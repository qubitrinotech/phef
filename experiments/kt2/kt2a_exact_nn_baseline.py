"""KT-2A: exact nearest-neighbour baseline for latest-under-overwrite retrieval.

SCOPE
-----
This script measures ONLY an external, non-neural retrieval rule. It does not
load a PHEF checkpoint, does not run PHEF, does not import faiss, and does not
use CUDA. PHEF numbers are NOT recomputed here; comparison against the recorded
KT-1 figures is performed outside this script.

WHAT IS MEASURED
----------------
For each episode the candidate set is the per-episode compact codebook
`codebook[b]` of shape (n_eff, key_dim), i.e. the distinct memory keys actually
occurring in that episode. The query is the perturbed query key produced by
`phef.query_noise.perturb_query_keys`, exactly the function used by KT-1, so the
noise semantics are "imprecise query", not "corrupted memory".

Two hit definitions are recorded separately:

* slot_hit      -- the retrieved candidate is the slot of the query key.
* position_hit  -- the LAST occurrence index of the retrieved slot equals
                   `answer_positions[b]`, i.e. the latest-under-overwrite
                   ground truth produced by the generator.

RECENCY TIE-BREAK
-----------------
Declared before any run: among candidates whose squared distance to the query
is within `--tie-eps` of the minimum, the candidate whose LAST occurrence in
`slots[b]` is latest is selected. This is the recency tie-break required by the
roadmap KT-2 formulation.

EXACTNESS
---------
Search is exact (brute force over all n_eff candidates). Exact NN is an upper
bound on the accuracy attainable by any approximate index (HNSW, IVF-PQ) in the
same space, which is why a parity outcome here is already decisive for the
accuracy axis of KT-2.

OUTPUT
------
A single JSON report with, per sigma point: successes, n, Wilson 95% interval
for both hit definitions, timing, and candidate-array memory. Verdict is
computed against PHEF reference intervals supplied via --phef-reference.
"""

import argparse
import json
import math
import os
import platform
import resource
import sys
import time

from pathlib import Path

import numpy as np
import torch

SCHEMA = "kt2a_exact_nn_v1"

# Recorded KT-1 PHEF retrieval figures (successes/n at 256 episodes per point),
# carried over as historical reference only. Not recomputed by this script.
PHEF_KT1_REFERENCE = {
    "0.00": [249, 256],
    "0.30": [199, 256],
    "0.35": [170, 256],
    "0.42": [119, 256],
    "0.50": [76, 256],
    "0.70": [14, 256],
    "1.00": [3, 256],
    "1.50": [1, 256],
}

# Points on which the KT-2 accuracy verdict is decided. Declared before the run.
DECISION_SIGMAS = ("0.30", "0.35", "0.42", "0.50")

DEFAULT_SIGMAS = "0.00,0.30,0.35,0.42,0.50,0.70,1.00,1.50"


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def wilson95(successes, n):
    """Wilson score interval at 95% confidence."""
    if n <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return [(centre - half) / denom, (centre + half) / denom]


def last_occurrence_table(slots_row, n_eff):
    """For each compact slot rank, the index of its LAST occurrence.

    slots_row is a 1-D int64 array of compact ranks in [0, n_eff).
    Returns an int64 array of length n_eff. Every rank occurs at least once by
    construction of the compaction, so no sentinel is required; an assertion
    guards that invariant.
    """
    table = np.full(n_eff, -1, dtype=np.int64)
    positions = np.arange(slots_row.shape[0], dtype=np.int64)
    # Later positions overwrite earlier ones, leaving the last occurrence.
    table[slots_row] = positions
    if int(table.min()) < 0:
        raise AssertionError("compact slot rank never occurs in slots row")
    return table


def exact_nn_with_recency(candidates, query, last_occ, tie_eps):
    """Exact nearest neighbour with recency tie-break.

    candidates: (n_eff, key_dim) float32
    query:      (key_dim,) float32
    last_occ:   (n_eff,) int64, last occurrence index per candidate rank
    Returns (chosen_rank, n_tied).
    """
    diff = candidates - query[None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    best = float(dist2.min())
    tied = np.nonzero(dist2 <= best + tie_eps)[0]
    if tied.shape[0] == 1:
        return int(tied[0]), 1
    # Recency tie-break: latest last occurrence wins.
    winner = tied[int(np.argmax(last_occ[tied]))]
    return int(winner), int(tied.shape[0])


def run_sigma_point(
    world_factory,
    sigma,
    episodes,
    batch_size,
    tie_eps,
    noise_seed,
    progress_every,
):
    """Evaluate one sigma point. Returns a metrics dict."""
    world = world_factory()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))

    slot_hits = 0
    position_hits = 0
    total = 0
    tie_events = 0

    n_eff_sum = 0
    n_eff_min = None
    n_eff_max = None

    candidate_bytes_max = 0
    build_seconds = 0.0
    query_seconds = 0.0

    key_scale_sum = 0.0
    raw_cos_sum = 0.0
    noise_batches = 0

    t_start = time.perf_counter()

    n_batches = episodes // batch_size
    if n_batches * batch_size != episodes:
        raise SystemExit(
            f"episodes={episodes} must be divisible by batch_size={batch_size}"
        )

    from phef.query_noise import key_scale_from_codebook, perturb_query_keys

    for _ in range(n_batches):
        (
            slots_t,
            codebook_t,
            _values_t,
            _timestamps_t,
            query_keys_t,
            _answers_t,
            answer_positions_t,
        ) = world.batch(batch_size)

        for value in world.last_n_eff:
            n_eff_sum += value
            n_eff_min = value if n_eff_min is None else min(n_eff_min, value)
            n_eff_max = value if n_eff_max is None else max(n_eff_max, value)

        if sigma > 0.0:
            scale = key_scale_from_codebook(codebook_t)
            key_scale_sum += scale
            clean_keys = query_keys_t
            query_keys_t = perturb_query_keys(
                query_keys_t, sigma, scale, generator
            )
            raw_cos_sum += float(
                torch.nn.functional.cosine_similarity(
                    clean_keys, query_keys_t, dim=-1
                )
                .mean()
                .item()
            )
            noise_batches += 1

        slots = slots_t.numpy()
        codebook = codebook_t.numpy()
        queries = query_keys_t.numpy()
        answer_positions = answer_positions_t.numpy()

        for b in range(batch_size):
            n_eff = int(world.last_n_eff[b])

            t0 = time.perf_counter()
            candidates = np.ascontiguousarray(
                codebook[b, :n_eff, :], dtype=np.float32
            )
            last_occ = last_occurrence_table(slots[b], n_eff)
            build_seconds += time.perf_counter() - t0

            candidate_bytes_max = max(candidate_bytes_max, candidates.nbytes)

            t1 = time.perf_counter()
            chosen, n_tied = exact_nn_with_recency(
                candidates,
                np.ascontiguousarray(queries[b], dtype=np.float32),
                last_occ,
                tie_eps,
            )
            query_seconds += time.perf_counter() - t1

            if n_tied > 1:
                tie_events += 1

            true_position = int(answer_positions[b])
            true_slot = int(slots[b, true_position])

            if chosen == true_slot:
                slot_hits += 1
            if int(last_occ[chosen]) == true_position:
                position_hits += 1

            total += 1

        if progress_every and total % progress_every == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"[sigma={sigma:g}] {total}/{episodes} episodes "
                f"elapsed={elapsed:.1f}s "
                f"slot_acc={slot_hits / total:.6f} "
                f"pos_acc={position_hits / total:.6f} "
                f"n_eff_mean={n_eff_sum / total:.1f} "
                f"rss={rss_gib():.2f}GiB",
                flush=True,
            )

    return {
        "sigma": sigma,
        "episodes": total,
        "slot_hit": {
            "successes": slot_hits,
            "n": total,
            "point": slot_hits / total,
            "wilson95": wilson95(slot_hits, total),
        },
        "position_hit": {
            "successes": position_hits,
            "n": total,
            "point": position_hits / total,
            "wilson95": wilson95(position_hits, total),
        },
        "tie_events": tie_events,
        "tie_rate": tie_events / total,
        "n_eff_mean": n_eff_sum / total,
        "n_eff_min": n_eff_min,
        "n_eff_max": n_eff_max,
        "build_seconds_total": build_seconds,
        "build_seconds_per_episode": build_seconds / total,
        "query_seconds_total": query_seconds,
        "query_seconds_per_episode": query_seconds / total,
        "candidate_bytes_max": candidate_bytes_max,
        "candidate_mib_max": candidate_bytes_max / (1024.0 * 1024.0),
        "wall_seconds": time.perf_counter() - t_start,
        "key_scale_mean": (
            key_scale_sum / noise_batches if noise_batches else None
        ),
        "query_key_cosine_raw": (
            raw_cos_sum / noise_batches if noise_batches else 1.0
        ),
    }


def decide_verdict(points, reference):
    """Compare exact-NN position_hit against the PHEF reference intervals.

    Rules declared before the run:
      ANN_PARITY      -- baseline Wilson lower >= PHEF Wilson upper in at least
                         one decision sigma.
      PHEF_DOMINATES  -- PHEF Wilson lower > baseline Wilson upper in ALL
                         decision sigmas.
      MIXED           -- anything else.
    """
    comparisons = {}
    parity_any = False
    phef_all = True

    for key in DECISION_SIGMAS:
        if key not in points or key not in reference:
            phef_all = False
            comparisons[key] = {"status": "MISSING"}
            continue

        base = points[key]["position_hit"]["wilson95"]
        successes, n = reference[key]
        phef = wilson95(successes, n)

        parity = base[0] >= phef[1]
        phef_wins = phef[0] > base[1]

        parity_any = parity_any or parity
        phef_all = phef_all and phef_wins

        comparisons[key] = {
            "baseline_wilson95": base,
            "phef_reference_successes": successes,
            "phef_reference_n": n,
            "phef_wilson95": phef,
            "baseline_parity_or_better": bool(parity),
            "phef_strictly_higher": bool(phef_wins),
        }

    if parity_any:
        verdict = "ANN_PARITY"
    elif phef_all:
        verdict = "PHEF_DOMINATES"
    else:
        verdict = "MIXED"

    return verdict, comparisons


def main():
    parser = argparse.ArgumentParser(
        description=(
            "KT-2A exact nearest-neighbour baseline for latest-under-overwrite "
            "retrieval. CPU only. Does not load PHEF checkpoints."
        )
    )
    parser.add_argument("--n-keys", type=int, default=262144)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=65536)
    parser.add_argument("--overwrite-p", type=float, default=0.3)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--episodes-per-sigma", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=11)
    parser.add_argument(
        "--sigmas",
        type=str,
        default=DEFAULT_SIGMAS,
        help="comma-separated sigma grid; defaults to the recorded KT-1 grid",
    )
    parser.add_argument(
        "--tie-eps",
        type=float,
        default=1e-12,
        help="squared-distance tolerance for the recency tie-break",
    )
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--phef-reference",
        type=str,
        default="",
        help=(
            "optional JSON file mapping sigma string to [successes, n] for the "
            "PHEF reference; defaults to the recorded KT-1 figures"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.episodes_per_sigma <= 0:
        raise SystemExit("--episodes-per-sigma must be > 0")
    if args.batch_size < 2:
        raise SystemExit(
            "--batch-size must be >= 2 to match the generator contract used by "
            "the compact evaluator"
        )
    if args.tie_eps < 0.0:
        raise SystemExit("--tie-eps must be >= 0")
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing {args.out}; pass --force to override"
        )

    try:
        sigmas = [float(part) for part in args.sigmas.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"could not parse --sigmas: {error}") from error
    if not sigmas:
        raise SystemExit("--sigmas produced an empty grid")
    for sigma in sigmas:
        if sigma < 0.0:
            raise SystemExit("sigma values must be >= 0")

    reference = dict(PHEF_KT1_REFERENCE)
    if args.phef_reference:
        with open(args.phef_reference, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        reference = {str(k): list(v) for k, v in loaded.items()}

    # This baseline is CPU-only by design: no CUDA context is created.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import importlib.util

    # `phef/experiments` is not a package (no __init__.py), so the compact
    # evaluator is loaded by explicit file path. Verified by import probe.
    _eval_path = Path(__file__).resolve().parents[2] / \
        "phef" / "experiments" / "eval_recall_compact.py"
    if not _eval_path.is_file():
        raise SystemExit(f"compact evaluator not found: {_eval_path}")
    _spec = importlib.util.spec_from_file_location(
        "kt2a_eval_recall_compact", _eval_path
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    CompactBranchingWorld = _module.CompactBranchingWorld

    def world_factory():
        return CompactBranchingWorld(
            seed=args.seed + 100_000,
            key_dim=args.key_dim,
            n_keys=args.n_keys,
            n_values=args.n_values,
            n_segments=args.n_segments,
            overwrite_p=args.overwrite_p,
        )

    print(
        f"schema={SCHEMA} device=cpu "
        f"n_keys={args.n_keys} n_segments={args.n_segments} "
        f"overwrite_p={args.overwrite_p} n_values={args.n_values} "
        f"key_dim={args.key_dim} "
        f"episodes_per_sigma={args.episodes_per_sigma} "
        f"batch_size={args.batch_size} "
        f"seed={args.seed} noise_seed={args.noise_seed} "
        f"tie_eps={args.tie_eps:g} "
        f"sigmas={','.join(f'{s:g}' for s in sigmas)}",
        flush=True,
    )

    points = {}
    for sigma in sigmas:
        result = run_sigma_point(
            world_factory,
            sigma,
            args.episodes_per_sigma,
            args.batch_size,
            args.tie_eps,
            args.noise_seed,
            args.progress_every,
        )
        key = f"{sigma:.2f}"
        points[key] = result
        slot = result["slot_hit"]
        pos = result["position_hit"]
        print(
            f"sigma={sigma:g} "
            f"slot={slot['successes']}/{slot['n']}={slot['point']:.6f} "
            f"wilson=[{slot['wilson95'][0]:.6f},{slot['wilson95'][1]:.6f}] "
            f"pos={pos['successes']}/{pos['n']}={pos['point']:.6f} "
            f"wilson=[{pos['wilson95'][0]:.6f},{pos['wilson95'][1]:.6f}] "
            f"ties={result['tie_events']} "
            f"n_eff_mean={result['n_eff_mean']:.1f} "
            f"build_per_ep={result['build_seconds_per_episode'] * 1000:.3f}ms "
            f"query_per_ep={result['query_seconds_per_episode'] * 1000:.3f}ms "
            f"cand_mib={result['candidate_mib_max']:.2f} "
            f"wall={result['wall_seconds']:.1f}s",
            flush=True,
        )

    verdict, comparisons = decide_verdict(points, reference)

    report = {
        "schema": SCHEMA,
        "device": "cpu",
        "config": vars(args),
        "decision_sigmas": list(DECISION_SIGMAS),
        "phef_reference_source": (
            args.phef_reference if args.phef_reference else "recorded_KT1_inline"
        ),
        "phef_reference": reference,
        "points": points,
        "comparisons": comparisons,
        "verdict": verdict,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "rss_gib_peak": rss_gib(),
        "caveats": [
            "Exact NN is an upper bound on any approximate index in this space; "
            "no HNSW or IVF-PQ measurement is performed here.",
            "PHEF figures are the recorded KT-1 values at 14M depth and are not "
            "recomputed; this run is at the n_segments given in config, so the "
            "comparison addresses the search rule, not matched resources.",
            "Candidate memory reported is the per-episode candidate array only, "
            "not a persistent index footprint.",
        ],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"VERDICT: {verdict}", flush=True)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
