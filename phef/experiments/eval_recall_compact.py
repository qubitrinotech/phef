"""
Compact-codebook evaluator for PHEF.

WHY
---
`phef/world.py:BranchingWorld.batch()` materialises a codebook of shape
(bs, n_keys, key_dim). At n_keys = 8_388_608, bs = 2, key_dim = 32 that is
536_870_912 float64 draws per batch (measured: 4.818 s/batch, 99.9% of
batch() wall time; projected 32.89 h for a 24576-batch evaluation).

`PHEF.forward` is POINTWISE in the codebook axis k:
  * key_encoder is applied row-wise;
  * einsum("bd,bkd->bk", ...) has no reduction over k;
  * torch.gather(..., 1, slots) keeps only the occurring entries.
Codebook rows that never occur in `slots` cannot influence logits or
retrieval scores. Measured occupancy at S = 65536, overwrite_p = 0.3:
45758 distinct slots out of 8_388_608 rows, i.e. 0.55%.

This evaluator materialises the codebook only for occurring slots and
re-indexes `slots` to compact ranks. Complexity drops from O(n_keys) to
O(n_segments).

WHAT IS PRESERVED
-----------------
n_keys is kept exactly where it carries meaning: the collision statistics
of `base ~ U{0..n_keys-1}` inside `_sample_slots`, which determine the
number of distinct competing keys
    N_eff(K) = K * (1 - (1 - 1/K)**R),  R = (1 - overwrite_p) * n_segments.

WHAT IS NOT PRESERVED
---------------------
The RNG stream differs from BranchingWorld (fewer normal draws per batch),
so results are NOT bit-identical to eval_recall_checkpoint.py. Equivalence
is distributional and MUST be validated as such.

Output JSON uses "compact_eval_v1" for the legacy dense path and
"compact_eval_v2_streamed_hard" when --stream-hard-chunk-size is positive.
Both are deliberately distinct from the old evaluator's schema.
"""

import argparse
import json
import math
import os
import random
import resource
import time

import numpy as np
import torch

from phef.model import PHEF
from phef.world import BranchingWorld

LEGACY_SCHEMA = "compact_eval_v1"
STREAMED_SCHEMA = "compact_eval_v2_streamed_hard"
HARD_EVAL = True


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def n_eff_theory(n_keys, n_segments, overwrite_p):
    r = (1.0 - overwrite_p) * n_segments
    return n_keys * (1.0 - math.exp(r * math.log1p(-1.0 / n_keys)))


class CompactBranchingWorld(BranchingWorld):
    """BranchingWorld with a lazily materialised, per-episode codebook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_n_eff = None

    def batch(self, bs):
        slots = self._sample_slots(bs)
        rows = np.arange(bs)

        values = self.rng.integers(
            0,
            self.n_values,
            size=(bs, self.n_segments),
        )

        timestamps = np.broadcast_to(
            np.arange(self.n_segments),
            (bs, self.n_segments),
        ).copy()

        pick = self.rng.integers(0, self.n_segments, size=bs)
        query_slots = slots[rows, pick]

        match = slots == query_slots[:, None]
        answer_positions = (
            self.n_segments
            - 1
            - np.argmax(match[:, ::-1], axis=1)
        )
        answers = values[rows, answer_positions]

        if self.symbolic:
            raise NotImplementedError(
                "symbolic-режим не поддерживает контракт "
                "(slots, codebook, values, timestamps, "
                "query_keys, answers, positions)"
            )

        # --- compaction: global slot id -> per-row rank ------------------
        compact = np.empty_like(slots)
        n_eff = np.empty(bs, dtype=np.int64)
        for b in range(bs):
            uniq, inv = np.unique(slots[b], return_inverse=True)
            compact[b] = inv.reshape(-1)
            n_eff[b] = uniq.size
        u_max = int(n_eff.max())
        self.last_n_eff = [int(x) for x in n_eff]

        # Same construction as BranchingWorld, only smaller in axis 1.
        codebook = self.rng.standard_normal(
            (bs, u_max, self.key_dim)
        ).astype(np.float32)

        codebook /= np.linalg.norm(
            codebook,
            axis=-1,
            keepdims=True,
        )

        compact_query_slots = compact[rows, pick]
        query_keys = codebook[rows, compact_query_slots]

        slots_t = torch.from_numpy(compact.astype(np.int64))
        query_slots_t = torch.from_numpy(compact_query_slots.astype(np.int64))
        values_t = torch.from_numpy(values.astype(np.int64))
        timestamps_t = torch.from_numpy(timestamps.astype(np.int64))
        answers_t = torch.from_numpy(answers.astype(np.int64))
        positions_t = torch.from_numpy(answer_positions.astype(np.int64))

        self.last_slots = slots_t
        self.last_query_slots = query_slots_t
        self.last_answer_positions = positions_t

        return (
            slots_t,
            torch.from_numpy(codebook),
            values_t,
            timestamps_t,
            torch.from_numpy(query_keys),
            answers_t,
            positions_t,
        )


def move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


@torch.inference_mode()
def evaluate(
    model,
    world,
    batch_size,
    device,
    n_batches,
    label,
    progress_every,
    shuffle=False,
    mask_memory=False,
    stream_hard_chunk_size=0,
):
    model.eval()
    correct = 0
    retrieval_correct = 0
    total = 0
    score_retrieval = not mask_memory
    n_eff_sum = 0
    n_eff_min = None
    n_eff_max = None
    t_start = time.perf_counter()

    for i in range(n_batches):
        batch = world.batch(batch_size)
        for v in world.last_n_eff:
            n_eff_sum += v
            n_eff_min = v if n_eff_min is None else min(n_eff_min, v)
            n_eff_max = v if n_eff_max is None else max(n_eff_max, v)

        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = move_batch(batch, device)

        use_streamed_hard = (
            stream_hard_chunk_size > 0 and not mask_memory
        )
        if use_streamed_hard:
            logits, retrieval_indices = model(
                slots_b,
                codebook,
                values,
                timestamps,
                query_keys,
                shuffle=shuffle,
                hard=HARD_EVAL,
                stream_hard_chunk_size=stream_hard_chunk_size,
                return_retrieval_indices=True,
            )
        else:
            logits, retrieval_scores = model(
                slots_b,
                codebook,
                values,
                timestamps,
                query_keys,
                shuffle=shuffle,
                mask_memory=mask_memory,
                hard=HARD_EVAL,
                return_retrieval=True,
            )
        correct += logits.argmax(dim=-1).eq(answers).sum().item()
        if score_retrieval:
            if use_streamed_hard:
                retrieval_correct += (
                    retrieval_indices.eq(answer_positions).sum().item()
                )
            else:
                retrieval_correct += (
                    retrieval_scores.argmax(dim=-1)
                    .eq(answer_positions)
                    .sum()
                    .item()
                )
        total += answers.numel()

        done = i + 1
        if progress_every and (done % progress_every == 0 or done == n_batches):
            elapsed = time.perf_counter() - t_start
            gpu = (
                torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"[{label}] {done}/{n_batches} batches "
                f"elapsed={elapsed:.1f}s "
                f"per_batch={elapsed / done:.4f}s "
                f"eta={(n_batches - done) * elapsed / done:.1f}s "
                f"answer_acc={correct / total:.6f} "
                f"retrieval_acc="
                f"{(retrieval_correct / total) if score_retrieval else float('nan'):.6f} "
                f"n_eff_mean={n_eff_sum / total:.1f} "
                f"rss={rss_gib():.2f}GiB gpu_alloc_peak={gpu:.2f}GiB",
                flush=True,
            )

    return {
        "answer_acc": correct / total,
        "retrieval_acc": retrieval_correct / total if score_retrieval else None,
        "episodes": total,
        "n_eff_mean": n_eff_sum / total,
        "n_eff_min": n_eff_min,
        "n_eff_max": n_eff_max,
        "wall_seconds": time.perf_counter() - t_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-keys", type=int, default=512)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=65536)
    parser.add_argument("--overwrite-p", type=float, default=0.3)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--stream-hard-chunk-size", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.eval_batch_size < 2:
        raise SystemExit(
            "eval_batch_size must be >= 2: PHEF.forward(shuffle=True) calls "
            "torch.randint(1, batch_size, ...), which is invalid for 1"
        )
    if args.stream_hard_chunk_size < 0:
        raise SystemExit("--stream-hard-chunk-size must be >= 0")

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing {args.out}; pass --force to override"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ckpt_key_dim = checkpoint["state_dict"]["key_encoder.0.weight"].shape[1]
    if ckpt_key_dim != args.key_dim:
        raise SystemExit(
            f"checkpoint key_dim={ckpt_key_dim} != --key-dim {args.key_dim}"
        )

    eval_world = CompactBranchingWorld(
        seed=args.seed + 100_000,
        key_dim=args.key_dim,
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        overwrite_p=args.overwrite_p,
    )

    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=True,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    schema = (
        STREAMED_SCHEMA
        if args.stream_hard_chunk_size > 0
        else LEGACY_SCHEMA
    )
    print(
        f"schema={schema} device={device} checkpoint={args.checkpoint} "
        f"checkpoint_epoch={checkpoint['epoch']} "
        f"checkpoint_sel_acc={checkpoint['sel_acc']:.6f} "
        f"key_dim={args.key_dim} n_keys={args.n_keys} "
        f"n_segments={args.n_segments} "
        f"n_eff_theory={n_eff_theory(args.n_keys, args.n_segments, args.overwrite_p):.1f} "
        f"episodes={args.eval_batch_size * args.eval_batches} "
        f"stream_hard_chunk_size={args.stream_hard_chunk_size}",
        flush=True,
    )

    conditions = {}
    for label, kwargs in (
        ("real", {}),
        ("shuffled", {"shuffle": True}),
        ("masked", {"mask_memory": True}),
    ):
        conditions[label] = evaluate(
            model,
            eval_world,
            args.eval_batch_size,
            device,
            args.eval_batches,
            label,
            args.progress_every,
            stream_hard_chunk_size=args.stream_hard_chunk_size,
            **kwargs,
        )

    report = {
        "schema": schema,
        "stream_hard_chunk_size": args.stream_hard_chunk_size,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sel_acc": checkpoint["sel_acc"],
        "hard_eval": HARD_EVAL,
        "answer_chance": 1.0 / args.n_values,
        "n_eff_theory": n_eff_theory(
            args.n_keys, args.n_segments, args.overwrite_p
        ),
        "real": conditions["real"],
        "shuffled": conditions["shuffled"],
        "masked": conditions["masked"],
        "config": vars(args),
    }

    for label in ("real", "shuffled", "masked"):
        c = conditions[label]
        ret = c["retrieval_acc"]
        ret_s = "None" if ret is None else f"{ret:.6f}"
        print(
            f"{label:9s} answer={c['answer_acc']:.6f} retrieval={ret_s} "
            f"episodes={c['episodes']} n_eff_mean={c['n_eff_mean']:.1f} "
            f"wall={c['wall_seconds']:.1f}s",
            flush=True,
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
