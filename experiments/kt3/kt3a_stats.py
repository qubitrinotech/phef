"""KT-3A statistics and immutable bridge verdict."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

Z95 = 1.959963984540054
SCHEMA = "kt3a_hard_soft_bridge_v1"
EXPECTED_CELLS = {(7, 7), (7, 11), (23, 7), (23, 11)}
EXPECTED_CONFIG = {
    "n_keys": 262144,
    "n_segments": 65536,
    "n_values": 64,
    "overwrite_p": 0.3,
    "key_dim": 32,
    "dim": 128,
    "eval_batch_size": 16,
    "eval_batches": 64,
}
EPISODES_PER_CELL = 1024
TIE_RATE_MAX = 0.01
PRIMARY_LOWER_BOUND_MIN = 0.95
ABS_HARD_SOFT_GAP_MAX = 0.05


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def require_int(value, name: str, lower: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < lower:
        raise ValueError(f"{name}={value} < {lower}")
    return value


def validate_artifact(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {data.get('schema')!r}")

    train_seed = require_int(data.get("train_seed"), f"{path}: train_seed")
    eval_seed = require_int(data.get("eval_seed"), f"{path}: eval_seed")
    if (train_seed, eval_seed) not in EXPECTED_CELLS:
        raise ValueError(f"{path}: unexpected cell {(train_seed, eval_seed)}")

    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be an object")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{path}: config[{key!r}]={config.get(key)!r}, expected {expected!r}"
            )
    if config.get("cpu") is not False:
        raise ValueError(f"{path}: KT-3A requires CUDA; config['cpu'] must be false")

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: metrics must be an object")
    episodes = require_int(metrics.get("episodes"), f"{path}: episodes", 1)
    if episodes != EPISODES_PER_CELL:
        raise ValueError(
            f"{path}: episodes={episodes}, expected {EPISODES_PER_CELL}"
        )

    for key in (
        "hard_answer_correct",
        "soft_answer_correct",
        "retrieval_correct",
        "retrieval_value_correct",
    ):
        require_int(metrics.get(key), f"{path}: {key}", 0)
        if metrics[key] > episodes:
            raise ValueError(f"{path}: {key} > episodes")

    tie_rows = require_int(metrics.get("tie_rows"), f"{path}: tie_rows", 0)
    tie_total = require_int(metrics.get("tie_total"), f"{path}: tie_total", 1)
    if tie_rows > tie_total or tie_total != episodes:
        raise ValueError(f"{path}: invalid tie counts")

    entropy_total = require_int(metrics.get("entropy_total"), f"{path}: entropy_total", 1)
    if entropy_total != episodes:
        raise ValueError(f"{path}: entropy_total must equal episodes")
    if not isinstance(metrics.get("entropy_sum"), (int, float)):
        raise ValueError(f"{path}: entropy_sum must be numeric")

    for key in ("bucket_answer_correct", "bucket_slot_hit", "bucket_total"):
        values = metrics.get(key)
        if not isinstance(values, list) or len(values) != 8:
            raise ValueError(f"{path}: {key} must be a length-8 list")
        for value in values:
            require_int(value, f"{path}: {key} item", 0)
    if sum(metrics["bucket_total"]) != episodes:
        raise ValueError(f"{path}: bucket_total does not sum to episodes")
    for correct_key in ("bucket_answer_correct", "bucket_slot_hit"):
        if any(
            hit > count
            for hit, count in zip(metrics[correct_key], metrics["bucket_total"])
        ):
            raise ValueError(f"{path}: invalid {correct_key} counts")

    return data


def metric_row(successes: int, n: int) -> dict:
    lo, hi = wilson(successes, n)
    return {
        "successes": successes,
        "n": n,
        "point": successes / n,
        "wilson95": [lo, hi],
    }


def summarize(cells: list[dict]) -> dict:
    metrics = [cell["metrics"] for cell in cells]
    total = sum(m["episodes"] for m in metrics)
    hard_answer = sum(m["hard_answer_correct"] for m in metrics)
    soft_answer = sum(m["soft_answer_correct"] for m in metrics)
    retrieval = sum(m["retrieval_correct"] for m in metrics)
    retrieval_value = sum(m["retrieval_value_correct"] for m in metrics)
    tie_rows = sum(m["tie_rows"] for m in metrics)
    tie_total = sum(m["tie_total"] for m in metrics)
    entropy_sum = sum(float(m["entropy_sum"]) for m in metrics)
    entropy_total = sum(m["entropy_total"] for m in metrics)

    per_cell = []
    all_ties_ok = True
    for cell in sorted(cells, key=lambda d: (d["train_seed"], d["eval_seed"])):
        m = cell["metrics"]
        tie_rate = m["tie_rows"] / m["tie_total"]
        all_ties_ok = all_ties_ok and tie_rate <= TIE_RATE_MAX
        per_cell.append(
            {
                "train_seed": cell["train_seed"],
                "eval_seed": cell["eval_seed"],
                "checkpoint": cell["checkpoint"],
                "checkpoint_epoch": cell["checkpoint_epoch"],
                "hard_answer": metric_row(m["hard_answer_correct"], m["episodes"]),
                "soft_answer": metric_row(m["soft_answer_correct"], m["episodes"]),
                "retrieval": metric_row(m["retrieval_correct"], m["episodes"]),
                "retrieval_value": metric_row(m["retrieval_value_correct"], m["episodes"]),
                "hard_soft_gap": (
                    m["hard_answer_correct"] - m["soft_answer_correct"]
                ) / m["episodes"],
                "tie_rate": tie_rate,
                "entropy_norm": float(m["entropy_sum"]) / m["entropy_total"],
                "wall_seconds": m.get("wall_seconds"),
                "gpu_alloc_peak_gib": m.get("gpu_alloc_peak_gib"),
                "rss_peak_gib": m.get("rss_peak_gib"),
            }
        )

    hard_answer_row = metric_row(hard_answer, total)
    retrieval_row = metric_row(retrieval, total)
    soft_answer_row = metric_row(soft_answer, total)
    hard_soft_gap = (hard_answer - soft_answer) / total
    pass_conditions = {
        "all_cells_valid": True,
        "all_cell_tie_rates_le_0_01": all_ties_ok,
        "hard_retrieval_wilson_lower_ge_0_95": (
            retrieval_row["wilson95"][0] >= PRIMARY_LOWER_BOUND_MIN
        ),
        "hard_answer_wilson_lower_ge_0_95": (
            hard_answer_row["wilson95"][0] >= PRIMARY_LOWER_BOUND_MIN
        ),
        "abs_hard_soft_gap_le_0_05": abs(hard_soft_gap) <= ABS_HARD_SOFT_GAP_MAX,
    }
    outcome = "BRIDGE_PASS" if all(pass_conditions.values()) else "BRIDGE_FAIL"

    return {
        "schema": "kt3a_hard_soft_bridge_stats_v1",
        "expected_cells": sorted([list(cell) for cell in EXPECTED_CELLS]),
        "criteria": {
            "tie_rate_max": TIE_RATE_MAX,
            "primary_wilson_lower_bound_min": PRIMARY_LOWER_BOUND_MIN,
            "abs_hard_soft_gap_max": ABS_HARD_SOFT_GAP_MAX,
        },
        "aggregate": {
            "hard_answer": hard_answer_row,
            "soft_answer": soft_answer_row,
            "hard_retrieval": retrieval_row,
            "retrieval_value": metric_row(retrieval_value, total),
            "hard_soft_gap": hard_soft_gap,
            "tie_rate": tie_rows / tie_total,
            "entropy_norm": entropy_sum / entropy_total,
            "episodes": total,
        },
        "cells": per_cell,
        "pass_conditions": pass_conditions,
        "outcome": outcome,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        result = {
            "schema": "kt3a_hard_soft_bridge_stats_v1",
            "outcome": "BRIDGE_INCONCLUSIVE",
            "reason": "no artifacts matched --glob",
        }
    else:
        try:
            cells = [validate_artifact(path) for path in paths]
            seen = {(cell["train_seed"], cell["eval_seed"]) for cell in cells}
            if len(cells) != len(EXPECTED_CELLS):
                raise ValueError(
                    f"expected {len(EXPECTED_CELLS)} artifacts, found {len(cells)}"
                )
            if seen != EXPECTED_CELLS:
                raise ValueError(f"cells={sorted(seen)}, expected={sorted(EXPECTED_CELLS)}")
            if len(seen) != len(cells):
                raise ValueError("duplicate train_seed/eval_seed cell")
            result = summarize(cells)
        except Exception as error:
            result = {
                "schema": "kt3a_hard_soft_bridge_stats_v1",
                "outcome": "BRIDGE_INCONCLUSIVE",
                "reason": str(error),
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["outcome"] == "BRIDGE_INCONCLUSIVE":
        sys.exit(1)


if __name__ == "__main__":
    main()
