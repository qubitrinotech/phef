"""KT-3A checkpoint-only evaluator: hard/soft readout bridge."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from phef.model import PHEF
from phef.world import BranchingWorld


SCHEMA = "kt3a_hard_soft_bridge_v1"
N_BUCKETS = 8


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def provenance() -> dict:
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": git_value("status", "--porcelain") != "",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def entropy_sum(scores: torch.Tensor) -> float:
    weights = torch.softmax(scores.float(), dim=1)
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
    return float((entropy / math.log(scores.shape[1])).sum().item())


def load_model(checkpoint_path: str, device: torch.device, args: argparse.Namespace) -> tuple[PHEF, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"]
    key_weight = state_dict["key_encoder.0.weight"]
    if key_weight.shape[1] != args.key_dim:
        raise ValueError(
            f"checkpoint key_dim={key_weight.shape[1]} != --key-dim {args.key_dim}"
        )
    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=True,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def evaluate(model: PHEF, world: BranchingWorld, device: torch.device, args: argparse.Namespace) -> dict:
    hard_answer_correct = 0
    soft_answer_correct = 0
    retrieval_correct = 0
    retrieval_value_correct = 0
    total = 0
    tie_rows = 0
    tie_total = 0
    entropy_total = 0.0
    bucket_answer_correct = [0] * N_BUCKETS
    bucket_slot_hit = [0] * N_BUCKETS
    bucket_total = [0] * N_BUCKETS

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.eval_batches):
            batch = world.batch(args.eval_batch_size)
            (
                slots,
                codebook,
                values,
                timestamps,
                query_keys,
                answers,
                answer_positions,
            ) = tuple(t.to(device, non_blocking=True) for t in batch)

            hard_logits, scores = model(
                slots,
                codebook,
                values,
                timestamps,
                query_keys,
                hard=True,
                return_retrieval=True,
            )
            soft_logits = model(
                slots,
                codebook,
                values,
                timestamps,
                query_keys,
                hard=False,
            )

            hard_ok = hard_logits.argmax(dim=-1).eq(answers)
            soft_ok = soft_logits.argmax(dim=-1).eq(answers)
            rows = torch.arange(answers.numel(), device=device)
            picked = scores.argmax(dim=1)
            retrieval_ok = picked.eq(answer_positions)
            retrieval_value_ok = values[rows, picked].eq(answers)

            max_scores = scores.max(dim=1, keepdim=True).values
            ties = scores.ge(max_scores - 1e-6).sum(dim=1).gt(1)

            distance = (
                args.n_segments - 1 - answer_positions
            ).float() / max(1, args.n_segments - 1)
            buckets = (distance * N_BUCKETS).long().clamp(0, N_BUCKETS - 1)
            query_slots = world.last_query_slots.to(device)
            slot_hit = slots[rows, picked].eq(query_slots)

            hard_answer_correct += int(hard_ok.sum().item())
            soft_answer_correct += int(soft_ok.sum().item())
            retrieval_correct += int(retrieval_ok.sum().item())
            retrieval_value_correct += int(retrieval_value_ok.sum().item())
            total += int(answers.numel())
            tie_rows += int(ties.sum().item())
            tie_total += int(ties.numel())
            entropy_total += entropy_sum(scores)

            for bucket in range(N_BUCKETS):
                selected = buckets.eq(bucket)
                bucket_total[bucket] += int(selected.sum().item())
                bucket_answer_correct[bucket] += int(hard_ok[selected].sum().item())
                bucket_slot_hit[bucket] += int(slot_hit[selected].sum().item())

    return {
        "episodes": total,
        "hard_answer_correct": hard_answer_correct,
        "soft_answer_correct": soft_answer_correct,
        "retrieval_correct": retrieval_correct,
        "retrieval_value_correct": retrieval_value_correct,
        "tie_rows": tie_rows,
        "tie_total": tie_total,
        "entropy_sum": entropy_total,
        "entropy_total": total,
        "bucket_answer_correct": bucket_answer_correct,
        "bucket_slot_hit": bucket_slot_hit,
        "bucket_total": bucket_total,
        "wall_seconds": time.perf_counter() - started,
        "rss_peak_gib": rss_gib(),
        "gpu_alloc_peak_gib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            if device.type == "cuda"
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--n-keys", required=True, type=int)
    parser.add_argument("--n-segments", required=True, type=int)
    parser.add_argument("--n-values", required=True, type=int)
    parser.add_argument("--overwrite-p", required=True, type=float)
    parser.add_argument("--key-dim", required=True, type=int)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--eval-batch-size", required=True, type=int)
    parser.add_argument("--eval-batches", required=True, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.eval_batch_size < 1:
        raise SystemExit("--eval-batch-size must be >= 1")
    if args.eval_batches < 1:
        raise SystemExit("--eval-batches must be >= 1")
    if args.n_keys < 1 or args.n_segments < 1 or args.n_values < 1:
        raise SystemExit("--n-keys, --n-segments, and --n-values must be >= 1")
    if args.key_dim < 1 or args.dim < 1:
        raise SystemExit("--key-dim and --dim must be >= 1")
    if not 0.0 <= args.overwrite_p <= 1.0:
        raise SystemExit("--overwrite-p must be in [0, 1]")
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, checkpoint = load_model(args.checkpoint, device, args)
    world = BranchingWorld(
        seed=args.eval_seed,
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        overwrite_p=args.overwrite_p,
        key_dim=args.key_dim,
    )

    print(
        f"schema={SCHEMA} device={device} checkpoint={args.checkpoint} "
        f"train_seed={args.train_seed} eval_seed={args.eval_seed} "
        f"episodes={args.eval_batch_size * args.eval_batches}",
        flush=True,
    )
    metrics = evaluate(model, world, device, args)
    report = {
        "schema": SCHEMA,
        "protocol": "KT-3A hard-soft bridge",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sel_acc": checkpoint["sel_acc"],
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "config": vars(args),
        "metrics": metrics,
        "provenance": provenance(),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(
        f"hard_answer={metrics['hard_answer_correct']}/{metrics['episodes']} "
        f"soft_answer={metrics['soft_answer_correct']}/{metrics['episodes']} "
        f"retrieval={metrics['retrieval_correct']}/{metrics['episodes']} "
        f"tie_rate={metrics['tie_rows'] / metrics['tie_total']:.6f} "
        f"saved={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
