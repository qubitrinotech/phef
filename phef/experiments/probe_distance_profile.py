"""Профиль точности по АБСОЛЮТНОЙ дистанции цели от конца эпизода.

probe_recency.py режет дистанцию в долях эпизода; при n_keys=512 все
цели лежат в последних ~S/n_keys событиях, то есть в первом бакете, и
профиль вырождается. Здесь границы бакетов квантильные по абсолютной
дистанции, поэтому вопрос "падает ли точность для далёких целей"
получает ответ. Читать slot_acc наравне с answer_acc.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from phef.model import PHEF
from phef.world import BranchingWorld


@torch.inference_mode()
def collect(model, world, batch_size, device, n_batches):
    model.eval()
    dists, ans_ok, slot_ok, last_ok = [], [], [], []

    for _ in range(n_batches):
        batch = tuple(t.to(device, non_blocking=True) for t in world.batch(batch_size))
        slots_b, codebook, values, timestamps, query_keys, answers, answer_positions = batch

        logits, scores = model(
            slots_b, codebook, values, timestamps, query_keys,
            hard=True, return_retrieval=True,
        )
        rows = torch.arange(answers.numel(), device=answers.device)
        picked = scores.argmax(dim=-1)
        slots = world.last_slots.to(device)
        query_slots = world.last_query_slots.to(device)

        dists.append(((world.n_segments - 1) - answer_positions).cpu().numpy())
        ans_ok.append(logits.argmax(dim=-1).eq(answers).cpu().numpy())
        slot_ok.append(slots[rows, picked].eq(query_slots).cpu().numpy())
        last_ok.append(picked.eq(answer_positions).cpu().numpy())

    cat = lambda xs: np.concatenate(xs).astype(np.float64)
    return np.concatenate(dists), cat(ans_ok), cat(slot_ok), cat(last_ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-keys", type=int, default=512)
    p.add_argument("--n-values", type=int, default=64)
    p.add_argument("--n-segments", type=int, default=524288)
    p.add_argument("--overwrite-p", type=float, default=0.3)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=256)
    p.add_argument("--eval-seed-offset", type=int, default=100_000)
    p.add_argument("--n-buckets", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available() and not a.cpu:
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    world = BranchingWorld(
        seed=a.seed + a.eval_seed_offset, key_dim=a.key_dim, n_keys=a.n_keys,
        n_values=a.n_values, n_segments=a.n_segments, overwrite_p=a.overwrite_p,
    )
    model = PHEF(
        n_values=a.n_values, key_dim=a.key_dim, dim=a.dim,
        n_segments=a.n_segments, top_k=1, use_memory=True,
    ).to(device)
    ck = torch.load(a.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])

    print(
        f"device={device} ckpt={a.checkpoint} epoch={ck['epoch']} "
        f"S={a.n_segments} episodes={a.eval_batch_size * a.eval_batches} "
        f"score_scale={model.score_scale.item():.4f} "
        f"recency_scale={model.recency_scale.item():.4f}",
        flush=True,
    )

    d, ans, slot, last = collect(model, world, a.eval_batch_size, device, a.eval_batches)

    pct = {f"p{q}": float(np.percentile(d, q)) for q in (0, 10, 25, 50, 75, 90, 99, 100)}
    edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, a.n_buckets + 1)))
    idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)

    buckets = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket": b, "dist_lo": float(edges[b]), "dist_hi": float(edges[b + 1]),
            "n": n, "answer_acc": float(ans[sel].mean()),
            "slot_acc": float(slot[sel].mean()), "last_acc": float(last[sel].mean()),
        })

    rep = {
        "checkpoint": a.checkpoint, "checkpoint_epoch": int(ck["epoch"]),
        "n_segments": a.n_segments, "episodes": int(d.size),
        "overall_answer_acc": float(ans.mean()), "overall_slot_acc": float(slot.mean()),
        "overall_last_acc": float(last.mean()),
        "score_scale": float(model.score_scale.item()),
        "recency_scale": float(model.recency_scale.item()),
        "dist_percentiles": pct, "dist_mean": float(d.mean()),
        "expected_dist_uniform": a.n_segments / a.n_keys,
        "buckets": buckets, "config": vars(a),
    }

    print("dist percentiles:", pct, flush=True)
    print(f"dist_mean={d.mean():.1f} expected~{a.n_segments / a.n_keys:.1f}", flush=True)
    print(f"overall answer={ans.mean():.6f} slot={slot.mean():.6f} last={last.mean():.6f}", flush=True)
    for b in buckets:
        print(
            f"bucket {b['bucket']:2d} dist [{b['dist_lo']:8.0f}, {b['dist_hi']:8.0f}] "
            f"n={b['n']:5d} answer={b['answer_acc']:.4f} slot={b['slot_acc']:.4f} last={b['last_acc']:.4f}",
            flush=True,
        )

    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(rep, h, indent=2)
    print(f"saved: {a.out}", flush=True)


if __name__ == "__main__":
    main()
