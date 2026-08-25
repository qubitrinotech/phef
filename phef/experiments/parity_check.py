"""
Проверка, что GPU-мир эквивалентен numpy-миру, и замер пропускной способности.

Побитовой RNG-совместимости нет; проверяются инварианты задачи:
  1. распределение позиции ответа (последнее вхождение);
  2. доля запросов, у которых ответ не в последнем сегменте;
  3. число различных слотов на эпизод;
  4. геометрия ключей: unit-норма, oracle raw top-1 = 1.0;
  5. потолок «вернуть последнее» (recency-shortcut baseline).

Плюс: сколько батчей в секунду даёт каждый бэкенд.

batch() возвращает (slots, codebook), поэтому норма проверяется прямо
на codebook, а similarity получается gather'ом einsum по n_keys.

    LC_ALL=C PYTHONPATH=. python -u experiments/parity_check.py \
        --n-segments 256 --n-keys 512 --key-dim 32 --batch-size 256
"""

import argparse
import time

import torch

from phef.world import BranchingWorld
from phef.world_gpu import BranchingWorldGPU


def stats(world, bs, n_batches, device):
    pos_all = []
    uniq = []
    last_frac = 0.0
    orc = 0.0
    total = 0
    norm_err = 0.0

    for _ in range(n_batches):
        _, cb, val, ts, qk, ans, pos = world.batch(bs)
        cb = cb.to(device).float()
        qk = qk.to(device).float()
        slots = world.last_slots.to(device)
        qslots = world.last_query_slots.to(device)
        pos = pos.to(device)

        pos_all.append(pos.float().mean().item())
        uniq.append(
            float(
                sum(len(torch.unique(row)) for row in slots.cpu()) / slots.shape[0]
            )
        )
        last_frac += (pos == world.n_segments - 1).float().mean().item()

        norm_err = max(norm_err, (cb.norm(dim=-1) - 1).abs().max().item())

        rows = torch.arange(slots.shape[0], device=device)
        raw = torch.gather(torch.einsum("bd,bkd->bk", qk, cb), 1, slots)
        orc += (slots[rows, raw.argmax(1)] == qslots).float().mean().item()
        total += 1

    return {
        "mean_answer_position": sum(pos_all) / total,
        "mean_unique_slots": sum(uniq) / total,
        "frac_answer_is_last_segment": last_frac / total,
        "oracle_raw_top1": orc / total,
        "max_key_norm_error": norm_err,
    }


def throughput(world, bs, n_batches):
    world.batch(bs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        world.batch(bs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    return {"batches_per_sec": round(n_batches / dt, 2), "ms_per_batch": round(1000 * dt / n_batches, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-segments", type=int, default=256)
    p.add_argument("--n-keys", type=int, default=512)
    p.add_argument("--n-values", type=int, default=64)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    common = dict(
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        key_dim=args.key_dim,
        seed=0,
    )

    wn = BranchingWorld(**common)
    wg = BranchingWorldGPU(**common, device=args.device)

    sn = stats(wn, args.batch_size, args.batches, dev)
    sg = stats(wg, args.batch_size, args.batches, dev)

    print(f"{'metric':32s} {'numpy':>12s} {'gpu':>12s} {'delta':>10s}")
    for k in sn:
        d = sg[k] - sn[k]
        print(f"{k:32s} {sn[k]:12.6f} {sg[k]:12.6f} {d:10.6f}")

    print("\nthroughput (world generation only)")
    print("numpy", throughput(wn, args.batch_size, args.batches))
    print("gpu  ", throughput(wg, args.batch_size, args.batches))


if __name__ == "__main__":
    main()
