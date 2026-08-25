"""
Замер, куда уходит время эпохи. Ничего не оптимизирует — только измеряет.

Рецепт шага повторяет experiments/analyze_memory.py: AdamW(wd=1e-4),
clip_grad_norm_(1.0), autocast(bfloat16) на cuda, allow_tf32.

Раскладывает эпоху на четыре части:
  world.batch      — генерация мира на CPU (numpy)
  move             — host -> device
  fwd+bwd+step     — собственно обучение
  epoch            — то же целиком, для сверки суммы

Запуск:
    cd ~/qubitrino-projects/agi/phef
    LC_ALL=C PYTHONPATH=. python -u experiments/profile_step.py \
        --n-segments 256 --n-keys 512 --key-dim 32 --dim 128 \
        --batch-size 256 --iters 200
"""

import argparse
import time

import torch
import torch.nn.functional as F

from phef.model import PHEF
from phef.world import BranchingWorld


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, iters, device):
    fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-keys", type=int, default=512)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=256)
    parser.add_argument("--overwrite-p", type=float, default=0.3)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--world", type=str, default="numpy", choices=["numpy", "gpu"]
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    wkw = dict(
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        overwrite_p=args.overwrite_p,
        key_dim=args.key_dim,
        seed=args.seed,
    )
    if args.world == "gpu":
        from phef.world_gpu import BranchingWorldGPU

        world = BranchingWorldGPU(device=str(device), **wkw)
    else:
        world = BranchingWorld(**wkw)

    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    def gen():
        return world.batch(args.batch_size)

    batch_cpu = gen()

    def move_only():
        return tuple(
            t.to(device, non_blocking=True) for t in batch_cpu
        )

    batch_gpu = move_only()

    def train_step():
        model.train()
        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            _,
        ) = batch_gpu
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                slots_b, codebook, values, timestamps, query_keys
            )
            loss = F.cross_entropy(logits, answers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def full_epoch():
        model.train()
        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            _,
        ) = tuple(
            t.to(device, non_blocking=True) for t in world.batch(args.batch_size)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                slots_b, codebook, values, timestamps, query_keys
            )
            loss = F.cross_entropy(logits, answers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    parts = [
        ("world.batch", timeit(gen, args.iters, device)),
        ("move", timeit(move_only, args.iters, device)),
        ("fwd+bwd+step", timeit(train_step, args.iters, device)),
        ("epoch", timeit(full_epoch, args.iters, device)),
    ]

    epoch_time = parts[-1][1]

    print(
        f"device={device} n_segments={args.n_segments} "
        f"n_keys={args.n_keys} key_dim={args.key_dim} "
        f"dim={args.dim} batch={args.batch_size} iters={args.iters}"
    )
    print(f"{'part':16s} {'ms':>9s} {'share':>8s}")
    for name, seconds in parts:
        share = seconds / epoch_time if epoch_time > 0 else 0.0
        tag = "" if name != "epoch" else "  (= 1.00 по определению)"
        print(f"{name:16s} {1000 * seconds:9.3f} {share:8.2%}{tag}")

    print(f"\nepochs/sec = {1.0 / epoch_time:.2f}")
    print(
        "Вывод про узкое место делать только по этим числам: "
        "если world.batch < 0.3 эпохи, переносить генерацию на GPU "
        "бессмысленно."
    )

    if device.type == "cuda":
        print(
            f"peak VRAM = "
            f"{torch.cuda.max_memory_allocated() / 2 ** 20:.1f} MiB"
        )


if __name__ == "__main__":
    main()
