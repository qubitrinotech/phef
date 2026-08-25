"""Измерение геометрии семантических скоров на сохранённых весах.

Шаг 8.2, часть первая: НИЧЕГО не решает и не выбирает. Только меряет
распределение зазоров, чтобы порог «почти равенства» назначался по
данным, а не наугад.

Чистая семантика достаётся через сам forward: при score_scale=1.0 и
recency_scale=0.0 строки model.py:91-92 дают scores = 1.0 * semantic
+ 0.0 * recency, что побитово равно semantic. Свои множители модели
сохраняются и восстанавливаются после замера.

Что меряется на каждом эпизоде:
  own_spread = max - min семантики среди событий СВОЕГО слота
               (world.py:111-116 выдаёт им одинаковые ключи, так что
               ожидается ноль; проверяем, а не предполагаем);
  gap        = own_max - other_max, зазор до лучшего ЧУЖОГО события;
               gap > 0 означает, что семантика сама попадает в слот;
  по сетке порогов tau: множество кандидатов C = {s : sem_s >= max - tau},
               его размер, и есть ли в нём чужие события.

Никакого нового ридаута здесь нет и никакая точность не считается.
"""

import argparse
import glob as globmod
import json
import os

import torch

from phef.model import PHEF
from phef.world import BranchingWorld


def make_world(cfg, seed, device):
    common = dict(
        n_keys=cfg["n_keys"],
        n_values=cfg["n_values"],
        n_segments=cfg["n_segments"],
        overwrite_p=cfg["overwrite_p"],
        key_dim=cfg["key_dim"],
    )

    if cfg.get("world", "numpy") == "gpu":
        from phef.world_gpu import BranchingWorldGPU

        return BranchingWorldGPU(
            seed=seed, device=str(device), **common
        )

    return BranchingWorld(seed=seed, **common)


@torch.inference_mode()
def measure(model, world, device, batch_size, n_batches, taus):
    """Распределения зазоров. Ридаут не трогается."""
    model.eval()

    own_counts = []
    spreads = []
    gaps = []

    n_tau = len(taus)
    cand_size = torch.zeros(n_tau, dtype=torch.float64, device=device)
    cand_has_foreign = torch.zeros(n_tau, dtype=torch.float64, device=device)
    cand_has_own = torch.zeros(n_tau, dtype=torch.float64, device=device)
    cand_clean = torch.zeros(n_tau, dtype=torch.float64, device=device)
    total = 0

    for _ in range(n_batches):
        batch = world.batch(batch_size)
        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = tuple(t.to(device, non_blocking=True) for t in batch)

        _, sem = model(
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            hard=True,
            return_retrieval=True,
        )

        slots = world.last_slots.to(device)
        query_slots = world.last_query_slots.to(device)
        own = slots.eq(query_slots.unsqueeze(1))

        big = torch.finfo(sem.dtype).max
        small = torch.finfo(sem.dtype).min

        own_max = sem.masked_fill(~own, small).max(dim=1).values
        own_min = sem.masked_fill(~own, big).min(dim=1).values
        other_max = sem.masked_fill(own, small).max(dim=1).values
        global_max = sem.max(dim=1).values

        own_counts.append(own.sum(dim=1).to(torch.float32))
        spreads.append(own_max - own_min)
        gaps.append(own_max - other_max)

        for i, tau in enumerate(taus):
            cand = sem.ge(global_max.unsqueeze(1) - tau)
            has_foreign = (cand & ~own).any(dim=1)
            has_own = (cand & own).any(dim=1)

            cand_size[i] += cand.sum().to(torch.float64)
            cand_has_foreign[i] += has_foreign.sum().to(torch.float64)
            cand_has_own[i] += has_own.sum().to(torch.float64)
            cand_clean[i] += (has_own & ~has_foreign).sum().to(torch.float64)

        total += query_slots.numel()

    own_counts = torch.cat(own_counts)
    spreads = torch.cat(spreads)
    gaps = torch.cat(gaps)

    probs = torch.tensor(
        [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 1.0],
        dtype=torch.float32,
        device=device,
    )

    def quant(x):
        return [float(v) for v in torch.quantile(x.float(), probs)]

    per_tau = []
    for i, tau in enumerate(taus):
        per_tau.append(
            {
                "tau": tau,
                "mean_cand": float(cand_size[i].item() / total),
                "frac_foreign_in_cand": float(cand_has_foreign[i].item() / total),
                "frac_own_in_cand": float(cand_has_own[i].item() / total),
                "frac_own_and_clean": float(cand_clean[i].item() / total),
            }
        )

    return {
        "n_episodes": total,
        "own_count_mean": float(own_counts.mean().item()),
        "own_count_max": float(own_counts.max().item()),
        "frac_own_count_1": float(own_counts.eq(1).float().mean().item()),
        "spread_q": quant(spreads),
        "frac_spread_exact_zero": float(spreads.eq(0).float().mean().item()),
        "gap_q": quant(gaps),
        "frac_gap_pos": float(gaps.gt(0).float().mean().item()),
        "frac_gap_zero": float(gaps.eq(0).float().mean().item()),
        "frac_gap_neg": float(gaps.lt(0).float().mean().item()),
        "per_tau": per_tau,
        "quantile_probs": [float(p) for p in probs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        type=str,
        required=True,
        help="шаблон путей к чекпойнтам .pt",
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        help="пороги почти-равенства В СЫРЫХ единицах semantic "
        "(без score_scale)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="по умолчанию берётся из конфига чекпойнта",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="по умолчанию берётся из конфига чекпойнта",
    )
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=100_000,
        help="смещение сида оценочного мира относительно cfg['seed']; "
        "100000 — тот же мир, что в analyze_memory.py",
    )
    parser.add_argument("--out", type=str, default="")

    args = parser.parse_args()

    paths = sorted(globmod.glob(args.glob))
    if not paths:
        raise SystemExit(f"по шаблону {args.glob} ничего не найдено")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"устройство: {device}, чекпойнтов: {len(paths)}")
    print(f"пороги tau (сырые единицы semantic): {args.taus}")
    print(f"смещение сида оценочного мира: {args.eval_seed_offset}\n")

    everything = []

    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        model = PHEF(
            n_values=cfg["n_values"],
            key_dim=cfg["key_dim"],
            dim=cfg["dim"],
            n_segments=cfg["n_segments"],
            top_k=1,
            use_memory=True,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        base_recency = model.recency_scale.item()
        base_score = model.score_scale.item()

        eval_bs = (
            args.eval_batch_size
            if args.eval_batch_size is not None
            else cfg["eval_batch_size"]
        )
        eval_nb = (
            args.eval_batches
            if args.eval_batches is not None
            else cfg["eval_batches"]
        )
        eval_seed = cfg["seed"] + args.eval_seed_offset

        name = os.path.basename(path)
        print("=" * 72)
        print(
            f"{name}\n  seg={cfg['n_segments']} dim={cfg['dim']} "
            f"seed={cfg['seed']} world={cfg.get('world', 'numpy')} "
            f"epochs={cfg['epochs']}"
        )
        print(
            f"  score_scale={base_score:.4f} "
            f"recency_scale={base_recency:.4f}"
        )
        print(
            f"  оценка: {eval_nb} батчей по {eval_bs} "
            f"= {eval_nb * eval_bs} эпизодов, сид мира {eval_seed}"
        )

        with torch.no_grad():
            model.score_scale.fill_(1.0)
            model.recency_scale.fill_(0.0)

        world = make_world(cfg, eval_seed, device)
        stats = measure(model, world, device, eval_bs, eval_nb, args.taus)

        with torch.no_grad():
            model.score_scale.fill_(base_score)
            model.recency_scale.fill_(base_recency)

        restored = (
            model.score_scale.item() == base_score
            and model.recency_scale.item() == base_recency
        )
        print(
            f"  score_scale={base_score:.4f} "
            f"(восстановлен: {'да' if restored else 'НЕТ'})"
        )
        if not restored:
            raise SystemExit("множители модели не восстановились, стоп")

        print(
            f"\n  событий своего слота: среднее "
            f"{stats['own_count_mean']:.2f}, максимум "
            f"{int(stats['own_count_max'])}, ровно одно у "
            f"{stats['frac_own_count_1']:.4f} эпизодов"
        )

        labels = ["min", "p0.1", "p1", "p5", "p50", "p95", "max"]
        head = " ".join(lab.rjust(11) for lab in labels)
        print(f"\n  квантили, сырые единицы semantic:")
        print(f"    {'':>10} {head}")
        print(
            f"    {'own_spread':>10} "
            + " ".join(f"{v:11.3e}" for v in stats["spread_q"])
        )
        print(
            f"    {'gap':>10} "
            + " ".join(f"{v:11.3e}" for v in stats["gap_q"])
        )
        print(
            f"    own_spread ровно ноль у "
            f"{stats['frac_spread_exact_zero']:.6f} эпизодов"
        )
        print(
            f"    gap > 0 у {stats['frac_gap_pos']:.4f}, "
            f"= 0 у {stats['frac_gap_zero']:.4f}, "
            f"< 0 у {stats['frac_gap_neg']:.4f}"
        )
        print(
            f"    те же квантили gap в единицах scores "
            f"(x score_scale={base_score:.2f}): "
            + " ".join(f"{v * base_score:.3e}" for v in stats["gap_q"])
        )

        print(
            f"\n  {'tau':>10} {'x score_sc':>11} {'|C| сред.':>10} "
            f"{'чужие в C':>11} {'своё в C':>10} {'своё и чисто':>13}"
        )
        for row in stats["per_tau"]:
            print(
                f"  {row['tau']:10.1e} {row['tau'] * base_score:11.3e} "
                f"{row['mean_cand']:10.3f} "
                f"{row['frac_foreign_in_cand']:11.4f} "
                f"{row['frac_own_in_cand']:10.4f} "
                f"{row['frac_own_and_clean']:13.4f}"
            )

        everything.append(
            {
                "path": path,
                "config": cfg,
                "eval_seed_offset": args.eval_seed_offset,
                "eval_seed": eval_seed,
                "base_recency": base_recency,
                "base_score": base_score,
                "stats": stats,
            }
        )
        print()

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(everything, handle, indent=2)
        print(f"сводка сохранена: {args.out}")


if __name__ == "__main__":
    main()
