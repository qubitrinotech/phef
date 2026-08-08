"""Абляция слагаемого давности на сохранённых весах, без обучения.

Гипотеза: спад точности по дистанции создаётся слагаемым
recency_scale * t/(S-1) в model.py:91-92, которое сдвигает аргмакс
к свежим событиям ЧЕРЕЗ слоты. Проверка: домножить recency_scale
на набор коэффициентов и посмотреть на профиль slot_hit по дистанции.

Читать надо slot_hit, а не answer_acc. При нулевой давности события
одного слота дают побитово равные скоры (world.py:111-116 выдаёт им
одинаковые ключи), аргмакс среди них произволен, и answer_acc падает
из-за выбора не последней записи, а не из-за промаха мимо слота.
"""

import argparse
import glob as globmod
import json
import math
import os

import numpy as np
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
def profile(model, world, device, batch_size, n_batches, n_buckets):
    """Профиль по дистанции для одного значения recency_scale."""
    model.eval()

    b_ans = np.zeros(n_buckets)
    b_slot = np.zeros(n_buckets)
    b_tot = np.zeros(n_buckets)

    correct = 0
    slot_ok = 0
    last_ok = 0
    total = 0
    tie_sum = 0.0

    for _ in range(n_batches):
        batch = world.batch(batch_size)
        (
            event_keys,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = tuple(t.to(device, non_blocking=True) for t in batch)

        logits, scores = model(
            event_keys,
            values,
            timestamps,
            query_keys,
            hard=True,
            return_retrieval=True,
        )

        tie_sum += model.last_tie_rate

        rows = torch.arange(answers.numel(), device=answers.device)
        picked = scores.argmax(dim=-1)

        slots = world.last_slots.to(device)
        query_slots = world.last_query_slots.to(device)

        is_correct = logits.argmax(dim=-1).eq(answers)
        slot_hit = slots[rows, picked].eq(query_slots)

        correct += is_correct.sum().item()
        slot_ok += slot_hit.sum().item()
        last_ok += picked.eq(answer_positions).sum().item()
        total += answers.numel()

        n_seg = world.n_segments
        distance = (
            n_seg - 1 - answer_positions
        ).float() / max(1, n_seg - 1)
        bucket = (distance * n_buckets).long().clamp(0, n_buckets - 1)

        for b in range(n_buckets):
            sel = bucket.eq(b)
            b_tot[b] += sel.sum().item()
            b_ans[b] += is_correct[sel].sum().item()
            b_slot[b] += slot_hit[sel].sum().item()

    def per_bucket(arr):
        return [
            (arr[b] / b_tot[b]) if b_tot[b] > 0 else float("nan")
            for b in range(n_buckets)
        ]

    ans_prof = per_bucket(b_ans)
    slot_prof = per_bucket(b_slot)
    filled = [i for i in range(n_buckets) if b_tot[i] > 0]

    return {
        "answer_acc": correct / total,
        "slot_acc": slot_ok / total,
        "last_acc": last_ok / total,
        "tie_rate": tie_sum / n_batches,
        "ans_prof": ans_prof,
        "slot_prof": slot_prof,
        "counts": b_tot.tolist(),
        "ans_drop": (
            ans_prof[filled[0]] - ans_prof[filled[-1]]
            if len(filled) > 1
            else float("nan")
        ),
        "slot_drop": (
            slot_prof[filled[0]] - slot_prof[filled[-1]]
            if len(filled) > 1
            else float("nan")
        ),
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
        "--multipliers",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.0, 2.0],
        help="множители для recency_scale; 1.0 обязателен первым "
        "как проверка воспроизведения",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="по умолчанию берётся из конфига чекпойнта, иначе "
        "сверка с json заведомо не сойдётся",
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
        "100000 воспроизводит eval_world из analyze_memory.py, любое "
        "другое значение даёт независимый набор эпизодов, и сверка "
        "с json тогда не выполняется",
    )
    parser.add_argument("--n-buckets", type=int, default=8)
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
    print(f"множители recency_scale: {args.multipliers}")
    print(
        f"смещение сида оценочного мира: {args.eval_seed_offset}"
        + ("" if args.eval_seed_offset == 100_000 else " (независимый набор)")
        + "\n"
    )

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

        json_path = os.path.splitext(path)[0] + ".json"
        reported = None
        if os.path.exists(json_path):
            reported = json.load(open(json_path)).get("answer_acc")

        name = os.path.basename(path)
        print("=" * 72)
        print(
            f"{name}\n  seg={cfg['n_segments']} dim={cfg['dim']} "
            f"seed={cfg['seed']} world={cfg.get('world', 'numpy')} "
            f"epochs={cfg['epochs']}"
        )
        print(
            f"  score_scale={base_score:.4f} "
            f"recency_scale={base_recency:.4f} "
            f"отношение rec/score={base_recency / base_score:.4f}"
        )

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
        print(
            f"  оценка: {eval_nb} батчей по {eval_bs} "
            f"= {eval_nb * eval_bs} эпизодов, сид мира {eval_seed}"
        )

        results = {}
        for mult in args.multipliers:
            with torch.no_grad():
                model.recency_scale.fill_(base_recency * mult)

            world = make_world(cfg, eval_seed, device)
            results[mult] = profile(
                model,
                world,
                device,
                eval_bs,
                eval_nb,
                args.n_buckets,
            )

        with torch.no_grad():
            model.recency_scale.fill_(base_recency)

        base = results[args.multipliers[0]]
        if args.eval_seed_offset != 100_000:
            print(
                f"  воспроизведение: не проверяется, оценочный мир смещён "
                f"(offset={args.eval_seed_offset}), это независимый набор"
            )
        elif reported is not None and args.multipliers[0] == 1.0:
            delta = abs(base["answer_acc"] - reported)
            verdict = "совпало" if delta < 1e-9 else f"РАСХОЖДЕНИЕ {delta:.2e}"
            print(
                f"  воспроизведение: answer_acc {base['answer_acc']:.6f} "
                f"против {reported:.6f} в json -> {verdict}"
            )

        print(
            f"\n  {'mult':>6} {'rec_sc':>8} {'answer':>8} {'slot':>8} "
            f"{'last':>8} {'tie':>7} {'ans_drop':>9} {'slot_drop':>10}"
        )
        for mult in args.multipliers:
            r = results[mult]
            print(
                f"  {mult:6.2f} {base_recency * mult:8.3f} "
                f"{r['answer_acc']:8.4f} {r['slot_acc']:8.4f} "
                f"{r['last_acc']:8.4f} {r['tie_rate']:7.4f} "
                f"{r['ans_drop']:9.4f} {r['slot_drop']:10.4f}"
            )

        print(f"\n  slot_hit по дистанции (свежее -> далёкое):")
        header = " ".join(f"b{b}".rjust(7) for b in range(args.n_buckets))
        print(f"  {'mult':>6} {header}")
        for mult in args.multipliers:
            prof = results[mult]["slot_prof"]
            print(
                f"  {mult:6.2f} "
                + " ".join(f"{v:7.4f}" for v in prof)
            )

        print(f"\n  n по бакетам: "
              f"{[int(c) for c in base['counts']]}")

        everything.append(
            {
                "path": path,
                "config": cfg,
                "eval_seed_offset": args.eval_seed_offset,
                "eval_seed": eval_seed,
                "base_recency": base_recency,
                "base_score": base_score,
                "reported_answer_acc": reported,
                "results": {str(m): results[m] for m in args.multipliers},
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
