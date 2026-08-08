import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from phef.model import PHEF
from phef.world import BranchingWorld


def move(batch, device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


@torch.inference_mode()
def oracle_bound(
    event_keys,
    query_keys,
    answer_positions,
    slots=None,
    query_slots=None,
    scale=1.0,
):
    """Доля эпизодов, где целевой ключ строго ближе всех к запросу.
    Это потолок задачи, а не модели.

    scale: делитель similarity. Для сырых unit-ключей 1.0 (косинус).
    Для геометрии модели — sqrt(dim), как в model.forward.
    """
    similarity = torch.einsum(
        "bd,bsd->bs",
        query_keys,
        event_keys,
    ) / scale

    batch_size, n_events = similarity.shape
    rows = torch.arange(batch_size, device=similarity.device)

    target_similarity = similarity[rows, answer_positions]

    if slots is not None and query_slots is not None:
        same_slot = slots.eq(query_slots.unsqueeze(1))
    else:
        same_slot = similarity.ge(
            target_similarity.unsqueeze(1) - 1e-5
        )

    distractor_similarity = similarity.masked_fill(same_slot, -2.0)
    best_distractor = distractor_similarity.max(dim=1).values

    bounds = {}
    for margin in (0.0, 0.02, 0.05, 0.10):
        separable = target_similarity > best_distractor + margin
        bounds[f"m{margin}"] = separable.float().mean().item()

    return bounds


@torch.inference_mode()
def top1_slot_recall(similarity, slots, query_slots):
    """Доля эпизодов, где argmax по similarity попал в слот запроса.

    Это soft-oracle: критерий совпадает с тем, чем реально пользуется
    считывание (порядок скоров), а не с фиксированным зазором 0.05.
    """
    rows = torch.arange(similarity.shape[0], device=similarity.device)
    picked = slots[rows, similarity.argmax(dim=1)]
    return picked.eq(query_slots).float().mean().item()


@torch.inference_mode()
def entropy_norm(scores):
    """Энтропия softmax-распределения похожестей, делённая на ln(n)."""
    weights = torch.softmax(scores.float(), dim=1)
    ent = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
    return (ent / math.log(scores.shape[1])).mean().item()


@torch.inference_mode()
def quick_acc(model, world, device, n=2048, batch_size=512, hard=False):
    """Валидационная accuracy для отбора чекпойнта и детекции escape.

    Отбор по training loss выбирал вырожденные recency-решения,
    поэтому критерий — порядок скоров, а не уверенность.

    hard: ридаут должен совпадать с тем, которым печатается итог,
    иначе отбирается один чекпойнт, а сообщается метрика другого.
    """
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    while total < n:
        (
            event_keys,
            values,
            timestamps,
            query_keys,
            answers,
            _,
        ) = move(world.batch(batch_size), device)

        logits = model(
            event_keys,
            values,
            timestamps,
            query_keys,
            hard=hard,
        )
        correct += logits.argmax(dim=-1).eq(answers).sum().item()
        total += answers.numel()

    if was_training:
        model.train()

    return correct / total


@torch.inference_mode()
def analyze(model, world, device, batch_size, n_batches, n_buckets=8):
    model.eval()

    bucket_correct = np.zeros(n_buckets)
    bucket_slot_correct = np.zeros(n_buckets)
    bucket_total = np.zeros(n_buckets)

    correct = 0
    correct_soft = 0
    retrieved = 0
    retrieved_value = 0
    total = 0
    oracle_sums = {}
    tie_sum = 0.0
    entropy_sum = 0.0

    for _ in range(n_batches):
        (
            event_keys,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = move(world.batch(batch_size), device)

        logits, scores = model(
            event_keys,
            values,
            timestamps,
            query_keys,
            hard=True,
            return_retrieval=True,
        )

        tie_sum += getattr(model, "last_tie_rate", 0.0)

        logits_soft = model(
            event_keys,
            values,
            timestamps,
            query_keys,
            hard=False,
        )

        is_correct = logits.argmax(dim=-1).eq(answers)
        correct += is_correct.sum().item()
        correct_soft += logits_soft.argmax(dim=-1).eq(answers).sum().item()

        rows = torch.arange(answers.numel(), device=answers.device)
        picked = scores.argmax(dim=-1)
        retrieved += picked.eq(answer_positions).sum().item()
        retrieved_value += values[rows, picked].eq(answers).sum().item()
        total += answers.numel()

        entropy_sum += entropy_norm(scores)

        slots = world.last_slots.to(device)
        query_slots = world.last_query_slots.to(device)

        # Попадание аргмакса реальных скоров в слот запроса, поэпизодно.
        # Агрегат этого же — oracle_bounds["score_top1"]; здесь он нужен
        # поэпизодно, чтобы разложить его по дистанции до ответа.
        slot_hit = slots[rows, picked].eq(query_slots)

        bounds = oracle_bound(
            event_keys,
            query_keys,
            answer_positions,
            slots,
            query_slots,
        )

        encoded_events = torch.nn.functional.normalize(
            model.key_encoder(event_keys), dim=-1
        )

        encoded_query = torch.nn.functional.normalize(
            model.query_encoder(query_keys), dim=-1
        )

        encoded_bounds = oracle_bound(
            encoded_events,
            encoded_query,
            answer_positions,
            slots,
            query_slots,
        )

        # То же самое, но в геометрии, которой пользуется model.forward:
        # dot / sqrt(dim), без нормировки выходов энкодеров.
        raw_events = model.key_encoder(event_keys)
        raw_query = model.query_encoder(query_keys)

        model_bounds = oracle_bound(
            raw_events,
            raw_query,
            answer_positions,
            slots,
            query_slots,
            scale=math.sqrt(model.dim),
        )

        for name, value in encoded_bounds.items():
            key = f"enc_{name}"
            oracle_sums[key] = oracle_sums.get(key, 0.0) + value

        for name, value in model_bounds.items():
            key = f"mg_{name}"
            oracle_sums[key] = oracle_sums.get(key, 0.0) + value

        for name, value in bounds.items():
            oracle_sums[name] = oracle_sums.get(name, 0.0) + value

        # soft-oracle: top-1 slot recall в трёх геометриях
        raw_similarity = torch.einsum(
            "bd,bsd->bs", query_keys, event_keys
        )
        model_similarity = torch.einsum(
            "bd,bsd->bs", raw_query, raw_events
        ) / math.sqrt(model.dim)

        for key, value in (
            (
                "raw_top1",
                top1_slot_recall(raw_similarity, slots, query_slots),
            ),
            (
                "mg_top1",
                top1_slot_recall(model_similarity, slots, query_slots),
            ),
            (
                "score_top1",
                top1_slot_recall(scores, slots, query_slots),
            ),
        ):
            oracle_sums[key] = oracle_sums.get(key, 0.0) + value

        distance = (
            world.n_segments - 1 - answer_positions
        ).float() / max(1, world.n_segments - 1)

        bucket = (distance * n_buckets).long().clamp(0, n_buckets - 1)

        for b in range(n_buckets):
            selected = bucket.eq(b)
            bucket_total[b] += selected.sum().item()
            bucket_correct[b] += is_correct[selected].sum().item()
            bucket_slot_correct[b] += slot_hit[selected].sum().item()

    accuracy_by_distance = [
        (bucket_correct[b] / bucket_total[b])
        if bucket_total[b] > 0
        else float("nan")
        for b in range(n_buckets)
    ]

    slot_accuracy_by_distance = [
        (bucket_slot_correct[b] / bucket_total[b])
        if bucket_total[b] > 0
        else float("nan")
        for b in range(n_buckets)
    ]

    filled = [
        accuracy_by_distance[b]
        for b in range(n_buckets)
        if bucket_total[b] > 0
    ]

    filled_slot = [
        slot_accuracy_by_distance[b]
        for b in range(n_buckets)
        if bucket_total[b] > 0
    ]

    return {
        "answer_acc": correct / total,
        "answer_acc_soft": correct_soft / total,
        "hard_soft_gap": (correct - correct_soft) / total,
        "retrieval_acc": retrieved / total,
        "retrieval_value_acc": retrieved_value / total,
        "tie_rate_mean": tie_sum / n_batches,
        "entropy_norm": entropy_sum / n_batches,
        "oracle_bounds": {
            name: value / n_batches
            for name, value in oracle_sums.items()
        },
        "oracle_bound": oracle_sums.get("m0.05", 0.0) / n_batches,
        "acc_by_distance": accuracy_by_distance,
        "count_by_distance": bucket_total.tolist(),
        "acc_recent": filled[0] if filled else None,
        "acc_remote": filled[-1] if filled else None,
        "drop_recent_remote": (
            filled[0] - filled[-1] if len(filled) > 1 else None
        ),
        "slot_acc_by_distance": slot_accuracy_by_distance,
        "slot_acc_recent": filled_slot[0] if filled_slot else None,
        "slot_acc_remote": filled_slot[-1] if filled_slot else None,
        "slot_drop_recent_remote": (
            filled_slot[0] - filled_slot[-1]
            if len(filled_slot) > 1
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-keys", type=int, default=512)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=64)
    parser.add_argument("--overwrite-p", type=float, default=0.3)
    parser.add_argument("--key-dim", type=int, default=32)

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--freeze-recency", action="store_true")
    parser.add_argument(
        "--recency-value",
        type=float,
        default=None,
        help=(
            "зафиксировать recency_scale на данном значении и исключить "
            "его из оптимизации. Отличается от --freeze-recency, который "
            "жёстко обнуляет скаляр. Взаимно исключается с ним"
        ),
    )
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--acc-every", type=int, default=250)
    parser.add_argument("--acc-n", type=int, default=2048)
    parser.add_argument(
        "--select-metric",
        type=str,
        choices=("soft", "hard"),
        default="soft",
        help=(
            "ридаут пробника для отбора чекпойнта. soft — прежнее "
            "поведение (сопоставимо с серией capacity), hard — тот же "
            "ридаут, которым печатается answer_acc"
        ),
    )
    parser.add_argument(
        "--escape-factor",
        type=float,
        default=2.0,
        help="порог выхода с плато в кратности chance (2.0 => 2/n_values)",
    )
    parser.add_argument("--lr-decay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/analysis.json")
    parser.add_argument(
        "--world",
        type=str,
        default="numpy",
        choices=["numpy", "gpu"],
        help="numpy = phef.world.BranchingWorld (по умолчанию, как раньше); "
        "gpu = phef.world_gpu.BranchingWorldGPU, побитовой совместимости нет",
    )

    args = parser.parse_args()

    if args.freeze_recency and args.recency_value is not None:
        parser.error(
            "--freeze-recency и --recency-value взаимно исключаются: "
            "первый обнуляет recency_scale, второй задаёт значение"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    common = dict(
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        overwrite_p=args.overwrite_p,
        key_dim=args.key_dim,
    )

    if args.world == "gpu":
        from phef.world_gpu import BranchingWorldGPU

        make_world = lambda seed: BranchingWorldGPU(
            seed=seed, device=str(device), **common
        )
    else:
        make_world = lambda seed: BranchingWorld(seed=seed, **common)

    train_world = make_world(args.seed)
    eval_world = make_world(args.seed + 100_000)
    sel_world = make_world(args.seed + 200_000)

    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=True,
    ).to(device)

    if args.freeze_recency:
        with torch.no_grad():
            model.recency_scale.zero_()
        model.recency_scale.requires_grad_(False)

    if args.recency_value is not None:
        with torch.no_grad():
            model.recency_scale.fill_(args.recency_value)
        model.recency_scale.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )
        if args.lr_decay
        else None
    )

    chance = 1.0 / args.n_values
    chance_loss = math.log(args.n_values)
    escape_threshold = args.escape_factor * chance
    select_hard = args.select_metric == "hard"

    print(
        f"device={device} n_segments={args.n_segments} "
        f"key_dim={args.key_dim} dim={args.dim} "
        f"freeze_recency={args.freeze_recency} "
        f"recency_value={args.recency_value} "
        f"use_best={args.use_best} "
        f"select_metric={args.select_metric} world={args.world}"
    )
    print(
        f"chance={chance:.6f} chance_loss={chance_loss:.4f} "
        f"escape_threshold_acc={escape_threshold:.6f}"
    )

    escape_epoch = [None]
    best_loss = [float("inf"), 0]
    best_acc = [0.0]
    best_state = [None]
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        (
            event_keys,
            values,
            timestamps,
            query_keys,
            answers,
            _,
        ) = move(train_world.batch(args.batch_size), device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                event_keys,
                values,
                timestamps,
                query_keys,
            )

            loss = F.cross_entropy(logits, answers)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        current = loss.item()

        if args.use_best and epoch % args.acc_every == 0:
            probe = quick_acc(
                model,
                sel_world,
                device,
                n=args.acc_n,
                batch_size=args.eval_batch_size,
                hard=select_hard,
            )

            # escape по accuracy: loss измеряет уверенность и расходится
            # с точностью, поэтому порог по loss давал ложные вердикты.
            if escape_epoch[0] is None and probe > escape_threshold:
                escape_epoch[0] = epoch

            if probe > best_acc[0]:
                best_acc[0] = probe
                best_loss[0] = current
                best_loss[1] = epoch
                best_state[0] = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }

            history.append(
                {
                    "epoch": epoch,
                    "sel_acc": probe,
                    "loss": current,
                    "score_scale": model.score_scale.item(),
                    "recency_scale": model.recency_scale.item(),
                    "memory_gate": model.memory_gate.item(),
                }
            )

            print(
                f"epoch {epoch:5d} sel_acc {probe:.4f} "
                f"best {best_acc[0]:.4f} @ {best_loss[1]} "
                f"escape {escape_epoch[0]}"
            )

        if epoch % 500 == 0:
            print(
                f"epoch {epoch:5d} loss {current:.4f} "
                f"chance {chance_loss:.4f}"
            )

    if args.use_best and best_state[0] is not None:
        model.load_state_dict(best_state[0])
        print(
            f"\nвосстановлен best checkpoint: "
            f"epoch {best_loss[1]} sel_acc {best_acc[0]:.4f}"
        )

    report = analyze(
        model,
        eval_world,
        device,
        args.eval_batch_size,
        args.eval_batches,
    )

    report["tie_rate"] = getattr(model, "last_tie_rate", 0.0)
    report["best_acc"] = best_acc[0] if args.use_best else None
    report["best_loss"] = (
        None if math.isinf(best_loss[0]) else best_loss[0]
    )
    report["best_epoch"] = best_loss[1]
    report["escape_epoch"] = escape_epoch[0]
    report["escape_criterion"] = "val_accuracy"
    report["escape_threshold"] = escape_threshold
    report["select_metric"] = args.select_metric
    report["chance"] = chance
    report["chance_loss"] = chance_loss
    report["recency_scale"] = model.recency_scale.item()
    report["score_scale"] = model.score_scale.item()
    report["memory_gate"] = model.memory_gate.item()
    report["selection"] = "val_accuracy" if args.use_best else "final"
    report["history"] = history
    report["config"] = vars(args)

    print("\n=== ANALYSIS ===")
    print(f"answer_acc     = {report['answer_acc']:.6f}")
    print(f"answer_soft    = {report['answer_acc_soft']:.6f}")
    print(f"hard_soft_gap  = {report['hard_soft_gap']:+.6f}")
    print(f"retrieval_acc  = {report['retrieval_acc']:.6f}")
    print(f"retrieval_val  = {report['retrieval_value_acc']:.6f}")
    print(f"entropy_norm   = {report['entropy_norm']:.6f}")

    for name, value in sorted(report["oracle_bounds"].items()):
        print(f"oracle {name:10} = {value:.6f}")

    print(f"tie_rate       = {report['tie_rate']:.6f}")
    print(f"tie_rate_mean  = {report['tie_rate_mean']:.6f}")
    print(f"recency_scale  = {report['recency_scale']:.4f}")
    print(f"score_scale    = {report['score_scale']:.4f}")
    print(f"memory_gate    = {report['memory_gate']:.4f}")
    print(f"escape_epoch   = {report['escape_epoch']}")

    if report["drop_recent_remote"] is not None:
        print(
            f"drop           = {report['drop_recent_remote']:.4f} "
            f"({report['acc_recent']:.4f} -> {report['acc_remote']:.4f})"
        )

    if report["slot_drop_recent_remote"] is not None:
        print(
            f"slot_drop      = {report['slot_drop_recent_remote']:.4f} "
            f"({report['slot_acc_recent']:.4f} -> "
            f"{report['slot_acc_remote']:.4f})"
        )

    if args.use_best:
        print(
            f"best_sel_acc   = {best_acc[0]:.4f} "
            f"@ epoch {best_loss[1]}"
        )

    if report["escape_epoch"] is None:
        print(
            "WARNING: модель не покинула chance-плато "
            f"(val acc не превысила {escape_threshold:.4f}). "
            "Увеличь --epochs, прежде чем делать вывод."
        )

    print("\nacc by distance (recent -> remote):")
    print("  bucket   answer   slot_hit        n")
    for b, (acc, slot_acc, count) in enumerate(
        zip(
            report["acc_by_distance"],
            report["slot_acc_by_distance"],
            report["count_by_distance"],
        )
    ):
        print(
            f"  {b:6d} {acc:8.4f} {slot_acc:10.4f} {int(count):8d}"
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Веса теряли уже дважды: прогон на 600k эпох невоспроизводим,
    # а любая геометрическая проверка требует state_dict. Сохраняем
    # всегда, рядом с --out, с тем же именем и расширением .pt.
    weights_path = os.path.splitext(args.out)[0] + ".pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "best_epoch": best_loss[1],
            "best_acc": best_acc[0] if args.use_best else None,
            "selection": report["selection"],
            "escape_epoch": escape_epoch[0],
        },
        weights_path,
    )
    report["weights_path"] = weights_path
    print(f"\nвеса сохранены: {weights_path}")

    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
