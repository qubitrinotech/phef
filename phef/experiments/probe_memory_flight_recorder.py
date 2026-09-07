"""Memory Flight Recorder (MFR) — read-only механическая диагностика PHEF retrieval.

Назначение
----------
Скрипт НЕ обучает, НЕ сохраняет .pt, НЕ меняет checkpoint и НЕ заменяет
experiments/eval_recall_checkpoint.py. Он не влияет на operational PASS-критерий.
Он объясняет, ПОЧЕМУ данный checkpoint проходит или не проходит hard-eval,
разлагая механизм адресации на семантический и recency-компоненты.

Воспроизводимая семантика (сверено с phef/model.py)
---------------------------------------------------
    semantic = gather(einsum("bd,bkd->bk", query_state, memory_keys) / sqrt(dim), 1, slots)
    recency  = timestamps.float() / max(1, n_segments - 1)
    scores   = score_scale * semantic + recency_scale * recency
    hard     = argmax(scores, dim=1)

Ключевое наблюдение задачи (сверено с phef/world.py)
----------------------------------------------------
answer_positions — ПОСЛЕДНЕЕ вхождение query_slot в slots. Все вхождения одного
и того же slot имеют ОДИНАКОВЫЙ semantic-score, поэтому различить их может только
recency. Отсюда ошибки строго делятся на два несовместимых класса:

    same_key_recency_error   : slots[pred] == query_slot, но pred != answer_position
    wrong_key_semantic_error : slots[pred] != query_slot

Recency budget
--------------
Правильная позиция P побеждает конкурента j, если

    score_scale * (sem_P - sem_j) > recency_scale * (t_j - t_P) / (S - 1)

Для конкурентов с ДРУГИМ ключом определяем семантический зазор

    delta_sem = min_over_wrong_keys(sem_P - sem_j)

ВАЖНО (исправление v2): delta_sem может быть отрицательным. Это значит, что
правильный ключ семантически ПРОИГРЫВАЕТ чужому ключу, и никакая recency его
не спасёт. Такие эпизоды НЕ являются "бюджетом" и учитываются отдельно как
semantic_inversion. Смешивание их с положительными значениями делает средний
"бюджет" бессмысленным.

Для эпизодов с delta_sem > 0 определяются две величины:

    B      = score_scale * delta_sem * (S - 1) / |recency_scale|   [timestamp-шаги]
    B_norm = score_scale * delta_sem / |recency_scale|             [безразмерная]

B — максимальное число timestamp-шагов, на которое recency способна перебить
семантику. B_norm сравним между разными S: если B_norm > 1, recency не может
перебить семантику ни на какой дистанции внутри эпизода, и адресация опирается
на содержание ключа. Сжатие B_norm указывает на переход к recency-эвристике,
которая не масштабируется с ростом числа ключей.

Выход
-----
Один небольшой JSON. Никаких сырых тензоров, никаких codebook, никаких
per-episode ключей — только потоковые агрегаты и гистограммы.
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch

from phef.model import PHEF
from phef.world import BranchingWorld


class Running:
    """Потоковые агрегаты без хранения per-episode массивов."""

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, tensor):
        tensor = tensor.detach().to(torch.float64).flatten()
        if tensor.numel() == 0:
            return
        self.n += tensor.numel()
        self.total += tensor.sum().item()
        self.total_sq += tensor.pow(2).sum().item()
        self.minimum = min(self.minimum, tensor.min().item())
        self.maximum = max(self.maximum, tensor.max().item())

    def summary(self):
        if self.n == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.n
        variance = max(0.0, self.total_sq / self.n - mean * mean)
        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class Quantiles:
    """Гистограмма по фиксированным краям; квантили оцениваются по бинам."""

    def __init__(self, edges):
        self.edges = torch.tensor(edges, dtype=torch.float64)
        self.counts = torch.zeros(len(edges) + 1, dtype=torch.float64)

    def update(self, tensor):
        tensor = tensor.detach().to(torch.float64).flatten().cpu()
        if tensor.numel() == 0:
            return
        index = torch.bucketize(tensor, self.edges, right=False)
        self.counts += torch.bincount(index, minlength=self.counts.numel()).to(
            torch.float64
        )

    def summary(self):
        total = self.counts.sum().item()
        return {
            "edges": [float(x) for x in self.edges.tolist()],
            "counts": [float(x) for x in self.counts.tolist()],
            "total": total,
            "fraction": (
                [float(x) / total for x in self.counts.tolist()] if total > 0 else None
            ),
        }


def bucket_by_tail(distance, n_buckets, n_segments):
    """Логарифмические бакеты по (S - 1 - answer_position)."""
    distance = distance.to(torch.float64).clamp(min=0.0)
    scaled = torch.log2(distance + 1.0)
    span = math.log2(max(2.0, float(n_segments)))
    index = (scaled / span * n_buckets).floor().to(torch.int64)
    return index.clamp(0, n_buckets - 1)


@torch.inference_mode()
def record(model, world, args, device, condition):
    """Один проход по батчам с полной механической декомпозицией."""

    mask_memory = condition == "masked"
    shuffle = condition == "shuffled"

    n_segments = args.n_segments
    recency_denominator = max(1, n_segments - 1)

    alpha = float(model.score_scale.item())
    beta = float(model.recency_scale.item())

    stats = {
        "answer_correct": 0,
        "retrieval_correct": 0,
        "episodes": 0,
        "same_key_recency_error": 0,
        "wrong_key_semantic_error": 0,
        "margin_nonpositive": 0,
        "tie_at_max": 0,
        "semantic_inversion": 0,
        "delta_sem_evaluated": 0,
    }

    running = {
        "margin": Running(),
        "delta_sem": Running(),
        "delta_sem_positive": Running(),
        "recency_budget": Running(),
        "recency_budget_norm": Running(),
        "dup_count": Running(),
        "tail_distance": Running(),
        "entropy": Running(),
        "true_rank": Running(),
        "semantic_true": Running(),
        "recency_term_true": Running(),
    }

    hist = {
        "margin": Quantiles([-1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]),
        "delta_sem": Quantiles(
            [-1.0, -0.1, -0.01, 0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        ),
        "recency_budget": Quantiles([1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9]),
        "recency_budget_norm": Quantiles(
            [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        ),
        "dup_count": Quantiles([1.5, 2.5, 4.5, 8.5, 16.5, 64.5, 256.5, 1024.5]),
    }

    n_buckets = args.tail_buckets
    buckets = {
        "count": np.zeros(n_buckets, dtype=np.int64),
        "retrieval_correct": np.zeros(n_buckets, dtype=np.int64),
        "answer_correct": np.zeros(n_buckets, dtype=np.int64),
        "same_key_recency_error": np.zeros(n_buckets, dtype=np.int64),
        "wrong_key_semantic_error": np.zeros(n_buckets, dtype=np.int64),
        "margin_sum": np.zeros(n_buckets, dtype=np.float64),
    }

    for _ in range(args.batches):
        batch = world.batch(args.batch_size)
        (
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = (tensor.to(device, non_blocking=True) for tensor in batch)

        batch_size = values.shape[0]
        rows = torch.arange(batch_size, device=device)

        logits, scores = model(
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            shuffle=shuffle,
            mask_memory=mask_memory,
            hard=True,
            return_retrieval=True,
        )

        predicted_answer = logits.argmax(dim=-1)
        stats["answer_correct"] += predicted_answer.eq(answers).sum().item()
        stats["episodes"] += batch_size

        if mask_memory:
            continue

        predicted_position = scores.argmax(dim=1)
        stats["retrieval_correct"] += (
            predicted_position.eq(answer_positions).sum().item()
        )

        max_scores = scores.max(dim=1, keepdim=True).values
        stats["tie_at_max"] += (
            scores.ge(max_scores - 1e-6).sum(dim=1).gt(1).sum().item()
        )

        scores_float = scores.to(torch.float32)
        true_score = scores_float[rows, answer_positions]

        masked_scores = scores_float.clone()
        masked_scores[rows, answer_positions] = -math.inf
        best_other = masked_scores.max(dim=1).values

        margin = true_score - best_other
        running["margin"].update(margin)
        hist["margin"].update(margin)
        stats["margin_nonpositive"] += margin.le(0).sum().item()

        probabilities = torch.softmax(scores_float, dim=1)
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
        running["entropy"].update(entropy)

        rank = scores_float.gt(true_score.unsqueeze(1)).sum(dim=1) + 1
        running["true_rank"].update(rank.to(torch.float32))

        query_slots = slots[rows, answer_positions]
        same_key = slots.eq(query_slots.unsqueeze(1))

        dup_count = same_key.sum(dim=1)
        running["dup_count"].update(dup_count.to(torch.float32))
        hist["dup_count"].update(dup_count.to(torch.float32))

        tail_distance = (n_segments - 1) - answer_positions
        running["tail_distance"].update(tail_distance.to(torch.float32))

        predicted_slot = slots[rows, predicted_position]
        wrong = predicted_position.ne(answer_positions)
        same_key_error = wrong & predicted_slot.eq(query_slots)
        wrong_key_error = wrong & predicted_slot.ne(query_slots)
        stats["same_key_recency_error"] += same_key_error.sum().item()
        stats["wrong_key_semantic_error"] += wrong_key_error.sum().item()

        recency = timestamps.to(torch.float32) / recency_denominator
        if alpha != 0.0:
            semantic = (scores_float - beta * recency) / alpha
        else:
            semantic = torch.zeros_like(scores_float)

        semantic_true = semantic[rows, answer_positions]
        running["semantic_true"].update(semantic_true)
        running["recency_term_true"].update(beta * recency[rows, answer_positions])

        semantic_wrong_key = semantic.masked_fill(same_key, -math.inf)
        best_wrong_key_semantic = semantic_wrong_key.max(dim=1).values
        valid = torch.isfinite(best_wrong_key_semantic)

        if valid.any():
            delta_sem = semantic_true[valid] - best_wrong_key_semantic[valid]
            running["delta_sem"].update(delta_sem)
            hist["delta_sem"].update(delta_sem)
            stats["delta_sem_evaluated"] += delta_sem.numel()

            # delta_sem <= 0 означает семантическую инверсию: правильный ключ
            # проигрывает чужому ключу ДО учёта recency. Это не бюджет.
            inverted = delta_sem.le(0)
            stats["semantic_inversion"] += inverted.sum().item()

            positive = delta_sem[~inverted]
            if positive.numel() > 0:
                running["delta_sem_positive"].update(positive)
                if beta != 0.0 and alpha != 0.0:
                    budget_norm = alpha * positive / abs(beta)
                    budget = budget_norm * recency_denominator
                    running["recency_budget"].update(budget)
                    hist["recency_budget"].update(budget)
                    running["recency_budget_norm"].update(budget_norm)
                    hist["recency_budget_norm"].update(budget_norm)

        bucket_index = (
            bucket_by_tail(tail_distance.to(torch.float32), n_buckets, n_segments)
            .cpu()
            .numpy()
        )

        retrieval_hit = predicted_position.eq(answer_positions).cpu().numpy()
        answer_hit = predicted_answer.eq(answers).cpu().numpy()
        same_key_np = same_key_error.cpu().numpy()
        wrong_key_np = wrong_key_error.cpu().numpy()
        margin_np = margin.to(torch.float64).cpu().numpy()

        np.add.at(buckets["count"], bucket_index, 1)
        np.add.at(buckets["retrieval_correct"], bucket_index, retrieval_hit)
        np.add.at(buckets["answer_correct"], bucket_index, answer_hit)
        np.add.at(buckets["same_key_recency_error"], bucket_index, same_key_np)
        np.add.at(buckets["wrong_key_semantic_error"], bucket_index, wrong_key_np)
        np.add.at(buckets["margin_sum"], bucket_index, margin_np)

    episodes = max(1, stats["episodes"])
    evaluated = max(1, stats["delta_sem_evaluated"])

    result = {
        "condition": condition,
        "episodes": stats["episodes"],
        "answer_acc": stats["answer_correct"] / episodes,
        "retrieval_acc": (
            None if mask_memory else stats["retrieval_correct"] / episodes
        ),
        "tie_at_max_rate": None if mask_memory else stats["tie_at_max"] / episodes,
        "margin_nonpositive_rate": (
            None if mask_memory else stats["margin_nonpositive"] / episodes
        ),
        "error_decomposition": (
            None
            if mask_memory
            else {
                "same_key_recency_error": stats["same_key_recency_error"],
                "wrong_key_semantic_error": stats["wrong_key_semantic_error"],
                "same_key_recency_rate": stats["same_key_recency_error"] / episodes,
                "wrong_key_semantic_rate": stats["wrong_key_semantic_error"]
                / episodes,
            }
        ),
        "semantic_health": (
            None
            if mask_memory
            else {
                "delta_sem_evaluated": stats["delta_sem_evaluated"],
                "semantic_inversion": stats["semantic_inversion"],
                "semantic_inversion_rate": stats["semantic_inversion"] / evaluated,
                "semantic_lead_rate": 1.0
                - stats["semantic_inversion"] / evaluated,
            }
        ),
        "running": {name: agg.summary() for name, agg in running.items()},
        "histograms": {name: h.summary() for name, h in hist.items()},
        "tail_buckets": (
            None
            if mask_memory
            else {
                "n_buckets": n_buckets,
                "count": buckets["count"].tolist(),
                "retrieval_acc": [
                    (float(c) / float(n) if n > 0 else None)
                    for c, n in zip(buckets["retrieval_correct"], buckets["count"])
                ],
                "answer_acc": [
                    (float(c) / float(n) if n > 0 else None)
                    for c, n in zip(buckets["answer_correct"], buckets["count"])
                ],
                "same_key_recency_rate": [
                    (float(c) / float(n) if n > 0 else None)
                    for c, n in zip(
                        buckets["same_key_recency_error"], buckets["count"]
                    )
                ],
                "wrong_key_semantic_rate": [
                    (float(c) / float(n) if n > 0 else None)
                    for c, n in zip(
                        buckets["wrong_key_semantic_error"], buckets["count"]
                    )
                ],
                "margin_mean": [
                    (float(s) / float(n) if n > 0 else None)
                    for s, n in zip(buckets["margin_sum"], buckets["count"])
                ],
            }
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Read-only механическая диагностика PHEF retrieval (MFR)."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-keys", type=int, required=True)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, required=True)
    parser.add_argument("--overwrite-p", type=float, default=0.3)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tail-buckets", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--conditions",
        type=str,
        default="real,shuffled,masked",
        help="через запятую: real, shuffled, masked",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for condition in conditions:
        if condition not in {"real", "shuffled", "masked"}:
            parser.error(f"неизвестный condition: {condition}")

    if os.path.exists(args.out):
        parser.error(f"--out уже существует, отказ от перезаписи: {args.out}")

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

    common = dict(
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    alpha = float(model.score_scale.item())
    beta = float(model.recency_scale.item())
    gate = float(torch.sigmoid(model.memory_gate).item())
    ratio = (alpha / abs(beta)) if beta != 0.0 else None

    print(
        f"device={device} checkpoint={args.checkpoint} "
        f"checkpoint_epoch={checkpoint['epoch']} "
        f"checkpoint_sel_acc={checkpoint['sel_acc']:.6f} "
        f"score_scale={alpha:.6f} recency_scale={beta:.6f} "
        f"alpha_over_beta="
        + ("n/a" if ratio is None else f"{ratio:.6f}")
        + f" memory_gate_sigmoid={gate:.6f} "
        f"episodes_per_condition={args.batch_size * args.batches}",
        flush=True,
    )

    results = {}
    for condition in conditions:
        world = BranchingWorld(seed=args.seed + 100_000, **common)
        results[condition] = record(model, world, args, device, condition)
        summary = results[condition]
        health = summary.get("semantic_health")
        print(
            f"[{condition}] answer_acc={summary['answer_acc']:.6f} "
            f"retrieval_acc="
            + (
                "n/a"
                if summary["retrieval_acc"] is None
                else f"{summary['retrieval_acc']:.6f}"
            )
            + " semantic_lead_rate="
            + (
                "n/a"
                if health is None
                else f"{health['semantic_lead_rate']:.6f}"
            )
            + " B_norm_mean="
            + (
                "n/a"
                if summary["running"]["recency_budget_norm"]["mean"] is None
                else f"{summary['running']['recency_budget_norm']['mean']:.4f}"
            ),
            flush=True,
        )

    report = {
        "tool": "memory_flight_recorder",
        "format_version": 2,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sel_acc": checkpoint["sel_acc"],
        "hard_eval": True,
        "answer_chance": 1.0 / args.n_values,
        "retrieval_chance": 1.0 / args.n_segments,
        "scalars": {
            "score_scale": alpha,
            "recency_scale": beta,
            "alpha_over_abs_beta": ratio,
            "delta_sem_threshold_full_episode": (
                (abs(beta) / alpha) if alpha != 0.0 else None
            ),
            "memory_gate_raw": float(model.memory_gate.item()),
            "memory_gate_sigmoid": gate,
            "last_tie_rate": float(model.last_tie_rate),
        },
        "results": results,
        "config": vars(args),
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
