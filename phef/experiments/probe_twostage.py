"""Двухступенчатое считывание на сохранённых весах, без обучения.

Шаг 8.2, часть вторая. Первая часть (probe_gap.py) уже измерила:
own_spread ровно ноль у 1.000000 эпизодов на всех 15 чекпойнтах,
то есть события одного слота дают побитово равные семантические
скоры, и порог почти-равенства по данным равен нулю. Поэтому
tau по умолчанию 0.0, а не 1e-6 из model.py:98.

Ридаут:
  ступень 1 — кандидаты C = {s : semantic_s >= max(semantic) - tau};
  ступень 2 — среди C берётся событие с наибольшим timestamp.
Давность здесь разрешает ничьи и НЕ наклоняет спектр скоров.

Хвост модели (model.py:58, :94, :115-117) повторён здесь, потому что
forward не позволяет задать индекс извне. Повтор проверяется на
каждом чекпойнте: тот же хвост прогоняется с индексом штатного
аргмакса, и answer_acc обязан совпасть с forward побитово. Если не
совпал — скрипт останавливается.
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


def head_from_index(model, query_state, values, index):
    """Хвост модели при заданном индексе выбранного события.

    Повторяет model.py:94 (value_emb), :115 (gate), :116 (norm),
    :117 (head). Больше ничего не делает.
    """
    rows = torch.arange(values.shape[0], device=values.device)
    memory_values = model.value_emb(values)
    memory_state = memory_values[rows, index]
    gate = torch.sigmoid(model.memory_gate)
    hidden = model.norm(query_state + gate * memory_state)
    return model.head(hidden)


class Accum:
    """Счётчики одного варианта ридаута."""

    def __init__(self, n_buckets, device):
        self.n_buckets = n_buckets
        self.b_ans = torch.zeros(n_buckets, dtype=torch.float64, device=device)
        self.b_slot = torch.zeros(n_buckets, dtype=torch.float64, device=device)
        self.b_tot = torch.zeros(n_buckets, dtype=torch.float64, device=device)
        self.correct = 0
        self.slot_ok = 0
        self.last_ok = 0
        self.total = 0

    def add(self, is_correct, slot_hit, is_last, bucket):
        self.correct += int(is_correct.sum().item())
        self.slot_ok += int(slot_hit.sum().item())
        self.last_ok += int(is_last.sum().item())
        self.total += int(is_correct.numel())
        for b in range(self.n_buckets):
            sel = bucket.eq(b)
            self.b_tot[b] += sel.sum().to(torch.float64)
            self.b_ans[b] += is_correct[sel].sum().to(torch.float64)
            self.b_slot[b] += slot_hit[sel].sum().to(torch.float64)

    def summary(self):
        def per_bucket(arr):
            return [
                float(arr[b].item() / self.b_tot[b].item())
                if self.b_tot[b].item() > 0
                else float("nan")
                for b in range(self.n_buckets)
            ]

        ans_prof = per_bucket(self.b_ans)
        slot_prof = per_bucket(self.b_slot)
        filled = [
            i for i in range(self.n_buckets) if self.b_tot[i].item() > 0
        ]

        return {
            "answer_acc": self.correct / self.total,
            "slot_acc": self.slot_ok / self.total,
            "last_acc": self.last_ok / self.total,
            "ans_prof": ans_prof,
            "slot_prof": slot_prof,
            "counts": [float(v.item()) for v in self.b_tot],
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


@torch.inference_mode()
def run(model, world, device, batch_size, n_batches, n_buckets, tau,
        base_score, base_recency):
    model.eval()

    acc_base = Accum(n_buckets, device)
    acc_two = Accum(n_buckets, device)
    tail_mismatch = 0
    tail_max_diff = 0.0
    cand_sum = 0.0
    total = 0

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

        rows = torch.arange(answers.numel(), device=answers.device)
        slots = world.last_slots.to(device)
        query_slots = world.last_query_slots.to(device)

        n_seg = world.n_segments
        distance = (
            n_seg - 1 - answer_positions
        ).float() / max(1, n_seg - 1)
        bucket = (distance * n_buckets).long().clamp(0, n_buckets - 1)

        logits_fwd, scores_fwd = model(
            event_keys,
            values,
            timestamps,
            query_keys,
            hard=True,
            return_retrieval=True,
        )
        index_fwd = scores_fwd.argmax(dim=1)

        acc_base.add(
            logits_fwd.argmax(dim=-1).eq(answers),
            slots[rows, index_fwd].eq(query_slots),
            index_fwd.eq(answer_positions),
            bucket,
        )

        query_state = model.query_encoder(query_keys)
        logits_tail = head_from_index(model, query_state, values, index_fwd)
        tail_mismatch += int(
            logits_tail.argmax(dim=-1)
            .ne(logits_fwd.argmax(dim=-1))
            .sum()
            .item()
        )
        tail_max_diff = max(
            tail_max_diff,
            float((logits_tail - logits_fwd).abs().max().item()),
        )

        with torch.no_grad():
            model.score_scale.fill_(1.0)
            model.recency_scale.fill_(0.0)
        _, sem = model(
            event_keys,
            values,
            timestamps,
            query_keys,
            hard=True,
            return_retrieval=True,
        )
        with torch.no_grad():
            model.score_scale.fill_(base_score)
            model.recency_scale.fill_(base_recency)

        cand = sem.ge(sem.max(dim=1, keepdim=True).values - tau)
        cand_sum += float(cand.sum().item())
        total += int(answers.numel())

        ts = timestamps.float()
        masked_ts = torch.where(
            cand, ts, torch.full_like(ts, float("-inf"))
        )
        index_two = masked_ts.argmax(dim=1)

        logits_two = head_from_index(model, query_state, values, index_two)
        acc_two.add(
            logits_two.argmax(dim=-1).eq(answers),
            slots[rows, index_two].eq(query_slots),
            index_two.eq(answer_positions),
            bucket,
        )

    return {
        "baseline": acc_base.summary(),
        "twostage": acc_two.summary(),
        "tail_mismatch": tail_mismatch,
        "tail_max_abs_diff": tail_max_diff,
        "mean_cand": cand_sum / total,
        "n_episodes": total,
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
        "--tau",
        type=float,
        default=0.0,
        help="порог почти-равенства в СЫРЫХ единицах semantic; "
        "0.0 измерено по данным в probe_gap.py",
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
        help="смещение сида оценочного мира относительно cfg['seed']",
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
    print(f"tau (сырые единицы semantic): {args.tau}")
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

        world = make_world(cfg, eval_seed, device)
        res = run(
            model,
            world,
            device,
            eval_bs,
            eval_nb,
            args.n_buckets,
            args.tau,
            base_score,
            base_recency,
        )

        restored = (
            model.score_scale.item() == base_score
            and model.recency_scale.item() == base_recency
        )
        if not restored:
            raise SystemExit("множители модели не восстановились, стоп")

        if res["tail_mismatch"] != 0:
            print(
                f"  хвост НЕ воспроизводит forward: "
                f"{res['tail_mismatch']} расхождений, "
                f"max|dlogits|={res['tail_max_abs_diff']:.3e}"
            )
            raise SystemExit("повтор хвоста неверен, стоп")

        print(
            f"  хвост воспроизводит forward: да "
            f"(max|dlogits|={res['tail_max_abs_diff']:.3e}), "
            f"множители восстановлены"
        )
        print(f"  средний размер C: {res['mean_cand']:.3f}")

        b = res["baseline"]
        t = res["twostage"]
        print(
            f"\n  {'вариант':>14} {'answer':>8} {'slot':>8} "
            f"{'last':>8} {'ans_drop':>9} {'slot_drop':>10}"
        )
        for label, r in (("база (forward)", b), ("двухступ.", t)):
            print(
                f"  {label:>14} {r['answer_acc']:8.4f} "
                f"{r['slot_acc']:8.4f} {r['last_acc']:8.4f} "
                f"{r['ans_drop']:9.4f} {r['slot_drop']:10.4f}"
            )
        print(
            f"  {'разница':>14} "
            f"{t['answer_acc'] - b['answer_acc']:8.4f} "
            f"{t['slot_acc'] - b['slot_acc']:8.4f} "
            f"{t['last_acc'] - b['last_acc']:8.4f} "
            f"{t['ans_drop'] - b['ans_drop']:9.4f} "
            f"{t['slot_drop'] - b['slot_drop']:10.4f}"
        )

        print(f"\n  slot_hit по дистанции (свежее -> далёкое):")
        header = " ".join(f"b{i}".rjust(7) for i in range(args.n_buckets))
        print(f"  {'вариант':>14} {header}")
        for label, r in (("база", b), ("двухступ.", t)):
            print(
                f"  {label:>14} "
                + " ".join(f"{v:7.4f}" for v in r["slot_prof"])
            )
        print(
            f"\n  n по бакетам: {[int(c) for c in b['counts']]}"
        )

        everything.append(
            {
                "path": path,
                "config": cfg,
                "tau": args.tau,
                "eval_seed_offset": args.eval_seed_offset,
                "eval_seed": eval_seed,
                "base_recency": base_recency,
                "base_score": base_score,
                "mean_cand": res["mean_cand"],
                "tail_max_abs_diff": res["tail_max_abs_diff"],
                "baseline": b,
                "twostage": t,
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
