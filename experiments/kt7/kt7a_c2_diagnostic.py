"""
KT-7A: диагностика проигрыша PHEF и эмпирическое закрытие условия C2.

ПОСТАНОВКА
---------
На одних и тех же эпизодах CompactBranchingWorld и одном чекпойнте измеряются:

  raw_cos       косинус (noisy, clean) в СЫРОМ пространстве ключей
  state_cos     косинус (enc(noisy), enc(clean)) после query_encoder
  raw_nn        exact nearest neighbor по сырому зашумлённому запросу
                (ответ = значение последнего вхождения ближайшего ключа)
  enc_nn        то же, но в обученном пространстве enc(codebook) x enc(query),
                БЕЗ recency-члена (изолирует качество энкодера)
  phef          полная модель (semantic + recency), hard retrieval

ВЕРДИКТ (зафиксирован до запуска)
---------------------------------
  ENCODER_AMPLIFIES   state_cos < raw_cos - 0.05 при sigma=0.50:
                      обученное представление усиливает возмущение
  ENCODER_NEUTRAL     |state_cos - raw_cos| <= 0.05
  ENCODER_DENOISES    state_cos > raw_cos + 0.05 (маловероятно при
                      поточечном детерминированном энкодере)

  C2_CLOSED_NEGATIVE  raw_nn доминирует над phef при sigma=0.50
                      (Wilson-нижняя граница raw_nn > верхняя phef) И
                      enc_nn <= raw_nn: переобучение под шум не обещает
                      паритета, т.к. энкодер уже сейчас не извлекает
                      из запроса ничего сверх сырого косинуса.

ЯКОРЬ ВОСПРОИЗВОДИМОСТИ
-----------------------
При sigma=0.50 ожидается phef ~ 0.297 (KT-1) и raw_nn ~ 0.996 (KT-2A).
Расхождение > 0.05 по phef печатает предупреждение ANCHOR_MISMATCH.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

HARD_EVAL = True


def load_evaluator_module(repo: Path):
    candidates = [
        repo / "experiments" / "eval_recall_compact.py",
        repo / "phef" / "experiments" / "eval_recall_compact.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "eval_recall_compact", str(path)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"evaluator: {path}", flush=True)
            return module
    raise SystemExit(
        "eval_recall_compact.py не найден ни в одном из: "
        + ", ".join(str(p) for p in candidates)
    )


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def last_occurrence_values(slots_row, values_row, ranks):
    """values_row[last position where slots_row == rank] для каждого rank."""
    out = torch.empty_like(ranks)
    for i, r in enumerate(ranks.tolist()):
        pos = (slots_row == r).nonzero()
        out[i] = values_row[pos[-1].item()]
    return out


@torch.inference_mode()
def run_sigma(model, world, device, sigma, n_batches, batch_size, gen):
    from phef.query_noise import key_scale_from_codebook, perturb_query_keys

    raw_cos_sum = 0.0
    state_cos_sum = 0.0
    n_cos = 0
    hits = {"raw_nn": 0, "enc_nn": 0, "phef_answer": 0, "phef_retrieval": 0}
    total = 0
    n_eff_sum = 0

    for _ in range(n_batches):
        (
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = [t.to(device) for t in world.batch(batch_size)]
        n_eff_sum += sum(world.last_n_eff)

        scale = key_scale_from_codebook(codebook)
        clean = query_keys
        noisy = perturb_query_keys(query_keys, sigma, scale, gen)

        raw_cos_sum += torch.nn.functional.cosine_similarity(
            clean, noisy, dim=-1
        ).mean().item()
        clean_state = model.query_encoder(clean)
        noisy_state = model.query_encoder(noisy)
        state_cos_sum += torch.nn.functional.cosine_similarity(
            clean_state, noisy_state, dim=-1
        ).mean().item()
        n_cos += 1

        for b in range(batch_size):
            # --- raw NN: argmax косинуса по сырому codebook ---
            # codebook нормирован на сферу, noisy нет — нормируем здесь,
            # чтобы скоринг был именно косинусом.
            q_raw = torch.nn.functional.normalize(noisy[b], dim=0)
            raw_rank = int((codebook[b, : world.last_n_eff[b]] @ q_raw).argmax())
            raw_val = last_occurrence_values(
                slots[b], values[b], torch.tensor([raw_rank], device=device)
            )[0]
            hits["raw_nn"] += int(raw_val == answers[b])

            # --- enc NN: argmax в обученном пространстве, без recency ---
            enc_keys = model.key_encoder(codebook[b, : world.last_n_eff[b]])
            enc_rank = int((enc_keys @ noisy_state[b]).argmax())
            enc_val = last_occurrence_values(
                slots[b], values[b], torch.tensor([enc_rank], device=device)
            )[0]
            hits["enc_nn"] += int(enc_val == answers[b])

        # --- полный PHEF, hard ---
        logits, scores = model(
            slots,
            codebook,
            values,
            timestamps,
            noisy,
            hard=HARD_EVAL,
            return_retrieval=True,
        )
        hits["phef_answer"] += int(
            (logits.argmax(dim=-1) == answers).sum().item()
        )
        hits["phef_retrieval"] += int(
            (scores.argmax(dim=1) == answer_positions).sum().item()
        )
        total += answers.numel()

    result = {
        "sigma": sigma,
        "episodes": total,
        "n_eff_mean": n_eff_sum / total,
        "raw_cos": raw_cos_sum / n_cos,
        "state_cos": state_cos_sum / n_cos,
    }
    for name, k in hits.items():
        lo, hi = wilson(k, total)
        result[name] = {
            "correct": k,
            "acc": k / total,
            "wilson95": [lo, hi],
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-keys", type=int, default=1_073_741_824)
    ap.add_argument("--n-values", type=int, default=64)
    ap.add_argument("--n-segments", type=int, default=131_072)
    ap.add_argument("--overwrite-p", type=float, default=0.3)
    ap.add_argument("--key-dim", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--sigma-list", default="0.30,0.50,1.00")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=7)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "phef"))
    sys.path.insert(0, str(repo))

    ev = load_evaluator_module(repo)
    from phef.model import PHEF

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    torch.manual_seed(args.seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.noise_seed)

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=True,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    world = ev.CompactBranchingWorld(
        seed=args.seed + 100_000,
        key_dim=args.key_dim,
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        overwrite_p=args.overwrite_p,
    )

    print(
        f"device={device} checkpoint={args.checkpoint} "
        f"checkpoint_epoch={checkpoint.get('epoch', checkpoint.get('best_epoch'))} "
        f"sel_acc={checkpoint.get('sel_acc', checkpoint.get('best_acc')):.6f} "
        f"n_segments={args.n_segments} "
        f"episodes={args.eval_batch_size * args.eval_batches}",
        flush=True,
    )

    sigmas = [float(s) for s in args.sigma_list.split(",")]
    points = [
        run_sigma(
            model, world, device, s, args.eval_batches,
            args.eval_batch_size, gen,
        )
        for s in sigmas
    ]

    for p in points:
        print(
            f"sigma={p['sigma']:.2f} "
            f"raw_cos={p['raw_cos']:.4f} state_cos={p['state_cos']:.4f} "
            f"raw_nn={p['raw_nn']['acc']:.4f} "
            f"enc_nn={p['enc_nn']['acc']:.4f} "
            f"phef={p['phef_answer']['acc']:.4f} "
            f"(retrieval={p['phef_retrieval']['acc']:.4f})",
            flush=True,
        )

    ref = min(points, key=lambda p: abs(p["sigma"] - 0.50))
    gap = ref["raw_cos"] - ref["state_cos"]
    if gap > 0.05:
        enc_verdict = "ENCODER_AMPLIFIES"
    elif gap < -0.05:
        enc_verdict = "ENCODER_DENOISES"
    else:
        enc_verdict = "ENCODER_NEUTRAL"

    c2_closed = (
        ref["raw_nn"]["wilson95"][0] > ref["phef_answer"]["wilson95"][1]
        and ref["enc_nn"]["acc"] <= ref["raw_nn"]["acc"]
    )
    c2_verdict = "C2_CLOSED_NEGATIVE" if c2_closed else "C2_OPEN"

    anchor_delta = abs(ref["phef_answer"]["acc"] - 0.296875)
    anchor = "OK" if anchor_delta <= 0.05 else "ANCHOR_MISMATCH"

    report = {
        "schema": "kt7a_c2_diagnostic_v1",
        "checkpoint": args.checkpoint,
        "config": vars(args),
        "points": points,
        "encoder_verdict": enc_verdict,
        "c2_verdict": c2_verdict,
        "anchor_kt1": anchor,
        "anchor_delta": anchor_delta,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"encoder_verdict={enc_verdict}", flush=True)
    print(f"c2_verdict={c2_verdict}", flush=True)
    print(f"anchor_kt1={anchor} (delta={anchor_delta:.4f})", flush=True)
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
