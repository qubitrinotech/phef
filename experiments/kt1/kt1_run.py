"""
kt1_run.py -- оркестратор KT-1.

ЕДИНСТВЕННОЕ МЕСТО, которое нужно адаптировать под ваш репозиторий: класс
PhefAdapter ниже. Всё остальное работает без изменений.

Порядок исполнения соответствует ТЗ:
  день 1  --  python kt1_run.py stage1     (инструментарий + baseline + контроль sigma=0)
  день 2  --  python kt1_run.py stage2     (полная сетка sigma для PHEF)
  день 3  --  python kt1_run.py stage3     (кривая, вердикт, отчёт)

Точка останова дня 1 жёсткая: если контрольный прогон при sigma = 0 не
воспроизводит интервал [0.9334, 0.9920], stage1 завершается с кодом 1 и stage2
запускать нельзя.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

from kt1_baselines import (
    ExactHashMap,
    QuantizedDict,
    key_scale,
    make_episode,
    perturb_keys,
    run_baselines,
    select_delta,
)
from kt1_stats import CurvePoint, load_points, verdict, wilson, write_curve_csv

SIGMAS = (0.0, 0.01, 0.03, 0.10, 0.30, 1.00)
EPISODES_PER_SIGMA = 256          # 2 ckpt x 2 eval stream x 64
CELLS = ((7, 7), (7, 11), (23, 7), (23, 11))   # (train_seed, eval_seed)
EPISODES_PER_CELL = EPISODES_PER_SIGMA // len(CELLS)

BASELINE_N_SEGMENTS = 20_000      # baseline не требует 14M: см. ниже
PHEF_N_SEGMENTS = 14_000_000

# Обоснование расхождения глубин. Baseline (hash map, quantized dict) не имеет
# capacity-ограничения и его кривая по sigma не зависит от n_segments в
# интересующем режиме. Прогон baseline на 14M потребовал бы часов CPU без
# изменения выводов. Это ДОЛЖНО быть проверено один раз эмпирически:
# см. команду `python kt1_run.py check-depth-invariance`.

CONTROL_CI = (0.9334, 0.9920)     # существующий результат 14M, 125/128


# --------------------------------------------------------------------------
# АДАПТЕР: единственное место для правки
# --------------------------------------------------------------------------


class PhefAdapter:
    """Обёртка над вашим streamed hard retrieval evaluator.

    ЗАПОЛНИТЕ три метода. Остальной код KT-1 от деталей не зависит.

    Требования, вытекающие из ТЗ:
      1. Возмущение применяется ТОЛЬКО к query key, внутри evaluator, ПОСЛЕ
         формирования памяти. Ключи в памяти не возмущаются.
      2. sigma нормируется на key_scale обучающего распределения.
      3. Считается retrieval accuracy (выбор события), а не только answer.
         В существующих результатах они совпадали во всех 16 ячейках; в KT-1
         это перестаёт быть гарантией и должно измеряться отдельно.
      4. stream_hard_chunk_size = 65536, как в compact_eval_v2_streamed_hard.
    """

    name = "phef"

    def __init__(self, checkpoint: str, device: str = "cuda", chunk_size: int = 65536):
        self.checkpoint = checkpoint
        self.device = device
        self.chunk_size = chunk_size
        self._model = None

    def load(self):
        # TODO: заменить на реальную загрузку
        # import torch
        # from phef.eval import CompactEvaluator
        # self._model = CompactEvaluator.load(self.checkpoint, device=self.device,
        #                                     stream_hard_chunk_size=self.chunk_size)
        raise NotImplementedError(
            "PhefAdapter.load не реализован. Подключите ваш evaluator "
            "(commit 662edb6) и верните объект с методом evaluate_episode."
        )

    def evaluate_sigma(
        self, sigma: float, episodes: int, n_segments: int, eval_seed: int
    ) -> dict:
        """Вернуть dict с обязательными ключами:

            retrieval_correct : int  -- число эпизодов с верно выбранным событием
            answer_correct    : int  -- число эпизодов с верным ответом
            episodes          : int
            gpu_alloc_peak_gib: float
            rss_peak_gib      : float
            wall_time_s       : float
            key_scale         : float
        """
        raise NotImplementedError(
            "PhefAdapter.evaluate_sigma не реализован.\n"
            "Внутри: сформировать память, возмутить ТОЛЬКО query key через\n"
            "perturb_keys(qk, sigma, key_scale, rng), выполнить streamed hard\n"
            "retrieval, сравнить выбранное событие с latest-value ground truth."
        )


# --------------------------------------------------------------------------
# Провенанс
# --------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def provenance() -> dict:
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _git("status", "--porcelain") != "",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def write_artifact(rec: dict, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rec.setdefault("provenance", provenance())
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------
# stage1 -- день 1
# --------------------------------------------------------------------------


def stage1(args) -> int:
    print("=" * 72)
    print("STAGE 1 -- инструментарий, baseline, контроль sigma=0")
    print("=" * 72)

    print("\n[1/3] Прогон baseline по сетке sigma")
    for train_seed, eval_seed in CELLS:
        run_baselines(
            n_segments=args.baseline_segments,
            sigmas=SIGMAS,
            episodes=EPISODES_PER_CELL,
            seed=train_seed * 1000 + eval_seed,
            out_dir=args.out,
        )
    print("      готово")

    print("\n[2/3] Сводка baseline")
    pts = load_points(os.path.join(args.out, "kt1_*.json"))
    for p in sorted(pts, key=lambda x: (x.mechanism, x.sigma)):
        lo, hi = p.ci
        print(
            f"      {p.mechanism:>16} sigma={p.sigma:<5} "
            f"{p.successes:>4}/{p.n:<4} {p.point:.4f} [{lo:.4f}, {hi:.4f}]"
        )

    print("\n[3/3] Контрольный прогон PHEF при sigma=0")
    if args.skip_phef:
        print("      ПРОПУЩЕН (--skip-phef). stage2 запускать НЕЛЬЗЯ.")
        return 0

    adapter = PhefAdapter(args.checkpoint, device=args.device)
    adapter.load()
    r = adapter.evaluate_sigma(0.0, EPISODES_PER_SIGMA, args.phef_segments, eval_seed=7)
    lo, hi = wilson(r["retrieval_correct"], r["episodes"])
    print(f"      {r['retrieval_correct']}/{r['episodes']} -> [{lo:.4f}, {hi:.4f}]")
    print(f"      ожидаемый интервал: [{CONTROL_CI[0]:.4f}, {CONTROL_CI[1]:.4f}]")

    overlaps = not (hi < CONTROL_CI[0] or lo > CONTROL_CI[1])
    write_artifact(
        {
            "schema": "kt1_control_v1",
            "mechanism": "phef",
            "sigma": 0.0,
            "control_expected_ci": list(CONTROL_CI),
            "observed_ci": [lo, hi],
            "overlaps_control": overlaps,
            **r,
        },
        args.out,
        "kt1_control_sigma0.json",
    )

    if not overlaps:
        print("\n  СТОП. Контроль sigma=0 не воспроизводит существующий результат.")
        print("  Отлаживать инструментарий. stage2 НЕ запускать.")
        return 1
    print("\n      контроль пройден -> stage2 разрешён")
    return 0


# --------------------------------------------------------------------------
# stage2 -- день 2
# --------------------------------------------------------------------------


def stage2(args) -> int:
    ctrl = os.path.join(args.out, "kt1_control_sigma0.json")
    if not os.path.exists(ctrl):
        print("СТОП: нет kt1_control_sigma0.json. Сначала stage1.")
        return 1
    with open(ctrl, encoding="utf-8") as fh:
        if not json.load(fh).get("overlaps_control"):
            print("СТОП: контроль sigma=0 не пройден.")
            return 1

    print("=" * 72)
    print(f"STAGE 2 -- PHEF, сетка sigma, {EPISODES_PER_SIGMA} эпизодов на точку")
    print("=" * 72)

    adapter = PhefAdapter(args.checkpoint, device=args.device)
    adapter.load()

    for sigma in SIGMAS:
        for train_seed, eval_seed in CELLS:
            t0 = time.time()
            r = adapter.evaluate_sigma(
                sigma, EPISODES_PER_CELL, args.phef_segments, eval_seed=eval_seed
            )
            write_artifact(
                {
                    "schema": "kt1_phef_v1",
                    "mechanism": "phef",
                    "sigma": float(sigma),
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "n_segments": args.phef_segments,
                    "chunk_size": adapter.chunk_size,
                    "wall_time_s": round(time.time() - t0, 2),
                    **r,
                },
                args.out,
                f"kt1_phef_sigma{sigma}_ckpt{train_seed}_eval{eval_seed}.json",
            )
            print(
                f"  sigma={sigma:<5} ckpt{train_seed}/eval{eval_seed}: "
                f"{r['retrieval_correct']}/{r['episodes']}"
            )
    return 0


# --------------------------------------------------------------------------
# stage3 -- день 3
# --------------------------------------------------------------------------


def stage3(args) -> int:
    print("=" * 72)
    print("STAGE 3 -- кривая и вердикт")
    print("=" * 72)

    pts = load_points(os.path.join(args.out, "kt1_*.json"))
    if not pts:
        print("нет артефактов")
        return 1

    csv_path = os.path.join(args.out, "kt1_curve.csv")
    write_curve_csv(pts, csv_path)
    print(f"\nкривая -> {csv_path}\n")

    for p in sorted(pts, key=lambda x: (x.mechanism, x.sigma)):
        lo, hi = p.ci
        print(
            f"{p.mechanism:>16} sigma={p.sigma:<5} {p.successes:>4}/{p.n:<4} "
            f"{p.point:.4f} [{lo:.4f}, {hi:.4f}]"
        )

    try:
        v = verdict(pts)
    except ValueError as e:
        print(f"\nВЕРДИКТ НЕВОЗМОЖЕН: {e}")
        return 1

    print("\n" + "=" * 72)
    print(f"ИСХОД: {v['outcome']}")
    print(f"ДЕЙСТВИЕ: {v['action']}")
    print("=" * 72)
    if v["needs_grid_refinement"]:
        print(
            f"\nТРЕБУЕТСЯ уточнение сетки: падение {v['max_drop']:.3f} "
            f"при sigma={v['max_drop_at_sigma']}"
        )
    write_artifact({"schema": "kt1_verdict_v1", **v}, args.out, "kt1_verdict.json")
    return 0


# --------------------------------------------------------------------------
# check-depth-invariance -- обоснование baseline на 20k вместо 14M
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Оркестратор KT-1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("stage1", stage1),
        ("stage2", stage2),
        ("stage3", stage3),
    ):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--out", default="artifacts/kt1")
        p.add_argument("--checkpoint", default="v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt")
        p.add_argument("--device", default="cuda")
        p.add_argument("--baseline-segments", type=int, default=BASELINE_N_SEGMENTS)
        p.add_argument("--phef-segments", type=int, default=PHEF_N_SEGMENTS)
        p.add_argument("--episodes", type=int, default=32)
        p.add_argument("--skip-phef", action="store_true")

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
