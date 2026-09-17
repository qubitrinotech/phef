"""
kt1_depth_invariance.py -- проверка инвариантности кривой baseline по глубине.

Заменяет сломанную функцию check_depth_invariance из kt1_run.py.
В той версии блок подбора delta был написан некорректно (TypeError: 'int' object
is not iterable) и фактически всегда использовал delta = 3.0. Здесь delta
подбирается по validation независимо для каждой пары (глубина, sigma), как и
должно быть.

НАЗНАЧЕНИЕ. Baseline (hash map, quantized dict) в KT-1 прогоняется на
редуцированной глубине 20 000 segments, тогда как PHEF -- на 14M. Расхождение
глубин требует эмпирического обоснования, иначе это методологическая дыра:
рецензент справедливо спросит, не выиграл ли PHEF просто за счёт того, что
baseline мерили на другой задаче.

Скрипт проверяет гипотезу: кривая baseline по sigma не зависит от n_segments в
интересующем режиме. Обоснование гипотезы: и hash map, и quantized dict имеют
O(1) доступ и не имеют capacity-ограничения; при росте глубины растёт только
число записей в таблице, а вероятность попадания возмущённого ключа в верный
бакет определяется отношением sigma к delta, а не длиной истории.

ЗАПУСК:
    python kt1_depth_invariance.py                    # быстрая проверка
    python kt1_depth_invariance.py --episodes 32 --depths 5000 20000 80000
    python kt1_depth_invariance.py --out artifacts/kt1/kt1_depth_invariance.json

ИНТЕРПРЕТАЦИЯ. Если 95% интервалы Уилсона для всех глубин перекрываются на
каждом уровне sigma -- прогон baseline на редуцированной глубине обоснован, и
этот факт фиксируется в отчёте KT-1. Если хотя бы на одном sigma интервалы
расходятся -- baseline обязан гоняться на 14M, и бюджет машинного времени
пересчитывается.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kt1_baselines import (  # noqa: E402
    DELTA_GRID,
    QuantizedDict,
    key_scale,
    make_episode,
    perturb_keys,
    select_delta,
)
from kt1_stats import separated, wilson  # noqa: E402

SIGMAS = (0.0, 0.01, 0.03, 0.10, 0.30)
DEPTHS = (5_000, 20_000)


def run(
    depths: tuple[int, ...] = DEPTHS,
    sigmas: tuple[float, ...] = SIGMAS,
    episodes: int = 16,
    seed: int = 7,
    delta_grid: tuple[float, ...] = DELTA_GRID,
) -> dict:
    """Прогоняет quantized dict по сетке (глубина x sigma) с честным подбором delta."""
    counts: dict[int, list[int]] = {}

    for n_seg in depths:
        row: list[int] = []
        for sigma in sigmas:
            hits = 0
            for ep in range(episodes):
                d = make_episode(n_seg, rng=np.random.default_rng([seed, ep]))
                # validation split: отдельный эпизод той же природы, меньшей длины
                vd = make_episode(
                    max(1024, n_seg // 100),
                    rng=np.random.default_rng([seed, ep, 999]),
                )
                best_delta, _ = select_delta(
                    vd["keys"], vd["values"], vd["query_key"], vd["answer"],
                    sigma, key_scale(vd["keys"]),
                    np.random.default_rng([seed, ep, 999]),
                    delta_grid=delta_grid,
                )
                sc = key_scale(d["keys"])
                m = QuantizedDict(delta=best_delta, scale=sc)
                m.build(d["keys"], d["values"])
                q = perturb_keys(
                    d["query_key"], sigma, sc,
                    np.random.default_rng([seed, ep, int(sigma * 1e6)]),
                )
                hits += int(np.all(m.query(q) == d["answer"]))
            row.append(hits)
        counts[n_seg] = row
        print(
            f"  n_segments={n_seg:>7}: "
            + "  ".join(f"{h}/{episodes}={h/episodes:.3f}" for h in row)
        )

    # попарное сравнение крайних глубин на каждом sigma
    lo_d, hi_d = depths[0], depths[-1]
    checks = []
    all_overlap = True
    for i, sigma in enumerate(sigmas):
        a = (counts[lo_d][i], episodes)
        b = (counts[hi_d][i], episodes)
        sep = separated(a, b)
        if sep:
            all_overlap = False
        la, ha = wilson(*a)
        lb, hb = wilson(*b)
        checks.append(
            {
                "sigma": sigma,
                f"depth_{lo_d}": {"hits": a[0], "ci": [round(la, 4), round(ha, 4)]},
                f"depth_{hi_d}": {"hits": b[0], "ci": [round(lb, 4), round(hb, 4)]},
                "intervals_overlap": not sep,
            }
        )

    return {
        "schema": "kt1_depth_invariance_v1",
        "mechanism": "quantized_dict",
        "depths": list(depths),
        "sigmas": list(sigmas),
        "episodes_per_cell": episodes,
        "seed": seed,
        "counts": {str(k): v for k, v in counts.items()},
        "pairwise": checks,
        "all_intervals_overlap": all_overlap,
        "conclusion": (
            "Кривая baseline инвариантна по глубине в проверенном диапазоне. "
            "Прогон baseline на редуцированной глубине ОБОСНОВАН."
            if all_overlap
            else "ОБНАРУЖЕНА зависимость от глубины. Baseline обязан гоняться на "
            "целевой глубине 14M. Бюджет машинного времени пересчитать."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Инвариантность baseline по глубине")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("Проверка инвариантности кривой baseline по глубине")
    print(f"sigmas={SIGMAS}, episodes={a.episodes}, depths={a.depths}\n")

    rec = run(tuple(a.depths), SIGMAS, a.episodes, a.seed)

    print("\nПопарное сравнение крайних глубин:")
    for c in rec["pairwise"]:
        print(f"  sigma={c['sigma']:<5} перекрываются={c['intervals_overlap']}")

    print(f"\n{rec['conclusion']}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        print(f"\n-> {a.out}")

    return 0 if rec["all_intervals_overlap"] else 2


if __name__ == "__main__":
    sys.exit(main())
