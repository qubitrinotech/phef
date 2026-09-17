"""
kt1_stats.py -- статистический слой KT-1.

Не зависит от torch и от кода PHEF. Не требует GPU.
Запускается автономно: python kt1_stats.py --selftest

Назначение:
  1. Интервалы Уилсона для всех точек.
  2. Проверка непересечения интервалов (критерий успеха KT-1).
  3. Планирование числа эпизодов.
  4. Сводная таблица кривой из артефактов прогонов.

Обоснование отдельного модуля: критерии KT-1 сформулированы через непересечение
интервалов, а не через сравнение точечных оценок. Логика должна быть тестируема
без GPU и неизменна между прогонами.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Iterable

Z95 = 1.959963984540054

# --------------------------------------------------------------------------
# Интервалы
# --------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Интервал Уилсона (score interval) для биномиальной пропорции.

    Выбран как метод по умолчанию: корректен при малых n и при p вблизи 0 и 1,
    не выходит за [0, 1], не вырождается в нулевую ширину при successes == 0.
    Последнее существенно: shuffled retrieval == 0 -- основной результат набора.
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} вне [0, {n}]")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def separated(a: tuple[int, int], b: tuple[int, int], z: float = Z95) -> bool:
    """True, если 95% интервалы не перекрываются, т.е. различие разрешено.

    a, b -- кортежи (successes, n).
    Порядок не важен: проверяется факт непересечения в любую сторону.
    """
    la, ha = wilson(*a, z=z)
    lb, hb = wilson(*b, z=z)
    return hb < la or ha < lb


def dominates(a: tuple[int, int], b: tuple[int, int], z: float = Z95) -> bool:
    """True, если нижняя граница a строго выше верхней границы b.

    Это операционализация критерия успеха KT-1: механизм превосходит baseline
    при данном sigma тогда и только тогда, когда dominates(phef, baseline).
    """
    la, _ = wilson(*a, z=z)
    _, hb = wilson(*b, z=z)
    return la > hb


def contains(successes: int, n: int, value: float, z: float = Z95) -> bool:
    """Содержит ли интервал заданное значение. Для проверки chance-уровня."""
    lo, hi = wilson(successes, n, z=z)
    return lo <= value <= hi


# --------------------------------------------------------------------------
# Планирование
# --------------------------------------------------------------------------


def min_episodes_to_separate(
    p_ref: float, p_alt: float, z: float = Z95, n_max: int = 100_000
) -> int | None:
    """Минимальное n, при котором истинные p_ref и p_alt разрешаются интервалами.

    Используется для обоснования числа эпизодов на точку sigma.
    """
    if p_ref == p_alt:
        return None
    n = 8
    while n <= n_max:
        a = (round(p_ref * n), n)
        b = (round(p_alt * n), n)
        if separated(a, b, z=z):
            return n
        n += 4
    return None


def plan_table(p_ref: float = 0.976, alts: Iterable[float] = (0.95, 0.90, 0.85, 0.80, 0.70, 0.50)):
    return [(a, min_episodes_to_separate(p_ref, a)) for a in alts]


# --------------------------------------------------------------------------
# Точки кривой
# --------------------------------------------------------------------------


@dataclass
class CurvePoint:
    mechanism: str
    sigma: float
    successes: int
    n: int
    delta: float | None = None  # шаг сетки для quantized dict, иначе None

    @property
    def point(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.successes, self.n)

    def row(self) -> dict:
        lo, hi = self.ci
        return {
            "mechanism": self.mechanism,
            "sigma": self.sigma,
            "delta": self.delta,
            "successes": self.successes,
            "n": self.n,
            "point": round(self.point, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "ci_width_pp": round((hi - lo) * 100, 3),
        }


def verdict(points: list[CurvePoint], mechanism: str = "phef", baseline: str = "quantized_dict") -> dict:
    """Применяет критерии раздела 4 ТЗ. Формулировки не подлежат ревизии постфактум."""
    by = {}
    for p in points:
        by.setdefault(p.mechanism, {})[p.sigma] = p

    if mechanism not in by:
        raise ValueError(f"нет точек механизма {mechanism!r}")
    if baseline not in by:
        raise ValueError(
            f"нет точек baseline {baseline!r}. KT-1 без baseline не интерпретируется."
        )

    sigmas = sorted(set(by[mechanism]) & set(by[baseline]))
    win_interval, overlap_all = [], True
    detail = []
    for s in sigmas:
        m, b = by[mechanism][s], by[baseline][s]
        dom = dominates((m.successes, m.n), (b.successes, b.n))
        sep = separated((m.successes, m.n), (b.successes, b.n))
        if dom:
            win_interval.append(s)
        if sep:
            overlap_all = False
        detail.append(
            {
                "sigma": s,
                f"{mechanism}_point": round(m.point, 6),
                f"{baseline}_point": round(b.point, 6),
                "mechanism_dominates": dom,
                "intervals_separated": sep,
            }
        )

    # монотонность механизма по sigma
    seq = [by[mechanism][s].point for s in sigmas]
    monotone = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))

    # резкость порога: максимальное падение между соседними точками
    drops = [(sigmas[i + 1], seq[i] - seq[i + 1]) for i in range(len(seq) - 1)]
    max_drop_sigma, max_drop = max(drops, key=lambda t: t[1]) if drops else (None, 0.0)

    if win_interval:
        outcome = "SUCCESS_WIDE" if len(win_interval) >= 3 else "SUCCESS_NARROW"
    elif overlap_all:
        outcome = "FAIL_NO_ADVANTAGE"  # п.2 критериев провала
    else:
        outcome = "FAIL_DOMINATED"  # п.1 критериев провала
    if not monotone:
        outcome = "FAIL_NONMONOTONE"  # п.3: отладка, не вывод

    return {
        "outcome": outcome,
        "dominating_sigmas": win_interval,
        "monotone": monotone,
        "max_drop": round(max_drop, 6),
        "max_drop_at_sigma": max_drop_sigma,
        "needs_grid_refinement": max_drop > 0.30,
        "per_sigma": detail,
        "action": {
            "SUCCESS_WIDE": "Переход к KT-2 и KT-3 в полном объёме; IP-трек активируется.",
            "SUCCESS_NARROW": "Переход к KT-2; сузить формулировку ценности до конкретного режима.",
            "FAIL_DOMINATED": "ОСТАНОВКА программы. Переформулирование задачи или закрытие.",
            "FAIL_NO_ADVANTAGE": "Остановка патентного трека; механизм остаётся research-артефактом.",
            "FAIL_NONMONOTONE": "Отладка реализации/обучения. KT-1 перезапускается.",
        }[outcome],
    }


# --------------------------------------------------------------------------
# Ввод-вывод
# --------------------------------------------------------------------------


def load_points(pattern: str) -> list[CurvePoint]:
    """Читает артефакты kt1_*.json, агрегируя по (mechanism, sigma) через ячейки."""
    agg: dict[tuple[str, float], dict] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        key = (d["mechanism"], float(d["sigma"]))
        slot = agg.setdefault(key, {"successes": 0, "n": 0, "delta": d.get("delta")})
        slot["successes"] += int(d["retrieval_correct"])
        slot["n"] += int(d["episodes"])
    return [
        CurvePoint(mechanism=m, sigma=s, successes=v["successes"], n=v["n"], delta=v["delta"])
        for (m, s), v in sorted(agg.items())
    ]


def write_curve_csv(points: list[CurvePoint], path: str) -> None:
    import csv

    rows = [p.row() for p in points]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Самопроверка
# --------------------------------------------------------------------------


def _selftest() -> None:
    # 1. Воспроизведение опубликованных интервалов существующих результатов.
    for succ, n, exp_lo, exp_hi in [
        (125, 128, 0.9334, 0.9920),
        (122, 128, 0.9015, 0.9783),
        (126, 128, 0.9448, 0.9957),
    ]:
        lo, hi = wilson(succ, n)
        assert abs(lo - exp_lo) < 5e-4 and abs(hi - exp_hi) < 5e-4, (succ, lo, hi)

    # 2. Нулевой числитель не даёт нулевую ширину. Основной результат набора.
    lo, hi = wilson(0, 512)
    assert lo == 0.0 and 0.007 < hi < 0.008, (lo, hi)
    # shuffled retrieval отличим от chance 1/64 в сторону нуля
    assert not contains(0, 512, 1 / 64)

    # 3. Контроли содержат chance-уровень.
    for succ in (1, 2, 3, 4):
        assert contains(succ, 128, 1 / 64), succ

    # 4. Планирование согласуется с решением о 256 эпизодах.
    assert min_episodes_to_separate(0.976, 0.90) <= 256
    assert min_episodes_to_separate(0.976, 0.95) > 256

    # 5. Логика вердикта.
    good = [
        CurvePoint("phef", 0.0, 250, 256),
        CurvePoint("phef", 0.03, 240, 256),
        CurvePoint("phef", 0.10, 225, 256),
        CurvePoint("quantized_dict", 0.0, 256, 256),
        CurvePoint("quantized_dict", 0.03, 150, 256),
        CurvePoint("quantized_dict", 0.10, 90, 256),
    ]
    v = verdict(good)
    assert v["outcome"].startswith("SUCCESS"), v

    dominated = [
        CurvePoint("phef", 0.0, 250, 256),
        CurvePoint("phef", 0.03, 20, 256),
        CurvePoint("quantized_dict", 0.0, 256, 256),
        CurvePoint("quantized_dict", 0.03, 240, 256),
    ]
    assert verdict(dominated)["outcome"] == "FAIL_DOMINATED"

    # 6. Отсутствие baseline -- ошибка, не тихий успех.
    try:
        verdict([CurvePoint("phef", 0.0, 250, 256)])
    except ValueError:
        pass
    else:
        raise AssertionError("KT-1 без baseline не должен интерпретироваться")

    print("kt1_stats selftest: OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Статистический слой KT-1")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--glob", default="artifacts/kt1/kt1_*.json")
    ap.add_argument("--csv", default="artifacts/kt1/kt1_curve.csv")
    ap.add_argument("--plan", action="store_true", help="таблица планирования эпизодов")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.plan:
        print(f"{'accuracy при sigma':>20} | {'нужно эпизодов':>14}")
        for alt, n in plan_table():
            print(f"{alt:>20} | {str(n):>14}")
        return

    pts = load_points(args.glob)
    if not pts:
        raise SystemExit(f"нет артефактов по шаблону {args.glob}")
    for p in pts:
        r = p.row()
        print(
            f"{r['mechanism']:>16} sigma={r['sigma']:<5} "
            f"{r['successes']:>4}/{r['n']:<4} {r['point']:.4f} "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] w={r['ci_width_pp']:.2f}pp"
        )
    write_curve_csv(pts, args.csv)
    print(f"\n-> {args.csv}")
    print(json.dumps(verdict(pts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
