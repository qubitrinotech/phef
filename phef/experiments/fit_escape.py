"""
Пересчёт степенного закона escape по JSON-отчётам train_v3.py.

В отчёте от 5 августа зафиксировано escape ~= 750 * (128/dim)^0.79,
но независимая OLS-проверка дала префактор 646 при том же показателе.
Скрипт считает и показатель, и префактор с bootstrap-интервалом,
и печатает остатки — чтобы префактор больше не был якорем «на глаз».

    LC_ALL=C PYTHONPATH=. python -u experiments/fit_escape.py \
        --root results/capacity_v3 --out output/escape_fit.json
"""

import argparse
import glob
import json
import math
import os
import random
import statistics


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    return slope, intercept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--pattern", default="**/*.json")
    p.add_argument("--anchor-dim", type=int, default=128)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    points = {}
    for path in glob.glob(os.path.join(args.root, args.pattern), recursive=True):
        with open(path) as f:
            r = json.load(f)
        if not r.get("escape_epoch"):
            continue
        cfg = r.get("config") or r.get("args") or {}
        if "dim" not in cfg:
            continue
        d = cfg["dim"]
        points.setdefault(d, []).append(r["escape_epoch"])

    if len(points) < 3:
        raise SystemExit(f"недостаточно точек: {sorted(points)}")

    dims = sorted(points, reverse=True)
    xs, ys, rows = [], [], []
    for d in dims:
        e = statistics.mean(points[d])
        sd = statistics.stdev(points[d]) if len(points[d]) > 1 else 0.0
        rows.append({"dim": d, "n": len(points[d]), "escape_mean": e, "escape_sd": sd})
        xs.append(math.log(args.anchor_dim / d))
        ys.append(math.log(e))

    slope, intercept = ols(xs, ys)
    prefactor = math.exp(intercept)

    boots = []
    idx = list(range(len(xs)))
    for _ in range(args.boot):
        s = [random.choice(idx) for _ in idx]
        if len(set(xs[i] for i in s)) < 2:
            continue
        sl, ic = ols([xs[i] for i in s], [ys[i] for i in s])
        boots.append((sl, math.exp(ic)))
    boots.sort(key=lambda t: t[0])
    lo_s = boots[int(0.025 * len(boots))][0]
    hi_s = boots[int(0.975 * len(boots))][0]
    boots.sort(key=lambda t: t[1])
    lo_p = boots[int(0.025 * len(boots))][1]
    hi_p = boots[int(0.975 * len(boots))][1]

    resid = [
        {
            "dim": rows[i]["dim"],
            "observed": rows[i]["escape_mean"],
            "predicted": prefactor * math.exp(slope * xs[i]),
        }
        for i in range(len(xs))
    ]
    for r in resid:
        r["ratio"] = r["observed"] / r["predicted"]

    out = {
        "law": f"escape = {prefactor:.0f} * ({args.anchor_dim}/dim)^{slope:.4f}",
        "exponent": slope,
        "exponent_ci95": [lo_s, hi_s],
        "prefactor": prefactor,
        "prefactor_ci95": [lo_p, hi_p],
        "points": rows,
        "residuals": resid,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
