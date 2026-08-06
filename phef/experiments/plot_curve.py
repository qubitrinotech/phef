import csv
import glob
import json
import math
import os
from collections import defaultdict


def main():
    grouped = defaultdict(list)

    for path in glob.glob("results/v2/v2_kd*.json"):
        with open(path) as handle:
            report = json.load(handle)

        grouped[report["config"]["key_dim"]].append(report)

    if not grouped:
        print("Нет файлов results/v2/v2_kd*.json")
        return

    os.makedirs("output", exist_ok=True)
    rows = []

    for key_dim in sorted(grouped):
        reports = grouped[key_dim]

        accuracies = [r["answer_acc"] for r in reports]
        ties = [r.get("tie_rate", 0.0) for r in reports]
        bounds = reports[0].get("oracle_bounds", {})

        chance = 1.0 / reports[0]["config"]["n_values"]
        mean = sum(accuracies) / len(accuracies)

        rows.append({
            "key_dim": key_dim,
            "n_seeds": len(accuracies),
            "acc_mean": round(mean, 6),
            "acc_min": round(min(accuracies), 6),
            "acc_max": round(max(accuracies), 6),
            "acc_spread": round(max(accuracies) - min(accuracies), 6),
            "chance": round(chance, 6),
            "ratio_to_chance": round(mean / chance, 2),
            "tie_rate_mean": round(sum(ties) / len(ties), 6),
            "oracle_m002": round(bounds.get("m0.02", float("nan")), 6),
            "oracle_m005": round(bounds.get("m0.05", float("nan")), 6),
            "gap_m005": round(
                bounds.get("m0.05", float("nan")) - mean, 6
            ),
            "recency_scale": round(
                sum(r["recency_scale"] for r in reports) / len(reports), 4
            ),
            "escape_epochs": str(
                [r.get("escape_epoch") for r in reports]
            ),
        })

    out = "output/kd_capacity_curve.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'kd':>3} {'n':>2} {'acc':>8} {'spread':>8} "
        f"{'xchance':>8} {'orc.05':>8} {'gap':>9} {'tie':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['key_dim']:>3} "
            f"{row['n_seeds']:>2} "
            f"{row['acc_mean']:8.4f} "
            f"{row['acc_spread']:8.4f} "
            f"{row['ratio_to_chance']:8.1f} "
            f"{row['oracle_m005']:8.4f} "
            f"{row['gap_m005']:9.4f} "
            f"{row['tie_rate_mean']:7.4f}"
        )

    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
