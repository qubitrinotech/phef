import argparse
import csv
import glob
import json
import os


def load(path):
    with open(path) as handle:
        return json.load(handle)


def run_name(path, root):
    """results/capacity/dim16_kd32.json -> capacity/dim16_kd32

    Имя включает подкаталог, иначе одноимённые файлы
    из разных папок затрут друг друга в сводке.
    """
    relative = os.path.relpath(path, root)
    return os.path.splitext(relative)[0]


def fmt(value, width, digits=4):
    if value is None:
        return f"{'-':>{width}}"
    if isinstance(value, float):
        return f"{value:{width}.{digits}f}"
    return f"{str(value):>{width}}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--pattern", default="**/*.json")
    parser.add_argument("--out", default="output/capacity_summary.csv")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["_scratch"],
        help="подстроки пути, которые пропускать",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.root, args.pattern)
    paths = sorted(glob.glob(pattern, recursive=True))

    rows = []

    for path in paths:
        if any(token in path for token in args.exclude):
            continue

        try:
            report = load(path)
        except (json.JSONDecodeError, OSError) as error:
            print(f"пропущен {path}: {error}")
            continue

        if not isinstance(report, dict) or "answer_acc" not in report:
            continue

        config = report.get("config", {})
        oracle = report.get("oracle_bound")
        answer = report.get("answer_acc")
        distances = report.get("acc_by_distance") or []

        rows.append({
            "run": run_name(path, args.root),
            "n_segments": config.get("n_segments"),
            "n_keys": config.get("n_keys"),
            "key_dim": config.get("key_dim"),
            "dim": config.get("dim"),
            "overwrite_p": config.get("overwrite_p"),
            "epochs": config.get("epochs"),
            "selection": report.get("selection"),
            "answer_acc": answer,
            "retrieval_acc": report.get("retrieval_acc"),
            "oracle_bound": oracle,
            "gap_to_oracle": (
                oracle - answer
                if oracle is not None and answer is not None
                else None
            ),
            "tie_rate": report.get("tie_rate"),
            "recency_scale": report.get("recency_scale"),
            "score_scale": report.get("score_scale"),
            "memory_gate": report.get("memory_gate"),
            "best_acc": report.get("best_acc"),
            "best_epoch": report.get("best_epoch"),
            "escape_epoch": report.get("escape_epoch"),
            "acc_recent": distances[0] if distances else None,
            "acc_remote": distances[-1] if distances else None,
            "drop_recent_remote": (
                distances[0] - distances[-1] if distances else None
            ),
        })

    if not rows:
        print(f"Нет результатов по шаблону {pattern}")
        return

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(rows[0])

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'run':32} {'kd':>4} {'dim':>5} {'acc':>8} "
        f"{'oracle':>8} {'gap':>8} {'drop':>8} {'esc':>7}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['run']:32} "
            f"{fmt(row['key_dim'], 4)} "
            f"{fmt(row['dim'], 5)} "
            f"{fmt(row['answer_acc'], 8)} "
            f"{fmt(row['oracle_bound'], 8)} "
            f"{fmt(row['gap_to_oracle'], 8)} "
            f"{fmt(row['drop_recent_remote'], 8)} "
            f"{fmt(row['escape_epoch'], 7)}"
        )

    print(f"\nСтрок: {len(rows)}. Сохранено: {args.out}")


if __name__ == "__main__":
    main()
