#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

ROOT = Path.home() / "qubitrino-projects/agi/phef/results"
DIRS = [
    "capacity_scan",
    "capacity_scan_dim32",
    "capacity_scan_dim64_extended",
    "capacity_scan_dim128",
    "capacity_scan_dim128_extended",
    "capacity_scan_dim256",
    "capacity_scan_dim128_keydim64",
    "capacity_scan_dim128_nkeys1024",
]
NAME_RE = re.compile(r"dim(\d+)_seg(\d+)_s(\d+)_e(\d+)\.json$")

ALIASES = {
    "answer_acc":    ["answer_acc", "final_answer_acc", "answer_accuracy", "acc_answer"],
    "retrieval_acc": ["retrieval_acc", "final_retrieval_acc", "retrieval_accuracy"],
    "best_sel_acc":  ["best_sel_acc", "best_selection_acc", "best_sel_accuracy", "best_sel"],
    "escape_epoch":  ["escape_epoch", "escape", "escaped_at_epoch"],
    "dim":           ["dim", "hidden_dim"],
    "n_segments":    ["n_segments", "seg", "segments"],
    "seed":          ["seed"],
    "epochs":        ["epochs", "n_epochs"],
    "key_dim":       ["key_dim"],
    "n_keys":        ["n_keys"],
    "n_values":      ["n_values"],
    "lr":            ["lr", "learning_rate"],
}

def flatten(obj, out=None, prefix=""):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                flatten(v, out, f"{prefix}{k}.")
            else:
                out.setdefault(k, v)
                out[f"{prefix}{k}"] = v
    return out

def pick(flat, field):
    for k in ALIASES[field]:
        if k in flat and flat[k] is not None:
            return flat[k]
        for fk, fv in flat.items():
            if fk.split(".")[-1] == k and fv is not None:
                return fv
    return None

rows, bad, allkeys = [], [], set()
for d in DIRS:
    p = ROOT / d
    if not p.is_dir():
        bad.append((d, "-", "DIR_MISSING"))
        continue
    files = sorted(p.glob("dim*_seg*_s*_e*.json"))
    if not files:
        bad.append((d, "-", "NO_JSON_MATCHING_PATTERN"))
    for f in files:
        try:
            raw = json.loads(f.read_text())
        except Exception as e:
            bad.append((d, f.name, f"UNREADABLE: {type(e).__name__}: {e}"))
            continue
        flat = flatten(raw)
        allkeys.update(flat.keys())
        m = NAME_RE.search(f.name)
        fn = {"dim": m.group(1), "n_segments": m.group(2), "seed": m.group(3), "epochs": m.group(4)} if m else {}
        row = {"dir": d, "file": f.name}
        for field in ["dim", "n_segments", "seed", "epochs", "key_dim", "n_keys", "n_values", "lr",
                      "answer_acc", "retrieval_acc", "best_sel_acc", "escape_epoch"]:
            v = pick(flat, field)
            if v is None and field in fn:
                v = int(fn[field])
            row[field] = v
        rows.append(row)

def fmt(v):
    if v is None:
        return "MISSING"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

def sk(r):
    g = lambda k: r[k] if isinstance(r[k], int) else -1
    return (g("dim"), g("n_segments"), g("seed"), r["dir"])

cols = ["dir", "dim", "n_segments", "seed", "epochs", "key_dim", "n_keys", "n_values",
        "answer_acc", "retrieval_acc", "best_sel_acc", "escape_epoch"]
rows.sort(key=sk)
table = [cols] + [[fmt(r[c]) for c in cols] for r in rows]
w = [max(len(r[i]) for r in table) for i in range(len(cols))]
print("  ".join(h.ljust(w[i]) for i, h in enumerate(table[0])))
print("  ".join("-" * w[i] for i in range(len(cols))))
for r in table[1:]:
    print("  ".join(r[i].ljust(w[i]) for i in range(len(cols))))

print(f"\nTOTAL ROWS: {len(rows)}")
print("KEYS SEEN IN JSON:", ", ".join(sorted(allkeys)) or "none")
if bad:
    print("\nPROBLEMS:")
    for d, f, why in bad:
        print(f"  {d}/{f}: {why}")
else:
    print("\nPROBLEMS: none")

miss = sorted({c for r in rows for c in cols if r[c] is None})
print("FIELDS WITH MISSING VALUES:", ", ".join(miss) if miss else "none")