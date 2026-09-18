"""Пороги baseline v2 с консервативной монотонной огибающей.

ЗАЧЕМ ОГИБАЮЩАЯ. Baseline считается на 64 эпизодах против 256 у PHEF,
поэтому при малых счётчиках оценка нестабильна. Измерено: sigma=0.42 дала
1/64, sigma=0.50 дала 5/64 -- baseline "вырос" при большем шуме, что
физически невозможно. Интервалы Уилсона перекрываются, различие не значимо.

Занижение порога случайным шумом создало бы ложную победу PHEF. Поэтому
порог при sigma берётся как максимум верхних границ по всем sigma' >= sigma:
случайное занижение в одной точке не может создать ложное доминирование.

Проверено: вердикт KT-1 (SUCCESS_NARROW, интервал [0.30, 0.50]) не меняется
при переходе от raw-порогов к огибающей.
"""
import glob, json, math

def wilson(x, n, z=1.959963984540054):
    if n <= 0:
        return (0.0, 1.0)
    p = x/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)


def load_baseline(pattern='artifacts/kt1/baseline_v2/kt1_quantized_dict_sigma*.json'):
    out = {}
    for f in glob.glob(pattern):
        d = json.load(open(f))
        out[round(float(d['sigma']), 2)] = (d['retrieval_correct'], d['episodes'])
    return out


def thresholds(envelope=True):
    """{sigma: порог}. envelope=True -> консервативная монотонная огибающая."""
    bl = load_baseline()
    srt = sorted(bl)
    raw = {s: wilson(*bl[s])[1] for s in srt}
    if not envelope:
        return raw
    return {s: max(raw[t] for t in srt[i:]) for i, s in enumerate(srt)}


if __name__ == "__main__":
    bl = load_baseline()
    raw = thresholds(envelope=False)
    env = thresholds(envelope=True)
    print(f"{'sigma':>6} {'qdict':>8} {'raw':>8} {'envelope':>9}")
    for s in sorted(bl):
        mark = " (поднят)" if env[s] > raw[s] + 1e-9 else ""
        print(f"{s:>6} {bl[s][0]:>4}/{bl[s][1]:<3} {raw[s]:>8.4f} {env[s]:>9.4f}{mark}")
