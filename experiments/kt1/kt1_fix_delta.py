"""Исправленный подбор delta: validation той же длины, много запросов."""
import sys, numpy as np
sys.path.insert(0, 'experiments/kt1')
from kt1_baselines import QuantizedDict, key_scale, make_episode, perturb_keys
from kt1_stats import wilson

GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
N_SEG = 20000
N_VAL_QUERIES = 32


def select_delta_fixed(sigma, seed, n_seg=N_SEG, grid=GRID):
    """delta по validation-эпизоду ТОЙ ЖЕ длины, усреднение по 32 запросам."""
    rng = np.random.default_rng([seed, 777])
    d = make_episode(n_seg, rng=rng)
    sc = key_scale(d["keys"])
    latest = {}
    for t in range(n_seg):
        latest[int(d["key_ids"][t])] = int(d["values"][t])
    uniq = np.unique(d["key_ids"])
    picks = rng.choice(len(uniq), size=min(N_VAL_QUERIES, len(uniq)), replace=False)

    scores = {}
    for delta in grid:
        m = QuantizedDict(delta=delta, scale=sc)
        m.build(d["keys"], d["values"])
        hits = 0
        for p in picks:
            kid = int(uniq[p])
            qk = d["keys"][np.flatnonzero(d["key_ids"] == kid)[-1]][None, :]
            qn = perturb_keys(qk, sigma, sc, rng)
            hits += int(m.query(qn)[0] == latest[kid])
        scores[delta] = hits / len(picks)
    best = max(scores, key=lambda x: (scores[x], -x))
    return best, scores


if __name__ == "__main__":
    print("Исправленный подбор delta (validation = 20000 сегментов, 32 запроса)\n")
    for sigma in (0.0, 0.30, 0.50, 0.70, 1.0, 1.5):
        best, sc = select_delta_fixed(sigma, seed=7)
        top = sorted(sc.items(), key=lambda kv: -kv[1])[:3]
        print(f"sigma={sigma:<5} best_delta={best:<6} "
              f"top3={[(d, round(v,3)) for d, v in top]}")
