"""
kt1_baselines.py -- обязательные baseline для KT-1.

Зависимости: numpy. Torch и GPU не требуются.

Содержит:
  * ExactHashMap       -- контрольная точка, ожидается обвал при любом sigma > 0
  * QuantizedDict      -- ГЛАВНЫЙ конкурент; delta подбирается по validation
  * perturb_keys       -- модель возмущения, единая для механизма и baseline

Принцип, определяющий ценность всего KT-1:
    Кривая деградации одного механизма НЕ является результатом.
    Результат -- разность кривых.

Поэтому quantized dict реализован как СИЛЬНЫЙ baseline: delta подбирается
независимо для каждого уровня sigma по validation split. Словарь с фиксированным
delta -- соломенное чучело, и это будет первым замечанием рецензента.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# --------------------------------------------------------------------------
# Модель возмущения -- единая для всех механизмов
# --------------------------------------------------------------------------


def key_scale(keys: np.ndarray) -> float:
    """Масштаб ключа: среднее стандартное отклонение по компонентам.

    sigma в KT-1 выражается В ЕДИНИЦАХ этой величины. Абсолютные значения sigma
    бессмысленны без нормировки и не подлежат публикации.
    """
    return float(np.mean(np.std(np.asarray(keys, dtype=np.float64), axis=0)))


def perturb_keys(
    query_keys: np.ndarray, sigma: float, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Аддитивный гауссов шум ТОЛЬКО на query key.

    Ключи, сохранённые в памяти, не возмущаются. Это существенно: моделируется
    неточность запроса, а не порча истории.
    """
    q = np.asarray(query_keys, dtype=np.float32)
    if sigma == 0.0:
        return q.copy()
    noise = rng.normal(0.0, sigma * scale, size=q.shape).astype(np.float32)
    return q + noise


# --------------------------------------------------------------------------
# Общий интерфейс
# --------------------------------------------------------------------------


class Mechanism(Protocol):
    name: str

    def build(self, keys: np.ndarray, values: np.ndarray) -> None: ...
    def query(self, qkeys: np.ndarray) -> np.ndarray: ...
    def memory_bytes(self) -> int: ...


# --------------------------------------------------------------------------
# Exact hash map -- контрольная точка
# --------------------------------------------------------------------------


@dataclass
class ExactHashMap:
    """Точный словарь по байтовому представлению ключа, latest-value семантика.

    Назначение в KT-1: продемонстрировать, что при sigma > 0 задача действительно
    НЕ решается точным поиском. Ожидается accuracy 1.0 при sigma = 0 и обвал до
    chance при любом sigma > 0.

    При sigma = 0 этот механизм доминирует PHEF по всем осям. Это известно
    заранее и не является предметом проверки: именно поэтому ниша механизма
    может существовать только при неточном ключе.
    """

    name: str = "exact_hash_map"
    _table: dict[bytes, int] = field(default_factory=dict, repr=False)

    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        self._table = {}
        k = np.ascontiguousarray(keys, dtype=np.float32)
        for i in range(k.shape[0]):
            # latest-value: поздняя запись затирает раннюю
            self._table[k[i].tobytes()] = int(values[i])

    def query(self, qkeys: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(qkeys, dtype=np.float32)
        return np.array(
            [self._table.get(q[i].tobytes(), -1) for i in range(q.shape[0])], dtype=np.int64
        )

    def memory_bytes(self) -> int:
        if not self._table:
            return 0
        klen = len(next(iter(self._table)))
        return len(self._table) * (klen + 8 + 64)  # + накладные расходы dict


# --------------------------------------------------------------------------
# Quantized dict -- главный конкурент
# --------------------------------------------------------------------------


@dataclass
class QuantizedDict:
    """Округление ключа до сетки с шагом delta, затем exact lookup.

    Это тот baseline, который может убить проект. Он обязан быть сильным:
      * delta подбирается по validation, независимо для каждого sigma;
      * при коллизии сохраняется последнее значение (latest-value семантика,
        согласованная с overwrite_p = 0.3).

    delta выражается в тех же единицах, что sigma: в масштабах key_scale.
    """

    delta: float
    scale: float = 1.0
    name: str = "quantized_dict"
    _table: dict[bytes, int] = field(default_factory=dict, repr=False)

    def _bucket(self, k: np.ndarray) -> np.ndarray:
        step = self.delta * self.scale
        return np.rint(np.asarray(k, dtype=np.float64) / step).astype(np.int32)

    def build(self, keys: np.ndarray, values: np.ndarray) -> None:
        self._table = {}
        b = np.ascontiguousarray(self._bucket(keys))
        for i in range(b.shape[0]):
            self._table[b[i].tobytes()] = int(values[i])

    def query(self, qkeys: np.ndarray) -> np.ndarray:
        b = np.ascontiguousarray(self._bucket(qkeys))
        return np.array(
            [self._table.get(b[i].tobytes(), -1) for i in range(b.shape[0])], dtype=np.int64
        )

    def memory_bytes(self) -> int:
        if not self._table:
            return 0
        klen = len(next(iter(self._table)))
        return len(self._table) * (klen + 8 + 64)


DELTA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def select_delta(
    mem_keys: np.ndarray,
    mem_values: np.ndarray,
    val_qkeys: np.ndarray,
    val_answers: np.ndarray,
    sigma: float,
    scale: float,
    rng: np.random.Generator,
    delta_grid: tuple[float, ...] = DELTA_GRID,
) -> tuple[float, dict[float, float]]:
    """Подбор delta по VALIDATION split. Никогда по test.

    Возвращает (лучшая delta, {delta: accuracy на validation}).
    """
    q = perturb_keys(val_qkeys, sigma, scale, rng)
    scores: dict[float, float] = {}
    for d in delta_grid:
        m = QuantizedDict(delta=d, scale=scale)
        m.build(mem_keys, mem_values)
        pred = m.query(q)
        scores[d] = float(np.mean(pred == val_answers))
    best = max(scores, key=lambda d: (scores[d], -d))
    return best, scores


# --------------------------------------------------------------------------
# Генератор синтетической задачи
# --------------------------------------------------------------------------


def make_episode(
    n_segments: int,
    n_keys: int = 1 << 30,
    n_values: int = 64,
    overwrite_p: float = 0.3,
    key_dim: int = 32,
    rng: np.random.Generator | None = None,
) -> dict:
    """Генерирует один эпизод latest-value associative retrieval.

    Соответствует протоколу compact_eval_v2_streamed_hard: key_dim = 32,
    n_values = 64, overwrite_p = 0.3, n_keys = 2^30 как key-ID universe для
    статистики коллизий, а НЕ число записей.

    ВНИМАНИЕ. Это автономный референсный генератор для baseline-ветки. Он должен
    быть сверен с генератором репозитория до содержательных прогонов
    (см. kt1_run.py, шаг verify_generator_parity).
    """
    rng = rng or np.random.default_rng(0)

    key_ids = np.empty(n_segments, dtype=np.int64)
    active: list[int] = []
    for t in range(n_segments):
        if active and rng.random() < overwrite_p:
            key_ids[t] = active[rng.integers(len(active))]
        else:
            kid = int(rng.integers(n_keys))
            active.append(kid)
            key_ids[t] = kid

    values = rng.integers(0, n_values, size=n_segments, dtype=np.int64)

    # Детерминированное отображение key_id -> вектор
    uniq = np.unique(key_ids)
    vec = {}
    for kid in uniq:
        r = np.random.default_rng(int(kid))
        vec[int(kid)] = r.normal(0.0, 1.0, size=key_dim).astype(np.float32)
    keys = np.stack([vec[int(k)] for k in key_ids])

    # latest-value ground truth
    latest: dict[int, int] = {}
    for t in range(n_segments):
        latest[int(key_ids[t])] = int(values[t])

    n_eff = len(latest)
    probe = int(rng.integers(len(uniq)))
    qk = int(uniq[probe])

    return {
        "keys": keys,
        "values": values,
        "key_ids": key_ids,
        "query_key": vec[qk][None, :],
        "answer": np.array([latest[qk]], dtype=np.int64),
        "n_eff": n_eff,
        "n_segments": n_segments,
    }


# --------------------------------------------------------------------------
# Прогон baseline по сетке sigma
# --------------------------------------------------------------------------


def run_baselines(
    n_segments: int,
    sigmas: tuple[float, ...],
    episodes: int,
    seed: int,
    out_dir: str,
    n_values: int = 64,
    overwrite_p: float = 0.3,
    key_dim: int = 32,
) -> list[dict]:
    """Прогоняет exact hash map и quantized dict по сетке sigma.

    Пишет по одному артефакту на (mechanism, sigma), совместимому со схемой,
    которую читает kt1_stats.load_points.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[dict] = []

    for sigma in sigmas:
        hits = {"exact_hash_map": 0, "quantized_dict": 0}
        deltas: list[float] = []
        mem_bytes = {"exact_hash_map": 0, "quantized_dict": 0}

        for ep in range(episodes):
            rng = np.random.default_rng([seed, ep, int(sigma * 1e6)])
            d = make_episode(
                n_segments,
                n_values=n_values,
                overwrite_p=overwrite_p,
                key_dim=key_dim,
                rng=rng,
            )
            scale = key_scale(d["keys"])

            # validation split для подбора delta: отдельный эпизод, тот же sigma
            vrng = np.random.default_rng([seed, ep, int(sigma * 1e6), 999])
            vd = make_episode(
                max(1024, n_segments // 100),
                n_values=n_values,
                overwrite_p=overwrite_p,
                key_dim=key_dim,
                rng=vrng,
            )
            best_delta, _ = select_delta(
                vd["keys"], vd["values"], vd["query_key"], vd["answer"],
                sigma, key_scale(vd["keys"]), vrng,
            )
            deltas.append(best_delta)

            q = perturb_keys(d["query_key"], sigma, scale, rng)

            for mech in (ExactHashMap(), QuantizedDict(delta=best_delta, scale=scale)):
                mech.build(d["keys"], d["values"])
                pred = mech.query(q)
                hits[mech.name] += int(np.all(pred == d["answer"]))
                mem_bytes[mech.name] = max(mem_bytes[mech.name], mech.memory_bytes())

        for name in hits:
            rec = {
                "schema": "kt1_baseline_v1",
                "mechanism": name,
                "sigma": float(sigma),
                "episodes": episodes,
                "retrieval_correct": hits[name],
                "n_segments": n_segments,
                "n_values": n_values,
                "overwrite_p": overwrite_p,
                "key_dim": key_dim,
                "seed": seed,
                "delta": (
                    float(np.median(deltas)) if name == "quantized_dict" else None
                ),
                "memory_bytes": mem_bytes[name],
            }
            path = os.path.join(out_dir, f"kt1_{name}_sigma{sigma}_seed{seed}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
            written.append(rec)

    return written


def _selftest() -> None:
    rng = np.random.default_rng(0)
    d = make_episode(20_000, rng=rng)
    assert d["n_eff"] < d["n_segments"], "overwrite должен уменьшать число активных записей"
    ratio = d["n_eff"] / d["n_segments"]
    assert 0.55 < ratio < 0.95, f"подозрительное n_eff/n_segments = {ratio:.3f}"

    scale = key_scale(d["keys"])
    assert 0.5 < scale < 2.0, scale

    # sigma = 0: точный словарь обязан быть идеален
    m = ExactHashMap()
    m.build(d["keys"], d["values"])
    assert np.all(m.query(d["query_key"]) == d["answer"])

    # sigma > 0: точный словарь обязан развалиться
    q = perturb_keys(d["query_key"], 0.10, scale, rng)
    assert m.query(q)[0] == -1

    # quantized dict при малом sigma должен переживать возмущение
    qd = QuantizedDict(delta=1.0, scale=scale)
    qd.build(d["keys"], d["values"])
    q_small = perturb_keys(d["query_key"], 0.01, scale, rng)
    assert qd.query(q_small)[0] != -1

    print(f"kt1_baselines selftest: OK (n_eff/n_seg={ratio:.3f}, key_scale={scale:.3f})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-segments", type=int, default=100_000)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts/kt1")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
    else:
        recs = run_baselines(
            a.n_segments,
            (0.0, 0.01, 0.03, 0.10, 0.30, 1.00),
            a.episodes,
            a.seed,
            a.out,
        )
        for r in recs:
            print(
                f"{r['mechanism']:>16} sigma={r['sigma']:<5} "
                f"{r['retrieval_correct']:>4}/{r['episodes']}"
            )
