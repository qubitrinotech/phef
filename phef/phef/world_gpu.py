"""
GPU-резидентный генератор BranchingWorld.

Мотивация. Первое узкое место — Python-цикл в _sample_slots: 26.6 мкс
launch overhead на итерацию, при n_segments=131072 это 3487 мс на вызов
batch(), то есть 96.2% времени эпохи. Цикл перенесён на numpy: замерено
563 мс против 3425 мс на batch(8). Передача host->device — int16 pinned
с кастом в int64 на устройстве, 0.79 мс, буфер кэшируется между вызовами.

Второе узкое место — развёрнутый event_keys формы (bs, n_segments,
key_dim): 1024 MiB в batch() при bs=64, s=131072, kd=32, и 9216 MiB на
forward-пути key_encoder. batch() возвращает (slots, codebook), gather
выполняется после key_encoder. Замерено: 9216 MiB -> 48.1 MiB,
50.9 мс -> 0.5 мс, расхождение двух путей на GPU ровно нулевое.

Семантика полностью совпадает с phef/world.py:
  * slots: случайные, с вероятностью overwrite_p переписывающие
    случайный ПРЕДЫДУЩИЙ слот (цикл по сегментам последователен);
  * query выбирается по случайному СЕГМЕНТУ (pick ~ U{0..S-1}),
    не по уникальному слоту — это зафиксированная смена семантики
    из раздела 8 отчёта;
  * ответ — значение ПОСЛЕДНЕГО вхождения слота запроса;
  * codebook пересэмплируется каждый батч (lookup-таблица невозможна).

Побитовой RNG-совместимости с numpy-версией нет и быть не может.
Совместимость проверяется распределённо: experiments/parity_check.py.
"""

import numpy as np
import torch


class BranchingWorldGPU:
    def __init__(
        self,
        n_keys=512,
        n_values=64,
        n_segments=256,
        overwrite_p=0.3,
        key_dim=32,
        seed=0,
        symbolic=False,
        device="cuda",
    ):
        if n_keys > 32767:
            raise ValueError(
                f"n_keys={n_keys} не влезает в int16; "
                "передача слотов host->device рассчитана на int16"
            )

        self.n_keys = n_keys
        self.n_values = n_values
        self.n_segments = n_segments
        self.overwrite_p = overwrite_p
        self.key_dim = key_dim
        self.symbolic = symbolic
        self.device = torch.device(device)

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

        self.np_rng = np.random.default_rng(int(seed))
        self._slot_pin = None

        self.last_slots = None
        self.last_query_slots = None
        self.last_answer_positions = None

    # ------------------------------------------------------------------ #

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randint(self, high, shape):
        return torch.randint(
            0,
            high,
            shape,
            generator=self.gen,
            device=self.device,
        )

    def _sample_slots(self, bs):
        s = self.n_segments

        base = self.np_rng.integers(0, self.n_keys, size=(bs, s))
        reuse = self.np_rng.random((bs, s)) < self.overwrite_p
        reuse[:, 0] = False
        back = self.np_rng.random((bs, s))

        # slots[ts] = base[root(ts)], где root - корень цепочки reuse.
        # Корни ищутся удвоением указателей parent = parent[parent]:
        # 5-6 итераций вместо s последовательных шагов. Порядок
        # обращений к RNG не изменён, результат побитово тот же.
        ts = np.arange(s)
        source = np.clip((back * ts).astype(np.int64), 0, np.maximum(ts - 1, 0))
        parent = np.where(reuse, source, ts[None, :])

        while True:
            nxt = np.take_along_axis(parent, parent, axis=1)
            if np.array_equal(nxt, parent):
                break
            parent = nxt

        slots = np.take_along_axis(base, parent, axis=1)

        if self._slot_pin is None or self._slot_pin.shape != (bs, s):
            self._slot_pin = torch.empty(
                (bs, s), dtype=torch.int16, pin_memory=True
            )

        self._slot_pin.copy_(torch.from_numpy(slots.astype(np.int16)))

        return self._slot_pin.to(self.device, non_blocking=True).long()

    # ------------------------------------------------------------------ #

    def batch(self, bs):
        s = self.n_segments
        dev = self.device

        slots = self._sample_slots(bs)
        rows = torch.arange(bs, device=dev)

        values = self._randint(self.n_values, (bs, s))
        timestamps = torch.arange(s, device=dev).expand(bs, s).contiguous()

        pick = self._randint(s, (bs,))
        query_slots = slots[rows, pick]

        match = slots == query_slots.unsqueeze(1)
        # индекс последнего вхождения: argmax по реверсу
        rev = torch.flip(match, dims=[1]).float()
        answer_positions = s - 1 - rev.argmax(dim=1)
        answers = values[rows, answer_positions]

        self.last_slots = slots
        self.last_query_slots = query_slots
        self.last_answer_positions = answer_positions

        if self.symbolic:
            raise NotImplementedError(
                "symbolic-режим не поддерживает контракт "
                "(slots, codebook, values, timestamps, "
                "query_keys, answers, positions)"
            )

        codebook = torch.randn(
            (bs, self.n_keys, self.key_dim),
            generator=self.gen,
            device=dev,
        )
        codebook = codebook / codebook.norm(dim=-1, keepdim=True)

        query_keys = codebook[rows, query_slots]

        return (
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        )
