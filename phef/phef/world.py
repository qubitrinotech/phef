import numpy as np
import torch


class BranchingWorld:
    """
    Segmented key-value world with content-based keys.

    Генерация векторизована по батчу: цикл идёт только по сегментам,
    а не по эпизодам. Ключи сэмплируются заново каждый батч,
    поэтому lookup-таблица структурно невозможна.
    """

    def __init__(
        self,
        n_keys=32,
        n_values=64,
        n_segments=16,
        overwrite_p=0.3,
        key_dim=32,
        seed=0,
        symbolic=False,
    ):
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_segments = n_segments
        self.overwrite_p = overwrite_p
        self.key_dim = key_dim
        self.symbolic = symbolic
        self.rng = np.random.default_rng(seed)

    def _sample_slots(self, bs):
        slots = self.rng.integers(
            0,
            self.n_keys,
            size=(bs, self.n_segments),
        )

        reuse = self.rng.random((bs, self.n_segments)) < self.overwrite_p
        reuse[:, 0] = False

        back = self.rng.random((bs, self.n_segments))
        rows = np.arange(bs)

        for ts in range(1, self.n_segments):
            source = (back[:, ts] * ts).astype(np.int64)
            source = np.clip(source, 0, ts - 1)

            candidate = slots[rows, source]
            slots[:, ts] = np.where(
                reuse[:, ts],
                candidate,
                slots[:, ts],
            )

        return slots

    def batch(self, bs):
        slots = self._sample_slots(bs)
        rows = np.arange(bs)

        values = self.rng.integers(
            0,
            self.n_values,
            size=(bs, self.n_segments),
        )

        timestamps = np.broadcast_to(
            np.arange(self.n_segments),
            (bs, self.n_segments),
        ).copy()

        pick = self.rng.integers(0, self.n_segments, size=bs)
        query_slots = slots[rows, pick]

        match = slots == query_slots[:, None]
        answer_positions = (
            self.n_segments
            - 1
            - np.argmax(match[:, ::-1], axis=1)
        )
        answers = values[rows, answer_positions]

        values_t = torch.from_numpy(values.astype(np.int64))
        timestamps_t = torch.from_numpy(timestamps.astype(np.int64))
        answers_t = torch.from_numpy(answers.astype(np.int64))
        positions_t = torch.from_numpy(
            answer_positions.astype(np.int64)
        )

        if self.symbolic:
            return (
                torch.from_numpy(slots.astype(np.int64)),
                values_t,
                timestamps_t,
                torch.from_numpy(query_slots.astype(np.int64)),
                answers_t,
                positions_t,
            )

        codebook = self.rng.standard_normal(
            (bs, self.n_keys, self.key_dim)
        ).astype(np.float32)

        codebook /= np.linalg.norm(
            codebook,
            axis=-1,
            keepdims=True,
        )

        event_keys = np.take_along_axis(
            codebook,
            slots[:, :, None],
            axis=1,
        )
        query_keys = codebook[rows, query_slots]

        self.last_slots = torch.from_numpy(slots.astype(np.int64))
        self.last_query_slots = torch.from_numpy(
            query_slots.astype(np.int64)
        )

        return (
            torch.from_numpy(event_keys),
            values_t,
            timestamps_t,
            torch.from_numpy(query_keys),
            answers_t,
            positions_t,
        )
