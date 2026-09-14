"""CPU regression checks for legacy and streamed hard retrieval.

The streamed path is production code. This script compares it with the
current dense hard path, including the shuffled-memory control.
"""

import torch

from phef.model import PHEF


def make_inputs(seed, batch_size, n_events, n_slots, key_dim, n_values):
    generator = torch.Generator().manual_seed(seed)
    slots = torch.randint(
        n_slots, (batch_size, n_events), generator=generator
    )
    codebook = torch.randn(
        batch_size, n_slots, key_dim, generator=generator
    )
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)
    values = torch.randint(
        n_values, (batch_size, n_events), generator=generator
    )
    timestamps = torch.arange(n_events).expand(batch_size, -1).clone()
    query_keys = torch.randn(
        batch_size, key_dim, generator=generator
    )
    return slots, codebook, values, timestamps, query_keys


def check_case(seed, batch_size, n_events, chunk_size, shuffle):
    torch.manual_seed(seed)
    n_slots = 11
    key_dim = 4
    dim = 8
    n_values = 5

    model = PHEF(
        n_values=n_values,
        key_dim=key_dim,
        dim=dim,
        n_segments=n_events,
    ).eval()
    slots, codebook, values, timestamps, query_keys = make_inputs(
        seed + 10_000,
        batch_size,
        n_events,
        n_slots,
        key_dim,
        n_values,
    )

    with torch.inference_mode():
        rng_state = torch.get_rng_state()
        legacy_logits, legacy_scores = model(
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            shuffle=shuffle,
            hard=True,
            return_retrieval=True,
        )
        torch.set_rng_state(rng_state)
        logits, indices = model(
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            shuffle=shuffle,
            hard=True,
            stream_hard_chunk_size=chunk_size,
            return_retrieval_indices=True,
        )
        if shuffle:
            shift_generator = torch.Generator()
            shift_generator.set_state(rng_state)
            shift = torch.randint(1, batch_size, (), generator=shift_generator)
            permutation = torch.roll(
                torch.arange(batch_size), shifts=int(shift.item())
            )
        else:
            permutation = torch.arange(batch_size)

    legacy_indices = legacy_scores.argmax(dim=1)
    legacy_values = values[permutation][
        torch.arange(batch_size), legacy_indices
    ]
    if not torch.equal(indices, legacy_indices):
        print(
            "INDEX_MISMATCH "
            f"seed={seed} batch_size={batch_size} "
            f"n_events={n_events} chunk_size={chunk_size} "
            f"shuffle={shuffle} permutation={permutation.tolist()}",
            flush=True,
        )
        for b in range(batch_size):
            dense_top2 = torch.topk(legacy_scores[b], k=2)
            print(
                f"row={b} dense_index={int(legacy_indices[b])} "
                f"streamed_index={int(indices[b])} "
                f"dense_top2_indices={dense_top2.indices.tolist()} "
                f"dense_top2_scores={dense_top2.values.tolist()}",
                flush=True,
            )
        raise AssertionError("chunked retrieval index mismatch")
    streamed_values = values[permutation][
        torch.arange(batch_size), indices
    ]
    assert torch.equal(streamed_values, legacy_values)
    torch.testing.assert_close(logits, legacy_logits, rtol=1e-6, atol=1e-7)


def check_tie_case():
    """Ensure chunk boundaries preserve earliest-position tie resolution."""
    torch.manual_seed(77)
    batch_size = 2
    n_events = 9
    n_slots = 3
    key_dim = 4
    dim = 8
    n_values = 5
    chunk_size = 4

    model = PHEF(
        n_values=n_values,
        key_dim=key_dim,
        dim=dim,
        n_segments=n_events,
    ).eval()
    with torch.no_grad():
        for parameter in model.key_encoder.parameters():
            parameter.zero_()
        for parameter in model.query_encoder.parameters():
            parameter.zero_()
        model.score_scale.zero_()
        model.recency_scale.zero_()

    slots = torch.tensor(
        [
            [1, 0, 2, 0, 1, 2, 0, 2, 1],
            [2, 1, 0, 1, 2, 0, 1, 0, 2],
        ],
        dtype=torch.long,
    )
    codebook = torch.randn(batch_size, n_slots, key_dim)
    values = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2, 3],
            [4, 3, 2, 1, 0, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    timestamps = torch.arange(n_events).expand(batch_size, -1).clone()
    query_keys = torch.randn(batch_size, key_dim)

    with torch.inference_mode():
        legacy_logits, legacy_scores = model(
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            shuffle=False,
            hard=True,
            return_retrieval=True,
        )
        logits, indices = model(
            slots,
            codebook,
            values,
            timestamps,
            query_keys,
            shuffle=False,
            hard=True,
            stream_hard_chunk_size=chunk_size,
            return_retrieval_indices=True,
        )
        permutation = torch.arange(batch_size)

    legacy_indices = legacy_scores.argmax(dim=1)
    legacy_values = values[
        torch.arange(batch_size), legacy_indices
    ]

    assert torch.equal(legacy_scores, torch.zeros_like(legacy_scores))
    assert torch.equal(legacy_indices, torch.zeros(batch_size, dtype=torch.long))
    assert torch.equal(indices, legacy_indices)
    streamed_values = values[
        torch.arange(batch_size), indices
    ]
    assert torch.equal(streamed_values, legacy_values)
    torch.testing.assert_close(logits, legacy_logits, rtol=0, atol=0)



def check_api_guards():
    """Reject unsupported streamed-mode combinations explicitly."""
    model = PHEF(
        n_values=3,
        key_dim=2,
        dim=4,
        n_segments=3,
    )
    slots = torch.zeros(2, 3, dtype=torch.long)
    codebook = torch.randn(2, 1, 2)
    values = torch.zeros(2, 3, dtype=torch.long)
    timestamps = torch.arange(3).expand(2, -1).clone()
    query_keys = torch.randn(2, 2)

    try:
        with torch.inference_mode():
            model(
                slots, codebook, values, timestamps, query_keys,
                hard=True,
                stream_hard_chunk_size=2,
                return_retrieval_indices=True,
            )
    except ValueError as error:
        assert "model.eval()" in str(error)
    else:
        raise AssertionError("streamed mode accepted model.train()")

    model.eval()
    try:
        model(
            slots, codebook, values, timestamps, query_keys,
            hard=True,
            stream_hard_chunk_size=2,
            return_retrieval_indices=True,
        )
    except ValueError as error:
        assert "disabled gradients" in str(error)
    else:
        raise AssertionError("streamed mode accepted enabled gradients")

    with torch.inference_mode():
        try:
            model(
                slots, codebook, values, timestamps, query_keys,
                hard=True,
                return_retrieval=True,
                stream_hard_chunk_size=2,
                return_retrieval_indices=True,
            )
        except ValueError as error:
            assert "exclusive" in str(error)
        else:
            raise AssertionError("streamed mode accepted two retrieval outputs")

def main():
    cases = (
        (7, 2, 17, 5),
        (11, 2, 19, 6),
        (19, 3, 17, 4),
        (23, 3, 22, 7),
    )
    checked = 0

    for seed, batch_size, n_events, chunk_size in cases:
        for shuffle in (False, True):
            check_case(seed, batch_size, n_events, chunk_size, shuffle)
            checked += 1

    check_tie_case()
    check_api_guards()
    print(
        "cpu chunked-hard equivalence matrix: PASS "
        f"random_cases={checked} tie_case=PASS api_guards=PASS"
    )


if __name__ == "__main__":
    main()
