import math

import torch
import torch.nn as nn


class PHEF(nn.Module):
    def __init__(
        self,
        n_values,
        key_dim=32,
        dim=128,
        n_segments=16,
        top_k=1,
        use_memory=True,
    ):
        super().__init__()

        self.dim = dim
        self.n_segments = n_segments
        self.top_k = top_k
        self.use_memory = use_memory

        self.key_encoder = nn.Sequential(
            nn.Linear(key_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(key_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.value_emb = nn.Embedding(n_values, dim)

        self.recency_scale = nn.Parameter(torch.tensor(0.0))
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.memory_gate = nn.Parameter(torch.tensor(0.0))

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_values)
        self.last_tie_rate = 0.0

    def _streamed_hard_retrieval(
        self,
        slots,
        codebook,
        values,
        timestamps,
        query_state,
        permutation,
        chunk_size,
    ):
        """Return hard retrieval indices and selected values without dense scores."""
        batch_size, n_events = values.shape
        indices = []
        selected_values = []

        for b in range(batch_size):
            src = int(permutation[b])
            memory_keys = self.key_encoder(codebook[src])
            best_score = None
            best_index = None

            for start in range(0, n_events, chunk_size):
                stop = min(start + chunk_size, n_events)
                event_slots = slots[src, start:stop]
                event_times = timestamps[src, start:stop]
                semantic = (
                    memory_keys[event_slots] @ query_state[b]
                ) / math.sqrt(self.dim)
                chunk_scores = (
                    self.score_scale * semantic
                    + self.recency_scale
                    * event_times.float()
                    / max(1, self.n_segments - 1)
                )
                local_score, local_index = chunk_scores.max(dim=0)
                candidate_index = start + int(local_index)

                # Strict > preserves torch.argmax's earliest-index tie rule.
                if best_score is None or local_score > best_score:
                    best_score = local_score
                    best_index = candidate_index

            indices.append(best_index)
            selected_values.append(values[src, best_index])

        return (
            torch.tensor(indices, device=values.device, dtype=torch.long),
            torch.stack(selected_values),
        )

    def forward(
        self,
        slots,
        codebook,
        values,
        timestamps,
        query_keys,
        shuffle=False,
        mask_memory=False,
        hard=False,
        return_retrieval=False,
        stream_hard_chunk_size=0,
        return_retrieval_indices=False,
    ):
        if stream_hard_chunk_size < 0:
            raise ValueError("stream_hard_chunk_size must be >= 0")
        if return_retrieval and return_retrieval_indices:
            raise ValueError(
                "return_retrieval and return_retrieval_indices are exclusive"
            )
        if stream_hard_chunk_size:
            if not hard:
                raise ValueError("streamed hard retrieval requires hard=True")
            if not return_retrieval_indices:
                raise ValueError(
                    "streamed hard retrieval requires return_retrieval_indices=True"
                )
            if self.training:
                raise ValueError("streamed hard retrieval requires model.eval()")
            if torch.is_grad_enabled():
                raise ValueError(
                    "streamed hard retrieval requires disabled gradients"
                )
        elif return_retrieval_indices:
            raise ValueError(
                "return_retrieval_indices requires stream_hard_chunk_size > 0"
            )

        batch_size, n_events = values.shape
        query_state = self.query_encoder(query_keys)

        if stream_hard_chunk_size and (not self.use_memory or mask_memory):
            raise ValueError(
                "streamed hard retrieval requires enabled, unmasked memory"
            )

        if stream_hard_chunk_size:
            if shuffle:
                shift = torch.randint(
                    1,
                    batch_size,
                    (),
                    device=values.device,
                )
                permutation = torch.roll(
                    torch.arange(batch_size, device=values.device),
                    shifts=int(shift.item()),
                )
            else:
                permutation = torch.arange(batch_size, device=values.device)

            index, selected_values = self._streamed_hard_retrieval(
                slots,
                codebook,
                values,
                timestamps,
                query_state,
                permutation,
                stream_hard_chunk_size,
            )
            self.last_tie_rate = None
            memory_state = self.value_emb(selected_values).to(
                query_state.dtype
            )
            gate = torch.sigmoid(self.memory_gate)
            hidden = self.norm(query_state + gate * memory_state)
            logits = self.head(hidden)
            return logits, index

        if not self.use_memory or mask_memory:
            memory_state = torch.zeros_like(query_state)
            scores = torch.zeros(
                batch_size,
                n_events,
                device=values.device,
                dtype=query_state.dtype,
            )
        else:
            if shuffle:
                shift = torch.randint(
                    1,
                    batch_size,
                    (),
                    device=values.device,
                )
                permutation = torch.roll(
                    torch.arange(batch_size, device=values.device),
                    shifts=int(shift.item()),
                )
                slots = slots[permutation]
                codebook = codebook[permutation]
                values = values[permutation]
                timestamps = timestamps[permutation]

            memory_keys = self.key_encoder(codebook)

            semantic = torch.gather(
                torch.einsum(
                    "bd,bkd->bk",
                    query_state,
                    memory_keys,
                )
                / math.sqrt(self.dim),
                1,
                slots,
            )

            recency = timestamps.float() / max(
                1,
                self.n_segments - 1,
            )

            scores = self.score_scale * semantic
            scores = scores + self.recency_scale * recency

            if hard:
                max_scores = scores.max(dim=1, keepdim=True).values
                tie_count = scores.ge(max_scores - 1e-6).sum(dim=1)
                self.last_tie_rate = tie_count.gt(1).float().mean().item()
                index = scores.argmax(dim=1)
                selected_values = values[
                    torch.arange(batch_size, device=values.device),
                    index,
                ]
                memory_state = self.value_emb(selected_values).to(
                    query_state.dtype
                )
            else:
                memory_values = self.value_emb.weight.to(query_state.dtype)[
                    values
                ]
                weights = torch.softmax(scores, dim=1)
                memory_state = torch.einsum(
                    "bs,bsd->bd",
                    weights,
                    memory_values,
                )

        gate = torch.sigmoid(self.memory_gate)
        hidden = self.norm(query_state + gate * memory_state)
        logits = self.head(hidden)

        if return_retrieval:
            return logits, scores

        return logits
