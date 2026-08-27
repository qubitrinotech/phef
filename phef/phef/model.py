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
    ):
        batch_size, n_events = values.shape

        query_state = self.query_encoder(query_keys)

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
                ) / math.sqrt(self.dim),
                1,
                slots,
            )

            recency = timestamps.float() / max(
                1,
                self.n_segments - 1,
            )

            scores = self.score_scale * semantic
            scores = scores + self.recency_scale * recency

            memory_values = self.value_emb.weight.to(query_state.dtype)[values]

            if hard:
                max_scores = scores.max(dim=1, keepdim=True).values
                tie_count = scores.ge(max_scores - 1e-6).sum(dim=1)
                self.last_tie_rate = (
                    tie_count.gt(1).float().mean().item()
                )
                index = scores.argmax(dim=1)
                memory_state = memory_values[
                    torch.arange(batch_size, device=values.device),
                    index,
                ]
            else:
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
