import torch, torch.nn as nn

class EventMemory(nn.Module):
    """Persistent per-episode store of event atoms with learned retrieval.
    Хранит (key_emb, value_emb, ts). read() возвращает top-k по learned impact.
    """
    def __init__(self, n_keys, n_values, dim=128, n_segments=16):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, dim)
        self.val_emb = nn.Embedding(n_values, dim)
        self.ts_emb  = nn.Embedding(n_segments + 1, dim)
        self.q_proj  = nn.Linear(dim, dim)
        self.rerank  = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.dim = dim

    def encode(self, keys, vals, ts):
        return self.key_emb(keys) + self.val_emb(vals) + self.ts_emb(ts)  # (B,S,D)

    def read(self, atoms, query_key, top_k, shuffle=False, mask_memory=False):
        B, S, D = atoms.shape
        if mask_memory:
            return torch.zeros(B, top_k, D, device=atoms.device)
        if shuffle:  # shuffled-memory probe: атомы перемешиваются между элементами батча
            atoms = atoms[torch.randperm(B, device=atoms.device)]
        q = self.q_proj(self.key_emb(query_key)).unsqueeze(1).expand(-1, S, -1)  # (B,S,D)
        feat = torch.cat([q, atoms, q * atoms], dim=-1)
        impact = self.rerank(feat).squeeze(-1)  # (B,S)
        idx = impact.topk(min(top_k, S), dim=1).indices
        return torch.gather(atoms, 1, idx.unsqueeze(-1).expand(-1, -1, D))
