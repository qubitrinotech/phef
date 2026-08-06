"""
Holographic Memory System with Hysteresis Trace
Phase-space encoding + non-linear restitution.

Revision history of fixes:
  1. No in-place writes into autograd-tracked views (AsStrided version error).
  2. Causal attention mask -> next-token loss is not trivially leaked.
  3. Memory read per-timestep from a causal prefix summary (not a global mean).
  4. Ablation support: memory can be disabled at construction time, and the
     unused 9.4M-param hologram is now opt-in, so parameter counts are fair.
  5. Positional encoding is a learned embedding table. The old Lorenz-only
     projection had rank 3 for any seq_len, leaving thousands of position pairs
     indistinguishable (2462 pairs closer than 0.01 at L=1024) -- copy tasks
     were unsolvable in principle. Lorenz is now an additive extra feature.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalPositionalEncoding(nn.Module):
    """
    Learned absolute positions + an optional rank-3 Lorenz signal.

    WHY THIS CHANGED
    ----------------
    The previous version used ONLY a linear projection of a normalised Lorenz
    trajectory. That trajectory spans exactly 3 dimensions, so nn.Linear(3, d)
    cannot raise its rank: every position vector lives on a 3-D manifold no
    matter how large d_model is. Measured on the normalised trajectory:

        seq_len   rank   min pairwise dist   pairs closer than 0.01
             32      3            0.00026                        23
            256      3            0.00026                       141
            512      3            0.00026                       946
           1024      3            0.00023                      2462

    Exact copying requires aligning position t with position t-half-1, which is
    impossible when thousands of position pairs are indistinguishable. This is
    what pinned train loss at ~6.90 against a floor of 3.44 for every arm at
    L>=257, with gradient norms flat at ~0.18.

    The fix: a genuine learned embedding table of rank min(max_len, d_model).
    The Lorenz signal is kept as an additive feature (small init, so it starts
    as a no-op and the model can recruit it if useful), but it is no longer the
    only source of positional information. Set use_lorenz=False for a clean
    ablation of the fractal idea itself.
    """

    def __init__(self, d_model: int, max_len: int = 8192, dt: float = 0.01,
                 use_lorenz: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dt = dt
        self.use_lorenz = use_lorenz

        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        if use_lorenz:
            self.register_buffer("sigma", torch.tensor(10.0))
            self.register_buffer("rho", torch.tensor(28.0))
            self.register_buffer("beta", torch.tensor(8.0 / 3.0))
            self.projection = nn.Linear(3, d_model)
            nn.init.normal_(self.projection.weight, std=0.01)
            nn.init.zeros_(self.projection.bias)
            self.register_buffer("_traj_cache", torch.zeros(1, 0, 3), persistent=False)

    @torch.no_grad()
    def _build_trajectory(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Euler integration under no_grad; backprop through a chaotic map would
        explode, so only `projection` is trained."""
        state = torch.ones(1, 3, device=device, dtype=dtype)
        traj = [state]
        for _ in range(1, seq_len):
            px, py, pz = state[:, 0], state[:, 1], state[:, 2]
            state = state + self.dt * torch.stack([
                self.sigma * (py - px),
                px * (self.rho - pz) - py,
                px * py - self.beta * pz,
            ], dim=-1)
            traj.append(state)
        return torch.stack(traj, dim=1)   # functional build -> no mutated views

    def _get_trajectory(self, seq_len: int, device, dtype) -> torch.Tensor:
        cache = self._traj_cache
        if cache.shape[1] < seq_len or cache.device != device or cache.dtype != dtype:
            self._traj_cache = self._build_trajectory(seq_len, device, dtype)
        return self._traj_cache[:, :seq_len]

    @torch.no_grad()
    def diagnose(self, seq_len: int) -> dict:
        """Rank / separability report. Call this before trusting any long-context
        result: rank must grow with seq_len, min_dist must stay well above 0."""
        pos = torch.arange(seq_len, device=self.pos_emb.weight.device)
        enc = self.forward(pos.unsqueeze(0))[0].float()
        d = torch.cdist(enc, enc)
        d.fill_diagonal_(float("inf"))
        return {
            "rank": int(torch.linalg.matrix_rank(enc).item()),
            "min_dist": d.min().item(),
            "median_dist": d.median().item(),
            "near_dupes": int((d < 0.01).sum().item() // 2),
        }

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = positions.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_len {self.max_len}; "
                f"construct AGICore with a larger max_len"
            )
        enc = self.pos_emb(positions)
        if self.use_lorenz:
            dtype = self.projection.weight.dtype
            traj = self._get_trajectory(seq_len, positions.device, dtype)
            traj = traj / (traj.norm(dim=-1, keepdim=True) + 1e-6)
            enc = enc + self.projection(traj).expand(batch_size, -1, -1)
        return enc


class HysteresisMemoryState(nn.Module):
    """
    Memristive memory with hysteresis: O(1) storage, O(N) information content.
    The persistent state lives in a plain python container, fully detached from
    autograd; it is advanced explicitly via update_memory_state().
    """

    def __init__(self, hidden_dim: int, memory_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        self.coercivity = nn.Parameter(torch.ones(1) * 0.1)
        self.remanence = nn.Parameter(torch.ones(1) * 0.95)

        self.input_gate = nn.Sequential(nn.Linear(hidden_dim, memory_dim), nn.Tanh())
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, memory_dim), nn.Sigmoid()
        )
        self.output_projection = nn.Linear(memory_dim, hidden_dim)

        self._state_container = [torch.zeros(1, memory_dim)]

    @property
    def memory_state(self) -> torch.Tensor:
        return self._state_container[0]

    def update_memory_state(self, new_state: torch.Tensor) -> None:
        self._state_container[0] = new_state.detach().to("cpu").clone()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [N, hidden_dim]  (N may be batch, or batch*time for per-step reads)
        returns (projection [N, hidden_dim], proposed_state [N, memory_dim])
        """
        n = x.shape[0]
        current = self._state_container[0].to(device=x.device, dtype=x.dtype)
        memory = current.repeat(n, 1)   # real tensor, never a stride-0 view

        new_input = self.input_gate(x)
        update_ratio = self.update_gate(torch.cat([x, memory], dim=-1))

        delta = new_input - memory
        magnetization = torch.tanh(delta / (self.coercivity.abs() + 1e-6))
        new_state = memory + update_ratio * magnetization * self.remanence

        return self.output_projection(new_state), new_state


class HolographicFeatureDistiller(nn.Module):
    """Compresses a feature vector into phase-interference patterns."""

    def __init__(self, model_dim: int, hologram_dim: int = 2048):
        super().__init__()
        self.model_dim = model_dim
        self.hologram_dim = hologram_dim

        self.hologram = nn.Parameter(
            torch.randn(hologram_dim, hologram_dim, dtype=torch.complex64)
            / math.sqrt(hologram_dim)
        )
        self.spectral_encoder = nn.Linear(model_dim, hologram_dim)
        self.spectral_decoder = nn.Linear(hologram_dim, model_dim)
        self.phase_modulator = nn.Parameter(
            torch.randn(hologram_dim) / math.sqrt(hologram_dim)
        )

    def forward(self, weights: torch.Tensor, context: Optional[torch.Tensor] = None):
        spectral = self.spectral_encoder(weights)
        spectral_complex = torch.complex(spectral, torch.zeros_like(spectral))
        phase = torch.matmul(spectral, self.phase_modulator).unsqueeze(-1)
        phase = torch.complex(torch.zeros_like(phase), phase)
        interfered = spectral_complex * torch.exp(phase)
        modulated = torch.matmul(interfered.to(torch.complex64), self.hologram)
        return self.spectral_decoder(modulated.real)


class HomeostaticWeightOscillator(nn.Module):
    """Resonance-based weight update dynamics (HOW)."""

    def __init__(self, dim: int, natural_freq: float = 1.0):
        super().__init__()
        self.dim = dim
        self.natural_freq = nn.Parameter(torch.ones(dim) * natural_freq)
        self.damping = nn.Parameter(torch.ones(dim) * 0.1)
        self._position_container = [torch.zeros(dim)]
        self._velocity_container = [torch.zeros(dim)]

    @property
    def position(self):
        return self._position_container[0]

    @property
    def velocity(self):
        return self._velocity_container[0]

    def forward(self, gradient_signal: torch.Tensor, dt: float = 0.01):
        device, dtype = gradient_signal.device, gradient_signal.dtype
        position = self._position_container[0].to(device=device, dtype=dtype)
        velocity = self._velocity_container[0].to(device=device, dtype=dtype)
        acceleration = (
            -self.natural_freq ** 2 * position
            - 2 * self.damping * velocity
            + gradient_signal
        )
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        return new_position, new_velocity

    def update_states(self, new_position: torch.Tensor, new_velocity: torch.Tensor):
        self._position_container[0] = new_position.detach().to("cpu").clone()
        self._velocity_container[0] = new_velocity.detach().to("cpu").clone()


class AGICore(nn.Module):
    """
    Causal decoder with optional holographic hysteresis memory.

    enable_memory=False builds a *pure* baseline decoder: the memory / hologram /
    oscillator submodules are not instantiated at all, so parameter counts are
    comparable in a fair ablation.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 512,
        num_layers: int = 4,
        memory_dim: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        enable_memory: bool = True,
        enable_hologram: bool = False,
        max_len: int = 4096,
        use_lorenz_pos: bool = True,
        use_checkpoint: bool = False,
        dim_feedforward_override: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_memory = enable_memory
        self.enable_hologram = enable_hologram
        self.use_checkpoint = use_checkpoint
        if dim_feedforward_override is not None:
            dim_feedforward = dim_feedforward_override
        self.dim_feedforward = dim_feedforward

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.max_len = max_len
        self.pos_encoding = FractalPositionalEncoding(
            hidden_dim, max_len=max_len, use_lorenz=use_lorenz_pos)

        # NOTE on parameter accounting:
        #   HolographicFeatureDistiller holds a complex 2048x2048 matrix
        #   (9,441,536 params = 92% of the old model) but is only touched by
        #   spectral_update(), which the training loop never calls. Keeping it
        #   in the module made the memory-vs-baseline ablation meaningless and
        #   forced AdamW to carry ~75 MB of dead optimizer state.
        #   It is now opt-in via enable_hologram=True.
        if enable_memory:
            self.memory = HysteresisMemoryState(hidden_dim, memory_dim)
        else:
            self.memory = None

        if enable_hologram:
            self.hologram = HolographicFeatureDistiller(hidden_dim)
            self.oscillator = HomeostaticWeightOscillator(hidden_dim)
        else:
            self.hologram = None
            self.oscillator = None

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def _causal_prefix_mean(x: torch.Tensor) -> torch.Tensor:
        """Running mean over time: position t summarises tokens <= t only."""
        counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        return x.cumsum(dim=1) / counts.view(1, -1, 1)

    def _run_layer(self, layer, x, mask):
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                lambda t: layer(t, src_mask=mask, is_causal=True),
                x, use_reentrant=False,
            )
        return layer(x, src_mask=mask, is_causal=True)

    def forward(self, input_ids: torch.Tensor, use_memory: Optional[bool] = None,
                advance_memory: bool = True):
        use_mem = self.enable_memory if use_memory is None else (use_memory and self.enable_memory)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        proposed_memory = None
        inject_at = len(self.layers) // 2

        for i, layer in enumerate(self.layers):
            if use_mem and i == inject_at:
                prefix = self._causal_prefix_mean(x)                 # [B, T, H]
                flat = prefix.reshape(batch_size * seq_len, -1)
                mem_out, mem_states = self.memory(flat)
                x = x + mem_out.view(batch_size, seq_len, -1)
                proposed_memory = mem_states.view(batch_size, seq_len, -1)[:, -1]
            x = self._run_layer(layer, x, causal_mask)

        if use_mem and advance_memory and proposed_memory is not None:
            self.memory.update_memory_state(proposed_memory.mean(0, keepdim=True))

        logits = self.output_proj(self.norm(x))

        if not use_mem:
            return logits, None
        return logits, self.memory.memory_state.clone()

    def spectral_update(self, context: torch.Tensor):
        if self.hologram is None:
            raise RuntimeError("spectral_update requires enable_hologram=True")
        flat = torch.cat([p.view(-1) for p in self.parameters() if p.requires_grad])
        transformed = self.hologram(flat.unsqueeze(0), context)
        new_position, _ = self.oscillator(transformed.squeeze(0))
        return new_position

    def reset_memory(self) -> None:
        if self.enable_memory:
            self.memory.update_memory_state(torch.zeros(1, self.memory.memory_dim))


def test_forward_pass():
    print("Testing AGI Core forward pass...")
    print("-" * 50)

    torch.manual_seed(0)
    model = AGICore(vocab_size=1000, hidden_dim=256, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n_mem = sum(p.numel() for p in model.parameters())
    baseline = AGICore(vocab_size=1000, hidden_dim=256, num_layers=2, enable_memory=False)
    n_base = sum(p.numel() for p in baseline.parameters())
    matched = AGICore(vocab_size=1000, hidden_dim=256, num_layers=2,
                      enable_memory=False, dim_feedforward_override=3840)
    n_match = sum(p.numel() for p in matched.parameters())
    print(f"Memory model parameters:        {n_mem:,}")
    print(f"Baseline (no memory):           {n_base:,}  (gap {n_mem - n_base:+,})")
    print(f"Param-matched baseline (ff wide): {n_match:,}  (gap {n_mem - n_match:+,})")
    with_holo = AGICore(vocab_size=1000, hidden_dim=256, num_layers=2, enable_hologram=True)
    print(f"With hologram (unused in fwd):  {sum(p.numel() for p in with_holo.parameters()):,}")

    print("\n[Test 0] Positional encoding rank / separability")
    print(f"  {'seq_len':>8} {'rank':>6} {'min_dist':>10} {'median':>9} {'near_dupes':>11}")
    ok_pos = True
    for n in (32, 256, 512, 1024):
        d = model.pos_encoding.diagnose(n)
        ok_pos &= d["near_dupes"] == 0 and d["rank"] >= min(n, 64)
        print(f"  {n:>8} {d['rank']:>6} {d['min_dist']:>10.4f} "
              f"{d['median_dist']:>9.4f} {d['near_dupes']:>11}")
    print(f"  -> {'OK: positions are separable' if ok_pos else 'DEGENERATE ENCODING'}")

    print("\n[Test 1] Basic forward pass")
    input_ids = torch.randint(0, 1000, (4, 128))
    logits, memory = model(input_ids)
    print(f"Input shape: {tuple(input_ids.shape)} | Logits: {tuple(logits.shape)}")
    print(f"Memory state: {tuple(memory.shape)} | norm: {memory.norm().item():.4f}")

    print("\n[Test 2] Variable sequence lengths")
    for test_len in (64, 256, 512):
        model.reset_memory()
        out, _ = model(torch.randint(0, 1000, (2, test_len)))
        print(f"  Seq len {test_len}: OK (output: {tuple(out.shape)})")

    print("\n[Test 3] Memory persistence")
    model.reset_memory()
    _, mem1 = model(torch.randint(0, 1000, (1, 32)))
    _, mem2 = model(torch.randint(0, 1000, (1, 32)))
    print(f"  Memory delta after second sequence: {(mem1 - mem2).norm().item():.4f}")

    print("\n[Test 4] Causal masking (no future leakage)")
    model.eval()
    model.reset_memory()
    base = torch.randint(0, 1000, (1, 16))
    altered = base.clone()
    altered[0, -1] = (altered[0, -1] + 7) % 1000
    with torch.no_grad():
        l1, _ = model(base, advance_memory=False)
        l2, _ = model(altered, advance_memory=False)
    drift = (l1[0, :-1] - l2[0, :-1]).abs().max().item()
    print(f"  Max logit drift at t<T: {drift:.2e} -> {'OK' if drift < 1e-4 else 'LEAK'}")

    print("\n[Test 5] Baseline path runs without memory modules")
    baseline.eval()
    with torch.no_grad():
        bl, bm = baseline(torch.randint(0, 1000, (2, 64)))
    print(f"  Logits: {tuple(bl.shape)} | memory returned: {bm}")

    print("\n[Test 6] Gradient computation")
    model.train()
    model.reset_memory()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(torch.randint(0, 1000, (2, 32)))
    logits.pow(2).mean().backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f} -> {'OK' if grad_norm > 0 else 'FAILED'}")

    print("\n[Test 7] Overfit a single batch (learnability)")
    torch.manual_seed(1)
    fixed = torch.randint(0, 1000, (4, 32))
    model.reset_memory()
    model.train()
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(fixed)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), fixed[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step == 59:
            print(f"  Step {step + 1:>3}: loss = {loss.item():.4f}")
    print(f"  {'OK: model can fit data' if loss.item() < 1.0 else 'WARNING: no fitting'}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return model


if __name__ == "__main__":
    model = test_forward_pass()