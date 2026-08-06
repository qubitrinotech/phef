"""
Training / ablation harness for the Holographic Memory AGI prototype.

Design notes
------------
* Data is generated ON THE FLY. A fixed tensor of samples lets the model
  memorise the unpredictable prefix, which drags train loss below its
  information-theoretic floor and destroys the diagnostic value of the number.
  With fresh samples every step, train loss must converge to

      floor = (half - 1) * ln(V) / (2*half)      [nats/token]

  and any value below that is a bug or a leak.

* `--no-memory` builds a pure causal decoder (memory submodules are not even
  instantiated), so the ablation is fair.

* The copy task is scored only on the second (copied) half, where chance
  accuracy is 1/V.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from core.holographic_memory import AGICore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available, skipping visualizations")

SEP_OFFSET = 1  # highest token id is reserved as the separator


class CopyTaskStream(IterableDataset):
    """Infinite stream of [prefix, SEP, prefix]; nothing to memorise."""

    def __init__(self, seq_len: int, vocab_size: int, steps: int, batch_size: int, seed: int = 0):
        self.sep_id = vocab_size - SEP_OFFSET
        self.half = (seq_len - 1) // 2
        self.steps = steps
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + int(time.time_ns() % 10_000))
        for _ in range(self.steps):
            prefix = torch.randint(0, self.sep_id, (self.batch_size, self.half), generator=g)
            sep = torch.full((self.batch_size, 1), self.sep_id, dtype=torch.long)
            yield torch.cat([prefix, sep, prefix], dim=1)


class CopyTaskDataset(Dataset):
    """Fixed-sample variant, used for a held-out validation set."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 0):
        self.sep_id = vocab_size - SEP_OFFSET
        self.half = (seq_len - 1) // 2
        g = torch.Generator().manual_seed(seed)
        prefix = torch.randint(0, self.sep_id, (num_samples, self.half), generator=g)
        sep = torch.full((num_samples, 1), self.sep_id, dtype=torch.long)
        self.data = torch.cat([prefix, sep, prefix], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NoiseStream(IterableDataset):
    """Pure noise control: loss cannot fall below ln(V)."""

    def __init__(self, seq_len, vocab_size, steps, batch_size, seed=0):
        self.seq_len, self.vocab_size = seq_len, vocab_size
        self.steps, self.batch_size, self.seed = steps, batch_size, seed
        self.half = (seq_len - 1) // 2

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            yield torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), generator=g)


def loss_floor(half: int, vocab_size: int) -> float:
    """Information-theoretic minimum of the full-sequence next-token loss."""
    predicted = 2 * half            # positions 1..2*half
    unpredictable = half - 1        # prefix tokens after the first
    return unpredictable * np.log(vocab_size) / predicted


@torch.no_grad()
def evaluate_copy(model, dataloader, device, half: int, amp: bool):
    model.eval()
    tot_loss = tot_correct = tot_tokens = 0
    seq_exact = seq_total = 0

    for batch in dataloader:
        batch = batch.to(device)
        model.reset_memory()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits, _ = model(batch, advance_memory=False)
        logits = logits.float()

        pred = logits[:, half:-1]
        targets = batch[:, half + 1:]
        tot_loss += F.cross_entropy(
            pred.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
        ).item()
        correct = pred.argmax(-1) == targets
        tot_correct += correct.sum().item()
        tot_tokens += targets.numel()
        seq_exact += correct.all(dim=1).sum().item()
        seq_total += targets.shape[0]

    return tot_loss / tot_tokens, tot_correct / tot_tokens, seq_exact / seq_total


@torch.no_grad()
def probe_memory(model, device, vocab_size, seq_len):
    """Does the persistent state discriminate two different sequences?"""
    if not model.enable_memory:
        return None
    model.eval()
    hi = vocab_size - SEP_OFFSET

    model.reset_memory()
    s1 = torch.randint(0, hi, (1, seq_len), device=device)
    s1[0, 0] = 42
    logits1, mem1 = model(s1)

    model.reset_memory()
    s2 = torch.randint(0, hi, (1, seq_len), device=device)
    s2[0, 0] = 99
    logits2, mem2 = model(s2)

    diff = torch.norm(mem1 - mem2).item()
    rel = diff / (mem1.norm().item() + 1e-9)
    top1 = logits1[0, -1].topk(5).indices.tolist()
    top2 = logits2[0, -1].topk(5).indices.tolist()
    return {"abs": diff, "rel": rel, "differ": top1 != top2,
            "norm1": mem1.norm().item(), "norm2": mem2.norm().item()}


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "nomem" if args.no_memory else "mem"
    if args.no_memory and args.ffn_dim:
        mode = "nomem-matched"
    if args.loss_on_copy_only:
        mode += "-copyloss"
    tag = f"{args.task}_L{args.seq_len}_{mode}"

    print("=" * 68)
    print(f"AGI Holographic Memory - run: {tag}")
    print("=" * 68)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    model = AGICore(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        memory_dim=args.memory_dim,
        enable_memory=not args.no_memory,
        enable_hologram=args.enable_hologram,
        max_len=max(4096, 2 * args.seq_len),
        use_lorenz_pos=not args.no_lorenz_pos,
        use_checkpoint=args.checkpoint,
        dim_feedforward_override=args.ffn_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} | memory: {'OFF' if args.no_memory else 'ON'}"
          f" | ffn: {model.dim_feedforward} | hologram: {'ON' if args.enable_hologram else 'OFF'}"
          f" | lorenz-pos: {'OFF' if args.no_lorenz_pos else 'ON'}")

    pos_diag = model.pos_encoding.diagnose(min(args.seq_len, model.max_len))
    print(f"Positional encoding: rank {pos_diag['rank']} | min dist "
          f"{pos_diag['min_dist']:.4f} | near-duplicate pairs {pos_diag['near_dupes']}")
    if pos_diag["near_dupes"] > 0:
        print("  WARNING: indistinguishable positions -> exact copying is impossible")

    steps = args.steps_per_epoch
    if args.task == "copy":
        stream = CopyTaskStream(args.seq_len, args.vocab_size, steps, args.batch_size, args.seed)
    else:
        stream = NoiseStream(args.seq_len, args.vocab_size, steps, args.batch_size, args.seed)
    half = stream.half

    val_ds = CopyTaskDataset(args.val_samples, args.seq_len, args.vocab_size, seed=args.seed + 9999)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size)

    floor = 0.0 if args.loss_on_copy_only else loss_floor(half, args.vocab_size)
    print(f"Task: {args.task} | prefix={half} | total len={2 * half + 1}")
    if args.loss_on_copy_only:
        print("Loss on the copied half ONLY -> floor is 0.0; train and copy loss "
              "are directly comparable (their gap measures overfitting)")
    else:
        print(f"Train-loss floor (perfect copy, zero memorisation): {floor:.4f}")
    print(f"Chance loss on copied half: {np.log(args.vocab_size):.4f} "
          f"| chance acc: {100 / args.vocab_size:.3f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    total_steps = max(args.epochs * steps, 1)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
            1 + np.cos(np.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp = args.amp and device.type == "cuda"

    print("\nTraining")
    print("-" * 68)
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = nb = 0
        loader = DataLoader(stream, batch_size=None)
        for batch in tqdm(loader, total=steps, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False):
            batch = batch.to(device, non_blocking=True)
            model.reset_memory()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(batch)
                if args.loss_on_copy_only:
                    pred = logits[:, half:-1].reshape(-1, logits.size(-1)).float()
                    tgt = batch[:, half + 1:].reshape(-1)
                else:
                    pred = logits[:, :-1].reshape(-1, logits.size(-1)).float()
                    tgt = batch[:, 1:].reshape(-1)
                loss = F.cross_entropy(pred, tgt)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            nb += 1

        train_loss = running / max(nb, 1)
        val_loss, tok_acc, seq_acc = evaluate_copy(model, val_dl, device, half, amp)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            "token_acc": tok_acc, "seq_acc": seq_acc,
            "lr": scheduler.get_last_lr()[0], "grad_norm": float(gnorm),
            "vram_gb": peak,
        })
        print(f"Ep {epoch + 1:>3}/{args.epochs} | train {train_loss:.4f} "
              f"(floor {floor:.3f}) | copy loss {val_loss:.4f} | tok {tok_acc * 100:5.1f}% "
              f"| seq {seq_acc * 100:5.1f}% | gn {float(gnorm):.2f} | lr {scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    val_loss, tok_acc, seq_acc = evaluate_copy(model, val_dl, device, half, amp)
    below = (not args.loss_on_copy_only) and train_loss < floor - 0.02

    print("\n" + "=" * 68)
    print("Final")
    print("=" * 68)
    print(f"Copy-half loss {val_loss:.4f} | token acc {tok_acc * 100:.2f}% "
          f"| exact-sequence acc {seq_acc * 100:.2f}%")
    print(f"Train loss {train_loss:.4f} vs floor {floor:.4f} "
          f"-> {'BELOW FLOOR (leak/memorisation!)' if below else 'consistent'}")
    print(f"Wall time: {elapsed / 60:.1f} min"
          + (f" | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
             if device.type == "cuda" else ""))

    mem = probe_memory(model, device, args.vocab_size, args.seq_len)
    if mem:
        print(f"\nMemory probe: |delta| {mem['abs']:.4f} ({mem['rel'] * 100:.2f}% of norm) "
              f"| top-5 differ: {'yes' if mem['differ'] else 'no'}")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"history_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    summary = {
        "tag": tag, "params": n_params, "memory_enabled": not args.no_memory,
        "seq_len": args.seq_len, "half": half, "floor": floor,
        "final_train_loss": train_loss, "copy_loss": val_loss,
        "token_acc": tok_acc, "seq_acc": seq_acc,
        "below_floor": bool(below), "minutes": elapsed / 60,
        "memory_probe": mem, "pos_diag": pos_diag, "config": vars(args),
    }
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if MATPLOTLIB_AVAILABLE:
        ep = [h["epoch"] for h in history]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        a1.plot(ep, [h["train_loss"] for h in history], label="train", lw=2)
        a1.plot(ep, [h["val_loss"] for h in history], label="copy-half val", lw=2)
        a1.axhline(floor, ls="--", c="tab:red", label=f"train floor {floor:.2f}")
        a1.axhline(np.log(args.vocab_size), ls=":", c="grey", label="chance ln(V)")
        a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(); a1.grid(alpha=0.3)
        a1.set_title(f"Loss ({tag})")
        a2.plot(ep, [h["token_acc"] * 100 for h in history], marker="o", lw=2, label="token")
        a2.plot(ep, [h["seq_acc"] * 100 for h in history], marker="s", lw=2, label="exact seq")
        a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy, %"); a2.legend(); a2.grid(alpha=0.3)
        a2.set_ylim(-2, 102); a2.set_title("Copy accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"curves_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    torch.save({"model_state_dict": model.state_dict(), "summary": summary},
               os.path.join(args.out_dir, f"ckpt_{tag}.pt"))
    print(f"\nArtifacts in {args.out_dir}/ (history_{tag}.csv, summary_{tag}.json, curves_{tag}.png)")
    return summary


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=65, help="2*half+1")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--memory-dim", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=1000)
    # Defaults below are the VALIDATED gate config: a plain causal decoder
    # reaches 99.94% token / 98.05% exact-seq on L=65 with these. The old
    # lr=3e-4 / 15-epoch budget cut training off at epoch ~11, exactly when the
    # induction head forms (grad norm spikes to ~5.8), so every arm looked dead.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--task", choices=["copy", "noise"], default="copy")
    p.add_argument("--no-memory", action="store_true", help="ablation: pure causal decoder")
    p.add_argument("--ffn-dim", type=int, default=None,
                   help="override dim_feedforward; use 2944 on the no-memory "
                        "baseline (4 layers, h=256) to param-match the memory model")
    p.add_argument("--loss-on-copy-only", action="store_true",
                   help="train only on the copied half; the prefix is pure noise "
                        "and contributes ~78%% of the loss at L=65 (99.8%% at "
                        "L=1025) while being unlearnable, which drowns the copy "
                        "signal in the gradient")
    p.add_argument("--no-lorenz-pos", action="store_true",
                   help="ablate the Lorenz term, keep learned positions only")
    p.add_argument("--enable-hologram", action="store_true",
                   help="instantiate the 9.4M-param HolographicFeatureDistiller "
                        "(unused by forward(); off by default)")
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--checkpoint", action="store_true", help="gradient checkpointing for long seqs")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())