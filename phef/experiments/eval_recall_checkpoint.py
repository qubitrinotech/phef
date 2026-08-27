import argparse
import json
import os
import random

import numpy as np
import torch

from phef.model import PHEF
from phef.world import BranchingWorld

HARD_EVAL = False


def move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


@torch.inference_mode()
def evaluate(model, world, batch_size, device, n_batches, shuffle=False, mask_memory=False):
    model.eval()
    correct = 0
    retrieval_correct = 0
    total = 0
    score_retrieval = not mask_memory

    for _ in range(n_batches):
        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = move_batch(world.batch(batch_size), device)

        logits, retrieval_scores = model(
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            shuffle=shuffle,
            mask_memory=mask_memory,
            hard=HARD_EVAL,
            return_retrieval=True,
        )
        correct += logits.argmax(dim=-1).eq(answers).sum().item()
        if score_retrieval:
            retrieval_correct += (
                retrieval_scores.argmax(dim=-1).eq(answer_positions).sum().item()
            )
        total += answers.numel()

    return {
        "answer_acc": correct / total,
        "retrieval_acc": retrieval_correct / total if score_retrieval else None,
        "episodes": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-keys", type=int, default=512)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=65536)
    parser.add_argument("--overwrite-p", type=float, default=0.3)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    global HARD_EVAL
    HARD_EVAL = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    common = dict(
        key_dim=args.key_dim,
        n_keys=args.n_keys,
        n_values=args.n_values,
        n_segments=args.n_segments,
        overwrite_p=args.overwrite_p,
    )
    eval_world = BranchingWorld(seed=args.seed + 100_000, **common)

    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=True,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    print(
        f"device={device} checkpoint={args.checkpoint} "
        f"checkpoint_epoch={checkpoint['epoch']} "
        f"checkpoint_sel_acc={checkpoint['sel_acc']:.6f} "
        f"episodes={args.eval_batch_size * args.eval_batches}"
    )

    real = evaluate(model, eval_world, args.eval_batch_size, device, args.eval_batches)
    shuffled = evaluate(
        model,
        eval_world,
        args.eval_batch_size,
        device,
        args.eval_batches,
        shuffle=True,
    )
    masked = evaluate(
        model,
        eval_world,
        args.eval_batch_size,
        device,
        args.eval_batches,
        mask_memory=True,
    )

    chance = 1.0 / args.n_values
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sel_acc": checkpoint["sel_acc"],
        "hard_eval": HARD_EVAL,
        "answer_chance": chance,
        "real": real,
        "shuffled": shuffled,
        "masked": masked,
        "config": vars(args),
    }

    print(f"real answer      = {real['answer_acc']:.6f}")
    print(f"real retrieval   = {real['retrieval_acc']:.6f}")
    print(f"shuffled answer  = {shuffled['answer_acc']:.6f}")
    print(f"shuffled retrieval= {shuffled['retrieval_acc']:.6f}")
    print(f"masked answer    = {masked['answer_acc']:.6f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
