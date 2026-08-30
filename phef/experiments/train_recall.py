import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from phef.model import PHEF
from phef.world import BranchingWorld

HARD_EVAL = False


def move_batch(batch, device):
    return tuple(
        tensor.to(device, non_blocking=True)
        for tensor in batch
    )


@torch.inference_mode()
def evaluate(
    model,
    world,
    batch_size,
    device,
    n_batches=10,
    shuffle=False,
    mask_memory=False,
    use_memory=True,
):
    model.eval()

    answer_correct = 0
    retrieval_correct = 0
    total = 0

    # scores вырождены в нули, когда канал памяти отключён,
    # поэтому retrieval там не измеряется вообще.
    score_retrieval = use_memory and not mask_memory

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

        answer_correct += (
            logits.argmax(dim=-1).eq(answers).sum().item()
        )

        if score_retrieval:
            retrieval_correct += (
                retrieval_scores.argmax(dim=-1)
                .eq(answer_positions)
                .sum()
                .item()
            )

        total += answers.numel()

    return {
        "answer_acc": answer_correct / total,
        "retrieval_acc": (
            retrieval_correct / total
            if score_retrieval
            else None
        ),
    }


@torch.inference_mode()
def quick_acc(model, world, device, n=2048, batch_size=1024):
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    while total < n:
        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            _,
        ) = move_batch(world.batch(batch_size), device)

        logits = model(
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            hard=HARD_EVAL,
        )
        correct += logits.argmax(dim=-1).eq(answers).sum().item()
        total += answers.numel()

    if was_training:
        model.train()

    return correct / total


def fmt(value):
    return "n/a" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=16)
    parser.add_argument("--overwrite-p", type=float, default=0.3)

    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--hard-eval", action="store_true")
    parser.add_argument("--dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--retrieval-weight", type=float, default=1.0)
    parser.add_argument(
        "--aux-value-weight",
        type=float,
        default=0.0,
        help="вес auxiliary loss для декодирования value_emb",
    )

    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--acc-every", type=int, default=250)
    parser.add_argument("--acc-n", type=int, default=2048)
    parser.add_argument(
        "--escape-factor",
        type=float,
        default=2.0,
        help="порог выхода с плато в кратности chance (2.0 => 2/n_values)",
    )

    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="путь к JSON-отчёту; без него результат только в stdout",
    )
    parser.add_argument(
        "--best-out",
        type=str,
        default=None,
        help="путь к .pt лучшего checkpoint по sel_acc; без него checkpoint не сохраняется",
    )

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="путь к .pt для инициализации весов перед обучением (curriculum)",
    )

    args = parser.parse_args()

    global HARD_EVAL
    HARD_EVAL = args.hard_eval

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

    train_world = BranchingWorld(seed=args.seed, **common)
    eval_world = BranchingWorld(seed=args.seed + 100_000, **common)
    sel_world = BranchingWorld(seed=args.seed + 200_000, **common)

    use_memory = not args.no_memory

    model = PHEF(
        n_values=args.n_values,
        key_dim=args.key_dim,
        dim=args.dim,
        n_segments=args.n_segments,
        top_k=1,
        use_memory=use_memory,
    ).to(device)

    if args.init_checkpoint is not None:
        init_state = torch.load(args.init_checkpoint, map_location=device)
        if isinstance(init_state, dict):
            for wrap in ("state_dict", "model"):
                if wrap in init_state:
                    init_state = init_state[wrap]
                    break
        own = model.state_dict()
        loaded, skipped = [], []
        for k, v in init_state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(own)
        print(
            f"init-checkpoint={args.init_checkpoint} loaded={len(loaded)} skipped={len(skipped)}",
            flush=True,
        )
        if skipped:
            print("skipped keys: " + ", ".join(skipped), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = device.type == "cuda"
    chance = 1.0 / args.n_values
    retrieval_chance = 1.0 / args.n_segments
    escape_threshold = args.escape_factor * chance

    emergent = use_memory and args.retrieval_weight == 0.0

    print(
        f"device={device} "
        f"memory={use_memory} "
        f"retrieval_weight={args.retrieval_weight} "
        f"emergent_mode={emergent} "
        f"answer_chance={chance:.6f} "
        f"retrieval_chance={retrieval_chance:.6f} "
        f"escape_threshold={escape_threshold:.6f}"
    )

    best_acc = 0.0
    best_epoch = 0
    best_state = None
    escape_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        (
            slots_b,
            codebook,
            values,
            timestamps,
            query_keys,
            answers,
            answer_positions,
        ) = move_batch(
            train_world.batch(args.batch_size),
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits, retrieval_scores = model(
                slots_b,
                codebook,
                values,
                timestamps,
                query_keys,
                return_retrieval=True,
            )

            answer_loss = F.cross_entropy(logits, answers)

            if not use_memory or args.retrieval_weight == 0.0:
                retrieval_loss = logits.new_zeros(())
            else:
                retrieval_loss = F.cross_entropy(
                    retrieval_scores.float(),
                    answer_positions,
                )

            aux_value_loss = F.cross_entropy(
                model.head(model.norm(model.value_emb.weight)),
                torch.arange(args.n_values, device=device),
            )

            loss = (
                answer_loss
                + args.retrieval_weight * retrieval_loss
                + args.aux_value_weight * aux_value_loss
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if args.use_best and epoch % args.acc_every == 0:
            probe = quick_acc(
                model,
                sel_world,
                device,
                n=args.acc_n,
                batch_size=args.eval_batch_size,
            )

            if escape_epoch is None and probe > escape_threshold:
                escape_epoch = epoch

            if probe > best_acc:
                best_acc = probe
                best_epoch = epoch
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }
                if args.best_out:
                    best_dir = os.path.dirname(args.best_out)
                    if best_dir:
                        os.makedirs(best_dir, exist_ok=True)
                    torch.save(
                        {"epoch": best_epoch, "sel_acc": best_acc, "state_dict": best_state},
                        args.best_out,
                    )

            history.append({"epoch": epoch, "sel_acc": probe})

            print(
                f"epoch {epoch:5d} sel_acc {probe:.4f} "
                f"best {best_acc:.4f} @ {best_epoch} "
                f"escape {escape_epoch}"
            )

        if epoch == 1 or epoch % 100 == 0:
            metrics = evaluate(
                model,
                eval_world,
                args.eval_batch_size,
                device,
                n_batches=8,
                use_memory=use_memory,
            )

            print(
                f"epoch {epoch:5d} "
                f"total {loss.item():.4f} "
                f"answer {answer_loss.item():.4f} "
                f"retrieval {retrieval_loss.item():.4f} "
                f"answer_acc {metrics['answer_acc']:.4f} "
                f"retrieval_acc {fmt(metrics['retrieval_acc'])}"
            )

    if args.use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"\nвосстановлен best checkpoint: "
            f"epoch {best_epoch} sel_acc {best_acc:.4f}"
        )

    real = evaluate(
        model,
        eval_world,
        args.eval_batch_size,
        device,
        n_batches=args.eval_batches,
        use_memory=use_memory,
    )

    report = {
        "task": "segmented_kv_recall",
        "use_memory": use_memory,
        "retrieval_weight": args.retrieval_weight,
        "emergent_mode": emergent,
        "answer_chance": chance,
        "retrieval_chance": retrieval_chance,
        "answer_acc": real["answer_acc"],
        "retrieval_acc": real["retrieval_acc"],
        "shuffled_answer_acc": None,
        "masked_answer_acc": None,
        "control_status": None,
        "best_acc": best_acc if args.use_best else None,
        "best_epoch": best_epoch if args.use_best else None,
        "escape_epoch": escape_epoch,
        "escape_threshold": escape_threshold,
        "escape_criterion": "val_accuracy",
        "selection": "val_accuracy" if args.use_best else "final",
        "best_checkpoint": args.best_out,
        "recency_scale": model.recency_scale.item(),
        "score_scale": model.score_scale.item(),
        "memory_gate": model.memory_gate.item(),
        "tie_rate": getattr(model, "last_tie_rate", 0.0),
        "history": history,
        "config": vars(args),
    }

    print("\n=== CONTROLS ===")
    print(f"answer chance    = {chance:.6f}")
    print(f"retrieval chance = {retrieval_chance:.6f}")
    print(f"real answer      = {real['answer_acc']:.6f}")
    print(f"real retrieval   = {fmt(real['retrieval_acc'])}")

    if use_memory:
        shuffled = evaluate(
            model,
            eval_world,
            args.eval_batch_size,
            device,
            n_batches=args.eval_batches,
            shuffle=True,
            use_memory=use_memory,
        )

        masked = evaluate(
            model,
            eval_world,
            args.eval_batch_size,
            device,
            n_batches=args.eval_batches,
            mask_memory=True,
            use_memory=use_memory,
        )

        print(f"shuffled answer  = {shuffled['answer_acc']:.6f}")
        print(f"masked answer    = {masked['answer_acc']:.6f}")

        passed = (
            real["answer_acc"] > 0.90
            and real["retrieval_acc"] is not None
            and real["retrieval_acc"] > 0.90
            and shuffled["answer_acc"] < chance * 2
            and masked["answer_acc"] < chance * 2
        )

        report["shuffled_answer_acc"] = shuffled["answer_acc"]
        report["masked_answer_acc"] = masked["answer_acc"]
        report["control_status"] = "PASS" if passed else "FAIL"

        print(
            f"control status   = "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if emergent and passed:
            print(
                "адресация возникла без retrieval-loss "
                "(retrieval_weight=0)"
            )
    else:
        report["control_status"] = (
            "baseline_at_chance"
            if real["answer_acc"] < chance * 2
            else "baseline_leaks"
        )

        print(
            "baseline без памяти: retrieval не измеряется, "
            f"ожидаем answer_acc около chance={chance:.6f}"
        )
        print(f"control status   = {report['control_status']}")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)

        print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
