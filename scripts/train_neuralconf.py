#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train NeuralConf on token-level confidence trajectories.

This script is intentionally single-seed and single-max-length. To reproduce a
length sweep, run it once per max_len and evaluate each produced checkpoint
explicitly; no checkpoint discovery logic is used.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralconf.data import (
    TraceDataset,
    build_qid2gt,
    collate_fn,
    compute_bce_class_weights,
    ensure_is_correct,
    load_jsonl,
    save_json,
    set_global_seed,
)
from neuralconf.losses import bce_loss_from_logits_weighted, pairwise_logistic_loss
from neuralconf.metrics import eval_epoch_loss, evaluate_paper_metrics
from neuralconf.model import HybridModel


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_history_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "epoch", "train_loss", "val_loss", "neural_auc", "trace_acc_at_0.5",
        "brier", "majority_vote_acc", "tailconf_vote_acc", "neuralconf_vote_acc",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_global_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    train_items = ensure_is_correct(load_jsonl(args.train_jsonl))
    val_items = ensure_is_correct(load_jsonl(args.val_jsonl))

    w_pos, w_neg, n_pos, n_neg = compute_bce_class_weights(train_items)
    qid2gt_val = build_qid2gt(val_items)

    train_ds = TraceDataset(train_items, max_len=args.max_len, tail_k_default=args.max_len)
    val_ds = TraceDataset(val_items, max_len=args.max_len, tail_k_default=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = HybridModel(d_model=args.d_model, in_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    best_ckpt_path = os.path.join(args.out_dir, f"{args.dataset_name}_maxlen{args.max_len}_seed{args.seed}_best.pt")
    final_ckpt_path = os.path.join(args.out_dir, f"{args.dataset_name}_maxlen{args.max_len}_seed{args.seed}_final.pt")

    def evaluate_epoch(epoch: int, train_loss_value: float = None) -> Dict[str, Any]:
        val_loss = eval_epoch_loss(
            model,
            val_loader,
            device,
            rank_weight=args.rank_weight,
            w_pos=w_pos,
            w_neg=w_neg,
        )
        metrics = evaluate_paper_metrics(
            model,
            val_loader,
            device,
            items=val_items,
            qid2gt=qid2gt_val,
            tail_window=args.max_len,
            bottom_window=1024,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss_value,
            "val_loss": val_loss,
            "neural_auc": metrics.get("neural_auc"),
            "trace_acc_at_0.5": metrics.get("trace_acc_at_0.5"),
            "brier": metrics.get("brier"),
            "majority_vote_acc": metrics.get("majority_vote_acc"),
            "tailconf_vote_acc": metrics.get("tailconf_vote_acc"),
            "neuralconf_vote_acc": metrics.get("neuralconf_vote_acc"),
        }
        history.append(row)
        return row

    row0 = evaluate_epoch(0, None)
    print(
        f"[epoch 0] val_loss={row0['val_loss']:.6f} "
        f"AUC={row0['neural_auc']} acc@0.5={row0['trace_acc_at_0.5']} "
        f"majority={row0['majority_vote_acc']} tail={row0['tailconf_vote_acc']} neural={row0['neuralconf_vote_acc']}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0
        for batch in train_loader:
            seq, mask, _qids, _answers, _tails, is_correct = batch
            seq = seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = is_correct.float().to(device)

            logits = model(seq, mask)
            bce = bce_loss_from_logits_weighted(logits, y, w_pos=w_pos, w_neg=w_neg)
            rank = pairwise_logistic_loss(logits, y) if args.rank_weight > 0.0 else logits.new_tensor(0.0)
            loss = bce + args.rank_weight * rank

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            bs = seq.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

        train_loss = total_loss / max(1, total_samples)
        row = evaluate_epoch(epoch, train_loss)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.6f} val_loss={row['val_loss']:.6f} "
            f"AUC={row['neural_auc']} acc@0.5={row['trace_acc_at_0.5']} "
            f"majority={row['majority_vote_acc']} tail={row['tailconf_vote_acc']} neural={row['neuralconf_vote_acc']}"
        )

        cur_auc = row["neural_auc"] if row["neural_auc"] is not None else -1.0
        if cur_auc > best_auc:
            best_auc = float(cur_auc)
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_auc": best_auc,
                    "config": vars(args),
                    "class_weights": {"w_pos": w_pos, "w_neg": w_neg, "n_pos": n_pos, "n_neg": n_neg},
                },
                best_ckpt_path,
            )
            print(f"[save] best checkpoint -> {best_ckpt_path}")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "best_auc": best_auc,
            "config": vars(args),
            "class_weights": {"w_pos": w_pos, "w_neg": w_neg, "n_pos": n_pos, "n_neg": n_neg},
        },
        final_ckpt_path,
    )

    history_csv = os.path.join(args.out_dir, f"{args.dataset_name}_maxlen{args.max_len}_seed{args.seed}_history.csv")
    write_history_csv(history_csv, history)

    summary = {
        "dataset": args.dataset_name,
        "max_len": args.max_len,
        "seed": args.seed,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "class_weights": {"w_pos": w_pos, "w_neg": w_neg, "n_pos": n_pos, "n_neg": n_neg},
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "best_ckpt": best_ckpt_path,
        "final_ckpt": final_ckpt_path,
        "history_csv": history_csv,
    }
    save_json(os.path.join(args.out_dir, f"{args.dataset_name}_maxlen{args.max_len}_seed{args.seed}_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NeuralConf from token-level confidence trajectories.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--dataset-name", default="dataset")
    p.add_argument("--out-dir", default="outputs/train")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--rank-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
