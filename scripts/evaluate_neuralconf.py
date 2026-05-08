#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate NeuralConf and confidence-summary baselines for the paper.

This script intentionally does not perform checkpoint discovery and does not
require any precomputed score file. All scores are computed directly from:
  1) the evaluation JSONL, and
  2) the explicitly provided NeuralConf checkpoint.

Standard evaluation writes the scalar metrics and the aggregation table. Per-
trace scores are saved only when --save-trace-scores is passed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader

from neuralconf.data import (
    TraceDataset,
    build_qid2gt,
    collate_fn,
    ensure_is_correct,
    get_confidences,
    load_jsonl,
    make_fixed_sequence,
    save_json,
)
from neuralconf.io import infer_d_model_from_state_dict, load_state_dict_from_checkpoint
from neuralconf.metrics import compute_auc_from_scores, compute_dbi, compute_voting_accuracy
from neuralconf.model import HybridModel
from neuralconf.voting import compute_bottom10_group_confidence, compute_tail_confidence


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def format_float(x: Any) -> Any:
    if x is None:
        return None
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return None
        return xf
    except Exception:
        return x


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "dataset",
        "alignment",
        "max_len",
        "select_window_idx",
        "ckpt",
        "n_traces",
        "n_questions",
        "raw_input_dbi",
        "raw_input_dbi_n",
        "neural_dbi",
        "neural_auc",
        "tail_auc",
        "bottom10_auc",
        "bottom10_grouping_length",
        "bottom10_source",
        "trace_acc_at_0.5",
        "brier",
        "majority_vote_acc",
        "bottom10_vote_acc_top0.1",
        "bottom10_vote_acc_top0.9",
        "tailconf_vote_acc_top0.1",
        "tailconf_vote_acc_top0.9",
        "neuralconf_vote_acc",
        "aggregation_csv",
        "trace_scores_csv",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_model_from_ckpt(ckpt_path: str, device: torch.device, d_model: Optional[int], non_strict: bool) -> HybridModel:
    state = load_state_dict_from_checkpoint(ckpt_path)
    inferred = infer_d_model_from_state_dict(state, default=128)
    model = HybridModel(d_model=int(d_model or inferred), in_channels=1).to(device)
    model.load_state_dict(state, strict=not non_strict)
    model.eval()
    return model


@torch.no_grad()
def predict_model_outputs(model: HybridModel, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    logits_all: List[float] = []
    probs_all: List[float] = []
    labels_all: List[int] = []
    embeddings_all: List[np.ndarray] = []

    for batch in dataloader:
        seq, mask, qids, answers, _tails, is_correct = batch
        seq = seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits_t = model(seq, mask).view(-1)
        probs_t = torch.sigmoid(logits_t)
        emb_t = model.encode(seq, mask)

        logits = logits_t.detach().cpu().numpy().astype(np.float64)
        probs = probs_t.detach().cpu().numpy().astype(np.float64)
        labels = is_correct.detach().cpu().numpy().astype(int)
        qids_np = qids.detach().cpu().numpy().astype(int)
        emb = emb_t.detach().cpu().numpy().astype(np.float32)

        logits_all.extend(logits.tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.tolist())
        embeddings_all.append(emb)

        for i in range(len(qids_np)):
            rows.append({
                "qid": int(qids_np[i]),
                "pred_answer": answers[i],
                "is_correct": int(labels[i]),
                "logit": float(logits[i]),
                "neuralconf": float(probs[i]),
            })

    return {
        "rows": rows,
        "logits": np.asarray(logits_all, dtype=np.float64),
        "neural_scores": np.asarray(probs_all, dtype=np.float64),
        "labels": np.asarray(labels_all, dtype=np.int64),
        "embeddings": np.concatenate(embeddings_all, axis=0) if embeddings_all else np.zeros((0, 1), dtype=np.float32),
    }


def attach_full_trajectory_scores(
    rows: List[Dict[str, Any]],
    items: Sequence[Dict[str, Any]],
    tail_window: int,
    bottom_window: int,
    bottom_stride: int,
    bottom_ratio: float,
) -> None:
    """Attach TailConf and Bottom-10Conf using raw full confidence trajectories.

    Bottom-10Conf is deliberately independent of NeuralConf's max_len. The raw
    trace confidence sequence is read from the JSONL, then grouped with the
    default grouping length 1024. Shorter traces use their full available length
    as one effective group, matching the standard aggregation/distribution use.
    """
    for row, item in zip(rows, items):
        conf = get_confidences(item)
        row["conf_len"] = int(len(conf))
        row["tailconf"] = compute_tail_confidence(conf, window=tail_window)
        row["bottom10conf"] = compute_bottom10_group_confidence(
            conf,
            window_size=bottom_window,
            stride=bottom_stride,
            bottom_ratio=bottom_ratio,
            require_full_window=False,
        )


def compute_raw_input_dbi(
    items: Sequence[Dict[str, Any]],
    labels: Sequence[int],
    max_len: int,
    pad_val: float,
    alignment: str,
    stride: int,
    select_window_idx: int,
    window_last_idx: int,
    require_full_length: bool,
) -> Tuple[Optional[float], int]:
    raw_features: List[np.ndarray] = []
    raw_labels: List[int] = []
    for item, lab in zip(items, labels):
        conf = get_confidences(item)
        if require_full_length and len(conf) < int(max_len):
            continue
        seq, mask, eff_len = make_fixed_sequence(
            confs=conf,
            max_len=max_len,
            pad_val=pad_val,
            alignment=alignment,
            stride=stride,
            select_window_idx=select_window_idx,
            window_last_idx=window_last_idx,
            min_len_required=1,
        )
        if eff_len <= 0:
            continue
        raw_features.append(seq.astype(np.float32))
        raw_labels.append(int(lab))
    if not raw_features:
        return None, 0
    return compute_dbi(np.stack(raw_features, axis=0), raw_labels), len(raw_features)


def make_aggregation_rows(dataset_name: str, score_rows: Sequence[Dict[str, Any]], qid2gt: Dict[int, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    specs = [
        ("Majority Voting", None, 1.0, "majority_vote_acc"),
        ("Bottom-10Conf (10%)", "bottom10conf", 0.1, "bottom10_vote_acc_top0.1"),
        ("Bottom-10Conf (90%)", "bottom10conf", 0.9, "bottom10_vote_acc_top0.9"),
        ("TailConf (10%)", "tailconf", 0.1, "tailconf_vote_acc_top0.1"),
        ("TailConf (90%)", "tailconf", 0.9, "tailconf_vote_acc_top0.9"),
        ("NeuralConf", "neuralconf", 1.0, "neuralconf_vote_acc"),
    ]
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for method, score_key, top_ratio, out_key in specs:
        acc, correct, total = compute_voting_accuracy(score_rows, qid2gt, score_key=score_key, top_ratio=top_ratio)
        summary[out_key] = acc
        rows.append({
            "dataset": dataset_name,
            "method": method,
            "score_key": score_key or "uniform",
            "top_ratio": top_ratio,
            "accuracy": acc,
            "correct": correct,
            "total": total,
        })
    return rows, summary


def evaluate_standard(args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    items = ensure_is_correct(load_jsonl(args.eval_jsonl))
    qid2gt = build_qid2gt(items)
    labels_from_items = np.asarray([int(x.get("is_correct", 0)) for x in items], dtype=np.int64)

    ds = TraceDataset(
        items,
        max_len=args.max_len,
        pad_val=args.pad_val,
        tail_k_default=args.max_len,
        alignment=args.alignment,
        stride=args.stride,
        select_window_idx=args.select_window_idx,
        window_last_idx=args.window_last_idx,
        min_len_required=args.min_len_required,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = build_model_from_ckpt(args.ckpt, device, args.d_model, args.non_strict)
    pred = predict_model_outputs(model, dl, device)
    score_rows = pred["rows"]
    attach_full_trajectory_scores(
        score_rows,
        items,
        tail_window=args.max_len,
        bottom_window=args.bottom10_group_len,
        bottom_stride=args.bottom10_stride,
        bottom_ratio=args.bottom10_ratio,
    )

    labels = pred["labels"]
    neural_scores = pred["neural_scores"]
    tail_scores = np.asarray([r["tailconf"] for r in score_rows], dtype=np.float64)
    bottom_scores = np.asarray([r["bottom10conf"] for r in score_rows], dtype=np.float64)

    neural_auc = compute_auc_from_scores(neural_scores, labels)
    tail_auc = compute_auc_from_scores(tail_scores, labels)
    bottom_auc = compute_auc_from_scores(bottom_scores, labels)
    trace_acc = float(((neural_scores > 0.5).astype(int) == labels).mean()) if len(labels) else None
    brier = float(brier_score_loss(labels, neural_scores)) if len(labels) and len(np.unique(labels)) > 1 else None
    neural_dbi = compute_dbi(pred["embeddings"], labels)
    raw_dbi, raw_dbi_n = compute_raw_input_dbi(
        items=items,
        labels=labels_from_items,
        max_len=args.max_len,
        pad_val=args.pad_val,
        alignment=args.alignment,
        stride=args.stride,
        select_window_idx=args.select_window_idx,
        window_last_idx=args.window_last_idx,
        require_full_length=args.require_full_length_for_raw_dbi,
    )

    aggregation_rows, aggregation_summary = make_aggregation_rows(args.dataset_name, score_rows, qid2gt)
    prefix = f"{args.dataset_name}_{args.alignment}_maxlen{args.max_len}"
    aggregation_csv = os.path.join(args.out_dir, f"{prefix}_aggregation_table.csv")
    write_csv(
        aggregation_csv,
        aggregation_rows,
        fieldnames=["dataset", "method", "score_key", "top_ratio", "accuracy", "correct", "total"],
    )

    trace_scores_csv = ""
    if args.save_trace_scores:
        trace_scores_csv = os.path.join(args.out_dir, f"{prefix}_trace_scores.csv")
        write_csv(
            trace_scores_csv,
            score_rows,
            fieldnames=["qid", "pred_answer", "is_correct", "conf_len", "logit", "neuralconf", "tailconf", "bottom10conf"],
        )

    summary: Dict[str, Any] = {
        "dataset": args.dataset_name,
        "alignment": args.alignment,
        "max_len": int(args.max_len),
        "select_window_idx": None if args.alignment != "window" else int(args.select_window_idx),
        "window_last_idx": None if args.alignment != "window" else int(args.window_last_idx),
        "stride": None if args.alignment != "window" else int(args.stride),
        "eval_jsonl": args.eval_jsonl,
        "ckpt": args.ckpt,
        "n_traces": int(len(items)),
        "n_questions": int(len(qid2gt)),
        "raw_input_dbi": raw_dbi,
        "raw_input_dbi_n": int(raw_dbi_n),
        "neural_dbi": neural_dbi,
        "neural_auc": neural_auc,
        "tail_auc": tail_auc,
        "bottom10_auc": bottom_auc,
        "bottom10_grouping_length": int(args.bottom10_group_len),
        "bottom10_source": "raw_full_confidence_trajectory_from_eval_jsonl",
        "bottom10_short_trace_policy": "use_full_available_trace_as_one_effective_group",
        "trace_acc_at_0.5": trace_acc,
        "brier": brier,
        **aggregation_summary,
        "aggregation_csv": aggregation_csv,
        "trace_scores_csv": trace_scores_csv,
    }
    # Deliberately no trace_auc key. neural_auc is the trace-level ROC-AUC of NeuralConf.

    summary_json = os.path.join(args.out_dir, f"{prefix}_eval_summary.json")
    save_json(summary_json, summary)
    if args.metrics_csv:
        append_metrics_csv(args.metrics_csv, summary)

    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def evaluate_bottom10_sweep(args: argparse.Namespace) -> List[Dict[str, Any]]:
    os.makedirs(args.out_dir, exist_ok=True)
    items = ensure_is_correct(load_jsonl(args.eval_jsonl))
    labels = np.asarray([int(x.get("is_correct", 0)) for x in items], dtype=np.int64)
    lengths = [int(x) for x in args.grouping_lengths.split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for group_len in lengths:
        scores: List[float] = []
        labs: List[int] = []
        for item, lab in zip(items, labels):
            conf = get_confidences(item)
            if args.exclude_shorter_in_bottom10_sweep and len(conf) < group_len:
                continue
            score = compute_bottom10_group_confidence(
                conf,
                window_size=group_len,
                stride=args.bottom10_stride,
                bottom_ratio=args.bottom10_ratio,
                require_full_window=args.exclude_shorter_in_bottom10_sweep,
            )
            if np.isfinite(score):
                scores.append(float(score))
                labs.append(int(lab))
        rows.append({
            "dataset": args.dataset_name,
            "grouping_length": int(group_len),
            "bottom10_auc": compute_auc_from_scores(scores, labs),
            "n_traces": int(len(scores)),
            "source": "raw_full_confidence_trajectory_from_eval_jsonl",
        })
    out_csv = os.path.join(args.out_dir, f"{args.dataset_name}_bottom10_grouping_auc.csv")
    write_csv(out_csv, rows, fieldnames=["dataset", "grouping_length", "bottom10_auc", "n_traces", "source"])
    print(f"Saved: {out_csv}")
    for row in rows:
        print(row)
    return rows


def parse_window_ckpts(values: Optional[Sequence[str]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--window-ckpt must have form IDX=PATH, got: {value}")
        k, v = value.split("=", 1)
        out[int(k)] = v
    return out


def window_idx_to_distance(idx: int, window_size: int, stride: int, last_idx: int) -> int:
    return int((int(last_idx) - int(idx)) * int(stride) + int(window_size) / 2)


def evaluate_window_position(args: argparse.Namespace) -> List[Dict[str, Any]]:
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    items = ensure_is_correct(load_jsonl(args.eval_jsonl))
    ckpts = parse_window_ckpts(args.window_ckpt)
    rows: List[Dict[str, Any]] = []

    for win_idx in sorted(ckpts):
        ckpt_path = ckpts[win_idx]
        ds = TraceDataset(
            items,
            max_len=args.max_len,
            pad_val=args.pad_val,
            tail_k_default=args.max_len,
            alignment="window",
            stride=args.stride,
            select_window_idx=win_idx,
            window_last_idx=args.window_last_idx,
            min_len_required=args.min_len_required,
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory)
        model = build_model_from_ckpt(ckpt_path, device, args.d_model, args.non_strict)
        pred = predict_model_outputs(model, dl, device)
        labels = pred["labels"]
        rec = {
            "dataset": args.dataset_name,
            "window_idx": int(win_idx),
            "distance_to_end": window_idx_to_distance(win_idx, args.max_len, args.stride, args.window_last_idx),
            "DBI": compute_dbi(pred["embeddings"], labels),
            "AUC": compute_auc_from_scores(pred["neural_scores"], labels),
            "ckpt": ckpt_path,
        }
        rows.append(rec)
        print(rec)

    out_csv = os.path.join(args.out_dir, f"{args.dataset_name}_window_position_metrics.csv")
    write_csv(out_csv, rows, fieldnames=["dataset", "window_idx", "distance_to_end", "DBI", "AUC", "ckpt"])
    print(f"Saved: {out_csv}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NeuralConf paper metrics.")
    p.add_argument("--mode", choices=["standard", "bottom10_sweep", "window_position"], default="standard")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--ckpt", default="", help="Required for --mode standard.")
    p.add_argument("--dataset-name", default="dataset")
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--metrics-csv", default="")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--alignment", choices=["tail", "head", "window"], default="tail")
    p.add_argument("--pad-val", type=float, default=0.0)
    p.add_argument("--select-window-idx", type=int, default=9)
    p.add_argument("--window-last-idx", type=int, default=9)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--min-len-required", type=int, default=1)
    p.add_argument("--require-full-length-for-raw-dbi", action="store_true")
    p.add_argument("--bottom10-group-len", type=int, default=1024)
    p.add_argument("--bottom10-stride", type=int, default=1)
    p.add_argument("--bottom10-ratio", type=float, default=0.10)
    p.add_argument("--grouping-lengths", default="4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--exclude-shorter-in-bottom10-sweep", action="store_true")
    p.add_argument("--window-ckpt", action="append", default=[])
    p.add_argument("--save-trace-scores", action="store_true")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--non-strict", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "standard":
        if not args.ckpt:
            raise ValueError("--ckpt is required for --mode standard")
        evaluate_standard(args)
    elif args.mode == "bottom10_sweep":
        evaluate_bottom10_sweep(args)
    elif args.mode == "window_position":
        evaluate_window_position(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
