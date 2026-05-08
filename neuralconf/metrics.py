#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics, readouts, and paper-table evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, davies_bouldin_score, roc_auc_score

from .evaluator import math_equal
from .data import get_confidences
from .losses import bce_loss_from_logits_weighted, pairwise_logistic_loss
from .voting import (
    compute_bottom10_group_confidence,
    compute_tail_confidence,
    simple_majority_vote,
    weighted_majority_vote,
)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _to_float_or_none(x):
    if x is None:
        return None
    try:
        if not np.isfinite(float(x)):
            return None
        return float(x)
    except Exception:
        return None


def compute_auc_from_scores(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    valid = np.isin(labels, [0, 1]) & np.isfinite(scores)
    labels = labels[valid].astype(int)
    scores = scores[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_dbi(features: np.ndarray, labels: Sequence[int]) -> Optional[float]:
    labels = np.asarray(labels)
    valid = np.isin(labels, [0, 1])
    if valid.sum() < 2:
        return None
    y = labels[valid].astype(int)
    x = np.asarray(features, dtype=np.float32)[valid]
    if len(np.unique(y)) < 2:
        return None
    return float(davies_bouldin_score(x, y))


def eval_epoch_loss(model, dataloader, device, rank_weight=1.0, w_pos=1.0, w_neg=1.0) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            seq, mask, _qids, _answers, _tails, is_correct = batch
            seq = seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = is_correct.float().to(device)
            logits = model(seq, mask)
            bce = bce_loss_from_logits_weighted(logits, y, w_pos=w_pos, w_neg=w_neg)
            rank = pairwise_logistic_loss(logits, y) if rank_weight > 0.0 else logits.new_tensor(0.0)
            loss = bce + rank_weight * rank
            bs = seq.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
    return total_loss / max(1, total_samples)


@torch.no_grad()
def predict_trace_scores(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    logits_all, probs_all, labels_all = [], [], []
    embeddings_all = []

    for batch in dataloader:
        seq, mask, qids, answers, tails, is_correct = batch
        seq = seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits_t = model(seq, mask).view(-1)
        probs_t = torch.sigmoid(logits_t)
        if hasattr(model, "encode"):
            emb_t = model.encode(seq, mask).detach().cpu().numpy()
            embeddings_all.append(emb_t)

        logits = logits_t.detach().cpu().numpy()
        probs = probs_t.detach().cpu().numpy()
        labels = is_correct.detach().cpu().numpy().astype(int)
        tails_np = tails.detach().cpu().numpy()
        qids_np = qids.detach().cpu().numpy()

        logits_all.extend(logits.tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.tolist())
        for i in range(len(qids_np)):
            rows.append({
                "qid": int(qids_np[i]),
                "pred_answer": answers[i],
                "is_correct": int(labels[i]),
                "logit": float(logits[i]),
                "neuralconf": float(probs[i]),
                "tailconf": float(tails_np[i]),
            })

    embeddings = np.concatenate(embeddings_all, axis=0) if embeddings_all else None
    return {
        "rows": rows,
        "logits": np.asarray(logits_all, dtype=np.float64),
        "scores": np.asarray(probs_all, dtype=np.float64),
        "labels": np.asarray(labels_all, dtype=np.int64),
        "embeddings": embeddings,
    }


def _group_items_for_voting(items_or_rows: Sequence[Dict[str, Any]], score_key: Optional[str] = None):
    qid_to_answers = defaultdict(list)
    qid_to_weights = defaultdict(list)
    qid_to_labels = defaultdict(list)
    for it in items_or_rows:
        ans = it.get("pred_answer")
        if ans is None or str(ans).strip() == "":
            continue
        qid = int(it.get("qid", -1))
        qid_to_answers[qid].append(ans)
        qid_to_labels[qid].append(int(it.get("is_correct", 0)))
        if score_key is not None:
            w = it.get(score_key)
            if w is None:
                w = float("nan")
            qid_to_weights[qid].append(float(w))
    return qid_to_answers, qid_to_weights, qid_to_labels


def compute_voting_accuracy(
    rows: Sequence[Dict[str, Any]],
    qid2gt: Dict[int, Any],
    score_key: Optional[str] = None,
    top_ratio: float = 1.0,
) -> Tuple[Optional[float], int, int]:
    qid_to_answers, qid_to_weights, _qid_to_labels = _group_items_for_voting(rows, score_key=score_key)
    correct = total = 0
    for qid, answers in qid_to_answers.items():
        gt = qid2gt.get(qid)
        if gt is None:
            continue
        if score_key is None:
            pred = simple_majority_vote(answers)
        else:
            weights = qid_to_weights[qid]
            valid_pairs = [(a, w) for a, w in zip(answers, weights) if np.isfinite(w)]
            if not valid_pairs:
                continue
            pred = weighted_majority_vote([a for a, _ in valid_pairs], [w for _, w in valid_pairs], top_ratio=top_ratio)
        if pred is None:
            continue
        total += 1
        if math_equal(str(pred), str(gt)):
            correct += 1
    return ((correct / total) if total > 0 else None), correct, total


def compute_oracle_stats(rows: Sequence[Dict[str, Any]], qid2gt: Dict[int, Any]) -> Dict[str, Any]:
    qid_to_answers, _weights, qid_to_labels = _group_items_for_voting(rows, score_key=None)
    total_q = len(qid_to_answers)
    oracle_solvable = 0
    maj_wrong_oracle = 0
    rescued_by_neural = 0
    for qid, answers in qid_to_answers.items():
        labels = qid_to_labels[qid]
        if any(int(x) == 1 for x in labels):
            oracle_solvable += 1
        else:
            continue
        gt = qid2gt.get(qid)
        if gt is None:
            continue
        maj = simple_majority_vote(answers)
        maj_ok = maj is not None and math_equal(str(maj), str(gt))
        if not maj_ok:
            maj_wrong_oracle += 1
            # filled by caller if NeuralConf scores are present
    return {
        "oracle_solvable_q": oracle_solvable,
        "oracle_solvable_q_ratio": oracle_solvable / max(1, total_q),
        "maj_wrong_oracle_solvable_q": maj_wrong_oracle,
        "maj_wrong_oracle_solvable_q_ratio": maj_wrong_oracle / max(1, total_q),
        "n_questions_with_answers": total_q,
    }


def attach_handcrafted_scores(
    rows: List[Dict[str, Any]],
    items: Sequence[Dict[str, Any]],
    tail_window: int,
    bottom_window: int = 1024,
    bottom_stride: int = 1,
    bottom_ratio: float = 0.10,
) -> List[Dict[str, Any]]:
    for i, row in enumerate(rows):
        conf = get_confidences(items[i]) if i < len(items) else []
        row["conf_len"] = int(len(conf))
        row["tailconf"] = compute_tail_confidence(conf, window=tail_window)
        # Bottom-10Conf is computed from the raw full trajectory, not from the
        # tail-cropped NeuralConf input. For fixed-budget aggregation and
        # distribution plots, short traces are scored with their full available
        # trajectory as one effective group, matching the original script.
        row["bottom10conf"] = compute_bottom10_group_confidence(
            conf, window_size=bottom_window, stride=bottom_stride, bottom_ratio=bottom_ratio,
            require_full_window=False,
        )
    return rows


def evaluate_paper_metrics(
    model,
    dataloader,
    device,
    items: Sequence[Dict[str, Any]],
    qid2gt: Optional[Dict[int, Any]] = None,
    tail_window: int = 2048,
    bottom_window: int = 1024,
    bottom_stride: int = 1,
    bottom_ratio: float = 0.10,
) -> Dict[str, Any]:
    """Return the scalar metrics used throughout the paper.

    Aggregation rows match Table I: Majority Voting, Bottom-10Conf (10/90%),
    TailConf (10/90%), and NeuralConf. NeuralConf uses all traces as continuous
    weights, matching the original script's top_ratio=1.0.
    """
    pred = predict_trace_scores(model, dataloader, device)
    rows = attach_handcrafted_scores(pred["rows"], items, tail_window, bottom_window, bottom_stride, bottom_ratio)
    labels = pred["labels"]
    neural_scores = pred["scores"]
    tail_scores = np.asarray([r["tailconf"] for r in rows], dtype=np.float64)
    bottom_scores = np.asarray([r["bottom10conf"] for r in rows], dtype=np.float64)
    # Appendix Fig.6 policy: Bottom-10Conf AUC at a grouping length excludes
    # traces shorter than that grouping length. The default bottom10_auc below
    # is kept as the fixed-budget aggregation/distribution score; this explicit
    # companion avoids ambiguity.
    bottom_scores_excl = []
    bottom_labels_excl = []
    for it, lab in zip(items, labels):
        conf = get_confidences(it)
        if len(conf) < int(bottom_window):
            continue
        score = compute_bottom10_group_confidence(
            conf, window_size=bottom_window, stride=bottom_stride, bottom_ratio=bottom_ratio,
            require_full_window=True,
        )
        if np.isfinite(score):
            bottom_scores_excl.append(float(score))
            bottom_labels_excl.append(int(lab))

    cls_acc = float(((neural_scores > 0.5).astype(int) == labels).mean()) if len(labels) > 0 else None
    brier = float(brier_score_loss(labels, neural_scores)) if len(labels) > 0 and len(np.unique(labels)) > 1 else None
    neural_auc = compute_auc_from_scores(neural_scores, labels)
    tail_auc = compute_auc_from_scores(tail_scores, labels)
    bottom_auc = compute_auc_from_scores(bottom_scores, labels)
    bottom_auc_excl = compute_auc_from_scores(bottom_scores_excl, bottom_labels_excl)
    neural_dbi = compute_dbi(pred["embeddings"], labels) if pred["embeddings"] is not None else None

    aggregation_rows = []
    aggregation_summary: Dict[str, Optional[float]] = {}
    if qid2gt is not None:
        specs = [
            ("Majority Voting", None, 1.0, "majority_vote_acc"),
            ("Bottom-10Conf (10%)", "bottom10conf", 0.1, "bottom10_vote_acc_top0.1"),
            ("Bottom-10Conf (90%)", "bottom10conf", 0.9, "bottom10_vote_acc_top0.9"),
            ("TailConf (10%)", "tailconf", 0.1, "tailconf_vote_acc_top0.1"),
            ("TailConf (90%)", "tailconf", 0.9, "tailconf_vote_acc_top0.9"),
            ("NeuralConf", "neuralconf", 1.0, "neuralconf_vote_acc"),
        ]
        for method, score_key, top_ratio, out_key in specs:
            acc, correct, total = compute_voting_accuracy(rows, qid2gt, score_key=score_key, top_ratio=top_ratio)
            aggregation_summary[out_key] = acc
            aggregation_rows.append({
                "method": method,
                "score_key": score_key or "uniform",
                "top_ratio": top_ratio,
                "accuracy": acc,
                "correct": correct,
                "total": total,
            })

    return {
        "trace_acc_at_0.5": cls_acc,
        "neural_auc": neural_auc,
        "tail_auc": tail_auc,
        "bottom10_auc": bottom_auc,
        "bottom10_auc_exclude_shorter": bottom_auc_excl,
        "bottom10_grouping_length": int(bottom_window),
        "bottom10_source": "full_available_confidence_trajectory",
        "bottom10_shorter_policy": "include_shorter_for_standard_and_aggregation; exclude_shorter_for_bottom10_sweep",
        "brier": brier,
        "neural_dbi": neural_dbi,
        **aggregation_summary,
        "aggregation_rows": aggregation_rows,
        "scores": pred,
        "score_rows": rows,
    }


def bottom10_grouping_auc(
    items: Sequence[Dict[str, Any]],
    grouping_lengths: Sequence[int],
    exclude_shorter: bool = True,
    stride: int = 1,
    bottom_ratio: float = 0.10,
) -> List[Dict[str, Any]]:
    labels_all = np.asarray([int(x.get("is_correct", 0)) for x in items], dtype=np.int64)
    records = []
    for glen in grouping_lengths:
        scores, labels = [], []
        for it, lab in zip(items, labels_all):
            conf = get_confidences(it)
            if exclude_shorter and len(conf) < int(glen):
                continue
            score = compute_bottom10_group_confidence(
                conf, window_size=int(glen), stride=stride, bottom_ratio=bottom_ratio,
                require_full_window=exclude_shorter,
            )
            if np.isfinite(score):
                scores.append(float(score))
                labels.append(int(lab))
        auc = compute_auc_from_scores(scores, labels)
        records.append({
            "grouping_length": int(glen),
            "bottom10_auc": auc,
            "n_traces": len(scores),
        })
    return records
