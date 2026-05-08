#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data utilities for token-level confidence trajectories.

The default TraceDataset behavior preserves the paper implementation:
confidence-only input, tail-aligned cropping, zero left-padding, and a binary
mask over valid confidence positions. Optional head/window modes are provided
only for the positional analyses reported in the appendix/main text.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .evaluator import math_equal


CONFIDENCE_KEYS = ("confidences", "confs", "confidence", "token_confidences", "token_level_confidences")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, records: Sequence[Dict[str, Any]], append: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_confidences(item: Dict[str, Any]) -> np.ndarray:
    for key in CONFIDENCE_KEYS:
        if key in item and item[key] is not None:
            arr = np.asarray(item[key], dtype=np.float32).reshape(-1)
            arr = arr[np.isfinite(arr)]
            return arr.astype(np.float32)
    return np.zeros((0,), dtype=np.float32)


def ensure_is_correct(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill item['is_correct'] from gold_answer and pred_answer when absent."""
    for it in items:
        if "is_correct" not in it:
            gold = it.get("gold_answer")
            pred = it.get("pred_answer")
            it["is_correct"] = 1 if (gold is not None and pred is not None and math_equal(str(pred), str(gold))) else 0
        else:
            it["is_correct"] = int(bool(it["is_correct"]))
    return items


def build_qid2gt(items: Sequence[Dict[str, Any]]) -> Dict[int, Any]:
    qid2gt: Dict[int, Any] = {}
    for it in items:
        if it.get("gold_answer") is not None:
            qid2gt.setdefault(int(it.get("qid", -1)), it["gold_answer"])
    return qid2gt


def compute_bce_class_weights(items: Sequence[Dict[str, Any]]) -> Tuple[float, float, int, int]:
    n_pos = sum(int(x.get("is_correct", 0)) == 1 for x in items)
    n_neg = len(items) - n_pos
    pos_ratio = n_pos / max(1, n_pos + n_neg)
    neg_ratio = n_neg / max(1, n_pos + n_neg)
    w_pos = 0.5 / max(1e-6, pos_ratio)
    w_neg = 0.5 / max(1e-6, neg_ratio)
    return float(w_pos), float(w_neg), int(n_pos), int(n_neg)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def tail_avg_from_confidences(confidences, window: int = 512) -> float:
    """Robust tail average used as TailConf baseline in the original code."""
    if confidences is None:
        return 0.0
    if isinstance(confidences, (int, float)):
        return float(confidences)
    try:
        arr = list(confidences)
        if len(arr) == 0:
            return 0.0
        if len(arr) > window:
            arr = arr[-window:]
        return float(sum(arr) / len(arr))
    except Exception:
        try:
            return float(confidences)
        except Exception:
            return 0.0


def make_fixed_sequence(
    confs: np.ndarray,
    max_len: int,
    pad_val: float,
    alignment: str = "tail",
    stride: int = 32,
    select_window_idx: int = 9,
    window_last_idx: int = 9,
    min_len_required: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Construct a fixed-length sequence and mask.

    alignment='tail' is the default paper construction: keep the final max_len
    values and left-pad shorter trajectories with pad_val. alignment='head'
    keeps the first max_len values and right-pads shorter trajectories. alignment
    ='window' uses fixed windows counted from the sequence end. The window index
    convention matches the original positional plotting script: with
    window_last_idx=9, idx=9 is the final window and idx=0 is farthest from the
    end.
    """
    max_len = int(max_len)
    confs = np.asarray(confs, dtype=np.float32).reshape(-1)
    length = len(confs)
    token_conf = np.full((max_len,), float(pad_val), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)

    if length == 0:
        return token_conf, mask, 0

    alignment = str(alignment).lower()
    if alignment == "tail":
        if length >= max_len:
            segment = confs[-max_len:]
        else:
            segment = confs
        eff_len = len(segment)
        if eff_len > 0:
            token_conf[-eff_len:] = segment
            mask[-eff_len:] = 1.0
        return token_conf, mask, eff_len

    if alignment == "head":
        if length >= max_len:
            segment = confs[:max_len]
        else:
            segment = confs
        eff_len = len(segment)
        if eff_len > 0:
            token_conf[:eff_len] = segment
            mask[:eff_len] = 1.0
        return token_conf, mask, eff_len

    if alignment == "window":
        offset_from_end = max(0, int(window_last_idx) - int(select_window_idx)) * int(stride)
        end = length - offset_from_end
        start = end - max_len
        start = max(0, start)
        end = max(0, min(length, end))
        segment = confs[start:end]
        if len(segment) < int(min_len_required):
            return token_conf, mask, 0
        if len(segment) > max_len:
            segment = segment[-max_len:]
        eff_len = len(segment)
        token_conf[-eff_len:] = segment
        mask[-eff_len:] = 1.0
        return token_conf, mask, eff_len

    raise ValueError(f"Unsupported alignment: {alignment}")


class TraceDataset(Dataset):
    """Token-confidence-only dataset.

    Returned sample:
      seq: FloatTensor, shape (1, max_len)
      mask: FloatTensor, shape (max_len,)
      qid: int
      pred_answer: str or None
      tail_val: float
      is_correct: int in {0,1}
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        max_len: int = 2048,
        pad_val: float = 0.0,
        tail_k_default: int = 512,
        alignment: str = "tail",
        stride: int = 32,
        select_window_idx: int = 9,
        window_last_idx: int = 9,
        min_len_required: int = 1,
        warn_on_the_fly: bool = False,  # compatibility with original scripts
    ):
        self.items = items
        self.max_len = int(max_len)
        self.pad_val = float(pad_val)
        self.tail_k_default = int(tail_k_default)
        self.alignment = str(alignment).lower()
        self.stride = int(stride)
        self.select_window_idx = int(select_window_idx)
        self.window_last_idx = int(window_last_idx)
        self.min_len_required = int(min_len_required)
        self.warn_on_the_fly = bool(warn_on_the_fly)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        confs = get_confidences(it)
        token_conf, mask, _eff_len = make_fixed_sequence(
            confs=confs,
            max_len=self.max_len,
            pad_val=self.pad_val,
            alignment=self.alignment,
            stride=self.stride,
            select_window_idx=self.select_window_idx,
            window_last_idx=self.window_last_idx,
            min_len_required=self.min_len_required,
        )
        seq = token_conf[None, :]
        tail_val = tail_avg_from_confidences(confs, window=self.tail_k_default)
        return (
            torch.from_numpy(seq).float(),
            torch.from_numpy(mask).float(),
            int(it.get("qid", -1)),
            it.get("pred_answer", None),
            float(tail_val),
            int(it.get("is_correct", 0)),
        )


def build_raw_input_matrix(
    items: Sequence[Dict[str, Any]],
    max_len: int,
    alignment: str = "tail",
    pad_val: float = 0.0,
    stride: int = 32,
    select_window_idx: int = 9,
    window_last_idx: int = 9,
    min_len_required: int = 1,
    require_full_length: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for it in items:
        confs = get_confidences(it)
        if require_full_length and len(confs) < int(max_len):
            continue
        token_conf, mask, eff_len = make_fixed_sequence(
            confs, max_len, pad_val, alignment, stride, select_window_idx, window_last_idx, min_len_required
        )
        if eff_len <= 0:
            continue
        xs.append(token_conf)
        ys.append(int(it.get("is_correct", 0)))
    if not xs:
        return np.zeros((0, int(max_len)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def collate_fn(batch):
    seqs, masks, qids, answers, tails, is_correct = zip(*batch)
    return (
        torch.stack(seqs),
        torch.stack(masks),
        torch.tensor(qids, dtype=torch.long),
        list(answers),
        torch.tensor(tails, dtype=torch.float32),
        torch.tensor(is_correct, dtype=torch.long),
    )
