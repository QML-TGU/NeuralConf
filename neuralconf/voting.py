#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Voting and hand-crafted confidence baselines."""

import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


def simple_majority_vote(answers: List[str]) -> Optional[str]:
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def weighted_majority_vote(
    answers: List[str],
    weights: List[float],
    top_ratio: float = 0.9,
) -> Optional[str]:
    """Weighted answer aggregation used in the experiments.

    top_ratio=1.0 keeps all traces, which is the NeuralConf setting in the
    original script. top_ratio=0.9 reproduces the TailConf weighted-vote
    setting used there.
    """
    if not answers or not weights or len(answers) != len(weights):
        return None

    indexed = list(zip(answers, weights))
    indexed.sort(key=lambda x: x[1], reverse=True)

    cutoff = int(np.ceil(len(indexed) * float(top_ratio)))
    selected = indexed[: max(1, cutoff)]

    agg: Dict[str, float] = {}
    for answer, weight in selected:
        if answer is None:
            continue
        answer_str = str(answer)
        agg[answer_str] = agg.get(answer_str, 0.0) + float(weight)

    if not agg:
        return None
    return max(agg.keys(), key=lambda k: agg[k])


def to_valid_confidence_array(conf_seq) -> np.ndarray:
    if conf_seq is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(conf_seq, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr.astype(np.float32)


def compute_tail_confidence(conf_seq, window: int = 2048) -> float:
    arr = to_valid_confidence_array(conf_seq)
    if len(arr) == 0:
        return 0.0
    if len(arr) > window:
        arr = arr[-window:]
    return float(np.mean(arr))


def compute_bottom10_group_confidence(
    conf_seq,
    window_size: int = 1024,
    stride: int = 1,
    bottom_ratio: float = 0.10,
    require_full_window: bool = False,
) -> float:
    """Compute Bottom-10Conf from the full available confidence trajectory.

    ``window_size`` is the grouping length. This function does not crop the
    trace to NeuralConf's ``max_len``; callers pass the raw per-trace confidence
    sequence.

    ``require_full_window=False`` reproduces the fixed-budget aggregation and
    distribution setting: traces shorter than ``window_size`` are scored using
    their full available trajectory as one effective group.

    ``require_full_window=True`` is used for the appendix grouping-length AUC
    sweep, where traces shorter than the chosen grouping length are excluded.
    """
    conf_seq = to_valid_confidence_array(conf_seq)
    length = len(conf_seq)
    if length == 0:
        return float("nan")

    if require_full_window and length < int(window_size):
        return float("nan")

    eff_w = min(int(window_size), length)
    if eff_w <= 0:
        return float("nan")

    csum = np.concatenate([[0.0], np.cumsum(conf_seq, dtype=np.float64)])
    group_means = (csum[eff_w:] - csum[:-eff_w]) / float(eff_w)
    if stride > 1:
        group_means = group_means[::stride]

    if len(group_means) == 0:
        return float(np.mean(conf_seq))

    count = max(1, int(math.ceil(float(bottom_ratio) * len(group_means))))
    bottom_vals = np.partition(group_means, count - 1)[:count]
    return float(np.mean(bottom_vals))
