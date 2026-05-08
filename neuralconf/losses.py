#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training losses for NeuralConf."""

import torch
import torch.nn.functional as F


def bce_loss_from_logits_weighted(logits, labels, w_pos=1.0, w_neg=1.0):
    weights = torch.where(labels > 0.5, torch.full_like(labels, w_pos), torch.full_like(labels, w_neg))
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction="mean")


def pairwise_logistic_loss(logits, labels):
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(0.0, device=logits.device)
    diff = pos[:, None] - neg[None, :]
    return F.softplus(-diff).mean()
