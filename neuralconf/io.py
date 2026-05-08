#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint I/O."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def load_state_dict_from_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


def infer_d_model_from_state_dict(state_dict: Dict[str, Any], default: int = 128) -> int:
    key = "seq_encoder.proj.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return int(default)
