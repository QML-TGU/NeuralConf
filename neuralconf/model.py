#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NeuralConf model definition.

This preserves the token-only ResNet1D architecture used in the paper code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels=1, d_model=256, layers_cfg=(2, 2, 2, 2), base_channels=32):
        super().__init__()
        self.inplanes = base_channels
        self.conv1 = nn.Conv1d(in_channels, base_channels, 7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(base_channels, layers_cfg[0])
        self.layer2 = self._make_layer(base_channels * 2, layers_cfg[1])
        self.layer3 = self._make_layer(base_channels * 4, layers_cfg[2])
        self.layer4 = self._make_layer(base_channels * 8, layers_cfg[3])

        final_channels = base_channels * 8 * BasicBlock1D.expansion
        self.proj = nn.Linear(final_channels, d_model)

    def _make_layer(self, planes, blocks):
        downsample = None
        if self.inplanes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * BasicBlock1D.expansion, 1, bias=False),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )

        layers = [BasicBlock1D(self.inplanes, planes, stride=1, downsample=downsample)]
        self.inplanes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, seq, mask):
        x = seq if seq.dim() == 3 else seq.unsqueeze(1)
        m = mask if mask.dim() == 3 else mask.unsqueeze(1)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if m.shape[2] != out.shape[2]:
            if m.shape[2] % out.shape[2] == 0:
                factor = m.shape[2] // out.shape[2]
                if factor > 1:
                    m = (F.avg_pool1d(m, kernel_size=factor, stride=factor) > 0).float()
            else:
                m = F.interpolate(m, size=out.shape[2], mode="nearest")

        denom = m.sum(dim=2).clamp(min=1e-6)
        pooled = (out * m).sum(dim=2) / denom
        emb = self.proj(pooled)
        return emb


class HybridModel(nn.Module):
    def __init__(self, d_model=128, in_channels=1):
        super().__init__()
        self.seq_encoder = ResNet1DEncoder(
            in_channels=in_channels,
            d_model=d_model,
            layers_cfg=(2, 2, 2, 2),
            base_channels=32,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

    def encode(self, seq, mask):
        return self.seq_encoder(seq, mask)

    def forward(self, seq, mask):
        seq_emb = self.seq_encoder(seq, mask)
        return self.head(seq_emb).squeeze(-1)


NeuralConf = HybridModel
