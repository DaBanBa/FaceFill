from __future__ import annotations

import torch
import torch.nn as nn


class AbsoluteUpperHead(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.SiLU(),
            nn.Linear(256, d_out),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.net(h)


class ResidualUpperHead(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 11, max_delta: float = 0.15):
        super().__init__()
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.SiLU(),
            nn.Linear(256, d_out),
            nn.Tanh(),
        )

    def forward(self, h):
        return self.max_delta * self.net(h)


class UncertaintyHead(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.SiLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, h):
        return torch.clamp(self.net(h), -6.0, 2.0)


class BlinkHead(nn.Module):
    def __init__(self, d_in: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
        )

    def forward(self, h):
        raw = self.net(h)
        onset = raw[..., 0:3]  # logits
        duration = 0.10 + 0.30 * torch.sigmoid(raw[..., 3])
        amplitude = 0.5 + 0.5 * torch.sigmoid(raw[..., 4])
        return onset, duration, amplitude
