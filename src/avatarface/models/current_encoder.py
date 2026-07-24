from __future__ import annotations

import torch.nn as nn


class CurrentObservationEncoder(nn.Module):
    def __init__(self, d_in: int = 80, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)
