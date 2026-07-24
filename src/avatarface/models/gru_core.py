from __future__ import annotations

import torch
import torch.nn as nn


class GRUCore(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 384, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x, hidden=None):
        # x: [B, T, input_dim]
        out, hidden = self.gru(x, hidden)
        return out, hidden

    def init_hidden(self, batch: int, device=None):
        return torch.zeros(self.num_layers, batch, self.hidden_dim, device=device)
