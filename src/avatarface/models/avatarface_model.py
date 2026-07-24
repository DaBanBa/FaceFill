"""AvatarFace GRU morph completion model (spec §40)."""

from __future__ import annotations

import torch
import torch.nn as nn

from avatarface.models.current_encoder import CurrentObservationEncoder
from avatarface.models.gru_core import GRUCore
from avatarface.models.history_encoder import HistoricalStateEncoder
from avatarface.models.output_heads import (
    AbsoluteUpperHead,
    BlinkHead,
    ResidualUpperHead,
    UncertaintyHead,
)
from avatarface.morphs.channel_groups import N_CURRENT, N_HISTORY, N_UPPER


class AvatarFaceModel(nn.Module):
    def __init__(
        self,
        current_dim: int = N_CURRENT,
        history_dim: int = N_HISTORY,
        enc_dim: int = 256,
        temporal_hidden: int = 384,
        temporal_layers: int = 3,
        dropout: float = 0.1,
        max_delta: float = 0.15,
        alpha: float = 0.8,
        upper_dim: int = N_UPPER,
    ):
        super().__init__()
        self.alpha = alpha
        self.current_enc = CurrentObservationEncoder(current_dim, enc_dim)
        self.history_enc = HistoricalStateEncoder(history_dim, enc_dim)
        self.fuse = nn.Linear(enc_dim * 2, temporal_hidden)
        self.core = GRUCore(temporal_hidden, temporal_hidden, temporal_layers, dropout)
        self.abs_head = AbsoluteUpperHead(temporal_hidden, upper_dim)
        self.res_head = ResidualUpperHead(temporal_hidden, upper_dim, max_delta=max_delta)
        self.unc_head = UncertaintyHead(temporal_hidden, upper_dim)
        self.blink_head = BlinkHead(temporal_hidden)

    def combine(self, u_prev, u_abs, u_delta, alpha: float | None = None):
        a = self.alpha if alpha is None else alpha
        u = a * (u_prev + u_delta) + (1.0 - a) * u_abs
        return torch.clamp(u, 0.0, 1.0)

    def forward(self, current, history, u_prev, hidden=None):
        """
        current: [B, T, 80]
        history: [B, T, 176]
        u_prev:  [B, T, 11] previous upper (teacher or generated)
        """
        b, t, _ = current.shape
        cur_e = self.current_enc(current)
        hist_e = self.history_enc(history)
        fused = self.fuse(torch.cat([cur_e, hist_e], dim=-1))
        out, hidden = self.core(fused, hidden)

        u_abs = self.abs_head(out)
        u_delta = self.res_head(out)
        log_var = self.unc_head(out)
        onset, duration, amplitude = self.blink_head(out)
        u_hat = self.combine(u_prev, u_abs, u_delta)

        return {
            "u_abs": u_abs,
            "u_delta": u_delta,
            "u_hat": u_hat,
            "log_var": log_var,
            "blink_onset_logits": onset,
            "blink_duration": duration,
            "blink_amplitude": amplitude,
            "hidden": hidden,
        }

    def step(self, current_t, history_t, u_prev_t, hidden=None):
        """Single-frame streaming step. inputs: [B, D]."""
        out = self.forward(
            current_t.unsqueeze(1),
            history_t.unsqueeze(1),
            u_prev_t.unsqueeze(1),
            hidden=hidden,
        )
        return {k: (v.squeeze(1) if torch.is_tensor(v) and v.dim() > 1 and v.shape[1] == 1 else v)
                if k != "hidden" else v
                for k, v in out.items()}
