"""Scheduled sampling schedule (spec §20.2) — used in Stage 2 training."""

from __future__ import annotations


def teacher_forcing_prob(epoch: int) -> float:
    table = {1: 0.95, 3: 0.80, 5: 0.60, 8: 0.40}
    if epoch >= 20:
        return 0.10
    # stepwise interpolate
    keys = sorted(table)
    for k in reversed(keys):
        if epoch >= k:
            return table[k]
    return 0.95
