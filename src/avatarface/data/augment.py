"""Morph-space augmentation (spec §19)."""

from __future__ import annotations

import numpy as np


def augment_lower(
    lower: np.ndarray,
    conf: np.ndarray,
    *,
    morph_noise_std: float = 0.01,
    channel_dropout_p: float = 0.02,
    frame_dropout_p: float = 0.01,
    amplitude_scale_range=(0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    x = lower.copy()
    c = conf.copy()
    scale = rng.uniform(*amplitude_scale_range)
    x *= scale
    x += rng.normal(0.0, morph_noise_std, size=x.shape).astype(np.float32)
    drop = rng.random(x.shape) < channel_dropout_p
    x[drop] = 0.0
    if x.ndim == 2:
        fdrop = rng.random(x.shape[0]) < frame_dropout_p
        x[fdrop] = 0.0
        c[fdrop, :] *= 0.2
    np.clip(x, 0.0, 1.0, out=x)
    return x, c
