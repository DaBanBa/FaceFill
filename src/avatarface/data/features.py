"""Build current-frame and history tokens for the completion model."""

from __future__ import annotations

import numpy as np

from avatarface.morphs.arkit_schema import N_FULL
from avatarface.morphs.channel_groups import (
    LOWER_IDX,
    N_BLINK_STATE,
    N_CONF,
    N_CURRENT,
    N_HEAD,
    N_HISTORY,
    N_LOWER,
    UPPER_IDX,
)


def default_blink_state(frames_since: float = 30.0) -> np.ndarray:
    # leftPhase, rightPhase, leftAmp, rightAmp, framesSinceLastBlink (normalized /30)
    return np.array([0.0, 0.0, 0.0, 0.0, min(frames_since / 30.0, 5.0)], dtype=np.float32)


def blink_state_from_upper(upper_hist: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Derive simple blink-state features from GT blink channels for training."""
    # upper: [..., 11] with blink L/R at 0,1
    t = upper_hist.shape[0]
    out = np.zeros((t, N_BLINK_STATE), dtype=np.float32)
    blink = upper_hist[:, 0:2].max(axis=-1)
    since = 999
    for i in range(t):
        if blink[i] > 0.5:
            since = 0
            out[i, 0] = blink[i]
            out[i, 1] = blink[i]
            out[i, 2] = blink[i]
            out[i, 3] = blink[i]
        else:
            since += 1
        out[i, 4] = min(since / fps, 5.0)
    return out


def build_current_features(
    lower: np.ndarray,
    lower_vel: np.ndarray,
    head: np.ndarray,
    conf: np.ndarray,
    blink: np.ndarray,
) -> np.ndarray:
    """lower/vel [30], head [12], conf [3], blink [5] → [80]."""
    return np.concatenate(
        [lower, lower_vel, head, conf, blink], axis=-1
    ).astype(np.float32)


def build_history_token(
    full: np.ndarray,
    full_vel: np.ndarray,
    obs_mask: np.ndarray,
    head: np.ndarray,
    conf: np.ndarray,
    blink: np.ndarray,
) -> np.ndarray:
    """Each arg last-dim matching → [176]."""
    return np.concatenate(
        [full, full_vel, obs_mask.astype(np.float32), head, conf, blink],
        axis=-1,
    ).astype(np.float32)


def hmd_observed_mask(t: int) -> np.ndarray:
    """Simulate headset occlusion: lower (+extra tongue optionally) observed; upper/gaze generated."""
    mask = np.zeros((t, N_FULL), dtype=np.float32)
    mask[:, LOWER_IDX] = 1.0
    # extras: tongueOut observed if lower visible; cheekSquint often occluded → 0
    from avatarface.morphs.arkit_schema import ARKIT_INDEX

    mask[:, ARKIT_INDEX["tongueOut"]] = 1.0
    return mask


def assert_feature_dims():
    assert N_CURRENT == 80
    assert N_HISTORY == 176
