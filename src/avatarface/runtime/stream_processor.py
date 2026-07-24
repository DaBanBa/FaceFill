"""Streaming runtime state (spec §25–26)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from avatarface.data.features import build_current_features, build_history_token, default_blink_state, hmd_observed_mask
from avatarface.morphs.arkit_schema import N_FULL
from avatarface.morphs.channel_groups import LOWER_IDX, N_UPPER, UPPER_IDX
from avatarface.morphs.mediapipe_mapping import dict_to_vector, lower_from_full
from avatarface.runtime.blink_controller import BlinkController


@dataclass
class AvatarFaceState:
    previous_lower: np.ndarray = field(default_factory=lambda: np.zeros(30, np.float32))
    previous_lower_velocity: np.ndarray = field(default_factory=lambda: np.zeros(30, np.float32))
    previous_full_morphs: np.ndarray = field(default_factory=lambda: np.zeros(N_FULL, np.float32))
    previous_full_velocity: np.ndarray = field(default_factory=lambda: np.zeros(N_FULL, np.float32))
    previous_upper: np.ndarray = field(default_factory=lambda: np.zeros(N_UPPER, np.float32))
    temporal_hidden: object | None = None
    blink: BlinkController = field(default_factory=BlinkController)
    frames_since_last_blink: int = 999
    last_timestamp: float = 0.0
    initialized: bool = False


def process_frame(
    model,
    mediapipe_morphs: dict | np.ndarray,
    head_features: np.ndarray,
    tracking_confidence: np.ndarray,
    timestamp: float,
    state: AvatarFaceState,
    device: str = "cpu",
) -> tuple[np.ndarray, AvatarFaceState]:
    """
    Returns completed full ARKit-52 vector and updated state.
    """
    model.eval()
    if isinstance(mediapipe_morphs, dict):
        full_obs = dict_to_vector(mediapipe_morphs)
    else:
        full_obs = np.asarray(mediapipe_morphs, dtype=np.float32)

    lower = lower_from_full(full_obs)
    dt = max(timestamp - state.last_timestamp, 1e-3) if state.initialized else 1.0 / 30.0
    lower_vel = (lower - state.previous_lower) / dt if state.initialized else np.zeros_like(lower)

    conf = np.asarray(tracking_confidence, dtype=np.float32).reshape(3)
    head = np.asarray(head_features, dtype=np.float32).reshape(12)
    blink_feat = default_blink_state(state.frames_since_last_blink)

    # low confidence → damp motion (spec §27)
    alpha = 0.8
    delta_scale = 1.0
    if conf[0] < 0.3:
        delta_scale = 0.1
        alpha = 0.98

    curr = build_current_features(lower, lower_vel, head, conf, blink_feat)
    obs_mask = hmd_observed_mask(1)[0]
    hist = build_history_token(
        state.previous_full_morphs,
        state.previous_full_velocity,
        obs_mask,
        head if state.initialized else head,
        conf,
        blink_feat,
    )

    with torch.no_grad():
        cur_t = torch.from_numpy(curr).unsqueeze(0).to(device)
        hist_t = torch.from_numpy(hist).unsqueeze(0).to(device)
        u_prev_t = torch.from_numpy(state.previous_upper).unsqueeze(0).to(device)
        hidden = state.temporal_hidden
        out = model.step(cur_t, hist_t, u_prev_t, hidden=hidden)
        state.temporal_hidden = out["hidden"]
        u_abs = out["u_abs"].squeeze(0).cpu().numpy()
        u_delta = out["u_delta"].squeeze(0).cpu().numpy() * delta_scale
        u_hat = alpha * (state.previous_upper + u_delta) + (1.0 - alpha) * u_abs
        u_hat = np.clip(u_hat, 0.0, 1.0)

        onset = out["blink_onset_logits"].squeeze(0).cpu().numpy()
        dur = float(out["blink_duration"].squeeze(0).cpu().numpy())
        amp = float(out["blink_amplitude"].squeeze(0).cpu().numpy())
        u_hat = state.blink.apply(u_hat, onset, dur, amp)

    completed = full_obs.copy()
    completed[UPPER_IDX] = u_hat
    # gaze left as observed or zero — do not invent from mouth
    # lower stays from MediaPipe
    completed[LOWER_IDX] = lower

    full_vel = (completed - state.previous_full_morphs) / dt if state.initialized else np.zeros(N_FULL, np.float32)
    state.previous_lower = lower
    state.previous_lower_velocity = lower_vel
    state.previous_full_velocity = full_vel
    state.previous_full_morphs = completed
    state.previous_upper = u_hat
    state.frames_since_last_blink = state.blink.frames_since
    state.last_timestamp = timestamp
    state.initialized = True
    return completed, state
