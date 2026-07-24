"""Sequence dataset for teacher-forced / scheduled-sampling training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from avatarface.data.augment import augment_lower
from avatarface.data.features import (
    blink_state_from_upper,
    build_current_features,
    build_history_token,
    hmd_observed_mask,
)
from avatarface.morphs.channel_groups import (
    LOWER_IDX,
    N_CURRENT,
    N_HISTORY,
    N_UPPER,
    UPPER_IDX,
)
from avatarface.morphs.arkit_schema import N_FULL


class MorphCompletionDataset(Dataset):
    def __init__(
        self,
        seq_root: Path,
        split_rows: list[dict],
        sequence_length: int = 128,
        burn_in: int = 32,
        augment: bool = False,
    ):
        self.seq_root = Path(seq_root)
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []  # (path, start)

        for row in split_rows:
            path = self.seq_root / row["path"]
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            t = int(z["full_morphs"].shape[0])
            if t < sequence_length:
                continue
            for start in range(0, t - sequence_length + 1, sequence_length // 2):
                self.samples.append((path, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, start = self.samples[idx]
        z = np.load(path, allow_pickle=True)
        end = start + self.sequence_length
        full = z["full_morphs"][start:end].astype(np.float32)
        head = z["head_features"][start:end].astype(np.float32)
        conf = z["tracking_confidence"][start:end].astype(np.float32)
        lower = full[:, LOWER_IDX]
        upper = full[:, UPPER_IDX]

        if self.augment:
            lower, conf = augment_lower(lower, conf)

        # velocities
        lower_vel = np.zeros_like(lower)
        lower_vel[1:] = lower[1:] - lower[:-1]
        full_vel = np.zeros_like(full)
        full_vel[1:] = full[1:] - full[:-1]

        blink = blink_state_from_upper(upper)
        obs = hmd_observed_mask(self.sequence_length)

        # Teacher-forced completed full: GT everywhere in history during stage 1
        # but observed_mask still reflects HMD (upper=generated flag for runtime fidelity)
        curr = np.stack(
            [
                build_current_features(lower[i], lower_vel[i], head[i], conf[i], blink[i])
                for i in range(self.sequence_length)
            ],
            axis=0,
        )
        hist = np.zeros((self.sequence_length, N_HISTORY), dtype=np.float32)
        for i in range(self.sequence_length):
            prev_full = full[i - 1] if i > 0 else full[0]
            prev_vel = full_vel[i - 1] if i > 0 else np.zeros(N_FULL, np.float32)
            prev_blink = blink[i - 1] if i > 0 else blink[0]
            prev_head = head[i - 1] if i > 0 else head[0]
            prev_conf = conf[i - 1] if i > 0 else conf[0]
            hist[i] = build_history_token(
                prev_full, prev_vel, obs[i], prev_head, prev_conf, prev_blink
            )

        # blink onset labels
        blink_l = (upper[:, 0] > 0.5).astype(np.float32)
        blink_r = (upper[:, 1] > 0.5).astype(np.float32)
        onset_l = np.zeros(self.sequence_length, np.float32)
        onset_r = np.zeros(self.sequence_length, np.float32)
        onset_l[1:] = ((blink_l[1:] > 0.5) & (blink_l[:-1] <= 0.5)).astype(np.float32)
        onset_r[1:] = ((blink_r[1:] > 0.5) & (blink_r[:-1] <= 0.5)).astype(np.float32)
        onset_bi = (onset_l * onset_r)
        amp = np.maximum(upper[:, 0], upper[:, 1])
        # crude duration: run-length normalized
        duration = np.zeros(self.sequence_length, np.float32)
        run = 0
        for i in range(self.sequence_length):
            if max(upper[i, 0], upper[i, 1]) > 0.5:
                run += 1
            else:
                run = 0
            duration[i] = min(run / 30.0, 1.0)

        burn = np.zeros(self.sequence_length, np.float32)
        burn[self.burn_in :] = 1.0

        return {
            "current": torch.from_numpy(curr),
            "history": torch.from_numpy(hist),
            "upper": torch.from_numpy(upper),
            "upper_prev": torch.from_numpy(np.vstack([upper[0:1], upper[:-1]])),
            "blink_targets": torch.from_numpy(
                np.stack([onset_l, onset_r, onset_bi, duration, amp], axis=-1)
            ),
            "supervise": torch.from_numpy(burn),
            "full": torch.from_numpy(full),
        }
