"""Blink state machine (spec §14). Overrides eyeBlink L/R after model combine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class EyeBlinkSM:
    phase: str = "IDLE"  # IDLE CLOSING CLOSED OPENING REFRACTORY
    frame: int = 0
    amplitude: float = 1.0
    closing_frames: int = 3
    closed_frames: int = 1
    opening_frames: int = 4
    refractory_frames: int = 30

    def trigger(self, amplitude: float = 1.0):
        if self.phase != "IDLE":
            return
        self.phase = "CLOSING"
        self.frame = 0
        self.amplitude = float(np.clip(amplitude, 0.5, 1.0))

    def step(self, onset_prob: float = 0.0, amplitude: float = 1.0) -> float:
        if self.phase == "IDLE" and onset_prob > 0.55:
            self.trigger(amplitude)

        if self.phase == "CLOSING":
            p = (self.frame + 1) / max(self.closing_frames, 1)
            val = smoothstep(p) * self.amplitude
            self.frame += 1
            if self.frame >= self.closing_frames:
                self.phase = "CLOSED"
                self.frame = 0
            return val

        if self.phase == "CLOSED":
            self.frame += 1
            if self.frame >= self.closed_frames:
                self.phase = "OPENING"
                self.frame = 0
            return self.amplitude

        if self.phase == "OPENING":
            p = (self.frame + 1) / max(self.opening_frames, 1)
            val = (1.0 - smoothstep(p)) * self.amplitude
            self.frame += 1
            if self.frame >= self.opening_frames:
                self.phase = "REFRACTORY"
                self.frame = 0
            return val

        if self.phase == "REFRACTORY":
            self.frame += 1
            if self.frame >= self.refractory_frames:
                self.phase = "IDLE"
                self.frame = 0
            return 0.0

        return 0.0


@dataclass
class BlinkController:
    left: EyeBlinkSM = field(default_factory=EyeBlinkSM)
    right: EyeBlinkSM = field(default_factory=EyeBlinkSM)
    frames_since: int = 999

    def step(self, onset_logits: np.ndarray, duration: float, amplitude: float) -> tuple[float, float]:
        # onset_logits: [3] left, right, bilateral — sigmoid outside
        probs = 1.0 / (1.0 + np.exp(-onset_logits))
        bi = probs[2]
        l_on = max(probs[0], bi)
        r_on = max(probs[1], bi)
        # scale blink timing from predicted duration (normalized ~0-1)
        total = int(np.clip(4 + duration * 12, 4, 16))
        close_n = max(2, total // 3)
        open_n = max(2, total - close_n - 1)
        self.left.closing_frames = close_n
        self.right.closing_frames = close_n
        self.left.opening_frames = open_n
        self.right.opening_frames = open_n
        ref = int(np.clip(15 + duration * 60, 15, 60))
        self.left.refractory_frames = ref
        self.right.refractory_frames = ref
        bl = self.left.step(l_on, amplitude)
        br = self.right.step(r_on, amplitude)
        if bl > 0.3 or br > 0.3:
            self.frames_since = 0
        else:
            self.frames_since += 1
        return float(bl), float(br)

    def apply(self, upper: np.ndarray, onset_logits, duration, amplitude) -> np.ndarray:
        out = upper.copy()
        bl, br = self.step(np.asarray(onset_logits), float(duration), float(amplitude))
        # override blink channels (indices 0,1 in UPPER_FACE_CHANNELS)
        out[0] = max(out[0], bl)
        out[1] = max(out[1], br)
        return out
