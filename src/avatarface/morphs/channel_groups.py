"""Channel group partitions from the AvatarFace morph completion spec."""

from __future__ import annotations

from avatarface.morphs.arkit_schema import ARKIT_52, ARKIT_INDEX, N_FULL

# Spec §3.1 — observable lower face (30). Order is training-locked.
LOWER_FACE_CHANNELS: list[str] = [
    "jawOpen",
    "jawForward",
    "jawLeft",
    "jawRight",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "cheekPuff",
    "noseSneerLeft",
    "noseSneerRight",
]

# Spec §3.2 — hidden upper-face expression (11).
UPPER_FACE_CHANNELS: list[str] = [
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
]

# Spec §3.3 — gaze (8); not predicted from mouth.
GAZE_CHANNELS: list[str] = [
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
]

# Remaining MediaPipe channels (complete the 52).
EXTRA_CHANNELS: list[str] = [
    "cheekSquintLeft",
    "cheekSquintRight",
    "tongueOut",
]

N_LOWER = len(LOWER_FACE_CHANNELS)  # 30
N_UPPER = len(UPPER_FACE_CHANNELS)  # 11
N_GAZE = len(GAZE_CHANNELS)  # 8
N_EXTRA = len(EXTRA_CHANNELS)  # 3

LOWER_IDX = [ARKIT_INDEX[n] for n in LOWER_FACE_CHANNELS]
UPPER_IDX = [ARKIT_INDEX[n] for n in UPPER_FACE_CHANNELS]
GAZE_IDX = [ARKIT_INDEX[n] for n in GAZE_CHANNELS]
EXTRA_IDX = [ARKIT_INDEX[n] for n in EXTRA_CHANNELS]

# Current-frame feature dims (first production, no acceleration): §40
# 30 lower + 30 vel + 12 head + 3 conf + 5 blink = 80
N_HEAD = 12
N_CONF = 3
N_BLINK_STATE = 5
N_CURRENT = N_LOWER + N_LOWER + N_HEAD + N_CONF + N_BLINK_STATE  # 80

# History token: 52 + 52 + 52 + 12 + 3 + 5 = 176
N_HISTORY = N_FULL + N_FULL + N_FULL + N_HEAD + N_CONF + N_BLINK_STATE  # 176


def validate_partition() -> None:
    all_named = LOWER_FACE_CHANNELS + UPPER_FACE_CHANNELS + GAZE_CHANNELS + EXTRA_CHANNELS
    if len(all_named) != N_FULL:
        raise ValueError(f"partition size {len(all_named)} != {N_FULL}")
    if set(all_named) != set(ARKIT_52):
        missing = set(ARKIT_52) - set(all_named)
        extra = set(all_named) - set(ARKIT_52)
        raise ValueError(f"partition mismatch missing={missing} extra={extra}")
    if len(set(all_named)) != N_FULL:
        raise ValueError("duplicate channel in partition")


validate_partition()
