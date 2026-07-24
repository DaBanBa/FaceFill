"""Head-pose features from MediaPipe facial transformation matrix.

Returns 12-D: 6D rotation (first two columns of R), 3D angular velocity, 3D translation velocity.
"""

from __future__ import annotations

import numpy as np


def mat4_to_rotation6d(mat: np.ndarray) -> np.ndarray:
    """mat: 4x4 row-major facial transformation matrix."""
    r = mat[:3, :3]
    return np.concatenate([r[:, 0], r[:, 1]], axis=0).astype(np.float32)


def head_features_from_matrices(
    mats: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    mats: [T, 4, 4]
    returns: [T, 12]
    """
    t = mats.shape[0]
    out = np.zeros((t, 12), dtype=np.float32)
    if t == 0:
        return out

    rot6 = np.stack([mat4_to_rotation6d(m) for m in mats], axis=0)
    trans = mats[:, :3, 3].astype(np.float32)

    out[:, 0:6] = rot6
    for i in range(1, t):
        dt = float(timestamps[i] - timestamps[i - 1])
        dt = max(dt, 1e-3)
        # angular proxy: finite difference on 6D rotation
        out[i, 6:9] = ((rot6[i, :3] - rot6[i - 1, :3]) / dt)[:3]
        out[i, 9:12] = (trans[i] - trans[i - 1]) / dt
    return out


def empty_head_features(t: int) -> np.ndarray:
    return np.zeros((t, 12), dtype=np.float32)
