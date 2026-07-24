"""Map MediaPipe Face Landmarker outputs → canonical ARKit-52 vector."""

from __future__ import annotations

import numpy as np

from avatarface.morphs.arkit_schema import ARKIT_52, ARKIT_INDEX, N_FULL
from avatarface.morphs.channel_groups import LOWER_IDX, N_LOWER


def categories_to_vector(categories) -> np.ndarray:
    """categories: iterable of {categoryName, score} or MediaPipe Category objects."""
    out = np.zeros(N_FULL, dtype=np.float32)
    for c in categories:
        name = c.category_name if hasattr(c, "category_name") else c.get("categoryName") or c.get("category_name")
        score = c.score if hasattr(c, "score") else float(c.get("score", 0.0))
        idx = ARKIT_INDEX.get(name)
        if idx is not None:
            out[idx] = float(np.clip(score, 0.0, 1.0))
    return out


def dict_to_vector(blendshapes: dict) -> np.ndarray:
    out = np.zeros(N_FULL, dtype=np.float32)
    for name, score in blendshapes.items():
        idx = ARKIT_INDEX.get(name)
        if idx is not None:
            out[idx] = float(np.clip(score, 0.0, 1.0))
    return out


def lower_from_full(full: np.ndarray) -> np.ndarray:
    return full[..., LOWER_IDX].astype(np.float32)


def assert_mapping_complete() -> None:
    assert len(ARKIT_52) == N_FULL
    assert N_LOWER == 30
