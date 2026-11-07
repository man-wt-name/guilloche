"""Шум для финального изображения.

P3-1: удалены неиспользуемые `downsample` и `mask_to_alpha` (dead code;
compose.py использует собственные `_downsample_mask` и `_downsample_rgb`).
"""
from __future__ import annotations

import numpy as np


def add_noise(arr: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return arr
    return arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)