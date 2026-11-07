"""12 режимов наложения по W3C Compositing and Blending Level 1."""
from __future__ import annotations

import numpy as np

BLEND_MODES: tuple[str, ...] = (
    "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
    "add", "difference", "darken", "lighten", "color_dodge", "color_burn",
    "exclusion",
)


def blend(base: np.ndarray, top: np.ndarray, mode: str, opacity: float = 1.0) -> np.ndarray:
    """Смешать top над base. Оба — float32 в [0,1], форма (H,W,3)."""
    if mode not in BLEND_MODES:
        raise ValueError(f"unknown blend mode: {mode!r}")
    b = np.asarray(base, np.float32)
    t = np.asarray(top, np.float32)

    if mode == "normal":
        r = t
    elif mode == "multiply":
        r = b * t
    elif mode == "screen":
        r = 1.0 - (1.0 - b) * (1.0 - t)
    elif mode == "overlay":
        r = np.where(b <= 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
    elif mode == "soft_light":
        d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(np.maximum(b, 0.0)))
        r = np.where(t <= 0.5, b - (1.0 - 2.0 * t) * b * (1.0 - b),
                     b + (2.0 * t - 1.0) * (d - b))
    elif mode == "hard_light":
        r = np.where(t <= 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
    elif mode == "add":
        r = b + t
    elif mode == "difference":
        r = np.abs(b - t)
    elif mode == "darken":
        r = np.minimum(b, t)
    elif mode == "lighten":
        r = np.maximum(b, t)
    elif mode == "color_dodge":
        r = np.where(t >= 1.0, 1.0, b / np.maximum(1.0 - t, 1e-8))
    elif mode == "color_burn":
        r = np.where(t <= 0.0, 0.0, 1.0 - (1.0 - b) / np.maximum(t, 1e-8))
    elif mode == "exclusion":
        r = b + t - 2.0 * b * t
    else:  # pragma: no cover
        raise AssertionError(mode)

    r = np.clip(r, 0.0, 1.0)
    if opacity < 1.0:
        r = b * (1.0 - opacity) + r * opacity
    return r.astype(np.float32, copy=False)