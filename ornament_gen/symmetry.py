"""Симметрия финального canvas: vertical / horizontal / central / radialN.

Применяется в `compose.make` после блендинга всех слоёв, до смешивания с paper.
Используется `np.maximum` — линии из оригинала и отражения не теряются.

P2-2: radialN ограничен MAX_RADIAL_N=32.
P3-12: pad для нечётной разницы центрирован ((S-H+1)//2 вместо (S-H)//2).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

MAX_RADIAL_N = 32


def is_valid_symmetry(mode: str) -> bool:
    """Проверить строку симметрии. 'radialN' требует 2 <= N <= MAX_RADIAL_N."""
    if mode in ("none", "vertical", "horizontal", "central"):
        return True
    if mode.startswith("radial") and mode[6:].isdigit():
        n = int(mode[6:])
        return 2 <= n <= MAX_RADIAL_N
    return False


def _rotate_2d(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Поворот 2D (mode='F') или 3D (H,W,3) с билинейной интерполяцией."""
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        return np.asarray(
            img.rotate(angle_deg, resample=Image.BILINEAR, expand=False),
            np.float32,
        )
    out = np.empty(arr.shape, np.float32)
    for c in range(arr.shape[2]):
        ch = arr[..., c].astype(np.float32)
        img = Image.fromarray(ch, mode="F")
        out[..., c] = np.asarray(
            img.rotate(angle_deg, resample=Image.BILINEAR, expand=False),
            np.float32,
        )
    return out


def _mirror_vertical(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, np.flip(arr, axis=1))


def _mirror_horizontal(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, np.flip(arr, axis=0))


def _rotate_180(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, arr[::-1, ::-1])


def _radial(arr: np.ndarray, n: int) -> np.ndarray:
    if n < 2:
        return arr.copy()
    H, W = arr.shape[:2]
    S = max(H, W)

    if H == W:
        padded = arr.astype(np.float32, copy=True)
        oy = ox = 0
    else:
        extra = arr.shape[2:] if arr.ndim > 2 else ()
        padded = np.zeros((S, S, *extra), np.float32)
        # P3-12: для нечётной разницы (S-H) — округление вверх, центрируется точнее
        oy, ox = (S - H + 1) // 2, (S - W + 1) // 2
        padded[oy:oy + H, ox:ox + W] = arr

    result = padded.copy()
    for k in range(1, n):
        angle = 360.0 * k / n
        result = np.maximum(result, _rotate_2d(padded, angle))

    if H == W:
        return result
    return result[oy:oy + H, ox:ox + W]


def apply_symmetry(arr: np.ndarray, mode: str) -> np.ndarray:
    """Применить симметрию к (H,W) или (H,W,3) float32. Валидный mode обязателен."""
    if mode == "none":
        return arr
    if mode == "vertical":
        return _mirror_vertical(arr)
    if mode == "horizontal":
        return _mirror_horizontal(arr)
    if mode == "central":
        return _rotate_180(arr)
    if mode.startswith("radial") and mode[6:].isdigit():
        n = int(mode[6:])
        if n < 2:
            raise ValueError(f"radialN требует N >= 2, получено {n}")
        if n > MAX_RADIAL_N:
            raise ValueError(
                f"radialN: N={n} превышает максимум {MAX_RADIAL_N}. "
                f"Используйте меньшее N для производительности."
            )
        return _radial(arr, n)
    raise ValueError(f"unknown symmetry mode: {mode!r}")