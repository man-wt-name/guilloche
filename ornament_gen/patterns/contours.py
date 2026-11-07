"""Топографические контуры: тонкие линии уровня из гладкого поля (marching squares).

P0-1 fix: работа в уменьшенном разрешении (scale = max(1, min(W,H) // 1024)).
Для A4 SS=3 (7440) → scale=7, поле 1063×1504 вместо 7440×10524. Память 2.7 GB → 30 MB.
Координаты масштабируются обратно при создании Curve.
"""
from __future__ import annotations

import random

import numpy as np

from ..core import Curve, GenerationContext


def _make_field(W: int, H: int, rng: random.Random, n_waves: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field = np.zeros((H, W), np.float32)
    base = min(W, H)
    for _ in range(n_waves):
        fx = rng.uniform(1.0, 5.0) * 2.0 * np.pi / base
        fy = rng.uniform(1.0, 5.0) * 2.0 * np.pi / base
        ph = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        field += amp * np.sin(fx * xx + fy * yy + ph)
    return field


def _fft_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.5:
        return a
    pad = int(3 * sigma) + 1
    p = np.pad(a, pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    k = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(p) * k, s=p.shape)
    return out[pad:-pad, pad:-pad].astype(np.float32)


_CASE_EDGES = {
    1: [("left", "bottom")],
    2: [("right", "bottom")],
    3: [("right", "left")],
    4: [("top", "right")],
    5: [("top", "left"), ("right", "bottom")],
    6: [("top", "bottom")],
    7: [("top", "left")],
    8: [("top", "left")],
    9: [("top", "bottom")],
    10: [("top", "right"), ("left", "bottom")],
    11: [("top", "right")],
    12: [("left", "right")],
    13: [("right", "bottom")],
    14: [("left", "bottom")],
}


def _marching_squares_segments(field: np.ndarray, level: float) -> list[tuple[np.ndarray, np.ndarray]]:
    H, W = field.shape
    above = field >= level

    tl = above[:-1, :-1].astype(np.uint8)
    tr = above[:-1, 1:].astype(np.uint8)
    br = above[1:, 1:].astype(np.uint8)
    bl = above[1:, :-1].astype(np.uint8)

    case = (tl << 3) | (tr << 2) | (br << 1) | bl
    active = (case != 0) & (case != 15)
    if not active.any():
        return []

    ys, xs = np.nonzero(active)
    cases = case[ys, xs]

    f_tl = field[ys, xs]
    f_tr = field[ys, xs + 1]
    f_br = field[ys + 1, xs + 1]
    f_bl = field[ys + 1, xs]

    def interp(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        t = (level - f1) / (f2 - f1 + 1e-12)
        return np.clip(t, 0.0, 1.0).astype(np.float32)

    t_top = interp(f_tl, f_tr)
    t_right = interp(f_tr, f_br)
    t_bot = interp(f_bl, f_br)
    t_left = interp(f_tl, f_bl)

    x_top = (xs.astype(np.float32) + t_top)
    y_top = ys.astype(np.float32)
    x_right = (xs + 1).astype(np.float32)
    y_right = (ys.astype(np.float32) + t_right)
    x_bot = (xs.astype(np.float32) + t_bot)
    y_bot = (ys + 1).astype(np.float32)
    x_left = xs.astype(np.float32)
    y_left = (ys.astype(np.float32) + t_left)

    edge_xy = {
        "top": (x_top, y_top),
        "right": (x_right, y_right),
        "bottom": (x_bot, y_bot),
        "left": (x_left, y_left),
    }

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for c, pairs in _CASE_EDGES.items():
        sel = cases == c
        if not sel.any():
            continue
        for e1, e2 in pairs:
            x1 = edge_xy[e1][0][sel]
            y1 = edge_xy[e1][1][sel]
            x2 = edge_xy[e2][0][sel]
            y2 = edge_xy[e2][1][sel]
            n = x1.shape[0]
            for k in range(n):
                segments.append((
                    np.array([x1[k], x2[k]], np.float32),
                    np.array([y1[k], y2[k]], np.float32),
                ))
    return segments


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    m = min(W_ss, H_ss)
    n_waves = params.get("n_waves", 5)
    smooth_ss = params.get("smooth", 0.02) * m
    n_levels = params.get("n_levels", 6)
    w = max(1, round(params.get("width", 0.25) * m / 512))

    # P0-1: работа в уменьшенном разрешении для экономии памяти.
    # scale = max(1, min(W,H) // 1024): для A4 SS=3 (m=7440) → scale=7,
    # для 1110×756 SS=2 (m=1512) → scale=1 (bit-for-bit для маленьких).
    scale = max(1, m // 1024)
    W_work = (W_ss + scale - 1) // scale
    H_work = (H_ss + scale - 1) // scale

    field = _make_field(W_work, H_work, rng, n_waves)
    field = _fft_smooth(field, smooth_ss / scale)
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-6:
        return []
    level_min = params.get("level_min", 0.15)
    level_max = params.get("level_max", 0.85)
    levels = np.linspace(lo + (hi - lo) * level_min, lo + (hi - lo) * level_max, n_levels)

    curves: list[Curve] = []
    for lv in levels:
        for xs_seg, ys_seg in _marching_squares_segments(field, float(lv)):
            # Масштабируем координаты обратно в SS-разрешение
            curves.append(Curve(
                (xs_seg * scale).astype(np.float32),
                (ys_seg * scale).astype(np.float32),
                w,
            ))
    return curves