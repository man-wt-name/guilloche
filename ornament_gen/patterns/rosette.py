"""Розетка: N окружностей с общим/слегка смещённым центром — банкнотный элемент."""
from __future__ import annotations

import random

import numpy as np

from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    cx, cy = W / 2.0, H / 2.0
    n_circles = params.get("n_circles", 12)
    r_min, r_max = params.get("radius_range", (0.30, 0.45))
    offset_rel = params.get("offset", 0.10)
    jitter_rel = params.get("center_jitter", 0.01)
    w = max(1, round(params.get("width", 0.3) * m / 512))

    R = m / 2.0
    offset = offset_rel * R
    jitter = jitter_rel * R
    n_pts = max(64, ctx.curve_points // 4)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    curves: list[Curve] = []
    for i in range(n_circles):
        angle = 2.0 * np.pi * i / max(n_circles, 1)
        c_x = cx + offset * float(np.cos(angle)) + rng.uniform(-jitter, jitter)
        c_y = cy + offset * float(np.sin(angle)) + rng.uniform(-jitter, jitter)
        r = rng.uniform(r_min, r_max) * R
        xs = (c_x + r * np.cos(theta)).astype(np.float32)
        ys = (c_y + r * np.sin(theta)).astype(np.float32)
        curves.append(Curve(xs, ys, w, closed=True))
    return curves