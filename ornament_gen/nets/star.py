from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    cx, cy = W / 2, H / 2
    n = params.get("n", rng.randint(5, 12))
    k = params.get("k", 2)
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    R = m / 2 * 0.95
    n_pts = min(ctx.curve_points, 200)
    t = np.linspace(0.0, 1.0, n_pts)
    curves: list[Curve] = []
    for i in range(n):
        j = (i + k) % n
        th_i, th_j = 2 * np.pi * i / n, 2 * np.pi * j / n
        x0, y0 = cx + R * np.cos(th_i), cy + R * np.sin(th_i)
        x1, y1 = cx + R * np.cos(th_j), cy + R * np.sin(th_j)
        X = (x0 + (x1 - x0) * t).astype(np.float32)
        Y = (y0 + (y1 - y0) * t).astype(np.float32)
        curves.append(Curve(X, Y, w))
    return curves