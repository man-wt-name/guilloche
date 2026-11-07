from __future__ import annotations
import math
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    sp = m * params.get("spacing", rng.uniform(0.012, 0.025))
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    nu, nv = int(1.2 * W / sp) + 2, int(1.2 * H / sp) + 2
    a = rng.uniform(0.02, 0.06) * m
    f = [rng.uniform(1.0, 4.5) for _ in range(4)]
    p = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
    t = np.linspace(-0.1, 1.1, ctx.curve_points)

    def warp(x, y):
        dx = a * np.sin(2 * np.pi * f[0] * y / m + p[0]) + 0.6 * a * np.sin(2 * np.pi * f[1] * (x + y) / m + p[1])
        dy = a * np.sin(2 * np.pi * f[2] * x / m + p[2]) + 0.6 * a * np.sin(2 * np.pi * f[3] * (x - y) / m + p[3])
        return x + dx, y + dy

    curves: list[Curve] = []
    for i in range(nu):
        X, Y = warp(np.full_like(t, -0.1 * W + i * sp), t * H)
        curves.append(Curve(X.astype(np.float32), Y.astype(np.float32), w))
    for j in range(nv):
        X, Y = warp(t * W, np.full_like(t, -0.1 * H + j * sp))
        curves.append(Curve(X.astype(np.float32), Y.astype(np.float32), w))
    return curves