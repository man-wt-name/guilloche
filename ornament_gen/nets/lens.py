from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    sp = m * params.get("spacing", rng.uniform(0.012, 0.025))
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    nu, nv = int(1.2 * W / sp) + 2, int(1.2 * H / sp) + 2
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.3, 0.7)
    sig = m * rng.uniform(0.25, 0.5)
    k = rng.choice([rng.uniform(0.8, 1.6), -rng.uniform(0.3, 0.55)])
    t = np.linspace(-0.1, 1.1, ctx.curve_points)

    def lens(x, y):
        dx, dy = x - cx, y - cy
        g = 1 + k * np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))
        return cx + dx * g, cy + dy * g

    curves: list[Curve] = []
    for i in range(nu):
        X, Y = lens(np.full_like(t, -0.1 * W + i * sp), t * H)
        curves.append(Curve(X.astype(np.float32), Y.astype(np.float32), w))
    for j in range(nv):
        X, Y = lens(t * W, np.full_like(t, -0.1 * H + j * sp))
        curves.append(Curve(X.astype(np.float32), Y.astype(np.float32), w))
    return curves