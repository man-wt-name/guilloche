from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    cx, cy = W / 2, H / 2
    k = params.get("k", rng.randint(3, 9))
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    R = m / 2 * 0.95
    n = ctx.curve_points
    theta = np.linspace(0, 2 * np.pi, n)
    r = R * np.abs(np.cos(k * theta))
    X = (cx + r * np.cos(theta)).astype(np.float32)
    Y = (cy + r * np.sin(theta)).astype(np.float32)
    return [Curve(X, Y, w, closed=True)]