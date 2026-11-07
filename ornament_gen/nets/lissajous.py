from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    cx, cy = W / 2, H / 2
    a = params.get("a", rng.randint(3, 7))
    b = params.get("b", rng.randint(3, 7))
    delta = params.get("delta", rng.uniform(0.0, np.pi))
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    A = m / 2 * 0.95
    n = ctx.curve_points
    t = np.linspace(0, 2 * np.pi, n)
    X = (cx + A * np.sin(a * t + delta)).astype(np.float32)
    Y = (cy + A * np.sin(b * t)).astype(np.float32)
    return [Curve(X, Y, w, closed=True)]