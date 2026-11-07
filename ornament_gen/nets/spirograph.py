from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    cx, cy = W / 2, H / 2
    R = m / 2 * 0.95
    r = R * params.get("r_ratio", rng.uniform(0.2, 0.5))
    d = r * params.get("d_ratio", rng.uniform(0.5, 1.5))
    turns = params.get("turns", rng.uniform(16, 32))
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    n = max(ctx.curve_points, int(turns * 150))
    t = np.linspace(0, 2 * np.pi * turns, n)
    X = (cx + (R - r) * np.cos(t) + d * np.cos((R - r) / max(r, 1e-6) * t)).astype(np.float32)
    Y = (cy + (R - r) * np.sin(t) - d * np.sin((R - r) / max(r, 1e-6) * t)).astype(np.float32)
    return [Curve(X, Y, w, closed=True)]