from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    cx, cy = W / 2, H / 2
    turns = params.get("turns", rng.uniform(8, 20))
    kind = params.get("kind", "archimedean")
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))
    R_max = m / 2 * 0.95
    n = max(ctx.curve_points, int(turns * 200))
    theta = np.linspace(0, 2 * np.pi * turns, n)
    if kind == "archimedean":
        r = R_max * theta / (2 * np.pi * turns)
    else:  # log
        b = params.get("b", 0.15)
        r = R_max * (np.exp(b * theta / (2 * np.pi)) - 1.0) / (np.exp(b * turns) - 1.0 + 1e-8)
    X = (cx + r * np.cos(theta)).astype(np.float32)
    Y = (cy + r * np.sin(theta)).astype(np.float32)
    return [Curve(X, Y, w)]