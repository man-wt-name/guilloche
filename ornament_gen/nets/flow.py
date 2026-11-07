from __future__ import annotations
import random
import numpy as np
from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    n_waves = params.get("n_waves", 6)
    step = params.get("step", m * 0.005)
    n_steps = params.get("n_steps", 400)
    n_seeds = params.get("n_seeds", 30)
    w = max(1, round(params.get("width", rng.uniform(0.25, 0.6)) * m / 512))

    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    angle = np.zeros_like(xx)
    for _ in range(n_waves):
        fx = rng.uniform(0.5, 3.0) / m * 2 * np.pi
        fy = rng.uniform(0.5, 3.0) / m * 2 * np.pi
        ph = rng.uniform(0.0, 2 * np.pi)
        angle += np.sin(fx * xx + fy * yy + ph)
    angle *= 2.0

    def sample(x: float, y: float) -> float:
        xi = min(max(int(x / 4), 0), angle.shape[1] - 1)
        yi = min(max(int(y / 4), 0), angle.shape[0] - 1)
        return float(angle[yi, xi])

    curves: list[Curve] = []
    for _ in range(n_seeds):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        xs, ys = [x], [y]
        for _ in range(n_steps):
            a = sample(x, y)
            x += step * float(np.cos(a))
            y += step * float(np.sin(a))
            if not (0.0 <= x < W and 0.0 <= y < H):
                break
            xs.append(x)
            ys.append(y)
        if len(xs) > 5:
            curves.append(Curve(np.array(xs, np.float32), np.array(ys, np.float32), w))
    return curves