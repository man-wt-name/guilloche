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
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.25, 0.75)
    R = 1.02 * max(math.hypot(cx, cy), math.hypot(W - cx, cy),
                   math.hypot(cx, H - cy), math.hypot(W - cx, H - cy))
    nr, ns = int(R / sp) + 1, rng.randint(36, 110)
    mm = rng.randint(4, 18)
    a = m * rng.uniform(0.008, 0.03)
    q = rng.uniform(1.0, 4.0)

    t = np.linspace(0, 2 * np.pi, max(ctx.curve_points, 2400))
    curves: list[Curve] = []
    for j in range(1, nr + 1):
        r = j * sp + a * np.sin(mm * t + j * 0.35)
        curves.append(Curve((cx + r * np.cos(t)).astype(np.float32),
                            (cy + r * np.sin(t)).astype(np.float32), w, closed=True))
    s = np.linspace(0, R, max(ctx.curve_points, 1200))
    for k in range(ns):
        th0 = 2 * np.pi * k / ns
        th = th0 + 0.10 * np.sin(2 * np.pi * q * s / R + th0 * 2)
        curves.append(Curve((cx + s * np.cos(th)).astype(np.float32),
                            (cy + s * np.sin(th)).astype(np.float32), w))
    return curves