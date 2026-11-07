"""Точечный узор: сетка точек с jitter, каждая — маленькая окружность из N точек."""
from __future__ import annotations

import random

import numpy as np

from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    spacing = params.get("spacing", 0.03) * m
    radius = max(0.5, params.get("radius", 0.004) * m)
    jitter = params.get("jitter", 0.25) * spacing
    w = max(1, round(params.get("width", 0.25) * m / 512))

    nx = int(W / spacing) + 2
    ny = int(H / spacing) + 2
    n_pts = 16
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    curves: list[Curve] = []
    for i in range(nx):
        for j in range(ny):
            x = i * spacing + rng.uniform(-jitter, jitter)
            y = j * spacing + rng.uniform(-jitter, jitter)
            xs = (x + radius * np.cos(theta)).astype(np.float32)
            ys = (y + radius * np.sin(theta)).astype(np.float32)
            curves.append(Curve(xs, ys, w, closed=True))
    return curves