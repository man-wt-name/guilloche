"""Крестики: в каждой точке сетки — два перпендикулярных отрезка."""
from __future__ import annotations

import random

import numpy as np

from ..core import Curve, GenerationContext


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    spacing = params.get("spacing", 0.04) * m
    arm = params.get("arm_length", 0.012) * m
    angle = params.get("angle", rng.uniform(0.0, np.pi / 2))
    w = max(1, round(params.get("width", 0.25) * m / 512))

    nx = int(W / spacing) + 2
    ny = int(H / spacing) + 2

    ca, sa = float(np.cos(angle)), float(np.sin(angle))
    dx1, dy1 = arm * ca, arm * sa
    dx2, dy2 = -arm * sa, arm * ca

    curves: list[Curve] = []
    for i in range(nx):
        for j in range(ny):
            x0 = i * spacing
            y0 = j * spacing
            curves.append(Curve(
                np.array([x0 - dx1, x0 + dx1], np.float32),
                np.array([y0 - dy1, y0 + dy1], np.float32),
                w,
            ))
            curves.append(Curve(
                np.array([x0 - dx2, x0 + dx2], np.float32),
                np.array([y0 - dy2, y0 + dy2], np.float32),
                w,
            ))
    return curves