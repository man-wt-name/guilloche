"""Границы ячеек Вороного как линии. Точки обрезаются по холсту (Cohen–Sutherland).

P2-6 fix: убран мусорный первый column_stack (тратил n_cells RNG-вызовов впустую).
Порядок RNG-вызовов изменён (2*n_cells вместо 3*n_cells) — bit-for-bit для spec
с voronoi ломается, но детерминизм сохранён (тот же seed → тот же результат).
"""
from __future__ import annotations

import random

import numpy as np
from scipy.spatial import Voronoi

from ..core import Curve, GenerationContext


def _clip_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float] | None:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    def code(x: float, y: float) -> int:
        c = 0
        if x < xmin:
            c |= 1
        elif x > xmax:
            c |= 2
        if y < ymin:
            c |= 4
        elif y > ymax:
            c |= 8
        return c

    c1 = code(x1, y1)
    c2 = code(x2, y2)

    for _ in range(8):
        if (c1 | c2) == 0:
            return x1, y1, x2, y2
        if (c1 & c2) != 0:
            return None
        c = c1 if c1 != 0 else c2
        if c & 1:
            x = xmin
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1 + 1e-12)
        elif c & 2:
            x = xmax
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1 + 1e-12)
        elif c & 4:
            y = ymin
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1 + 1e-12)
        else:
            y = ymax
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1 + 1e-12)
        if c == c1:
            x1, y1 = x, y
            c1 = code(x1, y1)
        else:
            x2, y2 = x, y
            c2 = code(x2, y2)
    return None


def generate(ctx: GenerationContext, rng: random.Random, params: dict) -> list[Curve]:
    W, H = ctx.ss_width, ctx.ss_height
    m = min(W, H)
    n_cells = params.get("n_cells", 60)
    w = max(1, round(params.get("width", 0.25) * m / 512))

    # P2-6: только 2*n_cells RNG-вызовов (было 3*n_cells — мусорный column_stack)
    if n_cells < 3:
        return []
    xs_arr = np.array([rng.uniform(0.0, float(W)) for _ in range(n_cells)], np.float64)
    ys_arr = np.array([rng.uniform(0.0, float(H)) for _ in range(n_cells)], np.float64)
    pts = np.column_stack([xs_arr, ys_arr])
    if len(pts) < 3:
        return []

    try:
        vor = Voronoi(pts)
    except Exception:
        return []

    curves: list[Curve] = []
    for v1, v2 in vor.ridge_vertices:
        if v1 < 0 or v2 < 0:
            continue
        p1 = vor.vertices[v1]
        p2 = vor.vertices[v2]
        clipped = _clip_segment((p1[0], p1[1]), (p2[0], p2[1]), 0.0, 0.0, float(W), float(H))
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        curves.append(Curve(
            np.array([x1, x2], np.float32),
            np.array([y1, y2], np.float32),
            w,
        ))
    return curves