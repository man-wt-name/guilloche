"""Тесты patterns: реестр, smoke, детерминизм, покрытие, специфика каждого."""
from __future__ import annotations

import random

import numpy as np
import pytest

from ornament_gen.core import GenerationContext, Quality
from ornament_gen.patterns import PATTERNS


@pytest.fixture
def ctx():
    return GenerationContext(width=1110, height=756, seed=42, quality=Quality.DRAFT)


def test_registry_complete():
    assert set(PATTERNS) == {"dots", "crosses", "contours", "voronoi_fill", "rosette"}


@pytest.mark.parametrize("name", list(PATTERNS))
def test_pattern_smoke(name, ctx):
    curves = PATTERNS[name](ctx, random.Random(42), {})
    assert isinstance(curves, list)
    assert len(curves) >= 1
    for c in curves:
        assert c.xs.shape == c.ys.shape == c.zs.shape
        assert c.xs.dtype == np.float32
        assert c.ys.dtype == np.float32
        assert np.all(np.isfinite(c.xs))
        assert np.all(np.isfinite(c.ys))
        assert c.width >= 1


@pytest.mark.parametrize("name", list(PATTERNS))
def test_pattern_deterministic(name, ctx):
    a = PATTERNS[name](ctx, random.Random(123), {})
    b = PATTERNS[name](ctx, random.Random(123), {})
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        np.testing.assert_array_equal(ca.xs, cb.xs)
        np.testing.assert_array_equal(ca.ys, cb.ys)


@pytest.mark.parametrize("name", list(PATTERNS))
def test_pattern_within_canvas(name, ctx):
    """Хотя бы одна кривая внутри холста (с допуском в 1 пиксель)."""
    curves = PATTERNS[name](ctx, random.Random(7), {})
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    hit = False
    for c in curves:
        in_box = (
            (c.xs >= -1) & (c.xs <= W_ss + 1)
            & (c.ys >= -1) & (c.ys <= H_ss + 1)
        )
        if np.any(in_box):
            hit = True
            break
    assert hit, f"{name}: ни одна кривая не попала в холст"


# ---------- dots ----------

def test_dots_jitter_bounded(ctx):
    """P1-5: реальная проверка границ jitter (было: только r.max() > 0)."""
    spacing_rel = 0.05
    jitter_rel = 0.1
    curves = PATTERNS["dots"](ctx, random.Random(1),
                              {"spacing": spacing_rel, "jitter": jitter_rel})
    m = min(ctx.ss_width, ctx.ss_height)
    spacing_px = spacing_rel * m
    jitter_px = jitter_rel * spacing_px
    nx = int(ctx.ss_width / spacing_px) + 2
    ny = int(ctx.ss_height / spacing_px) + 2
    assert len(curves) == nx * ny
    # dots.generate: for i in range(nx): for j in range(ny): idx = i*ny + j
    for idx, c in enumerate(curves):
        cx = float(c.xs.mean())
        cy = float(c.ys.mean())
        i = idx // ny
        j = idx % ny
        assert abs(cx - i * spacing_px) <= jitter_px + 1e-3
        assert abs(cy - j * spacing_px) <= jitter_px + 1e-3


# ---------- crosses ----------

def test_crosses_orthogonal_arms(ctx):
    curves = PATTERNS["crosses"](ctx, random.Random(1), {"angle": 0.0, "arm_length": 0.01})
    assert len(curves) >= 4
    c0, c1 = curves[0], curves[1]
    assert np.isclose(c0.ys[0], c0.ys[1])
    assert np.isclose(c1.xs[0], c1.xs[1])


# ---------- contours ----------

def test_contours_produces_segments(ctx):
    curves = PATTERNS["contours"](ctx, random.Random(3), {
        "n_waves": 4, "n_levels": 4, "smooth": 0.02,
    })
    assert len(curves) > 100
    for c in curves:
        assert len(c.xs) == 2
        assert c.xs[0] >= 0 and c.xs[0] <= ctx.ss_width
        assert c.ys[0] >= 0 and c.ys[0] <= ctx.ss_height


# ---------- voronoi_fill ----------

def test_voronoi_clipped_to_canvas(ctx):
    curves = PATTERNS["voronoi_fill"](ctx, random.Random(5), {"n_cells": 30})
    assert len(curves) > 10
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    for c in curves:
        assert c.xs.min() >= -1 and c.xs.max() <= W_ss + 1
        assert c.ys.min() >= -1 and c.ys.max() <= H_ss + 1


def test_voronoi_too_few_cells(ctx):
    curves = PATTERNS["voronoi_fill"](ctx, random.Random(5), {"n_cells": 2})
    assert curves == []


# ---------- rosette ----------

def test_rosette_n_circles(ctx):
    curves = PATTERNS["rosette"](ctx, random.Random(9), {"n_circles": 8})
    assert len(curves) == 8
    for c in curves:
        assert c.closed is True
        cx, cy = c.xs.mean(), c.ys.mean()
        r = np.hypot(c.xs - cx, c.ys - cy)
        assert r.std() < 0.01


def test_rosette_centered(ctx):
    curves = PATTERNS["rosette"](ctx, random.Random(9), {
        "n_circles": 6, "offset": 0.0, "center_jitter": 0.0,
    })
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    for c in curves:
        cx, cy = c.xs.mean(), c.ys.mean()
        assert abs(cx - W_ss / 2) < 1e-2
        assert abs(cy - H_ss / 2) < 1e-2