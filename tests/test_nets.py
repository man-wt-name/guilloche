"""Смоук-тесты всех сеток: не падают, возвращают непустые Curves, координаты конечны."""
from __future__ import annotations

import random

import numpy as np
import pytest

from ornament_gen.core import GenerationContext, Quality
from ornament_gen.nets import NETS


@pytest.fixture
def ctx():
    return GenerationContext(width=1110, height=756, seed=42, quality=Quality.DRAFT)


def test_registry_complete():
    assert set(NETS) == {
        "warped", "lens", "polar", "spiral", "spirograph",
        "lissajous", "rose", "star", "flow",
    }


@pytest.mark.parametrize("name", list(NETS))
def test_each_net_smoke(name, ctx):
    rng = random.Random(42)
    curves = NETS[name](ctx, rng, {})
    assert isinstance(curves, list)
    assert len(curves) >= 1
    for c in curves:
        assert c.xs.shape == c.ys.shape == c.zs.shape
        assert c.xs.dtype == np.float32 and c.ys.dtype == np.float32
        assert np.all(np.isfinite(c.xs))
        assert np.all(np.isfinite(c.ys))
        assert c.width >= 1


@pytest.mark.parametrize("name", list(NETS))
def test_net_covers_canvas(name, ctx):
    """Хотя бы одна кривая должна заходить в центр холста."""
    rng = random.Random(7)
    curves = NETS[name](ctx, rng, {})
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    hit_center = False
    for c in curves:
        in_box = (
            (c.xs > W_ss * 0.25) & (c.xs < W_ss * 0.75)
            & (c.ys > H_ss * 0.25) & (c.ys < H_ss * 0.75)
        )
        if np.any(in_box):
            hit_center = True
            break
    assert hit_center, f"{name} не заходит в центр холста"


@pytest.mark.parametrize("name", list(NETS))
def test_net_deterministic(name, ctx):
    a = NETS[name](ctx, random.Random(123), {})
    b = NETS[name](ctx, random.Random(123), {})
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        np.testing.assert_array_equal(ca.xs, cb.xs)
        np.testing.assert_array_equal(ca.ys, cb.ys)