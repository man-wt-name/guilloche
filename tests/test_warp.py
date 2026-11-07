"""Тесты ref-warp: warp_fields, apply_warp_to_curves, edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from ornament_gen.core import Curve, GenerationContext, Quality
from ornament_gen.warp import (
    _sample_bilinear,
    apply_warp_to_curves,
    upscale_field,
    warp_fields,
)


@pytest.fixture
def ctx():
    return GenerationContext(width=200, height=150, seed=0, quality=Quality.DRAFT)


def test_warp_zero_strength_zeros():
    lum = np.random.rand(100, 100).astype(np.float32)
    dx, dy = warp_fields(lum, 0.0, 0.03)
    assert (dx == 0).all() and (dy == 0).all()


def test_warp_flat_luminance_no_displacement():
    lum = np.full((100, 100), 0.5, np.float32)
    dx, dy = warp_fields(lum, 1.0, 0.03)
    assert np.abs(dx).max() < 1e-5
    assert np.abs(dy).max() < 1e-5


def test_warp_fields_shape_and_dtype():
    lum = np.random.rand(100, 80).astype(np.float32)
    dx, dy = warp_fields(lum, 0.8, 0.03)
    assert dx.shape == (100, 80) and dy.shape == (100, 80)
    assert dx.dtype == np.float32 and dy.dtype == np.float32


def test_warp_fields_bounded():
    lum = np.random.rand(200, 200).astype(np.float32)
    dx, dy = warp_fields(lum, 1.0, 0.03)
    lim = 0.15 * 200
    assert np.hypot(dx, dy).max() <= lim + 1e-3


def test_warp_fields_contour_pulls():
    """На резком контуре смещение не нулевое."""
    lum = np.zeros((200, 200), np.float32)
    lum[:, 100:] = 1.0
    dx, dy = warp_fields(lum, 1.0, 0.02)
    assert np.abs(dx).max() > 0.1


def test_upscale_field_noop_when_same_size():
    a = np.random.rand(100, 100).astype(np.float32)
    out = upscale_field(a, 100, 100)
    np.testing.assert_array_equal(out, a)


def test_upscale_field_doubles():
    a = np.zeros((50, 50), np.float32)
    a[25, 25] = 1.0
    out = upscale_field(a, 100, 100)
    assert out.shape == (100, 100)
    assert out.max() > 0.3


def test_sample_bilinear_at_integer_coords():
    a = np.zeros((10, 10), np.float32)
    a[5, 5] = 1.0
    v = _sample_bilinear(a, np.array([5.0], np.float32),
                         np.array([5.0], np.float32), 10, 10)
    assert v[0] == pytest.approx(1.0, abs=1e-6)


def test_sample_bilinear_between():
    a = np.zeros((10, 10), np.float32)
    a[5, 5] = 1.0
    a[5, 6] = 1.0
    v = _sample_bilinear(a, np.array([5.5], np.float32),
                         np.array([5.0], np.float32), 10, 10)
    assert v[0] == pytest.approx(1.0, abs=1e-6)


def test_apply_warp_to_curves_no_displacement(ctx):
    c = Curve(np.array([10.0, 20.0, 30.0], np.float32),
              np.array([10.0, 20.0, 30.0], np.float32), 2)
    dx = np.zeros((150, 200), np.float32)
    dy = np.zeros((150, 200), np.float32)
    out = apply_warp_to_curves([c], dx, dy, ctx)
    np.testing.assert_array_equal(out[0].xs, c.xs)
    np.testing.assert_array_equal(out[0].ys, c.ys)
    assert out[0] is not c


def test_apply_warp_to_curves_displacement(ctx):
    c = Curve(np.array([100.0], np.float32),
              np.array([75.0], np.float32), 2)
    dx = np.full((150, 200), 5.0, np.float32)
    dy = np.full((150, 200), 3.0, np.float32)
    out = apply_warp_to_curves([c], dx, dy, ctx)
    assert out[0].xs[0] == pytest.approx(105.0, abs=0.5)
    assert out[0].ys[0] == pytest.approx(78.0, abs=0.5)


def test_apply_warp_to_curves_preserves_zs_closed(ctx):
    c = Curve(np.array([10.0, 20.0], np.float32),
              np.array([10.0, 20.0], np.float32), 2,
              zs=np.array([0.5, 0.7], np.float32), closed=True)
    dx = np.ones((150, 200), np.float32)
    dy = np.ones((150, 200), np.float32)
    out = apply_warp_to_curves([c], dx, dy, ctx)
    np.testing.assert_array_equal(out[0].zs, c.zs)
    assert out[0].closed is True


def test_apply_warp_to_curves_empty():
    dx = np.zeros((150, 200), np.float32)
    dy = np.zeros((150, 200), np.float32)
    ctx = GenerationContext(width=200, height=150, seed=0, quality=Quality.DRAFT)
    assert apply_warp_to_curves([], dx, dy, ctx) == []