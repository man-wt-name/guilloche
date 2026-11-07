"""Тесты weave: z-пресеты, растеризация с z, resolve_weave, интеграция."""
from __future__ import annotations

import random

import numpy as np
import pytest

from ornament_gen.core import Curve, GenerationContext, Quality
from ornament_gen.nets import NETS
from ornament_gen.weave import (
    apply_z_to_curves,
    make_z_along_curve,
    rasterize_with_z,
    resolve_weave,
)


# ---------- make_z_along_curve ----------

def test_z_none_zeros():
    z = make_z_along_curve(10, None, 4.0, 0.0)
    assert z.shape == (10,)
    np.testing.assert_array_equal(z, np.zeros(10, np.float32))


def test_z_none_string_zeros():
    z = make_z_along_curve(10, "none", 4.0, 0.0)
    np.testing.assert_array_equal(z, np.zeros(10, np.float32))


def test_z_zero_length():
    for preset in (None, "sine", "checkerboard", "saw", "constant"):
        assert make_z_along_curve(0, preset, 4.0, 0.0).shape == (0,)


def test_z_sine_range_and_bounds():
    z = make_z_along_curve(200, "sine", 2.0, 0.0)
    assert z.dtype == np.float32
    assert z.min() >= -1.0 - 1e-6 and z.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(z[0], 0.0, atol=1e-5)


def test_z_checkerboard_alternates():
    z = make_z_along_curve(9, "checkerboard", 2.0, 0.0)
    assert z[0] == -1.0
    assert z[-1] == -1.0
    assert -1.0 in z
    assert 1.0 in z


def test_z_saw_range():
    z = make_z_along_curve(100, "saw", 2.0, 0.0)
    assert z.min() >= -1.0 - 1e-6
    assert z.max() <= 1.0 + 1e-6


def test_z_constant():
    z = make_z_along_curve(50, "constant", 4.0, 0.5)
    assert (z == 0.5).all()


def test_z_unknown_raises():
    with pytest.raises(ValueError):
        make_z_along_curve(10, "unknown_preset", 1.0, 0.0)


# ---------- apply_z_to_curves ----------

def test_apply_z_modifies_in_place():
    c1 = Curve(np.array([0.0, 1.0], np.float32),
               np.array([0.0, 1.0], np.float32), 1)
    c2 = Curve(np.array([0.0, 1.0, 2.0], np.float32),
               np.array([0.0, 1.0, 2.0], np.float32), 1)
    apply_z_to_curves([c1, c2], "constant", 1.0, 0.7)
    assert (c1.zs == 0.7).all()
    assert c2.zs.shape == (3,)
    assert (c2.zs == 0.7).all()


# ---------- rasterize_with_z ----------

@pytest.fixture
def ctx():
    return GenerationContext(width=1110, height=756, seed=0, quality=Quality.DRAFT)


def test_rasterize_empty(ctx):
    mask, z = rasterize_with_z([], ctx)
    assert mask.shape == (ctx.ss_height, ctx.ss_width)
    assert (mask == 0).all()
    assert (z == 0).all()


def test_rasterize_single_point(ctx):
    c = Curve(np.array([100.0], np.float32),
              np.array([100.0], np.float32), 3)
    mask, z = rasterize_with_z([c], ctx)
    assert mask.any()
    assert mask.dtype == np.uint8
    assert z.dtype == np.float32
    assert z.max() == 0.0


def test_rasterize_z_range(ctx):
    c = Curve(
        np.array([100.0, 200.0, 300.0], np.float32),
        np.array([100.0, 150.0, 200.0], np.float32),
        4,
        zs=np.array([-1.0, 0.0, 1.0], np.float32),
    )
    mask, z = rasterize_with_z([c], ctx)
    covered = mask > 0
    assert covered.any()
    assert z[covered].min() >= -1.0 - 1e-5
    assert z[covered].max() <= 1.0 + 1e-5


# ---------- resolve_weave ----------

def _mk_mask(H, W, ys, xs):
    m = np.zeros((H, W), np.uint8)
    m[ys, xs] = 255
    return m


def test_resolve_weave_empty():
    visible, shadow = resolve_weave([])
    assert visible == {}
    assert shadow == {}


def test_resolve_weave_single_layer():
    H, W = 20, 20
    m = _mk_mask(H, W, slice(5, 10), slice(5, 10))
    z = np.ones((H, W), np.float32)
    visible, shadow = resolve_weave([("a", m, z)])
    np.testing.assert_array_equal(visible["a"], m)
    assert "a" not in shadow


def test_resolve_weave_two_layers_no_overlap():
    H, W = 20, 20
    ma = _mk_mask(H, W, slice(0, 5), slice(0, 20))
    mb = _mk_mask(H, W, slice(15, 20), slice(0, 20))
    za = np.ones((H, W), np.float32)
    zb = np.ones((H, W), np.float32)
    visible, _ = resolve_weave([("a", ma, za), ("b", mb, zb)])
    np.testing.assert_array_equal(visible["a"], ma)
    np.testing.assert_array_equal(visible["b"], mb)


def test_resolve_weave_overlap_higher_z_wins():
    H, W = 20, 20
    ma = _mk_mask(H, W, slice(5, 15), slice(5, 15))
    mb = _mk_mask(H, W, slice(5, 15), slice(5, 15))
    za = np.full((H, W), 0.5, np.float32)
    zb = np.full((H, W), 0.3, np.float32)
    visible, shadow = resolve_weave(
        [("a", ma, za), ("b", mb, zb)], shadow_strength=0.5
    )
    np.testing.assert_array_equal(visible["a"], ma)
    assert (visible["b"] == 0).all()
    assert (shadow["a"] == 1.0).all()
    assert np.isclose(shadow["b"].min(), 0.5)


def test_resolve_weave_tie_lower_index_wins():
    H, W = 20, 20
    m = _mk_mask(H, W, slice(5, 15), slice(5, 15))
    z = np.zeros((H, W), np.float32)
    visible, _ = resolve_weave([("a", m, z), ("b", m, z)])
    np.testing.assert_array_equal(visible["a"], m)
    assert (visible["b"] == 0).all()


# ---------- интеграция ----------

def test_weave_integration_two_warped(ctx):
    rng_a = random.Random(101)
    rng_b = random.Random(202)
    curves_a = NETS["warped"](ctx, rng_a, {"width": 0.5, "spacing": 0.02})
    curves_b = NETS["warped"](ctx, rng_b, {"width": 0.5, "spacing": 0.02})
    apply_z_to_curves(curves_a, "sine", 4.0, 0.0)
    apply_z_to_curves(curves_b, "sine", 4.0, np.pi)

    mask_a, z_a = rasterize_with_z(curves_a, ctx)
    mask_b, z_b = rasterize_with_z(curves_b, ctx)
    assert mask_a.any()
    assert mask_b.any()

    overlap = (mask_a > 0) & (mask_b > 0)
    assert overlap.any(), "warped layers should overlap"

    visible, _shadow = resolve_weave(
        [("a", mask_a, z_a), ("b", mask_b, z_b)],
        shadow_strength=0.3,
    )
    a_in_overlap = (visible["a"] > 0) & overlap
    b_in_overlap = (visible["b"] > 0) & overlap
    assert not (a_in_overlap & b_in_overlap).any(), "both layers visible at same pixel"
    assert visible["a"].any()
    assert visible["b"].any()

    total_a_before = int((mask_a > 0).sum())
    total_a_after = int((visible["a"] > 0).sum())
    total_b_before = int((mask_b > 0).sum())
    total_b_after = int((visible["b"] > 0).sum())
    assert total_a_after < total_a_before or total_b_after < total_b_before