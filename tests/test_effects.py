"""Тесты effects: каждый эффект, агрегатор, edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from ornament_gen.core import EffectSpec, GenerationContext, Quality
from ornament_gen.effects import (
    EffectLayer,
    apply_emboss,
    apply_glow,
    apply_inner_glow,
    apply_micro_echo,
    apply_shadow,
    build_effect_layers,
)


@pytest.fixture
def ctx():
    return GenerationContext(width=1110, height=756, seed=0, quality=Quality.DRAFT)


@pytest.fixture
def simple_mask():
    m = np.zeros((200, 300), np.float32)
    m[80:120, 100:200] = 1.0
    return m


@pytest.fixture
def simple_color(simple_mask):
    return np.full((*simple_mask.shape, 3), 0.5, np.float32)


# ---------- glow ----------

def test_glow_extends_mask(simple_mask, simple_color, ctx):
    layer = apply_glow(simple_mask, simple_color, radius_rel=0.02, strength=0.5, ctx=ctx)
    assert layer.mask.shape == simple_mask.shape
    assert layer.mask.dtype == np.float32
    assert layer.mask.min() >= 0.0 and layer.mask.max() <= 1.0
    assert layer.blend == "screen"
    outside = simple_mask == 0
    assert layer.mask[outside].max() > 0.0


def test_glow_brighter_than_base(simple_mask, simple_color, ctx):
    layer = apply_glow(simple_mask, simple_color, radius_rel=0.01, strength=0.5, ctx=ctx)
    inside = simple_mask > 0
    assert (layer.color[inside] >= simple_color[inside] - 1e-6).all()


# ---------- inner_glow ----------

def test_inner_glow_modifies_mask(simple_mask, ctx):
    out = apply_inner_glow(simple_mask, radius_rel=0.01, strength=0.5, ctx=ctx)
    assert out.shape == simple_mask.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert (out[simple_mask == 0] == 0.0).all()


# ---------- emboss (базовые) ----------

def test_emboss_returns_hi_and_lo(simple_mask, ctx):
    hi, lo = apply_emboss(simple_mask, depth=1.5, light_dir=(-0.5, -0.5, 0.7), ctx=ctx)
    assert isinstance(hi, EffectLayer)
    assert isinstance(lo, EffectLayer)
    assert hi.blend == "add"
    assert lo.blend == "multiply"
    assert hi.mask.shape == simple_mask.shape
    both = (hi.mask > 1e-4) & (lo.mask > 1e-4)
    assert not both.any()


def test_emboss_flat_mask_no_output(ctx):
    flat = np.ones((50, 50), np.float32)
    hi, lo = apply_emboss(flat, depth=1.5, light_dir=(-0.5, -0.5, 0.7), ctx=ctx)
    assert hi.mask.max() < 1e-3
    assert lo.mask.max() < 1e-3


# ---------- emboss (расширение 3/2: sigma_rel) ----------

def test_emboss_custom_sigma(simple_mask, ctx):
    hi, lo = apply_emboss(
        simple_mask, depth=1.0, light_dir=(-0.5, -0.5, 0.7), ctx=ctx,
        sigma_rel=0.01,
    )
    assert hi.mask.shape == simple_mask.shape
    assert lo.mask.shape == simple_mask.shape


def test_emboss_small_sigma_sharper(simple_mask, ctx):
    hi_small, _ = apply_emboss(
        simple_mask, depth=1.0, light_dir=(-0.5, -0.5, 0.7), ctx=ctx,
        sigma_rel=0.001,
    )
    hi_big, _ = apply_emboss(
        simple_mask, depth=1.0, light_dir=(-0.5, -0.5, 0.7), ctx=ctx,
        sigma_rel=0.01,
    )
    assert (hi_small.mask > 0.5).sum() >= (hi_big.mask > 0.5).sum()


def test_emboss_params_passed_from_build(simple_mask, simple_color, ctx):
    effects = [EffectSpec(kind="emboss", params={"depth": 1.0, "sigma_rel": 0.01})]
    res = build_effect_layers(simple_mask, simple_color, effects, ctx)
    assert len(res.above) == 2


# ---------- shadow ----------

def test_shadow_shifted_and_blurred(simple_mask, ctx):
    layer = apply_shadow(
        simple_mask, offset_rel=(0.02, 0.02), radius_rel=0.005,
        opacity=0.6, ctx=ctx,
    )
    assert layer.blend == "multiply"
    assert layer.opacity == pytest.approx(0.6, abs=1e-6)
    assert layer.mask.shape == simple_mask.shape
    H, W = simple_mask.shape
    cy, cx = 100, 150
    orig_val = simple_mask[cy, cx]
    shadow_val = layer.mask[cy, cx]
    assert orig_val == 1.0
    assert shadow_val < 0.95


def test_shadow_offset_zero_is_blurred(simple_mask, ctx):
    layer = apply_shadow(
        simple_mask, offset_rel=(0.0, 0.0), radius_rel=0.005,
        opacity=1.0, ctx=ctx,
    )
    inside = simple_mask > 0
    assert layer.mask[inside].max() > 0.3


# ---------- micro_echo ----------

def test_micro_echo_not_overlapping_main(simple_mask, simple_color, ctx):
    layer = apply_micro_echo(
        simple_mask, simple_color,
        offset_rel=(0.01, 0.01), radius_rel=0.003, fade=0.5, ctx=ctx,
    )
    inside = simple_mask > 0
    assert layer.mask[inside].max() < 0.05


def test_micro_echo_outside_has_some_signal(simple_mask, simple_color, ctx):
    layer = apply_micro_echo(
        simple_mask, simple_color,
        offset_rel=(0.03, 0.03), radius_rel=0.003, fade=1.0, ctx=ctx,
    )
    assert layer.mask.max() > 0.1


# ---------- агрегатор ----------

def test_build_no_effects(simple_mask, simple_color, ctx):
    res = build_effect_layers(simple_mask, simple_color, None, ctx)
    assert res.main_mask.shape == simple_mask.shape
    assert res.main_color.shape == simple_color.shape
    assert res.below == []
    assert res.above == []
    np.testing.assert_allclose(res.main_mask, simple_mask, atol=1e-6)


def test_build_multiple_effects(simple_mask, simple_color, ctx):
    effects = [
        EffectSpec(kind="glow", params={"radius": 0.005, "strength": 0.4}),
        EffectSpec(kind="shadow", params={"offset": (0.01, 0.01), "radius": 0.004, "opacity": 0.5}),
        EffectSpec(kind="inner_glow", params={"radius": 0.003, "strength": 0.3}),
        EffectSpec(kind="emboss", params={"depth": 1.5}),
        EffectSpec(kind="micro_echo", params={"offset": (0.005, 0.005), "fade": 0.3}),
    ]
    res = build_effect_layers(simple_mask, simple_color, effects, ctx)
    assert len(res.below) == 2
    assert len(res.above) == 3
    assert res.main_mask.min() >= 0.0 and res.main_mask.max() <= 1.0


def test_build_unknown_effect_ignored(simple_mask, simple_color, ctx):
    effects = [EffectSpec(kind="unknown_thing", params={"x": 1})]
    res = build_effect_layers(simple_mask, simple_color, effects, ctx)
    assert res.below == []
    assert res.above == []


def test_build_accepts_uint8_mask(simple_color, ctx):
    m = np.zeros((50, 50), np.uint8)
    m[10:30, 10:30] = 255
    res = build_effect_layers(m, simple_color[:50, :50], None, ctx)
    assert res.main_mask.dtype == np.float32
    assert res.main_mask.max() <= 1.0


def test_build_accepts_flat_color(simple_mask, ctx):
    res = build_effect_layers(simple_mask, 0.7, None, ctx)
    assert res.main_color.shape == (*simple_mask.shape, 3)
    assert np.allclose(res.main_color, 0.7)


def test_build_accepts_rgb_tuple(simple_mask, ctx):
    res = build_effect_layers(simple_mask, (0.2, 0.5, 0.8), None, ctx)
    assert res.main_color[..., 0].mean() == pytest.approx(0.2, abs=1e-5)
    assert res.main_color[..., 1].mean() == pytest.approx(0.5, abs=1e-5)
    assert res.main_color[..., 2].mean() == pytest.approx(0.8, abs=1e-5)