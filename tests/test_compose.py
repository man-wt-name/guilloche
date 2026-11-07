"""Тесты compose: полный пайплайн, weave, transparent, effects, symmetry, ref-warp, spec."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from ornament_gen.compose import (
    ComposeResult,
    WeaveConfig,
    _downsample_mask,
    _expand_with_thickness,
    _rasterize_plain,
    make,
)
from ornament_gen.core import (
    BackgroundKind,
    Curve,
    EffectSpec,
    GenerationContext,
    Layer,
    Quality,
    ThicknessSpec,
)


@pytest.fixture
def ctx():
    return GenerationContext(width=1110, height=756, seed=0, quality=Quality.DRAFT)


# ---------- базовые (Фаза 4a) ----------

def test_compose_returns_pil(ctx):
    layer = Layer(kind="net", name="warped", params={"width": 0.5})
    res = make(ctx, [layer])
    assert isinstance(res, ComposeResult)
    assert res.image.size == (ctx.width, ctx.height)
    assert res.image.mode == "RGB"
    arr = np.asarray(res.image, np.uint8)
    assert arr.shape == (ctx.height, ctx.width, 3)


def test_compose_empty_layers(ctx):
    res = make(ctx, [])
    assert res.image.size == (ctx.width, ctx.height)


def test_compose_unknown_layer_raises(ctx):
    with pytest.raises(ValueError):
        make(ctx, [Layer(kind="net", name="does_not_exist")])
    with pytest.raises(ValueError):
        make(ctx, [Layer(kind="weird", name="warped")])


def test_compose_transparent_background(ctx):
    ctx2 = GenerationContext(
        width=1110, height=756, seed=0, quality=Quality.DRAFT,
        background=BackgroundKind.NONE,
    )
    layer = Layer(kind="net", name="warped", params={"width": 0.5})
    res = make(ctx2, [layer])
    assert res.image.mode == "RGBA"
    arr = np.asarray(res.image, np.uint8)
    assert arr.shape == (ctx2.height, ctx2.width, 4)
    alpha = arr[..., 3]
    assert alpha.max() > 0
    assert (alpha == 0).any()


def test_compose_transparent_no_alpha_when_paper(ctx):
    layer = Layer(kind="net", name="warped")
    res = make(ctx, [layer])
    assert res.image.mode == "RGB"


def test_compose_weave_smoke(ctx):
    layers = [
        Layer(kind="net", name="warped", params={"width": 0.5, "spacing": 0.02},
              z_preset="sine", z_period=4.0, z_phase=0.0),
        Layer(kind="net", name="warped", params={"width": 0.5, "spacing": 0.02},
              z_preset="sine", z_period=4.0, z_phase=np.pi),
    ]
    res = make(ctx, layers, weave_config=WeaveConfig(mode="zbuffer", shadow=0.3))
    assert res.image.size == (ctx.width, ctx.height)
    assert res.spec["weave"]["mode"] == "zbuffer"
    assert res.spec["weave"]["shadow"] == pytest.approx(0.3)


def test_compose_effects_smoke(ctx):
    layer = Layer(
        kind="net", name="warped", params={"width": 0.5},
        effects=[EffectSpec(kind="glow", params={"radius": 0.005, "strength": 0.3})],
    )
    res = make(ctx, [layer])
    assert res.image.size == (ctx.width, ctx.height)


def test_compose_pattern_layer(ctx):
    layer = Layer(kind="pattern", name="dots", params={"spacing": 0.05, "radius": 0.008})
    res = make(ctx, [layer])
    assert res.image.size == (ctx.width, ctx.height)


def test_compose_max_ss_guard():
    ctx = GenerationContext(width=8000, height=6000, seed=0, quality=Quality.MAX)
    with pytest.raises(ValueError, match="слишком велико"):
        make(ctx, [Layer(kind="net", name="warped")])


def test_compose_deterministic(ctx):
    layer = Layer(kind="net", name="spirograph", params={"width": 0.5})
    res1 = make(ctx, [layer])
    res2 = make(ctx, [layer])
    np.testing.assert_array_equal(np.asarray(res1.image), np.asarray(res2.image))
    assert res1.spec == res2.spec


def test_compose_variable_thickness(ctx):
    ctx2 = GenerationContext(
        width=1110, height=756, seed=0, quality=Quality.DRAFT,
        variable_thickness=True,
    )
    layer = Layer(
        kind="net", name="warped", params={"width": 0.5},
        thickness=ThicknessSpec(kind="modulated", amplitude=0.5, period=4.0),
    )
    res = make(ctx2, [layer])
    assert res.image.size == (ctx2.width, ctx2.height)


# ---------- внутренние функции ----------

def test_expand_with_thickness_noop(ctx):
    c = Curve(np.array([0, 1, 2], np.float32), np.array([0, 1, 2], np.float32), 3)
    layer = Layer(kind="net", name="warped")
    out = _expand_with_thickness([c], layer)
    assert len(out) == 1
    assert out[0] is c


def test_expand_with_thickness_modulated(ctx):
    c = Curve(
        np.linspace(0, 100, 100, dtype=np.float32),
        np.linspace(0, 100, 100, dtype=np.float32),
        3,
    )
    layer = Layer(
        kind="net", name="warped",
        thickness=ThicknessSpec(kind="modulated", base=1.0, amplitude=0.5, period=4.0),
    )
    out = _expand_with_thickness([c], layer)
    assert len(out) == 16
    widths = [o.width for o in out]
    assert len(set(round(w, 3) for w in widths)) > 1
    assert all(w >= 1 for w in widths)


def test_expand_with_thickness_closed():
    """P1-2: для closed=True добавляется замыкающий сегмент."""
    n = 100
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy, r = 50.0, 50.0, 30.0
    c = Curve(
        xs=(cx + r * np.cos(theta)).astype(np.float32),
        ys=(cy + r * np.sin(theta)).astype(np.float32),
        width=2.0,
        closed=True,
    )
    layer = Layer(
        kind="net", name="polar",
        thickness=ThicknessSpec(kind="modulated", amplitude=0.3),
    )
    out = _expand_with_thickness([c], layer)
    assert len(out) == 17
    last = out[-1]
    assert np.isclose(last.xs[0], c.xs[-1], atol=1e-5)
    assert np.isclose(last.ys[0], c.ys[-1], atol=1e-5)
    assert np.isclose(last.xs[1], c.xs[0], atol=1e-5)
    assert np.isclose(last.ys[1], c.ys[0], atol=1e-5)
    assert last.closed is False
    assert last.width >= 1.0


def test_expand_with_thickness_not_closed_unchanged():
    """P1-2: для closed=False ничего не меняется (16 сегментов)."""
    c = Curve(
        np.linspace(0, 100, 100, dtype=np.float32),
        np.linspace(0, 100, 100, dtype=np.float32),
        3.0,
        closed=False,
    )
    layer = Layer(
        kind="net", name="warped",
        thickness=ThicknessSpec(kind="modulated", amplitude=0.5),
    )
    out = _expand_with_thickness([c], layer)
    assert len(out) == 16


def test_rasterize_plain_smoke(ctx):
    c = Curve(np.array([10, 100, 200], np.float32),
              np.array([10, 100, 200], np.float32), 3)
    mask = _rasterize_plain([c], ctx)
    assert mask.dtype == np.uint8
    assert mask.shape == (ctx.ss_height, ctx.ss_width)
    assert (mask > 0).any()


def test_rasterize_plain_empty(ctx):
    mask = _rasterize_plain([], ctx)
    assert mask.shape == (ctx.ss_height, ctx.ss_width)
    assert (mask == 0).all()


def test_downsample_mask_uint8(ctx):
    m = np.zeros((100, 200), np.uint8)
    m[40:60, 80:120] = 255
    out = _downsample_mask(m, 50, 25)
    assert out.shape == (25, 50)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_downsample_mask_float():
    m = np.zeros((100, 200), np.float32)
    m[40:60, 80:120] = 1.0
    out = _downsample_mask(m, 50, 25)
    assert out.shape == (25, 50)
    assert out.dtype == np.float32


# ---------- spec ----------

def test_spec_has_all_fields(ctx):
    layer = Layer(kind="net", name="warped", params={"width": 0.5})
    res = make(ctx, [layer])
    spec = res.spec
    for key in (
        "width", "height", "seed", "quality", "background", "noise",
        "flat_color", "variable_thickness", "symmetry", "ref_warp", "ref_smooth",
        "color", "weave", "layers",
    ):
        assert key in spec, f"missing {key}"
    assert spec["layers"][0]["name"] == "warped"
    assert spec["layers"][0]["kind"] == "net"


def test_spec_serializable(ctx):
    layer = Layer(kind="net", name="warped")
    res = make(ctx, [layer])
    s = json.dumps(res.spec, ensure_ascii=False)
    back = json.loads(s)
    assert back == res.spec


# ---------- регрессия glow (Фаза 4b) ----------

def test_compose_effect_glow_matches_size(ctx):
    """Regression: glow возвращает color в SS; должен даунсэмплиться без ошибок."""
    layer = Layer(
        kind="net", name="warped", params={"width": 0.5},
        effects=[
            EffectSpec(kind="glow", params={"radius": 0.005, "strength": 0.3}),
            EffectSpec(kind="shadow", params={"offset": (0.005, 0.005),
                                              "radius": 0.004, "opacity": 0.4}),
            EffectSpec(kind="emboss", params={"depth": 1.0}),
        ],
    )
    res = make(ctx, [layer])
    assert res.image.size == (ctx.width, ctx.height)
    arr = np.asarray(res.image, np.uint8)
    assert arr.std() > 1.0


# ---------- симметрия (расширение 3/1) ----------

def test_compose_symmetry_vertical():
    ctx_sym = GenerationContext(
        width=1110, height=756, seed=0, quality=Quality.DRAFT,
        symmetry="vertical",
    )
    layer = Layer(kind="net", name="warped", params={"width": 0.5})
    res = make(ctx_sym, [layer])
    arr = np.asarray(res.image, np.uint8)
    np.testing.assert_array_equal(arr, np.flip(arr, axis=1))


def test_compose_symmetry_in_spec():
    ctx_sym = GenerationContext(
        width=1110, height=756, seed=0, quality=Quality.DRAFT,
        symmetry="central",
    )
    res = make(ctx_sym, [Layer(kind="net", name="warped")])
    assert res.spec["symmetry"] == "central"


# ---------- ref-warp (расширение 3/2) ----------

def test_compose_ref_warp_smoke(tmp_path):
    """ref-warp: сетка деформируется — картинка отличается от без-warp."""
    ref_path = tmp_path / "ref.png"
    yy, xx = np.mgrid[0:200, 0:200]
    arr = np.stack([(np.sin(xx / 20) * 127 + 128).astype(np.uint8)] * 3, -1)
    Image.fromarray(arr).save(ref_path)

    ctx_local = GenerationContext(width=1110, height=756, seed=0, quality=Quality.DRAFT)
    layer = Layer(kind="net", name="polar", params={"width": 0.5})
    res_plain = make(ctx_local, [layer], refs=[ref_path])
    res_warp = make(ctx_local, [layer], refs=[ref_path], ref_warp=0.8)
    a = np.asarray(res_plain.image)
    b = np.asarray(res_warp.image)
    assert not np.array_equal(a, b), "ref-warp должен менять картинку"
    assert res_warp.spec["ref_warp"] == pytest.approx(0.8)


def test_compose_no_ref_warp_no_change(tmp_path):
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (200, 200), (128, 128, 128)).save(ref_path)
    ctx_local = GenerationContext(width=1110, height=756, seed=0, quality=Quality.DRAFT)
    layer = Layer(kind="net", name="polar", params={"width": 0.5})
    a = np.asarray(make(ctx_local, [layer], refs=[ref_path], ref_warp=0.0).image)
    b = np.asarray(make(ctx_local, [layer], refs=[ref_path]).image)
    np.testing.assert_array_equal(a, b)


def test_compose_ref_warp_ignored_without_reference(ctx):
    """ref_warp без reference — no-op, не падает."""
    layer = Layer(kind="net", name="warped")
    res = make(ctx, [layer], ref_warp=0.8)
    assert res.image.size == (ctx.width, ctx.height)