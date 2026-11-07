"""Юнит-тесты фундамента. Никакого PIL/numpy в тестовом окружении быть не должно
падать — если падает, значит зависимость не установлена."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ornament_gen.blend import BLEND_MODES, blend
from ornament_gen.core import (
    BackgroundKind,
    GenerationContext,
    Layer,
    Quality,
    ThicknessSpec,
)
from ornament_gen.metadata import MetadataWriter, _clean
from ornament_gen.sizes import Orientation, SizeSpecError, resolve_size


# ---------- sizes ----------

def test_size_iso_a4_landscape_300dpi():
    s = resolve_size(size="a4", dpi=300, orientation=Orientation.LANDSCAPE)
    assert (s.width, s.height) == (3508, 2480)
    assert s.orientation is Orientation.LANDSCAPE


def test_size_iso_a4_portrait_300dpi():
    s = resolve_size(size="a4", dpi=300, orientation=Orientation.PORTRAIT)
    assert (s.width, s.height) == (2480, 3508)
    assert s.orientation is Orientation.PORTRAIT


def test_size_dpi_bounds():
    with pytest.raises(SizeSpecError):
        resolve_size(size="a4", dpi=100)
    with pytest.raises(SizeSpecError):
        resolve_size(size="a4", dpi=700)
    resolve_size(size="a4", dpi=150)
    resolve_size(size="a4", dpi=600)


def test_size_minimum():
    resolve_size(width=1110, height=756)
    resolve_size(width=756, height=1110)
    with pytest.raises(SizeSpecError):
        resolve_size(width=1000, height=700)


def test_size_missing_spec():
    with pytest.raises(SizeSpecError):
        resolve_size()


def test_size_unknown_iso():
    with pytest.raises(SizeSpecError):
        resolve_size(size="b5", dpi=300)


# ---------- core ----------

def test_quality_params():
    for q in Quality:
        ctx = GenerationContext(width=1110, height=756, seed=0, quality=q)
        assert ctx.ss >= 2
        assert ctx.curve_points >= 1500


def test_layer_rng_deterministic():
    ctx = GenerationContext(width=1110, height=756, seed=42)
    a = ctx.layer_rng(0).random()
    b = ctx.layer_rng(0).random()
    assert a == b
    c = ctx.layer_rng(1).random()
    assert a != c


def test_layer_defaults():
    layer = Layer(kind="net", name="warped")
    assert layer.blend == "normal"
    assert layer.opacity == 1.0
    assert layer.weave_group == "A"
    assert isinstance(layer.thickness, ThicknessSpec)


# ---------- blend ----------

def _ar(*vals):
    return np.array(vals, dtype=np.float32)


def test_blend_normal():
    b = _ar([[[0.2, 0.4, 0.6]]])
    t = _ar([[[0.8, 0.6, 0.2]]])
    r = blend(b, t, "normal")
    np.testing.assert_allclose(r, t)


def test_blend_multiply_screen():
    b = _ar([[[0.5, 0.25, 0.0]]])
    t = _ar([[[0.5, 0.5, 0.5]]])
    np.testing.assert_allclose(blend(b, t, "multiply"), _ar([[[0.25, 0.125, 0.0]]]))
    np.testing.assert_allclose(blend(b, t, "screen"), _ar([[[0.75, 0.625, 0.5]]]))


def test_blend_add_clamps():
    b = _ar([[[0.9, 0.5, 0.1]]])
    t = _ar([[[0.9, 0.5, 0.1]]])
    r = blend(b, t, "add")
    np.testing.assert_allclose(r, _ar([[[1.0, 1.0, 0.2]]]))


def test_blend_opacity_zero_returns_base():
    b = _ar([[[0.3, 0.3, 0.3]]])
    t = _ar([[[0.9, 0.9, 0.9]]])
    r = blend(b, t, "normal", opacity=0.0)
    np.testing.assert_allclose(r, b)


def test_all_modes_shape_and_range():
    b = np.random.rand(8, 8, 3).astype(np.float32)
    t = np.random.rand(8, 8, 3).astype(np.float32)
    for mode in BLEND_MODES:
        r = blend(b, t, mode)
        assert r.shape == b.shape
        assert r.dtype == np.float32
        assert r.min() >= 0.0 and r.max() <= 1.0


def test_blend_unknown_raises():
    with pytest.raises(ValueError):
        blend(np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 3), np.float32), "xyz")


# ---------- metadata ----------

def test_clean_handles_enum_dataclass():
    layer = Layer(kind="net", name="spiral", opacity=0.8)
    ctx = GenerationContext(width=1110, height=756, seed=1, quality=Quality.HIGH,
                            background=BackgroundKind.NONE)
    out = _clean({"ctx": ctx, "layer": layer})
    assert out["ctx"]["quality"] == "high"
    assert out["ctx"]["background"] == "none"
    assert out["layer"]["name"] == "spiral"
    assert out["layer"]["opacity"] == 0.8


def test_metadata_writer_roundtrip(tmp_path):
    p = tmp_path / "meta.jsonl"
    with MetadataWriter(p) as w:
        w.write({"file_name": "00001.png", "width": 1110})
        w.write({"file_name": "00002.png", "width": 756})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["version"] == "2.0.0"
    assert a["file_name"] == "00001.png"
    assert b["width"] == 756