"""Тесты spec.py: round-trip, валидация, ошибки."""
from __future__ import annotations

import pytest

from ornament_gen.compose import WeaveConfig
from ornament_gen.core import BackgroundKind, Quality
from ornament_gen.spec import (
    SpecError,
    context_from_dict,
    layer_from_dict,
    load_spec_file,
    save_spec_file,
    weave_from_dict,
)


# ---------- layer_from_dict ----------

def test_layer_roundtrip():
    d = {
        "kind": "net",
        "name": "spirograph",
        "params": {"r_ratio": 0.3},
        "opacity": 0.8,
        "blend": "multiply",
        "weave_group": "A",
        "z_preset": "sine",
        "z_period": 3.0,
        "z_phase": 0.5,
        "effects": [{"kind": "glow", "params": {"radius": 0.005}}],
        "thickness": {"kind": "modulated", "amplitude": 0.4, "period": 5.0},
    }
    layer = layer_from_dict(d)
    assert layer.kind == "net"
    assert layer.name == "spirograph"
    assert layer.opacity == 0.8
    assert layer.blend == "multiply"
    assert layer.z_preset == "sine"
    assert len(layer.effects) == 1
    assert layer.effects[0].kind == "glow"
    assert layer.thickness.kind == "modulated"


def test_layer_invalid_kind():
    with pytest.raises(SpecError):
        layer_from_dict({"kind": "weird", "name": "x"})
    with pytest.raises(SpecError):
        layer_from_dict({"kind": "net", "name": ""})
    with pytest.raises(SpecError):
        layer_from_dict({"kind": "net"})


def test_layer_invalid_effect():
    with pytest.raises(SpecError):
        layer_from_dict({"kind": "net", "name": "warped", "effects": ["not_a_dict"]})


# ---------- context_from_dict ----------

def test_context_roundtrip():
    d = {
        "width": 1110, "height": 756, "seed": 42,
        "quality": "high", "background": "none",
        "noise": 0.01, "flat_color": True, "variable_thickness": True,
    }
    ctx = context_from_dict(d)
    assert ctx.width == 1110
    assert ctx.quality is Quality.HIGH
    assert ctx.background is BackgroundKind.NONE
    assert ctx.noise == 0.01
    assert ctx.flat_color is True
    assert ctx.variable_thickness is True


def test_context_invalid_quality():
    with pytest.raises(SpecError):
        context_from_dict({"width": 1110, "height": 756, "quality": "ultra"})


def test_context_invalid_background():
    with pytest.raises(SpecError):
        context_from_dict({"width": 1110, "height": 756, "background": "rainbow"})


# ---------- context_from_dict: symmetry ----------

def test_context_symmetry_roundtrip():
    ctx = context_from_dict({
        "width": 1110, "height": 756, "symmetry": "radial6",
    })
    assert ctx.symmetry == "radial6"


def test_context_invalid_symmetry():
    with pytest.raises(SpecError):
        context_from_dict({
            "width": 1110, "height": 756, "symmetry": "diagonal",
        })


def test_context_symmetry_default_none():
    ctx = context_from_dict({"width": 1110, "height": 756})
    assert ctx.symmetry == "none"


# ---------- weave_from_dict ----------

def test_weave_roundtrip():
    w = weave_from_dict({"mode": "zbuffer", "shadow": 0.3})
    assert w.mode == "zbuffer"
    assert w.shadow == pytest.approx(0.3)


def test_weave_default():
    assert weave_from_dict(None) == WeaveConfig()
    assert weave_from_dict({}).mode == "none"


def test_weave_invalid_mode():
    with pytest.raises(SpecError):
        weave_from_dict({"mode": "magic", "shadow": 0.2})


def test_weave_invalid_shadow():
    with pytest.raises(SpecError):
        weave_from_dict({"mode": "zbuffer", "shadow": 2.0})


# ---------- файлы ----------

def test_spec_file_roundtrip(tmp_path):
    p = tmp_path / "spec.json"
    spec = {"width": 1110, "height": 756, "layers": [{"kind": "net", "name": "warped"}]}
    save_spec_file(p, spec)
    assert load_spec_file(p) == spec


def test_spec_file_invalid(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json {", encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec_file(p)


# ---------- P1-7: OSError → SpecError ----------

def test_spec_file_not_found(tmp_path):
    """P1-7: FileNotFoundError → SpecError."""
    with pytest.raises(SpecError):
        load_spec_file(tmp_path / "nonexistent.json")


def test_spec_write_to_missing_dir(tmp_path):
    """P1-7: FileNotFoundError (нет директории) → SpecError."""
    with pytest.raises(SpecError):
        save_spec_file(tmp_path / "nonexistent_dir" / "spec.json", {"a": 1})