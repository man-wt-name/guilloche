"""Тесты симметрии: валидатор, каждый режим, edge cases."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ornament_gen.symmetry import (
    apply_symmetry,
    is_valid_symmetry,
)


# ---------- валидатор ----------

@pytest.mark.parametrize("mode", ["none", "vertical", "horizontal", "central",
                                  "radial2", "radial6", "radial12"])
def test_validator_accepts(mode):
    assert is_valid_symmetry(mode) is True


@pytest.mark.parametrize("mode", ["", "diagonal", "radial", "radial1",
                                  "radialX", "radial0", "vertical2", "None"])
def test_validator_rejects(mode):
    assert is_valid_symmetry(mode) is False


# ---------- none ----------

def test_none_returns_input():
    a = np.random.rand(20, 30).astype(np.float32)
    out = apply_symmetry(a, "none")
    np.testing.assert_array_equal(out, a)


# ---------- vertical ----------

def test_vertical_symmetry():
    a = np.zeros((20, 40), np.float32)
    a[:, 5] = 1.0
    out = apply_symmetry(a, "vertical")
    np.testing.assert_array_equal(out, np.flip(out, axis=1))
    assert (out[0] == 1.0).sum() == 2


def test_vertical_preserves_max():
    a = np.zeros((20, 40), np.float32)
    a[:, 10] = 0.5
    a[:, 25] = 1.0
    out = apply_symmetry(a, "vertical")
    assert np.isclose(out[0].max(), 1.0)


# ---------- horizontal ----------

def test_horizontal_symmetry():
    a = np.zeros((40, 20), np.float32)
    a[5, :] = 1.0
    out = apply_symmetry(a, "horizontal")
    np.testing.assert_array_equal(out, np.flip(out, axis=0))
    assert (out[:, 0] == 1.0).sum() == 2


# ---------- central ----------

def test_central_180():
    a = np.zeros((30, 30), np.float32)
    a[5, 5] = 1.0
    out = apply_symmetry(a, "central")
    np.testing.assert_array_equal(out, out[::-1, ::-1])
    assert out[5, 5] == 1.0
    assert out[30 - 1 - 5, 30 - 1 - 5] == 1.0


# ---------- radial ----------

def test_radial4_on_square():
    """Один пиксель в центре четверти → 4 точки после radial4."""
    a = np.zeros((200, 200), np.float32)
    a[50, 50] = 1.0
    out = apply_symmetry(a, "radial4")
    assert out.sum() >= 3.5
    np.testing.assert_allclose(out, np.rot90(out, k=-1), atol=0.05)


def test_radial6_on_square():
    a = np.zeros((300, 300), np.float32)
    a[40, 150] = 1.0
    out = apply_symmetry(a, "radial6")
    rotated = np.asarray(
        Image.fromarray(out, mode="F").rotate(-60.0, resample=Image.BILINEAR, expand=False)
    )
    assert np.abs(out - rotated).mean() < 0.05


def test_radial2_equals_central():
    a = np.random.rand(50, 50).astype(np.float32)
    a[a < 0.95] = 0.0
    out_r2 = apply_symmetry(a, "radial2")
    out_c = apply_symmetry(a, "central")
    np.testing.assert_allclose(out_r2, out_c, atol=1e-5)


def test_radial_on_rect_pads_and_crops():
    """Для прямоугольника radial не должен обрезать линии в углах."""
    a = np.zeros((60, 100), np.float32)
    a[30, 50] = 1.0
    out = apply_symmetry(a, "radial4")
    assert out.shape == (60, 100)
    assert out[30, 50] > 0.5


# ---------- invalid ----------

def test_invalid_mode_raises():
    a = np.zeros((10, 10), np.float32)
    with pytest.raises(ValueError):
        apply_symmetry(a, "diagonal")
    with pytest.raises(ValueError):
        apply_symmetry(a, "radial1")


# ---------- 3D ----------

def test_vertical_3d_rgb():
    a = np.zeros((20, 40, 3), np.float32)
    a[:, 5, 0] = 1.0
    out = apply_symmetry(a, "vertical")
    assert out.shape == (20, 40, 3)
    np.testing.assert_array_equal(out, np.flip(out, axis=1))


def test_radial4_3d_rgb():
    a = np.zeros((200, 200, 3), np.float32)
    a[50, 50, 0] = 1.0
    out = apply_symmetry(a, "radial4")
    assert out.shape == (200, 200, 3)
    assert out[..., 0].sum() >= 3.5