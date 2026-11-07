"""CLI: batch, spec round-trip, ошибки, RGBA, preview, оверрайды слоёв."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ornament_gen.cli import main


# ==================================================================
# Фаза 4b — 15 базовых тестов
# ==================================================================

def test_cli_batch_generates_files(tmp_path):
    out = tmp_path / "out1"
    rc = main(["--n", "3", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--out", str(out)])
    assert rc == 0
    files = sorted(out.glob("*.png"))
    assert [p.name for p in files] == ["00000.png", "00001.png", "00002.png"]
    meta = (out / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(meta) == 3
    entry = json.loads(meta[0])
    assert entry["version"] == "2.0.0"
    assert entry["file_name"] == "00000.png"
    assert "layers" in entry
    assert "caption" in entry


def test_cli_n_exceeds_limit(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "51", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--out", str(tmp_path / "x")])


def test_cli_n_zero(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "0", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--out", str(tmp_path / "x")])


def test_cli_recipe_unknown(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--recipe", "nope",
              "--out", str(tmp_path / "x")])


def test_cli_dpi_out_of_range(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "900",
              "--out", str(tmp_path / "x")])


def test_cli_too_small(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--width", "800", "--height", "600",
              "--out", str(tmp_path / "x")])


def test_cli_transparent_background(tmp_path):
    out = tmp_path / "out_rgba"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--background", "none", "--out", str(out)])
    assert rc == 0
    img = Image.open(out / "00000.png")
    assert img.mode == "RGBA"
    arr = np.asarray(img, np.uint8)
    assert (arr[..., 3] == 0).any()
    assert (arr[..., 3] > 0).any()


def test_cli_spec_roundtrip_bit_for_bit(tmp_path):
    """--dump-spec → --layers-spec даёт идентичный PNG."""
    out1 = tmp_path / "out_spec1"
    spec_path = tmp_path / "spec.json"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--seed", "42", "--dump-spec", str(spec_path),
               "--out", str(out1)])
    assert rc == 0
    assert spec_path.exists()

    out2 = tmp_path / "out_spec2"
    rc = main(["--layers-spec", str(spec_path), "--out", str(out2)])
    assert rc == 0

    a = np.asarray(Image.open(out1 / "00000.png"))
    b = np.asarray(Image.open(out2 / "00000.png"))
    np.testing.assert_array_equal(a, b)


def test_cli_random_layers(tmp_path):
    out = tmp_path / "out_rand"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--random-layers", "3",
               "--seed", "7", "--out", str(out)])
    assert rc == 0
    assert (out / "00000.png").exists()


def test_cli_random_layers_deterministic(tmp_path):
    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    for out in (out1, out2):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--random-layers", "3",
              "--seed", "7", "--out", str(out)])
    a = np.asarray(Image.open(out1 / "00000.png"))
    b = np.asarray(Image.open(out2 / "00000.png"))
    np.testing.assert_array_equal(a, b)


def test_cli_recipe_and_random_layers_conflict(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--recipe", "woven_classic", "--random-layers", "3",
              "--out", str(tmp_path / "x")])


def test_cli_preview(tmp_path):
    out = tmp_path / "out_prev"
    rc = main(["--n", "2", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--preview", "2", "--out", str(out)])
    assert rc == 0
    assert (out / "preview.png").exists()


def test_cli_flat_color(tmp_path):
    out = tmp_path / "out_flat"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--flat-color", "--background", "none", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["flat_color"] is True
    assert entry["color"]["source"] == "flat"


def test_cli_weave_override(tmp_path):
    out = tmp_path / "out_weave"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--weave", "none", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["weave"]["mode"] == "none"


def test_cli_as_module(tmp_path):
    """`python -m ornament_gen --help` не падает."""
    r = subprocess.run(
        [sys.executable, "-m", "ornament_gen", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    assert "--size" in r.stdout


# ==================================================================
# Расширение 3/1 — симметрия (3 теста)
# ==================================================================

def test_cli_symmetry_vertical(tmp_path):
    out = tmp_path / "out_sym_v"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--symmetry", "vertical", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["symmetry"] == "vertical"
    arr = np.asarray(Image.open(out / "00000.png"))
    np.testing.assert_array_equal(arr, np.flip(arr, axis=1))


def test_cli_symmetry_radial(tmp_path):
    out = tmp_path / "out_sym_r"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "rosette_banknote",
               "--symmetry", "radial4", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["symmetry"] == "radial4"


def test_cli_symmetry_invalid(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--symmetry", "diagonal",
              "--out", str(tmp_path / "x")])


# ==================================================================
# Расширение 3/2 — ref-warp (3 теста)
# ==================================================================

def test_cli_ref_warp_requires_ref(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--recipe", "woven_classic",
              "--ref-warp", "0.5", "--out", str(tmp_path / "x")])


def test_cli_ref_warp_with_ref(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (300, 300), (128, 100, 200)).save(ref)
    out = tmp_path / "out_warp"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--ref", str(ref), "--ref-warp", "0.6",
               "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["ref_warp"] == pytest.approx(0.6)


def test_cli_spec_roundtrip_with_reference(tmp_path):
    """--dump-spec с --ref → --layers-spec должен воспроизвести reference-цвет."""
    ref = tmp_path / "ref.png"
    yy, xx = np.mgrid[0:200, 0:200]
    arr = np.stack([
        (np.sin(xx / 15) * 100 + 128).astype(np.uint8),
        (np.cos(yy / 15) * 100 + 128).astype(np.uint8),
        (np.sin((xx + yy) / 20) * 100 + 128).astype(np.uint8),
    ], -1)
    Image.fromarray(arr).save(ref)

    out1 = tmp_path / "s1"
    spec_path = tmp_path / "spec.json"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--ref", str(ref), "--ref-fixed", "--ref-warp", "0.5",
               "--seed", "42", "--dump-spec", str(spec_path),
               "--out", str(out1)])
    assert rc == 0

    out2 = tmp_path / "s2"
    rc = main(["--layers-spec", str(spec_path), "--out", str(out2)])
    assert rc == 0

    a = np.asarray(Image.open(out1 / "00000.png"))
    b = np.asarray(Image.open(out2 / "00000.png"))
    np.testing.assert_array_equal(a, b)


# ==================================================================
# Расширение 3/3 — RAM и кэш (3 теста)
# ==================================================================

def test_cli_no_cache(tmp_path):
    out = tmp_path / "out_nocache"
    rc = main(["--n", "2", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--no-cache", "--out", str(out)])
    assert rc == 0
    assert (out / "00000.png").exists()
    assert (out / "00001.png").exists()


def test_cli_max_ram_triggers_downgrade(tmp_path, capsys):
    out = tmp_path / "out_ram"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "high", "--recipe", "woven_classic",
               "--max-ram", "200", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "auto-downgraded" in captured.err or "exceeds limit" in captured.err
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["quality"] in ("draft", "normal")


def test_cli_max_ram_disabled(tmp_path, capsys):
    out = tmp_path / "out_ram0"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--max-ram", "0", "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "auto-downgraded" not in captured.err


# ==================================================================
# Расширение 4 — оверрайды слоёв (9 тестов)
# ==================================================================

def test_cli_layer_opacity_override(tmp_path):
    out = tmp_path / "out_op"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--layer-opacity", "0.5", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for layer in entry["layers"]:
        assert layer["opacity"] == pytest.approx(0.5)


def test_cli_layer_opacity_invalid(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--recipe", "woven_classic",
              "--layer-opacity", "1.5", "--out", str(tmp_path / "x")])


def test_cli_layer_blend_override(tmp_path):
    out = tmp_path / "out_bl"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--layer-blend", "screen", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for layer in entry["layers"]:
        assert layer["blend"] == "screen"


def test_cli_layer_blend_invalid(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--recipe", "woven_classic",
              "--layer-blend", "magic", "--out", str(tmp_path / "x")])


def test_cli_weave_z_period_override(tmp_path):
    out = tmp_path / "out_zp"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--weave-z-period", "7.5", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for layer in entry["layers"]:
        assert layer["z_period"] == pytest.approx(7.5)


def test_cli_weave_z_phase_override(tmp_path):
    out = tmp_path / "out_zh"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--weave-z-phase", "1.57", "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for layer in entry["layers"]:
        assert layer["z_phase"] == pytest.approx(1.57)


def test_cli_thickness_requires_variable(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--recipe", "woven_classic",
              "--thickness-amplitude", "0.5",
              "--out", str(tmp_path / "x")])


def test_cli_variable_thickness_applied(tmp_path):
    out = tmp_path / "out_vt"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--variable-thickness", "--thickness-amplitude", "0.4",
               "--thickness-period", "5.0",
               "--out", str(out)])
    assert rc == 0
    entry = json.loads((out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for layer in entry["layers"]:
        assert layer["thickness"]["kind"] == "modulated"
        assert layer["thickness"]["amplitude"] == pytest.approx(0.4)
        assert layer["thickness"]["period"] == pytest.approx(5.0)


def test_cli_overrides_ignored_with_spec(tmp_path, capsys):
    """При --layers-spec оверрайды игнорируются + warning."""
    out1 = tmp_path / "spec_a"
    spec_path = tmp_path / "s.json"
    rc = main(["--n", "1", "--size", "a4", "--dpi", "150",
               "--quality", "draft", "--recipe", "woven_classic",
               "--seed", "42", "--dump-spec", str(spec_path),
               "--out", str(out1)])
    assert rc == 0

    out2 = tmp_path / "spec_b"
    rc = main(["--layers-spec", str(spec_path),
               "--layer-opacity", "0.5",
               "--out", str(out2)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "игнорируются" in captured.err

    entry1 = json.loads((out1 / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    entry2 = json.loads((out2 / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for l1, l2 in zip(entry1["layers"], entry2["layers"]):
        assert l2["opacity"] == l1["opacity"]


# ==================================================================
# P1-7 + P1-8 — валидация и OSError (2 теста)
# ==================================================================

def test_cli_random_layers_invalid(tmp_path):
    """P1-8: --random-layers < 1 → ap.error → SystemExit."""
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--random-layers", "0",
              "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--random-layers", "-3",
              "--out", str(tmp_path / "x")])


def test_cli_dump_spec_invalid_dir(tmp_path, capsys):
    """P1-7: --dump-spec в несуществующую директорию → ap.error → SystemExit."""
    out = tmp_path / "out"
    spec_path = tmp_path / "missing_dir" / "spec.json"
    with pytest.raises(SystemExit):
        main(["--n", "1", "--size", "a4", "--dpi", "150",
              "--quality", "draft", "--recipe", "woven_classic",
              "--dump-spec", str(spec_path),
              "--out", str(out)])
    captured = capsys.readouterr()
    assert "не удалось сохранить --dump-spec" in captured.err