"""JSON-спека: сериализация/десериализация ComposeResult.spec и Layer-ов.

P1-7: OSError (включая FileNotFoundError, PermissionError) → SpecError.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compose import WeaveConfig
from .core import (
    BackgroundKind,
    EffectSpec,
    GenerationContext,
    Layer,
    Quality,
    ThicknessSpec,
)


class SpecError(ValueError):
    pass


def layer_from_dict(d: dict[str, Any]) -> Layer:
    kind = d.get("kind")
    name = d.get("name")
    if kind not in ("net", "pattern"):
        raise SpecError(f"layer.kind must be 'net' or 'pattern', got {kind!r}")
    if not isinstance(name, str) or not name:
        raise SpecError(f"layer.name must be non-empty string, got {name!r}")

    effects: list[EffectSpec] = []
    for e in d.get("effects", []) or []:
        if not isinstance(e, dict) or "kind" not in e:
            raise SpecError(f"effect must be dict with 'kind', got {e!r}")
        effects.append(EffectSpec(kind=str(e["kind"]), params=dict(e.get("params", {}))))

    t = d.get("thickness") or {}
    thickness = ThicknessSpec(
        kind=str(t.get("kind", "constant")),
        base=float(t.get("base", 1.0)),
        amplitude=float(t.get("amplitude", 0.3)),
        period=float(t.get("period", 4.0)),
        phase=float(t.get("phase", 0.0)),
    )

    return Layer(
        kind=kind,
        name=name,
        params=dict(d.get("params", {}) or {}),
        opacity=float(d.get("opacity", 1.0)),
        blend=str(d.get("blend", "normal")),
        effects=effects,
        weave_group=str(d.get("weave_group", "A")),
        z_preset=d.get("z_preset"),
        z_period=float(d.get("z_period", 4.0)),
        z_phase=float(d.get("z_phase", 0.0)),
        thickness=thickness,
    )


def context_from_dict(d: dict[str, Any]) -> GenerationContext:
    try:
        quality = Quality(str(d.get("quality", "normal")))
    except ValueError as exc:
        raise SpecError(f"unknown quality: {d.get('quality')!r}") from exc
    try:
        background = BackgroundKind(str(d.get("background", "paper")))
    except ValueError as exc:
        raise SpecError(f"unknown background: {d.get('background')!r}") from exc

    symmetry = str(d.get("symmetry", "none"))

    try:
        return GenerationContext(
            width=int(d["width"]),
            height=int(d["height"]),
            seed=int(d.get("seed", 0)),
            quality=quality,
            background=background,
            noise=float(d.get("noise", 0.0)),
            flat_color=bool(d.get("flat_color", False)),
            variable_thickness=bool(d.get("variable_thickness", False)),
            symmetry=symmetry,
        )
    except ValueError as exc:
        raise SpecError(str(exc)) from exc


def weave_from_dict(d: dict[str, Any] | None) -> WeaveConfig:
    if not d:
        return WeaveConfig()
    mode = str(d.get("mode", "none"))
    if mode not in ("none", "zbuffer"):
        raise SpecError(f"weave.mode must be 'none' or 'zbuffer', got {mode!r}")
    shadow = float(d.get("shadow", 0.0))
    if not (0.0 <= shadow <= 1.0):
        raise SpecError(f"weave.shadow must be in [0,1], got {shadow}")
    return WeaveConfig(mode=mode, shadow=shadow)


def load_spec_file(path: Path) -> dict[str, Any]:
    """P1-7: OSError → SpecError (для --layers-spec и --dump-spec)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"cannot read spec file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON in {path}: {exc}") from exc


def save_spec_file(path: Path, spec: dict[str, Any]) -> None:
    """P1-7: OSError → SpecError."""
    try:
        path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise SpecError(f"cannot write spec file {path}: {exc}") from exc