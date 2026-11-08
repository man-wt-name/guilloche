"""Mask-based эффекты в SS-разрешении.

Каждый эффект возвращает один или несколько `EffectLayer`, которые `compose.py`
будет накладывать поверх (above) или под (below) основного слоя.

`mask` — в SS-разрешении. `color` — в SS или final (caller делает downsample).

ФИКС: apply_emboss использует mode="reflect" (flat mask → h=1 везде → gradient=0).
Остальные эффекты — mode="constant" (blur/shift за пределы маски по дизайну).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

from .core import EffectSpec, GenerationContext


@dataclass
class EffectLayer:
    """Один слой эффекта. mask — в SS, color — в SS или final."""
    mask: np.ndarray
    color: np.ndarray
    blend: str = "normal"
    opacity: float = 1.0


@dataclass
class EffectResult:
    """Результат применения всех эффектов к одному слою."""
    main_mask: np.ndarray
    main_color: np.ndarray
    below: list[EffectLayer] = field(default_factory=list)
    above: list[EffectLayer] = field(default_factory=list)


def _as_float_mask(mask: np.ndarray) -> np.ndarray:
    a = np.asarray(mask)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return a.astype(np.float32, copy=False)


def _as_color_array(color: Any, shape: tuple[int, int]) -> np.ndarray:
    """Привести color к (H, W, 3) float32.

    Если color — 3D RGB массив, возвращается как есть (любого размера).
    Caller отвечает за downsample/upsample при необходимости.
    """
    H, W = shape
    if isinstance(color, np.ndarray):
        c = color.astype(np.float32, copy=False)
        if c.size == 3:
            v = c.reshape(-1, 3)[0]
            return np.broadcast_to(v[None, None, :], (H, W, 3)).copy()
        if c.ndim == 3 and c.shape[2] == 3:
            return c
        if c.ndim == 2 and c.shape == shape:
            return np.repeat(c[..., None], 3, axis=-1)
        if c.ndim == 1 and c.shape[0] == 3:
            return np.broadcast_to(c[None, None, :], (H, W, 3)).copy()
    if isinstance(color, (int, float, np.floating)):
        v = float(np.clip(color, 0.0, 1.0))
        return np.full((H, W, 3), v, np.float32)
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return np.broadcast_to(np.asarray(color, np.float32)[None, None, :], (H, W, 3)).copy()
    return np.ones((H, W, 3), np.float32)


def _sigma_px(radius_rel: float, ctx: GenerationContext) -> float:
    return max(0.5, float(radius_rel) * min(ctx.ss_width, ctx.ss_height))


def _offset_px(offset_rel: tuple[float, float], ctx: GenerationContext) -> tuple[float, float]:
    m = min(ctx.ss_width, ctx.ss_height)
    return float(offset_rel[1]) * m, float(offset_rel[0]) * m  # (dy, dx)


def apply_glow(
    mask: np.ndarray,
    color: np.ndarray,
    radius_rel: float,
    strength: float,
    ctx: GenerationContext,
) -> EffectLayer:
    """Bloom: размытая расширенная маска, светлее основного цвета.

    mode="constant": glow должен выходить за пределы маски.
    """
    m = _as_float_mask(mask)
    sigma = _sigma_px(radius_rel, ctx)
    blur = gaussian_filter(m, sigma=sigma, mode="constant", cval=0.0)
    glow_mask = np.clip(blur * float(strength), 0.0, 1.0).astype(np.float32)
    base = _as_color_array(color, m.shape)
    glow_color = np.clip(base * 1.2 + 0.2, 0.0, 1.0).astype(np.float32)
    return EffectLayer(mask=glow_mask, color=glow_color, blend="screen", opacity=1.0)


def apply_inner_glow(
    mask: np.ndarray,
    radius_rel: float,
    strength: float,
    ctx: GenerationContext,
) -> np.ndarray:
    """Свечение внутри маски (яркая кромка). Возвращает модифицированную маску.

    mode="constant": inner = m - blur(m) — только внутри маски.
    """
    m = _as_float_mask(mask)
    sigma = _sigma_px(radius_rel, ctx)
    blur = gaussian_filter(m, sigma=sigma, mode="constant", cval=0.0)
    inner = np.clip(m - blur, 0.0, 1.0) * float(strength)
    return np.clip(m + inner, 0.0, 1.0).astype(np.float32)


def apply_emboss(
    mask: np.ndarray,
    depth: float,
    light_dir: tuple[float, float, float],
    ctx: GenerationContext,
    sigma_rel: float = 0.003,
) -> tuple[EffectLayer, EffectLayer]:
    """Эмбоссинг: highlight (add) + shadow (multiply) по Sobel от сглаженной маски.

    ФИКС: mode="reflect" (не "constant"). Flat mask → h=1 везде → gradient=0 →
    hi=lo=0. "constant" добавлял фейковый emboss на краях холста, что ломало
    test_emboss_flat_mask_no_output и давало артефакты для слоёв, идущих до края.
    """
    m = _as_float_mask(mask)
    sigma = _sigma_px(sigma_rel, ctx)
    h = gaussian_filter(m, sigma=sigma, mode="reflect")
    gy, gx = np.gradient(h)
    L = np.asarray(light_dir, np.float32)
    L = L / (np.linalg.norm(L) + 1e-8)
    nx = -gx * float(depth)
    ny = -gy * float(depth)
    nz = np.ones_like(nx)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = (nx * L[0] + ny * L[1] + nz * L[2]) / nrm
    base = float(L[2])
    diff = shade - base
    hi = np.clip(diff, 0.0, 1.0) * m
    lo = np.clip(-diff, 0.0, 1.0) * m
    H, W = m.shape
    hi_color = np.ones((H, W, 3), np.float32)
    lo_color = np.zeros((H, W, 3), np.float32)
    return (
        EffectLayer(mask=hi.astype(np.float32), color=hi_color, blend="add", opacity=1.0),
        EffectLayer(mask=lo.astype(np.float32), color=lo_color, blend="multiply", opacity=1.0),
    )


def apply_shadow(
    mask: np.ndarray,
    offset_rel: tuple[float, float],
    radius_rel: float,
    opacity: float,
    ctx: GenerationContext,
) -> EffectLayer:
    """Тень: сдвинутая размытая маска, тёмный multiply.

    mode="constant": тень должна выходить за пределы маски.
    """
    m = _as_float_mask(mask)
    dy, dx = _offset_px(offset_rel, ctx)
    shifted = nd_shift(m, (dy, dx), order=1, mode="constant", cval=0.0, prefilter=False)
    sigma = _sigma_px(radius_rel, ctx)
    blurred = gaussian_filter(shifted, sigma=sigma, mode="constant", cval=0.0)
    shadow_mask = np.clip(blurred, 0.0, 1.0).astype(np.float32)
    H, W = m.shape
    shadow_color = np.zeros((H, W, 3), np.float32)
    return EffectLayer(
        mask=shadow_mask, color=shadow_color,
        blend="multiply", opacity=float(np.clip(opacity, 0.0, 1.0)),
    )


def apply_micro_echo(
    mask: np.ndarray,
    color: np.ndarray,
    offset_rel: tuple[float, float],
    radius_rel: float,
    fade: float,
    ctx: GenerationContext,
) -> EffectLayer:
    """Тонкий контур-двойник со сдвигом (банкнотный признак).

    mode="constant": эхо = blur(shift(m)) - m — только за пределами маски.
    """
    m = _as_float_mask(mask)
    dy, dx = _offset_px(offset_rel, ctx)
    shifted = nd_shift(m, (dy, dx), order=1, mode="constant", cval=0.0, prefilter=False)
    sigma = _sigma_px(radius_rel, ctx)
    blurred = gaussian_filter(shifted, sigma=sigma, mode="constant", cval=0.0)
    echo = np.clip(blurred - m, 0.0, 1.0) * float(fade)
    echo_color = _as_color_array(color, m.shape)
    return EffectLayer(
        mask=echo.astype(np.float32), color=echo_color, blend="normal", opacity=1.0,
    )


def build_effect_layers(
    mask: np.ndarray,
    color: Any,
    effects: list[EffectSpec] | None,
    ctx: GenerationContext,
) -> EffectResult:
    """Применить список эффектов. Неизвестные эффекты молча пропускаются."""
    m = _as_float_mask(mask)
    base_color = _as_color_array(color, m.shape)
    below: list[EffectLayer] = []
    above: list[EffectLayer] = []

    if effects:
        for eff in effects:
            kind = eff.kind
            p = eff.params or {}
            if kind == "glow":
                above.append(apply_glow(
                    m, base_color,
                    radius_rel=float(p.get("radius", 0.004)),
                    strength=float(p.get("strength", 0.4)),
                    ctx=ctx,
                ))
            elif kind == "inner_glow":
                m = apply_inner_glow(
                    m,
                    radius_rel=float(p.get("radius", 0.003)),
                    strength=float(p.get("strength", 0.3)),
                    ctx=ctx,
                )
            elif kind == "emboss":
                hi, lo = apply_emboss(
                    m,
                    depth=float(p.get("depth", 1.5)),
                    light_dir=tuple(p.get("light_dir", (-0.5, -0.5, 0.7))),
                    ctx=ctx,
                    sigma_rel=float(p.get("sigma_rel", 0.003)),
                )
                above.append(lo)
                above.append(hi)
            elif kind == "shadow":
                below.append(apply_shadow(
                    m,
                    offset_rel=tuple(p.get("offset", (0.003, 0.003))),
                    radius_rel=float(p.get("radius", 0.004)),
                    opacity=float(p.get("opacity", 0.5)),
                    ctx=ctx,
                ))
            elif kind == "micro_echo":
                below.append(apply_micro_echo(
                    m, base_color,
                    offset_rel=tuple(p.get("offset", (0.002, 0.002))),
                    radius_rel=float(p.get("radius", 0.002)),
                    fade=float(p.get("fade", 0.4)),
                    ctx=ctx,
                ))

    return EffectResult(main_mask=m, main_color=base_color, below=below, above=above)