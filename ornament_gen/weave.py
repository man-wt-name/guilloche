"""Z-buffer weave (G3): построение и разрешение переплетения слоёв.

Идея: каждая кривая имеет z(t) вдоль параметра t. При растеризации каждый слой
получает (mask, z) в SS-разрешении. `resolve_weave` для каждого пикселя выбирает
слой с максимальным z; проигравший слой на этом пикселе скрывается.

P0-2: для SS > 20M EDT выполняется в уменьшенном разрешении (scale=2).
P2-3: shadow_strength клипается в [0, 1].
P2-4: валидация размеров масок/z-буферов в resolve_weave.
ФИКС: resolve_weave([]) возвращает ({}, {}), а не {} — контракт 2-tuple.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from .core import Curve, GenerationContext


def make_z_along_curve(
    n_points: int, preset: str | None, period: float, phase: float
) -> np.ndarray:
    """z-значения вдоль t ∈ [0, 1] для одной кривой."""
    if preset is None or preset == "none":
        return np.zeros(n_points, dtype=np.float32)
    if n_points <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    if preset == "sine":
        return np.sin(2.0 * np.pi * period * t + phase).astype(np.float32)
    if preset == "checkerboard":
        k = np.floor(t * period * 2.0)
        return ((k % 2.0) * 2.0 - 1.0).astype(np.float32)
    if preset == "saw":
        return (2.0 * ((t * period) % 1.0) - 1.0).astype(np.float32)
    if preset == "constant":
        return np.full(n_points, float(phase), dtype=np.float32)
    raise ValueError(f"unknown z_preset: {preset!r}")


def apply_z_to_curves(
    curves: list[Curve], preset: str | None, period: float, phase: float
) -> None:
    """Проставить curve.zs по выбранному пресету (мутирует список)."""
    for curve in curves:
        curve.zs = make_z_along_curve(len(curve.xs), preset, period, phase)


def rasterize_with_z(
    curves: list[Curve],
    ctx: GenerationContext,
    uniform_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Растеризовать кривые в (mask_ss uint8, z_ss float32) в SS-разрешении.

    z в каждом «толстом» пикселе берётся от ближайшего «тонкого» пикселя через EDT.
    uniform_width (в SS-пикселях) переопределяет медианную ширину всех кривых.

    P0-2: для SS > 20M EDT в уменьшенном разрешении (scale=2).
    """
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    if not curves:
        return (
            np.zeros((H_ss, W_ss), np.uint8),
            np.zeros((H_ss, W_ss), np.float32),
        )

    z_thin = np.full((H_ss, W_ss), -np.inf, dtype=np.float32)
    for curve in curves:
        if len(curve.xs) == 0:
            continue
        xi = np.clip(np.round(curve.xs).astype(np.int32), 0, W_ss - 1)
        yi = np.clip(np.round(curve.ys).astype(np.int32), 0, H_ss - 1)
        np.maximum.at(z_thin, (yi, xi), curve.zs)

    thin_mask = z_thin > -np.inf
    if not thin_mask.any():
        return (
            np.zeros((H_ss, W_ss), np.uint8),
            np.zeros((H_ss, W_ss), np.float32),
        )

    if thin_mask.all():
        return np.full((H_ss, W_ss), 255, np.uint8), z_thin.copy()

    widths = [int(round(c.width)) for c in curves if len(c.xs) > 0 and c.width >= 1]
    width = uniform_width if uniform_width is not None else int(np.median(widths))
    width = max(1, width)

    # P0-2: для больших SS — уменьшенное EDT (scale=2)
    if W_ss * H_ss > 20_000_000:
        scale = 2
        H_s = H_ss // scale
        W_s = W_ss // scale

        thin_small = thin_mask[:H_s * scale, :W_s * scale].reshape(
            H_s, scale, W_s, scale
        ).any(axis=(1, 3))
        z_thin_small = z_thin[:H_s * scale, :W_s * scale].reshape(
            H_s, scale, W_s, scale
        ).max(axis=(1, 3))
        del z_thin

        dist_s, (ny_s, nx_s) = distance_transform_edt(~thin_small, return_indices=True)
        dist_s = dist_s.astype(np.float32)
        ny_s = ny_s.astype(np.int32)
        nx_s = nx_s.astype(np.int32)
        del thin_small

        width_s = max(0.5, width / scale)
        thick_small = (dist_s <= width_s)
        del dist_s
        z_nearest_s = z_thin_small[ny_s, nx_s]
        del z_thin_small, ny_s, nx_s

        thick_full = np.repeat(np.repeat(thick_small, scale, axis=0), scale, axis=1)
        z_full = np.repeat(np.repeat(z_nearest_s, scale, axis=0), scale, axis=1)
        del thick_small, z_nearest_s

        if thick_full.shape != (H_ss, W_ss):
            pad_h = H_ss - thick_full.shape[0]
            pad_w = W_ss - thick_full.shape[1]
            thick_full = np.pad(thick_full, ((0, pad_h), (0, pad_w)), mode="edge")
            z_full = np.pad(z_full, ((0, pad_h), (0, pad_w)), mode="edge")

        z_thick = np.where(thick_full, z_full, 0.0).astype(np.float32)
        return thick_full.astype(np.uint8) * 255, z_thick

    # Маленькие SS: полный EDT (bit-for-bit для старых тестов)
    dist, (ny, nx) = distance_transform_edt(~thin_mask, return_indices=True)
    z_nearest = z_thin[ny, nx]
    thick_mask = dist <= width
    z_thick = np.where(thick_mask, z_nearest, 0.0).astype(np.float32)
    return thick_mask.astype(np.uint8) * 255, z_thick


def resolve_weave(
    layer_masks: list[tuple[Any, np.ndarray, np.ndarray]],
    shadow_strength: float = 0.0,
) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray]]:
    """Разрешить переплетение нескольких слоёв.

    P2-3: shadow_strength клипается в [0, 1].
    P2-4: все маски и z-буферы должны иметь одинаковую форму, иначе ValueError.
    ФИКС: пустой вход → ({}, {}), а не {} (контракт 2-tuple).
    """
    if not layer_masks:
        return {}, {}

    # P2-4: валидация размеров
    first_shape = layer_masks[0][1].shape
    for key, mask, z in layer_masks:
        if mask.shape != first_shape:
            raise ValueError(
                f"resolve_weave: mask shape mismatch for layer {key!r}: "
                f"{mask.shape} != {first_shape}"
            )
        if z.shape != first_shape:
            raise ValueError(
                f"resolve_weave: z shape mismatch for layer {key!r}: "
                f"{z.shape} != {first_shape}"
            )

    # P2-3: клип shadow_strength в [0, 1]
    shadow_strength = max(0.0, min(1.0, float(shadow_strength)))

    H_ss, W_ss = first_shape
    z_final = np.full((H_ss, W_ss), -np.inf, dtype=np.float32)
    id_final = np.full((H_ss, W_ss), -1, dtype=np.int32)

    for idx, (_key, mask, z) in enumerate(layer_masks):
        covered = mask > 0
        z_here = np.where(covered, z, -np.inf)
        update = covered & (z_here > z_final)
        z_final[update] = z_here[update]
        id_final[update] = idx

    visible: dict[Any, np.ndarray] = {}
    shadow: dict[Any, np.ndarray] = {}
    for idx, (key, mask, _z) in enumerate(layer_masks):
        covered = mask > 0
        wins = (id_final == idx) & covered
        visible[key] = wins.astype(np.uint8) * 255
        if shadow_strength > 0.0:
            sh = np.ones((H_ss, W_ss), dtype=np.float32)
            sh[covered & ~wins] = 1.0 - shadow_strength
            shadow[key] = sh
    return visible, shadow