"""Деформация сетки по яркости reference (ref-warp).

`warp_fields(lum, strength, smooth_rel)` — поле смещений (final-разрешение).
`apply_warp_to_curves(curves, dx, dy, ctx)` — применить поле к Curve в SS.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .core import Curve, GenerationContext


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.5:
        return a.astype(np.float32, copy=False)
    pad = int(3 * sigma) + 1
    p = np.pad(a, pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    k = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(p) * k, s=p.shape)
    return out[pad:-pad, pad:-pad].astype(np.float32)


def warp_fields(
    lum: np.ndarray, strength: float, smooth_rel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Поле смещений (final-разрешение) из яркости (H,W).

    p' = p + k·∇h. Сила нормирована на p99.5 гессиана:
    strength=1 сжимает шаг линий до ~30% в самом крутом контуре,
    strength>1.4 может складывать линии (пользователь предупреждён
    в docstring CLI). Смещение ограничено 15% от min(H,W).
    """
    H, W = lum.shape
    if strength == 0.0:
        return (
            np.zeros((H, W), np.float32),
            np.zeros((H, W), np.float32),
        )
    h = _smooth(lum, min(W, H) * smooth_rel)
    h = (h - h.min()) / (h.max() - h.min() + 1e-6)
    gy, gx = np.gradient(h)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    hxy = 0.5 * (gyx + gxy)
    tr = 0.5 * (gxx + gyy)
    rad = np.sqrt((0.5 * (gxx - gyy)) ** 2 + hxy ** 2)
    lam = np.maximum(np.abs(tr + rad), np.abs(tr - rad))
    k = strength * 0.7 / max(float(np.percentile(lam, 99.5)), 1e-8)
    dx, dy = k * gx, k * gy
    lim = 0.15 * min(W, H)
    sc = np.minimum(1.0, lim / np.maximum(np.hypot(dx, dy), 1e-6))
    return (dx * sc).astype(np.float32), (dy * sc).astype(np.float32)


def upscale_field(arr: np.ndarray, W_ss: int, H_ss: int) -> np.ndarray:
    if arr.shape == (H_ss, W_ss):
        return arr
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(img.resize((W_ss, H_ss), Image.BILINEAR), np.float32)


def _sample_bilinear(
    a: np.ndarray, x: np.ndarray, y: np.ndarray, W: int, H: int
) -> np.ndarray:
    xi = np.clip(x, 0.0, W - 1.001)
    yi = np.clip(y, 0.0, H - 1.001)
    x0 = xi.astype(np.int32)
    y0 = yi.astype(np.int32)
    tx = (xi - x0).astype(np.float32)
    ty = (yi - y0).astype(np.float32)
    v = (
        a[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + a[y0, x0 + 1] * tx * (1.0 - ty)
        + a[y0 + 1, x0] * (1.0 - tx) * ty
        + a[y0 + 1, x0 + 1] * tx * ty
    )
    return v.astype(np.float32)


def apply_warp_to_curves(
    curves: list[Curve],
    dx: np.ndarray,
    dy: np.ndarray,
    ctx: GenerationContext,
) -> list[Curve]:
    """Сместить точки всех кривых по полю (dx, dy) в final-разрешении.

    Возвращает новые Curve (не мутирует вход).
    """
    if not curves:
        return curves
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    dx_ss = upscale_field(dx, W_ss, H_ss)
    dy_ss = upscale_field(dy, W_ss, H_ss)

    out: list[Curve] = []
    for c in curves:
        nx = c.xs + _sample_bilinear(dx_ss, c.xs, c.ys, W_ss, H_ss)
        ny = c.ys + _sample_bilinear(dy_ss, c.xs, c.ys, W_ss, H_ss)
        out.append(Curve(
            xs=nx.astype(np.float32),
            ys=ny.astype(np.float32),
            width=c.width,
            zs=c.zs.copy() if c.zs is not None else None,
            closed=c.closed,
        ))
    return out