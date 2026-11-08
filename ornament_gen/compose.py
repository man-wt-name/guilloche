"""Пайплайн сборки гильоша: curves → mask_ss → weave → effects → blend → final.

P0-3: после resolve_weave освобождаем z_ss и mask_ss в layer_data.
P1-2: _expand_with_thickness добавляет замыкающий сегмент для closed=True.
ФИКС: _get_color_field для reference из color_spec использует fixed/blur из spec
      (а не из аргументов make) — воспроизводимость через --layers-spec.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .blend import blend
from .color import GRADIENTS, PAPERS, gradient_field, hex_rgb, load_ref, ref_color_field
from .core import BackgroundKind, Curve, GenerationContext, Layer
from .effects import EffectLayer, build_effect_layers
from .metadata import _clean
from .nets import NETS
from .patterns import PATTERNS
from .rasterize import add_noise
from .symmetry import apply_symmetry
from .warp import apply_warp_to_curves, warp_fields
from .weave import apply_z_to_curves, rasterize_with_z, resolve_weave

MAX_SS_PIXELS = 120_000_000


@dataclass
class WeaveConfig:
    mode: str = "none"
    shadow: float = 0.0


@dataclass
class ComposeResult:
    image: Image.Image
    spec: dict[str, Any] = field(default_factory=dict)


def _rasterize_plain(curves: list[Curve], ctx: GenerationContext) -> np.ndarray:
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    if not curves:
        return np.zeros((H_ss, W_ss), np.uint8)
    img = Image.new("L", (W_ss, H_ss), 0)
    d = ImageDraw.Draw(img)
    for c in curves:
        if len(c.xs) == 0:
            continue
        pts = np.stack([c.xs, c.ys], axis=1)
        if c.closed:
            pts = np.vstack([pts, pts[:1]])
        d.line(
            pts.flatten().tolist(),
            fill=255,
            width=int(max(1, round(float(c.width)))),
            joint="curve",
        )
    return np.asarray(img, dtype=np.uint8)


def _downsample_mask(mask_ss: np.ndarray, W_f: int, H_f: int) -> np.ndarray:
    if mask_ss.dtype == np.uint8:
        img = Image.fromarray(mask_ss, mode="L")
        return np.asarray(img.resize((W_f, H_f), Image.LANCZOS), np.float32) / 255.0
    a = np.clip(mask_ss, 0.0, 1.0).astype(np.float32)
    img = Image.fromarray(a, mode="F")
    return np.clip(np.asarray(img.resize((W_f, H_f), Image.BILINEAR), np.float32), 0.0, 1.0)


def _downsample_rgb(arr_ss: np.ndarray, W_f: int, H_f: int) -> np.ndarray:
    if arr_ss.shape[0] == H_f and arr_ss.shape[1] == W_f:
        return arr_ss.astype(np.float32, copy=False)
    out = np.empty((H_f, W_f, 3), np.float32)
    for c in range(3):
        ch = np.clip(arr_ss[..., c], 0.0, 1.0).astype(np.float32)
        img = Image.fromarray(ch, mode="F")
        out[..., c] = np.asarray(img.resize((W_f, H_f), Image.BILINEAR), np.float32)
    return out


def _expand_with_thickness(curves: list[Curve], layer: Layer) -> list[Curve]:
    """P1-2: для closed=True добавляется замыкающий сегмент от последней точки к первой."""
    ts = layer.thickness
    if ts.kind != "modulated" or not curves:
        return curves
    n_seg = 16
    base = max(1e-3, float(ts.base))
    amp = float(ts.amplitude)
    period = float(ts.period)
    phase = float(ts.phase)
    out: list[Curve] = []
    for c in curves:
        n = len(c.xs)
        if n < 2:
            out.append(c)
            continue
        idx = np.linspace(0, n - 1, n_seg + 1).astype(np.int32)
        for k in range(n_seg):
            i0, i1 = int(idx[k]), int(idx[k + 1])
            if i1 <= i0:
                continue
            t_mid = (k + 0.5) / n_seg
            mult = 1.0 + amp * float(np.sin(2.0 * np.pi * period * t_mid + phase))
            new_w = max(1.0, float(c.width) * base * mult)
            out.append(Curve(
                xs=c.xs[i0:i1 + 1].copy(),
                ys=c.ys[i0:i1 + 1].copy(),
                width=new_w,
                zs=c.zs[i0:i1 + 1].copy() if c.zs is not None else None,
                closed=False,
            ))
        if c.closed:
            t_mid = 1.0
            mult = 1.0 + amp * float(np.sin(2.0 * np.pi * period * t_mid + phase))
            new_w = max(1.0, float(c.width) * base * mult)
            out.append(Curve(
                xs=np.array([c.xs[-1], c.xs[0]], np.float32),
                ys=np.array([c.ys[-1], c.ys[0]], np.float32),
                width=new_w,
                zs=np.array([c.zs[-1], c.zs[0]], np.float32) if c.zs is not None else None,
                closed=False,
            ))
    return out


def _get_color_field(
    ctx: GenerationContext,
    refs: list | None,
    ref_fixed: bool,
    ref_blur: float,
    color_spec: dict | None,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """(color RGB, color_spec_dict, lum_or_None)."""
    if ctx.flat_color:
        black = np.zeros((ctx.height, ctx.width, 3), np.float32)
        return black, {"source": "flat", "color": [0.0, 0.0, 0.0]}, None

    if color_spec:
        src = color_spec.get("source")
        if src == "reference" and color_spec.get("path"):
            rng = random.Random(ctx.seed * 999_983 + 3)
            img = load_ref(str(color_spec["path"]))
            # ФИКС: fixed/blur из color_spec (spec авторитет), fallback на аргументы.
            # Без этого --layers-spec теряет --ref-fixed (random crop при reproduce).
            cs_fixed = bool(color_spec.get("fixed", ref_fixed))
            cs_blur = float(color_spec.get("blur", ref_blur))
            color, lum = ref_color_field(img, ctx.width, ctx.height, rng, cs_fixed, cs_blur)
            return color, dict(color_spec), lum
        if src == "gradient" and color_spec.get("stops"):
            rng = random.Random(ctx.seed * 999_983 + 5)
            color = gradient_field(ctx.width, ctx.height, color_spec["stops"], rng)
            return color, dict(color_spec), None
        if src == "flat":
            black = np.zeros((ctx.height, ctx.width, 3), np.float32)
            return black, dict(color_spec), None

    if refs:
        rng = random.Random(ctx.seed * 999_983 + 3)
        path = rng.choice(refs)
        img = load_ref(str(path))
        color, lum = ref_color_field(img, ctx.width, ctx.height, rng, ref_fixed, ref_blur)
        return color, {
            "source": "reference",
            "path": str(path),
            "fixed": bool(ref_fixed),
            "blur": float(ref_blur),
        }, lum

    rng = random.Random(ctx.seed * 999_983 + 5)
    g = rng.choice(GRADIENTS)
    color = gradient_field(ctx.width, ctx.height, g[1], rng)
    return color, {"source": "gradient", "name": g[0], "stops": g[1]}, None


def _paper_rgb(ctx: GenerationContext) -> np.ndarray:
    if ctx.background == BackgroundKind.WHITE:
        return np.ones((ctx.height, ctx.width, 3), np.float32)
    if ctx.background == BackgroundKind.BLACK:
        return np.zeros((ctx.height, ctx.width, 3), np.float32)
    rng = random.Random(ctx.seed * 999_983 + 11)
    return np.broadcast_to(
        hex_rgb(rng.choice(PAPERS))[None, None, :],
        (ctx.height, ctx.width, 3),
    ).copy()


def make(
    ctx: GenerationContext,
    layers: list[Layer],
    refs: list | None = None,
    ref_fixed: bool = False,
    ref_blur: float = 0.004,
    weave_config: WeaveConfig | None = None,
    color_spec: dict | None = None,
    ref_warp: float = 0.0,
    ref_smooth: float = 0.03,
    use_cache: object | None = None,
) -> ComposeResult:
    W_ss, H_ss = ctx.ss_width, ctx.ss_height
    if W_ss * H_ss > MAX_SS_PIXELS:
        raise ValueError(
            f"SS-разрешение {W_ss}x{H_ss} слишком велико "
            f"(лимит {MAX_SS_PIXELS} пикселей). Уменьшите --quality."
        )
    if weave_config is None:
        weave_config = WeaveConfig()

    color, color_desc, ref_lum = _get_color_field(
        ctx, refs, ref_fixed, ref_blur, color_spec
    )

    dx_w: np.ndarray | None = None
    dy_w: np.ndarray | None = None
    if ref_warp != 0.0 and ref_lum is not None:
        dx_w, dy_w = warp_fields(ref_lum, float(ref_warp), float(ref_smooth))

    layer_data: list[dict[str, Any]] = []
    for idx, layer in enumerate(layers):
        rng = ctx.layer_rng(idx)
        if layer.kind == "net":
            if layer.name not in NETS:
                raise ValueError(f"unknown net: {layer.name!r}")
            gen = NETS[layer.name]
        elif layer.kind == "pattern":
            if layer.name not in PATTERNS:
                raise ValueError(f"unknown pattern: {layer.name!r}")
            gen = PATTERNS[layer.name]
        else:
            raise ValueError(f"unknown layer kind: {layer.kind!r}")

        curves: list[Curve] | None = None
        cache_key: str | None = None
        if use_cache is not None:
            from .cache import _make_key, restore_curves, snapshot_curves
            cache_key = _make_key(layer, ctx, idx)
            snap = use_cache.get(cache_key)
            if snap is not None:
                curves = restore_curves(snap)
        if curves is None:
            curves = gen(ctx, rng, layer.params)
            if use_cache is not None and cache_key is not None:
                use_cache.put(cache_key, snapshot_curves(curves))

        if dx_w is not None and dy_w is not None:
            curves = apply_warp_to_curves(curves, dx_w, dy_w, ctx)

        if ctx.variable_thickness:
            curves = _expand_with_thickness(curves, layer)

        if weave_config.mode == "zbuffer":
            apply_z_to_curves(curves, layer.z_preset, layer.z_period, layer.z_phase)
            mask_ss, z_ss = rasterize_with_z(curves, ctx)
        else:
            mask_ss = _rasterize_plain(curves, ctx)
            z_ss = None

        layer_data.append({
            "index": idx,
            "layer": layer,
            "mask_ss": mask_ss,
            "z_ss": z_ss,
            "visible_mask_ss": mask_ss,
            "weave_shadow_ss": None,
        })

    if weave_config.mode == "zbuffer" and layer_data:
        visible, shadow_maps = resolve_weave(
            [(d["index"], d["mask_ss"], d["z_ss"]) for d in layer_data],
            shadow_strength=float(weave_config.shadow),
        )
        for d in layer_data:
            d["visible_mask_ss"] = visible.get(d["index"], d["mask_ss"])
            d["weave_shadow_ss"] = shadow_maps.get(d["index"])
            d["z_ss"] = None
            if d["visible_mask_ss"] is not d["mask_ss"]:
                d["mask_ss"] = None

    H_f, W_f = ctx.height, ctx.width

    if ctx.background == BackgroundKind.NONE:
        canvas = np.zeros((H_f, W_f, 3), np.float32)
        alpha_acc = np.zeros((H_f, W_f), np.float32)
        for d in layer_data:
            layer: Layer = d["layer"]
            mask_ss = d["visible_mask_ss"]
            eff = build_effect_layers(mask_ss, color, layer.effects, ctx)

            for e_layer in eff.below:
                m_f = _downsample_mask(e_layer.mask, W_f, H_f)
                c_f = _downsample_rgb(e_layer.color, W_f, H_f)
                blended = blend(canvas, c_f, e_layer.blend, e_layer.opacity)
                canvas = canvas * (1.0 - m_f[..., None]) + blended * m_f[..., None]

            main_mask_f = _downsample_mask(eff.main_mask, W_f, H_f)
            main_color_f = color.copy()
            if d["weave_shadow_ss"] is not None:
                sh_f = _downsample_mask(d["weave_shadow_ss"], W_f, H_f)
                main_color_f = main_color_f * sh_f[..., None]
            blended = blend(canvas, main_color_f, layer.blend, layer.opacity)
            canvas = canvas * (1.0 - main_mask_f[..., None]) + blended * main_mask_f[..., None]
            alpha_acc = np.maximum(alpha_acc, main_mask_f)

            for e_layer in eff.above:
                m_f = _downsample_mask(e_layer.mask, W_f, H_f)
                c_f = _downsample_rgb(e_layer.color, W_f, H_f)
                blended = blend(canvas, c_f, e_layer.blend, e_layer.opacity)
                canvas = canvas * (1.0 - m_f[..., None]) + blended * m_f[..., None]

            del eff

        if ctx.symmetry != "none":
            canvas = apply_symmetry(canvas, ctx.symmetry)
            alpha_acc = apply_symmetry(alpha_acc, ctx.symmetry)

        rgb = np.clip(canvas, 0.0, 1.0)
        if ctx.noise > 0:
            nrng = np.random.default_rng(ctx.seed * 1000 + 1)
            rgb = np.clip(add_noise(rgb, nrng, ctx.noise), 0.0, 1.0)
        alpha = np.clip(alpha_acc, 0.0, 1.0)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        out_img = Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), mode="RGBA")
    else:
        canvas = _paper_rgb(ctx)
        for d in layer_data:
            layer: Layer = d["layer"]
            mask_ss = d["visible_mask_ss"]
            eff = build_effect_layers(mask_ss, color, layer.effects, ctx)

            for e_layer in eff.below:
                m_f = _downsample_mask(e_layer.mask, W_f, H_f)
                c_f = _downsample_rgb(e_layer.color, W_f, H_f)
                blended = blend(canvas, c_f, e_layer.blend, e_layer.opacity)
                canvas = canvas * (1.0 - m_f[..., None]) + blended * m_f[..., None]

            main_mask_f = _downsample_mask(eff.main_mask, W_f, H_f)
            main_color_f = color.copy()
            if d["weave_shadow_ss"] is not None:
                sh_f = _downsample_mask(d["weave_shadow_ss"], W_f, H_f)
                main_color_f = main_color_f * sh_f[..., None]
            blended = blend(canvas, main_color_f, layer.blend, layer.opacity)
            canvas = canvas * (1.0 - main_mask_f[..., None]) + blended * main_mask_f[..., None]

            for e_layer in eff.above:
                m_f = _downsample_mask(e_layer.mask, W_f, H_f)
                c_f = _downsample_rgb(e_layer.color, W_f, H_f)
                blended = blend(canvas, c_f, e_layer.blend, e_layer.opacity)
                canvas = canvas * (1.0 - m_f[..., None]) + blended * m_f[..., None]

            del eff

        if ctx.symmetry != "none":
            canvas = apply_symmetry(canvas, ctx.symmetry)

        combined = np.clip(canvas, 0.0, 1.0)
        if ctx.noise > 0:
            nrng = np.random.default_rng(ctx.seed * 1000 + 1)
            combined = add_noise(combined, nrng, ctx.noise)
        out_img = Image.fromarray(
            (np.clip(combined, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB"
        )

    spec = {
        "width": ctx.width,
        "height": ctx.height,
        "seed": ctx.seed,
        "quality": ctx.quality.value,
        "background": ctx.background.value,
        "noise": ctx.noise,
        "flat_color": ctx.flat_color,
        "variable_thickness": ctx.variable_thickness,
        "symmetry": ctx.symmetry,
        "ref_warp": float(ref_warp),
        "ref_smooth": float(ref_smooth),
        "color": color_desc,
        "weave": {"mode": weave_config.mode, "shadow": weave_config.shadow},
        "layers": [{
            "kind": L.kind,
            "name": L.name,
            "params": L.params,
            "opacity": L.opacity,
            "blend": L.blend,
            "weave_group": L.weave_group,
            "z_preset": L.z_preset,
            "z_period": L.z_period,
            "z_phase": L.z_phase,
            "thickness": {
                "kind": L.thickness.kind,
                "base": L.thickness.base,
                "amplitude": L.thickness.amplitude,
                "period": L.thickness.period,
                "phase": L.thickness.phase,
            },
            "effects": [{"kind": e.kind, "params": e.params} for e in L.effects],
        } for L in layers],
    }
    return ComposeResult(image=out_img, spec=_clean(spec))