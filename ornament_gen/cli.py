"""CLI: argparse, генерация батча, --dump-spec / --layers-spec, --n <= 50.

P2-1: --max-ram работает и при --layers-spec (spec авторитет по умолчанию,
       но явный --max-ram переопределяет quality).
P2-9: полный warning про игнорируемые флаги при --layers-spec.
P2-10: --seed дефолт None → --seed 0 при spec явно применяется.
P2-11: --out = файл → понятная ошибка (не FileExistsError traceback).
P2-12: _build_caption для reference → "reference:<filename>".
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

from PIL import Image

from .blend import BLEND_MODES
from .compose import make
from .color import collect_refs
from .core import BackgroundKind, GenerationContext, Layer, Quality
from .metadata import MetadataWriter
from .recipes import DEFAULT_RECIPE, RECIPES, random_layers
from .sizes import Orientation, SizeSpecError, resolve_size
from .spec import (
    SpecError,
    context_from_dict,
    layer_from_dict,
    load_spec_file,
    save_spec_file,
    weave_from_dict,
)

MAX_BATCH = 50
DEFAULT_MAX_RAM_MB = 4096.0


def _make_cache():
    from .cache import RasterizationCache
    return RasterizationCache(max_entries=32)


def _build_caption(spec: dict) -> str:
    """P2-12: правильная обработка source='reference' и 'gradient'."""
    names = [l.get("name", "?") for l in spec.get("layers", [])]
    blends = [l.get("blend", "normal") for l in spec.get("layers", [])]
    color_spec = spec.get("color", {})
    source = color_spec.get("source", "flat")
    if source == "reference":
        color = "reference:" + Path(color_spec.get("path", "?")).name
    elif source == "gradient":
        color = color_spec.get("name", "gradient")
    else:
        color = source
    weave = spec.get("weave", {}).get("mode", "none")
    return (
        f"guilloche-style ornamental pattern, layers: {', '.join(names) or 'none'}; "
        f"blends: {', '.join(blends) or 'none'}; color: {color}; weave: {weave}; "
        f"document quality"
    )


def _build_grid(paths: list[Path], cols: int, tw: int, th: int) -> Image.Image:
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, p in enumerate(paths):
        sheet.paste(Image.open(p).resize((tw, th), Image.LANCZOS),
                    ((i % cols) * tw, (i // cols) * th))
    return sheet


def _list_ignored_overrides(args: argparse.Namespace) -> list[str]:
    """P2-9: список флагов, которые игнорируются при --layers-spec."""
    out: list[str] = []
    if args.quality != "high":
        out.append("--quality")
    if args.background != "paper":
        out.append("--background")
    if args.noise != 0.0:
        out.append("--noise")
    if args.flat_color:
        out.append("--flat-color")
    if args.variable_thickness:
        out.append("--variable-thickness")
    if args.symmetry != "none":
        out.append("--symmetry")
    if args.weave is not None:
        out.append("--weave")
    if args.weave_shadow is not None:
        out.append("--weave-shadow")
    if args.weave_z_preset is not None:
        out.append("--weave-z-preset")
    if args.weave_z_period is not None:
        out.append("--weave-z-period")
    if args.weave_z_phase is not None:
        out.append("--weave-z-phase")
    if args.layer_opacity is not None:
        out.append("--layer-opacity")
    if args.layer_blend is not None:
        out.append("--layer-blend")
    if args.thickness_base is not None:
        out.append("--thickness-base")
    if args.thickness_amplitude is not None:
        out.append("--thickness-amplitude")
    if args.thickness_period is not None:
        out.append("--thickness-period")
    if args.thickness_phase is not None:
        out.append("--thickness-phase")
    if args.ref is not None:
        out.append("--ref")
    if args.ref_fixed:
        out.append("--ref-fixed")
    if args.ref_blur != 0.004:
        out.append("--ref-blur")
    if args.ref_warp != 0.0:
        out.append("--ref-warp")
    if args.ref_smooth != 0.03:
        out.append("--ref-smooth")
    return out


def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ornament_gen",
        description="High-quality guilloche-style ornament generator (document-grade).",
    )
    # --- batch ---
    ap.add_argument("--n", type=int, default=1,
                    help=f"количество изображений (1..{MAX_BATCH})")
    ap.add_argument("--out", default="ornament_out", help="директория результата")
    ap.add_argument("--seed", type=int, default=None,
                    help="базовый seed (по умолчанию 0)")
    ap.add_argument("--preview", type=int, default=0,
                    help="превью-сетка из первых N изображений (0=выкл)")

    # --- размер ---
    ap.add_argument("--size", default=None,
                    help="ISO размер: a4, a5, a6, letter, passport, id_card")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=300,
                    help="150..600 (по умолчанию 300)")
    ap.add_argument("--orientation", default="landscape",
                    choices=[o.value for o in Orientation])

    # --- качество ---
    ap.add_argument("--quality", default="high",
                    choices=[q.value for q in Quality])
    ap.add_argument("--noise", type=float, default=0.0,
                    help="стандартное отклонение шума (0=выкл, рекомендовано для документа)")
    ap.add_argument("--no-noise", dest="noise", action="store_const", const=0.0,
                    help="синоним --noise 0")

    # --- стиль ---
    ap.add_argument("--recipe", default=None,
                    help=f"пресет ({', '.join(RECIPES)})")
    ap.add_argument("--random-layers", type=int, default=None,
                    help="случайный рецепт из N слоёв (несовместим с --recipe)")
    ap.add_argument("--layers-spec", default=None,
                    help="JSON-спека для воспроизведения (генерирует 1 изображение)")
    ap.add_argument("--dump-spec", default=None,
                    help="сохранить JSON-спеку первого изображения в файл")

    # --- weave / effects ---
    ap.add_argument("--weave", default=None, choices=["none", "zbuffer"],
                    help="режим плетения (по умолчанию — из recipe)")
    ap.add_argument("--weave-shadow", type=float, default=None,
                    help="сила тени на пересечениях [0,1]")
    ap.add_argument("--weave-z-preset", default=None,
                    choices=["sine", "checkerboard", "saw", "constant"],
                    help="глобально переопределить z_preset слоёв без своего z_preset")
    ap.add_argument("--weave-z-period", type=float, default=None,
                    help="глобально переопределить z_period всех слоёв")
    ap.add_argument("--weave-z-phase", type=float, default=None,
                    help="глобально переопределить z_phase всех слоёв")

    # --- оверрайды слоёв ---
    ap.add_argument("--layer-opacity", type=float, default=None,
                    help="глобально переопределить opacity всех слоёв [0,1]")
    ap.add_argument("--layer-blend", default=None, choices=list(BLEND_MODES),
                    help="глобально переопределить blend всех слоёв")

    # --- толщина линий (требует --variable-thickness) ---
    ap.add_argument("--thickness-base", type=float, default=None,
                    help="множитель толщины (требует --variable-thickness)")
    ap.add_argument("--thickness-amplitude", type=float, default=None,
                    help="амплитуда модуляции толщины (требует --variable-thickness)")
    ap.add_argument("--thickness-period", type=float, default=None,
                    help="период модуляции толщины (требует --variable-thickness)")
    ap.add_argument("--thickness-phase", type=float, default=None,
                    help="фаза модуляции толщины (требует --variable-thickness)")

    # --- цвет / фон ---
    ap.add_argument("--ref", default=None, help="изображение или папка-референс")
    ap.add_argument("--ref-fixed", action="store_true",
                    help="без случайного кропа/отражений референса")
    ap.add_argument("--ref-blur", type=float, default=0.004,
                    help="сглаживание цветового поля (доля от размера)")
    ap.add_argument("--ref-warp", type=float, default=0.0,
                    help="деформировать сетку по яркости reference: "
                         ">0 сгущение в ярких, <0 в тёмных; 0.3-1.2 разумно; "
                         ">1.4 сильный эффект, может складывать линии; 0 = выкл")
    ap.add_argument("--ref-smooth", type=float, default=0.03,
                    help="σ сглаживания яркости для ref-warp (доля от размера)")
    ap.add_argument("--background", default="paper",
                    choices=[b.value for b in BackgroundKind],
                    help="paper|white|black|none (none = прозрачный RGBA)")
    ap.add_argument("--flat-color", action="store_true",
                    help="векторизационный режим: однотонная чёрная линия без градиента")
    ap.add_argument("--variable-thickness", action="store_true",
                    help="переменная толщина линий вдоль параметра")
    ap.add_argument("--symmetry", default="none",
                    help="none|vertical|horizontal|central|radialN (например, radial6)")

    # --- RAM / cache ---
    ap.add_argument("--max-ram", type=float, default=None,
                    help=f"лимит пиковой RAM в MB; при превышении quality понижается "
                         f"(<=0 отключает проверку, по умолчанию {DEFAULT_MAX_RAM_MB:.0f})")
    ap.add_argument("--no-cache", action="store_true",
                    help="отключить кэш curves (по умолчанию включён)")

    # --- вывод ---
    ap.add_argument("--compress-level", type=int, default=0,
                    help="PNG compress_level 0..9 (0=без сжатия, по умолчанию)")
    return ap


def _load_spec_layers(spec: dict) -> list[Layer]:
    raw_layers = spec.get("layers", []) or []
    if not raw_layers:
        raise SpecError("spec.layers пуст")
    return [layer_from_dict(d) for d in raw_layers]


def _apply_layer_overrides(layers: list[Layer], args: argparse.Namespace) -> None:
    """Глобальные оверрайды слоёв. Мутирует layers in-place."""
    if args.weave_z_period is not None:
        for L in layers:
            L.z_period = float(args.weave_z_period)
    if args.weave_z_phase is not None:
        for L in layers:
            L.z_phase = float(args.weave_z_phase)
    if args.layer_opacity is not None:
        for L in layers:
            L.opacity = float(args.layer_opacity)
    if args.layer_blend is not None:
        for L in layers:
            L.blend = str(args.layer_blend)
    if args.variable_thickness:
        for L in layers:
            L.thickness = dataclasses.replace(
                L.thickness,
                kind="modulated",
                base=float(args.thickness_base)
                if args.thickness_base is not None else L.thickness.base,
                amplitude=float(args.thickness_amplitude)
                if args.thickness_amplitude is not None else L.thickness.amplitude,
                period=float(args.thickness_period)
                if args.thickness_period is not None else L.thickness.period,
                phase=float(args.thickness_phase)
                if args.thickness_phase is not None else L.thickness.phase,
            )


def main(argv: list[str] | None = None) -> int:
    ap = _make_parser()
    args = ap.parse_args(argv)

    if not (1 <= args.n <= MAX_BATCH):
        ap.error(f"--n должен быть в диапазоне 1..{MAX_BATCH}, получено {args.n}")
    if not (0 <= args.compress_level <= 9):
        ap.error("--compress-level должен быть 0..9")
    if args.recipe and args.random_layers:
        ap.error("--recipe и --random-layers несовместимы")
    if args.layers_spec and (args.recipe or args.random_layers):
        ap.error("--layers-spec несовместим с --recipe / --random-layers")
    if args.layer_opacity is not None and not (0.0 <= args.layer_opacity <= 1.0):
        ap.error("--layer-opacity должен быть в [0, 1]")

    if args.random_layers is not None and args.random_layers < 1:
        ap.error(f"--random-layers должен быть >= 1, получено {args.random_layers}")

    thickness_flags = (
        args.thickness_base, args.thickness_amplitude,
        args.thickness_period, args.thickness_phase,
    )
    if any(x is not None for x in thickness_flags) and not args.variable_thickness:
        ap.error("--thickness-* требуют --variable-thickness")

    refs = collect_refs(args.ref)

    if args.ref_warp != 0.0 and not refs and not args.layers_spec:
        ap.error("--ref-warp требует --ref")

    # P2-9: полный warning при --layers-spec
    if args.layers_spec:
        ignored = _list_ignored_overrides(args)
        if ignored:
            print(
                f"warning: при --layers-spec игнорируются: {', '.join(ignored)} "
                f"(spec авторитет)",
                file=sys.stderr,
            )

    cache = None if args.no_cache else _make_cache()

    # P2-10: --seed None (не передан) vs 0 (передан)
    seed = args.seed if args.seed is not None else 0

    if args.layers_spec:
        try:
            spec = load_spec_file(Path(args.layers_spec))
            ctx = context_from_dict(spec)
            layers = _load_spec_layers(spec)
            weave = weave_from_dict(spec.get("weave"))
            color_spec = spec.get("color")
            spec_ref_warp = float(spec.get("ref_warp", 0.0))
            spec_ref_smooth = float(spec.get("ref_smooth", 0.03))
        except (SpecError, KeyError, ValueError) as exc:
            ap.error(f"не удалось прочитать --layers-spec: {exc}")
        # P2-10: явный --seed (включая 0) переопределяет
        if args.seed is not None:
            ctx = dataclasses.replace(ctx, seed=args.seed)
        # P2-1: --max-ram при spec
        max_ram = args.max_ram if args.max_ram is not None else None
        if max_ram is not None and max_ram > 0:
            from .ram import auto_downgrade_quality
            ref_warp_active = spec_ref_warp != 0.0
            weave_active = weave.mode == "zbuffer"
            new_ctx, msg = auto_downgrade_quality(
                ctx, ref_warp_active, weave_active, max_ram,
            )
            if msg:
                print(f"warning: {msg}", file=sys.stderr)
            ctx = new_ctx
        n = 1
        reproduce = True
    else:
        try:
            size = resolve_size(
                width=args.width, height=args.height,
                size=args.size, dpi=args.dpi,
                orientation=Orientation(args.orientation),
            )
        except SizeSpecError as exc:
            ap.error(str(exc))

        try:
            ctx = GenerationContext(
                width=size.width,
                height=size.height,
                seed=seed,
                quality=Quality(args.quality),
                background=BackgroundKind(args.background),
                noise=0.0 if args.no_noise else args.noise,
                flat_color=args.flat_color,
                variable_thickness=args.variable_thickness,
                symmetry=args.symmetry,
            )
        except ValueError as exc:
            ap.error(str(exc))

        if args.random_layers:
            recipe = random_layers(args.random_layers, ctx.seed)
        else:
            name = args.recipe or DEFAULT_RECIPE
            if name not in RECIPES:
                ap.error(f"неизвестный recipe: {name!r} (доступно: {', '.join(RECIPES)})")
            recipe = RECIPES[name]()

        try:
            layers = [layer_from_dict(d) for d in recipe["layers"]]
            weave = weave_from_dict(recipe.get("weave"))
        except SpecError as exc:
            ap.error(f"recipe повреждён: {exc}")

        if args.weave is not None:
            weave.mode = args.weave
        if args.weave_shadow is not None:
            weave.shadow = float(args.weave_shadow)

        if args.weave_z_preset:
            for L in layers:
                if L.z_preset is None:
                    L.z_preset = args.weave_z_preset

        _apply_layer_overrides(layers, args)

        n = args.n
        color_spec = None
        reproduce = False
        spec_ref_warp = float(args.ref_warp)
        spec_ref_smooth = float(args.ref_smooth)

        from .ram import auto_downgrade_quality
        ref_warp_active = float(args.ref_warp) != 0.0
        weave_active = weave.mode == "zbuffer"
        # P2-1: --max-ram дефолт None → используем DEFAULT_MAX_RAM_MB
        max_ram = args.max_ram if args.max_ram is not None else DEFAULT_MAX_RAM_MB
        if max_ram > 0:
            new_ctx, msg = auto_downgrade_quality(
                ctx, ref_warp_active, weave_active, max_ram,
            )
            if msg:
                print(f"warning: {msg}", file=sys.stderr)
            ctx = new_ctx

    # P2-11: --out = файл → понятная ошибка
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        ap.error(f"--out: {out} существует и не является директорией")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        ap.error(f"--out: не удалось создать директорию {out}: {exc}")

    metadata_path = out / "metadata.jsonl"
    paths: list[Path] = []

    with MetadataWriter(metadata_path) as meta:
        for i in range(n):
            if reproduce:
                img_ctx = ctx
            else:
                img_ctx = dataclasses.replace(ctx, seed=ctx.seed * 1_000_003 + i)
            res = make(
                img_ctx, layers,
                refs=refs,
                ref_fixed=args.ref_fixed,
                ref_blur=args.ref_blur,
                weave_config=weave,
                color_spec=color_spec,
                ref_warp=spec_ref_warp,
                ref_smooth=spec_ref_smooth,
                use_cache=cache,
            )
            name = f"{i:05d}.png"
            img_path = out / name
            res.image.save(img_path, compress_level=args.compress_level)
            meta.write({**res.spec, "file_name": name, "caption": _build_caption(res.spec)})
            paths.append(img_path)
            if i == 0 and args.dump_spec:
                try:
                    save_spec_file(Path(args.dump_spec), res.spec)
                except SpecError as exc:
                    ap.error(f"не удалось сохранить --dump-spec: {exc}")

    if args.preview:
        k = min(args.preview, len(paths))
        if k > 0:
            tw = max(1, ctx.width // 2)
            th = max(1, ctx.height // 2)
            _build_grid(paths[:k], cols=4, tw=tw, th=th).save(out / "preview.png")

    if cache is not None:
        print(f"cache: hits={cache.hits} misses={cache.misses} size={len(cache)}")

    print(f"готово: {n} изображений {ctx.width}x{ctx.height} -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())