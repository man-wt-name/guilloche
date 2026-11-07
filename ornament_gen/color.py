"""Градиенты и цветовое поле из reference-изображения. Без букв."""
from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

GRADIENTS: list[tuple[str, list[tuple[float, float, float]]]] = [
    ("blue-to-magenta-to-orange", [(0.10, 0.25, 0.65), (0.70, 0.20, 0.60), (0.95, 0.62, 0.20)]),
    ("teal-to-gold", [(0.05, 0.45, 0.50), (0.55, 0.70, 0.40), (0.90, 0.72, 0.25)]),
    ("violet-to-rose", [(0.35, 0.18, 0.65), (0.80, 0.30, 0.55), (0.95, 0.55, 0.45)]),
    ("emerald-to-sapphire", [(0.05, 0.50, 0.35), (0.10, 0.40, 0.60), (0.20, 0.25, 0.70)]),
    ("crimson-to-amber", [(0.60, 0.10, 0.25), (0.85, 0.35, 0.20), (0.95, 0.70, 0.25)]),
    ("indigo-to-turquoise", [(0.20, 0.15, 0.55), (0.15, 0.45, 0.70), (0.20, 0.75, 0.70)]),
    ("olive-to-copper", [(0.35, 0.45, 0.20), (0.65, 0.50, 0.20), (0.75, 0.38, 0.22)]),
]
PAPERS = ["#f6f3ea", "#f7f1ec", "#f2f5ef", "#faf6ec", "#f4f1f8", "#f0f5f6"]


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def gradient_field(W: int, H: int, stops: list, rng: random.Random) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / H
    ar = W / H
    mode = rng.choice(["linear", "radial", "angular", "waves"])
    cx, cy = rng.uniform(0.1, max(0.2, ar - 0.1)), rng.uniform(0.2, 0.8)
    if mode == "linear":
        th = rng.uniform(0.0, math.pi)
        s = xx * math.cos(th) + yy * math.sin(th)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    elif mode == "radial":
        s = np.clip(np.hypot(xx - cx, yy - cy) / 0.8, 0.0, 1.0)
    elif mode == "angular":
        s = np.arctan2(yy - cy, xx - cx) / (2.0 * math.pi) + 0.5
        s = np.abs(2.0 * s - 1.0)
    else:
        s = 0.5 + 0.5 * np.sin(
            2.0 * math.pi * (rng.uniform(0.6, 1.6) * xx + rng.uniform(0.6, 1.6) * yy)
            + rng.uniform(0.0, 6.28)
        )
    s = np.clip(s + 0.07 * np.sin(2.0 * math.pi * (2.3 * xx - 1.7 * yy) + rng.uniform(0.0, 6.28)),
                0.0, 1.0)
    pos = np.linspace(0.0, 1.0, len(stops))
    cols = np.array(stops, np.float32)
    return np.stack([np.interp(s, pos, cols[:, c]) for c in range(3)], -1).astype(np.float32)


@lru_cache(maxsize=16)
def _load_ref_cached(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1600, 1600))
    return img


def load_ref(path: Path) -> Image.Image:
    return _load_ref_cached(str(path))


def ref_color_field(
    img: Image.Image,
    W: int,
    H: int,
    rng: random.Random,
    fixed: bool,
    blur: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cover-кроп reference под aspect W/H, без растяжения. Возвращает (color, lum)."""
    iw, ih = img.size
    ar = W / H
    z = 1.0 if fixed else rng.uniform(0.6, 1.0)
    if iw / ih > ar:
        ch = ih * z
        cw = ch * ar
    else:
        cw = iw * z
        ch = cw / ar
    x0 = (iw - cw) / 2.0 if fixed else rng.uniform(0.0, iw - cw)
    y0 = (ih - ch) / 2.0 if fixed else rng.uniform(0.0, ih - ch)
    crop = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BICUBIC)
    if not fixed:
        if rng.random() < 0.5:
            crop = ImageOps.mirror(crop)
        if rng.random() < 0.5:
            crop = ImageOps.flip(crop)
    if blur > 0.0:
        crop = crop.filter(ImageFilter.GaussianBlur(min(W, H) * blur))
    arr = np.asarray(crop, np.float32) / 255.0
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]
    safe = arr * np.minimum(1.0, 0.72 / np.maximum(lum, 1e-3))
    return safe, lum[..., 0]


def collect_refs(arg: str | None) -> list[Path] | None:
    if not arg:
        return None
    p = Path(arg)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
    elif p.is_file():
        if p.suffix.lower() not in IMG_EXT:
            raise SystemExit(f"--ref: файл не похож на изображение: {arg}")
        files = [p]
    else:
        raise SystemExit(f"--ref: путь не найден: {arg}")
    if not files:
        raise SystemExit(f"--ref: в папке нет изображений ({', '.join(sorted(IMG_EXT))})")
    return files