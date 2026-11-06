#!/usr/bin/env python3
"""Генератор декоративных сеток в стиле гильоша для обучения diffusion-моделей.

Один файл, без внешних модулей проекта (нужны numpy и Pillow).

Что умеет
  * только сетка: деформированная, «линза», полярная паутина
  * цвет линий: встроенные градиенты ИЛИ цветовое поле из вашего изображения (--ref)
  * неквадратный размер (по умолчанию 512x1024: ширина x высота)
  * буквы:
      --letters mesh    формируются самой сеткой: внутри букв линии сдвинуты на
                        полпериода, толще и темнее; края слегка «выпуклые» (по умолчанию)
      --letters emboss  классический рельеф: подсветка, тень, заливка

Цвета из изображения (--ref)
  Подойдёт картинка, залитая цветами, переходами, градиентами. Укажите файл или
  папку (тогда для каждой сетки берётся случайный файл). Картинка растягивается
  на холст, цвет каждой точки линии берётся из картинки. Для разнообразия по
  умолчанию применяются случайный кроп и отражения (--ref-fixed отключает).
  Слишком светлые цвета автоматически затемняются, чтобы линии были видны.

Примеры
    python ornament_gen.py --n 20000 --out ornament_ds --seed 0 --preview 8
    python ornament_gen.py --n 5000 --ref my_gradient.png --out ds_ref
    python ornament_gen.py --n 5000 --ref ./gradients/ --ref-fixed --out ds_ref2
    python ornament_gen.py --n 500 --words "LUMEN,NOVA" --letters emboss --out brand_ds
"""
import argparse
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

SS = 2  # суперсэмплинг
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

WORDS = ["AETERNA", "VALOR", "LUMEN", "ARBOR", "SOLARIS", "NOVA", "FORTUNA",
         "ORIGO", "ATLAS", "HARMONIA", "VERITAS", "AURORA", "CALIDA", "MERIDIAN"]

GRADIENTS = [
    ("blue-to-magenta-to-orange", [(0.10, 0.25, 0.65), (0.70, 0.20, 0.60), (0.95, 0.62, 0.20)]),
    ("teal-to-gold", [(0.05, 0.45, 0.50), (0.55, 0.70, 0.40), (0.90, 0.72, 0.25)]),
    ("violet-to-rose", [(0.35, 0.18, 0.65), (0.80, 0.30, 0.55), (0.95, 0.55, 0.45)]),
    ("emerald-to-sapphire", [(0.05, 0.50, 0.35), (0.10, 0.40, 0.60), (0.20, 0.25, 0.70)]),
    ("crimson-to-amber", [(0.60, 0.10, 0.25), (0.85, 0.35, 0.20), (0.95, 0.70, 0.25)]),
    ("indigo-to-turquoise", [(0.20, 0.15, 0.55), (0.15, 0.45, 0.70), (0.20, 0.75, 0.70)]),
    ("olive-to-copper", [(0.35, 0.45, 0.20), (0.65, 0.50, 0.20), (0.75, 0.38, 0.22)]),
]
PAPERS = ["#f6f3ea", "#f7f1ec", "#f2f5ef", "#faf6ec", "#f4f1f8", "#f0f5f6"]


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def _line(d, xs, ys, color, w, closed=False):
    pts = np.stack([xs, ys], 1)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    d.line(pts.flatten().tolist(), fill=color, width=w, joint="curve")


# ---------- сетки: рисуем белым по чёрному в маску L ----------
# Параметры берутся из seed, а phase (доля шага) сдвигает линии,
# поэтому один узор можно нарисовать дважды со сдвигом (так получаются буквы).

def _spacing(W, H, rng):
    return min(W, H) * rng.uniform(0.016, 0.036)


def mesh_warp(d, W, H, w, seed, phase=0.0):
    rng = random.Random(seed)
    sp = _spacing(W, H, rng)
    m = min(W, H)
    nu, nv = int(1.2 * W / sp) + 2, int(1.2 * H / sp) + 2
    a = rng.uniform(0.02, 0.06) * m
    f = [rng.uniform(1.0, 4.5) for _ in range(4)]
    p = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
    t = np.linspace(-0.1, 1.1, 1400)

    def warp(x, y):
        dx = a * np.sin(2 * np.pi * f[0] * y / m + p[0]) + 0.6 * a * np.sin(2 * np.pi * f[1] * (x + y) / m + p[1])
        dy = a * np.sin(2 * np.pi * f[2] * x / m + p[2]) + 0.6 * a * np.sin(2 * np.pi * f[3] * (x - y) / m + p[3])
        return x + dx, y + dy

    for i in range(nu):
        X, Y = warp(np.full_like(t, -0.1 * W + (i + phase) * sp), t * H)
        _line(d, X, Y, 255, w)
    for j in range(nv):
        X, Y = warp(t * W, np.full_like(t, -0.1 * H + (j + phase) * sp))
        _line(d, X, Y, 255, w)


def mesh_lens(d, W, H, w, seed, phase=0.0):
    rng = random.Random(seed)
    sp = _spacing(W, H, rng)
    nu, nv = int(1.2 * W / sp) + 2, int(1.2 * H / sp) + 2
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.3, 0.7)
    sig = min(W, H) * rng.uniform(0.25, 0.5)
    k = rng.choice([rng.uniform(0.8, 1.6), -rng.uniform(0.3, 0.55)])
    t = np.linspace(-0.1, 1.1, 1400)

    def lens(x, y):
        dx, dy = x - cx, y - cy
        g = 1 + k * np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))
        return cx + dx * g, cy + dy * g

    for i in range(nu):
        X, Y = lens(np.full_like(t, -0.1 * W + (i + phase) * sp), t * H)
        _line(d, X, Y, 255, w)
    for j in range(nv):
        X, Y = lens(t * W, np.full_like(t, -0.1 * H + (j + phase) * sp))
        _line(d, X, Y, 255, w)


def mesh_polar(d, W, H, w, seed, phase=0.0):
    rng = random.Random(seed)
    sp = _spacing(W, H, rng)
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.25, 0.75)
    R = 1.02 * max(math.hypot(cx, cy), math.hypot(W - cx, cy), math.hypot(cx, H - cy), math.hypot(W - cx, H - cy))
    nr, ns = int(R / sp) + 1, rng.randint(36, 110)
    m = rng.randint(4, 18)
    a = min(W, H) * rng.uniform(0.008, 0.03)
    q = rng.uniform(1.0, 4.0)
    t = np.linspace(0, 2 * np.pi, 2400)
    for j in range(1, nr + 1):
        r = (j + phase) * sp + a * np.sin(m * t + j * 0.35)
        _line(d, cx + r * np.cos(t), cy + r * np.sin(t), 255, w, closed=True)
    s = np.linspace(0, R, 1200)
    for k in range(ns):
        th0 = 2 * np.pi * (k + phase) / ns
        th = th0 + 0.10 * np.sin(2 * np.pi * q * s / R + th0 * 2)
        _line(d, cx + s * np.cos(th), cy + s * np.sin(th), 255, w)


NETS = {
    "warped mesh": mesh_warp,
    "lens-distorted mesh": mesh_lens,
    "polar web": mesh_polar,
}


# ---------- цвет: встроенные градиенты ----------

def gradient_field(Wd, Ht, stops, rng):
    yy, xx = np.mgrid[0:Ht, 0:Wd].astype(np.float32) / Ht  # единый масштаб по обеим осям
    ar = Wd / Ht
    mode = rng.choice(["linear", "radial", "angular", "waves"])
    cx, cy = rng.uniform(0.1, ar - 0.1), rng.uniform(0.2, 0.8)
    if mode == "linear":
        th = rng.uniform(0, math.pi)
        s = xx * math.cos(th) + yy * math.sin(th)
        s = (s - s.min()) / (s.max() - s.min())
    elif mode == "radial":
        s = np.clip(np.hypot(xx - cx, yy - cy) / 0.8, 0, 1)
    elif mode == "angular":
        s = np.arctan2(yy - cy, xx - cx) / (2 * np.pi) + 0.5
        s = np.abs(2 * s - 1)
    else:
        s = 0.5 + 0.5 * np.sin(2 * np.pi * (rng.uniform(0.6, 1.6) * xx + rng.uniform(0.6, 1.6) * yy)
                               + rng.uniform(0, 6.28))
    s = np.clip(s + 0.07 * np.sin(2 * np.pi * (2.3 * xx - 1.7 * yy) + rng.uniform(0, 6.28)), 0, 1)
    pos = np.linspace(0, 1, len(stops))
    cols = np.array(stops, np.float32)
    return np.stack([np.interp(s, pos, cols[:, c]) for c in range(3)], -1).astype(np.float32)


# ---------- цвет: из загруженного изображения ----------

@lru_cache(maxsize=16)
def load_ref(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((1600, 1600))
    return img


def ref_color_field(img, Wd, Ht, rng, fixed, blur):
    """Цветовое поле нужного размера из изображения-примера."""
    iw, ih = img.size
    ar = Wd / Ht
    z = 1.0 if fixed else rng.uniform(0.6, 1.0)  # доля изображения, попадающая в кадр
    if iw / ih > ar:
        ch = ih * z
        cw = ch * ar
    else:
        cw = iw * z
        ch = cw / ar
    x0 = (iw - cw) / 2 if fixed else rng.uniform(0, iw - cw)
    y0 = (ih - ch) / 2 if fixed else rng.uniform(0, ih - ch)
    crop = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((Wd, Ht), Image.BICUBIC)
    if not fixed:
        if rng.random() < 0.5:
            crop = ImageOps.mirror(crop)
        if rng.random() < 0.5:
            crop = ImageOps.flip(crop)
    if blur > 0:
        crop = crop.filter(ImageFilter.GaussianBlur(min(Wd, Ht) * blur))
    arr = np.asarray(crop, np.float32) / 255
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]
    return arr * np.minimum(1.0, 0.72 / np.maximum(lum, 1e-3))  # светлые цвета затемняем


# ---------- текст ----------

def load_font(path, size):
    for p in ([path] if path else []) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_words(cw, ch, words, font_path, rng, fill):
    m = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(m)
    n = len(words)
    size = int(ch * fill / n)
    while True:
        font = load_font(font_path, size)
        if max(d.textlength(wd, font=font) for wd in words) <= cw * 0.88 or size < 12:
            break
        size = int(size * 0.92)
    lh = size * 1.25
    cy = ch * rng.uniform(0.38, 0.62)
    for i, wd in enumerate(words):
        d.text((cw / 2, cy + (i - (n - 1) / 2) * lh), wd, font=font, fill=255, anchor="mm")
    return m


def draw_stacked(W, H, word, font_path, rng):
    """Слово столбиком: по одной букве в строке, буквы получаются крупными."""
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    n = len(word)
    size = int(H * 0.80 / n / 1.2)
    while True:
        font = load_font(font_path, size)
        if max(d.textlength(c, font=font) for c in word) <= W * 0.85 or size < 12:
            break
        size = int(size * 0.92)
    lh = H * 0.80 / n
    cy = H * rng.uniform(0.45, 0.55)
    for i, c in enumerate(word):
        d.text((W / 2, cy + (i - (n - 1) / 2) * lh), c, font=font, fill=255, anchor="mm")
    return m


def text_mask(W, H, words, font_path, rng):
    short = [x for x in words if len(x) <= 6]
    options = ["horizontal", "two lines"]
    if H > W:
        options += ["vertical", "vertical"] + (["stacked"] * 2 if short else [])
    layout = rng.choice(options)
    if layout == "stacked":
        picked = [rng.choice(short)]
        m = draw_stacked(W, H, picked[0], font_path, rng)
    elif layout == "two lines":
        picked = rng.sample(words, 2) if len(words) > 1 else [words[0], words[0]]
        m = draw_words(W, H, picked, font_path, rng, 0.30)
    elif layout == "vertical":
        picked = [rng.choice(words)]
        m = draw_words(H, W, picked, font_path, rng, 0.40).rotate(rng.choice([90, 270]), expand=True)
    else:
        picked = [rng.choice(words)]
        m = draw_words(W, H, picked, font_path, rng, 0.15)
    return m, layout, " / ".join(picked)


# ---------- рельеф ----------

def height_and_grad(mask, rng, lo=0.0035, hi=0.007):
    sigma = max(1.2, min(mask.size) * rng.uniform(lo, hi))
    h = np.asarray(mask.filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255
    gy, gx = np.gradient(h)
    return gx, gy, sigma


def displace(arr, dx, dy):
    H, W = arr.shape
    yy, xx = np.mgrid[0:H, 0:W]
    sx = np.clip(np.rint(xx + dx).astype(int), 0, W - 1)
    sy = np.clip(np.rint(yy + dy).astype(int), 0, H - 1)
    return arr[sy, sx]


def apply_relief(out, mask_img, gx, gy, sigma, tint, rng, fill, gain, shadow):
    mask = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
    s = sigma * rng.uniform(2.5, 4.5)
    nx, ny = -gx * s, -gy * s
    L = np.array([-0.55, -0.60, 0.58], np.float32)
    L /= np.linalg.norm(L)
    shade = (nx * L[0] + ny * L[1] + L[2]) / np.sqrt(nx ** 2 + ny ** 2 + 1) - L[2]
    off = max(1, int(sigma * 0.9))
    sh = np.roll(np.roll(mask, off, axis=0), off, axis=1)
    sh = np.asarray(Image.fromarray((sh * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(sigma * 1.2)), np.float32) / 255
    out = out * (1 - shadow * sh * (1 - mask))[..., None]
    if fill > 0:
        m3 = mask[..., None]
        out = out * (1 - fill * m3) + (0.75 * tint + 0.25) * fill * m3
    return out * (1 + gain * shade[..., None]) + 0.30 * gain * np.maximum(shade, 0)[..., None]


# ---------- сборка ----------

def make(Wd, Ht, rng, font_path, words, letters, refs=None, ref_fixed=False, ref_blur=0.004):
    W, H = Wd * SS, Ht * SS
    paper = hex_rgb(rng.choice(PAPERS))
    w = max(1, round(rng.uniform(0.6, 1.5) * SS * min(Wd, Ht) / 512))

    net_name = rng.choice(list(NETS))
    seed = rng.randint(0, 2 ** 31 - 1)
    Lhi, layout, text = text_mask(W, H, words, font_path, rng)

    mesh = Image.new("L", (W, H), 0)
    NETS[net_name](ImageDraw.Draw(mesh), W, H, w, seed, 0.0)
    if letters == "mesh":  # внутри букв та же сетка, сдвинутая на полпериода и более толстая
        mesh1 = Image.new("L", (W, H), 0)
        NETS[net_name](ImageDraw.Draw(mesh1), W, H, max(w + 1, round(w * 1.7)), seed, 0.5)
        mesh = Image.composite(mesh1, mesh, Lhi)

    tm = Lhi.resize((Wd, Ht), Image.LANCZOS)
    L = np.asarray(tm, np.float32) / 255
    if letters == "mesh":
        gx, gy, sigma = height_and_grad(tm, rng, 0.002, 0.004)
        disp = sigma * rng.uniform(0.8, 2.0)  # лёгкий изгиб линий у краёв букв
    else:
        gx, gy, sigma = height_and_grad(tm, rng)
        disp = sigma * rng.uniform(8, 16)

    if refs:
        grad = ref_color_field(load_ref(str(rng.choice(refs))), Wd, Ht, rng, ref_fixed, ref_blur)
        color_desc = "smooth multicolor gradient taken from a reference image"
    else:
        g = rng.choice(GRADIENTS)
        grad = gradient_field(Wd, Ht, g[1], rng)
        color_desc = f"smooth {g[0]} color gradient"

    a = np.asarray(mesh.resize((Wd, Ht), Image.LANCZOS), np.float32) / 255
    a = displace(a, -gx * disp, -gy * disp) * 0.88
    col = grad
    if letters == "mesh":  # линии внутри букв темнее и насыщеннее
        col = grad * (1 - L[..., None]) + (grad ** 1.7 * 0.9) * L[..., None]
    out = np.ones((Ht, Wd, 3), np.float32) * paper
    out = out * (1 - a[..., None]) + col * a[..., None]

    if letters == "mesh":
        out = apply_relief(out, tm, gx, gy, sigma, grad, rng, fill=0.0, gain=0.30, shadow=0.10)
    else:
        out = apply_relief(out, tm, gx, gy, sigma, grad, rng, fill=0.55, gain=0.85, shadow=0.22)

    nrng = np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    out = out + nrng.normal(0, 0.006, out.shape).astype(np.float32)
    img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))

    if letters == "mesh":
        lettering = f"lettering '{text}' formed by the line mesh itself ({layout} text)"
    else:
        lettering = f"embossed relief lettering '{text}' ({layout} text)"
    caption = f"guilloche-style ornamental mesh pattern, {net_name}, {color_desc}, {lettering}"
    return img, caption


def build_grid(paths, cols, tw, th):
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, p in enumerate(paths):
        sheet.paste(Image.open(p).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    return sheet


def collect_refs(arg):
    if not arg:
        return None
    p = Path(arg)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
    elif p.is_file():
        files = [p]
    else:
        raise SystemExit(f"--ref: путь не найден: {arg}")
    if not files:
        raise SystemExit(f"--ref: в папке нет изображений ({', '.join(sorted(IMG_EXT))})")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", default="ornament_ds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--letters", choices=["mesh", "emboss"], default="mesh")
    ap.add_argument("--font", default=None, help="путь к .ttf (по умолчанию DejaVu Serif Bold)")
    ap.add_argument("--words", default=None, help="слова через запятую")
    ap.add_argument("--ref", default=None, help="изображение или папка с изображениями-цветовыми примерами")
    ap.add_argument("--ref-fixed", action="store_true", help="без случайного кропа и отражений")
    ap.add_argument("--ref-blur", type=float, default=0.004, help="сглаживание цветового поля (доля от размера)")
    ap.add_argument("--preview", type=int, default=0, help="превью из первых N картинок")
    args = ap.parse_args()

    words = [x.strip() for x in args.words.split(",") if x.strip()] if args.words else WORDS
    refs = collect_refs(args.ref)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as meta:
        for i in range(args.n):
            rng = random.Random(args.seed * 1_000_003 + i)
            img, cap = make(args.width, args.height, rng, args.font, words, args.letters,
                            refs, args.ref_fixed, args.ref_blur)
            name = f"{i:05d}.png"
            img.save(out / name)
            meta.write(json.dumps({"file_name": name, "text": cap}, ensure_ascii=False) + "\n")
            paths.append(out / name)
    if args.preview:
        k = min(args.preview, len(paths))
        build_grid(paths[:k], 4, args.width // 2, args.height // 2).save(out / "preview.png")
    print(f"готово: {args.n} изображений {args.width}x{args.height} -> {out}")


if __name__ == "__main__":
    main()
