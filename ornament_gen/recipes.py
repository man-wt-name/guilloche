"""Готовые пресеты. Каждый — функция без аргументов, возвращает {layers, weave}.

P1-1: spiral_moire — turns=15/15.5 (разные частоты, биения).
P1-3: random_layers — contours с быстрыми дефолтами (n_waves=3, n_levels=3).
P3-11: spirograph r_ratio = 0.25/0.4/1/3 — точное замыкание (было 0.31/0.27/0.41/0.23,
       дававшее артефакт замыкания ~0.06-0.90 цикла).
"""
from __future__ import annotations

from typing import Any, Callable


def _net(name: str, params: dict | None = None, **kw: Any) -> dict:
    return {"kind": "net", "name": name, "params": params or {}, **kw}


def _pat(name: str, params: dict | None = None, **kw: Any) -> dict:
    return {"kind": "pattern", "name": name, "params": params or {}, **kw}


def woven_classic() -> dict:
    return {
        "layers": [
            _net("warped", {"width": 0.55, "spacing": 0.018},
                 opacity=0.95, blend="multiply",
                 z_preset="sine", z_period=4.0, z_phase=0.0),
            _net("warped", {"width": 0.55, "spacing": 0.018},
                 opacity=0.95, blend="multiply",
                 z_preset="sine", z_period=4.0, z_phase=3.14159),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.32},
    }


def woven_three() -> dict:
    # P3-11: r_ratio = 1/3, 0.25, 0.4 — точное замыкание спирографа
    # (было 0.31, 0.27, 0.41 — артефакт замыкания).
    return {
        "layers": [
            _net("spirograph", {"width": 0.5, "r_ratio": 1 / 3, "turns": 22},
                 opacity=0.85, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=0.0),
            _net("spirograph", {"width": 0.5, "r_ratio": 0.25, "turns": 28},
                 opacity=0.8, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=2.094),
            _net("spirograph", {"width": 0.5, "r_ratio": 0.4, "turns": 18},
                 opacity=0.75, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=4.189),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.35},
    }


def rosette_banknote() -> dict:
    return {
        "layers": [
            _pat("rosette", {"n_circles": 14, "radius_range": (0.28, 0.46), "offset": 0.10},
                 opacity=0.9, blend="multiply",
                 z_preset="sine", z_period=2.0, z_phase=0.0),
            _net("lissajous", {"a": 5, "b": 4, "width": 0.4},
                 opacity=0.8, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=1.571),
            _net("polar", {"width": 0.35, "spacing": 0.03},
                 opacity=0.5, blend="screen"),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.28},
    }


def multiscale() -> dict:
    # P3-11: r_ratio = 0.25 — точное замыкание (было 0.23 — артефакт 0.4 цикла).
    return {
        "layers": [
            _net("warped", {"width": 0.7, "spacing": 0.025},
                 opacity=0.85, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=0.0),
            _net("spirograph", {"width": 0.4, "r_ratio": 0.25, "turns": 30},
                 opacity=0.75, blend="multiply",
                 z_preset="sine", z_period=4.0, z_phase=1.0),
            _pat("dots", {"spacing": 0.04, "radius": 0.006, "width": 0.3},
                 opacity=0.5, blend="multiply"),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.22},
    }


def spiral_moire() -> dict:
    # P1-1: turns=15 и 15.5 — разные частоты, медленные биения = классический moiré.
    return {
        "layers": [
            _net("spiral", {"width": 0.35, "turns": 15, "kind": "archimedean"},
                 opacity=0.7, blend="multiply"),
            _net("spiral", {"width": 0.35, "turns": 15.5, "kind": "archimedean"},
                 opacity=0.7, blend="screen"),
        ],
        "weave": {"mode": "none", "shadow": 0.0},
    }


def flow_organic() -> dict:
    return {
        "layers": [
            _net("flow", {"n_seeds": 40, "n_waves": 6, "width": 0.4},
                 opacity=0.85, blend="multiply"),
            _net("flow", {"n_seeds": 30, "n_waves": 4, "width": 0.35},
                 opacity=0.7, blend="screen"),
        ],
        "weave": {"mode": "none", "shadow": 0.0},
    }


def lissajous_web() -> dict:
    return {
        "layers": [
            _net("lissajous", {"a": 4, "b": 3, "width": 0.45},
                 opacity=0.9, blend="multiply",
                 z_preset="checkerboard", z_period=6.0, z_phase=0.0),
            _net("lissajous", {"a": 5, "b": 7, "width": 0.35},
                 opacity=0.85, blend="multiply",
                 z_preset="checkerboard", z_period=6.0, z_phase=1.0),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.30},
    }


def star_geometric() -> dict:
    return {
        "layers": [
            _net("star", {"n": 8, "k": 3, "width": 0.5},
                 opacity=0.85, blend="multiply",
                 z_preset="sine", z_period=4.0, z_phase=0.0),
            _net("polar", {"width": 0.4, "spacing": 0.028},
                 opacity=0.65, blend="multiply",
                 z_preset="sine", z_period=4.0, z_phase=2.5),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.28},
    }


def embossed() -> dict:
    return {
        "layers": [
            _net("warped", {"width": 0.6, "spacing": 0.02},
                 opacity=0.95, blend="multiply",
                 z_preset="sine", z_period=4.0, z_phase=0.0,
                 effects=[
                     {"kind": "emboss", "params": {"depth": 1.8}},
                     {"kind": "shadow", "params": {"offset": (0.003, 0.003),
                                                   "radius": 0.004, "opacity": 0.4}},
                 ]),
        ],
        "weave": {"mode": "none", "shadow": 0.0},
    }


def document_classic() -> dict:
    return {
        "layers": [
            _net("polar", {"width": 0.5, "spacing": 0.022},
                 opacity=0.85, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=0.0),
            _pat("rosette", {"n_circles": 12, "radius_range": (0.32, 0.44)},
                 opacity=0.7, blend="multiply",
                 z_preset="sine", z_period=3.0, z_phase=1.2,
                 effects=[{"kind": "glow", "params": {"radius": 0.004, "strength": 0.25}}]),
            _pat("contours", {"n_waves": 4, "n_levels": 4, "width": 0.3},
                 opacity=0.35, blend="screen"),
        ],
        "weave": {"mode": "zbuffer", "shadow": 0.30},
    }


RECIPES: dict[str, Callable[[], dict]] = {
    "woven_classic": woven_classic,
    "woven_three": woven_three,
    "rosette_banknote": rosette_banknote,
    "multiscale": multiscale,
    "spiral_moire": spiral_moire,
    "flow_organic": flow_organic,
    "lissajous_web": lissajous_web,
    "star_geometric": star_geometric,
    "embossed": embossed,
    "document_classic": document_classic,
}

DEFAULT_RECIPE = "woven_classic"


def random_layers(n: int, seed: int) -> dict:
    """Случайный рецепт из n слоёв. Детерминирован по seed.

    P1-3: contours получает быстрые дефолты (n_waves=3, n_levels=3).
    """
    import random as _random
    rng = _random.Random(seed * 7_919 + n)
    nets = ["warped", "lens", "polar", "spiral", "spirograph",
            "lissajous", "rose", "star"]
    patterns = ["dots", "crosses", "contours", "voronoi_fill", "rosette"]
    blends = ["multiply", "screen", "overlay", "normal"]
    z_presets = ["sine", "checkerboard", "saw"]
    layers = []
    for i in range(n):
        if rng.random() < 0.75:
            name = rng.choice(nets)
            layers.append({
                "kind": "net", "name": name, "params": {},
                "opacity": round(rng.uniform(0.6, 0.95), 2),
                "blend": rng.choice(blends),
                "z_preset": rng.choice(z_presets),
                "z_period": round(rng.uniform(2.0, 5.0), 2),
                "z_phase": round(rng.uniform(0.0, 6.28), 2),
                "effects": [],
            })
        else:
            name = rng.choice(patterns)
            params = {"n_waves": 3, "n_levels": 3} if name == "contours" else {}
            layers.append({
                "kind": "pattern", "name": name, "params": params,
                "opacity": round(rng.uniform(0.5, 0.85), 2),
                "blend": rng.choice(blends),
                "effects": [],
            })
    use_weave = rng.random() < 0.7
    return {
        "layers": layers,
        "weave": {
            "mode": "zbuffer" if use_weave else "none",
            "shadow": round(rng.uniform(0.15, 0.4), 2) if use_weave else 0.0,
        },
    }