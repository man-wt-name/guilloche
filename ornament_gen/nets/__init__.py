"""Реєстр сеток. Каждая — generate(ctx, rng, params) -> list[Curve]."""
from . import (
    flow, lens, lissajous, polar, rose, spirograph, spiral, star, warped,
)

NETS: dict[str, object] = {
    "warped": warped.generate,
    "lens": lens.generate,
    "polar": polar.generate,
    "spiral": spiral.generate,
    "spirograph": spirograph.generate,
    "lissajous": lissajous.generate,
    "rose": rose.generate,
    "star": star.generate,
    "flow": flow.generate,
}

__all__ = ["NETS"]