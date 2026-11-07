"""Реєстр узоров. Каждый — generate(ctx, rng, params) -> list[Curve]."""
from . import contours, crosses, dots, rosette, voronoi_fill

PATTERNS: dict[str, object] = {
    "dots": dots.generate,
    "crosses": crosses.generate,
    "contours": contours.generate,
    "voronoi_fill": voronoi_fill.generate,
    "rosette": rosette.generate,
}

__all__ = ["PATTERNS"]