"""Преобразование физических размеров + DPI в пиксели с валидацией."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MM_PER_INCH = 25.4

# ISO 216 и типовые форматы, в мм
ISO_SIZES: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "a6": (105.0, 148.0),
    "letter": (215.9, 279.4),
    "passport": (125.0, 88.0),
    "id_card": (85.6, 54.0),
}

MIN_SHORT = 756
MIN_LONG = 1110
DPI_RANGE = (150, 600)


class Orientation(str, Enum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class SizeSpecError(ValueError):
    pass


@dataclass(frozen=True)
class CanvasSize:
    width: int
    height: int
    dpi: int

    @property
    def orientation(self) -> Orientation:
        return Orientation.LANDSCAPE if self.width >= self.height else Orientation.PORTRAIT


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / MM_PER_INCH * dpi))


def resolve_size(
    *,
    width: int | None = None,
    height: int | None = None,
    size: str | None = None,
    dpi: int = 300,
    orientation: Orientation = Orientation.LANDSCAPE,
) -> CanvasSize:
    lo, hi = DPI_RANGE
    if not (lo <= dpi <= hi):
        raise SizeSpecError(f"dpi должен быть в [{lo}, {hi}], получено {dpi}")

    if size is not None:
        key = size.lower()
        if key not in ISO_SIZES:
            raise SizeSpecError(
                f"неизвестный размер '{size}', допустимые: {', '.join(sorted(ISO_SIZES))}"
            )
        w_mm, h_mm = ISO_SIZES[key]
        w_px = _mm_to_px(w_mm, dpi)
        h_px = _mm_to_px(h_mm, dpi)
        if orientation is Orientation.LANDSCAPE:
            w_px, h_px = max(w_px, h_px), min(w_px, h_px)
        else:
            w_px, h_px = min(w_px, h_px), max(w_px, h_px)
    elif width is not None and height is not None:
        w_px, h_px = width, height
    else:
        raise SizeSpecError("укажите либо --size, либо --width и --height")

    if w_px < 1 or h_px < 1:
        raise SizeSpecError(f"размер должен быть положительным, получено {w_px}x{h_px}")

    short = min(w_px, h_px)
    long_ = max(w_px, h_px)
    if short < MIN_SHORT or long_ < MIN_LONG:
        raise SizeSpecError(
            f"минимальный размер {MIN_LONG}x{MIN_SHORT} (любая ориентация), "
            f"получено {w_px}x{h_px}"
        )

    return CanvasSize(w_px, h_px, dpi)