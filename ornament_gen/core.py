"""Структуры данных: Layer, GenerationContext, Quality, Background, Thickness, Effect."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Any, Literal

import numpy as np

from .symmetry import is_valid_symmetry


class Quality(str, Enum):
    DRAFT = "draft"
    NORMAL = "normal"
    HIGH = "high"
    MAX = "max"


QUALITY_PARAMS: dict[Quality, dict[str, int]] = {
    Quality.DRAFT:  {"ss": 2, "curve_points": 1500},
    Quality.NORMAL: {"ss": 3, "curve_points": 3000},
    Quality.HIGH:   {"ss": 4, "curve_points": 4000},
    Quality.MAX:    {"ss": 4, "curve_points": 6000},
}


class BackgroundKind(str, Enum):
    PAPER = "paper"
    WHITE = "white"
    BLACK = "black"
    NONE  = "none"


@dataclass
class GenerationContext:
    """Всё, что нужно любому генератору/эффекту для работы."""
    width: int
    height: int
    seed: int
    quality: Quality = Quality.NORMAL
    background: BackgroundKind = BackgroundKind.PAPER
    noise: float = 0.0
    flat_color: bool = False
    variable_thickness: bool = False
    symmetry: str = "none"

    ss: int = field(init=False)
    curve_points: int = field(init=False)

    def __post_init__(self) -> None:
        p = QUALITY_PARAMS[self.quality]
        self.ss = p["ss"]
        self.curve_points = p["curve_points"]
        if not is_valid_symmetry(self.symmetry):
            raise ValueError(
                f"invalid symmetry: {self.symmetry!r} "
                f"(ожидается none|vertical|horizontal|central|radialN, N>=2)"
            )

    @property
    def ss_width(self) -> int:
        return self.width * self.ss

    @property
    def ss_height(self) -> int:
        return self.height * self.ss

    def layer_rng(self, idx: int) -> Random:
        return Random(self.seed * 1_000_003 + idx)


@dataclass
class ThicknessSpec:
    kind: Literal["constant", "modulated"] = "constant"
    base: float = 1.0
    amplitude: float = 0.3
    period: float = 4.0
    phase: float = 0.0


@dataclass
class EffectSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Layer:
    kind: Literal["net", "pattern"]
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    opacity: float = 1.0
    blend: str = "normal"
    effects: list[EffectSpec] = field(default_factory=list)
    weave_group: str = "A"
    z_preset: str | None = None
    z_period: float = 4.0
    z_phase: float = 0.0
    thickness: ThicknessSpec = field(default_factory=ThicknessSpec)


@dataclass
class Curve:
    """Одна кривая в SS-координатах."""
    xs: np.ndarray
    ys: np.ndarray
    width: float
    zs: np.ndarray | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.zs is None:
            self.zs = np.zeros_like(self.xs, dtype=np.float32)