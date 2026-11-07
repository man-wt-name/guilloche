"""In-memory LRU-кэш снимков curves. Ключ: name + params + ss + seed слоя.

P3-3: удалён неиспользуемый `import numpy as np` (Curve.xs уже ndarray,
np не нужен — .copy() это метод ndarray).
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from .core import Curve, GenerationContext, Layer


def _make_key(layer: Layer, ctx: GenerationContext, layer_idx: int) -> str:
    """Стабильный ключ по всему, что влияет на curves ДО warp/thickness."""
    payload = {
        "name": layer.name,
        "params": layer.params,
        "ss": ctx.ss,
        "seed": ctx.seed * 1_000_003 + layer_idx,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def snapshot_curves(curves: list[Curve]) -> list[tuple]:
    """Глубокая копия curves без zs (zs проставляются позже)."""
    return [
        (c.xs.copy(), c.ys.copy(), float(c.width), bool(c.closed))
        for c in curves
    ]


def restore_curves(snapshot: list[tuple]) -> list[Curve]:
    """Восстановить свежие Curve из снимка — безопасно мутировать."""
    return [
        Curve(xs=xs.copy(), ys=ys.copy(), width=w, closed=closed)
        for xs, ys, w, closed in snapshot
    ]


class RasterizationCache:
    """LRU-кэш curves. Только в памяти, ограниченный размер."""

    def __init__(self, max_entries: int = 32):
        self.max_entries = max(1, int(max_entries))
        self._data: OrderedDict[str, list[tuple]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[tuple] | None:
        if key not in self._data:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return self._data[key]

    def put(self, key: str, snapshot: list[tuple]) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = snapshot
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._data)