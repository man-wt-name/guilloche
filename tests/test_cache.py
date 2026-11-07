"""Тесты кэша: снимки curves, LRU, интеграция с make."""
from __future__ import annotations

import numpy as np
import pytest

from ornament_gen.cache import (
    RasterizationCache,
    _make_key,
    restore_curves,
    snapshot_curves,
)
from ornament_gen.core import (
    Curve,
    GenerationContext,
    Layer,
    Quality,
)


@pytest.fixture
def ctx():
    return GenerationContext(width=1110, height=756, seed=42, quality=Quality.DRAFT)


def test_snapshot_restore_independent():
    c = Curve(np.array([1.0, 2.0], np.float32),
              np.array([3.0, 4.0], np.float32),
              2.0, zs=np.array([0.5, 0.6], np.float32), closed=True)
    snap = snapshot_curves([c])
    c.xs[0] = 999.0
    out = restore_curves(snap)
    assert out[0].xs[0] == 1.0
    out[0].xs[0] = -1.0
    out2 = restore_curves(snap)
    assert out2[0].xs[0] == 1.0


def test_snapshot_zs_not_preserved():
    c = Curve(np.array([1.0], np.float32),
              np.array([2.0], np.float32),
              1.0, zs=np.array([0.7], np.float32))
    snap = snapshot_curves([c])
    out = restore_curves(snap)
    assert (out[0].zs == 0.0).all()


def test_make_key_deterministic(ctx):
    layer = Layer(kind="net", name="polar", params={"width": 0.5})
    k1 = _make_key(layer, ctx, 0)
    k2 = _make_key(layer, ctx, 0)
    assert k1 == k2


def test_make_key_differs_by_idx(ctx):
    layer = Layer(kind="net", name="polar", params={"width": 0.5})
    k0 = _make_key(layer, ctx, 0)
    k1 = _make_key(layer, ctx, 1)
    assert k0 != k1


def test_make_key_differs_by_params(ctx):
    a = Layer(kind="net", name="polar", params={"width": 0.5})
    b = Layer(kind="net", name="polar", params={"width": 0.6})
    assert _make_key(a, ctx, 0) != _make_key(b, ctx, 0)


def test_make_key_differs_by_ss():
    layer = Layer(kind="net", name="polar")
    ctx_d = GenerationContext(width=1110, height=756, seed=42, quality=Quality.DRAFT)
    ctx_n = GenerationContext(width=1110, height=756, seed=42, quality=Quality.NORMAL)
    assert _make_key(layer, ctx_d, 0) != _make_key(layer, ctx_n, 0)


def test_cache_lru_evicts():
    c = RasterizationCache(max_entries=2)
    c.put("a", [])
    c.put("b", [])
    c.get("a")
    c.put("c", [])
    assert "a" in c._data
    assert "b" not in c._data
    assert "c" in c._data


def test_cache_hits_misses():
    c = RasterizationCache()
    assert c.get("x") is None
    assert c.misses == 1
    c.put("x", [])
    assert c.get("x") == []
    assert c.hits == 1


def test_cache_clear():
    c = RasterizationCache()
    c.put("x", [])
    c.get("x")
    c.clear()
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0


def test_cache_integration_with_make(ctx):
    from ornament_gen.compose import make
    cache = RasterizationCache()
    layer = Layer(kind="net", name="polar", params={"width": 0.5})
    a = np.asarray(make(ctx, [layer], use_cache=cache).image)
    b = np.asarray(make(ctx, [layer], use_cache=cache).image)
    np.testing.assert_array_equal(a, b)
    assert cache.hits >= 1
    assert cache.misses >= 1


def test_cache_does_not_change_result(ctx):
    from ornament_gen.compose import make
    layer = Layer(kind="net", name="polar", params={"width": 0.5})
    a = np.asarray(make(ctx, [layer]).image)
    b = np.asarray(make(ctx, [layer], use_cache=RasterizationCache()).image)
    np.testing.assert_array_equal(a, b)