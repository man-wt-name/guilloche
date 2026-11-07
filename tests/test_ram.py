"""Тесты оценки RAM и auto-downgrade."""
from __future__ import annotations

import pytest

from ornament_gen.core import BackgroundKind, GenerationContext, Quality
from ornament_gen.ram import (
    auto_downgrade_quality,
    estimate_peak_ram_mb,
)


def test_estimate_grows_with_size():
    a = GenerationContext(width=1110, height=756, seed=0, quality=Quality.NORMAL)
    b = GenerationContext(width=2220, height=1512, seed=0, quality=Quality.NORMAL)
    assert estimate_peak_ram_mb(b, False, False) > estimate_peak_ram_mb(a, False, False)


def test_estimate_grows_with_quality():
    a = GenerationContext(width=1110, height=756, seed=0, quality=Quality.DRAFT)
    b = GenerationContext(width=1110, height=756, seed=0, quality=Quality.HIGH)
    assert estimate_peak_ram_mb(b, False, False) > estimate_peak_ram_mb(a, False, False)


def test_estimate_ref_warp_costs_more():
    ctx = GenerationContext(width=2000, height=1500, seed=0, quality=Quality.NORMAL)
    a = estimate_peak_ram_mb(ctx, False, False)
    b = estimate_peak_ram_mb(ctx, True, False)
    assert b > a


def test_estimate_weave_costs_more():
    ctx = GenerationContext(width=2000, height=1500, seed=0, quality=Quality.NORMAL)
    a = estimate_peak_ram_mb(ctx, False, False)
    b = estimate_peak_ram_mb(ctx, False, True)
    assert b > a


def test_no_downgrade_needed():
    ctx = GenerationContext(width=1110, height=756, seed=0, quality=Quality.NORMAL)
    new_ctx, msg = auto_downgrade_quality(ctx, False, False, max_ram_mb=4096)
    assert msg is None
    assert new_ctx.quality is Quality.NORMAL


def test_downgrade_high_to_normal():
    """Фикс: max_ram_mb=4096 (A4 @ draft + weave ≈ 1704 MB, но normal ≈ 3409 MB — влезает)."""
    ctx = GenerationContext(width=3508, height=2480, seed=0, quality=Quality.HIGH)
    new_ctx, msg = auto_downgrade_quality(ctx, False, True, max_ram_mb=4096)
    assert msg is not None
    assert "auto-downgraded" in msg
    assert new_ctx.quality in (Quality.NORMAL, Quality.DRAFT)
    assert new_ctx.ss == {Quality.DRAFT: 2, Quality.NORMAL: 3,
                          Quality.HIGH: 4, Quality.MAX: 4}[new_ctx.quality]


def test_downgrade_to_draft_still_exceeds():
    """Огромный размер — даже draft не влезает, возвращается draft + msg."""
    ctx = GenerationContext(width=8000, height=6000, seed=0, quality=Quality.HIGH)
    new_ctx, msg = auto_downgrade_quality(ctx, False, True, max_ram_mb=256)
    assert new_ctx.quality is Quality.DRAFT
    assert msg is not None
    assert "exceeds limit" in msg


def test_downgrade_preserves_other_fields():
    ctx = GenerationContext(
        width=3508, height=2480, seed=42, quality=Quality.HIGH,
        background=BackgroundKind.NONE,
        noise=0.01, symmetry="vertical",
    )
    new_ctx, _ = auto_downgrade_quality(ctx, False, True, max_ram_mb=1024)
    assert new_ctx.seed == 42
    assert new_ctx.symmetry == "vertical"
    assert new_ctx.noise == 0.01