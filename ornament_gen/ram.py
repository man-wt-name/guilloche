"""Оценка пиковой RAM и авто-понижение quality при превышении --max-ram."""
from __future__ import annotations

import dataclasses

from .core import GenerationContext, Quality

DEFAULT_MAX_RAM_MB = 4096.0
QUALITY_DOWNGRADE = {
    Quality.MAX: Quality.HIGH,
    Quality.HIGH: Quality.NORMAL,
    Quality.NORMAL: Quality.DRAFT,
    Quality.DRAFT: None,
}


def estimate_peak_ram_mb(
    ctx: GenerationContext, ref_warp_active: bool, weave_active: bool
) -> float:
    """Грубая верхняя оценка пиковой RAM в MB. Умышленно консервативная."""
    H, W = ctx.height, ctx.width
    H_ss, W_ss = ctx.ss_height, ctx.ss_width
    b = 0
    b += 3 * H * W * 4          # canvas
    b += H * W * 4              # total_alpha
    b += 3 * H * W * 4          # color
    b += 2 * H_ss * W_ss * 4    # mask_ss + z_ss
    if weave_active:
        b += H_ss * W_ss * 4    # dist float32
        b += 2 * H_ss * W_ss * 8  # ny/nx int64 из EDT
    if ref_warp_active:
        b += 14 * H * W * 4     # пик warp_fields (h, gx, gy, dx, dy, ...)
        b += 2 * H_ss * W_ss * 4  # upscaled dx/dy
    b = int(b * 1.4)            # safety margin
    return b / (1024.0 * 1024.0)


def auto_downgrade_quality(
    ctx: GenerationContext,
    ref_warp_active: bool,
    weave_active: bool,
    max_ram_mb: float,
) -> tuple[GenerationContext, str | None]:
    """Понижать quality, пока оценка RAM не уложится в max_ram_mb.

    Возвращает (ctx, msg|None). Если даже draft не влезает — возвращает
    (draft_ctx, сообщение об ошибке); CLI решает, что делать.
    """
    current = ctx
    path: list[str] = []
    while True:
        est = estimate_peak_ram_mb(current, ref_warp_active, weave_active)
        if est <= max_ram_mb:
            if not path:
                return current, None
            return current, (
                f"quality auto-downgraded ({' → '.join(path)}): "
                f"peak RAM estimate {est:.0f} MB <= {max_ram_mb:.0f} MB"
            )
        nxt = QUALITY_DOWNGRADE.get(current.quality)
        if nxt is None:
            return current, (
                f"peak RAM estimate {est:.0f} MB exceeds limit {max_ram_mb:.0f} MB "
                f"even at draft; reduce canvas size"
            )
        path.append(f"{current.quality.value}→{nxt.value}")
        current = dataclasses.replace(current, quality=nxt)