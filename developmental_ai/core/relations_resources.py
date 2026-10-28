"""RW-14: read-only configured context capacity observation."""
from __future__ import annotations

from typing import Optional

from developmental_ai.core.relations import RelationRegistry


def _context_width(loop) -> Optional[float]:
    attention = getattr(loop, "_am", None)
    window = getattr(attention, "win", None) if attention is not None else None
    width = getattr(window, "W", None) if window is not None else None
    return None if width is None else float(width)


def register(reg: RelationRegistry) -> None:
    reg.register("ID-06", "WMC", "WM", "WP-7 active latent window",
                 "attention working-memory buffer", "context-window width",
                 "frames", "per segment", "data", probe=_context_width,
                 group="resources")
