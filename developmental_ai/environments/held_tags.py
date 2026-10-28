"""Which action drove a real-time interval (2026-10-18).

Kept OUT of developmental_ai.core on purpose: importing that package builds
the whole loop, and the real-time adapters call this on their first step —
inside the game's clock (tests/_vizdoom_smoke.py J measured the cost).
See developmental_ai/core/realtime_credit.py for the bug and the fix.
"""
from typing import Dict, Sequence, Tuple


def majority_slot(slots: Sequence[int], fallback: int) -> Tuple[int, float]:
    """The slot held for most of an interval's tics (ties -> the LATEST of
    the tied slots), and the fraction of tics it was held. An empty interval
    returns (fallback, 1.0)."""
    if not slots:
        return int(fallback), 1.0
    counts: Dict[int, int] = {}
    last: Dict[int, int] = {}
    for i, s in enumerate(slots):
        s = int(s)
        counts[s] = counts.get(s, 0) + 1
        last[s] = i
    best = max(counts, key=lambda s: (counts[s], last[s]))
    return best, counts[best] / float(len(slots))
