"""Evaluator-only progress milestones for held-out Doom maps (ASS-007 §17).

The detector reads only ViZDoom's RED evaluator record. It is used for the
Doom development proxy, not for the later cross-game program evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, List

from developmental_ai.evaluator.detector import CheckpointSpec, Detector


class DoomDetector(Detector):
    GAME = "doom"

    def specs(self) -> List[CheckpointSpec]:
        return [
            CheckpointSpec("first_kill", "first Doom kill"),
            CheckpointSpec("first_pickup", "first Doom item pickup"),
            CheckpointSpec("first_exit", "first completed map exit"),
        ]

    def _holds(self, spec_id: str, ev: Dict[str, Any]) -> bool:
        # kills/items: the adapter's CUMULATIVE totals (the game's own KILLCOUNT and
        # ITEMCOUNT reset on death or a new level, so they are per-life and a sparse log line
        # can read 0 after earlier kills). Fall back to the per-life value for old logs.
        keys = {"first_kill": ("kills_total", "kills"),
                "first_pickup": ("items_total", "items"),
                "first_exit": ("exits",)}
        for k in keys[spec_id]:
            if k in ev:
                return float(ev.get(k, 0) or 0) >= 1.0
        return False


def build() -> Detector:
    return DoomDetector()
