"""Checkpoint detector for the synthetic proof game (WP-16). Evaluator side only.

The ladder here is for TESTING THE MACHINERY, not a program ladder.
"""
from __future__ import annotations

from typing import Any, Dict, List

from developmental_ai.evaluator.detector import CheckpointSpec, Detector


class SyntheticDetector(Detector):
    GAME = "synthetic"

    def specs(self) -> List[CheckpointSpec]:
        return [
            CheckpointSpec("moved", "visited at least 4 distinct cells"),
            CheckpointSpec("touched", "touched a cell once", prerequisites=("moved",)),
            CheckpointSpec("collected", "collected 3 items", prerequisites=("touched",)),
        ]

    def _holds(self, spec_id: str, ev: Dict[str, Any]) -> bool:
        if spec_id == "moved":
            return float(ev.get("cells_visited", 0)) >= 4
        if spec_id == "touched":
            return float(ev.get("touches", 0)) >= 1
        return float(ev.get("collected", 0)) >= 3


def build() -> Detector:
    return SyntheticDetector()
