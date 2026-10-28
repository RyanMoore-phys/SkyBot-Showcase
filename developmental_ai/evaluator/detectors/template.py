"""Detector TEMPLATE: copy to detectors/<game>.py and fill in (WP-16, ARC-005).

Steps for a new game:
  1. Copy this file to `developmental_ai/evaluator/detectors/<game>.py`.
  2. Set GAME and list the checkpoints in difficulty order (`specs`), declaring
     prerequisites (a later checkpoint is scored only after its prerequisites).
  3. Implement `_holds` against the game's `info["evaluator"]` keys (RED
     channel). Counters that only grow ("kills", "items") are ideal: the first
     time `value >= n` is the first attainment.
  4. Keep `build()`.
Nothing in this file may be imported by learning code (ADR-002); the checkpoint
ladder is the OWNER's to define (ASS-007 section 15). Ids and descriptions here
are placeholders, not a ladder.
"""
from __future__ import annotations

from typing import Any, Dict, List

from developmental_ai.evaluator.detector import CheckpointSpec, Detector


class TemplateDetector(Detector):
    GAME = "template"

    def specs(self) -> List[CheckpointSpec]:
        return [
            CheckpointSpec("first_effect", "any first counter above zero"),
            CheckpointSpec("repeat_effect", "the counter reaches three",
                           prerequisites=("first_effect",)),
        ]

    def _holds(self, spec_id: str, ev: Dict[str, Any]) -> bool:
        n = float(ev.get("example_counter", 0.0))
        return n >= 1.0 if spec_id == "first_effect" else n >= 3.0


def build() -> Detector:
    return TemplateDetector()
