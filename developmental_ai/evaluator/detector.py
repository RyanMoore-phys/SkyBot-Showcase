"""Checkpoint detector base (ASS-007 12.4, 14.5). Evaluator side only.

A detector turns a stream of RED-channel dicts (`info["evaluator"]`) into
FIRST-ATTAINMENT records for an ordered list of checkpoints. A game's detector
subclasses `Detector`, lists its `CheckpointSpec`s and implements `_holds`.

Rules implemented here once, for every game (ASS-007 14.5):
  * first attainment only; times are measured from target entry (`begin_phase`);
  * a checkpoint is scored only after all its prerequisites were attained;
  * a checkpoint already true when the phase began is flagged `at_entry`
    (source-phase leakage or a trivial checkpoint) and excluded from scoring;
  * unreached checkpoints stay absent: the analysis right-censors them.
Detecting never changes the run: `observe` only reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CheckpointSpec:
    id: str
    description: str
    prerequisites: Tuple[str, ...] = ()


class Detector:
    GAME = ""

    def __init__(self) -> None:
        self._specs: List[CheckpointSpec] = list(self.specs())
        ids = [s.id for s in self._specs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.GAME}: duplicate checkpoint ids")
        for s in self._specs:
            for q in s.prerequisites:
                if q not in ids[:ids.index(s.id)]:
                    raise ValueError(f"{s.id}: prerequisite {q!r} must be listed earlier")
        self.begin_phase(None)

    # ------------------------------------------------------------ to override
    def specs(self) -> List[CheckpointSpec]:
        raise NotImplementedError("list the ordered checkpoints of this game")

    def _holds(self, spec_id: str, ev: Dict[str, Any]) -> bool:
        raise NotImplementedError("does checkpoint `spec_id` hold in evaluator dict `ev`?")

    # ------------------------------------------------------------------ api
    def begin_phase(self, ev_at_entry: Optional[Dict[str, Any]], step: int = 0,
                    t_s: float = 0.0) -> None:
        """Start (or restart) a phase. Checkpoints already true now are flagged."""
        self.step0, self.t0 = int(step), float(t_s)
        self.attained: Dict[str, Tuple[int, float]] = {}
        self.at_entry: List[str] = []
        if ev_at_entry is not None:
            for s in self._specs:
                if self._holds(s.id, ev_at_entry):
                    self.at_entry.append(s.id)

    def observe(self, ev: Dict[str, Any], step: int, t_s: float) -> List[str]:
        """Return checkpoint ids first attained at this observation."""
        new: List[str] = []
        for s in self._specs:
            if s.id in self.attained or s.id in self.at_entry:
                continue
            if any(q not in self.attained and q not in self.at_entry
                   for q in s.prerequisites):
                continue
            if self._holds(s.id, ev or {}):
                self.attained[s.id] = (int(step) - self.step0, float(t_s) - self.t0)
                new.append(s.id)
        return new

    def result(self) -> Dict[str, Any]:
        """Serialisable summary: per checkpoint (decisions, seconds) or None."""
        return {"game": self.GAME,
                "checkpoints": [s.id for s in self._specs],
                "attained": {k: {"decisions": v[0], "seconds": v[1]}
                             for k, v in self.attained.items()},
                "at_entry": list(self.at_entry),
                "not_attained": [s.id for s in self._specs
                                 if s.id not in self.attained and s.id not in self.at_entry]}
