"""Frozen-protocol constants and run planning (ASS-007 sections 5, 12, 14). Evaluator side.

Pure Python. Nothing here launches anything; it only says WHICH runs exist, in
what order, with what split. The runner (runner.py) executes the plan and the
analysis (stats.py) judges the results.

Defaults are the protocol's (ASS-007 12.2, 12.5, 14.3, 14.5):
    main seeds 1..5; A/A seeds 11..20 (ten, accepted by the owner, section 15);
    one hour = one complete seed run, split 2/3 source : 1/3 target
    (40 min : 20 min, identical in every arm, never tuned per change);
    evaluation windows of five minutes (twelve per hour).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

MAIN_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
AA_SEEDS: Tuple[int, ...] = tuple(range(11, 21))
TOTAL_MINUTES = 60.0
SOURCE_FRAC = 2.0 / 3.0
WINDOW_S = 300.0


@dataclass(frozen=True)
class Phase:
    name: str          # "source" | "target"
    minutes: float


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    arm: str
    seed: int
    phases: Tuple[Phase, ...]


def split_minutes(total: float = TOTAL_MINUTES,
                  source_frac: float = SOURCE_FRAC) -> Tuple[float, float]:
    if not 0.0 < source_frac < 1.0:
        raise ValueError("source_frac must be in (0, 1)")
    src = round(total * source_frac, 6)
    return src, round(total - src, 6)


def plan_runs(arms: Sequence[str], seeds: Sequence[int],
              total_minutes: float = TOTAL_MINUTES,
              source_frac: float = SOURCE_FRAC,
              fresh_control_arms: Sequence[str] = ()) -> List[RunSpec]:
    """Seed-major order with arms interleaved, so host drift hits every arm of a
    seed alike. A `fresh_control_arm` has no source phase (ASS-007 section 4: a
    brain with no source phase, run in the target game); it gets the target
    budget only."""
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate arm names")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    src, tgt = split_minutes(total_minutes, source_frac)
    out: List[RunSpec] = []
    for seed in seeds:
        for arm in arms:
            phases = ((Phase("target", tgt),) if arm in fresh_control_arms
                      else (Phase("source", src), Phase("target", tgt)))
            out.append(RunSpec(f"{arm}_s{int(seed)}", arm, int(seed), phases))
    return out


def plan_summary(runs: Sequence[RunSpec]) -> Dict[str, object]:
    total = sum(p.minutes for r in runs for p in r.phases)
    return {"runs": len(runs), "run_minutes": total, "run_hours": round(total / 60.0, 2),
            "arms": sorted({r.arm for r in runs}), "seeds": sorted({r.seed for r in runs})}
