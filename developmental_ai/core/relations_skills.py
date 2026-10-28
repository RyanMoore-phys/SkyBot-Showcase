"""RW-11 Stage A: read-only skill trend and option-binding channels."""
from __future__ import annotations

from typing import Optional

from developmental_ai.core.relations import RelationRegistry


def _summary(loop) -> dict:
    bank = getattr(loop, "skill_bank", None)
    if bank is None:
        return {}
    executor = getattr(loop, "option_executor", None)
    slots = getattr(getattr(executor, "bank", None), "slots", ())
    bound = sum(1 for slot in slots if slot is not None)
    try:
        return bank.trend_summary(bound)
    except Exception:
        return {}


def _falling_fraction(loop) -> Optional[float]:
    value = _summary(loop).get("falling_skill_fraction")
    return None if value is None else float(value)


def _bound_count(loop) -> Optional[float]:
    value = _summary(loop).get("bound_option_count")
    return None if value is None else float(value)


def register(reg: RelationRegistry) -> None:
    R = reg.register
    R("OU-03", "PROC", "SKILL", "persistent skill competence trend",
      "skill library", "falling-skill fraction", "fraction", "per segment",
      "data", probe=_falling_fraction, group="skills")
    R("OU-04", "AUTO", "SKILL", "resident option bindings",
      "skill library", "bound option count", "options", "per segment",
      "data", probe=_bound_count, group="skills")
