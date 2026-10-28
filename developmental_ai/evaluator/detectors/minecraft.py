"""Minecraft checkpoint ladder: NOT DEFINED (ASS-007 section 15). Evaluator side.

The owner sets the ladder later ("when we get there"): progress through the game
combined with survivability and thrivability. The six-step ladder in ASS-007
12.3 was an illustrative proposal for the progress dimension only and is NOT
this ladder. Until the owner defines it, asking for the Minecraft detector must
fail LOUDLY rather than score anything, so no result can be produced against an
invented ladder (learned, not declared; no silent fallback).

Release (CLAUDE.md 4.1): the owner supplies the ladder; replace `build()` with a
Detector subclass (see detectors/template.py) and delete the raise.
"""
from __future__ import annotations

from developmental_ai.evaluator.registry import DetectorNotDefined


def build():
    raise DetectorNotDefined(
        "Minecraft checkpoint ladder not defined: the owner sets progress, "
        "survivability and thrivability checkpoints later (ASS-007 section 15). "
        "Fill developmental_ai/evaluator/detectors/minecraft.py when it is set.")
