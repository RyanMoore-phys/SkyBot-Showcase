"""Detector registry: one detector per game, found by name (ASS-007 12.4).

Adding a game's detector = adding `developmental_ai/evaluator/detectors/<game>.py`
exposing `build() -> Detector`. Nothing here changes. Evaluator side only.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict

from developmental_ai.evaluator.detector import Detector

_REGISTERED: Dict[str, Callable[[], Detector]] = {}


class DetectorNotDefined(RuntimeError):
    """Raised loudly when a game has no (usable) detector. Never a silent skip."""


def register(game: str, factory: Callable[[], Detector]) -> None:
    _REGISTERED[game.lower()] = factory


def get_detector(game: str) -> Detector:
    g = game.lower()
    if g in _REGISTERED:
        return _REGISTERED[g]()
    try:
        mod = importlib.import_module(f"developmental_ai.evaluator.detectors.{g}")
    except ImportError as e:
        raise DetectorNotDefined(
            f"no checkpoint detector for game {g!r}: add "
            f"developmental_ai/evaluator/detectors/{g}.py with build() ({e})")
    return mod.build()
