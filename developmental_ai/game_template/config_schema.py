"""Game configuration schema and validator (WP-16, ARC-005). Pure Python.

`validate_game_config(doc, action_slots)` returns a list of problems for a
parsed configs/games/<g>.yaml (the whole document, with its `game:` block).
It checks what tools/compose_run_config.py and the contract need, once, in
words: the file can be vetted before any launch. It does not replace the
composer's own checks; it mirrors them so a new game fails early and clearly.

Schema (keys under `game:`):

  name        str, == file stem                                  required
  env_name    str, make_env dispatch name                        required
  backend     "plugin" (adapter class below) | "player" | "vizdoom" | "minerl"
  adapter     "pkg.module:Class" (required when backend is plugin)
  timing      "synchronous" | "realtime"                          default synchronous
  streams     int >= 1                                           default 1
  controller  list of slot dicts; slot 0 == {}; len <= action_slots
              (player games use the player controller grammar instead)
  audio       {enabled: bool, sample_rate: int}                  optional
  install     {apt, apt_optional, debs, pip, roms, files, note}   optional
  requires    list of files the game cannot start without        optional
  stuck / checkpoint   as Doom's                                 optional
"""
from __future__ import annotations

from typing import Any, Dict, List

from developmental_ai.game_template.contract import TIMINGS

BACKENDS = ("plugin", "player", "vizdoom", "minerl")
INSTALL_KEYS = ("apt", "apt_optional", "debs", "pip", "roms", "files", "note")


def validate_game_config(doc: Dict[str, Any], action_slots: int,
                         stem: str = "") -> List[str]:
    p: List[str] = []
    g = (doc or {}).get("game")
    if not isinstance(g, dict):
        return ["missing `game:` block"]
    name = g.get("name")
    if not name or (stem and name != stem):
        p.append(f"game.name {name!r} must equal the file stem {stem!r}")
    if not g.get("env_name"):
        p.append("game.env_name is required")
    be = g.get("backend")
    if be not in BACKENDS:
        p.append(f"game.backend {be!r} not in {BACKENDS}")
    if be == "plugin":
        ad = str(g.get("adapter") or "")
        if ":" not in ad or not all(ad.split(":", 1)):
            p.append("backend plugin needs adapter: 'pkg.module:Class'")
    if g.get("timing", "synchronous") not in TIMINGS:
        p.append(f"game.timing must be one of {TIMINGS}")
    try:
        if int(g.get("streams", 1)) < 1:
            p.append("game.streams must be >= 1")
    except (TypeError, ValueError):
        p.append("game.streams must be an integer")
    if be != "player":
        ctrl = g.get("controller")
        if not ctrl:
            p.append("game.controller is required (slot table, slot 0 == {})")
        else:
            if ctrl[0]:
                p.append("controller slot 0 must be {} (the no-op)")
            if len(ctrl) > int(action_slots):
                p.append(f"{len(ctrl)} slots > universal.action_slots={action_slots}")
    inst = g.get("install") or {}
    if not isinstance(inst, dict):
        p.append("game.install must be a mapping")
    else:
        bad = [k for k in inst if k not in INSTALL_KEYS]
        if bad:
            p.append(f"game.install has unknown keys {bad} (allowed: {list(INSTALL_KEYS)})")
    return p
