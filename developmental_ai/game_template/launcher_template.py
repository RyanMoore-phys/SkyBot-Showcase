"""Launcher and install skeleton for a new game (WP-16, ARC-005).

Most games need NO code here:

  * An in-process engine (like ViZDoom) is launched by its adapter; there is
    nothing to launch. Declare `install:` in the game YAML and
    tools/game_install.py installs it (apt, debs, pip, roms, files).
  * Any unmodified game seen through a screen and keyboard uses
    `backend: player` with a `launcher:` block (exec / steam / browser,
    player/launchers.py), which already has preflight, per-seat teardown and
    one-way-door checks.

Write a `GameLauncher` subclass ONLY for an engine that needs a custom start
and stop that neither of those covers. The contract:

    preflight(game_cfg)   -> list[str]  what is missing; [] = ready. Called by
                             the launch preflight; a non-empty list REFUSES the
                             launch with the strings printed (fail CLOSED:
                             an exception here is a refusal, never "ready").
    launch(game_cfg, stream_index) -> handle   start one instance, tagged with
                             the env var SKYBOT_SEAT=<unique id> so teardown
                             can find re-parented children.
    stop(handle)          kill by that marker, idempotent, never raises.

Guard-becomes-latch (CLAUDE.md 4.1): every refusal in preflight must name the
action that fixes it (the game's `install` action).
"""
from __future__ import annotations

from typing import Any, Dict, List


class GameLauncher:
    def preflight(self, game_cfg: Dict[str, Any]) -> List[str]:
        raise NotImplementedError("list what is missing; [] means ready")

    def launch(self, game_cfg: Dict[str, Any], stream_index: int) -> Any:
        raise NotImplementedError("start one instance with SKYBOT_SEAT set")

    def stop(self, handle: Any) -> None:
        raise NotImplementedError("kill by the SKYBOT_SEAT marker; idempotent")
