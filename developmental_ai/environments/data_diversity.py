"""WP-6 (GOV-009): data diversity, as CONFIGURATION ONLY. BUILT, UNTESTED.

Pure functions, no engine import. Three things, all absent-by-default (a
game YAML that sets none of the keys below composes and runs exactly as
before):

1. PER-STREAM MAP ASSIGNMENT (`game.map_assignment`). The composer resolves,
   for the k-th stream of a game, the list of maps that stream plays
   (`stream_maps`) and a time schedule of re-assignments (`map_schedule`:
   `[{"at_s": seconds since the stream started, "maps": [...]}, ...]`). The
   adapter applies the schedule at episode starts only.

       map_assignment:
         mode: spread            # round-robin partition of the source maps
         # per_stream: {0: [MAP01, MAP02], 3: [MAP09]}   # explicit lists win
         schedule:               # optional: the assignment changes over a run
           - {at_s: 1800, offset: 1}    # stream k takes the list of k+1
           - {at_s: 3600, offset: 2}

2. A HELD-OUT MAP LIST (`game.held_out_maps`) that the SOURCE phase can never
   use. `configure_stream` raises `LeakError` if a source-phase stream list or
   schedule names a held-out map, and drops held-out maps from the generic
   `maps` offer. The TARGET phase (`--phase target` on the composer) plays
   the held-out maps only. Held-out Doom maps are a PROXY for a transfer
   target (gate G1) and are labelled so.

3. VISUAL VARIATION OPTIONS (`game.visual`, `game.visual_per_stream`): engine
   settings (skill level, rendering switches, monsters on or off, extra
   engine arguments) applied BEFORE the engine starts. They configure the
   game; nothing is read back from it (ADR-002).

LEARNED-NOT-DECLARED: this only varies WHICH experience the learner gets;
no map has a meaning, a goal or a name the learner sees (the evaluator line
carries the map name for evaluation only, as before).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple


class LeakError(ValueError):
    """A held-out map would be used by the source phase."""


# ViZDoom render switches the YAML may set (name -> setter). Resolved by
# getattr on the game object, so an engine without one raises a clear error.
_RENDER = {
    "render_hud": "set_render_hud",
    "render_minimal_hud": "set_render_minimal_hud",
    "render_weapon": "set_render_weapon",
    "render_crosshair": "set_render_crosshair",
    "render_decals": "set_render_decals",
    "render_particles": "set_render_particles",
    "render_effects_sprites": "set_render_effects_sprites",
    "render_messages": "set_render_messages",
    "render_corpses": "set_render_corpses",
    "render_screen_flashes": "set_render_screen_flashes",
    "render_all_frames": "set_render_all_frames",
}
KNOWN_VISUAL = set(_RENDER) | {"skill", "nomonsters", "game_args"}


def _spread(maps: Sequence[str], n: int) -> List[List[str]]:
    """Round-robin partition: stream k gets maps[k], maps[k+n], ...; with
    fewer maps than streams the lists wrap so no stream is empty."""
    maps = list(maps)
    if not maps:
        return [[] for _ in range(n)]
    out = [maps[k::n] for k in range(n)]
    return [o if o else [maps[k % len(maps)]] for k, o in enumerate(out)]


def _resolve(spec: Dict[str, Any], maps: Sequence[str], n: int
             ) -> List[List[str]]:
    base = _spread(maps, n) if str(spec.get("mode", "spread")) == "spread" \
        else [list(maps) for _ in range(n)]
    off = int(spec.get("offset", 0) or 0)
    if off and n > 0:
        base = [base[(k + off) % n] for k in range(n)]
    for k, lst in (spec.get("per_stream") or {}).items():
        if 0 <= int(k) < n:
            base[int(k)] = list(lst)
    return base


def configure_stream(game_cfg: Dict[str, Any], k: int, n: int,
                     phase: str = "source") -> Dict[str, Any]:
    """Fill the k-th (of n) stream's resolved fields in `game_cfg` (a
    per-stream copy). Raises LeakError / ValueError on a bad config. A config
    with none of the WP-6 keys is returned unchanged."""
    held = [str(m) for m in (game_cfg.get("held_out_maps") or [])]
    ma = game_cfg.get("map_assignment")
    vis = game_cfg.get("visual")
    vps = game_cfg.get("visual_per_stream")
    if not (held or ma or vis or vps) and phase == "source":
        return game_cfg
    offer = [str(m) for m in (game_cfg.get("maps") or [])]
    if phase == "target":
        if not held:
            raise ValueError("phase 'target' needs game.held_out_maps")
        game_cfg["maps"] = list(held)
        game_cfg["stream_maps"] = _spread(held, n)[k]
        game_cfg["phase"] = "target"
    else:
        offer = [m for m in offer if m not in held]
        if not offer and (game_cfg.get("maps") or held):
            raise LeakError("no source maps are left after removing the "
                            "held-out list")
        game_cfg["maps"] = offer
        game_cfg["phase"] = "source"
        if ma:
            lists = _resolve(ma, offer, n)
            game_cfg["stream_maps"] = lists[k]
            sched = []
            for ph in (ma.get("schedule") or []):
                sched.append({"at_s": float(ph["at_s"]),
                              "maps": _resolve(ph, offer, n)[k]})
            sched.sort(key=lambda p: p["at_s"])
            if sched:
                game_cfg["map_schedule"] = sched
    if held:
        game_cfg["held_out_maps"] = held
    check_no_leak(game_cfg)
    if vis or vps:
        v = dict(vis or {})
        v.update((vps or {}).get(k, {}) or (vps or {}).get(str(k), {}) or {})
        bad = sorted(set(v) - KNOWN_VISUAL)
        if bad:
            raise ValueError(f"game.visual: unknown option(s) {bad} "
                             f"(known: {sorted(KNOWN_VISUAL)})")
        game_cfg["visual_stream"] = v
    return game_cfg


def check_no_leak(game_cfg: Dict[str, Any]) -> None:
    """The no-leakage rule, callable from the composer, the adapter and a
    test: in the source phase no map this stream can ever play is held out."""
    if str(game_cfg.get("phase", "source")) != "source":
        return
    held = set(str(m) for m in (game_cfg.get("held_out_maps") or []))
    if not held:
        return
    used = set(str(m) for m in (game_cfg.get("stream_maps") or []))
    used |= set(str(m) for m in (game_cfg.get("maps") or []))
    for p in (game_cfg.get("map_schedule") or []):
        used |= set(str(m) for m in p.get("maps", []))
    bad = sorted(used & held)
    if bad:
        raise LeakError(f"held-out map(s) {bad} are reachable in the source "
                        f"phase")


def adapter_maps(game_cfg: Dict[str, Any]) -> Tuple[List[str], List[dict]]:
    """(maps this stream starts on, time schedule) for the adapter. Same
    result as before when no WP-6 key is set."""
    check_no_leak(game_cfg)
    maps = list(game_cfg.get("stream_maps") or game_cfg.get("maps")
                or ["MAP01"])
    return maps, list(game_cfg.get("map_schedule") or [])


def schedule_maps(schedule: Sequence[dict], elapsed_s: float
                  ) -> Optional[List[str]]:
    """The maps of the latest schedule phase whose `at_s` has passed, or None
    before the first phase."""
    cur = None
    for p in schedule:
        if elapsed_s >= float(p["at_s"]):
            cur = list(p["maps"])
    return cur


def apply_visual(g, vzd, visual: Optional[Dict[str, Any]]) -> None:
    """Apply `game.visual_stream` to a ViZDoom game object BEFORE init().
    Nothing is called (so nothing changes) when `visual` is empty."""
    if not visual:
        return
    for key, setter in _RENDER.items():
        if key in visual:
            getattr(g, setter)(bool(visual[key]))
    if "skill" in visual:
        g.set_doom_skill(int(visual["skill"]))
    if visual.get("nomonsters"):
        g.add_game_args("+nomonsters 1")
    if visual.get("game_args"):
        g.add_game_args(str(visual["game_args"]))
