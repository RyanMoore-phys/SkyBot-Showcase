#!/usr/bin/env python3
"""Compose the run config for SkyBot's universal brain from the enabled games.

docs/architecture/arc-004-universal-games.md §3. Run by scripts/launch_skybot.sh at every launch
when runlogs/games.enabled exists:

    base       configs/minecraft_skybot.yaml   the tuned learner
  + brain      configs/universal.yaml          one brain's shapes and paths
  + primary    configs/games/<stream-0 game>.yaml `primary:` block
  + streams    one parallel_envs.streams entry per stream of every game
  = out        runlogs/run_config.yaml

    python tools/compose_run_config.py --games minecraft,doom
    python tools/compose_run_config.py --games-file runlogs/games.enabled

Refuses (exit 2, message names the problem) rather than writing a config the
brain cannot run: unknown game, controller slot 0 not the no-op, more
controller slots than the brain has, no games at all.
"""

from __future__ import annotations

import argparse
import json
import copy
import os
import sys
from typing import Any, Dict, List

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ComposeError(ValueError):
    pass


def deep_merge(base: Dict, over: Dict) -> Dict:
    """Maps merge key by key; everything else (lists included) replaces."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_games_file(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip().lower()
            if line and line not in out:
                out.append(line)
    return out


def order_games(games: List[str], primary_order: List[str]) -> List[str]:
    """primary_order first (in that order), then the rest alphabetically."""
    first = [g for g in primary_order if g in games]
    return first + sorted(g for g in games if g not in first)


def read_scale(path) -> Dict[str, Any]:
    if not path or not os.path.isfile(str(path)):
        return {}
    try:
        with open(str(path)) as f:
            return dict(yaml.safe_load(f) or {})
    except (OSError, yaml.YAMLError, ValueError):
        return {}


def seed_scale(init: Dict[str, Any]) -> Dict[str, Any]:
    """The scale record for a NEW brain from `universal.initial_scale`."""
    sys.path.insert(0, ROOT)
    from developmental_ai.infra.autoscaler import SIZE_LADDER
    size = str(init["size"])
    ladder = dict(SIZE_LADDER)
    if size not in ladder:
        raise ComposeError(f"universal.initial_scale.size {size!r} not in {sorted(ladder)}")
    det, st, cl, enc = ladder[size]
    out = {"size": size, "dims": {"det": det, "stoch": st, "classes": cl, "encoder": enc}}
    if init.get("streams"):
        out["streams"] = int(init["streams"])
    return out


def record_scale(path: str, scale: Dict[str, Any]) -> None:
    """Write a seeded scale record once (never over an existing file)."""
    if not path or os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(scale, seeded_from="universal.initial_scale"), f, indent=1)
    os.replace(tmp, path)


# Minecraft's limit is the Paper server, not the host (CLAUDE.md §6)
MAX_STREAMS = {"minerl": 2}


def _apply_streams(docs, ordered, total: int) -> None:
    """Distribute `total` streams: games keep their own count, and the
    difference goes to (or comes from) the uncapped games, first one first."""
    have = sum(int(docs[g]["game"].get("streams", 1)) for g in ordered)
    delta = total - have
    for g in ordered:
        if delta == 0:
            break
        game = docs[g]["game"]
        cap = MAX_STREAMS.get(game.get("backend"))
        cur = int(game.get("streams", 1))
        new = max(1, cur + delta)
        if cap is not None:
            new = min(new, cap)
        game["streams"] = new
        delta -= new - cur


def load_game(name: str, games_dir: str) -> Dict[str, Any]:
    path = os.path.join(games_dir, f"{name}.yaml")
    if not os.path.isfile(path):
        known = sorted(f[:-5] for f in os.listdir(games_dir) if f.endswith(".yaml"))
        raise ComposeError(f"unknown game {name!r}: no {path} (known: {known})")
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    g = doc.get("game") or {}
    if g.get("name") != name:
        raise ComposeError(f"{path}: game.name {g.get('name')!r} != {name!r}")
    if int(g.get("streams", 1)) < 1:
        raise ComposeError(f"{path}: game.streams must be >= 1")
    return doc


def compose(games: List[str], base_path: str, universal_path: str,
            games_dir: str, seed_new_brain: bool = False,
            phase: str = "source") -> Dict[str, Any]:
    if not games:
        raise ComposeError("no games enabled")
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(universal_path) as f:
        uni = yaml.safe_load(f)
    cfg = deep_merge(base, uni)
    ucfg = cfg.get("universal") or {}
    slots = int(ucfg.get("action_slots", 0))
    if slots < 1:
        raise ComposeError("universal.action_slots must be set")
    ordered = order_games(games, list(ucfg.get("primary_order") or []))
    docs = {g: load_game(g, games_dir) for g in ordered}

    # ---- THE BRAIN'S CURRENT SIZE (stage D, 2026-10-17) -------------------
    # tools/grow_brain.py resizes world_model.pt between runs and records the
    # result in the scale file; this run must BUILD that size or the resume
    # pre-flight refuses the checkpoint. Streams follow the same record. A
    # missing or unreadable file changes nothing.
    scale = read_scale(ucfg.get("scale_file"))
    # A NEW BRAIN STARTS SMALL (user, 2026-10-18: tiny x 8, the measured
    # Stage C start). Only when there is no scale file AND no saved world
    # model: an existing brain without a record keeps building what it was
    # saved as (a different size would be refused at resume and start fresh).
    # The launch records the seed in the scale file (--record-scale), so a
    # later edit of initial_scale never resizes a brain that already exists.
    # Opt-in (seed_new_brain): a plain compose — tests, experiment arms — is
    # unchanged and does not depend on what brain sits in the cwd.
    init = ucfg.get("initial_scale") or {}
    ckpt_wm = os.path.join(str((cfg.get("loop") or {}).get("log_dir") or ""),
                           "checkpoints", "world_model.pt")
    if (seed_new_brain and not scale and init.get("size")
            and not os.path.isfile(ckpt_wm)):
        scale = seed_scale(init)
        ucfg["_scale_seeded"] = scale
    if scale.get("dims"):
        d = scale["dims"]
        wm = cfg.setdefault("world_model", {})
        wm["deterministic_size"] = int(d["det"])
        wm["stochastic_size"] = int(d["stoch"])
        wm["stochastic_classes"] = int(d["classes"])
        # the encoder/decoder width is part of the size too (WorldModel has
        # one hidden_dim): without it "tiny" built a 768-wide encoder — ~6x
        # the parameters of the tiny that was measured
        if d.get("encoder"):
            wm["encoder_hidden"] = wm["decoder_hidden"] = int(d["encoder"])
    if scale.get("streams"):
        _apply_streams(docs, ordered, int(scale["streams"]))

    streams: List[Dict[str, Any]] = []
    for name in ordered:
        g = docs[name]["game"]
        if g.get("backend") == "player":
            # keys/buttons/mouse controller: whitelist, OS-key refusal,
            # slot 0 = release-all, no one-way doors (CLAUDE.md §4.5)
            sys.path.insert(0, ROOT)
            from developmental_ai.environments.player.controller import (
                ControllerError, validate)
            from developmental_ai.environments.player.launchers import (
                LauncherError, validate_launcher)
            try:
                parsed = validate(g, slots)
                # exec / steam / browser: steam client mode = 1 stream, browser
                # shortcuts are one-way doors (launchers.py)
                validate_launcher(g, parsed)
            except (ControllerError, LauncherError) as e:
                raise ComposeError(f"{name}: {e}")
        ctrl = g.get("controller")
        if ctrl is not None and g.get("backend") != "player":
            if not ctrl or ctrl[0]:
                raise ComposeError(f"{name}: controller slot 0 must be {{}} "
                                   f"(every game's slot 0 is its no-op)")
            if len(ctrl) > slots:
                raise ComposeError(
                    f"{name}: {len(ctrl)} controller slots > the brain's "
                    f"universal.action_slots={slots}")
        _n_g = int(g.get("streams", 1))
        for _k in range(_n_g):
            _gc = copy.deepcopy(g)
            # WP-6 (GOV-011): per-stream map assignment, held-out maps and
            # visual options. A game that sets none of those keys, in the
            # source phase, is returned unchanged.
            sys.path.insert(0, ROOT)
            from developmental_ai.environments.data_diversity import (
                LeakError, configure_stream)
            try:
                _gc = configure_stream(_gc, _k, _n_g, phase)
            except (LeakError, ValueError) as e:
                raise ComposeError(f"{name}: {e}")
            streams.append({"game": name, "env_name": g["env_name"],
                            "backend": g.get("backend"),
                            "game_cfg": _gc})

    primary = ordered[0]
    cfg = deep_merge(cfg, docs[primary].get("primary") or {})
    if docs[primary]["game"].get("backend") != "minerl":
        cfg["environment"]["name"] = docs[primary]["game"]["env_name"]
    par = cfg.setdefault("parallel_envs", {})
    par["enabled"] = True
    par["num_envs"] = len(streams)
    par["streams"] = streams
    cfg["universal"]["games"] = ordered
    if len(streams) < 2:
        # num_envs 1 drops into the unwired legacy single-env body (CLAUDE.md
        # §4.2) — the pre-Wave-1 agent. One game with one stream is not
        # "SkyBot on one client"; refuse instead of silently running it.
        raise ComposeError("a universal run needs >= 2 streams in total "
                           "(num_envs 1 selects the legacy single-env body)")
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--games", help="comma-separated, e.g. minecraft,doom")
    src.add_argument("--games-file", help="one game per line (runlogs/games.enabled)")
    ap.add_argument("--base", default=os.path.join(ROOT, "configs/minecraft_skybot.yaml"))
    ap.add_argument("--universal", default=os.path.join(ROOT, "configs/universal.yaml"))
    ap.add_argument("--games-dir", default=os.path.join(ROOT, "configs/games"))
    ap.add_argument("--out", default="runlogs/run_config.yaml")
    ap.add_argument("--phase", choices=("source", "target"), default="source",
                    help="WP-6: 'source' can never use game.held_out_maps; "
                         "'target' plays only them")
    ap.add_argument("--record-scale", action="store_true",
                    help="the launch: a NEW brain takes universal.initial_scale, recorded in the scale file")
    a = ap.parse_args(argv)
    games = ([g.strip().lower() for g in a.games.split(",") if g.strip()]
             if a.games else read_games_file(a.games_file))
    try:
        cfg = compose(games, a.base, a.universal, a.games_dir,
                      seed_new_brain=a.record_scale, phase=a.phase)
    except (ComposeError, OSError, yaml.YAMLError) as e:
        print(f"COMPOSE FAILED: {e}", file=sys.stderr)
        return 2
    seeded = cfg["universal"].pop("_scale_seeded", None)
    if seeded:
        record_scale(cfg["universal"].get("scale_file"), seeded)
        print(f"  new brain: seeded {cfg['universal'].get('scale_file')} at "
              f"{seeded['size']} x {seeded.get('streams', '?')} (universal.initial_scale)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("# GENERATED by tools/compose_run_config.py — do not edit.\n"
                f"# games (stream order): {', '.join(cfg['universal']['games'])}\n"
                f"# streams: {[s['game'] for s in cfg['parallel_envs']['streams']]}\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, a.out)
    print(f"COMPOSED {a.out}: games={','.join(cfg['universal']['games'])} "
          f"streams={len(cfg['parallel_envs']['streams'])} "
          f"primary={cfg['universal']['games'][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
