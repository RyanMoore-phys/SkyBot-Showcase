"""The shared environment contract, as checkable code (WP-16, ARC-005).

Pure Python + numpy: no torch, no gymnasium import at module load, no engine.
`check_*` functions return a list of PROBLEM strings (empty = contract holds),
so a test can assert emptiness and a human can read what is wrong.

What every game adapter must present (taken from the ViZDoom and Player
adapters, which already satisfy it):

  observation   flat float32, length 3*S*S, values in [0, 1] (CHW order)
  actions       Discrete(n) with n <= universal.action_slots; slot 0 is the
                no-op ({}); `valid_mask` (len n, bool) with mask[0] True; an
                invalid or out-of-range slot executes slot 0 (never a crash)
  reward        the adapter pays 0.0: reward is earned through the ordinary
                game interaction by learned drives, never read from the engine
  info keys     game, world (dict of BODY senses), proprio (float32, fixed
                width), gui_open, env_restarted, rewind; optional audio
                (float32 (2, mels, frames) in [0,1]); evaluator (RED)
  RED channel   everything only an evaluator may see lives under
                info["evaluator"] (ADR-002). Learning code never reads it.
  life edges    a life ending => terminated=True; a forced discontinuity
                (rewind, engine rebuilt) => env_restarted=True (and truncated)
  timing        "synchronous": the frame returned is the outcome of the action
                just taken; the adapter MUST NOT report executed_action.
                "realtime": the game runs on its own clock, so the frame is the
                outcome of the PREVIOUS held action. The adapter MUST report
                executed_action (slot that drove the returned interval),
                executed_purity in [0,1], held_action (slot now in flight),
                and expose a `held_action` attribute (find_attr reads it).
                (core/realtime_credit.py: the action-lag defect came from
                exactly this difference.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

TIMINGS = ("synchronous", "realtime")
RED_KEY = "evaluator"
REQUIRED_INFO = ("game", "world", "proprio", "gui_open", "env_restarted", "rewind")
REALTIME_INFO = ("executed_action", "executed_purity", "held_action")


def check_observation(obs: Any, image_size: int, channels: int = 3) -> List[str]:
    p: List[str] = []
    a = np.asarray(obs)
    want = channels * image_size * image_size
    if a.ndim != 1 or a.shape[0] != want:
        p.append(f"observation shape {a.shape} != ({want},) (3 x {image_size} x {image_size} flat)")
        return p
    if a.dtype != np.float32:
        p.append(f"observation dtype {a.dtype} != float32")
    if not np.isfinite(a).all() or a.min() < 0.0 or a.max() > 1.0:
        p.append("observation not finite / not in [0, 1]")
    return p


def check_info(info: Dict[str, Any], timing: str, proprio_dim: int) -> List[str]:
    p: List[str] = []
    if not isinstance(info, dict):
        return ["info is not a dict"]
    for k in REQUIRED_INFO:
        if k not in info:
            p.append(f"info missing {k!r}")
    pr = info.get("proprio")
    if pr is not None and np.asarray(pr).shape != (proprio_dim,):
        p.append(f"proprio shape {np.asarray(pr).shape} != ({proprio_dim},)")
    if "audio" in info:
        au = np.asarray(info["audio"])
        if au.ndim != 3 or au.shape[0] != 2 or au.min() < 0.0 or au.max() > 1.0:
            p.append("audio must be float (2, mels, frames) in [0, 1]")
    if timing == "realtime":
        for k in REALTIME_INFO:
            if k not in info:
                p.append(f"realtime game: info missing {k!r}")
        pur = info.get("executed_purity")
        if pur is not None and not (0.0 <= float(pur) <= 1.0):
            p.append("executed_purity outside [0, 1]")
    elif timing == "synchronous":
        for k in ("executed_action", "held_action"):
            if k in info:
                p.append(f"synchronous game must not report {k!r}")
    else:
        p.append(f"timing {timing!r} not in {TIMINGS}")
    return p


def check_action_table(controller: Sequence[Any], valid_mask: Sequence[bool],
                       action_slots: int) -> List[str]:
    p: List[str] = []
    if not controller:
        return ["controller is empty"]
    if controller[0]:
        p.append("slot 0 must be {} (the no-op) in every game")
    if len(controller) > action_slots:
        p.append(f"{len(controller)} slots > universal.action_slots={action_slots}")
    if len(valid_mask) != len(controller):
        p.append("valid_mask length != number of slots")
    elif not bool(valid_mask[0]):
        p.append("valid_mask[0] (the no-op) must be True")
    return p


def check_episode(env: Any, timing: str, image_size: int, proprio_dim: int,
                  n_steps: int = 60, seed: int = 0) -> List[str]:
    """Drive `env` with a fixed slot sequence and check every contract line.

    Includes the executed-action contract for BOTH timing styles and the
    invalid-slot rule. `env` is any gymnasium-style object (a template
    adapter or a wrapped one)."""
    p: List[str] = []
    obs, info = env.reset(seed=seed)
    p += [f"reset: {m}" for m in check_observation(obs, image_size)]
    p += [f"reset: {m}" for m in check_info(info, timing, proprio_dim)]
    n = int(env.action_space.n)
    seq = [(i * 7 + 3) % n for i in range(n_steps)]
    seq[5] = n + 40                                  # out of range -> slot 0, no crash
    saw_life_edge = False
    for t, a in enumerate(seq):
        obs, r, term, trunc, info = env.step(a)
        if r != 0.0:
            p.append(f"step {t}: reward {r} != 0.0 (adapters pay nothing)")
        p += [f"step {t}: {m}" for m in check_observation(obs, image_size)]
        p += [f"step {t}: {m}" for m in check_info(info, timing, proprio_dim)]
        if timing == "realtime" and "executed_action" in info:
            ex = int(info["executed_action"])
            if not 0 <= ex < n:
                p.append(f"step {t}: executed_action {ex} outside 0..{n - 1}")
            if int(info["held_action"]) != (a if 0 <= a < n else 0):
                p.append(f"step {t}: held_action {info['held_action']} != the action just chosen")
        if term or trunc:
            saw_life_edge = True
            obs, info = env.reset()
            p += [f"after life edge: {m}" for m in check_observation(obs, image_size)]
        if p:
            break
    return p
