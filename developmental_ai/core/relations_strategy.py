"""RW-6: metacognition and strategy selection (GOV-010 section 7; GOV-012 entry
RW-6). BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier
behaviour.

Nelson and Narens: a monitoring level (judgements of learning, calibration) feeds
a control level that picks study strategies. Here the CONTROL level is a small
bandit over a few existing training RECIPES of the WP-2 consolidation phase:

    recipe = (spaced-replay fraction, prioritised sampling on/off)
    arm 0 = the configured values (no override), arm 1 = spaced revisits,
    arm 2 = prioritised sampling, arm 3 = both

and the MONITORING level is the progress of the agent's own fresh-data loss on
each stream's newest never-trained window (`measure_streams`, learner-side,
ADR-002): every `epoch_len` segments the previous arm is credited with

    reward = (relative drop of the mean fresh loss over the epoch) - (EMA of that
             drop over all epochs)

(the baseline removes the improvement every arm gets from training, as in WP-3,
D-22), then the bandit (`curriculum.ContextBandit`, reused) draws the next arm.
The recipe is applied through two single-owner coefficients
(`strategy.spaced_fraction`, `strategy.prioritized`, owner `rw6_strategy`) to
`Consolidator.old_frac` and `Consolidator.prioritized`.

WIRED (control):
    CP-37  SRL -> STRAT_ENTRY   the selector: chooses the arm (value = arm index,
                                neutral 0 = base). ON activates the selector.
    ST-39  STRAT_ENTRY -> SP    the arm's spaced fraction is applied (value = change
                                from the configured fraction, neutral 0)
    ST-40  STRAT_ENTRY -> INTL  the arm's sampling schedule is applied (value =
                                change in prioritised sampling, neutral 0)
    CP-35  CALIB -> SRL         calibration sets the bandit's exploration rate:
                                epsilon = eps0 * (1 + 0.5 * clip(z of the policy's
                                expected calibration error, -1, 1)), bounded
                                [0.02, 0.5]; worse calibration than its own
                                baseline: more strategy exploration (a design
                                choice, revisable)

OBSERVATION (read-only): CP-34 (expected calibration error of the policy), CP-36
(accepted trials of the WP-13 hyper-parameter adapter), MO-11 (competence progress
vs the selector's reward), MO-14 (action attribution inside the competence
tracker).

NOT BUILT (reasons in GOV-012 RW-6): ST-41 (the augmentation weight lives in the
world model's auxiliary module, which the async trainer copies; a runtime write
may not reach it), ST-38 (no evaluation-scheduling knob), CP-33 (no flow from the
adapter to the confidence estimate), MO-10, MO-12, MO-13, MO-27, MO-30, MO-07
(persistence and capacity blocks absent or not established), ID-23 (no
learner-side data-coverage measure), ST-12, ST-13, SO-28 (RES-009 D-10, D-12).

Escape path (CLAUDE.md 4.1): the base arm is always a member; the bandit's floor
keeps every arm drawn; on failure or when the consolidation phase is absent the
coefficients are released and `old_frac` / `prioritized` return to the configured
values.

Switches: relations.enabled; relations.SRL.STRAT_ENTRY.enabled (the selector),
relations.STRAT_ENTRY.SP.enabled, relations.STRAT_ENTRY.INTL.enabled,
relations.CALIB.SRL.enabled and the observation lines (JOL.CALIB, META.SRL,
SELFE.SRL, ATTRIB.SELFE); relations.strategy.{epoch_len 4, epsilon 0.1,
spaced 0.25, seq_len 32, min_pulls 1, ucb_c 0.1, temperature 0.1, seed 0}.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from developmental_ai.core.coefficients import registry_for
from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import delta
from developmental_ai.core.relations_modulation import ZTrack, _pair

OWNER = "rw6_strategy"
WIRED = ("rel.SRL.STRAT_ENTRY", "rel.STRAT_ENTRY.SP", "rel.STRAT_ENTRY.INTL",
         "rel.CALIB.SRL")
_COEFS = ("strategy.spaced_fraction", "strategy.prioritized")
_S = "per segment"


def _P(loop):
    P = getattr(loop, "_plan", None)
    return P if P else None


def calibration_error(loop) -> Optional[float]:
    P = _P(loop)
    return None if P is None else P.mon.ece("policy")


def _accepts(loop) -> Optional[float]:
    Ma = getattr(loop, "_ma", None)
    return float(Ma.accepts) if Ma else None


def _progress(loop) -> Optional[float]:
    Cc = getattr(loop, "_comp", None)
    if not Cc or not Cc.ctx:
        return None
    return sum(s.lp for s in Cc.ctx.values()) / len(Cc.ctx)


def _attr(loop) -> Optional[float]:
    Cc = getattr(loop, "_comp", None)
    return float(Cc.attr_mean) if Cc else None


def _sel_reward(loop) -> Optional[float]:
    r = getattr(loop, "_rel", None)
    st = getattr(r, "state", {}).get("strat") if r else None
    return None if not st else st.get("last_reward")


def register(reg: RelationRegistry) -> None:
    g = "strategy"
    R = reg.register
    eps0 = float((reg.cfg.get("strategy", {}) or {}).get("epsilon", 0.1))
    R("CP-37", "SRL", "STRAT_ENTRY", "strategy bandit (progress of each recipe)",
      "recipe choice", "arm chosen (0 = configured recipe)", "arm index", _S,
      "control", neutral=0.0, group=g)
    R("ST-39", "STRAT_ENTRY", "SP", "chosen recipe", "WP-2 spaced-replay fraction",
      "change of the spaced fraction from its configured value", "fraction", _S,
      "control", neutral=0.0, group=g)
    R("ST-40", "STRAT_ENTRY", "INTL", "chosen recipe",
      "WP-2 replay sampling schedule", "change of prioritised sampling (-1, 0, +1)",
      "step", _S, "control", neutral=0.0, group=g)
    R("CP-35", "CALIB", "SRL", "policy calibration error (WP-4)",
      "strategy bandit exploration rate", "epsilon", "probability", _S, "control",
      neutral=eps0, group=g, deviation=True)
    R("CP-34", "JOL", "CALIB", "policy confidence", "calibration monitor",
      "expected calibration error of the policy", "ece", _S, "data",
      probe=calibration_error, group=g)
    R("CP-36", "META", "SRL", "WP-13 hyper-parameter adapter",
      "strategy bandit", "accepted adapter trials this segment", "count", _S,
      "data", probe=delta(_accepts, "rw6_acc"), group=g)
    R("MO-11", "SELFE", "SRL", "WP-8 competence progress", "strategy bandit",
      "competence progress vs the selector's reward (correlation over segments)",
      "r", _S, "data", probe=_pair(_progress, _sel_reward), pair=True, group=g)
    R("MO-14", "ATTRIB", "SELFE", "action attribution", "WP-8 competence tracker",
      "mean action attribution inside the tracker", "nats", _S, "data",
      probe=_attr, group=g)


RECIPES = ({}, {"spaced": True}, {"prioritized": True},
           {"spaced": True, "prioritized": True})


def _fresh(loop, seq_len: int) -> Optional[float]:
    from developmental_ai.core.curriculum import measure_streams
    with loop._wm_param_lock:
        meas = measure_streams(loop.world_model, loop.replay_buffer, seq_len,
                               loop.device)
    if not meas:
        return None
    return sum(float(v[0]) for v in meas.values()) / len(meas)


def _apply_recipe(loop, reg, st, arm: int) -> None:
    cons = getattr(loop, "_cons", None)
    cr = registry_for(loop)
    base_f, base_p = st["base"]
    c = dict(reg.cfg.get("strategy", {}) or {})
    rec = RECIPES[arm]
    f = base_f
    if rec.get("spaced"):
        f = max(base_f, float(c.get("spaced", 0.25)))
    p = 1.0 if (base_p or rec.get("prioritized")) else 0.0
    cr.claim("strategy.spaced_fraction", OWNER, base_f, 0.0, 1.0)
    cr.claim("strategy.prioritized", OWNER, float(base_p), 0.0, 1.0)
    if reg.is_on("rel.STRAT_ENTRY.SP"):
        f_out = base_f + reg.send("rel.STRAT_ENTRY.SP", f - base_f)
    else:
        f_out = base_f
    if reg.is_on("rel.STRAT_ENTRY.INTL"):
        p_out = float(base_p) + reg.send("rel.STRAT_ENTRY.INTL",
                                         p - float(base_p))
    else:
        p_out = float(base_p)
    f_set = cr.set("strategy.spaced_fraction", OWNER, f_out)
    p_set = cr.set("strategy.prioritized", OWNER, p_out)
    cons.old_frac = float(f_set)
    cons.prioritized = bool(p_set >= 0.5)


def apply(loop, reg: RelationRegistry) -> None:
    cons = getattr(loop, "_cons", None)
    if not cons or not reg.is_on("rel.SRL.STRAT_ENTRY"):
        return
    c = dict(reg.cfg.get("strategy", {}) or {})
    st = reg.state.get("strat")
    if st is None:
        from developmental_ai.core.curriculum import ContextBandit
        bc = {"min_pulls": int(c.get("min_pulls", 1)),
              "ucb_c": float(c.get("ucb_c", 0.1)),
              "temperature": float(c.get("temperature", 0.1)),
              "epsilon": float(c.get("epsilon", 0.1))}
        st = reg.state["strat"] = {
            "bandit": ContextBandit(len(RECIPES), bc,
                                    random.Random(int(c.get("seed", 0)))),
            "arm": 0, "seg": 0, "Lprev": None, "rbase": None,
            "last_reward": None, "eps0": float(c.get("epsilon", 0.1)),
            "zece": ZTrack(0.9, 5),
            "base": (float(cons.old_frac), bool(cons.prioritized))}
    st["seg"] += 1
    if st["seg"] % max(1, int(c.get("epoch_len", 4))) == 0:
        L = _fresh(loop, int(c.get("seq_len", 32)))
        if L is not None and not math.isfinite(float(L)):
            L = None            # GOV-015 F-16: min(1, nan) is 1; not evidence
        if L is None:
            # GOV-015 F-18: the recipe was held until a measurement returned.
            # After `blind_after` epochs without one, return to the configured
            # recipe (arm 0) and forget the stale baseline.
            st["blind"] = st.get("blind", 0) + 1
            if st["blind"] >= max(1, int(c.get("blind_after", 2))):
                st["arm"], st["Lprev"], st["rbase"] = 0, None, None
        else:
            st["blind"] = 0
            if st["Lprev"] is not None:
                r = max(-1.0, min(1.0, (st["Lprev"] - L)
                                  / (abs(st["Lprev"]) + 1e-8)))
                rb = r if st["rbase"] is None else st["rbase"]
                st["last_reward"] = r - rb
                st["bandit"].update(st["arm"], st["last_reward"])
                st["rbase"] = 0.8 * rb + 0.2 * r
            st["Lprev"] = L
            eps = st["eps0"]
            if reg.is_on("rel.CALIB.SRL"):
                e = calibration_error(loop)
                if e is not None:
                    z = st["zece"].update(float(e))
                    eps = st["eps0"] * (1.0 + 0.5 * max(-1.0, min(1.0, z)))
                eps = max(0.02, min(0.5, eps))
                eps = reg.send("rel.CALIB.SRL", eps)
            st["bandit"].eps = float(eps)
            st["arm"], _ = st["bandit"].choose()
    arm = reg.send("rel.SRL.STRAT_ENTRY", float(st["arm"]))
    _apply_recipe(loop, reg, st, int(arm))


def release(loop, reg: RelationRegistry) -> None:
    st = reg.state.pop("strat", None)
    cr = registry_for(loop)
    for n in _COEFS:
        try:
            cr.release(n, OWNER)
        except Exception:
            pass
    cons = getattr(loop, "_cons", None)
    if cons and st:
        cons.old_frac, cons.prioritized = st["base"][0], st["base"][1]
