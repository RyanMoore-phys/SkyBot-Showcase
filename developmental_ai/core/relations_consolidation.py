"""RW-3: the consolidation chain (GOV-010 section 7; GOV-012 entry RW-3).
BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier behaviour.

Blocks used (all built, GOV-011): WP-2 `Consolidator` (the offline phase: snapshot
distillation, online EWC, advantage-weighted offline RL, spaced replay), the
replay buffer, WP-11 risk, WP-9/WP-4 telemetry. NM-06 (gain to consolidation) was
built in RW-1 (relations_modulation) and shares this chain's strength mixer.

OBSERVATION channels (read-only, one reading per segment of the Consolidator's
own last-phase statistics): PH-01, PH-03, PH-04, PH-05, SY-03, SY-07, CP-14,
CP-15, NS-14, ST-03.

WIRED channels (control; each contributes ONE FACTOR to the shared
consolidation-strength product, which relations_modulation.finalize applies to
`Consolidator.strength` through the single-owner coefficient
`consol.strength_scale`; neutral 1.0):

    MO-16  EMO   -> CONS   reward salience: z of the reward income (ledger total)
    SY-16  AMY   -> CONS   risk: z of WP-11's risk feature for stream 0
    ST-02  RP    -> CONS   learner-side evaluation: z of the FRESH-DATA loss on
                           each stream's newest never-trained window (the same
                           measurement the curriculum, competence and meta
                           blocks take; one extra forward per stream every
                           `fresh_every` segments). NEVER an evaluator
                           checkpoint (ADR-002).
    ST-34  BREAK -> CONS   the PPO KL stop firing (the policy hit its movement
                           budget) as z of its segment value

    factor = 1 + amp * clip(z, -1, 1)           (direction: more of the signal,
                                                 stronger consolidation)

z is against the signal's own running baseline (relations_modulation.ZTrack), so
no scale is declared. The factors multiply; the product is bounded
[strength_lo, strength_hi] and slew-limited by the coefficient registry. A
missing source sends the neutral 1.0, so a vanished source cannot strand a
value (the product walks back at `max_step`).

NOT BUILT (reasons in GOV-012 RW-3): CP-13, CP-18, PH-02, PH-07, PH-09, SY-32 (no
mechanism or the endpoint block is absent), CP-19, CP-20, CP-21, ST-04 (they need
a forgetting probe; none exists), NM-06 (done in RW-1), PH-06 (RES-009 D-08).

Switches: relations.enabled; relations.<FROM>.<TO>.enabled per line;
relations.consolidation.{amp 0.3, fresh_every 4, seq_len 32}.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import (
    _replay_len, delta, ledger_total)
from developmental_ai.core.relations_modulation import ZTrack, _pair

WIRED = ("rel.EMO.CONS", "rel.AMY.CONS", "rel.RP.CONS", "rel.BREAK.CONS")
_S = "per segment"


def _cl(key: str) -> Callable[[Any], Optional[float]]:
    """Read one key of the Consolidator's last phase (None if absent)."""
    def probe(loop):
        C = getattr(loop, "_cons", None)
        if not C:
            return None
        v = (getattr(C, "last", None) or {}).get(key)
        return None if v is None else float(v)
    return probe


def _spaced_fraction(loop) -> Optional[float]:
    C = getattr(loop, "_cons", None)
    return None if not C else float(getattr(C, "old_frac", 0.0))


_rows = delta(_replay_len, "rw3_rows")
_ckpt = delta(lambda l: getattr(l, "_last_checkpoint_step", None), "rw3_ckpt")


def risk_feature(loop) -> Optional[float]:
    rk = getattr(loop, "_rk", None)
    v = getattr(rk, "vec", None) if rk else None
    return None if v is None or len(v) == 0 else float(v[0])


def fresh_loss(loop, reg: RelationRegistry) -> Optional[float]:
    """Mean fresh-data reconstruction loss over streams on each stream's newest
    never-trained window; measured every `fresh_every` segments, cached between
    (learner-side only; one forward per stream)."""
    c = dict(reg.cfg.get("consolidation", {}) or {})
    st = reg.state.setdefault("cons", {"seg": 0, "fresh": None})
    st["seg"] += 1
    if st["fresh"] is not None and st["seg"] % max(1, int(c.get("fresh_every", 4))):
        return st["fresh"]
    from developmental_ai.core.curriculum import measure_streams
    with loop._wm_param_lock:
        meas = measure_streams(loop.world_model, loop.replay_buffer,
                               int(c.get("seq_len", 32)), loop.device)
    if meas:
        st["fresh"] = sum(float(v[0]) for v in meas.values()) / len(meas)
    return st["fresh"]


def register(reg: RelationRegistry) -> None:
    g = "consolidation"
    R = reg.register
    # observation
    R("PH-01", "SLEEP", "SWS", "WP-2 consolidation phase", "snapshot distillation",
      "distillation KL to the kept snapshots", "nats", _S, "data",
      probe=_cl("kl"), group=g)
    R("PH-03", "SWS", "SWR", "snapshot distillation", "replayed rows",
      "snapshots kept (teachers)", "count", _S, "data",
      probe=_cl("teachers"), group=g)
    R("PH-04", "SWR", "REPL", "replay with policy columns", "advantage-weighted "
      "offline RL", "fraction of replayed rows with valid policy columns",
      "fraction", _S, "data", probe=_cl("awr_valid"), group=g)
    R("PH-05", "REPL", "SYSCON", "advantage-weighted offline RL",
      "consolidation loss", "AWR loss", "loss", _S, "data",
      probe=_cl("awr"), group=g)
    R("SY-03", "HIP", "SYSCON", "replay buffer", "consolidation phase",
      "rows added vs consolidation steps taken (correlation over segments)",
      "r", _S, "data", probe=_pair(_rows, _cl("steps")), pair=True, group=g)
    R("SY-07", "HIP", "SWR", "replay buffer", "experience replay",
      "rows added to the replay buffer", "rows", _S, "data", probe=_rows,
      group=g)
    R("CP-14", "CONS", "SYSCON", "online EWC", "consolidation loss",
      "EWC penalty", "loss", _S, "data", probe=_cl("ewc"), group=g)
    R("CP-15", "SYSCON", "LTM", "consolidation phase", "policy weights",
      "consolidation optimiser steps applied", "steps", _S, "data",
      probe=_cl("steps"), group=g)
    R("NS-14", "IEG", "CONS", "checkpoint writer", "consolidation phase",
      "steps since checkpoint vs consolidation steps (correlation over "
      "segments); a checkpoint ending a consolidation is NOT wired (GOV-012)",
      "r", _S, "data", probe=_pair(_ckpt, _cl("steps")), pair=True, group=g)
    R("ST-03", "SP", "CONS", "spaced-replay fraction (WP-2)",
      "consolidation sampling", "share of each batch from the oldest rows",
      "fraction", _S, "data", probe=_spaced_fraction, group=g)
    # wired factors (neutral 1.0, deviation recorded)
    R("MO-16", "EMO", "CONS", "reward income (ledger total)",
      "consolidation-strength mixer", "factor 1 + amp*clip(z of reward income)",
      "multiplier", _S, "control", neutral=1.0, group=g, deviation=True)
    R("SY-16", "AMY", "CONS", "WP-11 risk feature (stream 0)",
      "consolidation-strength mixer", "factor 1 + amp*clip(z of risk)",
      "multiplier", _S, "control", neutral=1.0, group=g, deviation=True)
    R("ST-02", "RP", "CONS", "fresh-data loss on never-trained windows",
      "consolidation-strength mixer", "factor 1 + amp*clip(z of fresh loss)",
      "multiplier", _S, "control", neutral=1.0, group=g, deviation=True)
    R("ST-34", "BREAK", "CONS", "PPO KL stop",
      "consolidation-strength mixer", "factor 1 + amp*clip(z of KL-stop)",
      "multiplier", _S, "control", neutral=1.0, group=g, deviation=True)


def apply(loop, reg: RelationRegistry) -> None:
    c = dict(reg.cfg.get("consolidation", {}) or {})
    amp = max(0.0, float(c.get("amp", 0.3)))
    tr = reg.state.setdefault("cons_z", {})
    srcs = {
        "rel.EMO.CONS": ledger_total,
        "rel.AMY.CONS": risk_feature,
        "rel.RP.CONS": lambda l: fresh_loss(l, reg),
        "rel.BREAK.CONS": lambda l: (
            None if not isinstance(getattr(l, "_last_ppo", None), dict)
            or l._last_ppo.get("kl_stopped") is None
            else float(bool(l._last_ppo["kl_stopped"]))),
    }
    for name in WIRED:
        if not reg.is_on(name):
            continue
        v = srcs[name](loop)
        f = 1.0
        if v is not None:
            z = tr.setdefault(name, ZTrack()).update(float(v))
            f = 1.0 + amp * max(-1.0, min(1.0, z))
        reg.add_factor("consol", name, reg.send(name, f))


def release(loop, reg: RelationRegistry) -> None:
    """Nothing is claimed here (factors only); strength is released by the
    mixer's owner (relations_modulation)."""
    return None
