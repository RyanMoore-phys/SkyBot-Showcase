"""RW-7: from motivation to objective and attention (GOV-010 section 7; GOV-012
entry RW-7). BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier
behaviour exactly.

Self-determination theory and intrinsically motivated goal exploration: what the
agent pursues should follow perceived progress and learnability, and goals come
from its own effects, never from a list.

WIRED (control; neutral 1.0, recorded as deviation from 1.0):
    ST-20  ZPD -> MOT    learnability frontier -> objective mixture. Source: the
                         WP-3 curriculum's best-context progress value (mean over
                         pools), z-scored against its own history. Gain
                         f = 1 + amp * clip(z, -1, 1), bounded [1-amp, 1+amp],
                         slew-limited, written through the single-owner
                         coefficient `objective.frontier_gain` (owner
                         `rw7_motivation`) onto `RewardMixer.frontier_gain`
                         (a second multiplier beside WP-8's `intrinsic_gain`, so
                         the two controllers never share a name). A higher
                         frontier value gives more weight to the intrinsic term (a
                         design choice, revisable; WP-8 does the opposite for FLAT
                         progress, and its sign is untouched).
    SO-30  SAFE -> CUR   safe exploration -> information gain. Source: the WP-11
                         risk feature, z-scored. Factor s = 1 - safe_amp *
                         clip(z, 0, 1), bounded [1-safe_amp, 1] (safe_amp <= 0.9,
                         default 0.5): risk can DAMP the information-gain weight
                         but never silence it (a ceiling on the effect, GOV-010
                         RW-10 guard). Coefficient `info.weight_scale`; applied to
                         `InfoGainTerm.weight`; restored on release.

OBSERVATION (read-only): ST-19 (curriculum progress by context), MO-22 (realised
assignment entropy: how spread the curriculum is), MO-01 (the effective intrinsic
multiplier), MO-02 (the extrinsic weight), MO-03 (competence flat signal vs the
intrinsic gain), MO-29 (action attribution inside the competence tracker as the
agent's sense of control), MO-09 (critic value loss), MO-08 (number of effect keys
the agent has found: its goal vocabulary), ME-16 (policy entropy vs information-gain
agency), ME-17 (policy entropy vs effect-discovery fires), ID-12 (entropy
coefficient vs information-gain agency).

NOT BUILT (reasons in GOV-012 RW-7): MO-28, SO-14, SO-15, SO-18 (persona, data
distribution norms and belonging: no learned analogue, RES-009); ME-24 (no
variational-inference / active-inference block was built).

Escape path (CLAUDE.md 4.1): a vanished source slews each coefficient back to 1.0
at `max_step` per segment; failure or release restores the mixer's
`frontier_gain` and the info-gain weight to base; a floor on the damping keeps the
information-gain term alive under saturated risk.

Switches: relations.enabled; relations.ZPD.MOT.enabled, relations.SAFE.CUR.enabled
and the observation lines (SCAF.ZPD, FLOW.MOT, MOT.INTR, MOT.EXTR, SDT.INTR,
SENSE.SDT, EXPV.MOT, GOAL.MOT, PLAY.CUR, PLAY.CAUS, OPEN.CUR);
relations.motivation.{amp 0.3, max_step 0.05, ema 0.9, min_n 5, safe_amp 0.5}.
"""
from __future__ import annotations

from typing import Optional

from developmental_ai.core.coefficients import registry_for
from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import delta, ppo
from developmental_ai.core.relations_modulation import (
    ZTrack, _entropy_coef, _pair)

OWNER = "rw7_motivation"
WIRED = ("rel.ZPD.MOT", "rel.SAFE.CUR")
_COEFS = ("objective.frontier_gain", "info.weight_scale")
_S = "per segment"


# ---------------------------------------------------------------------------
# producers / probes (learner-side only)
# ---------------------------------------------------------------------------
def _bandits(loop):
    cu = getattr(loop, "_curr", None)
    b = getattr(cu, "bandits", None) if cu else None
    return list(b.values()) if b else []


def frontier_value(loop) -> Optional[float]:
    """Mean over curriculum pools of the best context's progress value."""
    bs = _bandits(loop)
    if not bs:
        return None
    # GOV-015 F-25: bandit values are derived from the fresh-data measurement;
    # while that has been absent they are stale, not evidence, so the factor
    # steps back to neutral instead of following them.
    cu = getattr(loop, "_curr", None)
    if getattr(cu, "empty_steps", 0) >= 2:
        return None
    return sum(max(b.value) for b in bs) / len(bs)


def assignment_entropy(loop) -> Optional[float]:
    bs = _bandits(loop)
    if not bs:
        return None
    return sum(float(b.realised_entropy()) for b in bs) / len(bs)


def risk_feature(loop) -> Optional[float]:
    rk = getattr(loop, "_rk", None)
    if not rk:
        return None
    try:
        return float(rk.vec[0])
    except Exception:
        return None


def _mixer(loop):
    return getattr(loop, "reward_mixer", None)


def intrinsic_multiplier(loop) -> Optional[float]:
    m = _mixer(loop)
    if m is None:
        return None
    return float(m.intrinsic_weight) * float(m.intrinsic_gain) * float(
        getattr(m, "frontier_gain", 1.0))


def extrinsic_weight(loop) -> Optional[float]:
    m = _mixer(loop)
    return None if m is None else float(m.extrinsic_weight)


def _flat(loop) -> Optional[float]:
    Cc = getattr(loop, "_comp", None)
    return float(Cc.flat) if Cc else None


def _gain(loop) -> Optional[float]:
    m = _mixer(loop)
    return None if m is None else float(m.intrinsic_gain)


def _attr(loop) -> Optional[float]:
    Cc = getattr(loop, "_comp", None)
    return float(Cc.attr_mean) if Cc else None


def _entropy(loop) -> Optional[float]:
    return ppo("entropy")(loop)


def _ig_agency(loop) -> Optional[float]:
    t = getattr(loop, "_ig_term", None)
    return float(t.last_mean_agency) if t else None


def _n_keys(loop) -> Optional[float]:
    d = getattr(loop, "_disc", None)
    ks = getattr(getattr(d, "keys", None), "keys", None) if d else None
    return None if ks is None else float(len(ks))


def _fires(loop) -> Optional[float]:
    d = getattr(loop, "_disc", None)
    return float(d.stats["fires"]) if d else None


def register(reg: RelationRegistry) -> None:
    g = "motivation"
    R = reg.register
    R("ST-20", "ZPD", "MOT", "WP-3 curriculum (best-context progress)",
      "objective mixture (RewardMixer.frontier_gain)",
      "frontier gain f = 1 + amp*clip(z of the best-context value)",
      "multiplier", _S, "control", neutral=1.0, group=g, deviation=True)
    R("SO-30", "SAFE", "CUR", "WP-11 risk feature",
      "WP-5 information-gain weight",
      "damping factor s = 1 - safe_amp*clip(z of risk, 0, 1)", "multiplier", _S,
      "control", neutral=1.0, group=g, deviation=True)
    R("ST-19", "SCAF", "ZPD", "WP-3 curriculum", "learnability frontier",
      "best-context progress value", "relative loss drop", _S, "data",
      probe=frontier_value, group=g)
    R("MO-22", "FLOW", "MOT", "WP-3 curriculum", "objective mixture",
      "realised assignment entropy over contexts", "nats", _S, "data",
      probe=assignment_entropy, group=g)
    R("MO-01", "MOT", "INTR", "objective mixture", "intrinsic reward term",
      "effective intrinsic multiplier (weight x gains)", "multiplier", _S,
      "data", probe=intrinsic_multiplier, group=g)
    R("MO-02", "MOT", "EXTR", "objective mixture", "extrinsic reward term",
      "extrinsic weight", "weight", _S, "data", probe=extrinsic_weight, group=g)
    R("MO-03", "SDT", "INTR", "WP-8 competence (flat-progress signal)",
      "intrinsic weight gain",
      "flat signal vs intrinsic gain (correlation over segments)", "r", _S,
      "data", probe=_pair(_flat, _gain), pair=True, group=g)
    R("MO-29", "SENSE", "SDT", "WP-8 competence tracker",
      "autonomy / competence report", "mean action attribution (sense of "
      "control)", "nats", _S, "data", probe=_attr, group=g)
    R("MO-09", "EXPV", "MOT", "critic", "objective (return)",
      "critic value loss", "loss", _S, "data", probe=ppo("value_loss"), group=g)
    R("MO-08", "GOAL", "MOT", "WP-1 effect discovery",
      "objective (goal vocabulary)", "effect keys found so far", "count", _S,
      "data", probe=_n_keys, group=g)
    R("ME-16", "PLAY", "CUR", "policy entropy", "WP-5 information gain",
      "entropy vs information-gain agency (correlation over segments)", "r", _S,
      "data", probe=_pair(_entropy, _ig_agency), pair=True, group=g)
    R("ME-17", "PLAY", "CAUS", "policy entropy", "WP-1 effect discovery",
      "entropy vs effect-discovery fires this segment (correlation over "
      "segments)", "r", _S, "data",
      probe=_pair(_entropy, delta(_fires, "rw7_fires")), pair=True, group=g)
    R("ID-12", "OPEN", "CUR", "policy entropy coefficient",
      "WP-5 information gain",
      "entropy coefficient vs information-gain agency (correlation over "
      "segments)", "r", _S, "data", probe=_pair(_entropy_coef, _ig_agency),
      pair=True, group=g)


# ---------------------------------------------------------------------------
# wiring (once per segment, before the probes)
# ---------------------------------------------------------------------------
def _cfg(reg: RelationRegistry) -> dict:
    return dict(reg.cfg.get("motivation", {}) or {})


def _drop(creg, name: str) -> None:
    try:
        creg.release(name, OWNER)
    except Exception:
        pass


def apply(loop, reg: RelationRegistry) -> None:
    c = _cfg(reg)
    amp = max(0.0, min(0.9, float(c.get("amp", 0.3))))
    safe = max(0.0, min(0.9, float(c.get("safe_amp", 0.5))))
    step = float(c.get("max_step", 0.05))
    ema, mn = float(c.get("ema", 0.9)), int(c.get("min_n", 5))
    st = reg.state.setdefault("motiv", {"zf": ZTrack(ema, mn),
                                        "zr": ZTrack(ema, mn),
                                        "ig_base": None})
    creg = registry_for(loop)
    mixer = _mixer(loop)

    # ST-20: frontier -> objective mixture
    if reg.is_on("rel.ZPD.MOT") and mixer is not None:
        v = frontier_value(loop)
        raw = 1.0 if v is None else 1.0 + amp * max(-1.0, min(
            1.0, st["zf"].update(v)))
        f = reg.send("rel.ZPD.MOT", raw)
        creg.claim("objective.frontier_gain", OWNER, 1.0, 1.0 - amp, 1.0 + amp)
        mixer.frontier_gain = float(creg.set(
            "objective.frontier_gain", OWNER, f, step))
    else:
        _drop(creg, "objective.frontier_gain")
        if mixer is not None:
            mixer.frontier_gain = 1.0

    # SO-30: risk damps (never silences) the information-gain weight
    ig = getattr(loop, "_ig_term", None)
    if reg.is_on("rel.SAFE.CUR") and ig:
        if st["ig_base"] is None:
            st["ig_base"] = float(ig.weight)
        r = risk_feature(loop)
        raw = 1.0 if r is None else 1.0 - safe * max(0.0, min(
            1.0, st["zr"].update(r)))
        s = reg.send("rel.SAFE.CUR", raw)
        creg.claim("info.weight_scale", OWNER, 1.0, 1.0 - safe, 1.0)
        ig.weight = st["ig_base"] * float(creg.set(
            "info.weight_scale", OWNER, s, step))
    else:
        _drop(creg, "info.weight_scale")
        if ig and st["ig_base"] is not None:
            ig.weight = st["ig_base"]


def release(loop, reg: RelationRegistry) -> None:
    st = reg.state.pop("motiv", None)
    creg = registry_for(loop)
    for n in _COEFS:
        _drop(creg, n)
    mixer = _mixer(loop)
    if mixer is not None:
        mixer.frontier_gain = 1.0
    ig = getattr(loop, "_ig_term", None)
    if ig and st and st.get("ig_base") is not None:
        ig.weight = st["ig_base"]
