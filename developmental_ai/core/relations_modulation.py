"""RW-1: learning-signal modulation (GOV-010 section 7; GOV-012 entry RW-1).
BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier behaviour.

One measured quantity, action concentration (1 - calibrated confidence of the
policy's most likely action on recent replay-excluded windows, from WP-4), is
turned into a bounded GAIN g around 1.0. This measures action predictability,
not competence. A missing or failed current calibration supplies no sample:

    z  = running z-score of the uncertainty against its own baseline
    g  = 1 + amp * clip(z, -1, 1)                            (channel NM-17)

and g is passed to three consumers, each through its own channel and its own
coefficient with ONE owner (core/coefficients.py, owner `rw1_modulation`):

    NM-07  NE -> AROUS   g scales the entropy TARGET of WP-9's controller via
                         `explore.uncertainty_scale` (read in adaptive_optim)
    NM-06  NE -> CONS    consolidation strength = 1 + cons_sign*(g-1), through
                         the shared factor mixer below (`consol.strength_scale`
                         -> `Consolidator.strength`); RW-3 adds more factors
    NM-05  NE -> ATT     attention softmax temperature = 1/g
                         (`attn.temperature_scale` -> `GatedWindowAttention
                         .temperature`): higher gain, sharper selection

Observation channels (read-only): MO-17 (exploration coefficient vs attention
entropy), NM-03 (attention entropy vs attention gate), NM-08 (discount horizon
vs return-normaliser scale), MO-26 (normaliser scale vs drift alarms).

ESCAPE PATH (guard-becomes-latch, CLAUDE.md 4.1). When the uncertainty source
gives no sample (planning off, no window yet) every coefficient is slewed back
toward 1.0 by `max_step` per segment, so a vanished source cannot strand a
modulated value; a failure in `apply` releases every coefficient (relations.py).

LEARNED, NOT DECLARED. The baseline for z is the stream's own history; the only
declared numbers are `amp`, `max_step` and `cons_sign` (placeholders for the
A/A noise floor and the A/B), and the SIGN conventions (higher uncertainty:
higher entropy target, sharper attention, weaker consolidation), which are
design choices recorded in GOV-012 and revisable by the T2 test. No game, no
engine value, no evaluator value (ADR-002); the source is the planning phase's
own calibration.

Switches: relations.enabled (master); relations.NE.AROUS.enabled,
relations.NE.CONS.enabled, relations.NE.ATT.enabled, relations.LC.NE.enabled and
relations.<FROM>.<TO>.enabled for the observation lines;
relations.modulation.{amp 0.3, max_step 0.05, cons_sign -1, ema 0.9, min_n 5}.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

from developmental_ai.core.coefficients import registry_for
from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import _state, delta

OWNER = "rw1_modulation"
FINALIZES_FACTORS = True        # the consolidation-strength mixer lives here
WIRED = ("rel.LC.NE", "rel.NE.AROUS", "rel.NE.CONS", "rel.NE.ATT")
_COEFS = ("explore.uncertainty_scale", "attn.temperature_scale",
          "consol.strength_scale")


class ZTrack:
    """Running mean / variance (EMA) and the z-score of the next value."""

    def __init__(self, ema: float = 0.9, min_n: int = 5):
        self.a, self.min_n = float(ema), int(min_n)
        self.m = 0.0
        self.v = 0.0
        self.n = 0

    def update(self, x: float) -> float:
        x = float(x)
        if not math.isfinite(x):
            # GOV-015 F-07: a NaN or infinite input is not evidence. It used
            # to pay the maximum factor once (min(1, nan) is 1) and then leave
            # the mean and variance NaN for good. Neutral z, state unchanged.
            self.refused = getattr(self, "refused", 0) + 1
            return 0.0
        z = 0.0
        if self.n >= self.min_n and self.v > 1e-12:
            z = (x - self.m) / math.sqrt(self.v)
        if self.n == 0:
            self.m = x
        else:
            d = x - self.m
            self.m += (1.0 - self.a) * d
            self.v = self.a * (self.v + (1.0 - self.a) * d * d)
        self.n += 1
        return z


# ---------------------------------------------------------------------------
# producers / probes (learner-side only)
# ---------------------------------------------------------------------------
def policy_uncertainty(loop) -> Optional[float]:
    """Action-concentration heuristic; None without valid current calibration."""
    P = getattr(loop, "_plan", None)
    if not P:
        return None
    last = getattr(P, "last", None) or {}
    if last.get("policy_calibration_ok") != 1.0:
        return None
    c = last.get("calibration_conf")
    if c is None or not math.isfinite(float(c)) or not 0.0 <= float(c) <= 1.0:
        return None
    return 1.0 - float(c)


def attention_entropy(loop) -> Optional[float]:
    am = getattr(loop, "_am", None)
    net = getattr(am, "net", None) if am else None
    w = getattr(net, "last_weights", None)
    if w is None:
        return None
    import torch
    p = w.clamp_min(1e-9)
    return float(-(p * p.log()).sum(-1).mean())


def attention_gate(loop) -> Optional[float]:
    am = getattr(loop, "_am", None)
    g = (getattr(am, "last", None) or {}).get("gate") if am else None
    return None if g is None else float(g)


def _entropy_coef(loop) -> Optional[float]:
    return getattr(getattr(loop, "policy", None), "entropy_coef", None)


def _horizon(loop) -> Optional[float]:
    g = getattr(getattr(loop, "policy", None), "gamma", None)
    if g is None or not (0.0 <= float(g) < 1.0):
        return None
    return 1.0 / (1.0 - float(g))


def _return_scale(loop) -> Optional[float]:
    ao = getattr(loop, "_ao", None)
    r = getattr(ao, "ret", None) if ao else None
    return None if r is None else float(r.scale)


def _drift_delta(loop) -> Optional[float]:
    return delta(lambda l: getattr(getattr(l, "infra", None),
                                   "drift_alarms", None), "drift")(loop)


def _pair(x: Callable[[Any], Any], y: Callable[[Any], Any]):
    def probe(loop):
        a, b = x(loop), y(loop)
        return None if a is None or b is None else (a, b)
    return probe


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
_S = "per segment"


def register(reg: RelationRegistry) -> None:
    g = "modulation"
    R = reg.register
    # wired (data / control); neutral 1.0; deviation recorded
    R("NM-17", "LC", "NE", "WP-4 planning phase (calibrated action predictability)",
      "gain computation (this module)", "gain g = 1 + amp*clip(z of uncertainty)",
      "multiplier", _S, "data", neutral=1.0, group=g, deviation=True)
    R("NM-07", "NE", "AROUS", "gain g", "WP-9 entropy controller target",
      "entropy-target multiplier (explore.uncertainty_scale)", "multiplier", _S,
      "control", neutral=1.0, group=g, deviation=True)
    R("NM-06", "NE", "CONS", "gain g", "WP-2 consolidation strength",
      "consolidation-strength factor (via the factor mixer)", "multiplier", _S,
      "control", neutral=1.0, group=g, deviation=True)
    R("NM-05", "NE", "ATT", "gain g", "WP-7 attention softmax",
      "attention temperature multiplier (1/g)", "multiplier", _S, "control",
      neutral=1.0, group=g, deviation=True)
    # observation (no write-back)
    R("MO-17", "AROUS", "ATT", "policy entropy coefficient",
      "attention entropy", "entropy coefficient vs attention entropy "
      "(correlation over segments)", "r", _S, "data",
      probe=_pair(_entropy_coef, attention_entropy), pair=True, group=g)
    R("NM-03", "ACH", "ATT", "attention entropy", "attention gate",
      "attention entropy vs attention gate (correlation over segments)", "r",
      _S, "data", probe=_pair(attention_entropy, attention_gate), pair=True,
      group=g)
    R("NM-08", "SER", "EMOREG", "policy discount (effective horizon)",
      "WP-9 return normaliser", "horizon 1/(1-gamma) vs normaliser scale "
      "(correlation over segments)", "r", _S, "data",
      probe=_pair(_horizon, _return_scale), pair=True, group=g)
    R("MO-26", "EMOREG", "ANX", "WP-9 return normaliser",
      "behaviour-drift alarms", "normaliser scale vs drift alarms this segment "
      "(correlation over segments)", "r", _S, "data",
      probe=_pair(_return_scale, _drift_delta), pair=True, group=g)


# ---------------------------------------------------------------------------
# wiring (once per segment, before the probes)
# ---------------------------------------------------------------------------
def _cfg(reg: RelationRegistry) -> dict:
    return dict(reg.cfg.get("modulation", {}) or {})


def _set(creg, name: str, lo: float, hi: float, value: float, max_step: float):
    creg.claim(name, OWNER, 1.0, lo, hi)
    return creg.set(name, OWNER, value, max_step)


def _drop(creg, name: str) -> None:
    try:
        creg.release(name, OWNER)
    except Exception:
        pass


def apply(loop, reg: RelationRegistry) -> None:
    c = _cfg(reg)
    amp = max(0.0, float(c.get("amp", 0.3)))
    step = float(c.get("max_step", 0.05))
    sign = -1.0 if float(c.get("cons_sign", -1.0)) < 0 else 1.0
    st = reg.state.setdefault("mod", {"z": ZTrack(float(c.get("ema", 0.9)),
                                                  int(c.get("min_n", 5)))})
    creg = registry_for(loop)
    u = policy_uncertainty(loop)
    if u is None:                        # source gone: slew back to neutral
        raw = 1.0
        z = 0.0
    else:
        z = st["z"].update(u)
        raw = 1.0 + amp * max(-1.0, min(1.0, z))
    g = reg.send("rel.LC.NE", raw)       # NM-17 (neutral 1.0 when off)
    g = max(1.0 - amp, min(1.0 + amp, g)) if amp > 0 else 1.0

    # NM-07: entropy target (WP-9 reads the coefficient next segment)
    if reg.is_on("rel.NE.AROUS"):
        v = reg.send("rel.NE.AROUS", g)
        _set(creg, "explore.uncertainty_scale", 1.0 - amp, 1.0 + amp, v, step)
    else:
        _drop(creg, "explore.uncertainty_scale")

    # NM-05: attention temperature = 1/g
    am = getattr(loop, "_am", None)
    net = getattr(am, "net", None) if am else None
    if reg.is_on("rel.NE.ATT"):
        v = reg.send("rel.NE.ATT", 1.0 / max(g, 1e-6))
        lo, hi = 1.0 / (1.0 + amp), 1.0 / max(1.0 - amp, 1e-6)
        t = _set(creg, "attn.temperature_scale", lo, hi, v, step)
        if net is not None:
            net.temperature = float(t)
    else:
        _drop(creg, "attn.temperature_scale")
        if net is not None:
            net.temperature = 1.0

    # NM-06: a factor into the shared consolidation-strength mixer (finalize)
    if reg.is_on("rel.NE.CONS"):
        reg.add_factor("consol", "NM-06",
                       reg.send("rel.NE.CONS", 1.0 + sign * (g - 1.0)))


def finalize(loop, reg: RelationRegistry) -> None:
    """Combine every factor other clusters added for consolidation strength
    (product, one owner, bounded, slew-limited) and apply it."""
    c = _cfg(reg)
    step = float(c.get("max_step", 0.05))
    creg = registry_for(loop)
    cons = getattr(loop, "_cons", None)
    fac = reg.factors.get("consol") or {}
    if not fac:
        _drop(creg, "consol.strength_scale")
        if cons:
            cons.strength = 1.0
        return
    prod = 1.0
    for v in fac.values():
        prod *= float(v)
    lo = float(c.get("strength_lo", 0.25))
    hi = float(c.get("strength_hi", 4.0))
    s = _set(creg, "consol.strength_scale", lo, hi, prod, step)
    if cons:
        cons.strength = float(s)


def release(loop, reg: RelationRegistry) -> None:
    creg = registry_for(loop)
    for n in _COEFS:
        _drop(creg, n)
    cons = getattr(loop, "_cons", None)
    if cons:
        cons.strength = 1.0
    am = getattr(loop, "_am", None)
    net = getattr(am, "net", None) if am else None
    if net is not None:
        net.temperature = 1.0
