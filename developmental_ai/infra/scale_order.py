"""WP-10 (GOV-009, GOV-011): the autoscaler's response to COMPUTE PRESSURE,
reconciled with ADR-005 and its Amendments 1 and 2. BUILT, UNTESTED.
Default OFF (`infra.autoscaler.adr005.enabled: false`): off, the autoscaler's
old `_shrink_or` fallback (shrink, else fewer streams, else hold) is unchanged.

ADR-005 order, when a budget is exceeded (acting latency over the game's
budget, VRAM over its cap):
  1. MODEL SIZE: shrink one rung, unless the model is AT SATURATION.
  2. STREAMS: if saturated, remove a stream, while the stream count is above
     "low".
  3. STEPPING RATE: if the model is saturated AND streams are low, slow the
     stepping; every slowing is logged with the game time per decision it
     implies in a real-time game.
Saturation (Amendment 1): the model is at the smallest rung, OR the next
smaller rung failed the quality tolerance. "Low" streams: at or below half of
the maximum count (four of eight).

The tolerance (Amendment 2) is measured by `tools/grow_brain.py` when it
relearns the smaller rung: fresh-data loss more than 10 % worse on the same
held-out windows (and, with `--full-tolerance`, the mechanism measure of
action attribution and a retention probe on the oldest windows). A refused
shrink is written to `scale.json` as `refused_shrinks`; this module reads it.
A refusal EXPIRES after `refusal_expiry_hours` and is ignored once the model
has changed size, so a refusal can never latch the order (CLAUDE.md 4.1).

Reduction goes one step per decision (the autoscaler's cooldown applies). When
pressure is gone, stepping is restored FIRST (reverse of the order), but only
if stepping is ACTUATED (`stepping_actuated`, default false: no stepping-rate
actuator exists in this build, D-56), so in this build the stepping decision
is advice that repeats after the cooldown, exactly like any unapplied advice.

Both arms of a comparison must run the same configuration (Amendment 2): the
state carries a configuration fingerprint (size, streams, step scale, actuate)
written to autoscale.json so a comparison can verify it. Frozen evaluation
runs keep their protocol configuration: set `actuate: false` there.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

SLOW, RESTORE = "slow_stepping", "restore_stepping"


def load_refusals(path: str, expiry_h: float, now: Optional[float] = None
                  ) -> Dict[str, Dict[str, Any]]:
    """Unexpired refused shrinks from scale.json: {to_rung: record}."""
    try:
        with open(path) as f:
            recs = (json.load(f) or {}).get("refused_shrinks") or {}
    except (OSError, ValueError):
        return {}
    now = time.time() if now is None else now
    out = {}
    for k, r in recs.items():
        try:
            if now - float(r.get("t", 0.0)) <= expiry_h * 3600.0:
                out[str(r.get("to", k))] = r
        except (TypeError, ValueError):
            continue
    return out


class PressureOrder:
    def __init__(self, cfg: Dict[str, Any], scaler):
        c = dict(cfg or {})
        self.a = scaler
        low = c.get("low_streams")
        self.low_streams = (int(low) if low is not None
                            else max(1, scaler.max_streams // 2))
        self.step_factor = float(c.get("step_factor", 1.25))
        self.max_step_scale = float(c.get("max_step_scale", 2.0))
        self.expiry_h = float(c.get("refusal_expiry_hours", 12.0))
        self.scale_file = str(c.get("scale_file", "brain_universal/scale.json"))
        self.stepping_actuated = bool(c.get("stepping_actuated", False))
        if not self.step_factor > 1.0:
            raise ValueError("adr005.step_factor must be > 1")
        self.step_scale = 1.0          # multiplier on game time per decision

    # ---- saturation (Amendment 1) ------------------------------------
    def saturated(self) -> tuple:
        a = self.a
        if a.size <= a.min_size:
            return True, "already the smallest rung"
        from developmental_ai.infra.autoscaler import SIZE_NAMES
        ref = load_refusals(self.scale_file, self.expiry_h)
        smaller = SIZE_NAMES[a.size - 1]
        r = ref.get(smaller)
        if r is not None and str(r.get("from")) == SIZE_NAMES[a.size]:
            return True, (f"the next smaller rung ({smaller}) failed the quality "
                          f"tolerance")
        return False, ""

    # ---- the order (ADR-005) -----------------------------------------
    def respond(self, sig, reason: str, releases: str):
        """Return the Decision for a budget breach."""
        from developmental_ai.infra.autoscaler import (
            FEWER_STREAMS, HOLD, SHRINK)
        a = self.a
        sat, why = self.saturated()
        if not sat:
            return a._dec(SHRINK, reason + " [ADR-005 step 1: model first]", sig,
                          releases, size=a.size - 1)
        if a.streams > self.low_streams and a.streams > a.min_streams:
            return a._dec(FEWER_STREAMS, f"{reason} [ADR-005 step 2: model saturated "
                          f"({why}); streams {a.streams} > low {self.low_streams}]",
                          sig, releases, streams=a.streams - 1)
        new = min(self.max_step_scale, self.step_scale * self.step_factor)
        if new <= self.step_scale + 1e-9:
            return a._dec(HOLD, f"{reason} — every dial at its limit (model "
                          f"saturated, streams low, stepping already x"
                          f"{self.step_scale:.2f} of max x{self.max_step_scale:.2f})",
                          sig, "pressure clears, or a bound is raised")
        d = a._dec(SLOW, f"{reason} [ADR-005 step 3: model saturated ({why}), "
                   f"streams low ({a.streams} <= {self.low_streams})]", sig, releases)
        self._log_game_time(d, sig, new)
        d.evidence["step_scale"] = round(new, 3)
        return d

    def restore(self, sig):
        """Pressure gone: undo stepping slowdown first (reverse order)."""
        a = self.a
        new = max(1.0, self.step_scale / self.step_factor)
        d = a._dec(RESTORE, "pressure cleared: restoring the stepping rate first "
                   "(reverse of the ADR-005 order)", sig)
        self._log_game_time(d, sig, new)
        d.evidence["step_scale"] = round(new, 3)
        return d

    def _log_game_time(self, d, sig, new_scale: float) -> None:
        """The amount of game time each decision covers, before and after, in a
        real-time game (ADR-005 Amendment 1); n/a for a synchronous game."""
        g = getattr(sig, "game_s_per_decision", None)
        if g is None:
            d.evidence["game_s_per_decision"] = "n/a (synchronous game)"
            return
        after = g * new_scale / max(1e-9, self.step_scale)
        d.evidence["game_s_per_decision"] = f"{g:.3f}->{after:.3f}"

    def apply(self, d) -> None:
        """Move the stepping state of an ACTUATED stepping decision."""
        if not self.stepping_actuated:
            return
        if d.action in (SLOW, RESTORE):
            self.step_scale = float(d.evidence.get("step_scale", self.step_scale))

    def state(self) -> Dict[str, Any]:
        a = self.a
        from developmental_ai.infra.autoscaler import SIZE_NAMES
        return {"step_scale": self.step_scale, "low_streams": self.low_streams,
                "fingerprint": {"size": SIZE_NAMES[a.size], "streams": a.streams,
                                "step_scale": self.step_scale,
                                "actuate": a.actuate}}
