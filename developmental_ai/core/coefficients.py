"""Coefficient-ownership registry (WP-8, GOV-011; WP-9 and WP-13 must use it).
BUILT, UNTESTED.

GOV-009 WP-8 and WP-9: "several controllers must not fight; single owner of
each coefficient". Every coefficient a controller moves is CLAIMED here under
a name by exactly one owner; only the owner may write it; a second claim by
another owner raises `OwnershipError` (the later controller is refused, loudly,
instead of silently overwriting). Reads are free for everyone.

A write is clamped to [lo, hi] and, when `max_step` is given, slew-limited
(relative to the current value) so one bad estimate cannot move a coefficient
far in one call. `release` returns the coefficient to its base value and frees
the name, so an owner that is switched off or fails leaves no value stranded
(the guard-becomes-latch rule: a released coefficient is back at base).

Coefficients used so far (name -> owner):
  mixer.intrinsic_gain          wp8_competence   multiplier on the intrinsic weight
  explore.entropy_target_scale  wp8_competence   multiplier on WP-9's entropy target
  policy.entropy_coef           wp9_adaptive     entropy coefficient (controller)
  policy.lr_scale               wp9_adaptive     multiplier on the policy learning rate
  wm.lr_scale                   wp9_adaptive     multiplier on the world-model rate
  meta.ppo_clip                wp13_meta        policy.clip_range (hill-climber)
  meta.gae_lambda              wp13_meta        policy.gae_lambda (hill-climber)
  meta.value_coef              wp13_meta        policy.value_coef (hill-climber)
  (WP-13 owns only the three `meta.*` names above; it may not claim the others.)

The registry is plain Python state; nothing is persisted (values restart at
base on relaunch, which is the safe direction).
"""
from __future__ import annotations

from typing import Dict, List, Optional


class OwnershipError(RuntimeError):
    pass


class Coefficient:
    def __init__(self, name: str, owner: str, base: float, lo: float, hi: float):
        if not lo <= base <= hi:
            raise ValueError(f"{name}: base {base} outside [{lo}, {hi}]")
        self.name, self.owner = name, owner
        self.base, self.lo, self.hi = float(base), float(lo), float(hi)
        self.value = float(base)
        self.writes = 0


class CoefficientRegistry:
    def __init__(self):
        self._c: Dict[str, Coefficient] = {}
        self.refused: List[str] = []          # logged refusals, for telemetry

    def claim(self, name: str, owner: str, base: float, lo: float, hi: float
              ) -> Coefficient:
        c = self._c.get(name)
        if c is not None:
            if c.owner != owner:
                self.refused.append(f"{owner} refused {name} (owned by {c.owner})")
                raise OwnershipError(
                    f"coefficient {name!r} is owned by {c.owner!r}; "
                    f"{owner!r} may not claim it")
            return c                          # idempotent for the owner
        c = self._c[name] = Coefficient(name, owner, base, lo, hi)
        return c

    def set(self, name: str, owner: str, value: float,
            max_step: Optional[float] = None) -> float:
        c = self._c.get(name)
        if c is None:
            raise KeyError(f"coefficient {name!r} is not claimed")
        if c.owner != owner:
            self.refused.append(f"{owner} write to {name} refused (owner {c.owner})")
            raise OwnershipError(
                f"{owner!r} may not write {name!r} (owner {c.owner!r})")
        v = float(value)
        if v != v:                            # NaN: keep the current value
            return c.value
        if max_step is not None:
            v = min(c.value + abs(max_step), max(c.value - abs(max_step), v))
        c.value = min(c.hi, max(c.lo, v))
        c.writes += 1
        return c.value

    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        c = self._c.get(name)
        return default if c is None else c.value

    def owner_of(self, name: str) -> Optional[str]:
        c = self._c.get(name)
        return None if c is None else c.owner

    def release(self, name: str, owner: str) -> None:
        c = self._c.get(name)
        if c is None:
            return
        if c.owner != owner:
            raise OwnershipError(f"{owner!r} does not own {name!r}")
        del self._c[name]                     # frees the name; readers get default

    def table(self) -> List[Dict]:
        return [{"name": c.name, "owner": c.owner, "value": c.value,
                 "base": c.base, "lo": c.lo, "hi": c.hi, "writes": c.writes}
                for c in self._c.values()]

    def report(self) -> str:
        return " ".join(f"{c.name}={c.value:.3f}({c.owner})"
                        for c in self._c.values()) or "none claimed"


def registry_for(obj) -> CoefficientRegistry:
    """The one registry of a loop-like object, created on first use."""
    r = getattr(obj, "_coef_registry", None)
    if r is None:
        r = obj._coef_registry = CoefficientRegistry()
    return r
