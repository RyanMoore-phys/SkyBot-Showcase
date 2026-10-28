"""WP-9 (GOV-009, GOV-011): adaptive optimisation and plasticity.
BUILT, UNTESTED. Default OFF (`adaptive_optim.enabled: false`, and every
part has its own sub-switch, also off).

Parts (each independent):
  schedules      learning-rate multipliers for the policy and the world model
                 (`lr_scale`): constant | linear | cosine | warmup_cosine, on
                 the lifetime step count. Applied to optimiser param groups
                 as base_lr x scale; base_lr is captured once per group.
  entropy        `EntropyController`: moves the policy's entropy coefficient
                 toward a TARGET normalised entropy measured over VALID
                 actions (the PPO update's `cond_entropy_norm`), a
                 multiplicative dual update, bounded. It replaces the one-time
                 post-resume boost (cancelled when the controller starts).
                 The target is multiplied by the registry coefficient
                 `explore.entropy_target_scale` when WP-8 publishes it.
  return norm    `ReturnNormalizer`: a running range (P95 - P5, EMA, floor 1)
                 of the PPO return targets; the policy divides its VALUE loss
                 by scale^2. Advantages are not touched, so their ordering is
                 unchanged; scale 1.0 = old behaviour exactly.
  plasticity     `PlasticityMonitor`: dormant-unit ratio per hidden layer of
                 the actor and critic on the policy's fixed probe (a unit is
                 dormant when its mean |activation| over the batch, relative
                 to the layer mean, is below `tau`; Sokar et al.). Optional
                 refresh (`refresh`: false): re-initialise the incoming
                 weights of dormant units and zero their outgoing weights
                 (function-preserving), reset their Adam moments, at most
                 `max_frac` of a layer per phase.
  drift response `DriftResponder` (separate switch `drift_response`): when the
                 existing behaviour-drift alarm (infra/monitors.BehaviorDrift)
                 fires, a transient boost of the policy learning rate and of
                 the entropy target decays linearly back over `hold_segments`.

ONE OWNER PER COEFFICIENT. Every value this module moves is claimed in the
registry (core/coefficients.py) as owner `wp9_adaptive`: policy.entropy_coef,
policy.lr_scale, wm.lr_scale. WP-8's names are different, so the two
controllers cannot fight; a failed phase releases all of them to base.

Fixed by design (not learned): schedule shapes, controller rates, bounds,
`tau`, the boost sizes. Nothing here names a game or task meaning.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
OWNER = "wp9_adaptive"
_ACTS = (nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.LeakyReLU, nn.Tanh)


# ------------------------------------------------------------------ schedule
def lr_scale(kind: str, t: float, horizon: float = 1.0e6, warmup: float = 0.0,
             floor: float = 0.1) -> float:
    """Multiplier in [floor, 1] (warmup ramps from `floor`). t, horizon and
    warmup are in the same unit (lifetime env steps)."""
    kind = str(kind)
    t = max(0.0, float(t))
    if kind == "constant":
        return 1.0
    w = 1.0
    if warmup > 0 and t < warmup:
        w = floor + (1.0 - floor) * t / warmup
        if kind in ("warmup", "warmup_cosine"):
            return w
    if kind == "warmup":
        return 1.0
    p = min(1.0, max(0.0, (t - max(0.0, warmup)) / max(1.0, horizon - warmup)))
    if kind == "linear":
        return floor + (1.0 - floor) * (1.0 - p)
    if kind in ("cosine", "warmup_cosine"):
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    raise ValueError(f"unknown lr schedule {kind!r}")


# ------------------------------------------------------------------- entropy
class EntropyController:
    """coef <- clip(coef * exp(eta * (target - H)))  (H, target in [0, 1]).
    Entropy below target raises the coefficient; above lowers it."""

    def __init__(self, base: float, target: float = 0.5, eta: float = 0.5,
                 lo: float = 1e-4, hi: float = 0.1):
        self.coef = float(base)
        self.target, self.eta = float(target), float(eta)
        self.lo, self.hi = float(lo), float(hi)
        self.last_h = float("nan")

    def update(self, h_norm: float, target: Optional[float] = None) -> float:
        tgt = self.target if target is None else float(target)
        if not math.isfinite(float(h_norm)):      # GOV-015 F-14: +-inf is not evidence either
            return self.coef
        self.last_h = float(h_norm)
        step = max(-1.0, min(1.0, self.eta * (tgt - float(h_norm))))
        self.coef = min(self.hi, max(self.lo, self.coef * math.exp(step)))
        return self.coef


# ------------------------------------------------------------ return scaling
class ReturnNormalizer:
    """Running range of return targets -> value-loss scale (>= 1)."""

    def __init__(self, decay: float = 0.99, lo_q: float = 5.0, hi_q: float = 95.0,
                 max_scale: float = 100.0, symlog: bool = False):
        self.decay, self.lo_q, self.hi_q = float(decay), float(lo_q), float(hi_q)
        self.max_scale, self.symlog = float(max_scale), bool(symlog)
        self.range: Optional[float] = None
        self.scale = 1.0

    def update(self, returns: np.ndarray) -> float:
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size < 2:
            return self.scale
        if self.symlog:
            r = np.sign(r) * np.log1p(np.abs(r))
        rng = float(np.percentile(r, self.hi_q) - np.percentile(r, self.lo_q))
        self.range = rng if self.range is None else (
            self.decay * self.range + (1.0 - self.decay) * rng)
        self.scale = min(self.max_scale, max(1.0, self.range))
        return self.scale


# ---------------------------------------------------------------- plasticity
class PlasticityMonitor:
    """Dormant-unit ratio per hidden layer, optional ReDo-style refresh."""

    def __init__(self, tau: float = 0.025, refresh: bool = False,
                 max_frac: float = 0.05, min_dormant_frac: float = 0.1,
                 probe_rows: int = 256):
        self.tau, self.refresh = float(tau), bool(refresh)
        self.max_frac, self.min_frac = float(max_frac), float(min_dormant_frac)
        self.rows = int(probe_rows)
        self.last: Dict[str, float] = {}
        self.refreshed_total = 0

    @staticmethod
    def _units(policy) -> List[Tuple[str, nn.Linear, nn.Module, nn.Linear]]:
        """(name, incoming Linear, activation, outgoing Linear) per hidden layer."""
        out = []

        def seq_units(name, seq, tail):
            ch = list(seq.children())
            for i, m in enumerate(ch):
                if not isinstance(m, nn.Linear):
                    continue
                j = i + 1
                while j < len(ch) and not isinstance(ch[j], _ACTS):
                    if isinstance(ch[j], nn.Linear):
                        break
                    j += 1
                if j >= len(ch) or not isinstance(ch[j], _ACTS):
                    continue
                k = j + 1
                nxt = None
                while k < len(ch):
                    if isinstance(ch[k], nn.Linear):
                        nxt = ch[k]
                        break
                    k += 1
                if nxt is None:
                    nxt = tail
                if nxt is not None:
                    out.append((f"{name}.{i}", m, ch[j], nxt))

        a = getattr(policy, "actor", None)
        c = getattr(policy, "critic", None)
        if a is not None and hasattr(a, "shared"):
            tail = getattr(a, "action_head", None) or getattr(a, "mean_head", None)
            seq_units("actor", a.shared, tail)
        if c is not None and hasattr(c, "net"):
            seq_units("critic", c.net, None)
        return out

    def _probe_inputs(self, policy):
        pr = getattr(policy, "_probe", None)
        if not pr or pr.get("feats") is None:
            return None
        n = min(self.rows, int(pr["feats"].shape[0]))
        sl = slice(0, n)
        k = pr.get("knowledge")
        p = pr.get("proprio")
        return policy._augment(pr["feats"][sl], None if k is None else k[sl],
                               None if p is None else p[sl])

    @torch.no_grad()
    def measure(self, policy) -> Dict[str, torch.Tensor]:
        """name -> bool mask of dormant units (also fills self.last)."""
        units = self._units(policy)
        x = self._probe_inputs(policy)
        if not units or x is None:
            return {}
        acts: Dict[str, torch.Tensor] = {}
        hooks = []
        for name, _l, act, _n in units:
            def mk(nm):
                def h(_m, _i, o):
                    acts[nm] = o.detach().reshape(-1, o.shape[-1]).abs().mean(0)
                return h
            hooks.append(act.register_forward_hook(mk(name)))
        try:
            policy.actor.shared(x)
            policy.critic(x)
        finally:
            for h in hooks:
                h.remove()
        masks, self.last = {}, {}
        for name, v in acts.items():
            score = v / (v.mean() + 1e-8)
            m = score <= self.tau
            masks[name] = m
            self.last[name] = float(m.float().mean())
        return masks

    def maybe_refresh(self, policy, masks: Dict[str, torch.Tensor]) -> int:
        if not self.refresh:
            return 0
        opt = getattr(policy, "optimizer", None)
        n_done = 0
        for name, lin, _act, nxt in self._units(policy):
            m = masks.get(name)
            if m is None or float(m.float().mean()) < self.min_frac:
                continue
            idx = torch.nonzero(m).flatten()
            cap = max(1, int(self.max_frac * m.numel()))
            if idx.numel() > cap:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:cap]]
            with torch.no_grad():
                fresh = torch.empty_like(lin.weight)
                nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
                lin.weight[idx] = fresh[idx]
                if lin.bias is not None:
                    lin.bias[idx] = 0.0
                nxt.weight[:, idx] = 0.0          # outgoing: function-preserving
            if opt is not None:
                for p, rows, cols in ((lin.weight, idx, None),
                                      (lin.bias, idx, None),
                                      (nxt.weight, None, idx)):
                    st = opt.state.get(p) if p is not None else None
                    if not st:
                        continue
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            if cols is not None:
                                st[key][:, cols] = 0.0
                            else:
                                st[key][rows] = 0.0
            n_done += int(idx.numel())
        self.refreshed_total += n_done
        return n_done


# --------------------------------------------------------------- drift boost
class DriftResponder:
    """Transient boost after a drift alarm, decaying linearly over `hold`."""

    def __init__(self, hold: int = 8, lr_boost: float = 1.5,
                 entropy_boost: float = 1.3):
        self.hold = max(1, int(hold))
        self.lr_boost, self.ent_boost = float(lr_boost), float(entropy_boost)
        self.left = 0
        self.seen = 0

    def alarm(self, count: int) -> bool:
        if count > self.seen:
            self.seen = count
            self.left = self.hold
            return True
        return False

    def factors(self) -> Tuple[float, float]:
        f = self.left / self.hold
        return (1.0 + (self.lr_boost - 1.0) * f, 1.0 + (self.ent_boost - 1.0) * f)

    def tick(self) -> None:
        self.left = max(0, self.left - 1)


# -------------------------------------------------------------------- facade
class AdaptiveOptim:
    def __init__(self, cfg: Dict[str, Any], registry, policy):
        c = dict(cfg or {})
        self.cfg, self.reg = c, registry
        sc = c.get("schedule", {}) or {}
        self.sch_on = bool(sc.get("enabled", False))
        self.sch = {"policy": dict(sc.get("policy", {}) or {}),
                    "wm": dict(sc.get("world_model", {}) or {})}
        ec = c.get("entropy", {}) or {}
        self.ent: Optional[EntropyController] = None
        self.ent_base: Optional[float] = None
        if bool(ec.get("enabled", False)) and not getattr(policy, "continuous", False):
            self.ent_base = float(policy.entropy_coef)
            self.ent = EntropyController(
                float(policy.entropy_coef), float(ec.get("target", 0.5)),
                float(ec.get("eta", 0.5)), float(ec.get("min_coef", 1e-4)),
                float(ec.get("max_coef", 0.1)))
            registry.claim("policy.entropy_coef", OWNER, self.ent.coef,
                           self.ent.lo, self.ent.hi)
            policy.start_entropy_boost(1.0, 0)         # replaces the one-time boost
        rc = c.get("return_norm", {}) or {}
        self.ret: Optional[ReturnNormalizer] = None
        if bool(rc.get("enabled", False)):
            self.ret = ReturnNormalizer(
                float(rc.get("decay", 0.99)), float(rc.get("lo_q", 5.0)),
                float(rc.get("hi_q", 95.0)), float(rc.get("max_scale", 100.0)),
                getattr(policy, "value_space", "") == "symlog")
            policy.return_normalizer = self.ret         # read in train_step
        pc = c.get("plasticity", {}) or {}
        self.plast: Optional[PlasticityMonitor] = None
        if bool(pc.get("enabled", False)):
            self.plast = PlasticityMonitor(
                float(pc.get("tau", 0.025)), bool(pc.get("refresh", False)),
                float(pc.get("max_frac", 0.05)),
                float(pc.get("min_dormant_frac", 0.1)),
                int(pc.get("probe_rows", 256)))
        dc = c.get("drift_response", {}) or {}
        self.drift: Optional[DriftResponder] = None
        if bool(dc.get("enabled", False)):
            self.drift = DriftResponder(int(dc.get("hold_segments", 8)),
                                        float(dc.get("lr_boost", 1.5)),
                                        float(dc.get("entropy_boost", 1.3)))
        if self.sch_on or self.drift is not None:
            registry.claim("policy.lr_scale", OWNER, 1.0, 0.05, 4.0)
            registry.claim("wm.lr_scale", OWNER, 1.0, 0.05, 4.0)
        self._base: Dict[int, float] = {}
        self._last_stats = None
        self.every = max(1, int(c.get("every_segments", 1)))
        self.seg = 0
        self.last_line = ""

    # -- helpers ------------------------------------------------------------
    def _apply_lr(self, opt, scale: float, base: Optional[float] = None) -> None:
        for g in getattr(opt, "param_groups", []):
            # GOV-015 F-19: with a known configured base (the policy's) the rate
            # is base x scale. Capturing the live rate lazily took the SCALED
            # rate restored with the optimizer as the new base, so a schedule
            # compounded on every relaunch.
            b = (float(base) if base is not None
                 else self._base.setdefault(id(g), float(g["lr"])))
            g["lr"] = b * float(scale)

    def _sched(self, which: str, t: float) -> float:
        s = self.sch[which]
        return lr_scale(str(s.get("kind", "constant")), t,
                        float(s.get("horizon", 1.0e6)),
                        float(s.get("warmup", 0.0)), float(s.get("floor", 0.1)))

    def release(self, policy=None) -> None:
        for n in ("policy.entropy_coef", "policy.lr_scale", "wm.lr_scale"):
            try:
                self.reg.release(n, OWNER)
            except Exception:
                pass
        if policy is not None:
            self._apply_lr(getattr(policy, "optimizer", None), 1.0,
                           getattr(policy, "base_lr", None))
            # GOV-015 F-17: the entropy controller also moved policy.entropy_coef;
            # releasing the claim without restoring the value left the last
            # controlled coefficient in force.
            if getattr(self, "ent_base", None) is not None:
                policy.entropy_coef = self.ent_base
            policy.return_normalizer = None
            policy.return_scale = 1.0

    # -- once per segment ----------------------------------------------------
    def run(self, policy, wm_optimizers: List[Any], t_life: float,
            drift_count: int) -> str:
        self.seg += 1
        if self.seg % self.every:
            return ""
        bits = []
        lr_f = ent_f = 1.0
        if self.drift is not None:
            if self.drift.alarm(int(drift_count)):
                bits.append("drift alarm -> boost")
            lr_f, ent_f = self.drift.factors()
            self.drift.tick()
        if self.sch_on or self.drift is not None:
            sp = self.reg.set("policy.lr_scale", OWNER,
                              self._sched("policy", t_life) * lr_f)
            sw = self.reg.set("wm.lr_scale", OWNER, self._sched("wm", t_life))
            self._apply_lr(policy.optimizer, sp, getattr(policy, "base_lr", None))
            for o in wm_optimizers:
                if o is not None and len(getattr(o, "state", {}) or {}) > 0:
                    self._apply_lr(o, sw)
            bits.append(f"lr policy x{sp:.3f} wm x{sw:.3f}")
        st = getattr(policy, "last_update_stats", None)
        if self.ent is not None and st is not None and st is not self._last_stats:
            self._last_stats = st
            h = st.get("cond_entropy_norm")
            if h is not None:
                scale = self.reg.get("explore.entropy_target_scale", 1.0) or 1.0
                # RW-1 (GOV-012): uncertainty gain on the target; 1.0 unless the
                # relation NE->AROUS is on and has claimed the coefficient.
                scale = scale * (self.reg.get("explore.uncertainty_scale", 1.0)
                                 or 1.0)
                tgt = min(0.99, self.ent.target * float(scale) * ent_f)
                c = self.ent.update(float(h), tgt)
                policy.entropy_coef = self.reg.set("policy.entropy_coef", OWNER, c)
                self.ent.coef = policy.entropy_coef
                bits.append(f"entropy {h:.3f}->target {tgt:.3f} coef {policy.entropy_coef:.4f}")
            else:
                bits.append("entropy: no cond_entropy_norm (stats off)")
        if self.ret is not None:
            bits.append(f"return scale {self.ret.scale:.2f}")
        if self.plast is not None:
            masks = self.plast.measure(policy)
            if masks:
                n = self.plast.maybe_refresh(policy, masks)
                bits.append("dormant " + " ".join(
                    f"{k} {v:.2f}" for k, v in self.plast.last.items())
                    + (f" refreshed {n}" if n else ""))
        self.last_line = " ".join(bits)
        return self.last_line
