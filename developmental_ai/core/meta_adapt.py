"""WP-13 (GOV-009, GOV-011): online hyperparameter adaptation inside the one
brain. BUILT, UNTESTED. Default OFF (`meta_adapt.enabled: false`).

ONLY the first version in the plan: a hill-climber over a few PPO coefficients
driven by learning progress. NOT built: population-based training (it needs
several brains; ADR-001 fixes one shared brain), and meta-learning proper
(gradient-based or recurrent): deferred to RES-007 H-13. One hour of one game
cannot supply the many tasks proper meta-learning needs, so the honest
expectation is null.

Coefficients (each CLAIMED in the registry, `core/coefficients.py`, owner
`wp13_meta`; the names are disjoint from WP-8's `mixer.*`/`explore.*` and
WP-9's `policy.entropy_coef`/`policy.lr_scale`/`wm.lr_scale`, so the one-owner
rule makes a fight impossible: a claim of a taken name raises):
  meta.ppo_clip     -> policy.clip_range
  meta.gae_lambda   -> policy.gae_lambda
  meta.value_coef   -> policy.value_coef
(`coefficients` in the config picks a subset.) Base = the value the policy had
when the adapter started; bounds = base / `range` .. base x `range` (gae_lambda
is also kept below 0.995).

SIGNAL. Learning progress, defined as in WP-3 / WP-8 and measured by the same
function: the pooled relative drop of the world model's fresh-data loss on each
stream's newest never-trained window between consecutive measurements
(`core.curriculum.measure_streams`). Never a reward; nothing is paid.

PROCEDURE (paired, so slow drift in progress cancels):
  ctrl_a  `epoch_len` measurements at the current values -> Pa
  trial   one coefficient moved by a relative `step` in its current direction
          -> Pt          (coefficients take turns, round robin)
  ctrl_b  back at the current values -> Pb
  diff = Pt - (Pa + Pb) / 2. Noise = running mean of |Pa - Pb| (two adjacent
  controls at identical settings). ACCEPT (the trial value becomes current)
  only if diff > accept_z x noise and the noise estimate has `min_noise_n`
  samples; otherwise the direction of that coefficient flips and Pb is reused
  as the next Pa. The first measurement of every epoch is dropped (its interval
  ran under the previous setting). A trial is never kept without evidence, so
  with no signal the values stay at base and the adapter costs only the
  measurement.

Guard-becomes-latch: values are bounded, slew-limited by `step`, directions
flip on a failed trial, so a coefficient pinned at a bound can leave it; a
failure of the phase releases every coefficient to base (the caller restores
the policy attributes).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

OWNER = "wp13_meta"

# registry name -> (policy attribute, hard lo, hard hi)
SPEC = {
    "ppo_clip": ("meta.ppo_clip", "clip_range", 0.02, 0.5),
    "gae_lambda": ("meta.gae_lambda", "gae_lambda", 0.5, 0.995),
    "value_coef": ("meta.value_coef", "value_coef", 0.05, 4.0),
}


class HyperAdapter:
    def __init__(self, cfg: dict, registry, policy):
        cfg = dict(cfg or {})
        self.reg = registry
        self.every = max(1, int(cfg.get("every_segments", 4)))
        self.seq_len = int(cfg.get("seq_len", 32))
        self.epoch_len = max(2, int(cfg.get("epoch_len", 4)))
        self.step = float(cfg.get("step", 0.1))
        self.range = max(1.05, float(cfg.get("range", 2.0)))
        self.accept_z = float(cfg.get("accept_z", 1.0))
        self.min_noise_n = max(1, int(cfg.get("min_noise_n", 3)))
        self.noise_decay = float(cfg.get("noise_decay", 0.8))
        self.blind_after = max(1, int(cfg.get("blind_after", 3)))
        self.empty = 0
        names = list(cfg.get("coefficients", ["ppo_clip", "gae_lambda"]))
        self.names: List[str] = [n for n in names if n in SPEC]
        self.attr: Dict[str, str] = {}
        self.cur: Dict[str, float] = {}
        self.dir: Dict[str, int] = {}
        for n in self.names:
            rname, attr, hlo, hhi = SPEC[n]
            base = float(getattr(policy, attr))
            lo = max(hlo, base / self.range)
            hi = min(hhi, base * self.range)
            lo, hi = min(lo, base), max(hi, base)
            self.reg.claim(rname, OWNER, base, lo, hi)   # raises if owned
            self.attr[n] = attr
            self.cur[n] = base
            self.dir[n] = 1
        self.calls = 0
        self.phase = "ctrl_a"
        self.k = 0                       # round-robin index
        self.target: Optional[str] = None
        self.trial_v = 0.0
        self.P: Dict[str, float] = {}
        self.epoch: List[float] = []
        self.n_in_epoch = 0
        self.prev_loss: Dict[int, float] = {}
        self.noise = 0.0
        self.noise_n = 0
        self.accepts = 0
        self.rejects = 0
        self.last = "none"

    # ---- schedule ---------------------------------------------------
    def due(self) -> bool:
        self.calls += 1
        return bool(self.names) and self.calls % self.every == 0

    # ---- signal -----------------------------------------------------
    def _progress(self, meas: Dict[int, tuple]) -> Optional[float]:
        drops = []
        for i, (loss, _attr) in meas.items():
            if not math.isfinite(float(loss)):
                # GOV-015 F-15: one NaN loss left noise = nan and disabled
                # acceptance for good. Not evidence: the reading is skipped.
                continue
            p = self.prev_loss.get(i)
            self.prev_loss[i] = float(loss)
            if p is not None and p > 1e-9:
                drops.append((p - float(loss)) / p)
        return sum(drops) / len(drops) if drops else None

    def update(self, meas: Dict[int, tuple]) -> Dict:
        d = self._progress(meas)
        if not meas:
            # GOV-015 F-18: with the measurement source gone a running trial
            # can never be judged; its value was held until the source
            # returned. After `blind_after` empty updates restore the accepted
            # value and start a fresh control.
            self.empty = getattr(self, "empty", 0) + 1
            if self.empty >= self.blind_after and self.phase == "trial":
                self.reg.set(SPEC[self.target][0], OWNER, self.cur[self.target],
                             max_step=abs(self.cur[self.target]) * 2.0)
                self.phase = "ctrl_a"
                self.epoch, self.n_in_epoch = [], 0
                self.last = f"released {self.target} (no measurement)"
        else:
            self.empty = 0
        if d is None:
            return self.state()
        self.n_in_epoch += 1
        if self.n_in_epoch > 1:          # first interval ran under the old setting
            self.epoch.append(d)
        if self.n_in_epoch > self.epoch_len:
            self._finish_epoch(sum(self.epoch) / max(1, len(self.epoch)))
            self.epoch, self.n_in_epoch = [], 0
        return self.state()

    # ---- state machine ----------------------------------------------
    def _start_trial(self):
        n = self.names[self.k % len(self.names)]
        self.k += 1
        v = self.cur[n] * (1.0 + self.dir[n] * self.step)
        rname = SPEC[n][0]
        got = self.reg.set(rname, OWNER, v, max_step=abs(self.cur[n] * self.step) * 1.01)
        if abs(got - self.cur[n]) < 1e-9:          # pinned at a bound: turn round
            self.dir[n] = -self.dir[n]
            got = self.reg.set(rname, OWNER, self.cur[n] * (1.0 + self.dir[n] * self.step),
                               max_step=abs(self.cur[n] * self.step) * 1.01)
        self.target, self.trial_v = n, got
        self.phase = "trial"

    def _finish_epoch(self, P: float):
        if self.phase == "ctrl_a":
            self.P["a"] = P
            self._start_trial()
        elif self.phase == "trial":
            self.P["t"] = P
            self.reg.set(SPEC[self.target][0], OWNER, self.cur[self.target],
                         max_step=abs(self.cur[self.target]) * 2.0)
            self.phase = "ctrl_b"
        else:                                       # ctrl_b: decide
            Pa, Pt, Pb = self.P["a"], self.P["t"], P
            diff = Pt - 0.5 * (Pa + Pb)
            dn = abs(Pb - Pa)
            self.noise = (dn if self.noise_n == 0 else
                          self.noise_decay * self.noise + (1 - self.noise_decay) * dn)
            self.noise_n += 1
            n = self.target
            if self.noise_n >= self.min_noise_n and diff > self.accept_z * self.noise:
                self.cur[n] = self.trial_v
                self.reg.set(SPEC[n][0], OWNER, self.cur[n],
                             max_step=abs(self.cur[n]) * 2.0)
                self.accepts += 1
                self.last = f"accept {n}->{self.cur[n]:.4g}"
                self.phase = "ctrl_a"               # fresh control at the new value
            else:
                self.dir[n] = -self.dir[n]
                self.rejects += 1
                self.last = f"reject {n}"
                self.P["a"] = Pb                    # Pb is a valid control (unchanged setting)
                self._start_trial()

    # ---- outputs ----------------------------------------------------
    def values(self) -> Dict[str, float]:
        """policy attribute -> value to apply NOW (the trial value while a
        trial runs, else the current value). Registry is the source."""
        return {self.attr[n]: float(self.reg.get(SPEC[n][0], self.cur[n]))
                for n in self.names}

    def base_values(self) -> Dict[str, float]:
        out = {}
        for n in self.names:
            c = self.reg._c.get(SPEC[n][0])
            out[self.attr[n]] = c.base if c is not None else self.cur[n]
        return out

    def release(self) -> Dict[str, float]:
        base = self.base_values()
        for n in self.names:
            try:
                self.reg.release(SPEC[n][0], OWNER)
            except Exception:
                pass
        self.names = []
        return base

    def state(self) -> Dict:
        return {"phase": self.phase, "target": self.target,
                "accepts": self.accepts, "rejects": self.rejects,
                "noise": self.noise, "last": self.last}

    def report(self) -> str:
        vals = " ".join(f"{a}={v:.4g}" for a, v in self.values().items())
        return (f"phase={self.phase} target={self.target} {vals} "
                f"accept={self.accepts} reject={self.rejects} "
                f"noise={self.noise:.4f}(n={self.noise_n}) last={self.last}")
