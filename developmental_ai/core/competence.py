"""WP-8 (GOV-009, GOV-011): self-evaluation, competence and autonomy.
BUILT, UNTESTED. Default OFF (`competence.enabled: false`).

A per-CONTEXT competence tracker (a context is a stream here, D-41),
generalising the skill-only `self_model.CompetencePredictor`: it is built from
the two quantities the agent already measures on each stream's newest,
never-trained window (`core.curriculum.measure_streams`):
  learning progress   relative drop of the world model's fresh-data loss
                      between consecutive measurements (EMA), and
  action attribution  how much rolling the actions changes the world model's
                      surprise (the agent's own agency).
Reported TOGETHER but kept as separate terms:
  competence   c = 1 - loss_ema / worst loss seen in the context (relative,
               not absolute), with its progress p;
  autonomy     attribution (squashed x/(x+running mean)); empowerment is not
               wired (D-42).

CONTROL. Competence NEVER becomes a reward. It moves two coefficients, each
owned by this module in the coefficient registry (core/coefficients.py, ONE
owner per coefficient):
  mixer.intrinsic_gain          multiplier on the intrinsic weight of the
                                reward mixer (applied in RewardMixer.mix);
  explore.entropy_target_scale  multiplier on WP-9's entropy-controller
                                target (read there; unused while that is off).
Flat or falling progress (the pooled signal vs its own running scale) raises
both (explore); rising progress lowers them (exploit). The signal is smoothed
(EMA), the output is bounded to [1 - amp, 1 + amp] and slew-limited per phase,
and nothing is written until every context has `min_meas` measurements.

SELF-MODEL HEAD. A small MLP predicts a context's NEXT relative loss drop from
[c, p, attribution, experience, relative loss]; trained online on the pair
(features then, outcome now). Its absolute error is logged next to the error
of the persistence baseline (predict "same progress as last time"). It is not
used for control.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn

OWNER = "wp8_competence"


def _sq(x: float, mean: float) -> float:
    x = max(0.0, float(x))
    return x / (x + max(mean, 1e-8)) if (x + mean) > 0 else 0.0


class _Ctx:
    def __init__(self, hist: int):
        self.n = 0
        self.prev: Optional[float] = None
        self.loss_ema: Optional[float] = None
        self.hist: deque = deque(maxlen=hist)
        self.lp = 0.0
        self.attr = 0.0
        self.f_prev: Optional[List[float]] = None
        self.pred_prev: Optional[float] = None
        self.hold = 0                         # updates still to skip after a context change


class SelfModelHead(nn.Module):
    """Predicts a context's next relative loss drop (in [-1, 1])."""

    def __init__(self, n_in: int = 5, hidden: int = 16, lr: float = 5e-3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1), nn.Tanh())
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    @torch.no_grad()
    def predict(self, f: List[float]) -> float:
        return float(self.net(torch.tensor([f], dtype=torch.float32)))

    def fit(self, f: List[float], y: float) -> float:
        p = self.net(torch.tensor([f], dtype=torch.float32))
        loss = (p - torch.tensor([[y]], dtype=torch.float32)).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)


class CompetenceController:
    def __init__(self, cfg: Dict, registry):
        c = dict(cfg or {})
        self.reg = registry
        self.every = max(1, int(c.get("every_segments", 4)))
        self.seq_len = int(c.get("seq_len", 32))
        self.alpha = float(c.get("progress_ema", 0.3))
        self.hist = int(c.get("history", 50))
        self.min_meas = int(c.get("min_meas", 3))
        self.smooth = float(c.get("signal_ema", 0.2))
        self.amp = float(c.get("gain_amp", 0.3))
        self.step = float(c.get("max_step", 0.05))
        self.mod_mix = bool(c.get("modulate_mixture", True))
        self.mod_explore = bool(c.get("modulate_exploration", False))
        self.eamp = float(c.get("entropy_amp", 0.3))
        # GOV-015 9.6 (BND-04): a curriculum context change moves the loss for reasons that are not learning. For this many
        # updates after one, that context's loss is re-baselined instead of scored. It expires by itself: the escape is a count.
        self.context_hold = max(0, int(c.get("context_hold", 2)))
        self.ctx: Dict[int, _Ctx] = {}
        self.head = SelfModelHead(lr=float(c.get("selfmodel_lr", 5e-3)))
        self.seg = 0
        self.empty = 0                        # consecutive empty updates
        self.flat = 0.5                       # smoothed "progress is flat" signal
        self.p_scale = 0.0                    # running mean |progress|
        self.attr_mean = 0.0
        self.err = deque(maxlen=200)          # self-model absolute errors
        self.err_base = deque(maxlen=200)     # persistence-baseline errors
        self.last_line = ""
        g = self.amp
        if self.mod_mix:
            self.reg.claim("mixer.intrinsic_gain", OWNER, 1.0,
                           1.0 - g, 1.0 + g)
        if self.mod_explore:
            self.reg.claim("explore.entropy_target_scale", OWNER, 1.0,
                           1.0 - self.eamp, 1.0 + self.eamp)

    def note_context_change(self, i: int) -> None:
        """The curriculum (WP-3) has queued a new context for stream `i`: do not score the next loss change as progress."""
        s = self.ctx.get(int(i))
        if s is not None and self.context_hold:
            s.hold = self.context_hold

    def due(self) -> bool:
        self.seg += 1
        return self.seg % self.every == 0

    def _features(self, s: _Ctx) -> List[float]:
        worst = max(s.hist) if s.hist else 1.0
        rel = (s.loss_ema or 0.0) / max(worst, 1e-8)
        c = max(0.0, min(1.0, 1.0 - rel))
        return [c, max(-1.0, min(1.0, s.lp)), _sq(s.attr, self.attr_mean),
                s.n / (s.n + 10.0), min(1.0, rel)]

    def update(self, meas: Dict[int, tuple]) -> Dict:
        # GOV-015 F-11: a NaN or infinite loss or attribution is not evidence.
        # One made loss_ema NaN for good and drifted the gain. Drop that
        # context's reading for this update; the rest are used as before.
        meas = {i: (l, a) for i, (l, a) in meas.items()
                if math.isfinite(float(l)) and math.isfinite(float(a))}
        if not meas:
            # GOV-015 F-25: no measurement is "unknown", not "unchanged". The
            # smoothed state is stale, so release toward neutral by at most
            # max_step per update (the same bound the controller moves by).
            self.empty += 1
            out = {"ready": bool(self.ctx) and all(
                s.n >= self.min_meas for s in self.ctx.values()), "empty": True}
            self.flat = (1 - self.smooth) * self.flat + self.smooth * 0.5
            if self.mod_mix:
                out["gain"] = self.reg.set("mixer.intrinsic_gain", OWNER,
                                           1.0, self.step)
            if self.mod_explore:
                out["entropy_scale"] = self.reg.set(
                    "explore.entropy_target_scale", OWNER, 1.0, self.step)
            return out
        self.empty = 0
        attrs = [a for (_, a) in meas.values()]
        if attrs:
            m = sum(max(0.0, a) for a in attrs) / len(attrs)
            self.attr_mean = m if self.attr_mean == 0 else 0.9 * self.attr_mean + 0.1 * m
        for i, (loss, attr) in meas.items():
            s = self.ctx.setdefault(int(i), _Ctx(self.hist))
            rel = 0.0
            if s.hold > 0:
                s.hold -= 1
                s.f_prev = s.pred_prev = None     # no self-model pair across the change
                s.prev = float(loss)              # the new baseline
                s.loss_ema = loss if s.loss_ema is None else (
                    (1 - self.alpha) * s.loss_ema + self.alpha * loss)
                s.attr = (1 - self.alpha) * s.attr + self.alpha * max(0.0, float(attr))
                s.n += 1
                continue
            if s.prev is not None:
                rel = max(-1.0, min(1.0, (s.prev - loss) / (abs(s.prev) + 1e-8)))
                if s.f_prev is not None:          # score + train the self-model
                    self.err.append(abs((s.pred_prev or 0.0) - rel))
                    self.err_base.append(abs(s.f_prev[1] - rel))
                    self.head.fit(s.f_prev, rel)
                s.lp = (1 - self.alpha) * s.lp + self.alpha * rel
            s.prev = float(loss)
            s.loss_ema = loss if s.loss_ema is None else (
                (1 - self.alpha) * s.loss_ema + self.alpha * loss)
            s.hist.append(float(s.loss_ema))
            s.attr = (1 - self.alpha) * s.attr + self.alpha * max(0.0, float(attr))
            s.n += 1
            s.f_prev = self._features(s)
            s.pred_prev = self.head.predict(s.f_prev)
        ready = bool(self.ctx) and all(s.n >= self.min_meas for s in self.ctx.values())
        out = {"ready": ready}
        if not ready:
            return out
        p_bar = sum(s.lp for s in self.ctx.values()) / len(self.ctx)
        absm = sum(abs(s.lp) for s in self.ctx.values()) / len(self.ctx)
        self.p_scale = absm if self.p_scale == 0 else 0.9 * self.p_scale + 0.1 * absm
        rising = max(0.0, min(1.0, p_bar / (self.p_scale + 1e-8)))
        flat = 1.0 - rising                   # falling progress counts as flat
        self.flat = (1 - self.smooth) * self.flat + self.smooth * flat
        sig = 2.0 * self.flat - 1.0           # +1 flat (explore) .. -1 rising
        if self.mod_mix:
            out["gain"] = self.reg.set("mixer.intrinsic_gain", OWNER,
                                       1.0 + self.amp * sig, self.step)
        if self.mod_explore:
            out["entropy_scale"] = self.reg.set(
                "explore.entropy_target_scale", OWNER,
                1.0 + self.eamp * sig, self.step)
        return out

    def release(self) -> None:
        for n in ("mixer.intrinsic_gain", "explore.entropy_target_scale"):
            try:
                self.reg.release(n, OWNER)
            except Exception:
                pass

    def report(self, out: Dict) -> str:
        cs = [self._features(s) for s in self.ctx.values()]
        k = max(1, len(cs))
        comp = sum(f[0] for f in cs) / k
        prog = sum(f[1] for f in cs) / k
        aut = sum(f[2] for f in cs) / k
        ea = sum(self.err) / max(1, len(self.err))
        eb = sum(self.err_base) / max(1, len(self.err_base))
        bits = [f"contexts {len(cs)}", f"competence {comp:.3f}",
                f"progress {prog:+.4f}", f"autonomy(attr) {aut:.3f}",
                "empowerment n/a", f"flat {self.flat:.2f}",
                f"selfmodel_err {ea:.4f} (persistence {eb:.4f})"]
        if "gain" in out:
            bits.append(f"mix_gain {out['gain']:.3f}")
        if "entropy_scale" in out:
            bits.append(f"ent_target_scale {out['entropy_scale']:.3f}")
        if not out.get("ready"):
            bits.append("warming up")
        return " ".join(bits)
