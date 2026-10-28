"""WP-4 (GOV-009, GOV-011): decision-time planning, arbitration, latency
budget. BUILT, UNTESTED. Every switch OFF (`planning.*` in universal.yaml).

FIRST VERSION (GOV-009 WP-4 item 3, ADR-005): planning runs in the LEARNING
PHASE (once per segment, after PPO, in `PlanningPhase.run`), NOT inside the
acting step. It plans from replayed real states through the world model's
imagined rollouts, scores candidate action sequences by the reward head and
the critic's value (minus a pessimism term from ensemble disagreement), and
reports how often and by how much the plan beats the policy's own imagined
rollouts. Optionally (`planning.distill.weight > 0`) it pulls the policy's
action distribution toward the plan on the states the arbiter picks
("planning to correct the policy"). Acting-time planning is NOT wired
(D-30): `Arbiter` and `LatencyBudget.acting_ok` exist so it can be, behind a
separate switch that must pass a MEASURED latency budget.

PARTS
* `CEMPlanner`: cross-entropy search over short categorical action sequences
  (random shooting is the first iteration). Model-agnostic: it takes
  `step_fn`, `reward_fn`, `value_fn`, `penalty_fn` callables so a toy model
  can test that it finds the known best sequence.
* `Arbiter`: plan only when ALL hold: the policy is unsure (calibrated
  action-distribution confidence at or below `conf_max`), the model is trusted at that
  state (ensemble disagreement below its `trust_quantile` in the running
  history), BOTH calibration channels are currently ok (else habit), and the
  time budget allows. Any failed condition falls back to the habit policy.
* `LatencyBudget`: measured seconds per phase / per plan call.
* `PlanningPhase`: orchestrates replay sampling, ensemble training,
  calibration, temperature fitting, planning, arbitration, the optional
  distillation step, telemetry, persistence.

Learned-not-declared: no game or reward meaning; the reward head and critic
are the agent's own learned quantities; thresholds are fixed mechanics.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import torch
import torch.nn.functional as F

from developmental_ai.core.uncertainty import (CalibrationMonitor, EnsembleHeads,
                                               fit_temperature)


def _symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


# --------------------------------------------------------------------------
class CEMPlanner:
    def __init__(self, cfg: Dict[str, Any]):
        c = dict(cfg or {})
        self.horizon = max(1, int(c.get("horizon", 6)))
        self.n = max(4, int(c.get("candidates", 64)))
        self.iters = max(1, int(c.get("iterations", 2)))
        self.elite_frac = min(0.5, max(0.02, float(c.get("elite_frac", 0.125))))
        self.smooth = min(1.0, max(0.1, float(c.get("smoothing", 0.7))))
        self.eps = float(c.get("explore_floor", 0.02))
        self.gamma = float(c.get("gamma", 0.97))
        self.risk = float(c.get("risk", 0.0))        # weight of the disagreement penalty

    def rollout(self, state: Dict[str, torch.Tensor], act_fn: Callable,
                n_actions: int, step_fn, reward_fn, value_fn, penalty_fn=None):
        """Roll R rows forward; `act_fn(state, t)` -> (R,) action indices.
        Returns (score, return, penalty, first_action), each (R,)."""
        st = state
        R = next(iter(st.values())).shape[0]
        dev = next(iter(st.values())).device
        ret = torch.zeros(R, device=dev)
        pen = torch.zeros(R, device=dev)
        first = None
        disc = 1.0
        for t in range(self.horizon):
            idx = act_fn(st, t)
            if first is None:
                first = idx
            a = F.one_hot(idx, n_actions).float()
            if penalty_fn is not None:
                pen = pen + disc * penalty_fn(st, a)
            st = step_fn(st, a)
            ret = ret + disc * reward_fn(st)
            disc *= self.gamma
        ret = ret + disc * value_fn(st)
        return ret - self.risk * pen, ret, pen, first

    @torch.no_grad()
    def plan(self, state: Dict[str, torch.Tensor], n_actions: int, step_fn,
             reward_fn, value_fn, penalty_fn=None,
             init_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = next(iter(state.values())).shape[0]
        dev = next(iter(state.values())).device
        N, H, A = self.n, self.horizon, int(n_actions)
        tiled = {k: v.repeat_interleave(N, 0) for k, v in state.items()}
        probs = torch.full((B, H, A), 1.0 / A, device=dev)
        if init_probs is not None:
            probs[:, 0, :] = 0.5 * init_probs.to(dev)[:, :A] + 0.5 / A
            probs = probs / probs.sum(-1, keepdim=True)
        best_s = torch.full((B,), -1e30, device=dev)
        best_first = torch.zeros(B, dtype=torch.long, device=dev)
        k = max(2, int(self.elite_frac * N))
        freq0 = probs[:, 0, :]
        for _ in range(self.iters):
            p = probs.reshape(B * H, A)
            seq = torch.multinomial(p, N, replacement=True)         # (B*H, N)
            seq = seq.view(B, H, N).permute(0, 2, 1).reshape(B * N, H)
            score, _, _, _ = self.rollout(
                tiled, lambda st, t, _s=seq: _s[:, t], A, step_fn, reward_fn,
                value_fn, penalty_fn)
            sc = score.view(B, N)
            el_s, el_i = sc.topk(k, dim=1)                          # (B,k)
            seq3 = seq.view(B, N, H)
            el = torch.gather(seq3, 1, el_i[:, :, None].expand(B, k, H))
            freq = F.one_hot(el, A).float().mean(1)                 # (B,H,A)
            probs = (1 - self.smooth) * probs + self.smooth * freq
            probs = (1 - self.eps) * probs + self.eps / A
            probs = probs / probs.sum(-1, keepdim=True)
            freq0 = freq[:, 0, :]
            top_s, top_i = sc.max(1)
            better = top_s > best_s
            best_s = torch.where(better, top_s, best_s)
            best_first = torch.where(
                better, seq3[torch.arange(B, device=dev), top_i, 0], best_first)
        return {"first_probs": freq0, "best_first": best_first,
                "best_score": best_s, "mean_score": sc.mean(1),
                "score_std": sc.std(1)}


# --------------------------------------------------------------------------
class LatencyBudget:
    def __init__(self, ema: float = 0.8):
        self.ema = float(ema)
        self.phase_s: Optional[float] = None
        self.plan_s: Optional[float] = None

    def _upd(self, old, x):
        return x if old is None else self.ema * old + (1 - self.ema) * x

    def record_phase(self, s: float) -> None:
        self.phase_s = self._upd(self.phase_s, float(s))

    def record_plan(self, s: float) -> None:
        self.plan_s = self._upd(self.plan_s, float(s))

    def acting_ok(self, act_s: float, interval_s: float,
                  limit_frac: float = 0.8) -> bool:
        """Would one more plan call keep the acting step inside the share of
        the decision interval ADR-005 allows? False until a plan was timed."""
        if self.plan_s is None or interval_s <= 0:
            return False
        return (float(act_s) + self.plan_s) <= float(limit_frac) * float(interval_s)


# --------------------------------------------------------------------------
class Arbiter:
    """Plan vs habit. Every failed condition means HABIT (the policy)."""

    def __init__(self, cfg: Dict[str, Any]):
        c = dict(cfg or {})
        self.conf_max = float(c.get("conf_max", 0.6))
        self.trust_q = float(c.get("trust_quantile", 0.7))
        self.require_calibrated = bool(c.get("require_calibrated", True))
        self._hist: Deque[float] = deque(maxlen=int(c.get("history", 512)))
        self.fired = 0
        self.seen = 0
        self.why = {"no_budget": 0, "uncalibrated": 0, "invalid": 0, "confident": 0,
                    "distrust": 0, "plan": 0}

    def decide(self, conf: List[float], disagreement: List[float],
               policy_ok: bool, model_ok: bool, budget_ok: bool) -> List[bool]:
        n = len(conf)
        if len(disagreement) != n:
            raise ValueError("arbitration requires one disagreement per confidence")
        out = [False] * n
        valid = [math.isfinite(c) and 0.0 <= c <= 1.0
                 and math.isfinite(d) and d >= 0.0
                 for c, d in zip(conf, disagreement)]
        for ok, d in zip(valid, disagreement):
            if ok:
                self._hist.append(float(d))
        thr = None
        if self._hist:
            s = sorted(self._hist)
            thr = s[min(len(s) - 1, int(self.trust_q * len(s)))]
        for i in range(n):
            self.seen += 1
            if not valid[i]:
                self.why["invalid"] += 1
            elif not budget_ok:
                self.why["no_budget"] += 1
            elif self.require_calibrated and not (policy_ok and model_ok):
                self.why["uncalibrated"] += 1
            elif conf[i] > self.conf_max:
                self.why["confident"] += 1
            elif thr is not None and disagreement[i] > thr:
                self.why["distrust"] += 1
            else:
                out[i] = True
                self.fired += 1
                self.why["plan"] += 1
        return out


# --------------------------------------------------------------------------
class PlanningPhase:
    """The learning-phase planner. One instance per run."""

    def __init__(self, cfg: Dict[str, Any], n_actions: int, latent_dim: int,
                 device):
        c = dict(cfg or {})
        self.cfg = c
        self.every = max(1, int(c.get("every_segments", 1)))
        self.rows = max(2, int(c.get("rows", 8)))
        self.min_rows = max(2, int(c.get("min_rows", 2)))
        self.seq_len = max(4, int(c.get("seq_len", 8)))
        self.max_phase_s = float(c.get("max_phase_s", 3.0))
        self.A, self.device = int(n_actions), device
        self.planner = CEMPlanner(c.get("planner", {}))
        self.arbiter = Arbiter(c.get("arbiter", {}))
        self.budget = LatencyBudget()
        ec = c.get("ensemble", {}) or {}
        self.ens: Optional[EnsembleHeads] = None
        if bool(ec.get("enabled", False)):
            self.ens = EnsembleHeads(latent_dim, n_actions,
                                     int(ec.get("heads", 4)),
                                     int(ec.get("hidden", 128)),
                                     float(ec.get("lr", 1e-3))).to(device)
            self.ens_steps = max(1, int(ec.get("train_steps", 4)))
        cal = c.get("calibration", {}) or {}
        self.mon = CalibrationMonitor(int(cal.get("window", 2000)),
                                      int(cal.get("min_n", 200)),
                                      float(cal.get("max_ece", 0.10)),
                                      float(cal.get("min_corr", 0.20)))
        self.T = 1.0
        self.fit_T = bool(cal.get("temperature_scaling", True))
        self.T_decay = float(cal.get("temperature_decay", 0.8))
        dc = c.get("distill", {}) or {}
        self.dist_w = float(dc.get("weight", 0.0))
        self.dist_steps = max(1, int(dc.get("steps", 2)))
        self.dist_lr = float(dc.get("lr_scale", 0.1))
        self._opt = None
        self._calls = 0
        self.last: Dict[str, float] = {}
        self._warned: set = set()

    # -- plumbing ----------------------------------------------------------
    def _warn(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            import logging
            logging.getLogger(__name__).warning("planning: %s", msg)

    def _kn0(self, policy, n):
        return (torch.zeros(n, policy.knowledge_dim, device=self.device)
                if policy.knowledge_dim > 0 else None)

    def _logits(self, policy, lat):
        aug = policy._augment(lat, self._kn0(policy, lat.shape[0]), None)
        lg = policy._bound_logits(policy.actor.action_head(policy.actor.shared(aug)))
        return lg[..., :self.A]

    def _value(self, policy, lat):
        aug = policy._augment(lat, self._kn0(policy, lat.shape[0]), None)
        return policy._value_of(aug).reshape(-1)

    @torch.no_grad()
    def _states(self, wm, batch):
        st, _ = wm.observe_sequence(batch["observations"], batch["actions"],
                                    batch.get("proprio"))
        return st, torch.cat([st["h"], st["z"]], -1)

    # -- the phase -------------------------------------------------------------
    def run(self, policy, wm, buf, device) -> Dict[str, float]:
        self._calls += 1
        self.last = {}
        if self._calls % self.every != 0:
            return {}
        if policy.continuous or policy.arch != "rssm":
            self._warn("arch", "idle: needs a discrete action space and arch 'rssm'")
            return {}
        if len(buf) < 4 * self.seq_len or not getattr(buf, "is_ready", True):
            return {}
        t0 = time.time()
        rows = self.rows
        if self.budget.phase_s is not None and self.budget.phase_s > self.max_phase_s:
            rows = max(self.min_rows, self.rows // 2)    # over budget: shrink (grows back when under)
        B = rows
        batch = buf.sample_sequences(B, self.seq_len, device, terminal_fraction=0.0)
        st, lat = self._states(wm, batch)
        act = batch["actions"].float()
        out: Dict[str, float] = {"rows": float(B)}
        # 1. ensemble training on real transitions (state_t, a_t) -> state_{t+1}
        if self.ens is not None:
            ls = []
            for _ in range(self.ens_steps):
                ls.append(self.ens.train_step(
                    lat[:, :-1].reshape(-1, lat.shape[-1]),
                    act[:, :-1].reshape(-1, act.shape[-1])[:, :self.A],
                    lat[:, 1:].reshape(-1, lat.shape[-1])))
            out["ens_loss"] = sum(ls) / len(ls)
        # 2. online diagnostic on recent replay-excluded windows
        out.update(self._calibrate(policy, wm, buf, device))
        temperature = out.get("calibration_temperature", self.T)
        out["T"] = temperature
        out["next_T"] = self.T
        # 3. start states: one random (non-final) step per sequence
        ti = torch.randint(0, self.seq_len - 1, (B,), device=device)
        ar = torch.arange(B, device=device)
        start = {"h": st["h"][ar, ti], "z": st["z"][ar, ti]}
        lat0 = lat[ar, ti]
        # 4. plan, and a policy-rollout baseline from the same states
        rssm = wm.rssm

        def step_fn(s, a):
            return rssm.imagine_step(s, a, knowledge=None)

        def reward_fn(s):
            return _symexp(wm._reward_symlog(rssm.get_latent(s))).reshape(-1)

        def value_fn(s):
            return self._value(policy, rssm.get_latent(s))

        def penalty_fn(s, a):
            if self.ens is None:
                return torch.zeros(a.shape[0], device=a.device)
            return self.ens.disagreement(rssm.get_latent(s), a)

        with torch.no_grad():
            lg0 = self._logits(policy, lat0)
            pi0 = F.softmax(lg0 / temperature, -1)
            t1 = time.time()
            res = self.planner.plan(start, self.A, step_fn, reward_fn, value_fn,
                                    penalty_fn if self.ens is not None else None,
                                    init_probs=pi0)
            self.budget.record_plan(time.time() - t1)
            N = self.planner.n
            tiled = {k: v.repeat_interleave(N, 0) for k, v in start.items()}

            def pol_act(s, t):
                g = self._logits(policy, rssm.get_latent(s))
                return torch.distributions.Categorical(logits=g).sample()

            base, _, _, _ = self.planner.rollout(
                tiled, pol_act, self.A, step_fn, reward_fn, value_fn,
                penalty_fn if self.ens is not None else None)
            base = base.view(B, N)
            conf = pi0.max(-1).values
            dis = (penalty_fn(start, F.one_hot(pi0.argmax(-1), self.A).float())
                   if self.ens is not None else torch.zeros(B, device=device))
        gain = res["best_score"] - base.mean(1)
        agree = (res["first_probs"].argmax(-1) == pi0.argmax(-1)).float().mean()
        out.update({"plan_gain": float(gain.mean()), "plan_gain_pos": float((gain > 0).float().mean()),
                    "agree": float(agree), "conf": float(conf.mean()),
                    "dis": float(dis.mean()), "score_std": float(res["score_std"].mean())})
        # 5. arbitration
        budget_ok = (self.budget.phase_s is None
                     or self.budget.phase_s <= self.max_phase_s)
        use = self.arbiter.decide(conf.tolist(), dis.tolist(),
                                  bool(out["policy_calibration_ok"]),
                                  bool(out["model_calibration_ok"]), budget_ok)
        out["plan_rows"] = float(sum(use))
        # 6. optional distillation on the rows the arbiter chose
        if self.dist_w > 0.0 and any(use):
            out["distill_kl"] = self._distill(policy, lat0, res["first_probs"], use)
        self.budget.record_phase(time.time() - t0)
        out["secs"] = time.time() - t0
        self.last = out
        return dict(out)

    def _calibrate(self, policy, wm, buf, device) -> Dict[str, float]:
        result = {"policy_calibration_ok": 0.0, "model_calibration_ok": 0.0}
        nb = (buf.newest_sequences(self.seq_len, device=device)
              if hasattr(buf, "newest_sequences") else None)
        if nb is None:
            return result
        st, lat = self._states(wm, nb)
        S, T, L = lat.shape
        with torch.no_grad():
            lg = self._logits(policy, lat.reshape(S * T, L))
            tgt = nb["actions"].reshape(S * T, -1)[:, :self.A].argmax(-1)
            # Score before fitting this window; the fitted T is for the next call.
            # Matching sampled behavior actions is not evidence of task success.
            p = F.softmax(lg / self.T, -1)
            result["calibration_temperature"] = self.T
            conf, pred = p.max(-1)
            policy_valid = bool(torch.isfinite(p).all())
            if policy_valid:
                self.mon.note("policy", conf.tolist(), (pred == tgt).float().tolist())
                result["calibration_conf"] = float(conf.mean())
                result["policy_calibration_ok"] = float(self.mon.policy_ok())
                if self.fit_T:
                    self.T = self.T_decay * self.T + (1 - self.T_decay) * fit_temperature(lg, tgt)
            if self.ens is not None and T > 1:
                a = nb["actions"].float()[:, :-1, :self.A].reshape(-1, self.A)
                l0 = lat[:, :-1].reshape(-1, L)
                l1 = lat[:, 1:].reshape(-1, L)
                dis = self.ens.disagreement(l0, a)
                err = self.ens.error(l0, a, l1)
                if bool(torch.isfinite(dis).all() and torch.isfinite(err).all()):
                    self.mon.note("model", dis.tolist(), err.tolist())
                    result["model_calibration_ok"] = float(self.mon.model_ok())
        return result

    def _distill(self, policy, lat0, target, use) -> float:
        import itertools
        idx = torch.as_tensor([i for i, u in enumerate(use) if u],
                              device=self.device, dtype=torch.long)
        params = list(itertools.chain(
            policy.actor.parameters(),
            policy.conditioner.parameters() if policy.conditioner is not None else []))
        if self._opt is None:
            lr = float(policy.optimizer.param_groups[0].get("lr", 3e-4)) * self.dist_lr
            self._opt = torch.optim.Adam(params, lr=lr)
        tot = 0.0
        tgt = target.index_select(0, idx).detach()
        x = lat0.index_select(0, idx).detach()
        for _ in range(self.dist_steps):
            lg = F.log_softmax(self._logits(policy, x), -1)
            kl = (tgt * (torch.log(tgt + 1e-8) - lg)).sum(-1).mean()
            loss = self.dist_w * kl
            self._opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            self._opt.step()
            tot += float(kl.detach())
        return tot / self.dist_steps

    # -- telemetry / persistence -------------------------------------------------
    def report(self) -> str:
        l = self.last
        pe = self.mon.ece("policy")
        mc = self.mon.corr("model")
        f = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"   # noqa
        a = self.arbiter
        return (f"rows {int(l.get('rows', 0))} gain {l.get('plan_gain', 0.0):+.3f} "
                f"(pos {l.get('plan_gain_pos', 0.0):.2f}) agree {l.get('agree', 0.0):.2f} "
                f"conf {l.get('conf', 0.0):.2f} T {l.get('T', self.T):.2f} ece {f(pe)} "
                f"policy_ok {int(l.get('policy_calibration_ok', 0))} "
                f"model_ok {int(l.get('model_calibration_ok', 0))} "
                f"model_corr {f(mc)} plan {a.why['plan']}/{a.seen} "
                f"({', '.join(f'{k} {v}' for k, v in a.why.items() if k != 'plan')}) "
                f"{l.get('secs', 0.0):.2f}s")

    def state_dict(self) -> dict:
        return {"T": self.T,
                "ens": self.ens.state_dict() if self.ens is not None else None,
                "ens_scale": self.ens.err_scale if self.ens is not None else None}

    def load_state_dict(self, d: dict) -> None:
        self.T = float(d.get("T", 1.0))
        if self.ens is not None and d.get("ens") is not None:
            try:
                # GOV-015 F-03: check every key and shape BEFORE any tensor is
                # copied. torch copies the matching tensors and then raises on
                # the rest, so a saved ensemble of another width left the new
                # one half-overwritten while the log said "starts fresh".
                have = self.ens.state_dict()
                new = d["ens"]
                if (set(have) != set(new) or any(
                        tuple(have[k].shape) != tuple(new[k].shape) for k in have)):
                    raise ValueError("shape mismatch")
                self.ens.load_state_dict(d["ens"])
                self.ens.err_scale = float(d.get("ens_scale") or 1.0)
            except Exception:
                self._warn("load", "ensemble state has another shape: starts fresh")
