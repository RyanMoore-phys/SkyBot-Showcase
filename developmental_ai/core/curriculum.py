"""WP-3 (GOV-009, GOV-011): curriculum and learnability frontier. BUILT, UNTESTED.

A scheduler that assigns a CONTEXT (a map and/or a difficulty level the game
interface offers) to each stream and re-assigns it from a progress signal,
by a bandit. Switch `curriculum.enabled` (default false). Off, nothing here is
imported.

WHAT IS LEARNED / MEASURED. Per stream, once per segment, two quantities from
the agent's own world model on that stream's NEWEST replay window (which
training has never sampled, so the loss is generalisation, not memory):

  * fresh-data loss (reconstruction) -> learning progress = relative drop
    between two consecutive measurements taken inside the same context;
  * action attribution = how much worse the prior does when the window's
    actions are rolled by one step (KL(post || prior) rolled minus true):
    positive when the model's predictions depend on what the agent did.

The arm reward is a clipped weighted sum of the two progress terms, each
MINUS the mean over the streams measured in the same round (so the
improvement every context gets from the world model simply training is
removed and a context is rewarded only for beating the others). NO extrinsic
reward, no engine value, no game or map name enters.

THE BANDIT. Arms = contexts of a pool (streams whose adapter reports the same
`pool`). Value = exponential average of the arm reward; selection = softmax of
value + a UCB bonus, then (1) forced exploration: an arm with fewer than
`min_pulls` assignments is taken first; (2) an epsilon of uniform mix;
(3) an ASSIGNMENT-ENTROPY FLOOR: the probabilities are mixed with uniform just
enough that their entropy is >= `entropy_floor` x ln(arms); and if the REALISED
(recent) assignment entropy is below the floor the draw is uniform. Every
guard is a function of counts, so it re-opens by itself (CLAUDE.md 4.1): a
collapsed bandit cannot stay collapsed.

THE INTERFACE. An adapter may expose `curriculum_options()` (-> None, or
{"pool": str, "maps": [...], "skills": [...]}) and `set_context(ctx)` (applied
at its next episode start). The scheduler never touches an engine. Streams
whose adapter offers nothing are left alone. A stream in a held-out (target)
phase offers nothing.

NOT BUILT (D-21): expanding revisit schedules; a separate "targeted
fine-tuning" mode (the same bandit already revisits a context whose
progress rises); the measured-latency check of the probe (it is one small
no-grad pass per stream per segment, to be timed).
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _entropy(p: Sequence[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0.0)


def floor_entropy(p: Sequence[float], frac: float) -> List[float]:
    """Mix `p` with uniform just enough that H >= frac x ln(n). Always
    reachable (uniform has H = ln n)."""
    n = len(p)
    if n <= 1 or frac <= 0.0:
        return list(p)
    target = min(1.0, frac) * math.log(n)
    if _entropy(p) >= target:
        return list(p)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        q = [(1 - mid) * x + mid / n for x in p]
        if _entropy(q) >= target:
            hi = mid
        else:
            lo = mid
    return [(1 - hi) * x + hi / n for x in p]


class ContextBandit:
    """Bandit over a fixed list of contexts (arms)."""

    def __init__(self, n: int, cfg: Dict[str, Any], rng: random.Random):
        self.n = int(n)
        self.rng = rng
        self.decay = float(cfg.get("value_decay", 0.8))
        self.ucb_c = float(cfg.get("ucb_c", 0.1))
        self.temp = max(1e-3, float(cfg.get("temperature", 0.1)))
        self.eps = float(cfg.get("epsilon", 0.1))
        self.min_pulls = int(cfg.get("min_pulls", 2))
        self.h_floor = float(cfg.get("entropy_floor", 0.6))
        self.h_window = float(cfg.get("entropy_window", 0.9))   # EMA of assignment counts
        self.value = [0.0] * self.n
        self.pulls = [0] * self.n
        self.rewards_n = [0] * self.n
        self.recent = [1.0] * self.n          # decaying assignment counts
        self.total = 0

    def probs(self) -> List[float]:
        t = max(1, self.total)
        sc = [self.value[i] + self.ucb_c * math.sqrt(
            math.log(1 + t) / (1 + self.pulls[i])) for i in range(self.n)]
        m = max(sc)
        ex = [math.exp((s - m) / self.temp) for s in sc]
        z = sum(ex)
        p = [e / z for e in ex]
        p = [(1 - self.eps) * x + self.eps / self.n for x in p]
        return floor_entropy(p, self.h_floor)

    def realised_entropy(self) -> float:
        z = sum(self.recent)
        return _entropy([x / z for x in self.recent]) / math.log(self.n) \
            if self.n > 1 and z > 0 else 1.0

    def choose(self) -> Tuple[int, str]:
        """(arm, why) with why in forced | floor | draw."""
        need = [i for i in range(self.n) if self.pulls[i] < self.min_pulls]
        if need:
            arm, why = min(need, key=lambda i: (self.pulls[i], i)), "forced"
        elif self.n > 1 and self.realised_entropy() < self.h_floor:
            arm, why = self.rng.randrange(self.n), "floor"
        else:
            p, r, acc = self.probs(), self.rng.random(), 0.0
            arm = self.n - 1
            for i, x in enumerate(p):
                acc += x
                if r <= acc:
                    arm = i
                    break
            why = "draw"
        self.pulls[arm] += 1
        self.total += 1
        self.recent = [x * self.h_window for x in self.recent]
        self.recent[arm] += 1.0
        return arm, why

    def update(self, arm: int, reward: float) -> None:
        d = self.decay
        if self.rewards_n[arm] == 0:
            self.value[arm] = float(reward)
        else:
            self.value[arm] = d * self.value[arm] + (1 - d) * float(reward)
        self.rewards_n[arm] += 1

    def state(self) -> dict:
        return {"value": self.value, "pulls": self.pulls,
                "rewards_n": self.rewards_n, "recent": self.recent,
                "total": self.total}

    def load(self, s: dict) -> None:
        for k in ("value", "pulls", "rewards_n", "recent"):
            v = list(s.get(k, []))
            if len(v) == self.n:
                setattr(self, k, v)
        self.total = int(s.get("total", self.total))


class CurriculumScheduler:
    """Pools of streams, one bandit per pool, per-stream measurement memory."""

    def __init__(self, cfg: Dict[str, Any], path: Optional[str] = None):
        self.cfg = dict(cfg or {})
        self.path = path
        self.every = max(2, int(self.cfg.get("every_segments", 4)))
        self.w_wm = float(self.cfg.get("w_wm", 1.0))
        self.w_attr = float(self.cfg.get("w_attr", 0.5))
        self.clip = float(self.cfg.get("reward_clip", 1.0))
        self.rng = random.Random(int(self.cfg.get("seed", 0)))
        self.seg = 0
        self.pool_of: Dict[int, str] = {}
        self.contexts: Dict[str, List[dict]] = {}
        self.bandits: Dict[str, ContextBandit] = {}
        # per stream: arm now, last (loss, attr) measured in this arm
        self.cur: Dict[int, Optional[int]] = {}
        self.last: Dict[int, Optional[Tuple[float, float]]] = {}
        self.why_count = {"forced": 0, "floor": 0, "draw": 0}
        self.last_report = ""
        self.empty_steps = 0      # consecutive steps without any measurement (GOV-015 F-25)

    # ---- setup ---------------------------------------------------------
    @staticmethod
    def build_contexts(opt: Dict[str, Any]) -> List[dict]:
        maps = list(opt.get("maps") or [None])
        skills = list(opt.get("skills") or [None])
        out = []
        for m in maps:
            for s in skills:
                c = {}
                if m is not None:
                    c["map"] = m
                if s is not None:
                    c["skill"] = s
                out.append(c)
        return out

    def register(self, options: Dict[int, Optional[Dict[str, Any]]]) -> int:
        """options: stream -> adapter's curriculum_options() (or None).
        Returns the number of streams the scheduler manages."""
        pools: Dict[str, List[int]] = {}
        for i, o in options.items():
            if not o:
                continue
            ctx = self.build_contexts(o)
            if len(ctx) < 2:
                continue                      # nothing to choose between
            pid = str(o.get("pool", "0"))
            self.contexts.setdefault(pid, ctx)
            if self.contexts[pid] != ctx:     # pool members must agree
                continue
            self.pool_of[int(i)] = pid
            self.cur[int(i)] = None
            self.last[int(i)] = None
            pools.setdefault(pid, []).append(int(i))
        for pid, ctx in self.contexts.items():
            self.bandits[pid] = ContextBandit(len(ctx), self.cfg, self.rng)
        self._load()
        return len(self.pool_of)

    # ---- one segment -----------------------------------------------------
    def step(self, meas: Dict[int, Tuple[float, float]]
             ) -> Dict[int, dict]:
        """meas: stream -> (fresh loss, attribution) measured this segment
        (missing = unknown). Returns {stream: context} to apply NOW."""
        self.seg += 1
        self.empty_steps = 0 if meas else self.empty_steps + 1
        # 1. progress of the arm each stream is in, baselined across streams
        raw: Dict[int, float] = {}
        for i, (loss, attr) in meas.items():
            prev = self.last.get(i)
            if self.cur.get(i) is not None and prev is not None \
                    and math.isfinite(loss) and math.isfinite(attr):
                dl = (prev[0] - loss) / max(abs(prev[0]), 1e-6)
                da = (attr - prev[1]) / max(abs(prev[1]), 1e-3)
                raw[i] = (self.w_wm * max(-1.0, min(1.0, dl))
                          + self.w_attr * max(-1.0, min(1.0, da)))
            if math.isfinite(loss) and math.isfinite(attr):
                self.last[i] = (float(loss), float(attr))
        if raw:
            mean = sum(raw.values()) / len(raw) if len(raw) > 1 else 0.0
            for i, r in raw.items():
                r = max(-self.clip, min(self.clip, r - mean))
                self.bandits[self.pool_of[i]].update(self.cur[i], r)
        # 2. (re)assign, staggered by stream index
        out: Dict[int, dict] = {}
        for i, pid in self.pool_of.items():
            if self.cur[i] is None or (self.seg + i) % self.every == 0:
                arm, why = self.bandits[pid].choose()
                self.why_count[why] += 1
                if arm != self.cur[i]:
                    out[i] = dict(self.contexts[pid][arm])
                    self.last[i] = None      # a new context: new baseline
                self.cur[i] = arm
        self._save()
        self.last_report = self.report()
        return out

    def report(self) -> str:
        parts = []
        for pid, b in self.bandits.items():
            top = max(range(b.n), key=lambda i: b.value[i])
            parts.append(f"pool {pid}: arms {b.n} realised_H {b.realised_entropy():.2f} "
                         f"best {self.contexts[pid][top]} v={b.value[top]:+.3f}")
        return (f"streams {len(self.pool_of)} | " + " | ".join(parts)
                + f" | forced {self.why_count['forced']} floor "
                  f"{self.why_count['floor']} draw {self.why_count['draw']}")

    # ---- persistence -------------------------------------------------------
    def _sig(self) -> str:
        return json.dumps(self.contexts, sort_keys=True, default=str)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            blob = {"sig": self._sig(), "seg": self.seg,
                    "bandits": {p: b.state() for p, b in self.bandits.items()}}
            with open(self.path + ".tmp", "w") as f:
                json.dump(blob, f)
            os.replace(self.path + ".tmp", self.path)
        except Exception:
            pass

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                blob = json.load(f)
            if blob.get("sig") != self._sig():
                return                       # contexts changed: start fresh
            self.seg = int(blob.get("seg", 0))
            for p, s in (blob.get("bandits") or {}).items():
                if p in self.bandits:
                    self.bandits[p].load(s)
        except Exception:
            pass


def measure_streams(world_model, replay_buffer, seq_len: int, device
                    ) -> Dict[int, Tuple[float, float]]:
    """Per stream: (fresh-data loss, action attribution) on its newest
    (never-trained-on) window. Caller holds the world-model lock. Streams
    without a window are absent. Never raises."""
    import torch
    out: Dict[int, Tuple[float, float]] = {}
    streams = getattr(replay_buffer, "streams", None)
    if not streams:
        return out
    S, C = world_model.rssm.stochastic_size, world_model.rssm.stochastic_classes
    lsm = torch.nn.functional.log_softmax

    def zinfo(obs, act, pro):
        _, infos = world_model.observe_sequence(obs, act, pro)
        pri = lsm(infos["prior_logits"].reshape(-1, S, C).float(), -1)
        post = lsm(infos["posterior_logits"].reshape(-1, S, C).float(), -1)
        return float((post.exp() * (post - pri)).sum(-1).sum(-1).mean())

    for i, s in enumerate(streams):
        try:
            p = s.newest_sequence(int(seq_len))
            if p is None:
                continue
            t = [torch.from_numpy(p[k]).to(device) for k in range(4)]
            pro = torch.from_numpy(p[4]).to(device) if p[4] is not None else None
            with torch.no_grad():
                L = world_model.compute_loss(t[0], t[1], t[2], t[3],
                                             proprio=pro)
                k_true = zinfo(t[0], t[1], pro)
                k_roll = zinfo(t[0], torch.roll(t[1], 1, dims=1), pro)
            out[i] = (float(L["reconstruction"]), k_roll - k_true)
        except Exception:
            continue
    return out
