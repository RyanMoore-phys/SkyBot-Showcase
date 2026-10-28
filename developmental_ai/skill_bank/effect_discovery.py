"""Game-agnostic skill discovery: trigger, effect keys, probation.

Built 2026-10-21 (GOV-011 entry WP-1; design RES-005, owner's choice
2026-10-18). BUILT, UNTESTED. Imported lazily; nothing runs unless
`goals.discovery.enabled` is true.

THE CHAIN
  1. TRIGGER (`SurpriseAgencyTrigger`): the world model is surprised
     (z-scored unclamped KL(post||prior) against the stream's own decaying
     baseline) AND the agent's own action explains the surprise (agency from
     `curiosity/action_surprise.py`, computed on spike rows only).
  2. EFFECT KEY (`EffectKeyClusterer`): `e = nk(view_{t+k}) - nk(view_{t-1})`,
     where nk is the frozen view-encoder embedding, normalised. Online leader
     clustering by cosine assigns `effect_N`. Each key keeps up to
     `exemplars` before/after frame pairs so its centroid can be re-embedded
     when the frozen encoder refreshes (a key keeps its meaning).
  3. The key goes to the goal broadcaster (`update_triggered`), which applies
     the EXISTING repeatability gate, minting and paging.
  4. PROBATION (`Probation`): a minted skill gets K invocations; success =
     its own key resolves while it runs; baseline = the rate that key
     resolves per step when NO option runs; kept only if a one-sided binomial
     test against that baseline gives p < alpha and >= 2 successes; else
     retired with a doubling back-off (distinct episodes) that re-opens by
     itself (CLAUDE.md 4.1: a retired key is never retired "until a human
     notices").

DISCOVERY PAYS NO REWARD. The trigger only creates goal candidates.

UNIVERSAL AND LEARNED (GOV-009 17.3): the inputs are the world model's
surprise, the executed action and the agent's own view embedding. No engine
counter, no game name, no named event. Which effects exist is whatever the
clustering finds; names are `effect_N`.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

EFFECT_KEY_RE = re.compile(r"^effect_\d+$")
OWN_KEY_RE = re.compile(r"^ach_\d+_(effect_\d+)$")


def is_effect_key_name(name) -> bool:
    return bool(EFFECT_KEY_RE.match(str(name or "")))


def own_effect_key(skill_id) -> Optional[str]:
    m = OWN_KEY_RE.match(str(skill_id or ""))
    return m.group(1) if m else None


def _nk(v: torch.Tensor) -> torch.Tensor:
    v = v.detach().reshape(-1).float().cpu()
    return v / (v.norm() + 1e-6)


def cue_for(centroid: torch.Tensor, dim: int) -> np.ndarray:
    """A recall cue of the width the goal store expects, from a centroid of
    any width: identity when equal, otherwise a fixed random projection
    (deterministic, so the same centroid always gets the same cue)."""
    c = centroid.detach().reshape(-1).float().cpu().numpy()
    if c.shape[0] != dim:
        g = np.random.RandomState(7).randn(int(dim), c.shape[0]).astype(
            np.float32) / math.sqrt(max(1, c.shape[0]))
        c = g @ c
    return (c / (np.linalg.norm(c) + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
class SurpriseAgencyTrigger:
    def __init__(self, z_thr: float = 3.0, s_floor: float = 1.0,
                 agency_frac: float = 0.25, horizon: int = 1000,
                 warmup: int = 200):
        self.z_thr, self.s_floor = float(z_thr), float(s_floor)
        self.agency_frac = float(agency_frac)
        self.alpha = 1.0 / max(2, int(horizon))
        self.warmup = int(warmup)
        self._mean: Dict[int, float] = {}
        self._var: Dict[int, float] = {}
        self._n: Dict[int, int] = defaultdict(int)
        self.stats = defaultdict(int)       # spikes, agency_pass, guarded, rows

    def step(self, surprise) -> List[dict]:
        """Fires for this step: [{stream, s, z, agency}]. A row inside a life
        boundary window neither fires nor moves its baseline."""
        n = int(surprise.s.shape[0])
        s_list = surprise.s.tolist()
        spikes = []
        for i in range(n):
            # guard[i] True = within k steps of a life boundary: skip the row
            # (StepSurprise.guard). This read True as eligible until 2026-10-22,
            # so the trigger fired on no mid-life spike (ASS-014 F-13, GOV-014 B1).
            if surprise.guard[i]:
                self.stats["guarded"] += 1
                continue
            s = float(s_list[i])
            if not math.isfinite(s):
                continue
            self.stats["rows"] += 1
            m = self._mean.get(i)
            if m is None:
                self._mean[i], self._var[i] = s, 1.0
                self._n[i] = 1
                continue
            sd = math.sqrt(max(self._var[i], 1e-8))
            z = (s - m) / sd
            if self._n[i] >= self.warmup and z >= self.z_thr \
                    and s >= self.s_floor:
                spikes.append((i, s, z))
            # baseline update with a winsorised sample: one freak step cannot
            # pin the bar (no latch), and the EMA decays it back anyway
            x = min(s, m + 4.0 * sd)
            d = x - m
            self._mean[i] = m + self.alpha * d
            self._var[i] = (1.0 - self.alpha) * (self._var[i]
                                                 + self.alpha * d * d)
            self._n[i] += 1
        fires: List[dict] = []
        if spikes:
            self.stats["spikes"] += len(spikes)
            ag = surprise.agency_for([r for r, _, _ in spikes])
            if ag is None:
                return fires
            for r, s, z in spikes:
                a = ag[r]
                if a > 0.0 and a >= self.agency_frac * s:
                    self.stats["agency_pass"] += 1
                    fires.append({"stream": r, "s": s, "z": z, "agency": a})
        return fires

    def state(self) -> dict:
        return {"mean": dict(self._mean), "var": dict(self._var),
                "n": dict(self._n)}

    def load(self, d: dict) -> None:
        self._mean = {int(k): float(v) for k, v in d.get("mean", {}).items()}
        self._var = {int(k): float(v) for k, v in d.get("var", {}).items()}
        self._n = defaultdict(int, {int(k): int(v)
                                    for k, v in d.get("n", {}).items()})


# ---------------------------------------------------------------------------
class EffectKeyClusterer:
    def __init__(self, tau: float = 0.8, max_keys: int = 48,
                 ema: float = 0.1, exemplars: int = 8,
                 min_change: float = 0.05, stale_after: int = 5000):
        self.tau, self.max_keys = float(tau), int(max_keys)
        self.ema, self.exemplars = float(ema), int(exemplars)
        self.min_change, self.stale_after = float(min_change), int(stale_after)
        self.keys: Dict[str, dict] = {}
        self._next = 0
        self._tick = 0
        self.stats = defaultdict(int)       # new, joined, nochange, overflow

    def _new_name(self) -> str:
        n = f"effect_{self._next}"
        self._next += 1
        return n

    def _evict(self, protected) -> bool:
        cands = [(k, v) for k, v in self.keys.items()
                 if k not in protected and (v["n"] < 2 or
                                            self._tick - v["last"]
                                            > self.stale_after)]
        if not cands:
            return False
        k = min(cands, key=lambda kv: (kv[1]["n"], kv[1]["last"]))[0]
        del self.keys[k]
        return True

    def assign(self, e: torch.Tensor, pair=None,
               protected=()) -> Optional[str]:
        """Key for effect vector `e` (difference of normalised embeddings),
        or None when the view did not change or the key table is full of
        keys worth keeping."""
        self._tick += 1
        e = e.detach().reshape(-1).float().cpu()
        mag = float(e.norm())
        if not math.isfinite(mag):
            # GOV-015 F-12: `nan < min_change` is False, so a NaN embedding
            # minted effect_0 with a NaN centroid. Not evidence: no key.
            self.stats["nonfinite"] = self.stats.get("nonfinite", 0) + 1
            return None
        if mag < self.min_change:
            self.stats["nochange"] += 1
            return None
        u = e / mag
        best, bc = None, -2.0
        for k, v in self.keys.items():
            c = float(u @ v["c"])
            if c > bc:
                best, bc = k, c
        if best is not None and bc >= self.tau:
            v = self.keys[best]
            c2 = (1.0 - self.ema) * v["c"] + self.ema * u
            v["c"] = c2 / (c2.norm() + 1e-8)
            v["n"] += 1
            v["last"] = self._tick
            self._remember(v, pair)
            self.stats["joined"] += 1
            return best
        if len(self.keys) >= self.max_keys and not self._evict(protected):
            self.stats["overflow"] += 1
            return None
        name = self._new_name()
        self.keys[name] = {"c": u.clone(), "n": 1, "last": self._tick,
                           "pairs": []}
        self._remember(self.keys[name], pair)
        self.stats["new"] += 1
        return name

    def _remember(self, v: dict, pair) -> None:
        if pair is None or len(v["pairs"]) >= self.exemplars:
            return
        v["pairs"].append((pair[0].half().cpu().clone(),
                           pair[1].half().cpu().clone()))

    def centroid(self, key: str) -> Optional[torch.Tensor]:
        v = self.keys.get(key)
        return None if v is None else v["c"]

    @torch.no_grad()
    def reembed(self, key_fn: Callable) -> int:
        """After the frozen encoder refreshed: recompute each key's centroid
        from its stored frame pairs. Returns keys re-embedded."""
        done = 0
        for v in self.keys.values():
            es = []
            for b, a in v["pairs"]:
                kb, ka = key_fn(b.float().reshape(1, -1)), key_fn(
                    a.float().reshape(1, -1))
                if kb is None or ka is None:
                    continue
                d = _nk(ka) - _nk(kb)
                if float(d.norm()) >= self.min_change:
                    es.append(d / d.norm())
            if es:
                c = torch.stack(es).mean(0)
                v["c"] = c / (c.norm() + 1e-8)
                done += 1
        return done

    def state(self) -> dict:
        return {"next": self._next, "tick": self._tick,
                "keys": {k: {"c": v["c"], "n": v["n"], "last": v["last"],
                             "pairs": v["pairs"]}
                         for k, v in self.keys.items()}}

    def load(self, d: dict) -> None:
        self._next = int(d.get("next", 0))
        self._tick = int(d.get("tick", 0))
        self.keys = {k: {"c": v["c"].float(), "n": int(v["n"]),
                         "last": int(v["last"]), "pairs": list(v["pairs"])}
                     for k, v in (d.get("keys") or {}).items()}


# ---------------------------------------------------------------------------
def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    p = min(max(p, 0.0), 1.0)
    return float(sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j)
                     for j in range(max(0, k), n + 1)))


class Probation:
    def __init__(self, K: int = 10, alpha: float = 0.05, min_succ: int = 2,
                 delay: int = 2, backoff0: int = 2, min_base_steps: int = 500):
        self.K, self.alpha, self.min_succ = int(K), float(alpha), int(min_succ)
        self.delay, self.backoff0 = int(delay), int(backoff0)
        self.min_base_steps = int(min_base_steps)
        self.base_steps = 0
        self.base_fires: Dict[str, int] = defaultdict(int)
        self._open: Dict[int, dict] = {}
        self._closed: List[dict] = []
        self._by_id: Dict[int, dict] = {}
        self._next_id = 0
        self._rec: Dict[str, dict] = {}
        self.retirements: Dict[str, int] = defaultdict(int)
        self.stats = defaultdict(int)       # passed, failed

    def observe(self, stream: int, sid: Optional[str], t: int) -> Optional[int]:
        """Call once per stream per step with the skill id running on the
        stream this step (or None under the base policy). Returns the
        invocation id to attribute a trigger to, or None."""
        key = own_effect_key(sid) if sid else None
        cur = self._open.get(stream)
        if cur is not None and (key is None or cur["sid"] != sid):
            cur["t_end"] = int(t)
            self._closed.append(cur)
            del self._open[stream]
            cur = None
        if key is not None and cur is None:
            cur = {"id": self._next_id, "sid": sid, "key": key, "steps": 0,
                   "hit": False, "t_end": None}
            self._next_id += 1
            self._open[stream] = cur
            self._by_id[cur["id"]] = cur
        if cur is not None:
            cur["steps"] += 1
            return cur["id"]
        if sid is None:
            self.base_steps += 1
        return None

    def note_fire(self, key: str, inv_id: Optional[int]) -> None:
        if inv_id is None:
            self.base_fires[key] += 1
            return
        inv = self._by_id.get(inv_id)
        if inv is not None and inv["key"] == key:
            inv["hit"] = True

    def drop(self, sid: str) -> None:
        """Forget a skill's record (retired skills restart on re-mint)."""
        self._rec.pop(sid, None)

    def verdicts(self, t: int) -> List[dict]:
        out: List[dict] = []
        still = []
        touched = set()
        for inv in self._closed:
            if t < inv["t_end"] + self.delay + 1:
                still.append(inv)
                continue
            r = self._rec.setdefault(inv["sid"], {
                "n": 0, "succ": 0, "durs": [], "done": False,
                "key": inv["key"]})
            r["n"] += 1
            r["succ"] += int(inv["hit"])
            r["durs"].append(int(inv["steps"]))
            self._by_id.pop(inv["id"], None)
            touched.add(inv["sid"])
        self._closed = still
        for sid in touched:
            r = self._rec[sid]
            if r["done"] or r["n"] < self.K:
                continue
            if self.base_steps < self.min_base_steps:
                continue            # no honest baseline yet: wait, do not guess
            r0 = (self.base_fires.get(r["key"], 0) + 0.5) / (
                self.base_steps + 1.0)
            pbar = float(np.mean([1.0 - (1.0 - r0) ** d for d in r["durs"]]))
            p = binom_tail(r["succ"], r["n"], pbar)
            keep = (r["succ"] >= self.min_succ) and (p < self.alpha)
            r["done"] = True
            if keep:
                self.stats["passed"] += 1
                out.append({"sid": sid, "key": r["key"], "verdict": "keep",
                            "succ": r["succ"], "n": r["n"], "p": p,
                            "base": pbar})
            else:
                self.stats["failed"] += 1
                self.retirements[r["key"]] += 1
                hold = self.backoff0 * 2 ** (self.retirements[r["key"]] - 1)
                out.append({"sid": sid, "key": r["key"], "verdict": "retire",
                            "succ": r["succ"], "n": r["n"], "p": p,
                            "base": pbar, "hold": int(hold)})
                self._rec.pop(sid, None)
        return out

    def state(self) -> dict:
        return {"base_steps": self.base_steps,
                "base_fires": dict(self.base_fires),
                "rec": {k: dict(v) for k, v in self._rec.items()},
                "retirements": dict(self.retirements),
                "stats": dict(self.stats)}

    def load(self, d: dict) -> None:
        self.base_steps = int(d.get("base_steps", 0))
        self.base_fires = defaultdict(int, {
            k: int(v) for k, v in d.get("base_fires", {}).items()})
        self._rec = {k: dict(v) for k, v in d.get("rec", {}).items()}
        self.retirements = defaultdict(int, {
            k: int(v) for k, v in d.get("retirements", {}).items()})
        for k, v in d.get("stats", {}).items():
            self.stats[k] = int(v)


# ---------------------------------------------------------------------------
class EffectDiscovery:
    """The facade the loop's one helper talks to."""

    def __init__(self, cfg: dict, path: Optional[str] = None,
                 cue_dim: int = 192):
        g = lambda k, d: cfg.get(k, d)           # noqa: E731
        self.delay = int(g("effect_delay", 2))
        self.cue_dim = int(cue_dim)
        self.trigger = SurpriseAgencyTrigger(
            z_thr=float(g("z_thr", 3.0)), s_floor=float(g("s_floor", 1.0)),
            agency_frac=float(g("agency_frac", 0.25)),
            horizon=int(g("baseline_horizon", 1000)),
            warmup=int(g("warmup", 200)))
        self.keys = EffectKeyClusterer(
            tau=float(g("tau", 0.8)), max_keys=int(g("max_keys", 48)),
            exemplars=int(g("exemplars", 8)),
            min_change=float(g("min_change", 0.05)))
        self.probation = Probation(
            K=int(g("probation_K", 10)), alpha=float(g("probation_alpha", 0.05)),
            min_succ=int(g("probation_min_successes", 2)), delay=self.delay,
            backoff0=int(g("backoff_start", 2)),
            min_base_steps=int(g("min_base_steps", 500)))
        self.path = path
        self._prev: Dict[int, Optional[torch.Tensor]] = {}
        self._pending: List[dict] = []
        self._stamp = None
        self._t0 = time.time()
        self._n_streams = 1
        self.stats = defaultdict(int)       # fires, resolved, candidates
        self.load()

    # -- one step ----------------------------------------------------------
    def step(self, surprise, keys: Sequence[Optional[torch.Tensor]],
             obs_t, next_obs_t, ended: Sequence[bool], t: int,
             running: Sequence[Optional[str]], protected=()) -> dict:
        """Process one step for all streams. Returns
        {'resolved': [(stream, key, cue)], 'verdicts': [...]}."""
        n = len(keys)
        self._n_streams = n
        inv_ids = [self.probation.observe(i, running[i] if i < len(running)
                                          else None, t) for i in range(n)]
        # boundaries cancel pending effects and invalidate the before-view
        for i in range(n):
            if ended[i]:
                self._prev[i] = None
        self._pending = [p for p in self._pending
                         if not (p["stream"] < len(ended) and ended[p["stream"]])]
        # new fires (this step), needing a valid before-view
        if surprise is not None:
            for f in self.trigger.step(surprise):
                i = f["stream"]
                self.stats["fires"] += 1
                before = self._prev.get(i)
                if before is None or ended[i]:
                    continue
                self._pending.append({
                    "stream": i, "due": t + self.delay, "before": before,
                    "frame": obs_t[i].detach().reshape(-1).half().cpu(),
                    "inv": inv_ids[i]})
        resolved = []
        keep = []
        prot = set(protected) | {k for k in self.probation.base_fires}
        for p in self._pending:
            if p["due"] > t:
                keep.append(p)
                continue
            i = p["stream"]
            k_after = keys[i] if i < n else None
            if k_after is None:
                continue
            e = _nk(k_after) - p["before"]
            pair = (p["frame"], next_obs_t[i].detach().reshape(-1).half().cpu())
            key = self.keys.assign(e, pair, protected=prot)
            if key is None:
                continue
            self.stats["resolved"] += 1
            self.probation.note_fire(key, p["inv"])
            resolved.append((i, key, cue_for(self.keys.centroid(key),
                                             self.cue_dim)))
        self._pending = keep
        for i in range(n):
            self._prev[i] = None if (ended[i] or keys[i] is None) \
                else _nk(keys[i])
        return {"resolved": resolved, "verdicts": self.probation.verdicts(t)}

    def refresh_if_needed(self, stamp, key_fn: Callable) -> None:
        """Re-embed centroids when the frozen view encoder was refreshed."""
        if self._stamp is None:
            self._stamp = stamp
            return
        if stamp != self._stamp:
            self._stamp = stamp
            self.stats["reembedded"] += self.keys.reembed(key_fn)

    def keys_seen(self) -> List[str]:
        return sorted(self.keys.keys)

    # -- telemetry ---------------------------------------------------------
    def report(self, reset: bool = True) -> str:
        dt = max(1e-6, time.time() - self._t0)
        tr = self.trigger.stats
        rate = self.stats["fires"] / (dt / 60.0) / max(1, self._n_streams)
        top = sorted(self.keys.keys.items(), key=lambda kv: -kv[1]["n"])[:3]
        pr = self.probation
        line = (f"fires {rate:.2f}/min/stream | spikes {tr['spikes']} -> "
                f"agency pass {tr['agency_pass']} (guarded rows "
                f"{tr['guarded']}) | keys {len(self.keys.keys)} "
                f"(new {self.keys.stats['new']}, joined "
                f"{self.keys.stats['joined']}, nochange "
                f"{self.keys.stats['nochange']}, overflow "
                f"{self.keys.stats['overflow']}) top "
                + ", ".join(f"{k}x{v['n']}" for k, v in top)
                + f" | resolved {self.stats['resolved']} | probation passed "
                f"{pr.stats['passed']} failed {pr.stats['failed']} (base "
                f"steps {pr.base_steps}) | retired keys "
                f"{ {k: v for k, v in pr.retirements.items()} }")
        if reset:
            self._t0 = time.time()
            self.stats["fires"] = 0
            self.stats["resolved"] = 0
            for k in list(tr.keys()):
                tr[k] = 0
            for k in list(self.keys.stats.keys()):
                self.keys.stats[k] = 0
        return line

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        try:
            tmp = self.path + ".tmp"
            torch.save({"trigger": self.trigger.state(),
                        "keys": self.keys.state(),
                        "probation": self.probation.state()}, tmp)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            d = torch.load(self.path, map_location="cpu")
            self.trigger.load(d.get("trigger", {}))
            self.keys.load(d.get("keys", {}))
            self.probation.load(d.get("probation", {}))
        except Exception:
            pass
