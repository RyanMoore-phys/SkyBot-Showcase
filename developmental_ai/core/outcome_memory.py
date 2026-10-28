"""WP-18 (GOV-009, GOV-011): queryable episodic memory. BUILT, UNTESTED.

A keyed store of (view embedding, action taken, short-horizon OUTCOME
SUMMARY) and a retrieval that hands the policy a small vector: "what
happened after the views most like this one". Switch `outcome_memory.enabled`
(default false); off, nothing here is imported and the policy input width is
unchanged.

KEY. The agent's own frozen view embedding (`DevelopmentalAI._view_key`, the
same one episodic novelty reads), L2-normalised: the view the agent was in
when it chose the action.

OUTCOME SUMMARY (LEARNED quantities only, never engine events) over the
`horizon` steps that follow the action, three numbers each squashed to [0, 1)
by x / (x + running mean of x), so no game-specific scale is declared:
  0  view change   1 - cos(view after, view before)  (latent change)
  1  surprise      mean per-step unclamped KL(posterior || prior)
  2  drive         mean intrinsic reward the stream was paid (the learning
                   signal the agent already receives; zero floor)
An outcome never contains reward from the game, a counter, a flag or a name.

STORE. Preallocated ring per stream (or one shared ring, `share_streams`),
normalised keys in one tensor, so the nearest-neighbour index IS the store
(one matmul per query, no rebuild). Not persisted across relaunches (a
decision for the test phase). Entries spanning a life boundary are never
written; the memory itself survives lives (that is its purpose).

RETRIEVAL VECTOR (width = retrieval_dim(action_dim) = 8 + action_dim): the
similarity-weighted mean outcome of the k nearest keys (3), the nearest key's
outcome (3), mean similarity of the k (1), the fraction of the k above
`min_sim` (1), and the nearest key's action one-hot (action_dim). Neighbours
younger than `min_age` writes are excluded (the agent's own last few frames
are not "a similar past situation"); with no neighbour above `min_sim` the
vector is zeros (neutral, like an absent knowledge vector).

WIRING. The vector is appended to the knowledge vector (the policy's
`knowledge_dim` grows by `retrieval_dim`), so it enters the policy in the
position the knowledge vector occupies, behind the same learned gate. Like
the knowledge vector it is computed for stream 0 only; the other streams act
with zeros there.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

D_OUT = 3


def retrieval_dim(action_dim: int) -> int:
    """Width of the retrieval vector (2 outcome vectors + 2 scalars + action)."""
    return 2 * D_OUT + 2 + max(0, int(action_dim))


class _Ring:
    """Preallocated key / action / outcome ring; the keys tensor is the index."""

    def __init__(self, capacity: int, dim: int, device):
        self.cap = int(capacity)
        self.keys = torch.zeros(self.cap, dim, device=device)
        self.acts = torch.full((self.cap,), -1, dtype=torch.long, device=device)
        self.outs = torch.zeros(self.cap, D_OUT, device=device)
        self.n = 0
        self.ptr = 0
        self.written = 0          # lifetime writes (age of a row = written - its write index)
        self.wid = torch.zeros(self.cap, dtype=torch.long, device=device)

    def add(self, key: torch.Tensor, act: int, out: torch.Tensor) -> None:
        i = self.ptr
        self.keys[i] = key
        self.acts[i] = int(act)
        self.outs[i] = out
        self.wid[i] = self.written
        self.written += 1
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.cap, self.n + 1)


class OutcomeMemory:
    def __init__(self, cfg: Dict[str, Any], action_dim: int, discrete: bool, device):
        c = dict(cfg or {})
        self.horizon = max(1, int(c.get("horizon", 4)))
        self.capacity = int(c.get("capacity", 4000))
        self.k = max(1, int(c.get("k", 8)))
        self.min_age = int(c.get("min_age", 32))
        self.min_sim = float(c.get("min_sim", 0.5))
        self.temp = max(1e-3, float(c.get("temperature", 0.1)))
        self.share = bool(c.get("share_streams", False))
        self.query_streams = set(int(s) for s in (c.get("query_streams") or [0]))
        self.decay = float(c.get("scale_decay", 0.999))
        self.A = int(action_dim) if discrete else 0
        self.dim = retrieval_dim(self.A)
        self.device = device
        self._rings: Dict[int, _Ring] = {}
        self._ring: Dict[int, Deque[Tuple[torch.Tensor, int, float, float]]] = {}
        self._prev: Dict[int, Optional[torch.Tensor]] = {}
        self._scale = [1e-3, 1e-3, 1e-3]
        self.vec: Dict[int, np.ndarray] = {}
        self.queries = 0
        self.hits = 0
        self.writes = 0

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _norm(key: torch.Tensor) -> torch.Tensor:
        e = key.detach().reshape(-1).float()
        return e / (e.norm() + 1e-6)

    def _squash(self, j: int, x: float) -> float:
        x = max(0.0, float(x))
        self._scale[j] = self.decay * self._scale[j] + (1 - self.decay) * x
        return x / (x + self._scale[j] + 1e-8)

    def _ring_for(self, stream: int, dim: int) -> _Ring:
        sid = 0 if self.share else int(stream)
        r = self._rings.get(sid)
        if r is None:
            r = self._rings[sid] = _Ring(self.capacity, dim, self.device)
        return r

    # ---- one step, one stream ----------------------------------------------
    @torch.no_grad()
    def observe(self, stream: int, key_after: Optional[torch.Tensor],
                action: Optional[int], surprise: float, drive: float,
                ended: bool) -> None:
        """Record the step that moved `stream` from its previous view to
        `key_after` by `action`; write the entry whose horizon has elapsed;
        refresh the retrieval vector for the view the agent is now in."""
        s = int(stream)
        ring = self._ring.setdefault(s, deque(maxlen=self.horizon))
        # GOV-015 F-05: a non-finite view key, surprise or drive is not
        # evidence. One infinite surprise froze the outcome scale at inf and a
        # NaN key poisoned every retrieval vector. Such a step is treated as an
        # unobserved one (the ring restarts, nothing is written).
        if key_after is not None:
            try:
                bad = (not bool(torch.isfinite(key_after.detach()).all())
                       or not np.isfinite(float(surprise))
                       or not np.isfinite(float(drive)))
            except Exception:
                bad = True
            if bad:
                self.refused = getattr(self, "refused", 0) + 1
                key_after = None
        if key_after is None:
            ring.clear()
            self._prev[s] = None
            return
        e = self._norm(key_after)
        prev = self._prev.get(s)
        if prev is not None and action is not None and int(action) >= 0:
            ring.append((prev, int(action), float(surprise), float(drive)))
            if len(ring) == self.horizon:
                vb, a0 = ring[0][0], ring[0][1]
                change = float((1.0 - torch.dot(vb, e)).clamp(0.0, 1.0))
                sm = sum(r[2] for r in ring) / len(ring)
                dm = sum(r[3] for r in ring) / len(ring)
                out = torch.tensor([self._squash(0, change), self._squash(1, sm),
                                    self._squash(2, dm)], device=self.device)
                self._ring_for(s, e.numel()).add(vb, a0, out)
                self.writes += 1
        if ended:
            ring.clear()
            self._prev[s] = None
            self.vec.pop(s, None)
            return
        self._prev[s] = e
        if s in self.query_streams:
            self.vec[s] = self._retrieve(s, e)

    # ---- query -----------------------------------------------------------------
    @torch.no_grad()
    def _retrieve(self, stream: int, e: torch.Tensor) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        r = self._rings.get(0 if self.share else int(stream))
        if r is None or r.n == 0:
            return out
        self.queries += 1
        sims = r.keys[:r.n] @ e
        old = (r.written - r.wid[:r.n]) > self.min_age
        sims = torch.where(old, sims, torch.full_like(sims, -2.0))
        k = min(self.k, r.n)
        top, idx = torch.topk(sims, k)
        ok = top >= self.min_sim
        if not bool(ok.any()):
            return out
        self.hits += 1
        w = torch.softmax(top / self.temp, 0) * ok.float()
        w = w / (w.sum() + 1e-8)
        mean_out = (w[:, None] * r.outs[idx]).sum(0)
        near = int(idx[0])
        out[0:3] = mean_out.cpu().numpy()
        out[3:6] = r.outs[near].cpu().numpy()
        out[6] = float(top[ok].mean())
        out[7] = float(ok.float().mean())
        if self.A > 0:
            a = int(r.acts[near])
            if 0 <= a < self.A:
                out[8 + a] = 1.0
        return out

    def feature(self, stream: int = 0) -> np.ndarray:
        """The cached retrieval vector for the view `stream` is in now."""
        v = self.vec.get(int(stream))
        return v if v is not None else np.zeros(self.dim, dtype=np.float32)

    def report(self) -> str:
        n = sum(r.n for r in self._rings.values())
        return (f"entries {n} writes {self.writes} queries {self.queries} "
                f"hit_rate {self.hits / max(1, self.queries):.2f}")
