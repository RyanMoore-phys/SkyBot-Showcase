"""WP-12 (GOV-009, GOV-011; ADR-006 and its Amendments 1 and 2): peer
observation between streams. BUILT, UNTESTED. Switch `peer_observation.enabled`
(default false); off, nothing here is imported.

GEOMETRY. Streams are SEPARATE game instances. A stream watches another as one
watches another player's screen. This module is the only path between them.

WHAT A STREAM OBSERVES OF A PEER (and nothing else): the peer's recent VIEWS
(the frozen view embedding of the frame, a pure function of pixels) and the
ACTIONS it chose (button inputs), over a short window aligned to the
OBSERVER'S decision count, with the lag recorded. A peer is identified only by
an anonymous slot index, assigned by a private per-observer shuffle. Sound is
NOT carried (the loop has no audio path; D-33).

WHAT IT NEVER CARRIES (ADR-006 Amendment 2): reward, intrinsic drive,
counters, engine state, probation or success flags, or any label that an
action was good. Enforced by an ALLOW-LIST: `publish` accepts exactly the
fields in `CHANNEL_FIELDS` and `assert_channel_clean` (also callable from a
test) raises `LeakError` on any other field or any forbidden name.

THREE USES, all independent:
1. CONTEXT (`peer_observation.context.enabled`, default false): a short vector
   (action histogram of the peers' recent actions, mean cosine of the peers'
   latest views to the observer's, mean peer view-change rate) appended to the
   knowledge vector like WP-18's retrieval vector (stream 0's, zeros without
   peers). Width = `context_dim(action_dim)` = action_dim + 2.
2. AUXILIARY PREDICTION (`PeerPredictor`): predict a peer's NEXT ACTION from
   its current view and previous action, and its NEXT VIEW from its current
   view and action. Trained on the board's own history in the learning phase;
   it rewards nothing and (in this build) shapes no shared representation: it
   measures how predictable peers are, against a repeat-previous-action and a
   no-change baseline.
3. DIVERSITY (`view_diversity`): coverage of a random-hyperplane hash of the
   view embedding per stream, mean pairwise Jaccard overlap between streams,
   pooled cell entropy. Never from engine state.

PEER-DERIVED EFFECTS ARE HYPOTHESES, NOT LABELS: a peer's unusually large view
change right after an action is logged as (action, normalised change vector)
in `hypotheses()`; nothing consumes it, and any consumer must verify it
through its own probation (WP-1). No example is weighted by a peer's success.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakError(ValueError):
    """The peer channel would carry a forbidden field."""


CHANNEL_FIELDS = ("stamp", "view", "action")
_FORBIDDEN = ("reward", "return", "intrinsic", "drive", "counter", "count",
              "engine", "flag", "probation", "success", "score", "info",
              "evaluator", "label", "good", "health", "kill", "done",
              "episode", "life", "position", "coord", "inventory")


def assert_channel_clean(rec: Dict[str, Any],
                         allowed: Sequence[str] = CHANNEL_FIELDS) -> None:
    for k, v in rec.items():
        name = str(k).lower()
        if name not in allowed:
            raise LeakError(f"peer channel field {k!r} is not allowed "
                            f"(allowed: {list(allowed)})")
        if any(w in name for w in _FORBIDDEN):
            raise LeakError(f"peer channel field {k!r} names a forbidden thing")
        if isinstance(v, (dict, list, tuple, set)):
            raise LeakError(f"peer channel field {k!r} is a container")


def context_dim(action_dim: int) -> int:
    return max(0, int(action_dim)) + 2


# --------------------------------------------------------------------------
class PeerBoard:
    def __init__(self, cfg: Dict[str, Any], action_dim: int, discrete: bool,
                 device):
        c = dict(cfg or {})
        self.window = max(2, int(c.get("window", 8)))
        self.capacity = max(self.window + 4, int(c.get("capacity", 512)))
        self.A = int(action_dim) if discrete else 0
        self.device = device
        self._seed0 = int(c.get("seed", 0))
        self.rng = np.random.RandomState(self._seed0)
        self.refused = 0                 # views refused as non-finite (GOV-015 F-20)
        self.z_hyp = float(c.get("hypothesis_z", 3.0))
        dv = c.get("diversity", {}) or {}
        self.bits = max(4, min(20, int(dv.get("bits", 12))))
        self.cells_window = int(dv.get("window", 512))
        self._ring: Dict[int, Deque[Tuple[int, torch.Tensor, int, bool]]] = {}
        self._count: Dict[int, int] = {}
        self._brk: Dict[int, bool] = {}
        self._slots: Dict[int, List[int]] = {}      # observer -> private peer order
        self._proj: Optional[torch.Tensor] = None
        self._cells: Dict[int, Deque[int]] = {}
        self._chg_mean, self._chg_var = 0.0, 1e-4
        self.hyp: Deque[dict] = deque(maxlen=256)
        self.last_lag = 0.0

    # -- publish ----------------------------------------------------------
    @torch.no_grad()
    def publish(self, stream: int, view: Optional[torch.Tensor],
                action: Optional[int]) -> None:
        rec = {"stamp": self._count.get(int(stream), 0), "view": view,
               "action": action}
        assert_channel_clean(rec)
        s = int(stream)
        # GOV-015 F-21 / F-27: validate BEFORE anything is counted or stored, in
        # every state of the ring (the width check used to run only with a
        # non-empty ring and no break, and a refused view burned a stamp).
        e = None
        if view is not None:
            e = view.detach().reshape(-1).float()
            held = self._ring.get(s)
            if held and held[-1][1].numel() != e.numel():
                raise ValueError(
                    "peer view of width %d does not match this stream's "
                    "width %d" % (e.numel(), held[-1][1].numel()))
            if not bool(torch.isfinite(e).all()):
                # GOV-015 F-20: a NaN or infinite view is not evidence; it is
                # treated as an absent view (a break), never stored.
                self.refused += 1
                view = e = None
        n = self._count.get(s, 0) + 1
        self._count[s] = n
        if view is None:
            self._brk[s] = True
            return
        e = (e / (e.norm() + 1e-6)).to(self.device)
        ring = self._ring.setdefault(s, deque(maxlen=self.capacity))
        brk = self._brk.pop(s, False) or not ring
        a = -1 if action is None else int(action)
        if ring and not brk:
            prev = ring[-1][1]
            chg = float((1.0 - torch.dot(prev, e)).clamp(0.0, 1.0))
            sd = math.sqrt(self._chg_var) + 1e-6
            if chg > self._chg_mean + self.z_hyp * sd and ring[-1][2] >= 0:
                d = e - prev
                self.hyp.append({"action": int(ring[-1][2]),
                                 "effect": (d / (d.norm() + 1e-6)).cpu()})
            self._chg_mean = 0.99 * self._chg_mean + 0.01 * chg
            self._chg_var = 0.99 * self._chg_var + 0.01 * (chg - self._chg_mean) ** 2
        ring.append((n, e, a, brk))
        self._cell(s, e)

    def mark_break(self, stream: int) -> None:
        """Internal: a life boundary on `stream`; no training pair spans it.
        Not part of the channel."""
        self._brk[int(stream)] = True

    # -- observer view ------------------------------------------------------
    def _peers(self, observer: int) -> List[int]:
        o = int(observer)
        if o not in self._slots or len(self._slots[o]) != len(
                [s for s in self._ring if s != o]):
            others = [s for s in sorted(self._ring) if s != o]
            # GOV-015 F-22: the slot order is drawn from a PRIVATE generator
            # keyed on the observer and the peer count, so READING the context
            # never moves the board's own RNG (which picks the training pairs).
            np.random.RandomState((self._seed0 * 1000003 + o * 7919
                                   + len(others)) % (2 ** 31 - 1)
                                  ).shuffle(others)    # the anonymous slot order
            self._slots[o] = others
        return self._slots[o]

    def context(self, observer: int) -> np.ndarray:
        """The observer's peer-context vector (zeros when there is nothing)."""
        out = np.zeros(context_dim(self.A), dtype=np.float32)
        o = int(observer)
        d = self._count.get(o, 0)
        own = self._ring.get(o)
        mine = own[-1][1] if own else None
        hist = np.zeros(max(self.A, 1), dtype=np.float32)
        cosv, chgs, lags, na = [], [], [], 0
        for p in self._peers(o):
            ring = self._ring.get(p)
            if not ring:
                continue
            ent = [x for x in ring if d - self.window < x[0] <= d]
            if not ent:
                ent = [ring[-1]]                       # a lagging peer: its latest
            lags.append(d - ent[-1][0])
            for x in ent:
                if self.A and 0 <= x[2] < self.A:
                    hist[x[2]] += 1.0
                    na += 1
            if mine is not None:
                cosv.append(float(torch.dot(mine, ent[-1][1])))
            for x0, x1 in zip(ent[:-1], ent[1:]):
                if x1[0] == x0[0] + 1 and not x1[3]:
                    chgs.append(float((1.0 - torch.dot(x0[1], x1[1])).clamp(0.0, 1.0)))
        if lags:
            self.last_lag = float(sum(lags) / len(lags))
        if self.A:
            out[:self.A] = hist[:self.A] / max(1, na)
        out[self.A] = float(np.mean(cosv)) if cosv else 0.0
        out[self.A + 1] = float(np.mean(chgs)) if chgs else 0.0
        return out

    # -- training pairs for the auxiliary task --------------------------------
    def pairs(self, n: int):
        """(v_t, a_prev, a_t, v_next) from consecutive entries of random
        streams; None if not enough. Actions are indices."""
        keys = [s for s, r in self._ring.items() if len(r) >= 3]
        if not keys:
            return None
        vt, ap, at, vn = [], [], [], []
        for _ in range(int(n) * 3):
            if len(vt) >= n:
                break
            r = self._ring[keys[self.rng.randint(len(keys))]]
            i = self.rng.randint(1, len(r) - 1)
            p, c, nx = r[i - 1], r[i], r[i + 1]
            if c[3] or nx[3] or c[2] < 0 or p[2] < 0 or c[0] != p[0] + 1 \
                    or nx[0] != c[0] + 1:
                continue
            vt.append(c[1]); ap.append(p[2]); at.append(c[2]); vn.append(nx[1])
        if len(vt) < 4:
            return None
        dev = self.device
        return (torch.stack(vt), torch.as_tensor(ap, device=dev),
                torch.as_tensor(at, device=dev), torch.stack(vn))

    # -- diversity ------------------------------------------------------------
    def _cell(self, stream: int, e: torch.Tensor) -> None:
        if self._proj is None or self._proj.shape[0] != e.numel():
            g = torch.Generator().manual_seed(int(self.rng.randint(1 << 30)))
            self._proj = torch.randn(e.numel(), self.bits, generator=g).to(self.device)
        bits = (e @ self._proj > 0).long()
        cid = int((bits * (2 ** torch.arange(self.bits, device=bits.device))).sum())
        self._cells.setdefault(int(stream), deque(maxlen=self.cells_window)).append(cid)

    def diversity(self) -> Dict[str, float]:
        sets = {s: set(q) for s, q in self._cells.items() if q}
        return view_diversity(sets, {s: list(q) for s, q in self._cells.items()})

    def hypotheses(self) -> List[dict]:
        return list(self.hyp)


def view_diversity(cells: Dict[int, set], seqs: Optional[Dict[int, list]] = None
                   ) -> Dict[str, float]:
    """Coverage, pairwise overlap and pooled entropy of hashed view cells.
    `cells`: stream -> set of cell ids; `seqs`: stream -> visit list."""
    ks = sorted(cells)
    if not ks:
        return {"coverage": 0.0, "overlap": 0.0, "entropy": 0.0, "streams": 0}
    cov = float(np.mean([len(cells[k]) for k in ks]))
    ov = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = cells[ks[i]], cells[ks[j]]
            u = len(a | b)
            ov.append(len(a & b) / u if u else 0.0)
    pooled: Dict[int, int] = {}
    for k in ks:
        for c in (seqs or {}).get(k, list(cells[k])):
            pooled[c] = pooled.get(c, 0) + 1
    tot = sum(pooled.values())
    ent = 0.0
    if tot and len(pooled) > 1:
        ent = -sum(v / tot * math.log(v / tot) for v in pooled.values()) \
            / math.log(len(pooled))
    return {"coverage": cov, "overlap": float(np.mean(ov)) if ov else 0.0,
            "entropy": ent, "streams": len(ks)}


# --------------------------------------------------------------------------
class PeerPredictor(nn.Module):
    """Predict a peer's next action and next view. Rewards nothing."""

    def __init__(self, view_dim: int, action_dim: int, hidden: int = 128,
                 lr: float = 1e-3):
        super().__init__()
        self.D, self.A = int(view_dim), int(action_dim)
        self.act_head = nn.Sequential(nn.Linear(self.D + self.A, hidden), nn.SiLU(),
                                      nn.Linear(hidden, self.A))
        self.frame_head = nn.Sequential(nn.Linear(self.D + self.A, hidden), nn.SiLU(),
                                        nn.Linear(hidden, self.D))
        self.opt = torch.optim.Adam(self.parameters(), lr=float(lr))

    def step(self, vt, a_prev, a_t, vn, train: bool = True) -> Dict[str, float]:
        oh = lambda a: F.one_hot(a.clamp(0, self.A - 1), self.A).float()   # noqa
        la = self.act_head(torch.cat([vt, oh(a_prev)], -1))
        pv = F.normalize(vt + self.frame_head(torch.cat([vt, oh(a_t)], -1)), dim=-1)
        l_act = F.cross_entropy(la, a_t.clamp(0, self.A - 1))
        l_frm = (1.0 - (pv * vn).sum(-1)).mean()
        if train:
            self.opt.zero_grad()
            (l_act + l_frm).backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.opt.step()
        with torch.no_grad():
            return {"act_loss": float(l_act), "frame_loss": float(l_frm),
                    "act_acc": float((la.argmax(-1) == a_t).float().mean()),
                    "repeat_acc": float((a_prev == a_t).float().mean()),
                    "frame_cos": float((pv * vn).sum(-1).mean()),
                    "nochange_cos": float((vt * vn).sum(-1).mean())}
