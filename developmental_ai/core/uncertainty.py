"""WP-4 (GOV-009, GOV-011): uncertainty and calibration. BUILT, UNTESTED.

Three small, separately testable parts, used by `core/planning.py`:

* `EnsembleHeads`: a few CHEAP heads (two-layer MLPs, not whole world models)
  predicting the next latent from (latent, action). Their DISAGREEMENT (mean
  over dimensions of the variance across heads) is the model's epistemic
  uncertainty about a transition. Trained on replayed real transitions, each
  head on its own bootstrap mask, so they differ. Switch `planning.ensemble`.
* `CalibrationMonitor` + `ece` + `spearman`: does the confidence the system
  reports match what happens? Two channels are used by the planner: the
  policy's confidence against whether its most-likely action matches the
  action actually taken on recent replay-excluded windows (expected
  calibration error), and the ensemble's disagreement against the real
  one-step latent error (rank correlation). A channel is "ok" only once it
  has `min_n` pairs AND passes its bar; until then the arbitration falls back
  to the habit policy (the guard re-opens as pairs accrue; nothing latches).
  Policy agreement measures action predictability, NOT action usefulness.
  Overlapping windows are an online diagnostic, not independent evidence.
* `fit_temperature`: temperature scaling of the policy's logits by minimising
  the negative log-likelihood of the observed actions (grid then refinement).

Nothing here reads an engine value, pays a reward, or names a game.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleHeads(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, n_heads: int = 4,
                 hidden: int = 128, lr: float = 1e-3, keep: float = 0.8):
        super().__init__()
        self.latent_dim, self.action_dim = int(latent_dim), int(action_dim)
        self.n_heads, self.keep = int(n_heads), float(keep)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, latent_dim))
            for _ in range(self.n_heads)])
        self.opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        self.err_scale = 1.0          # running mean squared target delta (for scale-free use)
        self.refused = 0              # rows refused as non-finite evidence (GOV-015 F-01)

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(heads, N, latent_dim) predicted NEXT latents."""
        x = torch.cat([latent, action], -1)
        return torch.stack([latent + h(x) for h in self.heads], 0)

    @torch.no_grad()
    def disagreement(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(N,) mean over dimensions of the variance across heads, in units of
        the running scale of real latent change (scale-free)."""
        p = self.predict(latent, action)
        return p.var(0, unbiased=False).mean(-1) / (self.err_scale + 1e-8)

    def train_step(self, lat: torch.Tensor, act: torch.Tensor,
                   nxt: torch.Tensor) -> float:
        """One step on real transitions (all detached). Returns mean loss."""
        lat, act, nxt = lat.detach(), act.detach(), nxt.detach()
        # GOV-015 F-01: a NaN or infinite row is not evidence (ASS-012 CT-03).
        # Refuse those rows; a finite batch takes the unchanged path below.
        ok = (torch.isfinite(lat).all(-1) & torch.isfinite(act).all(-1)
              & torch.isfinite(nxt).all(-1))
        if not bool(ok.all()):
            self.refused += int((~ok).sum())
            if not bool(ok.any()):
                return 0.0
            lat, act, nxt = lat[ok], act[ok], nxt[ok]
        delta = nxt - lat
        self.err_scale = 0.99 * self.err_scale + 0.01 * float(delta.pow(2).mean())
        x = torch.cat([lat, act], -1)
        loss = 0.0
        for h in self.heads:
            m = (torch.rand(lat.shape[0], device=lat.device) < self.keep).float()
            e = (h(x) - delta).pow(2).mean(-1)
            loss = loss + (e * m).sum() / m.sum().clamp_min(1.0)
        loss = loss / self.n_heads
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def error(self, lat: torch.Tensor, act: torch.Tensor, nxt: torch.Tensor
              ) -> torch.Tensor:
        """(N,) squared error of the ensemble MEAN against the real next latent."""
        return (self.predict(lat, act).mean(0) - nxt).pow(2).mean(-1)


def ece(conf: Sequence[float], correct: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error of confidences in [0,1] against 0/1 outcomes."""
    n = len(conf)
    if n == 0:
        return float("nan")
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n)
               if (conf[i] >= lo and (conf[i] < hi or (b == bins - 1 and conf[i] <= hi)))]
        if not idx:
            continue
        c = sum(conf[i] for i in idx) / len(idx)
        a = sum(correct[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(c - a)
    return tot


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation with average ranks for ties; undefined -> NaN."""
    if len(x) != len(y):
        raise ValueError("correlation requires equally sized paired observations")
    n = len(x)
    if n < 3:
        return float("nan")
    tx = torch.as_tensor(list(x), dtype=torch.float64)
    ty = torch.as_tensor(list(y), dtype=torch.float64)
    if not (torch.isfinite(tx).all() and torch.isfinite(ty).all()):
        return float("nan")

    def ranks(t):
        order = t.argsort()
        _, inverse, counts = torch.unique_consecutive(
            t[order], return_inverse=True, return_counts=True)
        ends = counts.cumsum(0).double()
        averages = ends - (counts.double() + 1.0) / 2.0
        result = torch.empty_like(t)
        result[order] = averages[inverse]
        return result

    rx, ry = ranks(tx), ranks(ty)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = (rx.norm() * ry.norm()).item()
    return float((rx * ry).sum().item() / d) if d > 0 else float("nan")


class CalibrationMonitor:
    """Rolling windows of (confidence, outcome) pairs per named channel."""

    def __init__(self, window: int = 2000, min_n: int = 200,
                 max_ece: float = 0.10, min_corr: float = 0.20):
        self.window, self.min_n = int(window), int(min_n)
        self.max_ece, self.min_corr = float(max_ece), float(min_corr)
        self._d: Dict[str, Deque[Tuple[float, float]]] = {}

    def note(self, name: str, a: Sequence[float], b: Sequence[float]) -> None:
        if len(a) != len(b):
            raise ValueError("calibration requires equally sized paired observations")
        q = self._d.setdefault(name, deque(maxlen=self.window))
        for x, y in zip(a, b):
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                q.append((float(x), float(y)))

    def n(self, name: str) -> int:
        return len(self._d.get(name, ()))

    def ece(self, name: str) -> Optional[float]:
        q = self._d.get(name)
        if not q or len(q) < self.min_n:
            return None
        return ece([p[0] for p in q], [p[1] for p in q])

    def corr(self, name: str) -> Optional[float]:
        q = self._d.get(name)
        if not q or len(q) < self.min_n:
            return None
        c = spearman([p[0] for p in q], [p[1] for p in q])
        return None if math.isnan(c) else c

    def policy_ok(self, name: str = "policy") -> bool:
        e = self.ece(name)
        return e is not None and e <= self.max_ece

    def model_ok(self, name: str = "model") -> bool:
        c = self.corr(name)
        return c is not None and c >= self.min_corr


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                    lo: float = 0.25, hi: float = 4.0) -> float:
    """T minimising NLL of softmax(logits / T) on `targets` (class indices).
    Grid in log space, then a finer grid around the best."""
    if logits.shape[0] < 2:
        return 1.0
    lg, tg = logits.detach().float(), targets.detach().long()

    def nll(t: float) -> float:
        return float(F.cross_entropy(lg / t, tg))

    grid = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / 24)
            for i in range(25)]
    best = min(grid, key=nll)
    a, b = best / 1.25, best * 1.25
    fine = [math.exp(math.log(a) + (math.log(b) - math.log(a)) * i / 14)
            for i in range(15)]
    return float(min(fine, key=nll))
