"""Risk estimation as a STATE INPUT (WP-11, GOV-009; GOV-011). BUILT, UNTESTED.

The world model already trains a CONTINUE head (a learned survival prediction
from the latent). Its complement, 1 - sigmoid(logit), is a per-step risk of
termination: a quantity learned from the game's own terminal event (what a
player sees), never from engine counters. This module turns it into a small
feature vector for the policy. It is an INPUT, never a reward and never a
penalty: nothing here pays or charges anything (GOV-009 WP-11 item 4).

Feature (width `RiskEstimator.dim`, stream 0 only, appended to the knowledge
vector so the policy reads it through the knowledge gate it already has):
  [ squash(continue risk) ]                      width 1
  [ ... , optional short-horizon risk head prob ]  width 2 when head.enabled
squash(r) = r / (r + running mean r), so no absolute scale is declared and the
feature stays usable when deaths are rare. `feature_scale: raw` passes r as is.

Optional short-horizon head: a small MLP on the (detached) latent trained with
binary cross-entropy on "a termination happens within `horizon` steps". Labels
come from the stream's own history: when a termination arrives, the last
`horizon` stored states of that stream are positives; a state that survived
`horizon` further steps is a negative; a restart that is not a termination
discards the open states (censored, never labelled). The head has its own Adam.

Calibration report (no engine value): base rate, rank discrimination (AUC) of
the raw continue risk and of the head against the horizon label, expected
calibration error and Brier score of the head, and the mean raw risk by lead
time before a termination (the "risk rises before deaths" check).

State is plain Python / torch; the head is saved beside the skill bank and a
latent-width mismatch starts it fresh. Nothing is paid; a failure idles the
helper (logged once by the caller) and the feature stays zeros.
"""
from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def risk_dim(cfg: dict) -> int:
    """Width of the risk feature for a `risk` config block (1 or 2)."""
    return 1 + int(bool((cfg.get("head", {}) or {}).get("enabled", False)))


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney) with ties averaged; nan when one class is absent."""
    pos = label > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(e)


class RiskEstimator:
    def __init__(self, cfg: dict, streams: int, device, path: Optional[str] = None):
        cfg = dict(cfg or {})
        self.streams = max(1, int(streams))
        self.device = device
        self.path = path
        self.H = max(1, int(cfg.get("horizon", 16)))
        self.scale = str(cfg.get("feature_scale", "squash"))
        self.cap = int(cfg.get("capacity", 20000))
        self.report_every = int(cfg.get("report_every", 1024))
        hc = dict(cfg.get("head", {}) or {})
        self.head_on = bool(hc.get("enabled", False))
        self.hidden = int(hc.get("hidden", 64))
        self.lr = float(hc.get("lr", 1e-3))
        self.train_every = int(hc.get("train_every", 128))
        self.steps = int(hc.get("steps", 2))
        self.batch = int(hc.get("batch", 128))
        self.min_pairs = int(hc.get("min_pairs", 256))
        self.dim = 1 + int(self.head_on)
        # per-stream open states: (raw risk, latent half or None)
        self._open = [deque() for _ in range(self.streams)]
        # labelled store (ring): raw risk, label, lead (steps before death, -1)
        self._r = np.zeros(self.cap, dtype=np.float32)
        self._y = np.zeros(self.cap, dtype=np.float32)
        self._lead = np.full(self.cap, -1, dtype=np.int32)
        self._lat: Optional[torch.Tensor] = None      # [cap, D] half, head only
        self._np = 0
        self._mean_r = 0.0
        self._mean_n = 0
        self.step_i = 0
        self.vec = np.zeros(self.dim, dtype=np.float32)
        self.head: Optional[nn.Module] = None
        self.opt = None
        self.last = {}
        self._d = None
        self._deaths = 0

    # ---- head --------------------------------------------------------
    def _build(self, d: int):
        self._d = d
        if self.head_on:
            self.head = nn.Sequential(
                nn.Linear(d, self.hidden), nn.LayerNorm(self.hidden), nn.ELU(),
                nn.Linear(self.hidden, 1)).to(self.device)
            self.opt = torch.optim.Adam(self.head.parameters(), lr=self.lr)
            self._lat = torch.zeros(self.cap, d, dtype=torch.float16)
            self._load()

    def _store(self, r: float, y: float, lead: int, lat: Optional[torch.Tensor]):
        j = self._np % self.cap
        self._r[j], self._y[j], self._lead[j] = r, y, lead
        if self._lat is not None and lat is not None:
            self._lat[j] = lat
        self._np += 1

    def _squash(self, r: float) -> float:
        if self.scale == "raw":
            return float(r)
        return float(r / (r + self._mean_r + 1e-8))

    # ---- per-step ----------------------------------------------------
    @torch.no_grad()
    def observe(self, latent: torch.Tensor, cont_head, died, ended) -> None:
        """latent[N,D] post-observe; cont_head: the world model's continue
        predictor (logits); died[N]: a TERMINAL event this step; ended[N]:
        terminal or restart (life boundary)."""
        z = latent.detach().float()
        if self._d is None:
            self._build(int(z.shape[-1]))
        self.step_i += 1
        risk = (1.0 - torch.sigmoid(cont_head(z).float())).reshape(-1)
        rl = risk.cpu().numpy()
        for i in range(min(self.streams, z.shape[0])):
            q = self._open[i]
            if bool(died[i]):
                self._deaths += 1
                n = len(q)
                for k, (r, lat) in enumerate(q):
                    self._store(r, 1.0, n - k, lat)
                q.clear()
                continue
            if bool(ended[i]):
                q.clear()                              # censored, never labelled
                continue
            r = float(rl[i])
            self._mean_n += 1
            self._mean_r += (r - self._mean_r) / min(self._mean_n, 5000)
            q.append((r, z[i].half().cpu() if self.head_on else None))
            if len(q) > self.H:
                r0, lat0 = q.popleft()                 # survived H more steps
                self._store(r0, 0.0, -1, lat0)
        # feature for stream 0 (zeros right after a boundary is fine: neutral)
        f = [self._squash(float(rl[0]))]
        if self.head_on and self.head is not None:
            self.head.eval()
            f.append(float(torch.sigmoid(self.head(z[:1]).float())[0, 0]))
        self.vec = np.asarray(f, dtype=np.float32)
        if (self.head_on and self.step_i % self.train_every == 0
                and min(self._np, self.cap) >= self.min_pairs):
            self._train()

    def feature(self) -> np.ndarray:
        return self.vec

    @torch.enable_grad()
    def _train(self) -> None:
        self.head.train()
        n = min(self._np, self.cap)
        tot = 0.0
        for _ in range(self.steps):
            idx = torch.randint(0, n, (self.batch,))
            x = self._lat[idx].float().to(self.device)
            y = torch.as_tensor(self._y[idx.numpy()], device=self.device)
            loss = F.binary_cross_entropy_with_logits(
                self.head(x).reshape(-1), y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.opt.step()
            tot += float(loss.detach())
        self.last["head_loss"] = tot / max(1, self.steps)

    # ---- calibration report -----------------------------------------
    @torch.no_grad()
    def calibration(self) -> dict:
        n = min(self._np, self.cap)
        out = {"n": n, "deaths": self._deaths}
        if n == 0:
            return out
        r, y, lead = self._r[:n], self._y[:n], self._lead[:n]
        out["base_rate"] = float(y.mean())
        out["auc_raw"] = _auc(r, y)
        pos = y > 0.5
        if pos.any() and (~pos).any():
            out["risk_pos"] = float(r[pos].mean())
            out["risk_neg"] = float(r[~pos].mean())
        # mean raw risk by lead time before a termination (4 buckets over H)
        b = max(1, self.H // 4)
        for k in range(4):
            m = (lead > k * b) & (lead <= (k + 1) * b)
            if m.any():
                out[f"lead_{k * b + 1}-{(k + 1) * b}"] = float(r[m].mean())
        if self.head is not None and self._lat is not None:
            self.head.eval()
            p = torch.sigmoid(self.head(
                self._lat[:n].float().to(self.device)).reshape(-1)
                .float()).cpu().numpy()
            out["auc_head"] = _auc(p, y)
            out["ece_head"] = _ece(p, y)
            out["brier_head"] = float(((p - y) ** 2).mean())
        return out

    def report(self) -> str:
        c = self.calibration()
        keys = ("n", "deaths", "base_rate", "auc_raw", "risk_pos", "risk_neg",
                "auc_head", "ece_head", "brier_head")
        s = " ".join(f"{k}={c[k]:.3f}" if isinstance(c.get(k), float)
                     else f"{k}={c[k]}" for k in keys if k in c)
        lead = " ".join(f"{k}={v:.3f}" for k, v in c.items()
                        if k.startswith("lead_"))
        return s + (" | " + lead if lead else "")

    # ---- persistence ------------------------------------------------
    def save(self) -> None:
        if self.head is None or not self.path:
            return
        torch.save({"d": self._d, "hidden": self.hidden,
                    "head": self.head.state_dict()}, self.path)

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            s = torch.load(self.path, map_location=self.device)
            if int(s.get("d", -1)) == self._d and int(s.get("hidden", -1)) == self.hidden:
                self.head.load_state_dict(s["head"])
        except Exception:
            pass                                   # start fresh
