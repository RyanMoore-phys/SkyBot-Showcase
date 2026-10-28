"""WP-7 (GOV-009, GOV-011): attention and working memory. BUILT, UNTESTED.

Two separate switches, both default OFF; off, nothing here is imported and
the policy input width is unchanged.

1. `slots.enabled` (EXISTING key, read by the loop when it builds the world
   model): the world model's slot-attention head (SAViSlots, trained on the
   flow field). Nothing new is built for it here except the COST REPORT
   (`slot_cost`): parameters, milliseconds per forward, memory.

2. `attention_memory.enabled` (NEW): a short WINDOW of the last `window`
   post-observe latents per stream, read by a small GATED attention layer
   (`GatedWindowAttention`) that turns it into a `out_dim` vector appended to
   the knowledge vector (stream 0 only, like WP-18 and WP-12), so the policy
   reads it behind the knowledge gate it already has. The layer is trained by
   a self-supervised objective that pays nothing: predict the NEXT latent from
   the window; the layer's own scalar gate sees only that objective, so it
   opens exactly as far as the window helps that prediction. Reported with a
   copy-the-last-latent baseline.

NOT in the world model's state_dict. The layer reads the latent but lives
beside the skill bank (`attention_memory.pt`), so `world_model/growth.py`
neither sees nor transforms it. A stage-D grow widens the latent; the saved
layer records the latent width it was built for and, on mismatch, is started
FRESH (never guessed at). Recorded as D-38 in GOV-011.

COST REPORT. `CostReport` writes parameters, mean milliseconds per call (the
summary read and the training step) and memory bytes (parameters, window
ring, pair store) to `attention_cost.json` and one telemetry line.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _nparams(m: Optional[nn.Module]) -> int:
    return 0 if m is None else int(sum(p.numel() for p in m.parameters()))


class GatedWindowAttention(nn.Module):
    """Multi-head attention of the newest latent over a window of recent
    latents, with a learned scalar gate on the attended read.

    `forward(window[B,W,D], mask[B,W])` -> (summary[B,out], pred[B,D], gate[B,1]).
    The last valid row of the window is the query. Attention weights sum to
    one over valid rows; with one valid row the read is that row itself.
    """

    def __init__(self, latent_dim: int, d_model: int = 64, heads: int = 2,
                 out_dim: int = 32, gate_bias: float = -2.0):
        super().__init__()
        assert d_model % heads == 0
        self.latent_dim, self.d, self.h = int(latent_dim), int(d_model), int(heads)
        self.out_dim = int(out_dim)
        self.inp = nn.Linear(latent_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(2 * d_model, 1)
        nn.init.constant_(self.gate.bias, float(gate_bias))   # starts mostly closed
        self.mix = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, out_dim)
        self.pred = nn.Linear(d_model, latent_dim)
        self.last_weights: Optional[torch.Tensor] = None
        # RW-1 (GOV-012): softmax temperature set by the relation layer (gain
        # NE->ATT); 1.0 = the previous behaviour exactly.
        self.temperature = 1.0

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:])

    def forward(self, window: torch.Tensor, mask: torch.Tensor, bias=None):
        B, W, _ = window.shape
        x = self.inp(self._norm(window.float())) + self.pos
        idx = (mask.long() * torch.arange(1, W + 1, device=mask.device)).argmax(1)
        cur = x[torch.arange(B, device=x.device), idx]            # newest valid
        q = self.q(cur).view(B, self.h, 1, self.d // self.h)
        k = self.k(x).view(B, W, self.h, -1).transpose(1, 2)
        v = self.v(x).view(B, W, self.h, -1).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / ((self.d // self.h) ** 0.5)
        if self.temperature != 1.0:
            s = s / float(self.temperature)
        if bias is not None:          # RW-2 (GOV-012): top-down logit bias [B, W]
            s = s + bias[:, None, None, :].to(s.dtype)
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        w = torch.softmax(s, dim=-1)                              # sums to 1
        self.last_weights = w.detach()
        att = self.mix((w @ v).transpose(1, 2).reshape(B, self.d))
        g = torch.sigmoid(self.gate(torch.cat([cur, att], -1)))
        mem = cur + g * att
        return self.out(mem), self.pred(mem), g


class LatentWindow:
    """Ring of the last W latents per stream (device tensor), life-aware."""

    def __init__(self, streams: int, window: int, dim: int, device):
        self.W, self.dim = int(window), int(dim)
        self.buf = torch.zeros(streams, self.W, self.dim, device=device)
        self.n = [0] * streams

    def clear(self, i: int) -> None:
        self.n[i] = 0

    def push(self, i: int, z: torch.Tensor) -> None:
        self.buf[i] = torch.roll(self.buf[i], -1, 0)
        self.buf[i, -1] = z
        self.n[i] = min(self.W, self.n[i] + 1)

    def get(self, i: int):
        """(window[W,D] right-aligned, mask[W])."""
        m = torch.zeros(self.W, dtype=torch.bool, device=self.buf.device)
        if self.n[i]:
            m[-self.n[i]:] = True
        return self.buf[i], m


class CostReport:
    """Parameters, mean ms per named call, memory bytes; json + one line."""

    def __init__(self, path: Optional[str]):
        self.path, self.ms, self.n = path, {}, {}
        self.params: Dict[str, int] = {}
        self.mem: Dict[str, int] = {}

    def timed(self, name: str, fn, *a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) * 1000.0
        self.ms[name] = self.ms.get(name, 0.0) + dt
        self.n[name] = self.n.get(name, 0) + 1
        return r

    def data(self) -> Dict[str, Any]:
        return {"params": dict(self.params), "mem_bytes": dict(self.mem),
                "ms_per_call": {k: self.ms[k] / max(1, self.n[k]) for k in self.ms},
                "calls": dict(self.n)}

    def line(self) -> str:
        d = self.data()
        return " ".join(
            [f"params {sum(d['params'].values())}",
             f"mem {sum(d['mem_bytes'].values()) / 1e6:.2f}MB"]
            + [f"{k} {v:.2f}ms" for k, v in d["ms_per_call"].items()])

    def write(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path + ".tmp", "w") as f:
                json.dump(self.data(), f)
            os.replace(self.path + ".tmp", self.path)
        except Exception:
            pass


def slot_cost(world_model, report: CostReport) -> None:
    """Cost of the world model's EXISTING slot head: parameters and memory now,
    forward milliseconds via hooks (only attached when asked)."""
    sm = getattr(world_model, "slot_module", None)
    if sm is None:
        return
    report.params["slots"] = _nparams(sm)
    report.mem["slots"] = sum(p.numel() * p.element_size() for p in sm.parameters())
    if getattr(sm, "_cost_hooked", False):
        return
    sm._cost_hooked = True
    state = {}

    def pre(_m, _i):
        state["t"] = time.perf_counter()

    def post(_m, _i, _o):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - state.get("t", time.perf_counter())) * 1000.0
        report.ms["slots_forward"] = report.ms.get("slots_forward", 0.0) + dt
        report.n["slots_forward"] = report.n.get("slots_forward", 0) + 1

    sm.register_forward_pre_hook(pre)
    sm.register_forward_hook(post)


class AttentionMemory:
    """Per-stream latent windows, the gated layer, its self-supervised
    trainer and the cost report. Driven by one loop helper per step."""

    def __init__(self, cfg: Dict[str, Any], streams: int, device, path: str):
        c = dict(cfg or {})
        self.cfg, self.device, self.path = c, device, path
        self.streams = int(streams)
        self.W = int(c.get("window", 8))
        self.out_dim = int(c.get("out_dim", 32))
        self.train_every = int(c.get("train_every", 128))
        self.steps = int(c.get("steps", 2))
        self.batch = int(c.get("batch", 64))
        self.cap = int(c.get("capacity", 1024))
        self.store_every = int(c.get("store_every", 2))
        self.report_every = int(c.get("report_every", 1024))
        self.lr = float(c.get("lr", 1e-3))
        self.net: Optional[GatedWindowAttention] = None
        self.opt = None
        self.win: Optional[LatentWindow] = None
        self.pairs = None            # (x[C,W,D] half, m[C,W], y[C,D] half)
        self.np_, self.step_i = 0, 0
        self.vec = np.zeros(self.out_dim, dtype=np.float32)
        self.cost = CostReport(os.path.join(os.path.dirname(path),
                                            "attention_cost.json")
                               if c.get("cost_report", True) else None)
        self.last: Dict[str, float] = {}
        self._fresh_note = ""
        # RW-2 (GOV-012): optional callable (ended0, window, mask) -> bias[W] or
        # None, set by the relation layer; None = the previous behaviour.
        self.bias_fn = None
        self.bias_error = None
        # RW-5 (GOV-012): optional callable (z0, window) -> bool (admit this
        # latent); None = always admitted (the previous behaviour).
        self.push_gate = None
        self.gate_error = None
        self.pushed0 = True

    # ---- lazy build once the latent width is known ------------------------
    def _build(self, dim: int) -> None:
        self.win = LatentWindow(self.streams, self.W, dim, self.device)
        self.net = GatedWindowAttention(
            dim, int(self.cfg.get("d_model", 64)), int(self.cfg.get("heads", 2)),
            self.out_dim, float(self.cfg.get("gate_bias", -2.0))).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        C = self.cap
        self.pairs = (torch.zeros(C, self.W, dim, device=self.device,
                                  dtype=torch.float16),
                      torch.zeros(C, self.W, dtype=torch.bool, device=self.device),
                      torch.zeros(C, dim, device=self.device, dtype=torch.float16))
        if os.path.exists(self.path):
            try:
                st = torch.load(self.path, map_location="cpu")
                if int(st.get("latent_dim", -1)) == dim and \
                        int(st.get("out_dim", -1)) == self.out_dim:
                    self.net.load_state_dict(st["net"])
                else:           # latent width changed (stage D grow): start fresh
                    self._fresh_note = "fresh (latent width changed)"
            except Exception:
                self._fresh_note = "fresh (state not restored)"
        self.cost.params["attention"] = _nparams(self.net)
        self.cost.mem["attention"] = sum(p.numel() * p.element_size()
                                         for p in self.net.parameters())
        self.cost.mem["ring"] = self.win.buf.numel() * 4
        self.cost.mem["pairs"] = sum(t.numel() * t.element_size() for t in self.pairs)

    def save(self) -> None:
        if self.net is None:
            return
        try:
            torch.save({"latent_dim": self.net.latent_dim, "out_dim": self.out_dim,
                        "net": self.net.state_dict()}, self.path + ".tmp")
            os.replace(self.path + ".tmp", self.path)
        except Exception:
            pass

    # ---- per step ----------------------------------------------------------
    @torch.no_grad()
    def observe(self, latent: torch.Tensor, ended) -> None:
        """latent[N,D] post-observe; ended[N] bool (life boundary / restart)."""
        z = latent.detach().float()
        if self.net is None:
            self._build(int(z.shape[-1]))
        self.step_i += 1
        self.pushed0 = True
        for i in range(min(self.streams, z.shape[0])):
            if (i == 0 and self.push_gate is not None and not bool(ended[0])):
                # RW-5 (GOV-012): the executive gate decides whether stream 0's
                # newest latent enters the window; a failure opens the gate.
                try:
                    if not self.push_gate(z[0], self.win):
                        self.pushed0 = False
                        continue
                except Exception as exc:
                    self.push_gate, self.gate_error = None, repr(exc)
            if bool(ended[i]):
                # GOV-015 F-08: the latent at an episode end belongs to the
                # episode that just ended; pushing it after the clear put it
                # first in the NEW episode's window and in its training pairs.
                # The window restarts empty and the next latent starts it.
                self.win.clear(i)
                continue
            if not bool(torch.isfinite(z[i]).all()):
                # GOV-015 F-06: a NaN or infinite latent is not evidence; it
                # would enter the window, the pairs and (through the loss) the
                # layer's weights for good. Skip the row.
                self.refused = getattr(self, "refused", 0) + 1
                continue
            if self.win.n[i] and self.step_i % self.store_every == 0:
                x, m = self.win.get(i)                  # window ending at t-1
                j = self.np_ % self.cap
                self.pairs[0][j] = x.half()
                self.pairs[1][j] = m
                self.pairs[2][j] = z[i].half()          # target: latent at t
                self.np_ += 1
            self.win.push(i, z[i])
        if not self.win.n[0]:
            # empty window (stream 0 just ended): no summary this step; an
            # all-masked window would softmax over nothing (NaN)
            if self.vec is not None:
                self.vec = np.zeros_like(self.vec)
            return
        x, m = self.win.get(0)                          # stream 0's summary
        self.net.eval()
        _kw = {}
        if self.bias_fn is not None:
            try:
                _b = self.bias_fn(bool(ended[0]), x, m)
                if _b is not None:
                    _kw["bias"] = _b[None]
            except Exception as exc:     # idle, never into the layer (RW-2)
                self.bias_fn, self.bias_error = None, repr(exc)
                self.bias_fails = getattr(self, "bias_fails", 0) + 1
        s, _, g = self.cost.timed("summary", self.net, x[None], m[None], **_kw)
        self.vec = s[0].float().cpu().numpy()
        self.last["gate"] = float(g.mean())
        if self.step_i % self.train_every == 0 and self.np_ >= self.batch:
            self.cost.timed("train", self._train)

    @torch.enable_grad()
    def _train(self) -> None:
        self.net.train()
        n = min(self.np_, self.cap)
        tot = base = 0.0
        for _ in range(self.steps):
            idx = torch.randint(0, n, (self.batch,), device=self.device)
            x, m, y = (t[idx] for t in self.pairs)
            x, y = x.float(), y.float()
            _, p, _ = self.net(x, m)
            tgt = F.layer_norm(y, y.shape[-1:])
            loss = F.mse_loss(p, tgt)
            if not bool(torch.isfinite(loss)):
                self.refused = getattr(self, "refused", 0) + 1
                continue
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            last = x[:, -1]                              # copy-the-last baseline
            base += float(F.mse_loss(F.layer_norm(last, last.shape[-1:]), tgt))
            tot += float(loss)
        self.net.eval()
        self.last["pred_loss"] = tot / self.steps
        self.last["copy_loss"] = base / self.steps

    def feature(self) -> np.ndarray:
        return self.vec

    def report(self) -> str:
        w = self.net.last_weights if self.net is not None else None
        ent = ""
        if w is not None:
            p = w.clamp_min(1e-9)
            ent = f" attn_entropy {float(-(p * p.log()).sum(-1).mean()):.2f}"
        return (" ".join(f"{k} {v:.3f}" for k, v in self.last.items()) + ent
                + (f" [{self._fresh_note}]" if self._fresh_note else "")
                + " | cost " + self.cost.line())
