"""Predictive-coding latent core with local (Hebbian) learning: the alternative to the RSSM posterior (WP-20, GOV-016).

THE QUESTION (docs/research/res-010-predictive-coding-core.md). The RSSM infers its stochastic latent z in ONE
amortised pass and trains every weight by backpropagating one loss. Predictive coding does neither: z is a belief
that SETTLES, over a few small steps, to reduce a precision-weighted PREDICTION ERROR, and the weights learn from
LOCAL quantities -- a synapse changes by (the prediction error at its output) x (the activity at its input), the
Hebbian / delta form, with no error signal carried back through the network.

WHAT IS LOCAL HERE (`learning="hebbian"`, the default of this core) AND WHAT IS NOT. The three networks of the
latent core learn locally, with no autograd gradient on their weights:

  prior_net      h              -> prior logits      target: the settled posterior   (delta rule on the softmax)
  posterior_net  [h, embedding] -> amortised logits  target: the settled logits      (delta rule)
  obs_pred_net   [h, z]         -> embedding         target: the observation         (precision-weighted delta rule)

Each is a `LocalPCMLP`: a two-layer network whose hidden layer is itself a predictive-coding node (a value node
and an error node), relaxed for a few steps; the update is  dW_l = lr * e_l * a_(l-1)^T  with e_l the layer's own
prediction error. At equilibrium that approaches the backpropagation gradient as the hidden precision grows (a known
predictive-coding result; tests/_predictive_coding_smoke.py H checks the cosine, not equality) but is computed from local terms only. The weights are BUFFERS, not
parameters: the optimiser never sees them, and autograd flows THROUGH them to their inputs only (a conduit), so
the rest of the model can still be trained.

NOT LOCAL, STATED PLAINLY. The encoder, the GRU sequence model (h), the decoder and every other head (reward,
continue, flow, inverse dynamics, ...) are still trained by backpropagation of the world model's loss. Making those
local too is a larger, separate change (RES-010 section 5). This package changes how the LATENT CORE infers and
learns; it does not make the whole world model Hebbian.

`learning="backprop"` keeps the earlier version of this core (same settling, the SAME three networks held as ordinary
parameters and trained by the optimiser) as an ablation, so a comparison can separate "iterative inference" from
"local learning" (the optimiser, Adam, is part of what that ablation changes).

Guards (ASS-014 / GOV-015 lessons): a non-finite settle gradient leaves that sample at its amortised guess (counted
in `refused`); a non-finite local update is dropped (counted in `hebbian_refused`); updates are clipped relative to
the weight norm; the precision is clamped. Nothing reads engine state (ADR-002).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from developmental_ai.world_model.rssm import RSSM


def _elu_prime(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, torch.ones_like(x), torch.exp(x.clamp(max=0)))


class LocalPCMLP(nn.Module):
    """x -> ELU(W1 x + b1) -> W2 . + b2, learned by a local predictive-coding rule. Weights are buffers."""

    def __init__(self, n_in: int, n_hid: int, n_out: int):
        super().__init__()
        l1, l2 = nn.Linear(n_in, n_hid), nn.Linear(n_hid, n_out)
        self.register_buffer("W1", l1.weight.detach().clone())
        self.register_buffer("b1", l1.bias.detach().clone())
        self.register_buffer("W2", l2.weight.detach().clone())
        self.register_buffer("b2", l2.bias.detach().clone())
        self.register_buffer("_refused", torch.zeros(()), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x @ self.W1.t() + self.b1) @ self.W2.t() + self.b2

    @property
    def refused(self) -> int:
        return int(self._refused.item())

    def n_values(self) -> int:
        return sum(int(b.numel()) for b in (self.W1, self.b1, self.W2, self.b2))

    @torch.no_grad()
    def deltas(self, x: torch.Tensor, target: torch.Tensor, kind: str = "mse", iters: int = 6,
               dt: float = 0.1, err_scale=1.0, groups: Optional[Tuple[int, int]] = None, pi: float = 8.0):
        """The local updates (dW1, db1, dW2, db2) per unit learning rate, and the mean output error.

        The hidden layer is a value node x1 with an error node e1 = x1 - W1 a0: it is relaxed so that it moves
        along what the layer above says is wrong (W2^T e2) while being pulled back to its own prediction.
        `pi` is the hidden layer's precision: the larger it is, the less the hidden value moves from its own prediction
        and the closer the update is to backpropagation (dt * pi must stay <= 1 for stable relaxation).
        At the fixed point pi * e1 = elu'(x1) * (W2^T e2): the backprop gradient, built from local terms."""
        x, target = x.float(), target.float()
        n = x.shape[0]
        v1 = x @ self.W1.t() + self.b1
        x1 = v1.clone()

        def out_err(x1_):
            phi_ = F.elu(x1_)
            y = phi_ @ self.W2.t() + self.b2
            if kind == "softmax":
                s, c = groups
                e = (target.view(n, s, c) - F.softmax(y.view(n, s, c), dim=-1)).view(n, -1)
            else:
                e = target - y
            return phi_, e * err_scale

        for _ in range(int(iters)):
            _, e2 = out_err(x1)
            x1 = x1 + dt * (_elu_prime(x1) * (e2 @ self.W2) - pi * (x1 - v1))
        phi, e2 = out_err(x1)
        e1 = pi * (x1 - v1)                      # the precision-weighted hidden error
        d = (e1.t() @ x / n, e1.mean(0), e2.t() @ phi / n, e2.mean(0))
        return d, e2.abs().mean()

    @torch.no_grad()
    def learn(self, x: torch.Tensor, target: torch.Tensor, kind: str = "mse", lr: float = 0.01, iters: int = 6,
              dt: float = 0.1, err_scale=1.0, clip: float = 0.05,
              groups: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        """One local update. No sync: a non-finite update is zeroed and counted, an over-large one is scaled
        so that no weight matrix moves by more than `clip` of its own norm in one step."""
        d, err = self.deltas(x, target, kind, iters, dt, err_scale, groups)
        ok = torch.stack([torch.isfinite(t).all() for t in d]).all()
        d = tuple(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) * ok.to(t.dtype) for t in d)
        self._refused += (~ok).to(self._refused.dtype)
        rel = torch.maximum(lr * d[0].norm() / (self.W1.norm() + 1e-8),
                            lr * d[2].norm() / (self.W2.norm() + 1e-8))
        scale = lr * (clip / (rel + 1e-12)).clamp(max=1.0)
        self.W1 += scale * d[0]
        self.b1 += scale * d[1]
        self.W2 += scale * d[2]
        self.b2 += scale * d[3]
        return {"err": err, "rel_step": rel * (scale / lr).clamp(max=1.0)}


class PlainMLP(nn.Module):
    """The same two-layer network as `LocalPCMLP` (linear, ELU, linear; the same initialisation draws) but with ordinary
    parameters, trained by the optimiser. It exists so the `backprop` ablation differs from the Hebbian core in the
    learning rule only, not in layer norms or topology (ASS-020 F-05)."""

    def __init__(self, n_in: int, n_hid: int, n_out: int):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_hid)
        self.l2 = nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.elu(self.l1(x)))


class PredictiveCodingCore(RSSM):
    is_predictive_coding = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stochastic_size: int = 32,
        stochastic_classes: int = 32,
        deterministic_size: int = 512,
        hidden_size: int = 256,
        film_conditioning: bool = False,
        pc_iters: int = 4,
        pc_step: float = 0.5,
        pc_prior_precision: float = 1.0,
        pc_grad_clip: float = 5.0,
        learning: str = "hebbian",
        hebbian_lr: float = 0.01,
        hebbian_iters: int = 6,
        hebbian_rows: int = 2048,
        hebbian_clip: float = 0.05,
        target_norm: bool = False,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            stochastic_size=stochastic_size, stochastic_classes=stochastic_classes,
            deterministic_size=deterministic_size, hidden_size=hidden_size,
            film_conditioning=film_conditioning)
        if learning not in ("hebbian", "backprop"):
            raise ValueError("pc learning must be 'hebbian' or 'backprop', got %r" % (learning,))
        self.pc_iters = max(0, int(pc_iters))
        self.pc_step = float(pc_step)
        self.pc_prior_precision = float(pc_prior_precision)
        self.pc_grad_clip = float(pc_grad_clip)
        self.learning = learning
        self.hebbian = learning == "hebbian"
        self.hebbian_lr = float(hebbian_lr)
        self.hebbian_iters = int(hebbian_iters)
        self.hebbian_rows = int(hebbian_rows)
        self.hebbian_clip = float(hebbian_clip)
        # GOV-016 11f: the generative head predicts the encoder's own, still-growing embedding (mean square 0.0016 -> 76 in 41
        # steps; the head's weights reached norms of 100+ with the step clip binding in pilot 2). With `target_norm` the target is
        # divided by the running RMS of the embedding (statistics only, no gradient) in the energy, the generative loss and the
        # local update, so the head chases a unit-scale signal. Off (default) leaves every number as before; the buffers exist
        # only when it is on, so default state dicts are unchanged.
        self.target_norm = bool(target_norm)
        if self.target_norm:
            self.register_buffer("embed_ms", torch.tensor(1.0))
            self.register_buffer("embed_n", torch.tensor(0.0))
        gen_in = deterministic_size + self.stoch_dim
        # The same three networks in both modes: only the learning rule differs (ASS-020 F-05). The Hebbian core holds
        # their weights as buffers; the backprop ablation holds the same shapes and initial values as parameters.
        net = LocalPCMLP if self.hebbian else PlainMLP
        self.prior_net = net(deterministic_size, hidden_size, self.stoch_dim)
        self.posterior_net = net(deterministic_size + obs_dim, hidden_size, self.stoch_dim)
        self.obs_pred_net = net(gen_in, hidden_size, obs_dim)
        # Estimated precision: running variance of the observation error (a scalar; saved with the model).
        self.register_buffer("obs_var", torch.tensor(1.0))
        # Samples whose settle gradient was non-finite (left at their amortised guess). A tensor: no host sync.
        self.register_buffer("_refused", torch.zeros(()), persistent=False)
        # Detached mean energy before / after settling for the LAST call (telemetry; no sync until read).
        self.last_energy: Dict[str, torch.Tensor] = {}
        self.last_hebbian: Dict[str, Dict[str, torch.Tensor]] = {}
        self._collecting = False
        self._cache: list = []

    # ------------------------------------------------------------------ readouts
    @property
    def refused(self) -> int:
        return int(self._refused.item())

    @property
    def hebbian_refused(self) -> int:
        return sum(n.refused for n in (self.prior_net, self.posterior_net, self.obs_pred_net)
                   if isinstance(n, LocalPCMLP))

    def local_values(self) -> int:
        """Weights held as buffers (they are not in .parameters(); parameter counts must add them)."""
        return sum(n.n_values() for n in (self.prior_net, self.posterior_net, self.obs_pred_net)
                   if isinstance(n, LocalPCMLP))

    def target_scale(self) -> float | torch.Tensor:
        """1 / the running RMS of the observation embedding when `target_norm` is on, else exactly 1 (a Python float)."""
        if not self.target_norm:
            return 1.0
        return torch.rsqrt(self.embed_ms.detach().clamp(min=1e-6))

    def precision(self) -> torch.Tensor:
        """Observation precision, clamped so a collapsed variance cannot make the energy explode."""
        return 1.0 / (self.obs_var.detach().clamp(min=1e-4, max=1e4))

    # ------------------------------------------------------------------ energy and settling
    def _energy(self, h: torch.Tensor, prior_logits: torch.Tensor, x: torch.Tensor,
                logits: torch.Tensor) -> torch.Tensor:
        """Per-sample energy of the belief `logits` given h, the prior and the observation embedding x."""
        b = logits.shape[0]
        S, C = self.stochastic_size, self.stochastic_classes
        lp = F.log_softmax(logits.view(b, S, C), dim=-1)
        p = lp.exp()
        pred = self.obs_pred_net(torch.cat([h, p.reshape(b, -1)], dim=-1))
        e_obs = 0.5 * self.precision() * ((x * self.target_scale() - pred) ** 2).mean(dim=-1)
        lq = F.log_softmax(prior_logits.view(b, S, C), dim=-1)
        kl = (p * (lp - lq)).sum(dim=-1).mean(dim=-1)
        return e_obs + self.pc_prior_precision * kl

    def _settle(self, h: torch.Tensor, prior_logits: torch.Tensor, x: torch.Tensor,
                l0: torch.Tensor) -> torch.Tensor:
        """K gradient steps on the energy, from the amortised guess. Returns the settled logits (detached)."""
        hd, pd, xd = h.detach().float(), prior_logits.detach().float(), x.detach().float()
        l = l0.detach().float().clone()
        e_first = None
        with torch.enable_grad():
            for _ in range(self.pc_iters):
                l = l.requires_grad_(True)
                e = self._energy(hd, pd, xd, l)
                if e_first is None:
                    e_first = e.detach()
                g, = torch.autograd.grad(e.sum(), l)
                ok = torch.isfinite(g).all(dim=-1, keepdim=True)
                g = torch.where(ok, g, torch.zeros_like(g))
                with torch.no_grad():
                    self._refused += (~ok).sum().to(self._refused.dtype)
                    gn = g.norm(dim=-1, keepdim=True)
                    g = g * (self.pc_grad_clip / (gn + 1e-8)).clamp(max=1.0)
                    l = (l.detach() - self.pc_step * g)
        if e_first is not None:
            with torch.no_grad():
                e_last = self._energy(hd, pd, xd, l.detach())      # energy AFTER the last step
            self.last_energy = {"before": e_first.mean(), "after": e_last.mean()}
        return l.detach()

    # ------------------------------------------------------------------ the one overridden step
    def observe_step(self, prev_state, action, observation) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        state, info = super().observe_step(prev_state, action, observation)
        if self.pc_iters <= 0:
            if self._collecting:
                self._remember(state["h"], observation, info["posterior_logits"], state["z"])
            return state, info
        l0 = info["posterior_logits"]
        settled = self._settle(state["h"], info["prior_logits"], observation, l0)
        # value = the settled belief; gradient = the amortised path's
        logits = l0 + (settled.to(l0.dtype) - l0.detach())
        z = self.categorical(logits)
        if self._collecting:
            self._remember(state["h"], observation, settled, z)
        state = {"h": state["h"], "z": z}
        info = dict(info, posterior_logits=logits, posterior_z=z, amortised_logits=l0.detach())
        return state, info

    # ------------------------------------------------------------------ the generative term (conduit)
    def generative_loss(self, h: torch.Tensor, z: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """Precision-weighted error of the top-down prediction of the (detached) embedding, mean over all steps.
        Updates the running variance that sets the precision (a gradient-enabled training pass only). With local learning the head's own
        weights get no gradient from this; the gradient reaches h and z (and through them the GRU and encoder)."""
        pred = self.obs_pred_net(torch.cat([h, z], dim=-1).float())
        target = embed.detach().float()
        if self.target_norm and self.training and torch.is_grad_enabled():
            with torch.no_grad():                        # the running scale moves in a gradient-enabled training pass only (as obs_var)
                ms = target.pow(2).mean()
                if torch.isfinite(ms):
                    first = (self.embed_n == 0).to(ms.dtype)
                    self.embed_ms.copy_(first * ms + (1 - first) * (0.99 * self.embed_ms + 0.01 * ms))
                    self.embed_n += 1
        err = (target * self.target_scale() - pred) ** 2
        mse = err.mean()
        self.last_mse = mse.detach()                      # raw error (telemetry; the loss is precision-weighted)
        # ASS-020 F-04: only a gradient-enabled training pass moves the precision. A measurement (curriculum,
        # competence, adaptation) reads this loss under no_grad, possibly with the model still in training mode.
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                m = mse.detach()
                if torch.isfinite(m):
                    self.obs_var.mul_(0.99).add_(0.01 * m)
        return 0.5 * self.precision() * mse

    # ------------------------------------------------------------------ local learning
    def begin_collect(self) -> None:
        self._cache = []
        self._collecting = bool(self.hebbian)

    def end_collect(self) -> None:
        self._collecting = False

    def discard_collected(self) -> None:
        self._cache = []
        self._collecting = False

    def _remember(self, h, obs, settled_logits, z) -> None:
        self._cache.append((h.detach().float(), obs.detach().float(), settled_logits.detach().float(),
                            z.detach().float()))

    @torch.no_grad()
    def hebbian_update(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Apply one local update to the three networks from the steps seen since `begin_collect`."""
        self._collecting = False
        cache, self._cache = self._cache, []
        if not self.hebbian or not cache:
            return {}
        h = torch.cat([c[0] for c in cache])
        obs = torch.cat([c[1] for c in cache])
        settled = torch.cat([c[2] for c in cache])
        z = torch.cat([c[3] for c in cache])
        n = h.shape[0]
        if n > self.hebbian_rows:
            idx = torch.randperm(n, device=h.device)[: self.hebbian_rows]
            h, obs, settled, z = h[idx], obs[idx], settled[idx], z[idx]
            n = self.hebbian_rows
        S, C = self.stochastic_size, self.stochastic_classes
        post_p = F.softmax(settled.view(n, S, C), dim=-1).view(n, -1)
        kw = dict(lr=self.hebbian_lr, iters=self.hebbian_iters, clip=self.hebbian_clip)
        out = {
            "prior": self.prior_net.learn(h, post_p, kind="softmax", groups=(S, C), **kw),
            "posterior": self.posterior_net.learn(torch.cat([h, obs], dim=-1), settled, kind="mse", **kw),
            "generative": self.obs_pred_net.learn(
                torch.cat([h, z], dim=-1), obs * self.target_scale(), kind="mse",
                err_scale=self.precision().clamp(0.01, 100.0), **kw),
        }
        self.last_hebbian = out
        return out
