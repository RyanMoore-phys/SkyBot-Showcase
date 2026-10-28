"""Action-explained surprise, from the world model's own quantities only.

Built 2026-10-21 (GOV-011, WP-5 and WP-1; both read this one module).
BUILT, UNTESTED. Nothing here is imported unless a switch asks for it
(`curiosity.info_gain.weight > 0` or `goals.discovery.enabled`).

WHAT IT COMPUTES, per stream, per step, from tensors the perception step
already has (the posterior and prior logits of `RSSM.observe_step`):

    s_t      = KL(posterior || prior(. | a_t))   UNCLAMPED, summed over the
               latent's categorical variables (the autoscaler's z-info, but
               per step and per stream; `RSSM.prior_divergence` averages and
               `compute_kl_loss` clamps at free nats, so neither can be used)
    pbar     = mean over every action a' of prior(. | h_{t-1}, z_{t-1}, a')
    agency_t = KL(post || pbar) - KL(post || prior(. | a_t))

`agency_t > 0` means the action actually taken explains the outcome better
than "any action would have" (RES-005). Something surprising under every
action (a monster entering view, flicker, another agent) has agency ~0.
The batched prior over all actions is computed lazily and only for the rows
asked for (`StepSurprise.agency_for`), so the discovery trigger pays it on
spike steps only and information gain pays it on the streams it prices.

UNIVERSAL AND LEARNED (GOV-009 17.3): inputs are the latent logits and the
action the agent executed; no engine value, no game name, no meaning is
assigned to any outcome. Discrete actions only: a continuous-action brain
gets `None` from `StepSurprise.agency_for` and the callers treat that as
"no estimate" (pays 0 / does not fire).

Ranges and the reasons for the clamps live with the callers; this module
only measures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def kl_per_row(post_logits: torch.Tensor, prior_logits: torch.Tensor,
               n_vars: int, n_classes: int) -> torch.Tensor:
    """KL(post || prior) per batch row, summed over the categorical variables.
    Unclamped; gradients are not needed (callers run under no_grad)."""
    b = post_logits.shape[0]
    pri = F.log_softmax(prior_logits.reshape(b, n_vars, n_classes), dim=-1)
    pst = F.softmax(post_logits.reshape(b, n_vars, n_classes), dim=-1)
    return (pst * (torch.log(pst + 1e-8) - pri)).sum(-1).sum(-1)


@torch.no_grad()
def prior_logits_all_actions(rssm, prev_state: Dict[str, torch.Tensor],
                             action_dim: int,
                             rows: Optional[Sequence[int]] = None
                             ) -> torch.Tensor:
    """Prior logits for EVERY one-hot action, from the state BEFORE the step.

    Returns (R, A, stoch_dim) for the R requested rows. Mirrors
    `RSSM.observe_step` up to the prior (FiLM or concatenated action, input
    projection, GRU, norm, prior head) so the entry for the action actually
    taken equals the prior `observe_step` produced (a contract to test).
    """
    h, z = prev_state["h"], prev_state["z"]
    if rows is not None:
        idx = torch.as_tensor(list(rows), device=h.device, dtype=torch.long)
        h, z = h.index_select(0, idx), z.index_select(0, idx)
    r = int(h.shape[0])
    a = int(action_dim)
    eye = torch.eye(a, device=h.device, dtype=h.dtype)
    h_rep = h.unsqueeze(1).expand(r, a, h.shape[-1]).reshape(r * a, -1)
    z_rep = z.unsqueeze(1).expand(r, a, z.shape[-1]).reshape(r * a, -1)
    act = eye.unsqueeze(0).expand(r, a, a).reshape(r * a, a)
    if rssm.film is not None:
        gin = rssm.gru_input_proj(rssm.film(act, z_rep))
    else:
        gin = rssm.gru_input_proj(torch.cat([z_rep, act], dim=-1))
    h2 = rssm.gru_norm(rssm.gru(gin, h_rep))
    return rssm.prior_net(h2).reshape(r, a, -1)


class StepSurprise:
    """One step's surprise for all streams, with lazy per-row agency."""

    def __init__(self, rssm, prev_state, action_tensor, model_info,
                 guard: List[bool], discrete: bool):
        self._rssm = rssm
        self._prev = prev_state
        self._act = action_tensor
        self._mi = model_info
        self.guard = guard                      # True = boundary: ignore row
        self._discrete = bool(discrete)
        nv, nc = int(rssm.stochastic_size), int(rssm.stochastic_classes)
        self._nv, self._nc = nv, nc
        post = model_info["posterior_logits"].detach()
        pri = model_info["prior_logits"].detach()
        self.s = kl_per_row(post, pri, nv, nc)  # (B,)
        self._post, self._pri = post, pri
        self._agency: Dict[int, float] = {}

    def agency_for(self, rows: Sequence[int]) -> Optional[Dict[int, float]]:
        """{row: agency in nats} for the rows asked; computes the batched
        prior over all actions for the rows not yet cached. None when the
        brain's actions are not discrete."""
        if not self._discrete:
            return None
        need = [int(r) for r in rows if int(r) not in self._agency]
        if need:
            a_dim = int(self._act.shape[-1])
            allp = prior_logits_all_actions(self._rssm, self._prev, a_dim, need)
            probs = F.softmax(
                allp.reshape(len(need), a_dim, self._nv, self._nc), dim=-1)
            pbar = probs.mean(1)                              # (R, nv, nc)
            idx = torch.as_tensor(need, device=self._post.device,
                                  dtype=torch.long)
            pst = F.softmax(self._post.index_select(0, idx).reshape(
                len(need), self._nv, self._nc), dim=-1)
            kl_bar = (pst * (torch.log(pst + 1e-8)
                             - torch.log(pbar + 1e-8))).sum(-1).sum(-1)
            kl_a = kl_per_row(self._post.index_select(0, idx),
                              self._pri.index_select(0, idx),
                              self._nv, self._nc)
            ag = (kl_bar - kl_a).tolist()
            for r, v in zip(need, ag):
                self._agency[r] = float(v)
        return {int(r): self._agency[int(r)] for r in rows}


class GuardClock:
    """Per-stream steps since a life boundary. A new life is maximally
    surprising and caused by nothing, so rows within `k_guard` steps of a
    boundary are ignored. The clock only counts up: nothing latches (CLAUDE.md
    4.1); it re-opens by itself after `k_guard` steps."""

    def __init__(self, k_guard: int = 8):
        self.k = int(k_guard)
        self._since: Dict[int, int] = {}

    def update(self, n: int, ended: Sequence[bool]) -> List[bool]:
        out = []
        for i in range(int(n)):
            if i < len(ended) and ended[i]:
                self._since[i] = 0
            else:
                self._since[i] = self._since.get(i, self.k) + 1
            out.append(self._since[i] <= self.k)
        return out
