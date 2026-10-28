"""Grow a world model in place — same brain, more room (stage D).

docs/research/res-004-rssm-size-experiment.md, stage D; the decision comes from
infra/autoscaler.py. User design (2026-10-17): SkyBot's brain grows as it
masters its worlds and keeps being the SAME brain — what it knows before a
grow, it knows after.

FUNCTION-PRESERVING BY CONSTRUCTION, VERIFIED BY MEASUREMENT. A grow is a
transform of the world model's state_dict (world_model.pt is a plain
state_dict) into the shapes of a bigger model:

  deterministic state h  (H -> k*H, k an integer: the ladder doubles)
      Every GRU unit is DUPLICATED k times (rows of every gate block), and
      every weight that READS h is split across the copies, w/k + d_j with
      sum_j d_j = 0. Because the copies stay equal, sum_j (w/k + d_j) h = w h:
      the outputs are exact, yet each copy receives a DIFFERENT gradient, so
      they diverge and the new capacity is real. The GRU's own LayerNorm sees
      the same mean and variance over k equal copies, so it is exact too —
      the reason duplication, not zero-padding, is used for h.
  stochastic state z     (S categoricals x C classes -> S2 x C2)
      Old (categorical, class) slots keep their place in the new layout; a new
      class of an old categorical gets logit bias -30 (never chosen); a new
      categorical gets fresh logits but ZERO outgoing weight, so nothing reads
      it until training gives it a use.
  everything that reads the latent [h, z] — decoder, reward, continue, flow
      head, inverse head (latent mode), symbolic decoder — gets its latent
      columns transformed the same way; every other tensor must keep its shape
      or the grow is REFUSED (an unknown layout is never guessed at).

`verify()` then runs old and grown models in eval mode on the same inputs and
reports the largest output difference; the tool refuses to write a grown
brain that does not match. Adam's moments are not carried (world_model.pt
holds none): the optimizer restarts on the grown shape, as it does every run.

Shrinking cannot be exact (information must be thrown away); it is a
separate, approximate path (tools/grow_brain.py --shrink) that relearns a
smaller model from the persisted replay buffer and is accepted only if its
held-out error is within tolerance of the bigger one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

ARCH_FILE = "world_model_arch.json"


class GrowthError(ValueError):
    pass


@dataclass(frozen=True)
class Dims:
    det: int          # deterministic_size (H)
    stoch: int        # stochastic_size (S, number of categoricals)
    classes: int      # stochastic_classes (C)

    @property
    def z(self) -> int:
        return self.stoch * self.classes

    @property
    def latent(self) -> int:
        return self.det + self.z


def dims_of(wm) -> Dims:
    r = wm.rssm
    return Dims(int(r.deterministic_size), int(r.stochastic_size), int(r.stochastic_classes))


# --------------------------------------------------------------- primitives
def _dup_rows(t: torch.Tensor, blocks: int, H: int, k: int) -> torch.Tensor:
    """dim0 = blocks*H -> blocks*(k*H): each block's rows repeated k times."""
    if t.shape[0] != blocks * H:
        raise GrowthError(f"expected dim0 {blocks}x{H}, got {tuple(t.shape)}")
    parts = t.split(H, dim=0)
    return torch.cat([p.repeat((k,) + (1,) * (t.dim() - 1)) for p in parts], 0)


def _zero_sum(shape, k: int, scale: torch.Tensor, gen: torch.Generator, delta: float):
    """k perturbations of `shape` that sum to exactly 0 (per element)."""
    if k == 1 or delta == 0.0:
        return [torch.zeros(shape) for _ in range(k)]
    d = torch.randn((k,) + tuple(shape), generator=gen) * delta * scale
    d = d - d.mean(0, keepdim=True)
    return list(d.unbind(0))


def _h_cols(W: torch.Tensor, off: int, H: int, k: int, gen, delta: float) -> torch.Tensor:
    """Columns [off, off+H) read h: split into k copies w/k + d_j (sum d_j = 0)."""
    blk = W[:, off:off + H]
    scale = blk.abs().mean().clamp(min=1e-8) / k
    ds = _zero_sum(blk.shape, k, scale, gen, delta)
    # the perturbations are zero-sum in exact arithmetic; float32 rounding of
    # w/k + d_j is far below the verify() tolerance
    copies = [blk / k + d for d in ds]
    return torch.cat([W[:, :off]] + copies + [W[:, off + H:]], 1)


def _z_index(old: Dims, new: Dims) -> torch.Tensor:
    """New flat index of every old (categorical, class) slot."""
    s = torch.arange(old.stoch).repeat_interleave(old.classes)
    c = torch.arange(old.classes).repeat(old.stoch)
    return s * new.classes + c


def _z_cols(W: torch.Tensor, off: int, old: Dims, new: Dims) -> torch.Tensor:
    """Columns [off, off+Z) read z: old slots moved to their new place, new = 0."""
    blk = W[:, off:off + old.z]
    out = W.new_zeros(W.shape[0], new.z)
    out[:, _z_index(old, new)] = blk
    return torch.cat([W[:, :off], out, W[:, off + old.z:]], 1)


def _z_rows(t: torch.Tensor, fresh: torch.Tensor, old: Dims, new: Dims,
            new_class_bias: Optional[float], is_bias: bool) -> torch.Tensor:
    """Rows produce z logits: old slots copied; new classes of an old
    categorical never fire; wholly new categoricals keep their fresh init."""
    out = fresh.clone()
    idx = _z_index(old, new)
    if old.classes < new.classes and new_class_bias is not None:
        ss = torch.arange(old.stoch).repeat_interleave(new.classes - old.classes)
        cc = torch.arange(old.classes, new.classes).repeat(old.stoch)
        nidx = ss * new.classes + cc
        out[nidx] = new_class_bias if is_bias else 0.0
    out[idx] = t
    return out


# --------------------------------------------------------------- the grow
_LATENT_CONSUMER_SKIP = ("rssm.",)


def grow_state_dict(old_sd: Dict[str, torch.Tensor], fresh_sd: Dict[str, torch.Tensor],
                    old: Dims, new: Dims, delta: float = 1e-3, seed: int = 0,
                    latent_consumers: bool = True) -> Dict[str, torch.Tensor]:
    """Transform `old_sd` (a model of dims `old`) into the shapes of `fresh_sd`
    (the same architecture built at dims `new`). See the module docstring."""
    if new.stoch < old.stoch or new.classes < old.classes or new.det < old.det:
        raise GrowthError(f"grow only: {old} -> {new} shrinks something "
                          f"(shrinking is the relearning path, tools/grow_brain.py)")
    if new.det % old.det:
        raise GrowthError(f"deterministic size must grow by an integer factor "
                          f"({old.det} -> {new.det})")
    k, H = new.det // old.det, old.det
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    for key, t in old_sd.items():
        if key not in fresh_sd:
            raise GrowthError(f"{key}: not in the grown model (architecture differs)")
        # BOTH sides on CPU (2026-10-17): growing a model built on CUDA mixed
        # the fresh model's GPU tensors with the old model's CPU ones and
        # raised "Expected all tensors to be on the same device" — the tool
        # always ran on CPU, so the Mac tests never saw it.
        f = fresh_sd[key].detach().cpu()
        t = t.detach().cpu().float() if t.is_floating_point() else t.detach().cpu()
        if not torch.is_floating_point(t):
            out[key] = t if t.shape == f.shape else f.clone()     # counters etc.
            continue
        if key == "rssm.gru.weight_ih":
            # reads the input projection (width unchanged): rows only
            out[key] = _dup_rows(t, 3, H, k)
        elif key == "rssm.gru.weight_hh":
            out[key] = _h_cols(_dup_rows(t, 3, H, k), 0, H, k, gen, delta)
        elif key in ("rssm.gru.bias_ih", "rssm.gru.bias_hh"):
            out[key] = _dup_rows(t, 3, H, k)
        elif key in ("rssm.gru_norm.weight", "rssm.gru_norm.bias"):
            out[key] = _dup_rows(t, 1, H, k)
        elif key in ("rssm.prior_net.0.weight", "rssm.posterior_net.0.weight"):
            out[key] = _h_cols(t, 0, H, k, gen, delta)
        elif re.fullmatch(r"rssm\.(prior|posterior)_net\.3\.(weight|bias)", key):
            out[key] = _z_rows(t, f.float(), old, new, new_class_bias=-30.0,
                               is_bias=key.endswith("bias"))
        elif re.fullmatch(r"rssm\.film\.net\.2\.(weight|bias)", key):
            is_b = key.endswith("bias")
            g_old, b_old = t[:old.z], t[old.z:]
            g = f[:new.z].float().clone()
            b = f[new.z:].float().clone()
            g.zero_(); b.zero_()
            if is_b:
                g.fill_(1.0)              # new dims: gamma 1, beta 0 (identity)
            idx = _z_index(old, new)
            g[idx], b[idx] = g_old, b_old
            out[key] = torch.cat([g, b], 0)
        elif re.fullmatch(r"rssm\.(gru_input_proj|_knowledge_proj)\.0\.weight", key):
            out[key] = _z_cols(t, 0, old, new)
        elif (key == "inverse_head.net.0.weight" and t.shape[1] == 2 * old.latent
              and f.shape[1] == 2 * new.latent):
            # inverse_input="latent": [latent_t | latent_t1], two latent blocks
            a = _z_cols(_h_cols(t[:, :old.latent], 0, H, k, gen, delta), k * H, old, new)
            b = _z_cols(_h_cols(t[:, old.latent:], 0, H, k, gen, delta), k * H, old, new)
            out[key] = torch.cat([a, b], 1)
        elif (latent_consumers and t.dim() == 2 and not key.startswith(_LATENT_CONSUMER_SKIP)
              and t.shape[1] >= old.latent and f.shape[1] - t.shape[1] == new.latent - old.latent
              and t.shape[0] == f.shape[0]):
            w = _h_cols(t, 0, H, k, gen, delta)                    # [h | z | rest]
            out[key] = _z_cols(w, k * H, old, new)
        else:
            if t.shape != f.shape:
                raise GrowthError(f"{key}: shape {tuple(t.shape)} -> {tuple(f.shape)} "
                                  f"has no growth rule — refusing to guess")
            out[key] = t
        if out[key].shape != f.shape:
            raise GrowthError(f"{key}: grew to {tuple(out[key].shape)}, model wants "
                              f"{tuple(f.shape)}")
        out[key] = out[key].to(f.dtype)
    missing = set(fresh_sd) - set(out)
    if missing:
        raise GrowthError(f"grown model has tensors the old one lacks: {sorted(missing)[:4]}")
    return out


@torch.no_grad()
def verify(old_wm, new_wm, obs: torch.Tensor, actions: torch.Tensor,
           proprio: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Max |old - grown| over what the model PRODUCES, eval mode (argmax
    latents, so both see the same discrete states): reconstruction, reward,
    continue, and imagined next frames from the observed state."""
    old_wm.eval(); new_wm.eval()
    res = {}
    so, _ = old_wm.observe_sequence(obs, actions, proprio)
    sn, _ = new_wm.observe_sequence(obs, actions, proprio)
    lo = torch.cat([so["h"], so["z"]], -1).reshape(-1, old_wm.rssm.latent_dim)
    ln = torch.cat([sn["h"], sn["z"]], -1).reshape(-1, new_wm.rssm.latent_dim)
    res["reconstruction"] = float((old_wm.decoder(lo) - new_wm.decoder(ln)).abs().max())
    res["reward"] = float((old_wm.predict_reward(lo) - new_wm.predict_reward(ln)).abs().max())
    res["continue"] = float((old_wm.continue_predictor(lo)
                             - new_wm.continue_predictor(ln)).abs().max())
    st_o = {"h": so["h"][:, -1], "z": so["z"][:, -1]}
    st_n = {"h": sn["h"][:, -1], "z": sn["z"][:, -1]}
    worst = 0.0
    for t in range(actions.shape[1]):
        st_o = old_wm.rssm.imagine_step(st_o, actions[:, t])
        st_n = new_wm.rssm.imagine_step(st_n, actions[:, t])
        a = old_wm.decoder(torch.cat([st_o["h"], st_o["z"]], -1))
        b = new_wm.decoder(torch.cat([st_n["h"], st_n["z"]], -1))
        worst = max(worst, float((a - b).abs().max()))
    res["imagined"] = worst
    return res


def grow_world_model(old_wm, build, new: Dims, delta: float = 1e-3, seed: int = 0):
    """`build(dims) -> WorldModel` makes the same architecture at `dims`.
    Returns (grown model, verify-ready)."""
    old = dims_of(old_wm)
    new_wm = build(new)
    sd = grow_state_dict(old_wm.state_dict(), new_wm.state_dict(), old, new,
                         delta=delta, seed=seed)
    new_wm.load_state_dict(sd, strict=True)
    return new_wm


# ------------------------------------------------------------ architecture
def _jsonable(v):
    # NumPy scalars arrive from the LIVE loop (obs_dim, layout indices ...):
    # found 2026-10-17 when a real Doom run logged "world_model_arch.json not
    # written: TypeError('not JSON-serialisable: int64')" — tests built the
    # model from plain ints and never saw it.
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            return v.item()
    except ImportError:                                   # pragma: no cover
        pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    raise TypeError(f"not JSON-serialisable: {type(v).__name__}")


def save_arch(wm, checkpoint_dir: str) -> str:
    """Write the world model's constructor arguments beside world_model.pt."""
    p = os.path.join(checkpoint_dir, ARCH_FILE)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(dict(wm.init_kwargs)), f, indent=1, sort_keys=True)
    os.replace(tmp, p)
    return p


def load_arch(checkpoint_dir: str) -> Dict:
    with open(os.path.join(checkpoint_dir, ARCH_FILE)) as f:
        return json.load(f)


def build_from_arch(arch: Dict, dims: Optional[Dims] = None):
    """The same architecture, optionally at new RSSM dims."""
    from developmental_ai.world_model.rssm import WorldModel
    kw = dict(arch)
    if dims is not None and kw.get("core", "rssm") != "rssm":
        # WP-20: growth knows the RSSM's h/z column layout only. A resized predictive-coding core would
        # silently lose its generative head; refuse until a growth rule is written for it (GOV-016).
        raise ValueError("growing a world model with core=%r is not supported" % kw["core"])
    if kw.get("sensor_image_specs") is not None:
        kw["sensor_image_specs"] = [tuple(x) for x in kw["sensor_image_specs"]]
    if dims is not None:
        kw.update(deterministic_size=dims.det, stochastic_size=dims.stoch,
                  stochastic_classes=dims.classes)
    return WorldModel(**kw)


# ------------------------------------------------------- latent consumers
# THE POLICY READS THE LATENT TOO (2026-10-18). A grow used to leave
# policy.pt sized for the old [h, z]; the resume refused it by shape and the
# whole policy restarted from scratch after every tiny -> small (measured on
# main: 'policy_features' REFUSED ... shape/key mismatch ['shared.0.weight']).
# These helpers apply the same column rules the world model's own latent
# consumers get (_h_cols then _z_cols), so the grown reader computes the same
# function of the grown latent. Callers must VERIFY before writing anything.

# StandaloneActorCritic (arch='rssm') submodules whose first layer reads
# [latent | proprio | gated knowledge] (actor, critic) or [latent | knowledge]
# (the knowledge gate): the latent is always the first block.
POLICY_LATENT_KEYS = (("actor", "shared.0.weight"),
                      ("critic", "net.0.weight"),
                      ("conditioner", "gate_net.0.weight"))
# DreamActorCritic: every first layer reads the latent alone.
DREAM_LATENT_KEYS = ("actor_net.0.weight", "critic_net.0.weight",
                     "target_net.0.weight")


def grow_latent_columns(W: torch.Tensor, old: Dims, new: Dims,
                        gen: torch.Generator, delta: float = 1e-3) -> torch.Tensor:
    """A weight whose first `old.latent` input columns read [h | z]."""
    if W.dim() != 2 or W.shape[1] < old.latent:
        raise GrowthError(f"latent reader expected (out, >= {old.latent}), "
                          f"got {tuple(W.shape)}")
    k, H = new.det // old.det, old.det
    w = _h_cols(W.detach().cpu().float(), 0, H, k, gen, delta)
    return _z_cols(w, k * H, old, new).to(W.dtype)


def grow_policy_state(sd: Dict, old: Dims, new: Dims, delta: float = 1e-3,
                      seed: int = 0) -> Dict:
    """policy.pt (StandaloneActorCritic.get_state_dict, arch 'rssm') grown to
    read the `new` latent. Raises GrowthError if it does not read a latent."""
    if sd.get("arch") != "rssm":
        raise GrowthError(f"policy arch {sd.get('arch')!r} does not read the "
                          f"world-model latent; nothing to grow")
    if int(sd.get("enc_dim", -1)) != old.latent:
        raise GrowthError(f"policy enc_dim {sd.get('enc_dim')} != old latent "
                          f"{old.latent}")
    gen = torch.Generator().manual_seed(seed)
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in sd.items()}
    for part, key in POLICY_LATENT_KEYS:
        if part in out and key in out[part]:
            out[part][key] = grow_latent_columns(out[part][key], old, new, gen, delta)
    out["enc_dim"] = int(new.latent)
    return out


def grow_dream_state(sd: Dict[str, torch.Tensor], old: Dims, new: Dims,
                     delta: float = 1e-3, seed: int = 0) -> Dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out = dict(sd)
    for key in DREAM_LATENT_KEYS:
        if key in out:
            if out[key].shape[1] != old.latent:
                raise GrowthError(f"dream actor {key} reads {out[key].shape[1]} "
                                  f"columns, not the old latent {old.latent}")
            out[key] = grow_latent_columns(out[key], old, new, gen, delta)
    return out


@torch.no_grad()
def verify_policy(old_sd: Dict, new_sd: Dict, lat_old: torch.Tensor,
                  lat_new: torch.Tensor, seed: int = 0) -> Dict[str, float]:
    """Max |old - grown| of the policy's action logits and critic output,
    the old policy on the old model's latents vs the grown policy on the
    grown model's latents for the SAME frames (rows of lat_*). Proprio and
    knowledge get the same random values on both sides."""
    from developmental_ai.policy.actor_critic import (
        ActorNetwork, CriticNetwork, KnowledgeConditioner)

    def build(sd, lat_w):
        a = sd["actor"]
        hid, inp = a["shared.0.weight"].shape
        n_act = a["action_head.weight"].shape[0]
        kd = (sd["conditioner"]["gate_net.0.weight"].shape[0]
              if sd.get("conditioner") else 0)
        actor = ActorNetwork(inp, n_act, hid)
        actor.load_state_dict(a)
        critic = CriticNetwork(inp, sd["critic"]["net.0.weight"].shape[0])
        critic.load_state_dict(sd["critic"])
        cond = None
        if kd:
            cond = KnowledgeConditioner(lat_w, kd)
            cond.load_state_dict(sd["conditioner"])
        return actor.eval(), critic.eval(), cond, inp - lat_w - kd, kd

    ao, co, ko, po, kdo = build(old_sd, lat_old.shape[1])
    an, cn, kn, pn, kdn = build(new_sd, lat_new.shape[1])
    if (po, kdo) != (pn, kdn):
        raise GrowthError(f"grown policy changed its non-latent inputs "
                          f"({po},{kdo}) -> ({pn},{kdn})")
    g = torch.Generator().manual_seed(seed)
    n = lat_old.shape[0]
    pro = torch.randn(n, po, generator=g)
    kv = torch.randn(n, kdo, generator=g)

    def aug(lat, cond):
        parts = [lat, pro]
        if cond is not None:
            parts.append(cond(lat, kv))
        return torch.cat(parts, -1)

    xo, xn = aug(lat_old.float(), ko), aug(lat_new.float(), kn)
    return {"policy_logits": float((ao.action_head(ao.shared(xo))
                                    - an.action_head(an.shared(xn))).abs().max()),
            "policy_value": float((co(xo) - cn(xn)).abs().max())}


@torch.no_grad()
def verify_dream(old_sd, new_sd, lat_old, lat_new) -> Dict[str, float]:
    worst = 0.0
    for key in DREAM_LATENT_KEYS:
        if key in old_sd and key in new_sd:
            bo = old_sd.get(key[:-len("weight")] + "bias")
            bn = new_sd.get(key[:-len("weight")] + "bias")
            yo = lat_old.float() @ old_sd[key].float().t() + (0 if bo is None else bo.float())
            yn = lat_new.float() @ new_sd[key].float().t() + (0 if bn is None else bn.float())
            worst = max(worst, float((yo - yn).abs().max()))
    return {"dream_first_layers": worst}
