"""Adam-state expansion across a world-model growth (WP-10, GOV-011).
BUILT, UNTESTED. Used only when `tools/grow_brain.py --expand-optim` (or
SKYBOT_GROW_EXPAND_OPTIM=1) wrote the sidecar below; without it a grow moves
`policy_optim.pt` aside and Adam restarts, exactly as before.

WHY. `world_model/growth.py` grows the policy's latent-reading input columns
function-preservingly (`grow_policy_state`), but Adam's moments are shaped for
the OLD input layer, so grow_brain moved them aside and the policy's optimiser
restarted cold: the first updates of a carried policy then take large,
uncalibrated steps. This module carries the moments across with the SAME
column layout the weights used (`growth._h_cols`, `_z_cols`):

  h columns   the new h is k copies of the old one (k = new.det / old.det);
              each copy's gradient equals the old column's, so its moments are
              the old column's moments, copied (NOT divided by k as weights are).
  z columns   old (categorical, class) slots move to their new place with their
              moments; a wholly new slot starts with exp_avg = 0 and exp_avg_sq
              = the mean of the old z block's exp_avg_sq (so its first steps
              are of the size of its neighbours', not unbounded).
  other columns (knowledge, proprio, ...) are copied.
`step` and all unchanged tensors are kept. Any state that does not fit either
way raises GrowthError and the caller falls back to a fresh Adam (the old
behaviour), removing the sidecar so a bad file cannot be retried forever.

The sidecar `policy_optim.grown.json` = {"old": Dims, "new": Dims, "stamp"} is
written by grow_brain next to the (grown) checkpoint and consumed (read and
removed) by the loop's policy_full resume.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from developmental_ai.world_model.growth import Dims, GrowthError, _z_index

SIDECAR = "policy_optim.grown.json"


def _expand_cols(M: torch.Tensor, old: Dims, new: Dims, z_fill: float) -> torch.Tensor:
    if new.det % old.det:
        raise GrowthError(f"det {new.det} is not a multiple of {old.det}")
    k, H = new.det // old.det, old.det
    if M.dim() != 2 or M.shape[1] < old.latent:
        raise GrowthError(f"optimiser tensor {tuple(M.shape)} cannot read latent "
                          f"{old.latent}")
    h = M[:, :H].repeat(1, k)
    z = M.new_full((M.shape[0], new.z), float(z_fill))
    z[:, _z_index(old, new)] = M[:, H:H + old.z]
    return torch.cat([h, z, M[:, H + old.z:]], 1)


def expand_adam_state(opt_state: Dict[str, Any], params: List[torch.Tensor],
                      old: Dims, new: Dims) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Expand a torch Adam state dict whose latent-reading tensors are at the
    OLD width to the live `params` (the grown policy's, in optimiser order).
    Returns (new state dict, report). Raises GrowthError if anything does not fit."""
    delta = new.latent - old.latent
    out_state: Dict[Any, Dict[str, Any]] = {}
    rep = {"expanded": [], "kept": 0}
    for idx, st in (opt_state.get("state") or {}).items():
        i = int(idx)
        if i >= len(params):
            raise GrowthError(f"optimiser state index {i} >= {len(params)} params")
        p, ns = params[i], {}
        for key, t in (st or {}).items():
            if not torch.is_tensor(t) or t.dim() == 0 or tuple(t.shape) == tuple(p.shape):
                ns[key] = t
                rep["kept"] += 1
                continue
            if (t.dim() == 2 and p.dim() == 2 and t.shape[0] == p.shape[0]
                    and t.shape[1] + delta == p.shape[1]):
                fill = 0.0
                if key == "exp_avg_sq":
                    fill = float(t[:, old.det:old.det + old.z].mean().clamp(min=1e-12))
                e = _expand_cols(t.detach().cpu().float(), old, new, fill).to(t.dtype)
                if tuple(e.shape) != tuple(p.shape):
                    raise GrowthError(f"expanded {key} {tuple(e.shape)} != "
                                      f"param {tuple(p.shape)}")
                ns[key] = e
                rep["expanded"].append((i, key, tuple(t.shape), tuple(e.shape)))
                continue
            raise GrowthError(f"optimiser tensor {key} of param {i} "
                              f"{tuple(t.shape)} does not fit {tuple(p.shape)}")
        out_state[idx] = ns
    new_sd = dict(opt_state)
    new_sd["state"] = out_state
    return new_sd, rep


# ---- sidecar ---------------------------------------------------------
def write_sidecar(ckpt: str, old: Dims, new: Dims, stamp: str) -> str:
    p = os.path.join(ckpt, SIDECAR)
    with open(p + ".tmp", "w") as f:
        json.dump({"old": old.__dict__, "new": new.__dict__, "stamp": stamp}, f)
    os.replace(p + ".tmp", p)
    return p


def consume_sidecar(ckpt: str) -> Optional[Tuple[Dims, Dims]]:
    """Read and REMOVE the sidecar (a bad one is never retried)."""
    p = os.path.join(ckpt, SIDECAR)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            j = json.load(f)
        return Dims(**j["old"]), Dims(**j["new"])
    except Exception:
        return None
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
