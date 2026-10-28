#!/usr/bin/env python
"""WP-19 transfer probe (GOV-009, GOV-011). MEASUREMENT ONLY. BUILT, UNTESTED.

Freeze an encoder, train a small head on frames from the SOURCE maps, and
report accuracy on frames from a HELD-OUT map (feature reuse). The label is
the agent's own action between consecutive frames (an inverse-dynamics
probe): it needs no engine value and no game meaning. A randomly initialised
encoder of the same architecture is probed the same way as the reference, so
the number to read is the gap, not the accuracy.

Data: two .npz files (--source, --target) with arrays
  obs     float [N, obs_dim]   flat frames, the model's own observation format
  action  int   [N]            action taken AFTER frame i (label for pair i,i+1)
  start   bool  [N] optional   True where frame i begins a new life/episode
Pairs (i, i+1) are used only when start[i+1] is False.

  PYTHONPATH=. python tools/transfer_probe.py --checkpoint brain_universal/checkpoints \\
      --source src.npz --target held.npz [--out probe.json]

Never imported by the learner; writes nothing but --out.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch


def _pairs(path, n_actions=None):
    d = np.load(path)
    obs = torch.as_tensor(d["obs"], dtype=torch.float32)
    act = torch.as_tensor(d["action"], dtype=torch.long)
    n = obs.shape[0]
    ok = torch.ones(n - 1, dtype=torch.bool)
    if "start" in d.files:
        ok = ~torch.as_tensor(d["start"][1:], dtype=torch.bool)
    idx = torch.nonzero(ok).flatten()
    return obs, idx, act[:-1][idx]


def _features(enc, obs, idx):
    from developmental_ai.world_model.representation import probe_features
    f = probe_features(enc, obs)
    return torch.cat([f[idx], f[idx + 1] - f[idx]], dim=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="dir holding world_model.pt and world_model_arch.json")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    import os
    from developmental_ai.world_model.growth import build_from_arch, load_arch
    from developmental_ai.world_model.representation import probe_fit
    arch = load_arch(a.checkpoint)
    torch.manual_seed(a.seed)
    trained = build_from_arch(arch)
    sd = torch.load(os.path.join(a.checkpoint, "world_model.pt"),
                    map_location="cpu")
    trained.load_state_dict(sd, strict=False)
    torch.manual_seed(a.seed + 1)
    random_ref = build_from_arch(arch)
    so, si, sy = _pairs(a.source)
    to, ti, ty = _pairs(a.target)
    n_cls = int(max(sy.max(), ty.max())) + 1
    res = {}
    for name, wm in (("trained", trained), ("random_init", random_ref)):
        res[name] = probe_fit(_features(wm.encoder, so, si), sy,
                              _features(wm.encoder, to, ti), ty, n_cls,
                              steps=a.steps, seed=a.seed)
    res["reuse_gap"] = res["trained"]["heldout_acc"] - res["random_init"]["heldout_acc"]
    res["n_actions"] = n_cls
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
