#!/usr/bin/env python3
"""WP-20 diagnostic: why did the generative error RISE in `_predictive_coding_smoke` contract G? (GOV-016 section 11c)

Contract G trained the Hebbian PC core 41 steps on one fixed batch and saw the head's error go 0.0379 -> 1.5309. Two
explanations fit that number and they predict different things; this tool separates them. It measures, never decides.

  H-drift   The head's TARGET is the encoder's embedding, which the optimiser is changing at the same time, so "error on a
            fixed batch" is not a fixed problem. Predicts: the embedding's own scale grows over the run, the backprop
            ablation (`pc_bp`) shows the same rise in raw error, and the error relative to the embedding's size does not rise.
  H-rule    The local rule does not converge (a step normalised to `clip` of the weights' norm never settles, or the rule
            diverges). Predicts: the isolated network, given a FIXED target, does not fall either; the rise is larger than the
            ablation's and is also present in relative terms.

Part 1 (isolated): a LocalPCMLP vs the same topology trained by Adam, on a fixed regression (x -> target), over a grid of
          learning rates and clips. No encoder, no GRU: nothing else moves.
Part 2 (in the model): the contract's own run (same build, data, seed, 41 steps) for core `pc` (Hebbian) and `pc_bp`, logging
          every step: raw error, the embedding's variance and mean square, error / mean square, weight norms, refusals.

    PYTHONPATH=. python tools/pc_hebbian_diag.py --out DIR [--device cpu]
Exit 0 always when it ran (it reports); 2 if it could not run. Output: DIR/diag.json and a printed summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.world_model.predictive_coding import LocalPCMLP, PlainMLP   # noqa: E402
from developmental_ai.world_model.rssm import WorldModel   # noqa: E402


def isolated(n_in=64, n_hid=32, n_out=32, steps=200, B=64, seed=0):
    """Fixed regression, same init for every row of the grid. Returns relative error (mse / mean square of target)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, n_in, generator=g)
    w = torch.randn(n_out, n_in, generator=g) / n_in ** 0.5
    t = torch.tanh(x @ w.t())                                # a learnable, bounded target
    ms = float((t ** 2).mean())
    rows = []
    for lr in (0.003, 0.01, 0.05, 0.2):
        for clip in (0.05, 0.5, 1e9):
            torch.manual_seed(seed)
            net = LocalPCMLP(n_in, n_hid, n_out)
            curve = []
            for k in range(steps):
                curve.append(float(((t - net(x)) ** 2).mean()) / ms)
                net.learn(x, t, "mse", lr=lr, clip=clip)
            rows.append({"rule": "hebbian", "lr": lr, "clip": clip, "rel_first": curve[0], "rel_min": min(curve),
                         "rel_last": curve[-1], "monotone": all(b <= a * 1.05 for a, b in zip(curve, curve[1:])),
                         "refused": net.refused})
    for lr in (1e-3, 3e-3, 1e-2):
        torch.manual_seed(seed)
        net = PlainMLP(n_in, n_hid, n_out)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        curve = []
        for k in range(steps):
            loss = ((t - net(x)) ** 2).mean()
            curve.append(float(loss) / ms)
            opt.zero_grad()
            loss.backward()
            opt.step()
        rows.append({"rule": "adam", "lr": lr, "clip": None, "rel_first": curve[0], "rel_min": min(curve),
                     "rel_last": curve[-1], "monotone": all(b <= a * 1.05 for a, b in zip(curve, curve[1:])), "refused": 0})
    return rows


def build(core, seed=0, **kw):                                # the contract's own build
    torch.manual_seed(seed)
    return WorldModel(obs_dim=3 * 32 * 32, action_dim=6, stochastic_size=4, stochastic_classes=4,
                      deterministic_size=32, hidden_dim=32, pixel_obs=True, image_size=32,
                      recon_reduction="sum", inverse_dynamics=True, inverse_input="embed", core=core, **kw)


def data(seed=0, B=2, T=6):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(B, T, 3 * 32 * 32, generator=g)
    a = F.one_hot(torch.randint(0, 6, (B, T), generator=g), 6).float()
    return o, a, torch.zeros(B, T), torch.ones(B, T)


def in_model(core, steps=41, **kw):
    m = build(core, learning_rate=3e-3, pc_iters=2, **kw)
    o, a, r, c = data()
    seen = {}
    orig = m.rssm.generative_loss

    def spy(h, z, embed):
        e = embed.detach().float()
        seen["var"], seen["ms"] = float(e.var()), float((e ** 2).mean())
        return orig(h, z, embed)

    m.rssm.generative_loss = spy
    log = []
    for k in range(steps):
        m.train_step(o, a, r, c)
        mse = float(m.rssm.last_mse)
        row = {"step": k, "mse": mse, "embed_var": seen["var"], "embed_ms": seen["ms"], "rel": mse / max(seen["ms"], 1e-12),
               "obs_var": float(m.rssm.obs_var)}
        if getattr(m.rssm, "hebbian", False):
            row["w_norm"] = {n: float(t.norm()) for n, t in m.rssm.named_buffers() if n.split(".")[-1] in ("W1", "W2")}
            row["refused"] = int(m.rssm.hebbian_refused)
        log.append(row)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        iso = isolated()
        heb = in_model("pc", pc_hebbian_lr=0.05)
        heb_default = in_model("pc", pc_hebbian_lr=0.01)
        bp = in_model("pc", pc_learning="backprop")
    except Exception as e:                                    # noqa: BLE001
        print("pc_hebbian_diag could not run: %r" % (e,))
        return 2
    out = {"isolated": iso, "hebbian_lr0.05": heb, "hebbian_lr0.01": heb_default, "backprop_ablation": bp}
    with open(os.path.join(a.out, "diag.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("== Part 1: isolated fixed regression (relative error = mse / mean square of target; lower is better)")
    for r in iso:
        print("  %-7s lr=%-6g clip=%-6s first %.3f  min %.3f  last %.3f  monotone=%s refused=%d"
              % (r["rule"], r["lr"], r["clip"], r["rel_first"], r["rel_min"], r["rel_last"], r["monotone"], r["refused"]))
    print("== Part 2: inside the model (41 steps on one fixed batch)")
    for name, log in (("hebbian lr 0.05 (the contract)", heb), ("hebbian lr 0.01 (the default)", heb_default),
                      ("backprop ablation", bp)):
        f, l = log[0], log[-1]
        print("  %-32s raw mse %.4f -> %.4f | embed mean-square %.4f -> %.4f | relative %.3f -> %.3f"
              % (name, f["mse"], l["mse"], f["embed_ms"], l["embed_ms"], f["rel"], l["rel"]))
    hl, bl = heb[-1], bp[-1]
    drift = hl["embed_ms"] > 3 * heb[0]["embed_ms"]
    rel_rise = hl["rel"] > 1.5 * heb[0]["rel"]
    iso_ok = any(r["rule"] == "hebbian" and r["rel_last"] < 0.5 * r["rel_first"] for r in iso)
    print("== reading (descriptive; the decision is recorded in GOV-016 11c):")
    print("  embedding scale grew >3x: %s | relative error rose >1.5x: %s | the isolated rule converged somewhere: %s"
          % (drift, rel_rise, iso_ok))
    print("  backprop ablation's raw error rose >3x as well: %s" % (bl["mse"] > 3 * bp[0]["mse"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
