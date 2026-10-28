#!/usr/bin/env python3
"""WP-20 pilot GATE: can the predictive-coding core learn at all, at a cost worth a five-seed A/B? (GOV-016 section 5)

Same fixed data, same batches in the same order, same size, the REAL WorldModel with the Doom run's composed
settings; the ONLY difference is `core: rssm` vs `core: pc` (Hebbian, the gated candidate). A third run, `pc_bp`
(the same settling with ordinary backprop-trained networks), is reported but NOT gated: it separates "iterative
inference" from "local learning" (tools/rssm_size_experiment.py does the training and the
held-out evaluation; this file collects the data, runs both cores, and applies the PRE-REGISTERED gate below).

THE GATE (written before any result; docs/governance/gov-016-predictive-coding-core-plan.md section 5). The PC core
passes only if ALL of:
  G0  parameters:   params(pc) <= 1.25 x params(rssm)                     (the comparison stays fair; ADR-005)
  G1  stability:    PC did not diverge and every logged value is finite; refused settle samples are reported
  G2  learns:       PC final open-loop error beats "the frame never changes" (open_loop_rel < 1.0)
  G3  not worse:    PC final open-loop error <= 1.25 x the RSSM's, and PC reconstruction <= 1.25 x the RSSM's
  G4  cost:         PC seconds per training step <= 3 x the RSSM's, acting ms <= 3 x, peak VRAM <= 1.25 x
A FAIL cancels the five-seed test (the stage that follows requires this one). A PASS says only that the core is worth
testing; it is NOT evidence that it is better (that needs the T1 A/B, five seeds, ASS-007).

    PYTHONPATH=. python tools/pc_pilot.py --out DIR [--frames 12000] [--steps 3000] [--device cuda]
Exit code 0 = gate passed, 1 = gate failed, 2 = the pilot itself could not run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _finite_tree(x) -> bool:
    if isinstance(x, dict):
        return all(_finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_finite_tree(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def verdict(res: dict, size: str = "tiny") -> dict:
    """Apply the gate to a rssm_size_experiment results dict (pure; unit-tested without training). The `pc_bp`
    run, if present, is reported under `ablation_pc_backprop` and never gated."""
    out = _gate(res, size)
    bp = res.get("runs", {}).get(f"{size}:pc_bp")
    if bp is not None:
        if "diverged_at" in bp or not bp.get("curve"):
            out["ablation_pc_backprop"] = {"diverged_at": bp.get("diverged_at")}
        else:
            last = bp["curve"][-1]
            out["ablation_pc_backprop"] = {k: last.get(k) for k in ("open_loop_rel", "open_loop_mse", "recon")}
    return out


def _gate(res: dict, size: str = "tiny") -> dict:
    runs = res.get("runs", {})
    a, b = runs.get(f"{size}:rssm"), runs.get(f"{size}:pc")
    out = {"size": size, "gates": {}, "passed": False}
    if a is None or b is None or "diverged_at" in a:
        out["reason"] = "the RSSM reference did not complete" if a is None or "diverged_at" in (a or {}) \
            else "the PC run did not complete"
        if b is not None and "diverged_at" in b:
            out["gates"]["G1"] = {"ok": False, "why": "PC diverged at step %s" % b["diverged_at"]}
        return out
    last = lambda r: r["curve"][-1]
    g = out["gates"]
    ratio = b["params"] / max(1, a["params"])
    g["G0"] = {"ok": ratio <= 1.25, "params_ratio": round(ratio, 3)}
    g["G1"] = {"ok": "diverged_at" not in b and _finite_tree(b), "pc_refused": b.get("pc_refused")}
    if "diverged_at" in b:
        out["reason"] = "PC diverged"
        return out
    rel_pc, rel_rs = last(b)["open_loop_rel"], last(a)["open_loop_rel"]
    g["G2"] = {"ok": rel_pc < 1.0, "pc_open_loop_rel": round(rel_pc, 4)}
    g["G3"] = {"ok": last(b)["open_loop_mse"] <= 1.25 * last(a)["open_loop_mse"]
               and last(b)["recon"] <= 1.25 * last(a)["recon"],
               "open_loop_rel_pc": round(rel_pc, 4), "open_loop_rel_rssm": round(rel_rs, 4),
               "recon_pc": round(last(b)["recon"], 5), "recon_rssm": round(last(a)["recon"], 5)}
    cost = {"s_per_step": b["s_per_step"] / max(1e-9, a["s_per_step"]),
            "act_ms": b["act_ms_2_streams"] / max(1e-9, a["act_ms_2_streams"])}
    ok = cost["s_per_step"] <= 3.0 and cost["act_ms"] <= 3.0
    if "peak_vram_mb" in a and "peak_vram_mb" in b:
        cost["vram"] = b["peak_vram_mb"] / max(1.0, a["peak_vram_mb"])
        ok = ok and cost["vram"] <= 1.25
    g["G4"] = {"ok": ok, **{k: round(v, 3) for k, v in cost.items()}}
    out["passed"] = all(v["ok"] for v in g.values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pc-iters", type=int, default=4)
    ap.add_argument("--pc-step", type=float, default=0.5)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from tools import rssm_size_experiment as X
    data = os.path.join(a.out, "data.npz")
    res_path = os.path.join(a.out, "results.json")
    try:
        if not os.path.exists(data):
            X.collect(a.frames, data, a.seed)
        X.train(data, [a.size], a.steps, max(100, a.steps // 12), a.device, res_path,
                cores=("rssm", "pc", "pc_bp"), pc={"pc_iters": a.pc_iters, "pc_step": a.pc_step})
        res = json.load(open(res_path))
    except Exception as e:                                   # the pilot failing to run is not a gate result
        import traceback
        traceback.print_exc()
        print(f"PILOT COULD NOT RUN: {e!r}")
        return 2
    v = verdict(res, a.size)
    v.update(steps=a.steps, frames=a.frames, pc_iters=a.pc_iters, pc_step=a.pc_step)
    with open(os.path.join(a.out, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    print(json.dumps(v, indent=1))
    print("PC PILOT GATE: " + ("PASS" if v["passed"] else "FAIL"))
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
