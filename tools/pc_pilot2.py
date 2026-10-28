#!/usr/bin/env python3
"""WP-20 pilot, SECOND ATTEMPT, with the learning-rate diagnostic in the same run (GOV-016 section 11e).

The first pilot (stages/pc_pilot, GOV-016 11d) FAILED G2, G3 and G4 at pc_hebbian_lr 0.01, pc_iters 4. Its curves showed the
Hebbian core's open-loop error was BEST early (0.816 of copy-last at step 250, better than the RSSM's 1.159 then) and got worse
with training (1.213 at step 3000) while its reconstruction improved slowly (0.0432 -> 0.0279; the RSSM reached 0.0177) and its
KL stayed small (0.2-0.5 against 1.48). This run asks why, and tries the two pre-registered settings, in ONE run on the SAME
data, the SAME batches and the SAME seed, so the arms differ in the named setting only.

ARMS (registered in GOV-016 11e before this ran). The unchanged gate of section 5 is applied to the REGISTERED arms only:
    p2a  Hebbian core, pc_hebbian_lr 0.05, pc_iters 4        (the one difference from attempt 1: the learning rate)
    p2b  Hebbian core, pc_hebbian_lr 0.05, pc_iters 2        (p2a plus the cheaper settling, aimed at G4's acting time)
DIAGNOSTIC arms (reported, never gated; they decide nothing):
    d_lr001   lr 0.01, iters 4   (attempt 1's setting, replicated here: does the failure reproduce?)
    d_lr02    lr 0.2,  iters 4   (a faster rate)
    d_clip    lr 0.05, iters 4, hebbian_clip 0.5   (is the per-step 5% norm clip what slows it?)
and the RSSM reference (the gate's comparator) on the same batches.

PER EVALUATION each Hebbian arm also logs: the three local networks' output error and relative step (a step equal to the
clip means the clip is binding), the weight norms, the running observation variance, the settle/update refusals, and the
training loss terms. The runs are therefore comparable curve by curve.

    PYTHONPATH=. python tools/pc_pilot2.py --data DATA.npz --out DIR [--steps 3000] [--device cuda]
Exit 0 = at least one registered arm passed the gate, 1 = none did, 2 = the run could not complete.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REGISTERED = ("p2a", "p2b")
ARMS = {                      # name: (core, pc kwargs for the constructor, attributes set on the core after building)
    "rssm": ("rssm", {}, {}),
    "p2a": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.05}, {}),
    "p2b": ("pc", {"pc_iters": 2, "pc_step": 0.5, "pc_hebbian_lr": 0.05}, {}),
    "d_lr001": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.01}, {}),
    "d_lr02": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.2}, {}),
    "d_clip": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.05}, {"hebbian_clip": 0.5}),
    # GOV-016 11f (diagnostic only, never registered): the generative target divided by its running RMS
    "d_norm": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.05, "pc_target_norm": True}, {}),
    "d_norm01": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.01, "pc_target_norm": True}, {}),
    # GOV-016 11i (diagnostic only): the normalised head's step is still pinned at the clip, so a smaller clip
    "d_norm_c01": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.05, "pc_target_norm": True}, {"hebbian_clip": 0.01}),
    "d_norm_c002": ("pc", {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.05, "pc_target_norm": True}, {"hebbian_clip": 0.002}),
}
DEFAULT_ARMS = ("rssm", "p2a", "p2b", "d_lr001", "d_lr02", "d_clip")      # what the second pilot ran (unchanged)


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def hebbian_state(m) -> dict:
    """What the local rule did at the last update, and where the weights are (empty for the RSSM)."""
    r = m.rssm
    if not getattr(r, "is_predictive_coding", False):
        return {}
    out = {"obs_var": _f(r.obs_var), "refused": r.refused + r.hebbian_refused, "clip": getattr(r, "hebbian_clip", None)}
    for name, net in (("prior", r.prior_net), ("posterior", r.posterior_net), ("generative", r.obs_pred_net)):
        d = {}
        h = getattr(r, "last_hebbian", {}).get(name)
        if h:
            d["err"] = _f(h["err"])
            d["rel_step"] = _f(h["rel_step"])
        if hasattr(net, "W1"):
            d["W1_norm"], d["W2_norm"] = _f(net.W1.norm()), _f(net.W2.norm())
        out[name] = d
    return out


def run_arm(name: str, data: str, steps: int, eval_every: int, device_name: str, X, n_eval: int = 64) -> dict:
    import numpy as np
    import torch
    device = torch.device(device_name)
    d = np.load(data)
    obs, act, ep = d["obs"], d["act"], d["ep"]
    eps = np.unique(ep)
    n_held = max(1, int(round(len(eps) * 0.15)))
    held, tr = set(eps[-n_held:].tolist()), set(eps[:-n_held].tolist())
    core, kw, attrs = ARMS[name]
    m, B, T = X.build("tiny", obs.shape[-1], device, core=core, pc=dict(kw))
    for k, v in attrs.items():
        setattr(m.rssm, k, v)
    tr_starts, ev_starts = X.windows(ep, T, tr), X.windows(ep, T, held)
    rng = np.random.default_rng(1234)                              # the SAME batches for every arm (as tools/pc_pilot.py)
    ev = ev_starts[np.random.default_rng(99).choice(len(ev_starts), min(n_eval, len(ev_starts)), replace=False)]
    params = sum(p.numel() for p in m.parameters()) + (m.rssm.local_values() if hasattr(m.rssm, "local_values") else 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    curve, diag, t_train = [{"step": 0, **X.evaluate(m, obs, act, ev, T, device)}], [], 0.0
    print(f"\n== {name}: core={core} {kw} {attrs} params {params / 1e6:.2f}M", flush=True)
    for s in range(1, steps + 1):
        o, a, r, c = X.make_batch(obs, act, tr_starts[rng.integers(len(tr_starts), size=B)], T, device)
        t0 = time.time()
        met = m.train_step(o, a, r, c)
        t_train += time.time() - t0
        bad = [k for k, v in met.items() if not math.isfinite(float(v))]
        if bad:
            print(f"  {name} DIVERGED at step {s} ({', '.join(bad)})", flush=True)
            return {"params": params, "diverged_at": s, "curve": curve, "diag": diag}
        if s % eval_every == 0 or s == steps:
            e = X.evaluate(m, obs, act, ev, T, device)
            curve.append({"step": s, "train_seconds": round(t_train, 1), **e})
            diag.append({"step": s, "train": {k: _f(v) for k, v in met.items()},
                         **hebbian_state(m)})
            print(f"  {name:8s} step {s:5d}  recon {e['recon']:.4f}  kl {e['kl']:.3f}  open-loop {e['open_loop_mse']:.5f} "
                  f"({e['open_loop_rel']:.3f}x copy-last)  {t_train / s:.3f}s/step", flush=True)
    run = {"params": params, "core": core, "kwargs": kw, "attrs": attrs, "s_per_step": t_train / steps, "curve": curve,
           "diag": diag, "act_ms_2_streams": X.act_ms(m, obs, act, device)}
    if getattr(m.rssm, "is_predictive_coding", False):
        run["pc_refused"] = m.rssm.refused + m.rssm.hebbian_refused
        run["pc_obs_var"] = float(m.rssm.obs_var)
    if device.type == "cuda":
        run["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"  {name:8s} acting {run['act_ms_2_streams']:.1f} ms per perception step (2 streams)", flush=True)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run


def gate_registered(runs: dict) -> dict:
    """The UNCHANGED gate (tools/pc_pilot.py, GOV-016 section 5) on each registered arm against the RSSM reference."""
    from tools import pc_pilot as P
    out = {}
    for name in REGISTERED:
        if name not in runs or "rssm" not in runs:
            out[name] = {"passed": False, "reason": "arm or reference missing"}
            continue
        out[name] = P._gate({"runs": {"tiny:rssm": runs["rssm"], "tiny:pc": runs[name]}}, "tiny")
    return out


def make_verdict(runs: dict, steps: int) -> dict:
    """The verdict of this attempt: the gate on the REGISTERED arms only; a diagnostic arm can never make it pass."""
    gates = gate_registered(runs)
    v = {"attempt": 2, "registered_arms": list(REGISTERED), "gates": gates,
         "passed": any(g.get("passed") for g in gates.values()),
         "note": "attempts so far on this question: 1 (stages/pc_pilot) + 2 registered arms here; a pass is a licence to run the "
                 "five-seed A/B, not evidence the core is better (GOV-016 11e)",
         "steps": steps}
    last = lambda r: (r["curve"][-1] if r and r.get("curve") else {})
    v["final"] = {n: {**{k: last(r).get(k) for k in ("recon", "kl", "open_loop_rel")}, "s_per_step": r.get("s_per_step"),
                      "act_ms": r.get("act_ms_2_streams")} for n, r in runs.items()}
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma list (default: the second pilot's arms)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from tools import rssm_size_experiment as X
    res_path = os.path.join(a.out, "results.json")
    runs = {}
    try:
        for name in [x.strip() for x in a.arms.split(",") if x.strip()]:
            runs[name] = run_arm(name, a.data, a.steps, max(100, a.steps // 12), a.device, X)
            with open(res_path, "w") as f:
                json.dump({"data": a.data, "steps": a.steps, "runs": runs}, f, indent=1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"PILOT 2 COULD NOT RUN: {e!r}")
        return 2
    v = make_verdict(runs, a.steps)
    if not set(runs) & set(REGISTERED):                    # a diagnostic-only run: nothing is gated, nothing can pass or fail
        v = {"attempt": None, "registered_arms": [], "gates": {}, "passed": None, "diagnostic_only": True,
             "final": v["final"], "steps": a.steps}
    with open(os.path.join(a.out, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    print(json.dumps(v, indent=1))
    if v.get("diagnostic_only"):
        print("PC PILOT 2 DIAGNOSTIC ONLY (no registered arm; nothing gated)")
        return 0
    print("PC PILOT 2 (registered arms): " + ("PASS" if v["passed"] else "FAIL"))
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
