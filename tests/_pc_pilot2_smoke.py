"""WP-20 pilot, second attempt (GOV-016 section 11e): the tool runs, applies the unchanged gate, and cannot be gamed.

Born 2026-10-27. The first pilot failed (GOV-016 11d). `tools/pc_pilot2.py` runs the two registered arms and three diagnostic arms
in one run, in about an hour on `main`; a defect found after that hour is an hour lost, so this suite runs a REAL few-step
run of each kind of arm first and the queue makes the long stage wait for it.

Contracts:
    A. THE REGISTRATION IS WHAT THE DOCUMENT SAYS: the registered arms are exactly p2a and p2b; p2a differs from the replicated
       attempt-1 arm (d_lr001) in the learning rate ONLY; p2b differs from p2a in the settling iterations ONLY; every
       diagnostic arm is outside the registered set.
    B. THE GATE IS THE UNCHANGED ONE: on synthetic results the registered arm's gate equals `tools/pc_pilot._gate` called directly;
       a good arm passes, a bad one fails G2/G3 and G4 each on its own.
    C. A DIAGNOSTIC ARM CANNOT MAKE THE VERDICT PASS: a diagnostic arm with perfect numbers beside two failing registered arms
       leaves `passed` False; a missing reference never passes.
    D. IT RUNS FOR REAL (CPU, a few steps, synthetic frames): every kind of arm (rssm, a Hebbian arm, the clip override, the fewer-iteration
       arm) completes with finite curves, the Hebbian arms report the three networks' error, relative step and weight norms,
       the clip override reaches the core (clip recorded as 0.5), and p2b really settles with 2 iterations.
    E. THE QUEUE (second pilot, the normalisation diagnostic and the re-verification stages): pc_pilot2 waits for its own smoke (pc_pilot2_tests) and for pc_tests, gates nothing (on_fail continue, required by
       no stage), and the A/B stage is NOT required to wait for it (the second attempt is a licence only if it passes, and the A/B
       needs its own overlay, decided after the result).

Run: PYTHONPATH=. python tests/_pc_pilot2_smoke.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

NAME = "pc-pilot2"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def run_curve(rel, recon=0.02, mse=0.035):
    return [{"step": 0, "recon": 0.1, "kl": 0.1, "open_loop_mse": 0.07, "open_loop_rel": 1.6},
            {"step": 100, "recon": recon, "kl": 1.0, "open_loop_mse": mse, "open_loop_rel": rel}]


def fake(rel, recon=0.02, mse=0.035, s_per=0.3, act=2.0, vram=1000.0, params=12_000_000):
    return {"params": params, "s_per_step": s_per, "act_ms_2_streams": act, "peak_vram_mb": vram,
            "curve": run_curve(rel, recon, mse), "pc_refused": 0}


def main():
    from tools import pc_pilot as P
    from tools import pc_pilot2 as Q

    # ---- A
    A = Q.ARMS
    check(tuple(Q.REGISTERED) == ("p2a", "p2b"), "A. the registered arms are exactly p2a and p2b")
    kw = lambda n: dict(A[n][1])
    diff = lambda x, y: {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
    check(diff(kw("p2a"), kw("d_lr001")) == {"pc_hebbian_lr"} and A["p2a"][2] == A["d_lr001"][2],
          "A. p2a differs from the replicated attempt-1 arm in the learning rate only")
    check(diff(kw("p2b"), kw("p2a")) == {"pc_iters"} and A["p2b"][1]["pc_iters"] == 2 and A["p2b"][2] == A["p2a"][2],
          "A. p2b differs from p2a in the settling iterations only")
    check(kw("d_lr001") == {"pc_iters": 4, "pc_step": 0.5, "pc_hebbian_lr": 0.01},
          "A. the replicated arm is attempt 1's setting (lr 0.01, 4 iterations, step 0.5)")
    check(all(n not in Q.REGISTERED for n in ("d_lr001", "d_lr02", "d_clip", "rssm")), "A. no diagnostic arm is registered")

    check(diff(kw("d_norm_c01"), kw("d_norm")) == set() and A["d_norm_c01"][2] == {"hebbian_clip": 0.01} and A["d_norm"][2] == {}
          and A["d_norm_c002"][2] == {"hebbian_clip": 0.002} and kw("d_norm")["pc_target_norm"] is True
          and "d_norm_c01" not in Q.REGISTERED and "d_norm_c002" not in Q.REGISTERED,
          "A. the clip diagnostic arms differ from d_norm in the clip only and are not registered")

    # ---- B
    good = {"rssm": fake(0.82, 0.0177, 0.0379), "p2a": fake(0.80, 0.0180, 0.037, s_per=0.5, act=2.5),
            "p2b": fake(1.20, 0.0279, 0.056, s_per=0.5, act=2.5)}
    g = Q.gate_registered(good)
    direct = P._gate({"runs": {"tiny:rssm": good["rssm"], "tiny:pc": good["p2a"]}}, "tiny")
    check(g["p2a"] == direct and g["p2a"]["passed"], "B. a good registered arm passes and equals the unchanged gate called directly")
    check(not g["p2b"]["passed"] and not g["p2b"]["gates"]["G2"]["ok"] and not g["p2b"]["gates"]["G3"]["ok"]
          and g["p2b"]["gates"]["G4"]["ok"], "B. a bad arm fails G2 and G3, and only those (its cost is fine)")
    slow = dict(good, p2a=fake(0.80, 0.0180, 0.037, act=5.0))                 # 5x the RSSM's 1.0? use explicit ratio
    slow["rssm"] = fake(0.82, 0.0177, 0.0379, act=1.0)
    gs = Q.gate_registered(slow)
    check(not gs["p2a"]["passed"] and not gs["p2a"]["gates"]["G4"]["ok"] and gs["p2a"]["gates"]["G2"]["ok"],
          "B. acting time alone (5x) fails G4 and nothing else")

    # ---- C
    runs = {"rssm": fake(0.82, 0.0177, 0.0379), "p2a": fake(1.2, 0.03, 0.06), "p2b": fake(1.3, 0.03, 0.07),
            "d_lr02": fake(0.5, 0.01, 0.02), "d_clip": fake(0.5, 0.01, 0.02), "d_lr001": fake(0.5, 0.01, 0.02)}
    v = Q.make_verdict(runs, 100)
    check(v["passed"] is False and set(v["gates"]) == {"p2a", "p2b"},
          "C. diagnostic arms with perfect numbers cannot make the verdict pass; only the registered arms are gated")
    runs2 = dict(runs, p2a=fake(0.8, 0.0180, 0.037, act=2.0, s_per=0.5))
    check(Q.make_verdict(runs2, 100)["passed"] is True, "C. a registered arm that passes makes it pass (the verdict can pass)")
    check(Q.make_verdict({"p2a": runs["p2a"]}, 100)["passed"] is False, "C. a missing RSSM reference never passes")

    # ---- D
    import torch
    from tools import rssm_size_experiment as X
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        n_ep, ln = 6, 60
        obs = rng.integers(0, 255, size=(n_ep * ln, 3, 128, 128), dtype=np.uint8)
        act = rng.integers(0, 16, size=n_ep * ln).astype(np.int64)
        ep = np.repeat(np.arange(n_ep), ln).astype(np.int64)
        path = os.path.join(d, "data.npz")
        np.savez(path, obs=obs, act=act, ep=ep)
        out = {}
        for name in ("rssm", "p2b", "d_clip"):
            out[name] = Q.run_arm(name, path, 3, 3, "cpu", X, n_eval=2)
    finite = lambda r: all(np.isfinite(c["open_loop_mse"]) and np.isfinite(c["recon"]) for c in r["curve"])
    check(all("diverged_at" not in r and finite(r) and len(r["curve"]) == 2 for r in out.values()),
          "D. rssm, p2b and d_clip each complete a real few-step run with finite curves")
    hs = out["p2b"]["diag"][-1]
    nets = [hs[k] for k in ("prior", "posterior", "generative")]
    check(all(n and np.isfinite(n["err"]) and np.isfinite(n["rel_step"]) and np.isfinite(n["W1_norm"]) for n in nets)
          and np.isfinite(hs["obs_var"]), "D. a Hebbian arm reports each network's error, relative step and weight norms")
    check(out["d_clip"]["diag"][-1]["clip"] == 0.5 and out["p2b"]["diag"][-1]["clip"] == 0.05,
          "D. the clip override reaches the core (0.5), and the default stays 0.05 elsewhere")
    check(out["p2b"]["kwargs"]["pc_iters"] == 2 and "diag" in out["rssm"] and out["rssm"]["diag"][-1].get("prior") is None,
          "D. p2b settles with 2 iterations; the RSSM reports no Hebbian state")
    check(all(np.isfinite(r["s_per_step"]) and np.isfinite(r["act_ms_2_streams"]) for r in out.values()),
          "D. cost numbers are measured")

    # ---- E
    import types
    import campaign
    a = types.SimpleNamespace(root="/r", code="/r/code", venv="/v", ab00_root="/a", poll=1, idle=1)
    S = {s["id"]: s for s in campaign.build_stages(campaign.Cfg(a))}
    p2, tt = S["pc_pilot2"], S["pc_pilot2_tests"]
    check("pc_pilot2_tests" in p2["requires"] and "pc_tests" in p2["requires"], "E. pc_pilot2 waits for its smoke and for pc_tests")
    check(p2.get("on_fail") == "continue" and not any("pc_pilot2" in s.get("requires", []) for s in S.values()),
          "E. pc_pilot2 gates nothing: on_fail continue and required by no stage")
    check("pc_pilot2" not in S["t1_wp20_pc_core"]["requires"], "E. the A/B does not wait for the second attempt")
    check(tt.get("on_fail") == "continue" and "pc_snapshot" in tt["requires"], "E. the smoke runs on the snapshot and cannot pause the campaign")
    d3, nt = S["pc_pilot3_diag"], S["pc_norm_tests"]
    cmd3 = d3["cmd"][-1]
    check(d3.get("on_fail") == "continue" and "pc_norm_tests" in d3["requires"] and not any("pc_pilot3_diag" in v.get("requires", []) for v in S.values()),
          "E. the normalisation diagnostic waits for its suite, gates nothing and is required by nothing")
    check("--arms rssm,d_norm,d_norm01" in cmd3 and "p2a" not in cmd3 and "p2b" not in cmd3,
          "E. the normalisation diagnostic runs no registered arm (it can neither pass nor fail the gate, and adds no attempt)")
    check(nt.get("on_fail") == "continue" and "_pc_target_norm_smoke" in nt["cmd"][-1], "E. the normalisation suite is queued on the snapshot")
    for k in ("gov015_diag3", "gov015_verify2"):
        check(S[k].get("on_fail") == "continue" and not any(k in v.get("requires", []) for v in S.values()),
              "E. %s reports, gates nothing and is required by nothing" % k)
    bt, bc = S["pc_boot_tests"], S["pc_boot_check2"]
    check("--strict" in bc["cmd"] and "wp20_pc_core" in bc["cmd"] and bc.get("on_fail") == "continue"
          and {"pc_snapshot", "pc_boot_tests"} <= set(bc["requires"]) and not any("pc_boot_check2" in v.get("requires", []) for v in S.values()),
          "E. the second PC boot check is strict, boots the WP-20 arm on the snapshot, waits for its suite and gates nothing")
    check("_arm_boot_check_smoke" in bt["cmd"][-1] and "_report_tools_smoke" in bt["cmd"][-1], "E. its suite runs the boot-check and report-tools tests")
    lt = S["pc_loop_tests"]
    check(lt.get("on_fail") == "continue" and "_pc_loop_smoke" in lt["cmd"][-1] and "pc_snapshot" in lt["requires"]
          and not any("pc_loop_tests" in v.get("requires", []) for v in S.values()),
          "E. the live-loop suite runs on the snapshot, gates nothing and is required by nothing")
    d4 = S["pc_pilot4_diag"]
    check(d4.get("on_fail") == "continue" and "--arms rssm,d_norm_c01,d_norm_c002" in d4["cmd"][-1] and "p2a" not in d4["cmd"][-1]
          and not any("pc_pilot4_diag" in v.get("requires", []) for v in S.values()),
          "E. the clip diagnostic runs no registered arm, gates nothing and is required by nothing")
    # the diagnostic-only verdict: no registered arm -> nothing gated, no pass or fail
    only = {"rssm": fake(0.82, 0.0177, 0.0379), "d_norm": fake(0.7, 0.02, 0.03)}
    v = Q.make_verdict(only, 100)
    check(v["passed"] is False and all(not g.get("passed") for g in v["gates"].values()),
          "E. make_verdict never passes a run that holds no registered arm (the tool reports it as diagnostic only and exits 0)")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
