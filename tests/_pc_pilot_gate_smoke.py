"""WP-20 (GOV-016): the pilot gate decides, it can fail, and the queue obeys it.

Born 2026-10-25. `tools/pc_pilot.py` applies a pre-registered gate to a head-to-head of the RSSM core and the
predictive-coding core; `tools/campaign.py` queues the five-seed A/B behind it. A gate that cannot fail, or a queue
that runs the A/B anyway, would make the pilot decoration. No training here: the verdict is a pure function of the
results file.

Contracts:
    A. A GOOD RESULT PASSES: all five gates true. The backprop ablation run (pc_bp) is reported, never gated.
    B. EACH GATE CAN FAIL ALONE: too many parameters (G0), divergence (G1), no better than "the frame never changes"
       (G2), worse than 1.25 x the RSSM (G3), 3x the step time / acting time or 1.25x the VRAM (G4). Each failing case
       fails ONLY its own gate and the verdict is "not passed".
    C. A MISSING OR DIVERGED RUN NEVER PASSES: no PC run, an RSSM reference that diverged.
    D. THE QUEUE OBEYS: t1_wp20_pc_core requires pc_pilot and pc_boot_check; pc_pilot and pc_boot_check require pc_tests (the
       REQUIRED PC suites, ASS-020 F-01) and the informational suites (pc_tests_info) are required by nothing, so a suite that is red
       for an unrelated reason cannot latch the experiment shut; pc_pilot is on_fail=continue (a failed gate must not pause the
       campaign); the A/B runs on the staged snapshot with its own baseline arm and does NOT
       use the shared ab_baseline; the frozen-code boot check list excludes the WP-20 arm (it would boot the old model).
    F. THE OPEN-LOOP EVALUATION IS CAUSALLY ALIGNED (ASS-020 F-02): the first imagined step after `context` observed frames is
       given the action that CAUSED frame `context` (a[:, context-1]), the next a[:, context], and so on; a model that is shown
       the action one frame late (the old behaviour) is caught.
    E. THE ARM IS REAL: its overlay keys exist in configs/universal.yaml (no silent no-op arm) and the shipped
       default is the RSSM.

Run: PYTHONPATH=. python tests/_pc_pilot_gate_smoke.py
"""
import copy
import os
import sys
import types

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import campaign                      # noqa: E402
import campaign_arms as A            # noqa: E402
from tools import pc_pilot           # noqa: E402
from tools import rssm_size_experiment as X   # noqa: E402

NAME = "pc-pilot-gate"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def run(rel=0.8, mse=0.010, recon=0.020, sps=0.25, act=4.0, vram=1000.0, params=1.1e6, **kw):
    r = {"params": int(params), "s_per_step": sps, "act_ms_2_streams": act, "peak_vram_mb": vram,
         "curve": [{"step": 0, "recon": 0.5, "open_loop_mse": 0.1, "open_loop_rel": 1.5},
                   {"step": 3000, "recon": recon, "open_loop_mse": mse, "open_loop_rel": rel}]}
    r.update(kw)
    return r


def results(pc=None, rs=None):
    out = {"runs": {}}
    if rs is not False:
        out["runs"]["tiny:rssm"] = rs or run(rel=0.78, mse=0.0095, recon=0.019, sps=0.20, act=3.0, vram=900.0,
                                             params=1.0e6)
    if pc is not False:
        out["runs"]["tiny:pc"] = pc or run()
    return out


def main():
    v = pc_pilot.verdict(results())
    check(v["passed"] and all(g["ok"] for g in v["gates"].values()) and set(v["gates"]) == {"G0", "G1", "G2", "G3", "G4"},
          "A. a good result passes all five gates")

    res = results()
    res["runs"]["tiny:pc_bp"] = run(rel=1.4, mse=0.05, recon=0.09)           # a terrible ablation run
    v = pc_pilot.verdict(res)
    check(v["passed"] and v["ablation_pc_backprop"]["open_loop_rel"] == 1.4 and "pc_bp" not in str(v["gates"]),
          "A. the backprop ablation run is reported under 'ablation_pc_backprop' and never gated")
    res["runs"]["tiny:pc_bp"] = {"params": 1, "diverged_at": 7, "curve": []}
    check(pc_pilot.verdict(res)["ablation_pc_backprop"] == {"diverged_at": 7},
          "A. a diverged ablation run is reported and does not fail the gate")

    cases = {
        "G0": run(params=1.4e6),
        "G2": run(rel=1.05),
        "G3": run(mse=0.020),
        "G4": run(sps=0.80),
    }
    for gate, pc in cases.items():
        v = pc_pilot.verdict(results(pc=pc))
        failed = sorted(k for k, g in v["gates"].items() if not g["ok"])
        check(not v["passed"] and gate in failed, f"B. {gate} fails on its own bad case and the verdict is not 'passed' ({failed})")
    v = pc_pilot.verdict(results(pc=run(act=10.0)))
    check(not v["passed"] and not v["gates"]["G4"]["ok"], "B. G4 fails on acting time alone")
    v = pc_pilot.verdict(results(pc=run(vram=1300.0)))
    check(not v["passed"] and not v["gates"]["G4"]["ok"], "B. G4 fails on VRAM alone")
    v = pc_pilot.verdict(results(pc=run(recon=0.030)))
    check(not v["passed"] and not v["gates"]["G3"]["ok"], "B. G3 fails on reconstruction alone")
    v = pc_pilot.verdict(results(pc=run(rel=float("nan"), mse=float("nan"))))
    check(not v["passed"], "B. a NaN result cannot pass")

    v = pc_pilot.verdict(results(pc={"params": 1, "diverged_at": 120, "curve": []}))
    check(not v["passed"] and not v["gates"]["G1"]["ok"], "C. a diverged PC run fails G1 and never passes")
    check(not pc_pilot.verdict(results(pc=False))["passed"], "C. no PC run never passes")
    check(not pc_pilot.verdict(results(rs={"params": 1, "diverged_at": 5, "curve": []}))["passed"],
          "C. a diverged RSSM reference never passes (no comparison)")

    a = types.SimpleNamespace(root="/r", code="/r/code", venv="/v", ab00_root="/a", poll=1, idle=1)
    S = {s["id"]: s for s in campaign.build_stages(campaign.Cfg(a))}
    ab = S["t1_wp20_pc_core"]
    check({"pc_pilot", "pc_boot_check"} <= set(ab["requires"]) and "ab_baseline" not in ab["requires"],
          "D. the A/B requires the pilot gate and its boot check, and not the shared baseline")
    check(S["pc_pilot"]["on_fail"] == "continue", "D. a failed gate does not pause the campaign")
    arms = [x.split("=")[0] for x in ab["cmd"] if isinstance(x, str) and "=" in x and not x.startswith("-")]
    check(sorted(arms) == ["baseline", "changed"] and ab["env"]["EVAL_CODE"] == "/r/pc_code"
          and ab["cmd"][1].startswith("/r/pc_code/"),
          "D. the A/B runs both arms on the staged snapshot, not on the frozen code")
    bc = S["arm_boot_checks"]["cmd"][S["arm_boot_checks"]["cmd"].index("--names") + 1].split(",")
    check("wp20_pc_core" not in bc, "D. the frozen-code boot check does not include the WP-20 arm")
    check("t1_wp20_pc_core" in S and "an_t1_wp20_pc_core" in S and S["an_t1_wp20_pc_core"]["base"] == "t1_wp20_pc_core",
          "D. the analysis compares the two arms of the one stage")

    check({"pc_snapshot", "pc_tests"} <= set(S["pc_pilot"]["requires"]) and "pc_tests" in S["pc_boot_check"]["requires"]
          and "pc_pilot" in ab["requires"],
          "D. the pilot and the boot check require the REQUIRED PC suites (pc_tests); the A/B reaches them through the pilot")
    required_by_anything = {r for st in S.values() for r in st.get("requires", [])}
    check("pc_tests_info" in S and "pc_tests_info" not in required_by_anything,
          "D. the informational suites (pc_tests_info) are required by nothing: no latch on an unrelated red suite")
    check("pc_hebbian_diag" in S and "pc_hebbian_diag" not in required_by_anything and S["pc_hebbian_diag"]["on_fail"] == "continue",
          "D. the diagnostic (pc_hebbian_diag, GOV-016 11c) reports, is required by nothing and cannot pause the campaign")
    check("gov015_verify" in S and "gov015_verify" not in required_by_anything and S["gov015_verify"]["on_fail"] == "continue"
          and "_context_hold_smoke" in S["gov015_verify"]["cmd"][2] and "must FAIL" in S["gov015_verify"]["cmd"][2],
          "D. the GOV-015 9.6 verification reports, gates nothing, and runs the context-hold suite on the new code (green) and the old (must fail)")
    check("tests/_predictive_coding_smoke.py" in S["pc_tests"]["cmd"][2] and "tests/_pc_pilot_gate_smoke.py" in S["pc_tests"]["cmd"][2]
          and "tests/_config_keys_smoke.py" in S["pc_tests"]["cmd"][2],
          "D. the required suites are the predictive-coding, pilot-gate and config-key suites")

    # ---- F: causal alignment of the open-loop evaluation (a fake model records the actions it is given)
    seen = []

    class FakeRSSM:
        def imagine_step(self, st, a):
            seen.append(a.argmax(-1).clone())
            return st

    class FakeM:
        rssm = FakeRSSM()

        def eval(self): pass
        def train(self): pass
        def compute_loss(self, o, a, r, c): return {"reconstruction": torch.tensor(0.0), "kl": torch.tensor(0.0)}

        def observe_sequence(self, o, a):
            b, t = o.shape[:2]
            return {"h": torch.zeros(b, t, 2), "z": torch.zeros(b, t, 2)}, {}

        def decoder(self, lat): return torch.zeros(lat.shape[0], 3 * 4 * 4)

    n = 64
    obs = (np.random.default_rng(0).random((n, 3, 4, 4)) * 255).astype(np.uint8)
    act = (np.arange(n) % X.ACTION_DIM).astype(np.int64)
    ctx, hor = 16, 12
    X.evaluate(FakeM(), obs, act, np.array([5]), ctx + hor, torch.device("cpu"), context=ctx, horizon=hor)
    want = [int(act[5 + ctx - 1 + k]) for k in range(hor)]
    got = [int(g[0]) for g in seen]
    check(got == want, "F. the first imagined step gets the action that caused frame `context` (a[context-1]), then a[context], ...")
    check(got != [int(act[5 + ctx + k]) for k in range(hor)], "F. falsifier: the old one-frame-late pairing differs from the corrected one")

    uni = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    check(A.validate(uni) == [], "E. every arm overlay key (WP-20 included) exists in configs/universal.yaml")
    check(uni["world_model"]["core"] == "rssm" and uni["world_model"]["pc_learning"] == "hebbian",
          "E. the shipped default core is the RSSM, and the PC core's learning is Hebbian")
    ov = A.T1["wp20_pc_core"][1]["world_model"]
    check(ov["core"] == "pc" and ov["pc_learning"] == "hebbian", "E. the WP-20 arm is the Hebbian PC core")
    bad = copy.deepcopy(A.T1["wp20_pc_core"][1])
    bad["world_model"]["corr"] = "pc"
    saved = A.T1["wp20_pc_core"]
    A.T1["wp20_pc_core"] = (saved[0], bad, None)
    try:
        check(any("corr" in p for p in A.validate(uni)), "E. falsifier: a misspelt WP-20 key is caught by validate")
    finally:
        A.T1["wp20_pc_core"] = saved
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
