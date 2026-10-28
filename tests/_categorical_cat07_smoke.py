"""CAT-07 categorical system test: runtime, resources and evaluation isolation (GOV-013, ASS-012 3).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat07_smoke.py

Question (GOV-013 2): do budgets, persistence and telemetry preserve the learner/evaluator boundary
through faults and restarts?

Members (tests/categorical_manifest.py CAT-07): WP-10 (infra/scale_order.py, world_model/growth_optim.py,
infra/autoscaler.py: the ADR-005 order model size -> streams -> stepping rate), WP-14 (the evaluator
package), WP-17 (tools/campaign.py, tools/eval_runner.py), RW-14 relations_resources, RW-15
relations_connected, and the registry (core/relations.py, EvaluatorLeakError).

Resource pressure is exercised ONLY with injected numbers on the pure Signals / PressureOrder objects
(ASS-012 6: no real memory exhaustion, no trainer, no GPU). Persistence uses temp directories. The
tools' subprocess launch is replaced by a recorder, so no phase, game or trainer is ever started.
Each scenario runs as built and again with a deliberate break (harness-side or monkeypatched copy,
never production code) that it must detect. A real violation is left FAILING and reported.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
import random
import sys
import tempfile
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs, reintroduce_nonfinite  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

H = Harness("cat07", "Runtime, resources and evaluation isolation")

from developmental_ai.infra import autoscaler as AS  # noqa: E402
from developmental_ai.infra import scale_order as SO  # noqa: E402
from developmental_ai.core import relations as R  # noqa: E402
from developmental_ai.core.relations import EvaluatorLeakError, RelationError, RelationRegistry  # noqa: E402


@contextlib.contextmanager
def patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def make_scaler(td, actuate=True, stepping=True, adr=True, size="live", streams=8, cooldown=3, **extra):
    sf = os.path.join(td, "scale.json")
    cfg = {"actuate": actuate, "cooldown_segments": cooldown, "min_streams": 2, "max_streams": 8}
    if adr:
        cfg["adr005"] = dict({"enabled": True, "scale_file": sf, "stepping_actuated": stepping,
                              "refusal_expiry_hours": 12}, **extra)
    return AS.Autoscaler(cfg, size=size, streams=streams), sf


def over(step, **kw):
    """A budget breach: acting 5x over the game's budget (injected number, seconds)."""
    return AS.Signals(step=step, z_info=5.0, act_s=0.5, act_budget_s=0.1, **kw)


def calm(step, **kw):
    return AS.Signals(step=step, z_info=5.0, act_s=0.01, act_budget_s=0.1, **kw)


def global_rng():
    return (torch.get_rng_state().numpy().tobytes(), np.random.get_state()[1].tobytes(), random.getstate())


# ============================================================================================
# CT-01 normal causal flow
# ============================================================================================
@H.scenario("CT-01", "WP-10 ADR-005 order: model size first, then streams, then stepping rate (injected pressure)",
            brk="model_rung_skipped", min_exercised=18,
            oracle="a persistent budget breach walks live x8 -> tiny x8 -> tiny x4 -> stepping x2.0 -> hold, one dial per decision")
def ct01_order(brk):
    with tempfile.TemporaryDirectory() as td:
        a, _ = make_scaler(td)
        ctx = patched(SO.PressureOrder, "saturated", lambda self: (True, "skipped")) if brk else contextlib.nullcontext()
        ex, bad = 0, []
        # units: 70 ms on a 100 ms budget is under the 80 % headroom (no decision); 90 ms is over it
        with ctx:
            d = a.observe(AS.Signals(step=0, z_info=5.0, act_s=0.07, act_budget_s=0.1))
            ex += 1
            if d.action != AS.HOLD:
                bad.append("70 ms of 100 ms triggered %s" % d.action)
            want = ([(AS.SHRINK, s, 8) for s in ("medium", "small", "tiny")]
                    + [(AS.FEWER_STREAMS, "tiny", n) for n in (7, 6, 5, 4)]
                    + [(SO.SLOW, "tiny", 4)] * 4 + [(AS.HOLD, "tiny", 4)])
            first = True
            scales = []
            for i, (act, size, streams) in enumerate(want):
                d = a.observe(AS.Signals(step=i + 1, z_info=5.0, act_s=0.09 if first else 0.5, act_budget_s=0.1,
                                         game_s_per_decision=0.1))
                first = False
                ex += 1
                if (d.action, d.target_size, d.target_streams) != (act, size, streams):
                    bad.append("decision %d: %s -> %s x %d, wanted %s -> %s x %d" % (
                        i, d.action, d.target_size, d.target_streams, act, size, streams))
                if d.action == SO.SLOW:
                    scales.append(d.evidence.get("step_scale"))
                    ex += 1
                    if "game_s_per_decision" not in d.evidence or "->" not in str(d.evidence["game_s_per_decision"]):
                        bad.append("slowing not logged with its game time per decision")
            ex += 1
            if scales != [1.25, 1.562, 1.953, 2.0]:
                bad.append("stepping scales %s" % scales)
            ex += 1
            if a.order.step_scale != 2.0 or AS.SIZE_NAMES[a.size] != "tiny" or a.streams != 4:
                bad.append("actuated state %s x %d scale %s" % (AS.SIZE_NAMES[a.size], a.streams, a.order.step_scale))
    return Obs(not bad, ex, "; ".join(bad[:3]) or "the full ADR-005 descent in order")


@H.scenario("CT-01", "WP-10 the Adam moments of the grown policy follow the weights' column layout",
            brk="moments_dropped", min_exercised=7,
            oracle="expand_adam_state: h columns copied per copy, old z slots moved with their moments, new z slots start at the neighbours' scale, other columns kept, step kept")
def ct01_adam(brk):
    from developmental_ai.world_model import growth_optim as GO
    from developmental_ai.world_model.growth import Dims, _z_index
    old, new = Dims(8, 2, 2), Dims(16, 3, 3)
    extra = 3
    torch.manual_seed(0)
    lin_old = torch.nn.Linear(old.latent + extra, 5)
    opt = torch.optim.Adam(lin_old.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        (lin_old(torch.randn(16, old.latent + extra)) ** 2).sum().backward()
        opt.step()
    lin_new = torch.nn.Linear(new.latent + extra, 5)
    sd = opt.state_dict()
    ctx = patched(GO, "_expand_cols", lambda M, o, n, z: M.new_zeros(M.shape[0], n.latent + (M.shape[1] - o.latent))) \
        if brk else contextlib.nullcontext()
    with ctx:
        new_sd, rep = GO.expand_adam_state(sd, list(lin_new.parameters()), old, new)
    st_old, st_new = sd["state"][0], new_sd["state"][0]
    ex, ok = 1, len(rep["expanded"]) == 2         # exp_avg and exp_avg_sq of the weight matrix
    k = new.det // old.det
    for key in ("exp_avg", "exp_avg_sq"):
        M, N = st_old[key], st_new[key]
        ex += 1
        ok &= tuple(N.shape) == tuple(lin_new.weight.shape)
        ex += 1
        ok &= all(torch.equal(N[:, i * old.det:(i + 1) * old.det], M[:, :old.det]) for i in range(k))
        zi = _z_index(old, new)
        ex += 1
        ok &= torch.equal(N[:, new.det:new.det + new.z][:, zi], M[:, old.det:old.latent])
        fresh = torch.ones(new.z, dtype=torch.bool)
        fresh[zi] = False
        want = 0.0 if key == "exp_avg" else float(M[:, old.det:old.latent].mean().clamp(min=1e-12))
        ex += 1
        ok &= bool(torch.allclose(N[:, new.det:new.det + new.z][:, fresh], torch.full((5, int(fresh.sum())), want)))
        ex += 1
        ok &= torch.equal(N[:, new.latent:], M[:, old.latent:])
    ex += 1
    ok &= torch.equal(st_new["step"], st_old["step"]) and tuple(new_sd["state"][1]["exp_avg"].shape) == (5,)
    return Obs(bool(ok), ex, "expanded %d tensors" % len(rep["expanded"]))


@H.scenario("CT-01", "WP-14/WP-17 plan -> resumable state -> scoring: order, split and attainment times",
            brk="arms_not_interleaved", min_exercised=7,
            oracle="seed-major arm-interleaved plan, 40/20 split (fresh control: target only); phases consumed in plan order; checkpoint times counted from target entry")
def ct01_eval(brk):
    from developmental_ai.evaluator import protocol, runner, scoring

    def arm_major(arms, seeds, total_minutes=60.0, source_frac=2.0 / 3.0, fresh_control_arms=()):
        real = arm_major.real
        return sorted(real(arms, seeds, total_minutes, source_frac, fresh_control_arms), key=lambda r: (r.arm, r.seed))
    arm_major.real = protocol.plan_runs
    ctx = patched(protocol, "plan_runs", arm_major) if brk else contextlib.nullcontext()
    ex, bad = 0, []
    with ctx:
        plan = protocol.plan_runs(["a", "b"], [1, 2], fresh_control_arms=["b"])
    ids = [r.run_id for r in plan]
    ex += 1
    if ids != ["a_s1", "b_s1", "a_s2", "b_s2"]:
        bad.append("plan order %s" % ids)
    ex += 1
    if [(p.name, p.minutes) for p in plan[0].phases] != [("source", 40.0), ("target", 20.0)]:
        bad.append("split %s" % (plan[0].phases,))
    ex += 1
    if [(p.name, p.minutes) for p in plan[1].phases] != [("target", 20.0)]:
        bad.append("fresh control phases")
    with tempfile.TemporaryDirectory() as td:
        st = runner.EvalState.load_or_create(os.path.join(td, "eval_state.json"), plan, {"f": "h"})
        order = []
        while True:
            t = st.next_task()
            if t is None:
                break
            order.append((t[0], t[1]))
            st.mark_started(t[0], t[1])
            st.mark_phase_done(t[0], t[1])
        ex += 1
        if order != [("a_s1", "source"), ("a_s1", "target"), ("b_s1", "target"), ("a_s2", "source"),
                     ("a_s2", "target"), ("b_s2", "target")]:
            bad.append("consumption order %s" % order)
        ex += 1
        if any(st.run(r)["status"] != "done" for r in ids):
            bad.append("not all done")
    # scoring: the synthetic detector (evaluator side), RED lines as the run log would carry them
    ev = [(50.0, 0, {"cells_visited": 5, "touches": 0, "collected": 0}),           # source: moved already true
          (105.0, 1, {"cells_visited": 0, "touches": 1, "collected": 0}),
          (110.0, 1, {"cells_visited": 5, "touches": 1, "collected": 0}),
          (120.0, 0, {"cells_visited": 5, "touches": 1, "collected": 0}),
          (160.0, 0, {"cells_visited": 6, "touches": 2, "collected": 3})]
    res = scoring.score_events("synthetic", ev, t_target_start=100.0, budget_s=100.0)
    ex += 1
    if res["attained"] != {"touched": 10.0, "collected": 60.0} or res["at_entry"] != ["moved"]:
        bad.append("scored %s at_entry %s" % (res["attained"], res["at_entry"]))
    late = scoring.score_events("synthetic", ev + [(500.0, 0, {"cells_visited": 9, "touches": 9, "collected": 9})],
                                100.0, 100.0)
    ex += 1
    if late["attained"] != res["attained"]:
        bad.append("an event after the budget changed the score")
    return Obs(not bad, ex, "; ".join(bad[:3]) or "plan, state machine and scoring agree")


@H.scenario("CT-01", "RW-14 / RW-15 probes deliver the loop's real numbers (and a missing key is a skip, not a zero)",
            brk="probe_key_misspelled", min_exercised=11,
            oracle="enabled channels' segment means equal the producer values; channels whose producer key is absent report skipped, n=0")
def ct01_probes(brk):
    from developmental_ai.core.attention_memory import LatentWindow
    ppo = {"entropy": 0.7, "approx_kl": 0.02, "value_loss": 1.5, "clip_frac": 0.1, "max_prob": 0.4}
    wmm = {"kl_raw": 2.5, "grad_norm": 1.2, "total": 3.0}
    if brk:
        ppo = {k + "_": v for k, v in ppo.items()}              # DEFECT: producer renamed its keys
        wmm = {k + "_": v for k, v in wmm.items()}
    loop = SimpleNamespace(_last_ppo=ppo, _last_wm_metrics=wmm,
                           _am=SimpleNamespace(win=LatentWindow(2, 16, 4, "cpu")))
    reg = R.build_default_registry({"enabled": True, "connected_core": {"enabled": True},
                                    "WMC": {"WM": {"enabled": True}}})
    got = reg.sample_all(loop)
    seg = reg.end_segment()["channels"]
    want = {"rel.WMC.WM": 16.0, "rel.BG.PROC": 0.7, "rel.LTP.STDP": None, "rel.BDNF.LTP": 0.02,
            "rel.BG.HABIT": 0.1, "rel.RL.RPE": 1.5, "rel.PRED.ERRL": 2.5, "rel.ERRL.PRED": 1.2,
            "rel.ACC.ERRL": 3.0, "rel.MOTOR.PROC": 0.4}
    ex, ok, why = 0, True, []
    for name, v in want.items():
        s = seg.get(name)
        ex += 1
        if v is None:                                           # applied actor grad norm: key absent in the loop
            ok &= bool(s) and s["n"] == 0 and s["skipped"] >= 1
            continue
        if not (s and s["n"] == 1 and abs(s["mean"] - v) < 1e-6):
            ok = False
            why.append(name)
    ex += 1
    ok &= got >= 1
    return Obs(bool(ok), ex, "sampled %d channels; wrong: %s" % (got, why[:4]))


# ============================================================================================
# CT-02 cut / neutral parity
# ============================================================================================
@H.scenario("CT-02", "WP-10 adr005 off is the old fallback exactly (shrink, then fewer streams, then hold)",
            brk="cut_arm_runs_the_order", min_exercised=14,
            oracle="decision-for-decision identical to an autoscaler built without the adr005 key, over a breach that reaches the floor")
def ct02_cut(brk):
    def trace(a):
        out = []
        for i in range(16):
            sig = over(i) if i % 5 != 4 else AS.Signals(step=i, z_info=5.0, vram_frac=0.95)
            d = a.observe(sig)
            out.append((d.action, d.size, d.streams, d.target_size, d.target_streams))
        return out
    with tempfile.TemporaryDirectory() as td:
        ref, _ = make_scaler(td, adr=False)
        if brk:
            cut, _ = make_scaler(td, adr=True)                        # DEFECT: the 'cut' arm still has the order on
        else:
            cut = AS.Autoscaler({"actuate": True, "cooldown_segments": 3, "min_streams": 2, "max_streams": 8,
                                 "adr005": {"enabled": False, "scale_file": os.path.join(td, "scale.json")}},
                                size="live", streams=8)
        ex = 1
        ok = cut.order is None
        ta, tb = trace(ref), trace(cut)
    ex += len(ta)
    ok &= ta == tb
    ex += 1
    ok &= not any(a[0] == SO.SLOW for a in tb)
    return Obs(bool(ok), ex, "first difference at %s" % next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None))


@H.scenario("CT-02", "RW-14 / RW-15 off: no sample, no segment entry, no loop write; a single channel cut leaves the others",
            brk="cut_ignored", min_exercised=8,
            oracle="a disabled registry reads nothing; disabling one channel removes only that one")
def ct02_registry(brk):
    from developmental_ai.core.attention_memory import LatentWindow
    loop = SimpleNamespace(_last_ppo={"entropy": 0.7}, _last_wm_metrics={"kl_raw": 2.5},
                           _am=SimpleNamespace(win=LatentWindow(2, 16, 4, "cpu")))
    before = copy.deepcopy({k: v for k, v in vars(loop).items() if k != "_am"})
    off = R.build_default_registry({"enabled": bool(brk), "connected_core": {"enabled": True},
                                    "WMC": {"WM": {"enabled": True}}})
    ex, ok = 1, off.sample_all(loop) == 0
    seg = off.end_segment()
    ex += 1
    ok &= seg["channels"] == {} and seg["live"] == 0
    ex += 1
    ok &= {k: v for k, v in vars(loop).items() if k != "_am"} == before
    ch = off.get("rel.BG.PROC")
    ex += 1
    ok &= not off.is_on("rel.BG.PROC") and off.send("rel.BG.PROC", 0.9) == ch.neutral
    on = R.build_default_registry({"enabled": True, "connected_core": {"enabled": True},
                                   "WMC": {"WM": {"enabled": True}}, "BG": {"PROC": {"enabled": False}}})
    on.sample_all(loop)
    s2 = on.end_segment()["channels"]
    ex += 1
    ok &= "rel.BG.PROC" not in s2 and "rel.WMC.WM" in s2 and "rel.PRED.ERRL" in s2
    ex += 1
    ok &= s2["rel.WMC.WM"]["mean"] == 16.0
    ex += 1
    ok &= s2["rel.PRED.ERRL"]["mean"] == 2.5
    ex += 1
    ok &= s2["rel.BG.HABIT"]["n"] == 0            # key absent: skipped, never a zero
    return Obs(bool(ok), ex, "cut: %d live; single cut respected" % seg["live"])


@H.scenario("CT-02", "WP-17 eval_runner without --execute starts nothing and writes nothing",
            brk="plan_only_executes", min_exercised=3,
            oracle="the default mode prints the plan; no subprocess call, no state file")
def ct02_runner(brk):
    import eval_runner
    calls = []
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "exp")
        argv = ["--root", root, "--arm", "baseline=-", "--seeds", "1,2"] + (["--execute"] if brk else [])
        env = {"EVAL_CODE": os.path.join(td, "code")}
        os.makedirs(env["EVAL_CODE"])
        open(os.path.join(env["EVAL_CODE"], "f.py"), "w").write("x=1\n")
        with patched(eval_runner.subprocess, "call", lambda cmd, shell=False: calls.append(cmd) or 0), \
                patched(os, "environ", {**os.environ, **env}), contextlib.redirect_stdout(out):
            rc = eval_runner.main(argv)
        wrote = os.path.exists(os.path.join(root, "eval_state.json"))
    ex = 3
    ok = rc == 0 and not calls and not wrote and "plan:" in out.getvalue()
    return Obs(bool(ok), ex, "calls %d, state file %s" % (len(calls), wrote))


# ============================================================================================
# CT-03 missing / invalid / stale
# ============================================================================================
@H.scenario("CT-03", "WP-10 NaN, absent and zero signals never trigger a resource decision; real pressure still does",
            brk="nan_treated_as_overload", min_exercised=11,
            oracle="NaN/None/zero latency or VRAM and a zero budget produce HOLD with unchanged size and streams; a real overload shrinks")
@reintroduce_nonfinite("developmental_ai.infra.autoscaler")
def ct03_signals(brk):
    nan, inf = float("nan"), float("inf")
    with tempfile.TemporaryDirectory() as td:
        a, _ = make_scaler(td)
        ex, bad = 0, []

        def fix(s):                                   # DEFECT: a sanitiser that maps NaN to +inf
            if brk:
                for f in ("act_s", "vram_frac", "act_budget_s"):
                    v = getattr(s, f)
                    if v is not None and v != v:
                        setattr(s, f, inf)
            return s
        cases = [AS.Signals(step=1, z_info=5.0, act_s=nan, act_budget_s=0.1),
                 AS.Signals(step=2, z_info=5.0, act_s=0.5, act_budget_s=nan),
                 AS.Signals(step=3, z_info=5.0, vram_frac=nan),
                 AS.Signals(step=4, z_info=nan, act_s=0.01, act_budget_s=0.1),
                 AS.Signals(step=5, z_info=5.0),
                 AS.Signals(step=6, z_info=5.0, act_s=0.5, act_budget_s=0.0),    # zero budget = no limit
                 AS.Signals(step=7, z_info=5.0, act_s=0.0, act_budget_s=0.1, vram_frac=0.0),   # a legitimate zero
                 AS.Signals(step=8, z_info=5.0, act_s=None, act_budget_s=0.1, vram_frac=None)]
        for s in cases:
            d = a.observe(fix(s))
            ex += 1
            if d.action != AS.HOLD or AS.SIZE_NAMES[a.size] != "live" or a.streams != 8:
                bad.append("step %d -> %s" % (s.step, d.action))
        d = a.observe(over(9))
        ex += 1
        if d.action != AS.SHRINK or d.target_size != "medium":
            bad.append("a real overload no longer shrinks: %s" % d.action)
        d = a.observe(AS.Signals(step=10, z_info=5.0, vram_frac=0.99))
        ex += 1
        if d.action not in (AS.SHRINK, AS.FEWER_STREAMS):
            bad.append("a real VRAM breach ignored: %s" % d.action)
        ex += 1
        if not (a.history and all(x.action == AS.HOLD for x in a.history[:8])):
            bad.append("history shows a non-hold for invalid input")
    return Obs(not bad, ex, "; ".join(bad[:3]) or "8 invalid/zero cases held; 2 real breaches acted")


@H.scenario("CT-03", "WP-10 stale, foreign or corrupt refusals never count as saturation; a fresh one does",
            brk="stale_refusal_honoured", min_exercised=7,
            oracle="saturated() is True only for an unexpired refusal of THIS size's next smaller rung")
def ct03_refusals(brk):
    now = time.time()
    with tempfile.TemporaryDirectory() as td:
        a, sf = make_scaler(td, size="small", streams=8)
        ctx = patched(SO, "load_refusals", lambda path, h, now_=None: SO.load_refusals(path, 1e9)) if brk \
            else contextlib.nullcontext()

        def put(rec):
            with open(sf, "w") as f:
                json.dump({"refused_shrinks": rec}, f)
        ex, bad = 0, []
        with ctx:
            def sat():
                return a.order.saturated()[0]
            ex += 1
            if sat():                                                        # no file at all
                bad.append("saturated with no scale file")
            put({"tiny": {"from": "small", "to": "tiny", "t": now}})
            ex += 1
            if not sat():
                bad.append("a fresh refusal was ignored")
            put({"tiny": {"from": "small", "to": "tiny", "t": now - 13 * 3600}})
            ex += 1
            if sat():
                bad.append("an expired refusal (13 h) still counts")
            put({"tiny": {"from": "medium", "to": "tiny", "t": now}})
            ex += 1
            if sat():
                bad.append("a refusal recorded for another size counts")
            put({"small": {"from": "medium", "to": "small", "t": now}})
            ex += 1
            if sat():
                bad.append("a refusal for a different rung counts")
            with open(sf, "w") as f:
                f.write("{not json")
            ex += 1
            if sat():
                bad.append("a corrupt file counts")
            put({"tiny": {"from": "small", "to": "tiny", "t": "garbage"}})
            ex += 1
            if sat():
                bad.append("a malformed timestamp counts")
    return Obs(not bad, ex, "; ".join(bad[:3]) or "7 file states classified correctly")


@H.scenario("CT-03", "WP-14/WP-17 evaluator and driver refuse loudly: no detector, changed frozen files, damaged state",
            brk="unknown_game_skipped_silently", min_exercised=6,
            oracle="DetectorNotDefined, FrozenConfigChanged and SystemExit are raised; malformed log lines are skipped, never scored")
def ct03_eval(brk):
    from developmental_ai.evaluator import detector as D
    from developmental_ai.evaluator import protocol, registry, runner, scoring
    import campaign as C
    ex, ok, why = 0, True, []

    class _Empty(D.Detector):
        def specs(self):
            return []

        def _holds(self, spec_id, ev):
            return False
    ctx = patched(scoring, "get_detector", lambda game: _Empty()) if brk else contextlib.nullcontext()
    with ctx:
        ex += 1
        try:
            scoring.score_events("no_such_game", [(1.0, 0, {"x": 1})], 0.0, 10.0)
            ok = False
            why.append("unknown game scored")
        except registry.DetectorNotDefined:
            pass
    ex += 1
    try:
        class _Bad(D.Detector):
            def specs(self):
                return [D.CheckpointSpec("b", "needs a", prerequisites=("a",)), D.CheckpointSpec("a", "first")]
            def _holds(self, s, e):
                return False
        _Bad()
        ok = False
        why.append("prerequisite listed after use")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "run.log")
        open(log, "w").write("garbage line\n10:00:00 synthetic stream 0: {not a dict\n"
                             "10:00:01 synthetic stream 0: {'cells_visited': 5}\n"
                             "10:00:02 synthetic stream x: {'cells_visited': 5}\n"
                             "10:00:03 other stream 1: {'cells_visited': 9}\n")
        ev = scoring.parse_evaluator_log(log, game="synthetic")
        ex += 1
        ok &= len(ev) == 1 and ev[0][1] == 0 and ev[0][2] == {"cells_visited": 5}
        plan = protocol.plan_runs(["a"], [1])
        sp = os.path.join(td, "eval_state.json")
        runner.EvalState.load_or_create(sp, plan, {"f": "h1"})
        ex += 1
        try:
            runner.EvalState.load_or_create(sp, plan, {"f": "h2"})
            ok = False
            why.append("changed frozen hash accepted")
        except runner.FrozenConfigChanged:
            pass
        ex += 1
        ok &= runner.EvalState.load_or_create(sp, plan, {"f": "h1"}).data["frozen"] == {"f": "h1"}
        # campaign: both state copies unreadable -> refuses to start over
        cfg = SimpleNamespace(root=td + "/camp", code=td + "/camp/code", venv=None, ab00_root=td + "/ab00", poll=1, idle=1)
        cfg = C.Cfg(cfg)
        os.makedirs(cfg.code)
        open(os.path.join(cfg.code, "f.py"), "w").write("x=1\n")
        os.makedirs(cfg.root, exist_ok=True)
        open(cfg.state_path, "w").write("{broken")
        open(cfg.state_path + ".bak", "w").write("{also broken")
        ex += 1
        try:
            C.Campaign(cfg, stages=[])
            ok = False
            why.append("campaign started over a damaged record")
        except SystemExit:
            pass
    return Obs(bool(ok), ex, "; ".join(why[:3]) or "loud refusals everywhere")


# ============================================================================================
# CT-04 competing controls
# ============================================================================================
@H.scenario("CT-04", "WP-10 the stepping scale has one writer: bounded, no compounding, advice moves nothing",
            brk="duplicate_scale_write", min_exercised=14,
            oracle="step_scale moves only through PressureOrder.apply on an actuated decision, by one factor per decision, within [1, max]; unactuated advice leaves size, streams and scale alone")
def ct04_owner(brk):
    with tempfile.TemporaryDirectory() as td:
        ex, bad = 0, []
        real_apply = SO.PressureOrder.apply

        def twice(self, d):                                         # DEFECT: a second writer applies the move again
            real_apply(self, d)
            if d.action in (SO.SLOW, SO.RESTORE):
                self.step_scale = min(self.max_step_scale, self.step_scale * self.step_factor)
        ctx = patched(SO.PressureOrder, "apply", twice) if brk else contextlib.nullcontext()
        a, _ = make_scaler(td, size="tiny", streams=4)
        seen = []
        with ctx:
            for i in range(12):
                a.observe(over(i))
                seen.append(round(a.order.step_scale, 4))
        want = [1.25, 1.562, 1.953, 2.0] + [2.0] * 8       # evidence rounds to 3 decimals, and the state follows it
        ex += len(seen)
        if seen != want:
            bad.append("scale trace %s" % seen[:6])
        ex += 1
        if not all(1.0 <= x <= 2.0 for x in seen):
            bad.append("out of bounds")
        # advice mode (actuate False): the SAME advice repeats, nothing is overwritten or compounded
        b, _ = make_scaler(td, actuate=False, stepping=False, size="tiny", streams=4)
        decs = [b.observe(over(i)) for i in range(8)]
        ex += 1
        if b.order.step_scale != 1.0 or AS.SIZE_NAMES[b.size] != "tiny" or b.streams != 4:
            bad.append("advice moved the state")
        ex += 1
        if {d.evidence.get("step_scale") for d in decs if d.action == SO.SLOW} != {1.25}:
            bad.append("unapplied advice compounded")
    return Obs(not bad, ex, "; ".join(bad[:3]) or "single writer, bounded, advice inert")


# ============================================================================================
# CT-05 temporal / state coherence
# ============================================================================================
@H.scenario("CT-05", "WP-14 phase and stream coherence in scoring: entry state per stream, earliest across streams, no source leakage",
            brk="phase_boundary_shifted_into_source", min_exercised=6,
            oracle="a checkpoint true at target entry (any stream) is excluded; times count from target entry; the earliest stream wins; events outside the budget are ignored")
def ct05_phase(brk):
    from developmental_ai.evaluator import scoring
    ev = [(50.0, 0, {"cells_visited": 5, "touches": 0, "collected": 0}),
          (60.0, 1, {"cells_visited": 0, "touches": 0, "collected": 0}),
          (105.0, 1, {"cells_visited": 0, "touches": 1, "collected": 0}),
          (110.0, 1, {"cells_visited": 5, "touches": 1, "collected": 0}),
          (120.0, 0, {"cells_visited": 5, "touches": 1, "collected": 0}),
          (160.0, 0, {"cells_visited": 6, "touches": 2, "collected": 3}),
          (999.0, 1, {"cells_visited": 9, "touches": 9, "collected": 9})]
    t0 = 40.0 if brk else 100.0                    # DEFECT: the boundary is placed inside the source phase
    res = scoring.score_events("synthetic", ev, t_target_start=t0, budget_s=100.0 if not brk else 160.0)
    ex = 1
    ok = res["at_entry"] == ["moved"]
    ex += 1
    ok &= res["attained"].get("touched") == 10.0       # stream 1 at 110 s (its prerequisite met), earliest across streams
    ex += 1
    ok &= res["attained"].get("collected") == 60.0
    ex += 1
    ok &= "moved" not in res["attained"]
    ex += 1
    ok &= res["streams_seen"] == [0, 1]
    ex += 1
    ok &= res["events_used"] == 4
    return Obs(bool(ok), ex, "attained %s at_entry %s" % (res["attained"], res["at_entry"]))


@H.scenario("CT-05", "WP-10 the recorded fingerprint always describes the state the next decision starts from",
            brk="decision_not_applied", min_exercised=12,
            oracle="after every actuated decision state()['fingerprint'] equals (size, streams, step_scale) of that decision's target")
def ct05_fingerprint(brk):
    with tempfile.TemporaryDirectory() as td:
        a, _ = make_scaler(td)
        ctx = patched(SO.PressureOrder, "apply", lambda self, d: None) if brk else contextlib.nullcontext()
        ex, bad = 0, []
        with ctx:
            for i in range(14):
                d = a.observe(over(i))
                fp = a.state()["order"]["fingerprint"]
                ex += 1
                want_scale = d.evidence.get("step_scale", fp["step_scale"]) if d.action in (SO.SLOW, SO.RESTORE) \
                    else fp["step_scale"]
                if (fp["size"], fp["streams"]) != (d.target_size, d.target_streams) or fp["step_scale"] != want_scale:
                    bad.append("decision %d %s: fingerprint %s vs target %s x %d" % (
                        i, d.action, fp, d.target_size, d.target_streams))
    return Obs(not bad, ex, "; ".join(bad[:2]) or "fingerprint tracked 14 decisions")


# ============================================================================================
# CT-06 failure and recovery
# ============================================================================================
@H.scenario("CT-06", "WP-10 no latch: a refusal expires or is outgrown, the cooldown expires, stepping is restored within its bound",
            brk="refusal_never_expires", min_exercised=8,
            oracle="saturation from a refusal ends at expiry and when the size changes; after pressure clears the stepping scale returns to 1.0 within ceil(log(max)/log(factor)) restores spaced by the cooldown")
def ct06_latch(brk):
    now = time.time()
    ex, bad = 0, []
    with tempfile.TemporaryDirectory() as td:
        a, sf = make_scaler(td, size="small", streams=8)
        if brk:
            a.order.expiry_h = float("inf")                       # DEFECT: refusals never lapse
        with open(sf, "w") as f:
            json.dump({"refused_shrinks": {"tiny": {"from": "small", "to": "tiny", "t": now}}}, f)
        ex += 1
        if not a.order.saturated()[0]:
            bad.append("a fresh refusal does not saturate")
        with open(sf, "w") as f:
            json.dump({"refused_shrinks": {"tiny": {"from": "small", "to": "tiny", "t": now - 13 * 3600}}}, f)
        ex += 1
        if a.order.saturated()[0]:
            bad.append("saturation latched past the expiry")
        with open(sf, "w") as f:
            json.dump({"refused_shrinks": {"tiny": {"from": "small", "to": "tiny", "t": now}}}, f)
        a.size = AS.SIZE_NAMES.index("medium")                     # the model changed size
        ex += 1
        if a.order.saturated()[0]:
            bad.append("saturation latched after the model changed size")
        # pressure, then calm: stepping returns to 1.0 within the bound
        b, _ = make_scaler(td, size="tiny", streams=4, cooldown=3)
        for i in range(8):
            b.observe(over(i))
        ex += 1
        if b.order.step_scale != 2.0:
            bad.append("setup: scale %s" % b.order.step_scale)
        restores, segs = 0, 0
        while b.order.step_scale > 1.0 and segs < 200:
            d = b.observe(calm(100 + segs))
            segs += 1
            restores += d.action == SO.RESTORE
        import math
        bound_restores = math.ceil(math.log(2.0) / math.log(1.25))
        ex += 1
        if b.order.step_scale != 1.0:
            bad.append("stepping not restored in %d calm segments" % segs)
        ex += 1
        if restores > bound_restores:
            bad.append("%d restores > bound %d" % (restores, bound_restores))
        ex += 1
        if segs > bound_restores * 3 + 3:
            bad.append("%d segments > %d x cooldown + slack" % (segs, bound_restores))
        ex += 1
        if "game_s_per_decision" not in b.history[-1].evidence:
            bad.append("restore not logged with game time")
    return Obs(not bad, ex, "; ".join(bad[:3]) or "released: expiry, size change, %d restores in %d segments" % (
        restores, segs))


@H.scenario("CT-06", "WP-17 driver: a crashed seed re-runs once with the same seed; a second crash is final, observable and bounded",
            brk="retry_unbounded", min_exercised=7,
            oracle="command sequence is exactly source/target of seed 1 twice then seed 2; baseline_s1 ends failed_final with two recorded crashes; 'NO CLAIM' is printed")
def ct06_driver(brk):
    import eval_runner
    from developmental_ai.evaluator import runner
    cmds = []
    # rc per call: s1 source ok, s1 target CRASH, s1 source ok, s1 target CRASH, then everything ok
    rcs = deque([0, 1, 0, 1])
    ctx2 = patched(runner, "MAX_ATTEMPTS", 3) if brk else contextlib.nullcontext()

    def fake(cmd, shell=False):
        cmds.append(cmd)
        return rcs.popleft() if rcs else 0
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as td:
        code = os.path.join(td, "code")
        os.makedirs(code)
        open(os.path.join(code, "f.py"), "w").write("x=1\n")
        root = os.path.join(td, "exp")
        with ctx2, patched(eval_runner.subprocess, "call", fake), \
                patched(os, "environ", {**os.environ, "EVAL_CODE": code}), contextlib.redirect_stdout(out):
            eval_runner.main(["--root", root, "--arm", "baseline=-", "--seeds", "1,2", "--execute"])
        st = json.load(open(os.path.join(root, "eval_state.json")))
    toks = [(c.split()[c.split().index("bash") + 2:][:3]) for c in cmds]
    seq = [(t[0], t[1], t[2]) for t in toks]
    want = [("source", "baseline_s1", "1"), ("target", "baseline_s1", "1")] * 2 + \
           [("source", "baseline_s2", "2"), ("target", "baseline_s2", "2")]
    ex, ok = 1, seq == want
    r1, r2 = st["runs"]["baseline_s1"], st["runs"]["baseline_s2"]
    ex += 1
    ok &= r1["status"] == "failed_final" and len(r1["crashes"]) == 2 and r1["attempts"] == 2
    ex += 1
    ok &= r2["status"] == "done"
    ex += 1
    ok &= "NO CLAIM" in out.getvalue() and "[1]" in out.getvalue()
    ex += 1
    ok &= all(c["reason"].startswith("exit code") for c in r1["crashes"])
    ex += 1
    ok &= out.getvalue().count("CRASH (rc=1)") == 2
    ex += 1
    ok &= json.loads(json.dumps(st))["runs"]["baseline_s2"]["crashes"] == []
    return Obs(bool(ok), ex, "commands %d (want %d)" % (len(seq), len(want)))


@H.scenario("CT-06", "RW-14/15 dead-wire alarm: raised when a channel stops flowing, releases itself when flow returns",
            brk="alarm_latched", min_exercised=5,
            oracle="a channel that carried flow and then went silent alarms after one window; one segment of renewed flow clears it")
def ct06_alarm(brk):
    reg = RelationRegistry({"enabled": True, "window": 3, "eps": 0.01, "WMC": {"WM": {"enabled": True}}})
    from developmental_ai.core import relations_resources
    relations_resources.register(reg)
    ch = reg.get("rel.WMC.WM")
    ctx = contextlib.nullcontext()
    if brk:
        real = type(ch)._dead
        ctx = patched(type(ch), "_dead", lambda self, w, e: bool(self.alarm) or real(self, w, e))   # DEFECT: sticky
    ex, ok = 0, True
    with ctx:
        for _ in range(4):
            ch.observe(16.0)
            reg.end_segment()
        ex += 1
        ok &= not ch.alarm
        for _ in range(4):                                       # flow drops to zero (a sample of 0.0)
            ch.observe(0.0)
            reg.end_segment()
        ex += 1
        ok &= ch.alarm
        ch.observe(16.0)
        reg.end_segment()
        for _ in range(3):
            ch.observe(16.0)
            reg.end_segment()
        ex += 1
        ok &= not ch.alarm
        ex += 1
        ok &= "DEAD-WIRE" not in reg.line()
        # absence is not zero: segments without any sample are COUNTED as skipped samples, never booked as 0.0
        n_hist = len(ch.hist)
        ch.skipped = 0
        reg.sample_all(SimpleNamespace())             # the probe finds no attention window -> None
        reg.end_segment()
        ex += 1
        ok &= ch.skipped == 1 and len(ch.hist) == n_hist
    return Obs(bool(ok), ex, "alarm after zero flow, released on return: %s" % (not ch.alarm))


# ============================================================================================
# CT-07 persistence
# ============================================================================================
@H.scenario("CT-07", "WP-10 the cooldown (and the refusal file) survive the restart that applied a change",
            brk="cooldown_not_restored", min_exercised=6,
            oracle="a rebuilt autoscaler given the saved state waits out the same remaining cooldown as the uninterrupted one; the saved state is atomic JSON")
def ct07_cooldown(brk):
    with tempfile.TemporaryDirectory() as td:
        def build():
            return AS.Autoscaler({"actuate": True, "cooldown_segments": 6, "window": 3, "min_streams": 2,
                                  "max_streams": 8}, size="tiny", streams=4)
        cont = build()
        cont.observe(over(0))                                       # a change; cooldown starts
        for i in range(2):
            cont.observe(AS.Signals(step=1 + i, z_info=5.0, wm_fresh=1.0, wm_train=1.0))
        path = os.path.join(td, "autoscale.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"decision": {}, "state": cont.state()}, f)
        os.replace(path + ".tmp", path)
        fresh = build()
        if not brk:
            fresh.restore(json.load(open(path))["state"])
        ex, bad = 1, []
        reasons_a, reasons_b = [], []
        for i in range(8):
            sig = AS.Signals(step=10 + i, z_info=5.0, wm_fresh=1.0, wm_train=1.0)
            reasons_a.append(cont.observe(sig).reason)
            reasons_b.append(fresh.observe(copy.copy(sig)).reason)
        ex += 1
        cool = [("cooling" in r) for r in reasons_a]
        cool_b = [("cooling" in r) for r in reasons_b]
        if cool != cool_b:
            bad.append("cooldown differs: %s vs %s" % (cool, cool_b))
        ex += 1
        if not any(cool):
            bad.append("fixture never cooled down")
        ex += 1
        if fresh.segments_since_change >= 10 ** 9:
            bad.append("restart reset the cooldown to 'never changed'")
        ex += 1
        if os.path.exists(path + ".tmp"):
            bad.append("temporary file left behind")
        # size/streams come from the BUILT brain, not from the file (declared behaviour)
        other = AS.Autoscaler({"actuate": True}, size="medium", streams=6)
        other.restore(json.load(open(path))["state"])
        ex += 1
        if AS.SIZE_NAMES[other.size] != "medium" or other.streams != 6:
            bad.append("restore overrode the built size/streams")
    return Obs(not bad, ex, "; ".join(bad[:2]) or "cooldown carried across the restart")


@H.scenario("CT-07", "WP-10 the ADR-005 stepping scale that autoscale.json records is restored at the next launch",
            brk="scale_not_restored", min_exercised=3,
            oracle="a saved state with step_scale 2.0 gives a rebuilt PressureOrder step_scale 2.0, so the comparison fingerprint and the restore-first rule continue")
def ct07_scale(brk):
    with tempfile.TemporaryDirectory() as td:
        a, _ = make_scaler(td, size="tiny", streams=4)
        for i in range(6):
            a.observe(over(i))
        saved = json.loads(json.dumps(a.state()))                    # what stack.py writes to autoscale.json
        b, _ = make_scaler(td, size="tiny", streams=4)
        if not brk:
            b.restore(saved)
        ex = 1
        ok = saved["order"]["step_scale"] == 2.0
        ex += 1
        ok &= b.order.step_scale == saved["order"]["step_scale"]
        ex += 1
        ok &= b.state()["order"]["fingerprint"] == saved["order"]["fingerprint"]
    return Obs(bool(ok), ex, "saved scale %s, rebuilt scale %s" % (saved["order"]["step_scale"], b.order.step_scale))


@H.scenario("CT-07", "WP-14/WP-17 evaluator state and the driver resume exactly; a changed frozen file refuses",
            brk="state_not_reloaded", min_exercised=8,
            oracle="re-running the same eval_runner command after completion starts nothing; a half-finished state resumes at the next phase; editing a frozen file raises FrozenConfigChanged until the state is deleted on purpose")
def ct07_driver(brk):
    import eval_runner
    from developmental_ai.evaluator import runner
    calls = []
    out = io.StringIO()
    ex, ok, why = 0, True, []
    with tempfile.TemporaryDirectory() as td:
        code = os.path.join(td, "code")
        os.makedirs(code)
        open(os.path.join(code, "f.py"), "w").write("x=1\n")
        root = os.path.join(td, "exp")
        argv = ["--root", root, "--arm", "baseline=-", "--seeds", "1,2", "--execute"]

        def run(rcs=()):
            q = deque(rcs)
            calls.clear()
            ctx = patched(runner.EvalState, "load_or_create",
                          classmethod(lambda cls, path, plan, frozen=None: cls(path, {
                              "frozen": frozen or {}, "order": [r.run_id for r in plan],
                              "runs": {r.run_id: {"arm": r.arm, "seed": r.seed,
                                                  "phases": [[p.name, p.minutes] for p in r.phases],
                                                  "status": "pending", "attempts": 1, "phase_done": [],
                                                  "in_progress": None, "crashes": [], "info": {}} for r in plan}})
                          )) if brk else contextlib.nullcontext()
            with ctx, patched(eval_runner.subprocess, "call", lambda c, shell=False: calls.append(c) or (q.popleft() if q else 0)), \
                    patched(os, "environ", {**os.environ, "EVAL_CODE": code}), contextlib.redirect_stdout(out):
                eval_runner.main(argv)
            return len(calls)
        # the driver dies after two phases: simulate by a state file left with seed 1 half done
        n1 = run()
        ex += 1
        ok &= n1 == 4
        n2 = run()                                             # complete -> a re-run starts nothing
        ex += 1
        if n2 != 0:
            ok = False
            why.append("re-run started %d phases" % n2)
        sp = os.path.join(root, "eval_state.json")
        st = json.load(open(sp))
        st["runs"]["baseline_s2"].update(status="pending", phase_done=["source"])
        json.dump(st, open(sp, "w"))
        n3 = run()
        ex += 1
        if n3 != 1:
            ok = False
            why.append("half-finished run resumed with %d phases, wanted 1 (target only)" % n3)
        # a driver killed mid-phase: in_progress is recovered as a crash (and the seed re-runs from its start)
        st = json.load(open(sp))
        st["runs"]["baseline_s1"].update(status="pending", phase_done=["source"], in_progress={"phase": "target", "t": 0.0})
        json.dump(st, open(sp, "w"))
        n4 = run()
        ex += 1
        s4 = json.load(open(sp))["runs"]["baseline_s1"]
        if not (n4 == 2 and s4["attempts"] == 2 and len(s4["crashes"]) == 1 and s4["status"] == "done"):
            ok = False
            why.append("mid-phase death not recovered as one crash (calls %d, %s)" % (n4, s4["attempts"]))
        open(os.path.join(code, "f.py"), "w").write("x=2\n")      # a frozen file changes
        ex += 1
        try:
            run()
            ok = False
            why.append("changed frozen code accepted")
        except runner.FrozenConfigChanged:
            pass
        os.remove(sp)                                                # the documented deliberate release
        ex += 1
        ok &= run() == 4
        ex += 1
        ok &= not os.path.exists(sp + ".tmp")
        ex += 1
        ok &= json.load(open(sp))["frozen"] != {}
    return Obs(bool(ok), ex, "; ".join(why[:3]) or "resume, crash recovery, frozen refusal and release verified")


@H.scenario("CT-07", "WP-17 campaign state: atomic save, restore from the backup copy, frozen code hash",
            brk="backup_ignored", min_exercised=7,
            oracle="a damaged state file is replaced by its .bak with stage statuses intact; a changed staged-code hash refuses unless accepted")
def ct07_campaign(brk):
    import campaign as C
    with tempfile.TemporaryDirectory() as td:
        ns = SimpleNamespace(root=td + "/camp", code=td + "/camp/code", venv=None, ab00_root=td + "/ab", poll=1, idle=1)
        cfg = C.Cfg(ns)
        os.makedirs(cfg.code)
        open(os.path.join(cfg.code, "f.py"), "w").write("x=1\n")
        stages = [{"id": "s1", "title": "t", "kind": "cmd", "requires": [], "on_fail": "pause", "est_hours": 0,
                   "blocked": None}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            c1 = C.Campaign(cfg, stages=stages)
            c1.set("s1", status="done", attempts=2)
            c1.set("s1", status="done", attempts=2)                  # a second save creates the .bak
        ex = 1
        ok = os.path.exists(cfg.state_path + ".bak") and not os.path.exists(cfg.state_path + ".tmp")
        open(cfg.state_path, "w").write("{trunc")                   # a power cut mid-write
        ctx = patched(C.Campaign, "valid_json", staticmethod(lambda p: False)) if brk else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(out):
            try:
                c2 = C.Campaign(cfg, stages=stages)
                restored = c2.st("s1")["status"] == "done" and c2.st("s1")["attempts"] == 2
            except SystemExit:
                restored = False
        ex += 1
        ok &= restored
        ex += 1
        ok &= "RESTORED campaign state" in out.getvalue()
        open(os.path.join(cfg.code, "f.py"), "w").write("x=2\n")
        ex += 1
        try:
            with contextlib.redirect_stdout(out):
                C.Campaign(cfg, stages=stages)
            ok = False
        except SystemExit:
            pass
        ex += 1
        with contextlib.redirect_stdout(out):
            c3 = C.Campaign(cfg, stages=stages, accept_code=True)
        ok &= c3.state["code_hash"] == c3.code_hash()
        ex += 1
        ok &= c3.st("s1")["status"] == "done"
        ex += 1
        ok &= os.path.isfile(cfg.state_path) and json.load(open(cfg.state_path))["stages"]["s1"]["attempts"] == 2
    return Obs(bool(ok), ex, "restored %s" % restored)


@H.scenario("CT-07", "WP-10 the Adam-expansion sidecar is consumed once; a bad one is removed, never retried",
            brk="sidecar_left_behind", min_exercised=6,
            oracle="write_sidecar -> consume_sidecar returns the dims and removes the file; a second consume is None; a corrupt sidecar yields None and is removed")
def ct07_sidecar(brk):
    from developmental_ai.world_model import growth_optim as GO
    from developmental_ai.world_model.growth import Dims
    old, new = Dims(8, 2, 2), Dims(16, 3, 3)
    with tempfile.TemporaryDirectory() as td:
        ctx = patched(GO.os, "remove", lambda p: None) if brk else contextlib.nullcontext()
        ex, ok = 0, True
        with ctx:
            p = GO.write_sidecar(td, old, new, "stamp")
            ex += 1
            ok &= os.path.isfile(p) and not os.path.exists(p + ".tmp")
            got = GO.consume_sidecar(td)
            ex += 1
            ok &= got == (old, new)
            ex += 1
            ok &= not os.path.exists(p)
            ex += 1
            ok &= GO.consume_sidecar(td) is None
            open(p, "w").write("{corrupt")
            ex += 1
            ok &= GO.consume_sidecar(td) is None
            ex += 1
            ok &= not os.path.exists(p)
    return Obs(bool(ok), ex, "sidecar consumed once")


# ============================================================================================
# CT-08 isolation and non-interference  (the most important contract here)
# ============================================================================================
@H.scenario("CT-08", "registry: an evaluator-tagged source or an evaluator-defined probe is rejected; none is registered anywhere",
            brk="evaluator_tag_accepted", min_exercised=60,
            oracle="RelationRegistry.register raises EvaluatorLeakError for the evaluator tag and for probes defined in the evaluator / detector modules; every channel of the default registry is learner-tagged with a learner-side probe")
def ct08_registry(brk):
    real = R.RelationRegistry.register
    if brk:
        def laundering(self, *a, **k):                # DEFECT: the tag is laundered at the entry point
            k["source_tag"] = R.LEARNER
            return real(self, *a, **k)
        ctx = patched(R.RelationRegistry, "register", laundering)
    else:
        ctx = contextlib.nullcontext()

    def probe_from(mod):
        def probe(loop):
            return 1.0
        probe.__module__ = mod
        return probe
    ex, ok, why = 0, True, []
    with ctx:
        reg = RelationRegistry({"enabled": True})
        for i, (kw, exc) in enumerate((
                ({"source_tag": "evaluator"}, EvaluatorLeakError),
                ({"source_tag": "engine"}, RelationError),
                ({"probe": probe_from("developmental_ai.evaluator.detector")}, EvaluatorLeakError),
                ({"probe": probe_from("developmental_ai.evaluator.scoring")}, EvaluatorLeakError),
                ({"probe": probe_from("developmental_ai.game_template.detectors.doom")}, EvaluatorLeakError))):
            ex += 1
            try:
                reg.register("ZZ-%02d" % i, "AAA", "B%d" % i, "p", "c", "s", "u", "t", **kw)
                ok = False
                why.append("accepted %s" % list(kw))
            except exc:
                pass
        ex += 1
        ok &= reg.channels() == []                    # none of the rejected channels exists
    full = R.build_default_registry({"enabled": False})
    for c in full.channels():
        ex += 1
        mod = getattr(c.probe, "__module__", "") or ""
        if c.source_tag != R.LEARNER or any(mod.startswith(p) for p in R._FORBIDDEN_PROBE_PREFIXES):
            ok = False
            why.append(c.name)
    return Obs(bool(ok), ex, "%d channels audited; %s" % (len(full.channels()), why[:3]))


def _py_sources(extra=()):
    out = []
    pkg = os.path.join(ROOT, "developmental_ai")
    for d, _, fs in os.walk(pkg):
        if "__pycache__" in d or os.path.join(pkg, "evaluator") in d:
            continue
        out += [(os.path.join(d, f), "learner") for f in fs if f.endswith(".py")]
    for f in sorted(os.listdir(ROOT)):
        if f.endswith(".py"):
            out.append((os.path.join(ROOT, f), "entry"))
    sd = os.path.join(ROOT, "scripts")
    if os.path.isdir(sd):
        out += [(os.path.join(sd, f), "entry") for f in sorted(os.listdir(sd)) if f.endswith(".py")]
    srcs = [(p, k, open(p, encoding="utf-8").read()) for p, k in out]
    return srcs + [(p, k, s) for (p, k, s) in extra]


@H.scenario("CT-08", "static import scan: no learner module or trainer entry point imports the evaluator, and it never imports the learner",
            brk="learner_module_imports_evaluator", min_exercised=240,
            oracle="the AST scan (statements, relative imports, literal importlib) finds no import of developmental_ai.evaluator from developmental_ai/ outside the package, the root entry scripts or scripts/; and no learner import inside developmental_ai/evaluator")
def ct08_imports(brk):
    TARGET = "developmental_ai.evaluator"

    def imports_of(src, modname, is_pkg=False):
        out, base = [], modname.split(".")
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                out += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    keep = len(base) - n.level + (1 if is_pkg else 0)
                    head = ".".join(base[:max(keep, 0)])
                    mod = (head + "." + n.module) if n.module else head
                else:
                    mod = n.module or ""
                out.append(mod)
                out += [mod + "." + a.name for a in n.names]
            elif isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if nm in ("import_module", "__import__") and n.args and isinstance(n.args[0], ast.Constant) \
                        and isinstance(n.args[0].value, str):
                    out.append(n.args[0].value)
        return out
    hit = lambda names: [x for x in names if x == TARGET or x.startswith(TARGET + ".")]      # noqa: E731
    ex, ok = 0, True
    for src, rel, want in (("from developmental_ai.evaluator import scoring", "developmental_ai.core.x", True),
                           ("import developmental_ai.evaluator.detector", "developmental_ai.core.x", True),
                           ("from ..evaluator import detector", "developmental_ai.core.x", True),
                           ("from .. import evaluator", "developmental_ai.core.x", True),
                           ("import importlib\nimportlib.import_module('developmental_ai.evaluator.x')",
                            "developmental_ai.core.x", True),
                           ("from developmental_ai.core import relations", "developmental_ai.core.x", False)):
        ex += 1
        ok &= bool(hit(imports_of(src, rel))) == want
    extra = []
    if brk:
        extra = [("developmental_ai/core/planted.py", "learner",
                  "import importlib\nimportlib.import_module('developmental_ai.evaluator.scoring')\n")]
    hits, n = [], 0
    for p, kind, src in _py_sources(extra):
        n += 1
        if os.path.isabs(p):
            rel = os.path.relpath(p, ROOT)
        else:
            rel = p
        mod = rel[:-3].replace(os.sep, ".").replace("/", ".")
        for h in hit(imports_of(src, mod, rel.endswith("__init__.py"))):
            hits.append((rel, h))
    ex += n
    ok &= not hits
    # the inverse boundary: detectors depend on dicts only (no learner import from inside the evaluator)
    evdir = os.path.join(ROOT, "developmental_ai", "evaluator")
    inv = []
    for d, _, fs in os.walk(evdir):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(d, f)
                mod = os.path.relpath(p, ROOT)[:-3].replace(os.sep, ".")
                names = imports_of(open(p, encoding="utf-8").read(), mod, f == "__init__.py")
                if brk and f == "scoring.py":
                    names = names + ["developmental_ai.core.relations"]
                inv += [(f, x) for x in names if x.startswith("developmental_ai.") and not (x == TARGET or x.startswith(TARGET + "."))]
                ex += 1
    ok &= not inv
    return Obs(bool(ok), ex, "%d modules scanned; learner->evaluator %s; evaluator->learner %s" % (
        n, [h[0] for h in hits[:3]], inv[:2]))


@H.scenario("CT-08", "read-only probes, autoscaler advice and logging frequency leave learner state, reward and every RNG unchanged",
            brk="probe_draws_rng", min_exercised=7,
            oracle="one probe pass per segment and six passes end with identical learner attributes, reward accumulators, autoscaler state and global RNG state")
def ct08_nonint(brk):
    from developmental_ai.core.attention_memory import LatentWindow

    def run(freq):
        torch.manual_seed(5)
        np.random.seed(5)
        random.seed(5)
        loop = SimpleNamespace(_last_ppo={"entropy": 0.7, "approx_kl": 0.02, "value_loss": 1.5},
                               _last_wm_metrics={"kl_raw": 2.5, "grad_norm": 1.2, "total": 3.0},
                               _am=SimpleNamespace(win=LatentWindow(2, 16, 4, "cpu")),
                               total_timesteps=1000, total_reward=3.5, mixer=SimpleNamespace(intrinsic_gain=1.0),
                               policy=torch.nn.Linear(3, 3), _wm_frames_trained=50,
                               replay_buffer=[0] * 10)
        reg = R.build_default_registry({"enabled": True, "connected_core": {"enabled": True},
                                        "WMC": {"WM": {"enabled": True}}})
        td = os.path.join(tempfile.gettempdir(), "cat07_never_created")
        a, _ = make_scaler(td, actuate=False, stepping=False, size="tiny", streams=4)
        w0 = {k: v.clone() for k, v in loop.policy.state_dict().items()}
        shown = []
        for seg in range(4):
            for _ in range(freq):
                reg.sample_all(loop)
                shown.append(reg.line())
                reg.end_segment()
                a.observe(over(seg))                        # advice mode: reads Signals, moves nothing
        snap = {k: copy.deepcopy(v) for k, v in vars(loop).items()
                if k not in ("policy", "_am", "_rel_state")}      # _rel_state: the registry's own telemetry scratch
        w1 = loop.policy.state_dict()
        return (snap, all(torch.equal(w0[k], w1[k]) for k in w0), global_rng(), a.state()["size"],
                a.streams, a.order.step_scale, hasattr(loop, "_rel_state"))
    ctx = patched(type(R.build_default_registry({}).get("rel.WMC.WM")), "observe",
                  (lambda real: (lambda self, v, x=None: (torch.rand(1), real(self, v, x))[1]))(
                      R.Channel.observe)) if brk else contextlib.nullcontext()
    with ctx:
        a = run(1)
        b = run(6)
    ex, ok = 0, True
    ex += 1
    ok &= a[0] == b[0]                                 # loop attributes, reward accumulators, counters
    ex += 1
    ok &= a[1] and b[1]                                # policy weights untouched
    ex += 1
    ok &= a[2][0] == b[2][0] and a[2][1] == b[2][1] and a[2][2] == b[2][2]
    ex += 1
    ok &= a[3:6] == b[3:6] == ("tiny", 4, 1.0)         # autoscaler advice moved nothing
    ex += 1
    ok &= a[0]["total_reward"] == 3.5 and a[0]["total_timesteps"] == 1000
    ex += 1
    ok &= a[6] is True                                 # the scratch the probes use is the only addition
    ex += 1
    ok &= a[2][0] == run(1)[2][0]                      # and the run is itself reproducible
    return Obs(bool(ok), ex, "attrs equal %s, rng equal %s" % (a[0] == b[0], a[2][0] == b[2][0]))


@H.scenario("CT-08", "the RED oracle sink at the loop's real entry touches evaluation fields only",
            brk="oracle_feeds_learner_state", min_exercised=5,
            oracle="DevelopmentalAI._oracle_observe fed wildly different info['oracle'] values changes only the _oracle_* evaluation fields and the oracle telemetry key, never any other loop attribute or RNG")
def ct08_oracle(brk):
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    real = DevelopmentalAI._oracle_observe

    def sink(self, infos, n):
        real(self, infos, n)
        if brk:                                       # DEFECT: the evaluation number leaks into the reward scale
            self._icm_base_scale = 1.0 / (1.0 + float(np.mean(self._oracle_drift or [0.0])))

    def run(scale):
        fake = SimpleNamespace(_explore_tracker=None, _oracle_enabled=True, _oracle_origin=None,
                               _oracle_travel=0.0, _oracle_last=None, _oracle_drift=deque(maxlen=100),
                               _oracle_steps=0, _oracle_every=2, total_timesteps=10, training_metrics={},
                               _ltel=None, _icm_base_scale=1.0, total_reward=1.0, coef={"g": 1.0},
                               policy_w=np.ones(3))
        torch.manual_seed(1)
        np.random.seed(1)
        before = {k: copy.deepcopy(v) for k, v in vars(fake).items()}
        rng0 = global_rng()
        for t in range(6):
            infos = [{"oracle": {"true_position": [t * scale, 0.0, 2.0 * t * scale]},
                      "world": {"dr_x": float(t), "dr_z": float(2 * t)}}]
            sink(fake, infos, 1)
        after = vars(fake)

        def same(x, y):
            try:
                return bool(np.array_equal(np.asarray(x, dtype=object), np.asarray(y, dtype=object))) \
                    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) else x == y
            except Exception:
                return False
        changed = sorted(k for k in after if k not in before or not same(after[k], before[k]))
        return changed, global_rng() == rng0, dict(after)
    c1, r1, a1 = run(1.0)
    c2, r2, a2 = run(1000.0)
    allowed = {"_oracle_origin", "_oracle_travel", "_oracle_last", "_oracle_drift", "_oracle_steps", "training_metrics"}
    ex, ok = 1, set(c1) <= allowed and set(c2) <= allowed
    ex += 1
    ok &= r1 and r2
    ex += 1
    ok &= a1["_icm_base_scale"] == a2["_icm_base_scale"] == 1.0 and a1["coef"] == a2["coef"]
    ex += 1
    ok &= a1["total_reward"] == a2["total_reward"] == 1.0
    ex += 1
    ok &= len(a2["_oracle_drift"]) > 0 and a1["_oracle_drift"] != a2["_oracle_drift"]      # the sink did read it
    return Obs(bool(ok), ex, "changed fields: %s / %s" % (c1, c2))


if __name__ == "__main__":
    sys.exit(H.finish())
