"""CAT-03 within-group contracts CT-01..CT-08: planning, uncertainty and risk (GOV-013, ASS-012 3).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat03_smoke.py

Members (tests/categorical_manifest.py): WP-4 (core/planning.py PlanningPhase/Arbiter,
core/uncertainty.py EnsembleHeads/CalibrationMonitor), WP-11 (core/risk.py RiskEstimator), RW-5
(core/relations_executive.py working-memory gate), the coefficient registry (core/coefficients.py),
the relation registry (core/relations.py) and RW-1's consumer of the calibration validity
(core/relations_modulation.py).

Every fixture drives the REAL module objects (a tiny CPU world model, a real policy, a real
multi-stream replay buffer, real AttentionMemory, Consolidator, RiskEstimator and registries).
A fixture injects numbers, NaN/inf, delays and faults; it never supplies a skill or a task meaning,
never reads the evaluator and never touches a game engine. The loop object is a SimpleNamespace that
only carries the attributes the wiring modules read (`_plan`, `_am`, `_cons`, `_rk`, `infra`).

Each scenario runs twice (tests/categorical_lib.py): as built, and with ONE named wiring defect
injected into this harness (never into production code); the defective run must fail. A scenario
that finds a real production defect is left FAILING and reported, not weakened.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import math
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs  # noqa: E402

from developmental_ai.core import relations_executive as executive  # noqa: E402
from developmental_ai.core import relations_modulation as modulation  # noqa: E402
from developmental_ai.core.attention_memory import AttentionMemory  # noqa: E402
from developmental_ai.core.coefficients import OwnershipError, registry_for  # noqa: E402
from developmental_ai.core.planning import Arbiter, PlanningPhase  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EvaluatorLeakError, RelationError, RelationRegistry, build_default_registry)
from developmental_ai.core.risk import RiskEstimator, risk_dim  # noqa: E402
from developmental_ai.core.uncertainty import CalibrationMonitor, EnsembleHeads, spearman  # noqa: E402
from developmental_ai.policy.actor_critic import StandaloneActorCritic  # noqa: E402
from developmental_ai.policy.consolidation import Consolidator  # noqa: E402
from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer  # noqa: E402
from developmental_ai.world_model.rssm import WorldModel  # noqa: E402

torch.set_num_threads(1)
H = Harness("cat03", "Planning, uncertainty and risk")
CPU = torch.device("cpu")
A_DIM = 4
STEP = 0.05


# ---------------------------------------------------------------------------------- helpers
class Chk:
    """Collects sub-checks so one failing contract never hides another one's evidence."""

    def __init__(self):
        self.fails, self.n = [], 0

    def __call__(self, cond, msg, n=1):
        self.n += n
        if not cond:
            self.fails.append(msg)

    def obs(self, ok_note):
        return Obs(not self.fails, self.n, "; ".join(self.fails) if self.fails else ok_note)


@contextlib.contextmanager
def patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def mk_stack(seed=0, ens=True, min_n=200, distill=0.0):
    """A tiny real world model + real rssm-arch policy + 2-stream replay + a PlanningPhase."""
    torch.manual_seed(seed)
    wm = WorldModel(obs_dim=3 * 16 * 16, action_dim=A_DIM, deterministic_size=32, stochastic_size=4,
                    stochastic_classes=4, hidden_dim=32, pixel_obs=True, image_size=16,
                    recon_reduction="sum", inverse_input="embed")
    L = wm.rssm.latent_dim
    pol = StandaloneActorCritic(obs_dim=L, action_dim=A_DIM, hidden_dim=32, arch="rssm",
                                enc_dim=L, device=CPU)
    buf = MultiStreamReplayBuffer(2, 4000, 3 * 16 * 16, A_DIM)
    rng = np.random.RandomState(seed)
    for s in range(2):
        for _ in range(240):
            buf.add(rng.rand(3 * 16 * 16).astype(np.float32), int(rng.randint(A_DIM)), 0.0, False, stream=s)
    cfg = {"rows": 4, "seq_len": 8, "planner": {"candidates": 8, "horizon": 2, "iterations": 1},
           "calibration": {"min_n": min_n},
           "ensemble": {"enabled": ens, "hidden": 16, "heads": 3},
           "distill": {"weight": distill}}
    P = PlanningPhase(cfg, A_DIM, L, CPU)
    return wm, pol, buf, P


def mk_attention(streams=2, dim=8, window=4, tmp=None):
    torch.manual_seed(5)
    am = AttentionMemory({"window": window, "out_dim": 8, "d_model": 16, "heads": 2, "capacity": 16,
                          "cost_report": False, "store_every": 1},
                         streams, CPU, os.path.join(tmp or tempfile.mkdtemp(), "attn.pt"))
    am.observe(torch.zeros(streams, dim), [False] * streams)      # builds the net and the window
    am.win.n = [0] * streams
    return am


def mk_loop(tmp=None, with_risk=False):
    loop = SimpleNamespace(_plan=SimpleNamespace(last={}), _am=mk_attention(tmp=tmp),
                           _cons=Consolidator({}), infra=SimpleNamespace(last_ledger_segment={}))
    if with_risk:
        loop._rk = RiskEstimator({"horizon": 4}, 1, CPU)
    return loop


def rel_cfg(*pairs, **extra):
    cfg = {"enabled": True}
    for a, b in pairs:
        cfg.setdefault(a, {})[b] = {"enabled": True}
    cfg.update(extra)
    return cfg


MOD = [("LC", "NE"), ("NE", "AROUS"), ("NE", "ATT"), ("NE", "CONS")]
COEFS = ("explore.uncertainty_scale", "attn.temperature_scale", "consol.strength_scale")


def coef_values(loop):
    c = registry_for(loop)
    return {n: c.get(n) for n in COEFS}


def set_cal(loop, conf, ok=1.0):
    loop._plan.last = {"calibration_conf": float(conf), "policy_calibration_ok": float(ok)}


def warm_modulation(reg, loop, n=8):
    for k in range(n):
        set_cal(loop, 0.70 + 0.04 * (k % 2))
        reg.wire_all(loop)


def reseed(n):
    """Torch AND numpy (the replay buffer samples with numpy's global RNG)."""
    torch.manual_seed(n)
    np.random.seed(n)


def phash(m):
    return torch.cat([p.detach().flatten().clone() for p in m.parameters()])


# ---------------------------------------------------------------------------------- CT-01
@H.scenario("CT-01", "valid evidence reaches planner, modulation and risk consumers with right sign",
            brk="ensemble_stale_modulation_unwired_risk_head_stale", min_exercised=20,
            oracle="arbiter input rises with injected head disagreement; coefficients and consumer "
                   "state move in the declared direction/bound; risk feature rises with risk, "
                   "consumer reads it")
def ct01(brk):
    c = Chk()
    # (a) producer: ensemble disagreement -> consumer: the Arbiter's actual input in PlanningPhase.run
    got = {}
    for arm in ("base", "pert"):
        wm, pol, buf, P = mk_stack()
        for g in P.ens.opt.param_groups:
            g["lr"] = 0.0                                       # hold the ensemble fixed during run()
        stale = copy.deepcopy(P.ens)
        if arm == "pert":
            with torch.no_grad():
                P.ens.heads[0][4].bias.add_(1.0)                # one head now disagrees with the rest
            if brk:
                P.ens = stale                                   # consumer wired to a stale model version
        seen = []
        orig = P.arbiter.decide

        def spy(conf, dis, pok, mok, bok, _o=orig, _s=seen):
            _s.append((list(conf), list(dis)))
            return _o(conf, dis, pok, mok, bok)
        P.arbiter.decide = spy
        reseed(123)
        P.run(pol, wm, buf, CPU)
        got[arm] = seen[-1]
    cb, db = got["base"]
    cp, dp = got["pert"]
    c(len(cb) == len(db) == 4 and len(dp) == 4, "arbiter input is not one value per sampled row")
    c(all(0.0 <= x <= 1.0 for x in cb + cp) and all(math.isfinite(x) and x >= 0 for x in db + dp),
      "arbiter input out of range")
    c(cb == cp, "policy confidence changed when only the ensemble was perturbed", n=4)
    c(np.mean(dp) > 5.0 * max(np.mean(db), 1e-9),
      f"perturbed ensemble disagreement did not reach the arbiter (base {np.mean(db):.4f} pert {np.mean(dp):.4f})",
      n=4)

    # (b) producer: calibrated confidence -> consumers: three coefficients and the actual consumer state
    loop = mk_loop()
    reg = build_default_registry(rel_cfg(*MOD))
    v0 = None
    if not brk:
        warm_modulation(reg, loop)
    v0 = coef_values(loop)
    path = [dict(v0)]
    for _ in range(4):                                          # evidence turns uncertain (conf 0.05)
        set_cal(loop, 0.05)
        if not brk:
            reg.wire_all(loop)
        path.append(coef_values(loop))
    hi = path[-1]
    steps_ok = all(abs((path[i + 1][n] or 1.0) - (path[i][n] or 1.0)) <= STEP + 1e-9
                   for i in range(len(path) - 1) for n in COEFS)
    c(steps_ok, "a coefficient moved by more than max_step in one update", n=12)
    c((hi["explore.uncertainty_scale"] or 1.0) > 1.0 + STEP - 1e-9, "entropy-target scale did not rise with uncertainty")
    c((hi["attn.temperature_scale"] or 1.0) < 1.0 - STEP + 1e-9, "attention temperature did not fall (sharper) with uncertainty")
    c((hi["consol.strength_scale"] or 1.0) < 1.0 - STEP + 1e-9, "consolidation strength did not fall with uncertainty")
    c(abs(loop._am.net.temperature - (hi["attn.temperature_scale"] or 1.0)) < 1e-9,
      "AttentionMemory net did not receive the temperature coefficient", n=1)
    c(abs(loop._cons.strength - (hi["consol.strength_scale"] or 1.0)) < 1e-9,
      "Consolidator did not receive the strength coefficient", n=1)
    for _ in range(14):                                         # evidence turns confident (conf 0.999)
        set_cal(loop, 0.999)
        if not brk:
            reg.wire_all(loop)
    lo = coef_values(loop)
    c((lo["explore.uncertainty_scale"] or 1.0) < (hi["explore.uncertainty_scale"] or 1.0) - STEP,
      "entropy-target scale did not fall back when confidence returned")
    c((lo["attn.temperature_scale"] or 1.0) > (hi["attn.temperature_scale"] or 1.0) + STEP,
      "attention temperature did not rise back when confidence returned")

    # (c) producer: continue-head risk -> consumers: feature vector and the RW-3 AMY consumer
    from developmental_ai.core.relations_consolidation import risk_feature
    for hc, want in (({}, 1), ({"head": {"enabled": True}}, 2)):
        c(risk_dim(hc) == want, f"risk_dim {hc} != {want}")
    feats = {}
    for arm in ("base", "pert"):
        torch.manual_seed(2)
        rk = RiskEstimator({"horizon": 4}, 1, CPU)
        head = nn.Linear(6, 1)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()                                   # continue 0.5 -> risk 0.5
        stale_head = copy.deepcopy(head)
        lat = torch.zeros(1, 6)
        for _ in range(6):
            rk.observe(lat, head, [False], [False])
        if arm == "pert":
            with torch.no_grad():
                head.bias.fill_(-3.0)                           # continue 0.047 -> risk 0.953
            use = stale_head if brk else head
            for _ in range(2):
                rk.observe(lat, use, [False], [False])
        lp = SimpleNamespace(_rk=rk)
        feats[arm] = (float(rk.feature()[0]), risk_feature(lp), rk.feature().shape)
    c(feats["base"][2] == feats["pert"][2] == (1,), "risk feature width is not risk_dim")
    c(feats["pert"][0] > feats["base"][0] + 0.1,
      f"risk feature did not rise with the continue head risk ({feats['base'][0]:.3f} -> {feats['pert'][0]:.3f})", n=2)
    c(feats["pert"][1] == feats["pert"][0], "RW-3 AMY consumer does not read the risk feature index 0", n=1)
    return c.obs("arbiter input, three coefficients, two consumer states and the risk feature all follow the producers")


# ---------------------------------------------------------------------------------- CT-02
@H.scenario("CT-02", "cut/neutral parity: a cut channel or group leaves neutral behaviour and unrelated state",
            brk="cut_switch_ignored_by_consumer", min_exercised=14,
            oracle="channel off -> send() is the neutral constant, no coefficient claimed, window/temperature/"
                   "strength exactly the never-wired values; planning without ensemble falls back to habit "
                   "and leaves the policy untouched")
def ct02(brk):
    c = Chk()
    pristine = lambda lp: (tuple(sorted((r["name"], r["owner"]) for r in registry_for(lp).table())),
                           lp._am.net.temperature, lp._cons.strength)
    ref = pristine(mk_loop())
    # group cut: master on, no wired channel on -> wire_all is a no-op
    loop = mk_loop()
    reg = build_default_registry({"enabled": True})
    for k in range(4):
        set_cal(loop, 0.1 + 0.2 * k)
        reg.wire_all(loop)
    c(pristine(loop) == ref, "group cut changed coefficients or consumer state", n=4)
    # master off: every wired channel returns its neutral constant exactly
    off = build_default_registry({"enabled": False})
    for name in ("rel.LC.NE", "rel.NE.AROUS", "rel.NE.ATT", "rel.NE.CONS", "rel.PFC.WM"):
        c(off.send(name, 1.7) == off.get(name).neutral and off.get(name).n == 0,
          f"{name} OFF is not the neutral constant", n=1)
    # per-channel cut: only NE.ATT on
    loop = mk_loop()
    reg = build_default_registry(rel_cfg(("NE", "ATT")))
    cutter = (patched(RelationRegistry, "is_on",
                      lambda self, n, _o=RelationRegistry.is_on: True if n == "rel.NE.AROUS" else _o(self, n))
              if brk else contextlib.nullcontext())
    with cutter:
        for k in range(4):
            set_cal(loop, 0.2 + 0.2 * (k % 2))
            reg.wire_all(loop)
    creg = registry_for(loop)
    c(creg.owner_of("attn.temperature_scale") == modulation.OWNER, "the enabled channel did not claim its coefficient")
    c(creg.owner_of("explore.uncertainty_scale") is None, "cut NE.AROUS still claimed its coefficient", n=1)
    c(creg.owner_of("consol.strength_scale") is None and loop._cons.strength == 1.0,
      "cut NE.CONS touched the consolidation strength", n=1)
    c(reg.get("rel.NE.AROUS").n == 0 and reg.get("rel.NE.ATT").n > 0, "cut channel recorded samples", n=2)
    # RW-5 cut: no gate installed -> the window is exactly the never-gated window
    tmp = tempfile.mkdtemp()
    a, b = mk_attention(tmp=tmp), mk_attention(tmp=tmp)
    reg5 = build_default_registry({"enabled": True})
    lp5 = SimpleNamespace(_am=b)
    reg5.wire_all(lp5)
    rng = np.random.RandomState(1)
    for _ in range(10):
        z = torch.tensor(rng.randn(2, 8), dtype=torch.float32)
        a.observe(z, [False, False])
        b.observe(z, [False, False])
    c(b.push_gate is None and torch.equal(a.win.buf, b.win.buf) and a.win.n == b.win.n
      and np.array_equal(a.vec, b.vec), "RW-5 cut changed the attention window", n=3)
    # planning cut: no ensemble -> habit, and the policy is not written
    wm, pol, buf, P = mk_stack(ens=False)
    before = phash(pol.actor)
    out = P.run(pol, wm, buf, CPU)
    c(P.ens is None and out["dis"] == 0.0 and out["model_calibration_ok"] == 0.0 and out["plan_rows"] == 0.0,
      f"ensemble cut did not fall back to habit: {out.get('plan_rows')}", n=1)
    c(torch.equal(before, phash(pol.actor)), "planning with distillation off wrote policy parameters", n=1)
    return c.obs("cut channels are exactly neutral and unrelated state is unchanged")


# ---------------------------------------------------------------------------------- CT-03
class NoGuardArbiter(Arbiter):
    """Disposable copy of Arbiter.decide with the validity check removed."""

    def decide(self, conf, disagreement, policy_ok, model_ok, budget_ok):
        n = len(conf)
        if len(disagreement) != n:
            raise ValueError("arbitration requires one disagreement per confidence")
        out = [False] * n
        for d in disagreement:
            self._hist.append(float(d))
        thr = None
        if self._hist:
            s = sorted(self._hist)
            thr = s[min(len(s) - 1, int(self.trust_q * len(s)))]
        for i in range(n):
            self.seen += 1
            if not budget_ok:
                self.why["no_budget"] += 1
            elif self.require_calibrated and not (policy_ok and model_ok):
                self.why["uncalibrated"] += 1
            elif conf[i] > self.conf_max:
                self.why["confident"] += 1
            elif thr is not None and disagreement[i] > thr:
                self.why["distrust"] += 1
            else:
                out[i] = True
                self.why["plan"] += 1
        return out


def lax_policy_uncertainty(loop):
    P = getattr(loop, "_plan", None)
    last = getattr(P, "last", None) or {}
    c = last.get("calibration_conf")
    return None if c is None else 1.0 - float(c)


@H.scenario("CT-03", "missing, NaN/inf, constant or out-of-range evidence is refused with no state update",
            brk="validity_guards_removed", min_exercised=25,
            oracle="arbiter, calibration and modulation refuse invalid rows, keep their history/z-score "
                   "unchanged, and a legitimate zero is still accepted")
def ct03(brk):
    c = Chk()
    nan, inf = float("nan"), float("inf")
    # arbiter
    a = (NoGuardArbiter if brk else Arbiter)({})
    c(a.decide([0.2], [0.1], True, True, True) == [True], "valid row was refused")
    h0 = len(a._hist)
    bad_rows = [([nan], [0.1]), ([1.1], [0.1]), ([-0.1], [0.1]), ([0.2], [inf]), ([0.2], [nan]), ([0.2], [-1.0])]
    for conf, dis in bad_rows:
        c(a.decide(conf, dis, True, True, True) == [False], f"invalid row {conf},{dis} was trusted")
    c(len(a._hist) == h0, "invalid rows entered the trust history", n=1)
    c(a.decide([0.0], [0.0], True, True, True) == [True], "a legitimate zero was treated as absence")
    c(a.decide([0.2], [0.1], True, False, True) == [False], "absent/failed model channel still planned")
    try:
        a.decide([0.2, 0.3], [0.1], True, True, True)
        c(False, "length mismatch accepted")
    except ValueError:
        c(True, "")
    # calibration (real PlanningPhase._calibrate) on a real buffer
    wm, pol, buf, P = mk_stack(min_n=8)
    r = P._calibrate(pol, wm, buf, CPU)
    c(P.mon.n("policy") == 16 and "calibration_conf" in r, "healthy window was not recorded", n=2)
    n_pol, T0 = P.mon.n("policy"), P.T
    with patched(P, "_logits", lambda p, x: torch.full((x.shape[0], A_DIM), nan)):
        r = P._calibrate(pol, wm, buf, CPU)
    c(r["policy_calibration_ok"] == 0.0 and P.mon.n("policy") == n_pol and P.T == T0,
      "NaN policy logits updated the monitor or the temperature")
    n_mod = P.mon.n("model")
    with patched(P.ens, "disagreement", lambda l, a_: torch.full((l.shape[0],), nan)):
        r = P._calibrate(pol, wm, buf, CPU)
    c(r["model_calibration_ok"] == 0.0 and P.mon.n("model") == n_mod, "NaN disagreement updated the model channel")
    n_pol, n_mod = P.mon.n("policy"), P.mon.n("model")
    with patched(buf, "newest_sequences", lambda *a_, **k: None):
        r = P._calibrate(pol, wm, buf, CPU)
    c(r == {"policy_calibration_ok": 0.0, "model_calibration_ok": 0.0}
      and P.mon.n("policy") == n_pol and P.mon.n("model") == n_mod,
      "absent sample did not fall back explicitly")
    wm_c, pol_c, buf_c, Pc = mk_stack(min_n=8)                   # fresh monitor: only constant evidence
    with patched(Pc.ens, "disagreement", lambda l, a_: torch.full((l.shape[0],), 0.5)), \
            patched(Pc.ens, "error", lambda l, a_, n_: torch.full((l.shape[0],), 0.25)):
        for _ in range(3):
            r = Pc._calibrate(pol_c, wm_c, buf_c, CPU)
    c(Pc.mon.n("model") >= 8 and r["model_calibration_ok"] == 0.0, "constant ensemble evidence was trusted")
    mon = CalibrationMonitor(window=5, min_n=3)
    mon.note("m", [nan, inf, 1.0], [1.0, 2.0, nan])
    c(mon.n("m") == 0 and math.isnan(spearman([0, 0, 0], [0, 0, 0])), "monitor accepted non-finite pairs")
    # modulation consumer of the validity
    loop = mk_loop()
    reg = build_default_registry(rel_cfg(*MOD))
    warm_modulation(reg, loop)
    z = reg.state["mod"]["z"]
    n0, snap = z.n, coef_values(loop)
    with (patched(modulation, "policy_uncertainty", lax_policy_uncertainty) if brk else contextlib.nullcontext()):
        for bad in ({"calibration_conf": nan, "policy_calibration_ok": 1.0},
                    {"calibration_conf": inf, "policy_calibration_ok": 1.0},
                    {"calibration_conf": 1.5, "policy_calibration_ok": 1.0},
                    {"calibration_conf": 0.5, "policy_calibration_ok": 0.0},
                    {"conf": 0.5}, {}):
            loop._plan.last = bad
            reg.wire_all(loop)
            c(z.n == n0 and math.isfinite(z.m), f"invalid calibration {bad} updated the z-score")
    c(all(math.isfinite(v) for v in coef_values(loop).values()), "a coefficient became non-finite", n=3)
    return c.obs("invalid evidence refused with no state update; legitimate zero accepted")


# ---------------------------------------------------------------------------------- CT-04
@H.scenario("CT-04", "one owner, one write per segment, bounded slew, product of declared factors",
            brk="segment_wired_twice", min_exercised=30,
            oracle="each coefficient owned by rw1_modulation, written exactly once per segment, moved <= max_step, "
                   "equal to the independently recomputed slewed product; a rival writer is refused")
def ct04(brk):
    c = Chk()
    loop = mk_loop(with_risk=True)
    reg = build_default_registry(rel_cfg(*MOD, ("EMO", "CONS"), ("AMY", "CONS")))
    creg = registry_for(loop)
    head = nn.Linear(6, 1)
    with torch.no_grad():
        head.weight.zero_()
    lo_hi = {"explore.uncertainty_scale": (0.7, 1.3), "attn.temperature_scale": (1 / 1.3, 1 / 0.7),
             "consol.strength_scale": (0.25, 4.0)}
    rng = np.random.RandomState(3)
    prev = {n: 1.0 for n in COEFS}
    writes = {n: 0 for n in COEFS}
    for seg in range(12):
        set_cal(loop, float(np.clip(0.7 - 0.06 * seg + 0.05 * rng.randn(), 0.01, 0.99)))
        with torch.no_grad():
            head.bias.fill_(float(rng.randn() * 2))
        loop._rk.observe(torch.zeros(1, 6), head, [False], [False])
        loop.infra.last_ledger_segment = {"total": float(rng.rand())}
        reg.wire_all(loop)
        if brk:
            reg.wire_all(loop)                                   # the segment hook runs twice (duplicated write)
        tab = {r["name"]: r for r in creg.table()}
        for n in COEFS:
            if n not in tab:
                c(False, f"{n} was never claimed")
                continue
            row = tab[n]
            c(row["owner"] == modulation.OWNER, f"{n} owner {row['owner']}")
            c(row["writes"] - writes[n] == 1, f"{n} written {row['writes'] - writes[n]}x in segment {seg}")
            writes[n] = row["writes"]
            c(abs(row["value"] - prev[n]) <= STEP + 1e-9 and lo_hi[n][0] - 1e-9 <= row["value"] <= lo_hi[n][1] + 1e-9,
              f"{n} left its slew/bound ({prev[n]:.3f}->{row['value']:.3f})")
            prev[n] = row["value"]
        fac = reg.factors.get("consol", {})
        if seg >= 1 and not brk:
            c(set(fac) == {"NM-06", "rel.EMO.CONS", "rel.AMY.CONS"}, f"consol factors {sorted(fac)}")
        c(abs(loop._cons.strength - prev["consol.strength_scale"]) < 1e-9, "consumer strength != coefficient")
    # recompute the consolidation target independently for the last segment
    fac = reg.factors["consol"]
    prod = 1.0
    for v in fac.values():
        prod *= v
    c(0.25 <= prod <= 4.0, "factor product out of declared bound")
    # rival writers are refused, value unchanged
    v0, nref = creg.get("consol.strength_scale"), len(creg.refused)
    for fn in (lambda: creg.set("consol.strength_scale", "wp9_adaptive", 3.0),
               lambda: creg.claim("explore.uncertainty_scale", "rw7_motivation", 1.0, 0.5, 2.0)):
        try:
            fn()
            c(False, "a rival writer was accepted")
        except OwnershipError:
            c(True, "")
    c(creg.get("consol.strength_scale") == v0 and len(creg.refused) == nref + 2, "rival refusal not recorded/neutral")
    # a NaN write does not move the coefficient
    c(creg.set("consol.strength_scale", modulation.OWNER, float("nan")) == v0, "NaN write moved a coefficient")
    return c.obs("single owner, one bounded write per segment, consumer equals coefficient")


# ---------------------------------------------------------------------------------- CT-05
@H.scenario("CT-05", "stream interleave and episode boundaries do not leak labels or window history",
            brk="stream_ids_swapped", min_exercised=14,
            oracle="a death in stream 1 labels only stream 1's states; a restart labels nothing; "
                   "an episode end in stream 0 clears only stream 0's window")
def ct05(brk):
    c = Chk()
    sw = (lambda x: list(reversed(x))) if brk else (lambda x: list(x))
    # risk labels (two streams, distinct raw risk per stream)
    rk = RiskEstimator({"horizon": 4, "capacity": 200}, 2, CPU)
    head = lambda z: -z[:, :1]                                    # risk = sigmoid(z0)
    lat = torch.tensor([[-3.0, 0, 0], [3.0, 0, 0]])               # stream 0 low risk, stream 1 high risk
    for _ in range(7):
        rk.observe(lat, head, [False, False], [False, False])
    rk.observe(lat, head, sw([False, True]), sw([False, True]))   # stream 1 dies
    n = min(rk._np, rk.cap)
    pos = rk._y[:n] > 0.5
    c(int(pos.sum()) == 4 and int(rk._deaths) == 1, f"expected the 4 open states of one stream labelled, got {int(pos.sum())}")
    hi = 1 / (1 + math.exp(-3.0))
    c(bool(np.all(np.abs(rk._r[:n][pos] - hi) < 1e-5)), "a stream-0 state was labelled by a stream-1 death", n=int(pos.sum()))
    c(sorted(rk._lead[:n][pos].tolist()) == [1, 2, 3, 4], "lead times are not 1..H", n=1)
    npos = int(pos.sum())
    for _ in range(3):
        rk.observe(lat, head, [False, False], [False, False])
    rk.observe(lat, head, sw([False, False]), sw([True, False]))  # stream 0 restarts: censored
    c(int((rk._y[:min(rk._np, rk.cap)] > 0.5).sum()) == npos and len(rk._open[0]) == 0 and len(rk._open[1]) > 0,
      "a restart was labelled as a death or cleared the other stream", n=3)
    # attention window: the executive gate acts on stream 0 only; ended clears only that stream
    tmp = tempfile.mkdtemp()
    am = mk_attention(tmp=tmp)
    loop = SimpleNamespace(_am=am)
    reg = build_default_registry(rel_cfg(("PFC", "WM"), executive={"min_n": 3, "max_skip": 4}))
    reg.wire_all(loop)
    c(am.push_gate is not None, "RW-5 gate was not installed by the real wiring")
    calls = []
    inner = am.push_gate
    am.push_gate = lambda z0, win: (calls.append(1), inner(z0, win))[1]
    rng = np.random.RandomState(4)
    held = 0
    for k in range(14):
        z = torch.tensor(rng.randn(2, 8), dtype=torch.float32)
        if k >= 6:
            z[0] = torch.ones(8)                                  # stream 0 stops changing -> gate holds
        am.observe(z, [False, False])
        held += int(not am.pushed0)
        c(torch.equal(am.win.buf[1, -1], z[1]), "stream 1 latent was held out by stream 0's gate", n=1)
    c(len(calls) == 14, "gate was not consulted exactly once per step", n=1)
    c(held >= 1 and held < 8, f"gate never held a repeated latent / never admitted (held {held})", n=1)
    z = torch.tensor(rng.randn(2, 8), dtype=torch.float32)
    keep1 = am.win.buf[1].clone()
    ncalls = len(calls)
    am.observe(z, sw([True, False]))                              # stream 0's episode ends
    c(len(calls) == ncalls, "gate consulted on an episode end", n=1)
    # GOV-015 F-08: the latent at an episode end belongs to the episode that ended, so the window
    # restarts EMPTY and the next latent (the new episode's first) starts it. (Before the fix this
    # asserted n == 1 holding the old episode's last latent, which put it in the new window.)
    c(am.win.n[0] == 0, "stream 0 window not emptied at the boundary", n=1)
    c(am.win.n[1] == 4 and torch.equal(am.win.buf[1, :-1], keep1[1:]),
      "stream 1 window was cleared by stream 0's boundary", n=1)
    z_new = torch.tensor(rng.randn(2, 8), dtype=torch.float32)
    am.observe(z_new, [False, False])
    c(am.win.n[0] == 1 and torch.equal(am.win.buf[0, -1], z_new[0]),
      "stream 0 window did not restart with the new episode's first latent", n=1)
    # ensemble training pairs: (state_t, a_t) -> state_{t+1}, with the action replayed at its own time index
    wm, pol, buf, P = mk_stack()
    cap = {}
    orig_ss = buf.sample_sequences

    def ss(*a_, **k_):
        b = orig_ss(*a_, **k_)
        cap["b"] = b
        if brk:                                                  # labels delayed by one step
            b = dict(b)
            b["actions"] = torch.roll(b["actions"], 1, 1)
        return b
    buf.sample_sequences = ss
    calls = []
    orig_ts = P.ens.train_step
    P.ens.train_step = lambda l_, a_, n_: (calls.append((l_.clone(), a_.clone(), n_.clone())), orig_ts(l_, a_, n_))[1]
    P.run(pol, wm, buf, CPU)
    b = cap["b"]
    T = b["actions"].shape[1]
    truth = np.stack([buf.streams[int(s)].actions[(int(i) + np.arange(T)) % buf.streams[int(s)].capacity]
                      for s, i in b["start_indices"]])
    lat_, act_, nxt_ = calls[0]
    Bn, L = b["actions"].shape[0], lat_.shape[-1]
    c(torch.equal(act_.reshape(Bn, T - 1, -1), torch.from_numpy(truth[:, :-1, :A_DIM]).float()),
      "ensemble was trained with actions not aligned to the replayed time index", n=Bn)
    c(torch.equal(nxt_.reshape(Bn, T - 1, L)[:, :-1], lat_.reshape(Bn, T - 1, L)[:, 1:]),
      "ensemble target is not the next state of the input state", n=Bn)
    return c.obs("labels and window state stay inside their stream and episode; action delay exact")


@H.scenario("CT-06", "dependency failure then recovery: release bound, bad batch, wiring fault, gate fault",
            brk="release_path_missing", min_exercised=18,
            oracle="source loss releases every coefficient in ceil(dev/step) updates; a transient NaN world-model "
                   "batch does not poison the ensemble for good; a raising wiring module is released and the "
                   "error is observable; a raising window gate opens")
def ct06(brk):
    c = Chk()
    # (1) bounded release of slew-limited coefficients and recovery
    loop = mk_loop()
    reg = build_default_registry(rel_cfg(*MOD))
    warm_modulation(reg, loop)
    for _ in range(6):
        set_cal(loop, 0.0)
        reg.wire_all(loop)
    dev = {n: abs((v or 1.0) - 1.0) for n, v in coef_values(loop).items()}
    c(max(dev.values()) > 0.15, f"fixture did not displace the coefficients ({dev})")
    bound = max(math.ceil(d / STEP - 1e-9) for d in dev.values())
    loop._plan.last = {}                                          # the calibration source vanishes
    steps = 0
    real_apply = modulation.apply
    gate = (lambda lp, rg: None if modulation.policy_uncertainty(lp) is None else real_apply(lp, rg)) if brk else real_apply
    with patched(modulation, "apply", gate):
        while steps < bound + 3 and any(abs((v or 1.0) - 1.0) > 1e-9 for v in coef_values(loop).values()):
            reg.wire_all(loop)
            steps += 1
    c(steps <= bound, f"release took {steps} updates, bound {bound} (coefficients {coef_values(loop)})", n=steps)
    c(abs(loop._am.net.temperature - 1.0) < 1e-9 and abs(loop._cons.strength - 1.0) < 1e-9,
      "consumer state stranded away from neutral", n=2)
    for k in range(8):                                            # valid, varying evidence returns
        set_cal(loop, 0.9 if k % 2 else 0.1)
        reg.wire_all(loop)
    set_cal(loop, 0.02)
    for _ in range(3):
        reg.wire_all(loop)
    c(any(abs((v or 1.0) - 1.0) > 1e-9 for v in coef_values(loop).values()), "modulation did not resume after recovery")
    # (2) arbiter budget gate re-opens
    a = Arbiter({})
    c(a.decide([0.2], [0.1], True, True, False) == [False] and a.decide([0.2], [0.1], True, True, True) == [True],
      "arbiter did not re-open when the budget returned")
    # (3) a transient bad world-model batch must not poison the uncertainty ensemble permanently
    wm, pol, buf, P = mk_stack(min_n=8)
    P.run(pol, wm, buf, CPU)
    orig = P._states

    def bad_states(w, b):
        st, lat = orig(w, b)
        return st, lat * float("nan")
    P._states = bad_states
    try:
        P.run(pol, wm, buf, CPU)
    except Exception:
        pass                                                     # the loop's _planning_phase swallows and idles
    P._states = orig
    for _ in range(3):
        out = P.run(pol, wm, buf, CPU)
    finite = all(torch.isfinite(p).all() for p in P.ens.parameters()) and math.isfinite(P.ens.err_scale)
    c(finite and math.isfinite(out.get("dis", float("nan"))),
      "ensemble stayed NaN after the world model recovered (err_scale="
      f"{P.ens.err_scale}); planning arbitration stays 'invalid' for the rest of the run", n=4)
    # (4) a raising wiring module is released, idled and the failure is observable
    loop = mk_loop()
    reg = build_default_registry(rel_cfg(*MOD))
    warm_modulation(reg, loop)
    set_cal(loop, 0.0)
    reg.wire_all(loop)
    reg.wire_all(loop)

    class Boom:
        @property
        def last(self):
            raise RuntimeError("source fault")
    loop._plan = Boom()
    records = []
    h = logging.Handler()
    h.emit = lambda rec: records.append(rec.getMessage())
    lg = logging.getLogger("developmental_ai.core.relations")
    lg.addHandler(h)
    try:
        reg.wire_all(loop)
    finally:
        lg.removeHandler(h)
    c("developmental_ai.core.relations_modulation" in reg._wire_off and any("failed" in m for m in records),
      "wiring fault was not observable (idle set + warning)")
    c(registry_for(loop).get("attn.temperature_scale") is None and abs(loop._am.net.temperature - 1.0) < 1e-9
      and abs(loop._cons.strength - 1.0) < 1e-9, "wiring fault stranded coefficients", n=3)
    # (5) RW-5 gate fault opens the window and is recorded
    am = mk_attention()
    am.push_gate = lambda z0, win: (_ for _ in ()).throw(RuntimeError("gate fault"))
    for _ in range(3):
        am.observe(torch.randn(2, 8), [False, False])
    c(am.push_gate is None and am.gate_error is not None and am.win.n[0] >= 2, "gate fault did not open the window", n=3)
    return c.obs("release within the documented bound; faults observable; recoveries reached")


# ---------------------------------------------------------------------------------- CT-07
@H.scenario("CT-07", "persistence: restored planner/risk behave equivalently; mismatched width starts clean",
            brk="restore_skipped", min_exercised=12,
            oracle="a recreated PlanningPhase/RiskEstimator reproduces T, err_scale and disagreement; an "
                   "unsaved calibration window restarts empty; a different latent width leaves the "
                   "ensemble exactly as freshly built (no partial hybrid)")
def ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp()
    wm, pol, buf, P = mk_stack(min_n=8)
    for _ in range(3):
        P.run(pol, wm, buf, CPU)
    path = os.path.join(tmp, "planning.pt")
    torch.save(P.state_dict(), path)
    wm2, pol2, buf2, Q = mk_stack(seed=9, min_n=8)               # different init, same architecture
    if not brk:
        Q.load_state_dict(torch.load(path, map_location=CPU))
    x = torch.randn(5, wm.rssm.latent_dim)
    a = F_onehot(torch.randint(0, A_DIM, (5,)))
    c(abs(Q.T - P.T) < 1e-12, f"temperature not restored ({Q.T} vs {P.T})", n=1)
    c(abs(Q.ens.err_scale - P.ens.err_scale) < 1e-12, "err_scale not restored", n=1)
    c(torch.allclose(Q.ens.disagreement(x, a), P.ens.disagreement(x, a), atol=1e-7), "disagreement differs after restore", n=5)
    c(Q.mon.n("policy") == 0 and Q._calls == 0 and len(Q.arbiter._hist) == 0,
      "unsaved calibration/arbiter state was claimed to persist", n=1)
    out = Q.run(pol2, wm2, buf2, CPU)
    c(out["policy_calibration_ok"] == 0.0 and out["plan_rows"] == 0.0, "restored planner skipped the documented warmup", n=1)
    # risk head
    rp = os.path.join(tmp, "risk.pt")
    cfg = {"horizon": 4, "head": {"enabled": True, "hidden": 8}}
    torch.manual_seed(1)
    r1 = RiskEstimator(cfg, 1, CPU, rp)
    head = nn.Linear(6, 1)
    r1.observe(torch.zeros(1, 6), head, [False], [False])
    with torch.no_grad():
        for p in r1.head.parameters():
            p.add_(0.37)
    r1.save()
    if brk:
        os.remove(rp)
    torch.manual_seed(2)
    r2 = RiskEstimator(cfg, 1, CPU, rp)
    r2.observe(torch.zeros(1, 6), head, [False], [False])
    c(all(torch.equal(p, q) for p, q in zip(r1.head.parameters(), r2.head.parameters())), "risk head not restored", n=1)
    torch.manual_seed(3)
    r3 = RiskEstimator(cfg, 1, CPU, rp)                            # latent width changed: must start fresh
    r3.observe(torch.zeros(1, 9), nn.Linear(9, 1), [False], [False])
    torch.manual_seed(3)
    r4 = RiskEstimator(cfg, 1, CPU, os.path.join(tmp, "none.pt"))
    r4.observe(torch.zeros(1, 9), nn.Linear(9, 1), [False], [False])
    c(all(torch.equal(p, q) for p, q in zip(r3.head.parameters(), r4.head.parameters())),
      "risk head with another width was not left fresh", n=1)
    # ensemble with another latent width: documented "starts fresh"
    P3 = PlanningPhase({"ensemble": {"enabled": True, "hidden": 16, "heads": 3}}, A_DIM, wm.rssm.latent_dim + 2, CPU)
    before = {k: v.clone() for k, v in P3.ens.state_dict().items()}
    P3.load_state_dict(torch.load(path, map_location=CPU))
    changed = [k for k, v in P3.ens.state_dict().items() if not torch.equal(before[k], v)]
    c(not changed, f"width-mismatched ensemble state was partially copied into the new ensemble ({len(changed)} tensors, "
                   f"e.g. {changed[:2]}); 'starts fresh' is false", n=1)
    return c.obs("state restored where saved, reset where not, no partial hybrid")


def F_onehot(idx):
    return torch.nn.functional.one_hot(idx, A_DIM).float()


# ---------------------------------------------------------------------------------- CT-08
@H.scenario("CT-08", "evaluator injection rejected at the real entry point; read-only probes do not change learner state",
            brk="lax_registry_and_rng_consuming_probe", min_exercised=12,
            oracle="register() raises for evaluator tag / evaluator-module probe / unknown tag; planner "
                   "state, policy, reward-side counters and torch RNG are identical with and without telemetry")
def ct08(brk):
    c = Chk()

    class Lax(RelationRegistry):
        def register(self, line_id, frm, to, *a, source_tag="learner", probe=None, **k):
            if probe is not None:
                probe.__module__ = "tests"
            return super().register(line_id, frm, to, *a, source_tag="learner", probe=probe, **k)

    Reg = Lax if brk else RelationRegistry
    base = ("producer", "consumer", "signal", "units", "per segment")
    r = Reg({"enabled": True})
    for tag in ("evaluator", "other"):
        try:
            r.register("ZZ-01", "AAA", "BBB", *base, "data", source_tag=tag)
            c(False, f"source tag {tag!r} accepted")
        except (EvaluatorLeakError, RelationError):
            c(True, "")

    def leaky(loop):
        return 1.0
    leaky.__module__ = "developmental_ai.evaluator.runner"
    try:
        r.register("ZZ-02", "AAA", "CCC", *base, "data", probe=leaky)
        c(False, "a probe defined in the evaluator package was accepted")
    except EvaluatorLeakError:
        c(True, "")
    full = build_default_registry({"enabled": False})
    bad = [ch.name for ch in full.channels() if ch.source_tag != "learner"
           or (getattr(ch.probe, "__module__", "") or "").startswith(("developmental_ai.evaluator",
                                                                       "developmental_ai.game_template.detectors"))]
    c(not bad, f"registered channels with evaluator provenance {bad[:3]}", n=len(full.channels()))
    c(not any(m.startswith("developmental_ai.evaluator") for m in sys.modules),
      "an evaluator module is imported by the CAT-03 modules", n=1)

    def arm(telemetry):
        wm, pol, buf, P = mk_stack(min_n=8)
        loop = SimpleNamespace(_plan=P)
        reg = build_default_registry(rel_cfg(("PFC", "EF"), ("FPN", "EF"), ("MB", "PFC"), ("PROC", "AUTO")))
        reseed(7)
        reads = 0
        for _ in range(3):
            P.run(pol, wm, buf, CPU)
            if telemetry:
                reads += reg.sample_all(loop)
                reg.end_segment()
                P.report()
                if brk:
                    torch.rand(1)                                 # a telemetry helper that samples
        last = {k: v for k, v in P.last.items() if k != "secs"}
        return (phash(P.ens), P.T, dict(P.arbiter.why), P.arbiter.seen, last,
                phash(pol.actor), torch.get_rng_state().clone(), reads)
    a, b = arm(False), arm(True)
    c(torch.equal(a[0], b[0]) and a[1] == b[1], "telemetry changed ensemble/temperature", n=2)
    c(a[2] == b[2] and a[3] == b[3] and a[4] == b[4], "telemetry changed arbiter/plan outputs", n=3)
    c(torch.equal(a[5], b[5]), "telemetry changed the policy", n=1)
    c(torch.equal(a[6], b[6]), "telemetry consumed learner RNG", n=1)
    c(a[7] == 0 and b[7] >= 1, "probes were not actually read in the telemetry arm", n=1)
    return c.obs("evaluator inputs rejected; telemetry is read-only")


sys.exit(H.finish())
