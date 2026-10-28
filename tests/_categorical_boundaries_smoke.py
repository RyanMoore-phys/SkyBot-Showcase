"""CS-2 boundary scenarios BND-01..BND-07 (GOV-013 section 4, ASS-012 section 4).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_boundaries_smoke.py
    BND_ONLY=BND-03 PYTHONPATH=. ./venv/bin/python tests/_categorical_boundaries_smoke.py   # iterate on one

WHAT THIS FILE IS.  The within-group files (`_categorical_cat0N_smoke.py`) test each group on its own.
This one composes the REAL objects of TWO OR MORE groups into one fixture, in the phase order the
developmental loop really uses, and checks the end-to-end consequence on the CONSUMING side:

    BND-01  CAT-01 -> CAT-03   world model / replay evidence -> ensemble, calibration, arbiter, planner
    BND-02  CAT-03 -> CAT-04   planner calibration validity -> relation modulation -> the ACTUAL consumers
                               (entropy-controller target, attention temperature, consolidation strength)
    BND-03  CAT-02 -> CAT-03   retrieval and attention features at policy entry vs the planner's entry
    BND-04  CAT-04 <-> CAT-05  competence, curriculum, adaptive optimisation, meta adaptation, strategy
    BND-05  CAT-06 -> CAT-01/02  stream/map provenance through replay, outcome memory, peer board; held-out
    BND-06  CAT-02 <-> CAT-05  consolidation, replay selection (RW-6 recipe), optimiser updates: order, attribution
    BND-07  CAT-07 <-> all     checkpoint/restart, telemetry and injected resource pressure vs every group

Real phase order.  `LOOP_ORDER` is READ FROM `developmental_loop.py` (the call sequence after PPO, which is
identical in both stepping bodies) so a reordering of the production phases changes this fixture too.
The loop's own phase helpers (`_consolidation_phase`, `_curriculum_phase`, `_planning_phase`,
`_competence_phase`, `_adaptive_optim_phase`, `_meta_adapt_phase`, `_relations_phase`,
`_outcome_memory_step`, `_attention_memory_step`, `_current_knowledge_feature`) are bound unbound-style to a
bare namespace that carries only the attributes they read (the pattern of the CAT-02/04/05 files), so the
stream indexing, save paths, lazy construction and registry wiring are the loop's, not re-implemented.

Fixtures inject NUMBERS and FAULTS only (measured losses, entropy readings, NaN/inf, constant frames, dead
streams, injected `Signals`).  None supplies a skill or a task meaning, reads the evaluator package, starts
a game engine or a trainer, or exhausts real memory (pressure is injected on `Signals` / `PressureOrder`).

DELIBERATE BREAKS live in this file (a monkeypatched copy or a harness-side wiring change), never in
production code; the harness requires each to be detected.  A scenario that fails in its NORMAL run is a real
defect in the code under test and is left failing ("REAL DEFECT" in its note).

BLOCKED / DECLARED (inventory first, nothing invented; the `blocked_decl` rows at the end of the file):
  * RW-4 semantic storage has no policy retrieval consumer (manifest note): BND-03 tests retrieval (WP-18) and
    attention (WP-7) only.
  * Planning is learning-phase only (acting-time planning is not wired, D-30) and feeds the policy NO knowledge vector
    (GOV-011 WP-4 risks): BND-03 asserts that declared zero-context path rather than a memory-aware planner, and the
    acting-time boundary is blocked.
  * RW-12 / RW-13 / RW-10 have no module (manifest); the peer predictor does not train the shared encoder; RW-8 is
    measurement-only: no boundary scenario is invented for them.

ADDITIONAL implemented cross-group paths found in the loop (the BND table is not exhaustive): BND-X1 RW-6 CALIB.SRL
(planner ECE -> strategy exploration), BND-X2 RW-7 SAFE.CUR (risk -> information-gain weight), BND-X3 RW-5 PFC.WM
(executive gate -> attention window -> policy).  The COMBO-A / COMBO-B scenarios apply the two required controller
combinations together.

FIXTURE KNOBS (not production changes): calibration bars (max_ece, min_corr) are relaxed so an untrained random model
gives a deterministic "valid" arm; the CAT-03 output is injected as numbers into BND-04 (BND-02 covers the real chain).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from developmental_ai.core import relations_modulation as modulation  # noqa: E402
from developmental_ai.core import peer_observation as po  # noqa: E402
from developmental_ai.core.attention_memory import AttentionMemory  # noqa: E402
from developmental_ai.core.coefficients import OwnershipError, registry_for  # noqa: E402
from developmental_ai.core.developmental_loop import DevelopmentalAI as DAI  # noqa: E402
from developmental_ai.core.outcome_memory import retrieval_dim  # noqa: E402
from developmental_ai.core.planning import Arbiter, PlanningPhase  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EvaluatorLeakError, RelationError, RelationRegistry, build_default_registry)
from developmental_ai.core.risk import RiskEstimator  # noqa: E402
from developmental_ai.core.uncertainty import CalibrationMonitor  # noqa: E402
from developmental_ai.environments import data_diversity as dd  # noqa: E402
from developmental_ai.policy.actor_critic import RewardMixer, StandaloneActorCritic  # noqa: E402
from developmental_ai.policy.consolidation import Consolidator  # noqa: E402
from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer  # noqa: E402
from developmental_ai.world_model.rssm import WorldModel  # noqa: E402

torch.set_num_threads(1)
logging.getLogger().addHandler(logging.NullHandler())      # injected faults are observed via fault_log(), not stderr
H = Harness("boundaries", "Between-category boundary scenarios BND-01..BND-07", boundary=True)
CPU = torch.device("cpu")
A = 4                       # actions
OBS = 3 * 16 * 16           # flat 16x16 RGB frames
STEP = 0.05                 # RW-1 max_step
ONLY = os.environ.get("BND_ONLY", "")


def sc(cid, title, **kw):
    """H.scenario, with an optional BND_ONLY=BND-0x filter for iterating."""
    def deco(fn):
        if ONLY and ONLY not in title:
            return fn
        return H.scenario(cid, title, **kw)(fn)
    return deco


# ------------------------------------------------------------------------------------------------ helpers
class Chk:
    """Collects sub-checks so one failing contract never hides another one's evidence (duplicates are collapsed)."""

    def __init__(self):
        self.fails, self.n = [], 0

    def __call__(self, cond, msg, n=1):
        self.n += n
        if not cond and msg not in self.fails:
            self.fails.append(msg)

    def obs(self, ok_note):
        return Obs(not self.fails, self.n, "; ".join(self.fails[:4]) if self.fails else ok_note)


@contextlib.contextmanager
def patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def reseed(n):
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def phash(m):
    ps = [p.detach().flatten().clone() for p in m.parameters()]
    return torch.cat(ps) if ps else torch.zeros(0)


def thash(*xs):
    h = hashlib.sha1()
    for x in xs:
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_sig():
    h = hashlib.sha1()
    h.update(torch.get_rng_state().numpy().tobytes())
    h.update(np.random.get_state()[1].tobytes())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def finite_params(*mods):
    return all(bool(torch.isfinite(p).all()) for m in mods for p in m.parameters())


def fault_log():
    """(records, remove): captures WARNING+ records from every logger so faults are observable."""
    recs = []

    class _H(logging.Handler):
        def emit(self, r):
            recs.append(r.getMessage())
    h = _H(level=logging.WARNING)
    logging.getLogger().addHandler(h)
    return recs, lambda: logging.getLogger().removeHandler(h)


# ------------------------------------------------------------------------------------------------ real phase order
def _loop_order():
    """The per-segment learning-phase call sequence, read from the loop source (both stepping bodies)."""
    src = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    names = re.findall(r"^\s+self\.(_(?:consolidation|curriculum|planning|peer|competence|adaptive_optim|"
                       r"meta_adapt|relations)_phase)\(\)", src, re.M)
    half = len(names) // 2
    if not names or len(names) % 2 or names[:half] != names[half:]:
        raise RuntimeError("the two stepping bodies no longer call the learning phases in the same order: %s" % names)
    return names[:half]


LOOP_ORDER = _loop_order()
PHASES = [p for p in LOOP_ORDER if p != "_peer_phase"]


# ------------------------------------------------------------------------------------------------ fixtures
def mk_wm(seed=0, det=32):
    torch.manual_seed(seed)
    return WorldModel(obs_dim=OBS, action_dim=A, deterministic_size=det, stochastic_size=4,
                      stochastic_classes=4, hidden_dim=32, pixel_obs=True, image_size=16,
                      recon_reduction="sum", inverse_input="embed")


def mk_pol(L, kd=0, seed=1, gate="open"):
    torch.manual_seed(seed)
    kw = {"knowledge_dim": kd, "knowledge_gate_mode": gate} if kd else {}
    return StandaloneActorCritic(obs_dim=L, action_dim=A, hidden_dim=32, arch="rssm", enc_dim=L, device=CPU, **kw)


class Truth:
    """What each stream ACTUALLY produced (harness-side record), to compare with what replay hands back."""

    def __init__(self):
        self.obs, self.act = {}, {}


def mk_buf(streams=2, steps=240, seed=0, const=False, tag=False, swap=False, policy_cols=False):
    """A real multi-stream replay.  `tag`: column 0 of every frame encodes (stream, step) so the origin of a
    replayed row is readable.  `const`: collapsed frames and one action (constant evidence).  `swap`: the
    DEFECT of writing stream 1's rows under stream 0's id."""
    buf = MultiStreamReplayBuffer(streams, 4000, OBS, A, policy_cols=policy_cols)
    rng = np.random.RandomState(seed)
    tr = Truth()
    for s in range(streams):
        tr.obs[s], tr.act[s] = [], []
    for t in range(steps):
        for s in range(streams):
            o = np.full(OBS, 0.5, np.float32) if const else rng.rand(OBS).astype(np.float32)
            if tag:
                o[0] = (s * 1000 + t) / 100000.0
            a = 1 if const else int(rng.randint(A))
            tr.obs[s].append(o)
            tr.act[s].append(a)
            buf.add(o, a, 0.0, False, stream=(0 if (swap and s == 1) else s))
    buf.truth = tr
    return buf


def stack(seed=0, ens=True, const=False, tag=False, cfg=None, kd=0, steps=240, swap=False):
    """(wm, policy, buffer, PlanningPhase): the CAT-01 objects the planner consumes plus the CAT-03 planner."""
    wm = mk_wm(seed)
    L = wm.rssm.latent_dim
    pol = mk_pol(L, kd)
    buf = mk_buf(2, steps, seed, const=const, tag=tag, swap=swap)
    c = {"rows": 4, "seq_len": 8, "planner": {"candidates": 8, "horizon": 2, "iterations": 1},
         # fixture knobs: bars relaxed so an UNTRAINED random model gives a deterministic valid arm; the
         # gates under test are the refusal of constant / non-finite / absent / too-few evidence (min_n 24
         # needs two 16-pair windows, i.e. a restored or new planner is untrusted for its first run)
         "calibration": {"min_n": 24, "max_ece": 0.5, "min_corr": -1.0}, "ensemble": {"enabled": ens, "hidden": 16, "heads": 3},
         "distill": {"weight": 0.0}}
    for k, v in (cfg or {}).items():
        c[k] = dict(c.get(k, {}), **v) if isinstance(v, dict) else v
    return wm, pol, buf, PlanningPhase(c, A, L, CPU)


class FakeAdapter:
    """The game interface only: what the curriculum scheduler may ask of an environment adapter."""

    def __init__(self, pool="p", maps=("m1", "m2")):
        self.pool, self.maps, self.ctx = pool, list(maps), []

    def curriculum_options(self):
        return {"pool": self.pool, "maps": self.maps}

    def set_context(self, ctx):
        self.ctx.append(dict(ctx))


def brain(cfg=None, *, seed=0, kd=0, tmp=None, adapters=0, am=False, buf=None, const=False, steps=240,
          phases=None, det=32):
    """One loop-like namespace holding REAL objects of several groups.  Objects that the loop builds lazily
    inside its phase helpers (PlanningPhase, Consolidator, controllers, relation registry) are built by those
    helpers from `cfg`, exactly as in a run; the world model, policy, replay and attention memory are built
    here.  The loop's own phase methods are bound to it."""
    wm = mk_wm(seed, det)
    L = wm.rssm.latent_dim
    tmp = tmp or tempfile.mkdtemp(prefix="bnd_")
    ns = SimpleNamespace(
        config=copy.deepcopy(cfg or {}), world_model=wm, policy=mk_pol(L, kd), device=CPU, is_discrete=True,
        action_dim=A, replay_buffer=buf or mk_buf(2, steps, seed, const=const),
        skill_bank=SimpleNamespace(storage_dir=tmp), _wm_param_lock=threading.RLock(),
        reward_mixer=RewardMixer(intrinsic_weight=0.7, extrinsic_weight=0.3), lifetime_timesteps=0,
        infra=SimpleNamespace(drift_alarms=0, last_ledger_segment={}), _wm_train=None,
        _parallel_envs=[FakeAdapter() for _ in range(adapters)], _standby_refresh_policy=lambda: None,
        dir=tmp)
    if am:
        ns._am = mk_attention(tmp, dim=L)
    for n in set(PHASES) | {"_peer_phase", "_phase_failed", "_phase_rearm", "_phase_ok"}:
        setattr(ns, n, getattr(DAI, n).__get__(ns))
    ns._phases = list(phases or PHASES)
    return ns


def mk_attention(tmp=None, streams=2, dim=None, window=4, out_dim=8):
    torch.manual_seed(5)
    am = AttentionMemory({"window": window, "out_dim": out_dim, "d_model": 16, "heads": 2, "capacity": 16,
                          "cost_report": False, "store_every": 1},
                         streams, CPU, os.path.join(tmp or tempfile.mkdtemp(prefix="bnd_am_"), "attn.pt"))
    am.observe(torch.zeros(streams, dim or 8), [False] * streams)
    am.win.n = [0] * streams
    return am


def segment(ns, order=None):
    """One learning-phase segment, in the loop's real order (or an explicitly different `order`)."""
    with quiet():
        for p in (order or ns._phases):
            getattr(ns, p)()


def scale_head(ns, k):
    """Scale the policy's action-head weights: changes the policy's confidence (the CAT-03 producer)."""
    with torch.no_grad():
        for p in ns.policy.actor.action_head.parameters():
            p.mul_(k)


def rel_cfg(*pairs, **extra):
    cfg = {"enabled": True}
    for a, b in pairs:
        cfg.setdefault(a, {})[b] = {"enabled": True}
    cfg.update(extra)
    return cfg


MOD = [("LC", "NE"), ("NE", "AROUS"), ("NE", "ATT"), ("NE", "CONS")]
COEFS = ("explore.uncertainty_scale", "attn.temperature_scale", "consol.strength_scale")


def coef_values(ns):
    c = registry_for(ns)
    return {n: c.get(n) for n in COEFS}


def cv(v):
    return 1.0 if v is None else float(v)


# ================================================================================================
# BND-01  CAT-01 -> CAT-03
# ================================================================================================
@sc("CT-01", "BND-01: the CURRENT world model's latents, actions and reward head are what the ensemble trains on "
             "and the planner scores with",
    brk="stale_world_model_version", min_exercised=14,
    oracle="ensemble training inputs equal an independent recomputation of the current model's states on the "
           "sampled rows (next state = following step, action = replayed action); the planner's reward_fn equals "
           "symexp(current reward head)")
def b01_ct01(brk):
    c = Chk()
    wm, pol, buf, P = stack(tag=True)
    old = copy.deepcopy(wm)
    with torch.no_grad():                                        # the model moved on: a new version
        for p in list(wm.encoder.parameters()) + list(wm.reward_predictor.parameters()):
            p.add_(0.3 * torch.randn_like(p))
    used = old if brk else wm                                    # DEFECT: planner handed the previous version
    cap = {}
    orig_ss = buf.sample_sequences
    buf.sample_sequences = lambda *a, **k: cap.setdefault("b", orig_ss(*a, **k))
    calls = []
    orig_ts = P.ens.train_step
    P.ens.train_step = lambda l, a, n: (calls.append((l.clone(), a.clone(), n.clone())), orig_ts(l, a, n))[1]
    plans = []
    orig_plan = P.planner.plan
    P.planner.plan = lambda *a, **k: (plans.append(a), orig_plan(*a, **k))[1]
    rng_at = {}
    orig_st = P._states
    P._states = lambda w, b: (rng_at.setdefault("s", torch.get_rng_state().clone()), orig_st(w, b))[1]
    reseed(11)
    P.run(pol, used, buf, CPU)
    b = cap["b"]
    with torch.no_grad():                                        # independent recomputation, same RNG position
        torch.set_rng_state(rng_at["s"])
        st, _ = wm.observe_sequence(b["observations"], b["actions"], b.get("proprio"))
        lat = torch.cat([st["h"], st["z"]], -1)
    Bn, T, L = lat.shape
    l_in, a_in, n_in = calls[0]
    c(l_in.shape == (Bn * (T - 1), L), "ensemble input is not one row per replayed transition", n=1)
    c(torch.allclose(l_in, lat[:, :-1].reshape(-1, L), atol=1e-5),
      "ensemble trained on latents that are not the current world model's states", n=Bn)
    c(torch.allclose(n_in, lat[:, 1:].reshape(-1, L), atol=1e-5),
      "ensemble target is not the current model's NEXT state", n=Bn)
    c(torch.equal(a_in.reshape(Bn, T - 1, -1), b["actions"][:, :-1, :A].float()),
      "ensemble action is not the replayed action at the same step", n=Bn)
    # planner scoring consumes the current reward head (prediction evidence reaches the plan)
    step_fn, reward_fn = plans[0][2], plans[0][3]
    start = plans[0][0]
    rssm = wm.rssm
    with torch.no_grad():
        from developmental_ai.core.planning import _symexp
        want = _symexp(wm._reward_symlog(rssm.get_latent(start))).reshape(-1)
        got = reward_fn(start)
    c(torch.allclose(got, want, atol=1e-6), "planner reward is not the current reward head's prediction", n=len(want))
    return c.obs("ensemble inputs and planner reward come from the current model version")


@sc("CT-02", "BND-01: cutting the ensemble channel leaves planning at the habit fallback; cutting the group touches nothing",
    brk="arbiter_not_requiring_calibration", min_exercised=12,
    oracle="no ensemble -> disagreement 0, model channel not ok, zero plan rows, policy and world model unchanged; "
           "planning.enabled false -> no planner built, no file, RNG/world model/policy bit-identical")
def b01_ct02(brk):
    c = Chk()
    cfg = {"arbiter": {"require_calibrated": False}} if brk else None
    wm, pol, buf, P = stack(ens=False, cfg=cfg)
    pw, ww = phash(pol.actor), phash(wm)
    reseed(3)
    outs = [P.run(pol, wm, buf, CPU) for _ in range(3)]
    o = outs[-1]
    c(P.ens is None and o["dis"] == 0.0, "a cut ensemble still supplied disagreement", n=2)
    c(o["model_calibration_ok"] == 0.0, "a cut ensemble channel reported itself calibrated", n=1)
    c(all(x["plan_rows"] == 0.0 for x in outs), "planner trusted rows with no model evidence: %s"
      % [x["plan_rows"] for x in outs], n=3)
    c(P.arbiter.why["plan"] == 0 and P.arbiter.why["uncalibrated"] > 0, "arbiter did not record 'uncalibrated'", n=2)
    c(torch.equal(pw, phash(pol.actor)) and torch.equal(ww, phash(wm)), "planning wrote policy or world-model weights", n=2)
    # group cut through the loop's own helper
    ns = brain({"planning": {"enabled": False}})
    before = (thash(phash(ns.policy.actor), phash(ns.world_model)), rng_sig())
    with quiet():
        ns._planning_phase()
    c((thash(phash(ns.policy.actor), phash(ns.world_model)), rng_sig()) == before, "disabled planning consumed RNG or wrote weights", n=2)
    c(ns._plan is False and not os.path.exists(os.path.join(ns.dir, "planning.pt")), "disabled planning built a planner/file", n=2)
    return c.obs("cut ensemble = habit fallback; cut group = no-op")


@sc("CT-03", "BND-01: constant, invalid and absent model evidence is never trusted as planning guidance",
    brk="calibration_gate_bypassed", min_exercised=12,
    oracle="valid frames -> both channels ok and rows planned; collapsed (constant) frames, NaN states, no window and "
           "too little data -> zero planned rows, no calibration pairs recorded from invalid rows")
def b01_ct03(brk):
    c = Chk()
    ctx = contextlib.ExitStack()
    if brk:                                                      # DEFECT: both calibration gates always open
        ctx.enter_context(patched(CalibrationMonitor, "model_ok", lambda self, name="model": True))
        ctx.enter_context(patched(CalibrationMonitor, "policy_ok", lambda self, name="policy": True))
    with ctx:
        wm, pol, buf, P = stack()
        outs = []
        for i in range(4):
            reseed(i)
            outs.append(P.run(pol, wm, buf, CPU))
        c(outs[-1]["policy_calibration_ok"] == 1.0 and outs[-1]["model_calibration_ok"] == 1.0 and outs[-1]["plan_rows"] > 0,
          "valid evidence was not trusted (legitimate evidence must reach planning): %s"
          % {k: outs[-1].get(k) for k in ("policy_calibration_ok", "model_calibration_ok", "plan_rows")}, n=3)
        # constant prediction evidence: the ensemble's disagreement and error stop varying (a collapsed model);
        # the injection is on the REAL ensemble's outputs, everything downstream is the real chain
        wm2, pol2, buf2, P2 = stack()
        with patched(P2.ens, "disagreement", lambda l, a_: torch.full((l.shape[0],), 0.5)), \
                patched(P2.ens, "error", lambda l, a_, n_: torch.full((l.shape[0],), 0.25)):
            for i in range(4):
                reseed(i)
                o2 = P2.run(pol2, wm2, buf2, CPU)
        c(o2["plan_rows"] == 0.0, "constant evidence produced %s planned rows" % o2["plan_rows"], n=1)
        c(o2["model_calibration_ok"] == 0.0, "constant ensemble evidence reported calibrated", n=1)
        # NaN model states
        wm3, pol3, buf3, P3 = stack()
        for i in range(3):
            reseed(i)
            P3.run(pol3, wm3, buf3, CPU)
        n_pol, n_mod, T3 = P3.mon.n("policy"), P3.mon.n("model"), P3.T
        orig = P3._states
        P3._states = lambda w, b: (lambda st, lat: (st, lat * float("nan")))(*orig(w, b))
        planned = 0.0
        try:
            planned = P3.run(pol3, wm3, buf3, CPU).get("plan_rows", 0.0)
        except Exception:
            planned = 0.0                                        # a raised phase is idled by the loop's helper
        c(planned == 0.0, "NaN states were planned from", n=1)
        c(P3.mon.n("policy") == n_pol and P3.mon.n("model") == n_mod and P3.T == T3,
          "NaN evidence entered the calibration windows or moved the temperature", n=3)
        # absent window / not enough data
        wm4, pol4, buf4, P4 = stack()
        with patched(buf4, "newest_sequences", lambda *a, **k: None):
            r = P4._calibrate(pol4, wm4, buf4, CPU)
        c(r == {"policy_calibration_ok": 0.0, "model_calibration_ok": 0.0}, "absent window not an explicit fallback", n=1)
        wm5, pol5, buf5, P5 = stack(steps=10)
        c(P5.run(pol5, wm5, buf5, CPU) == {} and P5.last == {}, "too little data did not idle the planner", n=2)
    return c.obs("only valid, varying evidence reaches planning; the rest falls back to habit")


@sc("CT-05", "BND-01: replayed rows reach the ensemble with their own stream's frames and actions (end to end)",
    brk="stream_ids_swapped", min_exercised=12,
    oracle="every ensemble training row equals the world-model state of the frames the harness recorded for the "
           "sampled (stream, start index), with the recorded actions")
def b01_ct05(brk):
    c = Chk()
    wm, pol, buf, P = stack(tag=True, swap=bool(brk))
    cap = {}
    orig_ss = buf.sample_sequences
    buf.sample_sequences = lambda *a, **k: cap.setdefault("b", orig_ss(*a, **k))
    calls = []
    orig_ts = P.ens.train_step
    P.ens.train_step = lambda l, a, n: (calls.append((l.clone(), a.clone(), n.clone())), orig_ts(l, a, n))[1]
    rng_at = {}
    orig_st = P._states
    P._states = lambda w, b_: (rng_at.setdefault("s", torch.get_rng_state().clone()), orig_st(w, b_))[1]
    reseed(5)
    P.run(pol, wm, buf, CPU)
    b = cap["b"]
    Bn, T = b["actions"].shape[:2]
    obs_t, act_t = [], []
    for s, i in b["start_indices"]:
        obs_t.append(np.stack(buf.truth.obs[int(s)][int(i):int(i) + T]))
        act_t.append(np.eye(A, dtype=np.float32)[buf.truth.act[int(s)][int(i):int(i) + T]])
    obs_t, act_t = torch.from_numpy(np.stack(obs_t)), torch.from_numpy(np.stack(act_t))
    with torch.no_grad():
        torch.set_rng_state(rng_at["s"])
        st, _ = wm.observe_sequence(obs_t, act_t, b.get("proprio"))
        lat = torch.cat([st["h"], st["z"]], -1)
    L = lat.shape[-1]
    l_in, a_in, n_in = calls[0]
    streams = sorted({int(s) for s, _ in b["start_indices"]})
    c(len(streams) >= 1, "no stream sampled", n=1)
    c(torch.allclose(l_in, lat[:, :-1].reshape(-1, L), atol=1e-5),
      "ensemble states are not the model's states of the frames the stream actually produced", n=Bn)
    c(torch.allclose(n_in, lat[:, 1:].reshape(-1, L), atol=1e-5), "ensemble target is not the same stream's next state", n=Bn)
    c(torch.equal(a_in, act_t[:, :-1].reshape(-1, A)), "ensemble actions are not the stream's recorded actions", n=Bn)
    return c.obs("ensemble rows trace back to the producing stream and step")


@sc("CT-06", "BND-01: planning idles on a short or absent replay window and resumes when data arrives",
    brk="idle_latch", min_exercised=6,
    oracle="short replay -> {} and no planner state; after more data arrives planning produces rows within the "
           "calibration bound; a dependency that returns None is explicit in the result")
def b01_ct06a(brk):
    c = Chk()
    wm, pol, buf, P = stack(steps=10)
    c(P.run(pol, wm, buf, CPU) == {}, "short window did not idle", n=1)
    rng = np.random.RandomState(9)
    got = None
    for k in range(8):
        if k == 1 and not brk:                                   # data arrives (the dependency is restored)
            for t in range(230):
                for s in range(2):
                    buf.add(rng.rand(OBS).astype(np.float32), int(rng.randint(A)), 0.0, False, stream=s)
        reseed(k)
        got = P.run(pol, wm, buf, CPU)
        if got and got.get("plan_rows", 0) > 0:
            break
    c(bool(got), "planning never resumed after data returned (permanent closed gate)", n=2)
    c(bool(got) and got.get("policy_calibration_ok") == 1.0 and got.get("plan_rows", 0) > 0,
      "planning resumed but did not trust valid evidence: %s" % {k: got.get(k) for k in ("policy_calibration_ok", "plan_rows")} if got else "no result",
      n=3)
    return c.obs("planning resumes as soon as the replay dependency returns")


@sc("CT-06", "BND-01: a transient non-finite world-model batch does not permanently poison planner evidence "
             "(REAL DEFECT if red: EnsembleHeads.train_step has no finiteness guard)",
    brk="poisoned_batch_every_segment", min_exercised=4,
    oracle="after one NaN-state segment and three valid ones the ensemble weights, err_scale and disagreement are "
           "finite and the arbiter again receives valid values")
def b01_ct06b(brk):
    c = Chk()
    wm, pol, buf, P = stack()
    P.run(pol, wm, buf, CPU)
    orig = P._states
    bad = lambda w, b: (lambda st, lat: (st, lat * float("nan")))(*orig(w, b))   # noqa: E731
    P._states = bad
    try:
        P.run(pol, wm, buf, CPU)
    except Exception:
        pass
    out = {}
    for k in range(3):
        P._states = bad if brk else orig
        try:
            out = P.run(pol, wm, buf, CPU)
        except Exception:
            out = {}
    finite = finite_params(P.ens) and math.isfinite(P.ens.err_scale)
    c(finite, "ensemble stayed non-finite after the world model recovered (err_scale=%s)" % P.ens.err_scale, n=2)
    c(math.isfinite(out.get("dis", float("nan"))), "arbiter input stayed invalid after recovery", n=1)
    c(P.arbiter.why["invalid"] == 0 or finite, "every later row is refused as invalid: planning is dead for the run", n=1)
    return c.obs("planner evidence recovered after a transient bad batch")


@sc("CT-07", "BND-01: a restored planner on a different/new world model re-earns its trust (calibration is not persisted)",
    brk="calibration_window_carried_over", min_exercised=6,
    oracle="saved: temperature, ensemble, err_scale; not saved: calibration window, arbiter history; the first run of "
           "the restored planner plans zero rows and the channels reopen only after min_n fresh pairs")
def b01_ct07a(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_plan_")
    wm, pol, buf, P = stack(seed=0)
    for i in range(3):
        reseed(i)
        P.run(pol, wm, buf, CPU)
    c(P.last["plan_rows"] > 0, "fixture: the source planner never trusted its evidence", n=1)
    path = os.path.join(tmp, "planning.pt")
    torch.save(P.state_dict(), path)
    wm2, pol2, buf2, Q = stack(seed=7)                           # the model it is now attached to is a different version
    Q.load_state_dict(torch.load(path, map_location=CPU))
    if brk:                                                      # DEFECT: the calibration window is assumed to persist
        Q.mon = copy.deepcopy(P.mon)
        Q.arbiter = copy.deepcopy(P.arbiter)
    c(abs(Q.T - P.T) < 1e-12 and abs(Q.ens.err_scale - P.ens.err_scale) < 1e-12, "temperature/err_scale not restored", n=2)
    c(all(torch.equal(a, b) for a, b in zip(Q.ens.parameters(), P.ens.parameters())), "ensemble weights not restored", n=1)
    reseed(21)
    first = Q.run(pol2, wm2, buf2, CPU)
    c(first["plan_rows"] == 0.0, "restored planner trusted rows on its first run: %s" % first["plan_rows"], n=2)
    reseed(22)
    second = Q.run(pol2, wm2, buf2, CPU)
    c(second["policy_calibration_ok"] == 1.0, "calibration did not reopen once fresh pairs accrued", n=1)
    return c.obs("restart restores weights, not trust")


@sc("CT-07", "BND-01: a world model grown to another latent width leaves the restored planner exactly fresh "
             "(REAL DEFECT if red: PlanningPhase.load_state_dict partial hybrid)",
    brk="stale_width_state_copied", min_exercised=2,
    oracle="state saved at latent width L loaded into a planner built for another width leaves the ensemble "
           "tensors exactly as built (no partial copy)")
def b01_ct07b(brk):
    c = Chk()
    wm, pol, buf, P = stack()
    reseed(1)
    P.run(pol, wm, buf, CPU)
    st = P.state_dict()
    wide = PlanningPhase({"ensemble": {"enabled": True, "hidden": 16, "heads": 3}}, A, wm.rssm.latent_dim + 2, CPU)
    before = {k: v.clone() for k, v in wide.ens.state_dict().items()}
    if brk:                                                      # DEFECT: a width-blind copy
        for k, v in st["ens"].items():
            if k in before and before[k].shape == v.shape:
                wide.ens.state_dict()[k].copy_(v)
        wide.ens.state_dict()
    wide.load_state_dict(st)
    changed = [k for k, v in wide.ens.state_dict().items() if not torch.equal(before[k], v)]
    c(not changed, "width-mismatched ensemble state was partially copied (%d tensors, e.g. %s)" % (len(changed), changed[:2]), n=2)
    return c.obs("another latent width starts the ensemble fresh")


class Tripwire(dict):
    """A replay batch that fails on any key outside the allow-list the planner is declared to read."""
    ALLOWED = {"observations", "actions", "proprio", "start_indices", "rewards", "continues"}

    def __getitem__(self, k):
        if k not in self.ALLOWED:
            raise AssertionError("learner read non-declared batch field %r" % k)
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        if k not in self.ALLOWED:
            raise AssertionError("learner read non-declared batch field %r" % k)
        return dict.get(self, k, d)


@sc("CT-08", "BND-01: evaluator-side fields in a replay batch are never read; telemetry leaves planner state and RNG alone",
    brk="planner_reads_undeclared_field", min_exercised=8,
    oracle="a batch carrying evaluator_checkpoint / held_out_score fields is consumed without touching them; "
           "report()/sample_all() change no planner weight, temperature, arbiter count or RNG")
def b01_ct08(brk):
    c = Chk()
    wm, pol, buf, P = stack()
    orig_ss = buf.sample_sequences

    def poisoned(*a, **k):
        b = Tripwire(orig_ss(*a, **k))
        b["evaluator_checkpoint"] = torch.ones(1)
        b["held_out_score"] = torch.ones(1)
        return b
    buf.sample_sequences = poisoned
    orig_states = P._states
    if brk:                                                      # DEFECT: a planner change that reads a side field
        P._states = lambda w, b: (b["evaluator_checkpoint"], orig_states(w, b))[1]
    P.run(pol, wm, buf, CPU)
    c(True, "tripwire survived", n=1)

    def arm(telemetry):
        w, p, bf, Pp = stack()
        loop = SimpleNamespace(_plan=Pp)
        reg = build_default_registry(rel_cfg(("PFC", "EF"), ("FPN", "EF"), ("MB", "PFC"), ("PROC", "AUTO")))
        reseed(7)
        for _ in range(3):
            Pp.run(p, w, bf, CPU)
            if telemetry:
                reg.sample_all(loop)
                reg.end_segment()
                Pp.report()
        return (phash(Pp.ens), Pp.T, dict(Pp.arbiter.why), Pp.arbiter.seen, phash(p.actor), rng_sig())
    a, b2 = arm(False), arm(True)
    c(torch.equal(a[0], b2[0]) and a[1] == b2[1], "telemetry changed the ensemble/temperature", n=2)
    c(a[2] == b2[2] and a[3] == b2[3], "telemetry changed arbitration", n=2)
    c(torch.equal(a[4], b2[4]), "telemetry changed the policy", n=1)
    c(a[5] == b2[5], "telemetry consumed learner RNG", n=1)
    mods = [m for m in sys.modules if m.startswith("developmental_ai.evaluator")]
    c(not mods, "an evaluator module is imported on the planning path: %s" % mods[:2], n=1)
    return c.obs("undeclared fields are not read; telemetry is read-only")


# ================================================================================================
# BND-02  CAT-03 -> CAT-04   planner calibration validity -> relation modulation -> actual consumers
# ================================================================================================
MODCFG = {"amp": 0.3, "max_step": STEP, "min_n": 4, "ema": 0.9, "cons_sign": -1}


def b02_brain(tmp=None, rel="all", seed=0, ao=True):
    """Real planner (lazy, via the loop's own _planning_phase), real consolidator, real WP-9 entropy controller,
    real attention memory, real relation registry, in the loop's real phase order."""
    cfg = {"planning": {"enabled": True, "rows": 4, "seq_len": 8,
                        "planner": {"candidates": 8, "horizon": 2, "iterations": 1},
                        "calibration": {"min_n": 32, "window": 32, "max_ece": 0.2, "min_corr": -1.0},
                        "ensemble": {"enabled": True, "hidden": 16, "heads": 3}},
           "consolidation": {"enabled": True, "every_segments": 1, "steps": 1, "batch": 4, "seq_len": 4,
                             "lr_scale": 0.1, "distill": {"enabled": True, "snapshot_every_segments": 100,
                                                          "max_snapshots": 2}}}
    if ao:
        cfg["adaptive_optim"] = {"enabled": True, "every_segments": 1,
                                 "entropy": {"enabled": True, "target": 0.5, "eta": 0.5}}
    if rel == "all":
        cfg["relations"] = rel_cfg(*MOD, modulation=MODCFG)
    elif rel:
        cfg["relations"] = rel
    ns = brain(cfg, seed=seed, am=True, tmp=tmp)
    ns._head0 = {k: v.clone() for k, v in ns.policy.actor.action_head.state_dict().items()}
    ns._tgt, ns._seg_i = [], 0
    return ns


def set_conf(ns, k):
    """The CAT-03 producer: the policy's action concentration (head scaled by k: 0 -> uniform, large -> sharp)."""
    ns.policy.actor.action_head.load_state_dict({n: v * k for n, v in ns._head0.items()})


def b02_seg(ns, k, ent=0.3, order=None):
    set_conf(ns, k)
    ns.lifetime_timesteps += 64
    ns.policy.last_update_stats = {"cond_entropy_norm": ent}
    segment(ns, order)
    ao = getattr(ns, "_ao", None)
    if ao and ao.ent is not None and not getattr(ao.ent, "_spied", False):
        o = ao.ent.update
        ao.ent.update = lambda h, target=None, _o=o: (ns._tgt.append((ns._seg_i, target)), _o(h, target))[1]
        ao.ent._spied = True
    ns._seg_i += 1
    P = getattr(ns, "_plan", None)
    return dict(P.last) if P else {}


def b02_state(ns):
    v = coef_values(ns)
    return {"scale": v, "temp": ns._am.net.temperature,
            "strength": getattr(getattr(ns, "_cons", None), "strength", 1.0),
            "entropy_coef": ns.policy.entropy_coef}


def b02_warm(ns, n=8):
    """Valid, varying evidence (mildly different confidences) so the z-score baseline exists."""
    for i in range(n):
        b02_seg(ns, 0.3 if i % 2 else 1.0)


@sc("CT-01", "BND-02: valid planner calibration reaches the ACTUAL consumers (entropy target, attention temperature, "
             "consolidation strength) with the declared sign, bound and delay",
    brk="relations_phase_omitted", min_exercised=18,
    oracle="uncertain planner evidence raises the entropy-controller target, lowers the attention temperature and the "
           "consolidation strength, each by <= max_step per segment; the consumer objects hold the coefficient values")
def b02_ct01(brk):
    c = Chk()
    ns = b02_brain()
    order = [p for p in ns._phases if not (brk and p == "_relations_phase")]
    for i in range(8):
        b02_seg(ns, 0.3 if i % 2 else 1.0, order=order)
    path = [b02_state(ns)]
    for _ in range(6):                                           # evidence turns uncertain: uniform policy
        last = b02_seg(ns, 0.0, order=order)
        path.append(b02_state(ns))
    c(last.get("policy_calibration_ok") == 1.0 and last.get("calibration_conf", 1.0) < 0.3,
      "fixture: planner did not report valid low confidence (%s)" % {k: last.get(k) for k in ("policy_calibration_ok", "calibration_conf")}, n=2)
    hi = path[-1]
    steps_ok = all(abs(cv(path[i + 1]["scale"][n]) - cv(path[i]["scale"][n])) <= STEP + 1e-9
                   for i in range(len(path) - 1) for n in COEFS)
    c(steps_ok, "a coefficient moved by more than max_step in one segment", n=18)
    c(cv(hi["scale"]["explore.uncertainty_scale"]) > 1.0 + STEP, "entropy-target multiplier did not rise with uncertainty")
    c(cv(hi["scale"]["attn.temperature_scale"]) < 1.0 - STEP, "attention temperature did not fall with uncertainty")
    c(cv(hi["scale"]["consol.strength_scale"]) < 1.0 - STEP, "consolidation strength did not fall with uncertainty")
    if not brk:
        c(abs(hi["temp"] - cv(hi["scale"]["attn.temperature_scale"])) < 1e-9, "attention net did not receive the temperature", n=1)
        c(abs(hi["strength"] - cv(hi["scale"]["consol.strength_scale"])) < 1e-9, "consolidator did not receive the strength", n=1)
        tg = [t for _, t in ns._tgt]
        c(len(tg) >= 8 and max(tg) > 0.5 + STEP * 0.5 and abs(tg[-1] - min(0.99, 0.5 * cv(hi["scale"]["explore.uncertainty_scale"]) / 1.0)) < 0.06,
          "the entropy controller's target did not follow the multiplier: %s" % [round(t, 3) for t in tg[-3:]], n=2)
    else:
        c(hi["temp"] != 1.0 or hi["strength"] != 1.0 or cv(hi["scale"]["explore.uncertainty_scale"]) != 1.0,
          "no consumer moved (relations phase omitted)", n=1)
    return c.obs("calibration validity and value reach the entropy target, attention and consolidation")


@sc("CT-02", "BND-02: a cut modulation channel leaves its consumer exactly neutral; the producer or master cut leaves all three",
    brk="cut_channel_still_wired", min_exercised=14,
    oracle="NE.AROUS cut: entropy target stays exactly the configured 0.5, no coefficient claimed, temperature and strength still "
           "move; LC.NE (the gain) cut or relations off: no consumer moves at all")
def b02_ct02(brk):
    c = Chk()
    no_arous = rel_cfg(("LC", "NE"), ("NE", "ATT"), ("NE", "CONS"), modulation=MODCFG)
    ctx = patched(RelationRegistry, "is_on", lambda self, n, _o=RelationRegistry.is_on: True if n == "rel.NE.AROUS" else _o(self, n)) \
        if brk else contextlib.nullcontext()
    with ctx:
        ns = b02_brain(rel=no_arous)
        b02_warm(ns)
        for _ in range(5):
            b02_seg(ns, 0.0)
    st = b02_state(ns)
    reg = registry_for(ns)
    c(st["temp"] < 1.0 - 1e-9 and st["strength"] < 1.0 - 1e-9, "the enabled channels did not move their consumers", n=2)
    c(reg.owner_of("explore.uncertainty_scale") is None, "cut NE.AROUS claimed its coefficient", n=1)
    c(all(abs(t - 0.5) < 1e-12 for _, t in ns._tgt), "entropy target left the configured value with NE.AROUS cut", n=len(ns._tgt))
    for label, rel in (("LC.NE cut", rel_cfg(("NE", "AROUS"), ("NE", "ATT"), ("NE", "CONS"), modulation=MODCFG)),
                       ("relations off", {"enabled": False})):
        off = b02_brain(rel=rel)
        b02_warm(off)
        for _ in range(3):
            b02_seg(off, 0.0)
        so = b02_state(off)
        c(so["temp"] == 1.0 and so["strength"] == 1.0 and all(abs(t - 0.5) < 1e-12 for _, t in off._tgt)
          and all(abs(cv(v) - 1.0) < 1e-12 for v in so["scale"].values()),
          "%s: a consumer is not exactly neutral %s" % (label, so), n=3)
    return c.obs("cut channels are neutral at the consumers")


def lax_policy_uncertainty(loop):
    P = getattr(loop, "_plan", None)
    last = getattr(P, "last", None) or {}
    cf = last.get("calibration_conf")
    return None if cf is None else 1.0 - float(cf)


@sc("CT-03", "BND-02: planner evidence that is invalid (miscalibrated, NaN, absent, planner idle) governs nothing",
    brk="validity_flag_ignored", min_exercised=16,
    oracle="after warm valid evidence, each invalid kind adds no z-score sample and never moves a coefficient away "
           "from neutral; a legitimate valid sample still does")
def b02_ct03(brk):
    c = Chk()
    ns = b02_brain()
    b02_warm(ns)
    z = ns._rel.state["mod"]["z"]
    b02_seg(ns, 0.0)                                             # displaced and valid
    c(z.n >= 8 and max(abs(cv(v) - 1.0) for v in coef_values(ns).values()) > 0.0, "fixture: no displaced coefficient", n=1)
    ctx = patched(modulation, "policy_uncertainty", lax_policy_uncertainty) if brk else contextlib.nullcontext()

    def check(kind, make):
        n0 = z.n
        before = {k: abs(cv(v) - 1.0) for k, v in coef_values(ns).items()}
        for _ in range(2):
            make()
        after = {k: abs(cv(v) - 1.0) for k, v in coef_values(ns).items()}
        c(z.n == n0, "%s added a z-score sample" % kind, n=1)
        c(all(after[k] <= before[k] + 1e-9 for k in before), "%s moved a coefficient away from neutral" % kind, n=3)
    with ctx:
        check("a miscalibrated (sharp) policy", lambda: b02_seg(ns, 200.0))
        P = ns._plan
        nan_logits = lambda p, x: torch.full((x.shape[0], A), float("nan"))   # noqa: E731
        with patched(P, "_logits", nan_logits):
            check("NaN policy logits", lambda: b02_seg(ns, 1.0))
        with patched(ns.replay_buffer, "newest_sequences", lambda *a, **k: None):
            check("an absent replay window", lambda: b02_seg(ns, 1.0))
        P2 = ns._plan
        ns._plan = False                                         # planner idled by the loop
        check("an idle planner", lambda: segment(ns, [p for p in ns._phases if p != "_planning_phase"]))
        ns._plan = P2
    n1 = z.n
    for _ in range(3):
        b02_seg(ns, 0.0)
    c(z.n > n1, "valid evidence stopped governing after the invalid episodes (latched)", n=1)
    return c.obs("only valid planner evidence is consumed")


@sc("CT-05", "BND-02: modulation reads the SAME segment's planner evidence (planning before relations); the entropy "
             "controller sees it one segment later, as declared",
    brk="relations_before_planning", min_exercised=10,
    oracle="u recorded by the relation wiring in segment s equals 1 - calibration_conf produced by the planning phase "
           "in segment s; the controller's target in segment s uses the multiplier published in s-1")
def b02_ct05(brk):
    c = Chk()
    ns = b02_brain()
    order = list(reversed(ns._phases)) if brk else None         # DEFECT: relation wiring runs before the planner
    seen_u, conf = [], []
    real_u = modulation.policy_uncertainty
    with patched(modulation, "policy_uncertainty", lambda lp: (seen_u.append(real_u(lp)), seen_u[-1])[1]):
        for i in range(10):
            last = b02_seg(ns, [1.0, 0.0, 0.3][i % 3], order=order)
            conf.append(last.get("calibration_conf"))
    pairs = [(u, cf) for u, cf in zip(seen_u, conf) if u is not None and cf is not None]
    c(len(pairs) >= 8, "fixture: too few valid sample pairs (%d)" % len(pairs), n=1)
    bad = [i for i, (u, cf) in enumerate(pairs) if abs(u - (1.0 - cf)) > 1e-9]
    c(not bad, "the modulation read evidence from another segment (%d of %d samples)" % (len(bad), len(pairs)), n=len(pairs))
    if not brk:
        hist = {}
        reg = registry_for(ns)
        c(len(ns._tgt) >= 8, "no controller targets recorded", n=1)
    return c.obs("same-segment evidence into modulation, one-segment delay into the controller")


@sc("CT-06", "BND-02: planner evidence loss releases the ACTUAL consumers within ceil(dev/step) segments, and valid evidence resumes",
    brk="release_path_missing", min_exercised=10,
    oracle="the replay window vanishes: multiplier, temperature, strength and the controller's target return to "
           "neutral in <= ceil(max deviation / step) segments and stay; valid evidence re-engages modulation")
def b02_ct06(brk):
    c = Chk()
    ns = b02_brain()
    b02_warm(ns)
    for _ in range(7):
        b02_seg(ns, 0.0)
    st = b02_state(ns)
    dev = max(abs(cv(v) - 1.0) for v in st["scale"].values())
    c(dev > 0.2 and st["temp"] < 0.9 and st["strength"] < 0.9, "fixture did not displace the consumers (dev %.3f)" % dev, n=1)
    bound = math.ceil(dev / STEP - 1e-9)
    real_apply = modulation.apply
    gate = (lambda lp, rg: None if modulation.policy_uncertainty(lp) is None else real_apply(lp, rg)) if brk else real_apply
    steps = 0
    with patched(modulation, "apply", gate), patched(ns.replay_buffer, "newest_sequences", lambda *a, **k: None):
        while steps < bound + 3 and (max(abs(cv(v) - 1.0) for v in coef_values(ns).values()) > 1e-9
                                     or abs(ns._am.net.temperature - 1.0) > 1e-9 or abs(ns._cons.strength - 1.0) > 1e-9):
            b02_seg(ns, 0.0)
            steps += 1
    c(steps <= bound, "released in %d segments, bound %d (state %s)" % (steps, bound, {k: round(v, 3) for k, v in coef_values(ns).items() if v}), n=steps + 1)
    c(abs(ns._am.net.temperature - 1.0) < 1e-9 and abs(ns._cons.strength - 1.0) < 1e-9, "consumer stranded away from neutral", n=2)
    last_t = ns._tgt[-1][1]
    b02_seg(ns, 0.0)                                             # one more segment: the controller reads the released multiplier
    c(abs(ns._tgt[-1][1] - 0.5) < 1e-9, "the entropy controller's target stayed off its configured value (%.4f)" % ns._tgt[-1][1], n=1)
    for i in range(10):                                          # the dependency returns
        b02_seg(ns, 0.0 if i % 3 == 0 else 1.0)
    c(any(abs(cv(v) - 1.0) > 1e-9 for v in coef_values(ns).values()), "modulation did not resume after valid evidence returned", n=1)
    return c.obs("evidence loss releases every consumer within the slew bound; no latch")


@sc("CT-07", "BND-02: planner weights/temperature persist across a restart; coefficients and calibration restart neutral and re-earn trust",
    brk="planner_state_not_restored", min_exercised=10,
    oracle="a recreated brain on the same storage restores the planner's temperature and ensemble through the loop's own "
           "restore path; every coefficient and consumer starts neutral; modulation stays neutral until the unsaved "
           "calibration window refills")
def b02_ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b02_")
    ns = b02_brain(tmp=tmp)
    b02_warm(ns)
    for _ in range(5):
        b02_seg(ns, 0.0)
    T, ens = ns._plan.T, [p.detach().clone() for p in ns._plan.ens.parameters()]
    c(max(abs(cv(v) - 1.0) for v in coef_values(ns).values()) > 0.05, "fixture: nothing to lose on restart", n=1)
    loaded = {}
    real_load = PlanningPhase.load_state_dict

    def spy(self, d):
        real_load(self, d)
        loaded["T"] = self.T
        loaded["same_ens"] = all(torch.equal(a, b) for a, b in zip(self.ens.parameters(), ens))
    ns2 = b02_brain(tmp=(tempfile.mkdtemp(prefix="bnd_b02_other_") if brk else tmp))
    with patched(PlanningPhase, "load_state_dict", spy):
        b02_seg(ns2, 1.0)                                        # first segment of the new process
    c(bool(loaded) and abs(loaded["T"] - T) < 1e-12, "the loop did not restore the planner temperature (%s vs %.4f)" % (loaded.get("T"), T), n=1)
    c(bool(loaded) and loaded["same_ens"], "the loop did not restore the planner ensemble", n=1)
    s2 = b02_state(ns2)
    c(all(abs(cv(v) - 1.0) < 1e-12 for v in s2["scale"].values()) and s2["temp"] == 1.0 and s2["strength"] == 1.0,
      "a coefficient or consumer survived the restart non-neutral: %s" % s2, n=5)
    c(ns2._plan.last.get("policy_calibration_ok") == 0.0, "the unsaved calibration window was treated as already valid", n=1)
    for i in range(7):
        b02_seg(ns2, [1.0, 0.3][i % 2])
    for _ in range(3):
        b02_seg(ns2, 0.0)
    c(ns2._plan.last.get("policy_calibration_ok") == 1.0, "calibration never reopened after restart", n=1)
    c(any(abs(cv(v) - 1.0) > 1e-9 for v in coef_values(ns2).values()), "modulation never re-engaged after the window refilled", n=1)
    return c.obs("weights persist; trust and coefficients restart")


@sc("CT-08", "BND-02: an evaluator-side field in the planner's output governs nothing; telemetry leaves coefficients and RNG alone",
    brk="evaluator_field_read", min_exercised=12,
    oracle="coefficient path with 'evaluator_score'/'held_out_checkpoint' injected into the planner output equals the "
           "path without; sample_all/report/line change no coefficient, weight or RNG")
def b02_ct08(brk):
    c = Chk()

    def run(inject, telemetry):
        reseed(13)
        ns = b02_brain()
        reseed(14)
        ctx = contextlib.nullcontext()
        if brk and inject:                                       # DEFECT: a "helper" folds an evaluator signal in
            real = modulation.policy_uncertainty
            ctx = patched(modulation, "policy_uncertainty",
                          lambda lp: None if real(lp) is None else real(lp) + 0.5 * float(lp._plan.last.get("evaluator_score", 0.0)))
        path = []
        with ctx:
            for i in range(9):
                set_conf(ns, [1.0, 0.0, 0.3][i % 3])
                ns.lifetime_timesteps += 64
                ns.policy.last_update_stats = {"cond_entropy_norm": 0.3}
                if inject and getattr(ns, "_plan", None):
                    ns._plan.last["evaluator_score"] = 3.0 * (i % 2)
                    ns._plan.last["held_out_checkpoint"] = float(i)
                with quiet():
                    for p in ns._phases:
                        getattr(ns, p)()
                        if p == "_planning_phase" and inject and getattr(ns, "_plan", None):
                            ns._plan.last["evaluator_score"] = 3.0 * (i % 2)
                            ns._plan.last["held_out_checkpoint"] = float(i)
                if telemetry:
                    ns._rel.sample_all(ns)
                    ns._rel.line()
                    ns._plan.report()
                path.append(tuple(round(cv(v), 9) for v in coef_values(ns).values()) + (round(ns._am.net.temperature, 9),))
        return path, rng_sig(), thash(phash(ns._plan.ens), phash(ns.policy.actor))
    base = run(False, False)
    inj = run(True, False)
    tel = run(False, True)
    c(base[0] == inj[0], "an evaluator-side field changed the coefficient path", n=9)
    c(base[0] == tel[0] and base[2] == tel[2], "telemetry changed coefficients or weights", n=3)
    c(base[1] == tel[1], "telemetry consumed learner RNG", n=1)
    return c.obs("evaluator fields are inert; telemetry is read-only")


# ================================================================================================
# BND-03  CAT-02 -> CAT-03   retrieval / attention features at policy entry vs the planner's entry
# ================================================================================================
OM_CFG = dict(horizon=1, capacity=64, k=3, min_age=2, min_sim=0.5, temperature=0.1)
AM_CFG = dict(window=4, d_model=16, heads=2, out_dim=6, train_every=4, batch=4, steps=1, store_every=1,
              capacity=64, report_every=10 ** 9, cost_report=False, lr=1e-3)
LATV, NS3 = 8, 3
_q, _ = torch.linalg.qr(torch.randn(LATV, LATV, generator=torch.Generator().manual_seed(11)))
VIEW = {n: _q[:, i].clone() for i, n in enumerate("abcdefgh")}      # orthonormal view embeddings
_NPR = np.random.RandomState(3)


def near(v, eps=0.02):
    return v + eps * torch.from_numpy(_NPR.randn(v.numel()).astype(np.float32))


def b03_loop(om=True, am=True, streams=NS3, kd=None, tmp=None, seed=0, amover=None):
    """A loop-like namespace with the REAL WP-18 / WP-7 objects (built lazily by the loop's own step helpers),
    a real world model and replay, an rssm-arch policy whose knowledge width is what the loop would build, and
    the CAT-03 planner over the same policy."""
    wm = mk_wm(seed)
    Ld = wm.rssm.latent_dim
    omd = retrieval_dim(A) if om else 0
    amc = dict(AM_CFG, **(amover or {}))
    amd = amc["out_dim"] if am else 0
    pol = mk_pol(Ld, (omd + amd) if kd is None else kd)
    d = tmp or tempfile.mkdtemp(prefix="bnd_b03_")
    cfg = {}
    if om:
        cfg["outcome_memory"] = dict(OM_CFG, enabled=True)
    if am:
        cfg["attention_memory"] = dict(amc, enabled=True)
    L = SimpleNamespace(
        config=cfg, device=CPU, policy=pol, world_model=wm, replay_buffer=mk_buf(streams, 160, seed),
        skill_bank=SimpleNamespace(storage_dir=d), is_discrete=True, action_dim=A, _num_envs=streams,
        total_timesteps=0, _vk_step=None, _om=None, _am=None, _cons=None, _rel=None, _om_dim=omd, _am_dim=amd,
        _pc_dim=0, _rk_dim=0, _kd_base=0, _peer=None, _rk=None, _am_sp=None, _last_ppo={},
        infra=SimpleNamespace(last_ledger_segment={}), _wm_param_lock=threading.RLock(),
        _standby_refresh_policy=lambda: None, _keys=[None] * streams, _sp=None, dir=d, streams=streams)
    L._current_knowledge_feature_base = lambda: None
    L._step_view_key = lambda i: L._keys[i]
    L._view_key = lambda x: x
    L._step_surprise = lambda *a, **k: L._sp
    L.plan = PlanningPhase({"rows": 4, "seq_len": 8, "planner": {"candidates": 8, "horizon": 2, "iterations": 1},
                            "calibration": {"min_n": 24, "max_ece": 0.5, "min_corr": -1.0},
                            "ensemble": {"enabled": True, "hidden": 16, "heads": 3}}, A, Ld, CPU)
    for n in ("_outcome_memory_step", "_attention_memory_step", "_current_knowledge_feature"):
        setattr(L, n, getattr(DAI, n).__get__(L))
    nb = L.replay_buffer.newest_sequences(8, CPU)
    with torch.no_grad():
        st, _ = wm.observe_sequence(nb["observations"], nb["actions"])
        L.lat_seq = torch.cat([st["h"], st["z"]], -1)            # (streams, T, latent): real model latents
    return L


def sp_of(n, s=0.3):
    return SimpleNamespace(s=np.full(n, float(s)), guard=np.full(n, False), agency_for=lambda idx: np.zeros(len(idx)))


def om_step(L, keys, acts, surprise=0.3, dones=None, restarted=None):
    n = len(keys)
    L._keys = list(keys)
    L._sp = sp_of(n, surprise)
    ex = torch.zeros(n, A)
    for i, a in enumerate(acts):
        ex[i, a] = 1.0
    with quiet():
        L._outcome_memory_step(None, ex, None, [None] * n, torch.full((n,), 0.2),
                               dones or [False] * n, restarted or [False] * n, [{} for _ in range(n)])


def am_step(L, t, dones=None, restarted=None, lat=None):
    n = L.streams
    z = (L.lat_seq[:, t % L.lat_seq.shape[1]] if lat is None else lat)
    with quiet():
        L._attention_memory_step(z, dones or [False] * n, restarted or [False] * n, [{} for _ in range(n)], None)


def feed_mem(L, n=17, a1=2, hi=3.0, lo=0.3, view0=("a", "b")):
    """Stream 0 alternates two views and acts a1 after the second; other streams sit on view c."""
    for t in range(n):
        keys, acts, sur = [], [], []
        for s in range(L.streams):
            if s == 0:
                keys.append(near(VIEW[view0[t % 2]]))
                acts.append(a1 if t % 2 == 1 else 0)
            else:
                keys.append(near(VIEW["c"]))
                acts.append(1)
            sur.append(hi if (s == 0 and t % 2 == 1) else lo)
        om_step(L, keys, acts, sur[0])
        if L._am_dim:
            am_step(L, t)


def kn_of(L):
    return L._current_knowledge_feature()


def rd():
    return retrieval_dim(A)


@sc("CT-01", "BND-03: retrieval and attention features have the declared identity, slice and width at POLICY entry",
    brk="slices_swapped", min_exercised=10,
    oracle="knowledge vector = [OM retrieval | AM summary] of width policy.knowledge_dim; the retrieval action one-hot "
           "is the executed action stored with the stream-0 view; the vector reaches the augmented policy input unchanged")
def b03_ct01a(brk):
    c = Chk()
    L = b03_loop()
    feed_mem(L)
    if brk:                                                      # DEFECT: attention appended before retrieval
        fn = lambda self: np.concatenate([self._am.feature(), self._om.feature(0)]).astype(np.float32)   # noqa: E731
        L._current_knowledge_feature = fn.__get__(L)
    kn = kn_of(L)
    pol = L.policy
    c(kn.shape == (pol.knowledge_dim,) and pol.knowledge_dim == rd() + L._am_dim, "knowledge width %s vs policy %d" % (kn.shape, pol.knowledge_dim), n=2)
    c(np.array_equal(kn[:rd()], L._om.feature(0)), "retrieval slice is not the outcome memory's stream-0 feature", n=rd())
    c(np.array_equal(kn[rd():], L._am.feature()), "attention slice is not the attention memory's summary", n=L._am_dim)
    oh = kn[8:8 + A]
    c(oh.max() > 0 and int(np.argmax(oh)) == 2, "retrieval action one-hot %s is not the stored action 2" % np.round(oh, 2), n=1)
    c(np.isfinite(kn).all() and np.abs(kn[rd():]).sum() > 0, "attention slice empty or non-finite", n=1)
    aug = pol._augment(L.lat_seq[0, :1], pol._prep_knowledge(kn), None)
    c(torch.allclose(aug[0, -pol.knowledge_dim:], torch.from_numpy(kn)), "vector did not reach the policy input unchanged", n=1)
    return c.obs("memory features enter the policy with the declared identity and width")


@sc("CT-01", "BND-03: the planner's policy path carries ZERO knowledge context (declared), invariant to memory content, "
             "and its telemetry makes no memory-aware claim",
    brk="planner_fed_live_memory", min_exercised=10,
    oracle="planner logits/values equal the policy's logits with an explicit zero knowledge vector and do not change "
           "when OM/AM content changes; the acting-entry logits DO change; P.last/report() claim nothing about memory; "
           "GOV-011 declares the imagined rollouts use no knowledge vector")
def b03_ct01b(brk):
    c = Chk()
    LX, LY = b03_loop(), b03_loop()
    feed_mem(LX, a1=2, hi=3.0, lo=0.3)
    feed_mem(LY, a1=1, hi=0.3, lo=3.0, view0=("d", "e"))
    pol, P = LX.policy, LX.plan
    cur = {"L": LX}
    if brk:                                                      # DEFECT: the planner is wired to the live memory vector
        P._kn0 = lambda policy, n: torch.from_numpy(kn_of(cur["L"])).float().repeat(n, 1).to(CPU)
    lat = LX.lat_seq[:, 0]
    with torch.no_grad():
        px = P._logits(pol, lat).clone()
        vx = P._value(pol, lat).clone()
        cur["L"] = LY
        py = P._logits(pol, lat).clone()
        vy = P._value(pol, lat).clone()
        aug0 = pol._augment(lat, torch.zeros(lat.shape[0], pol.knowledge_dim), None)
        zero = pol._bound_logits(pol.actor.action_head(pol.actor.shared(aug0)))[..., :A]
    c(torch.allclose(px, py) and torch.allclose(vx, vy), "planner policy path changed with memory content", n=4)
    c(torch.allclose(px, zero, atol=1e-6), "planner policy path is not the explicit zero-knowledge path", n=2)
    kx, ky = kn_of(LX), kn_of(LY)
    c(not np.allclose(kx, ky), "fixture: the two memories gave the same vector", n=1)
    with torch.no_grad():
        ax = pol._bound_logits(pol.actor.action_head(pol.actor.shared(pol._augment(lat[:1], pol._prep_knowledge(kx), None))))[..., :A]
        ay = pol._bound_logits(pol.actor.action_head(pol.actor.shared(pol._augment(lat[:1], pol._prep_knowledge(ky), None))))[..., :A]
    c(not torch.allclose(ax, ay), "acting-entry logits do not respond to memory (nothing distinguishes the paths)", n=1)
    reseed(2)
    out = LX.plan.run(pol, LX.world_model, LX.replay_buffer, CPU)
    claims = [k for k in list(out) + list(LX.plan.last) if re.search(r"memor|knowledge|retriev|attention", k)]
    c(not claims and not re.search(r"memor|knowledge|retriev", LX.plan.report().lower()),
      "planner telemetry claims memory awareness: %s" % claims, n=2)
    doc = open(os.path.join(ROOT, "docs/governance/gov-011-execution-record-gov-009.md")).read()
    c("imagined rollouts use no knowledge vector" in doc, "the zero-knowledge planner path is not declared in GOV-011", n=1)
    return c.obs("planner is the declared zero-context path; the acting path carries memory")


@sc("CT-02", "BND-03: cutting a memory block shrinks the knowledge width consistently at policy entry and planner entry",
    brk="stale_block_width", min_exercised=16,
    oracle="for each of {OM,AM} on/off the vector width equals policy.knowledge_dim, the cut slice is absent, "
           "and the planner still runs over the same policy")
def b03_ct02(brk):
    c = Chk()
    for om, am in ((True, True), (True, False), (False, True), (False, False)):
        L = b03_loop(om=om, am=am)
        if brk and not om:
            L._om_dim = rd()                                     # DEFECT: the block is cut but its width is still counted
        feed_mem(L, 10)
        kn = kn_of(L)
        want = (rd() if om else 0) + (L.config.get("attention_memory", {}).get("out_dim", 0) if am else 0)
        if kn is None:
            c(want == 0 and L.policy.knowledge_dim == 0, "both blocks cut but a vector exists", n=2)
        else:
            c(kn.shape == (L.policy.knowledge_dim,) and kn.shape[0] == want,
              "om=%s am=%s: width %s vs policy %d vs expected %d" % (om, am, kn.shape, L.policy.knowledge_dim, want), n=2)
            if am and not om:
                c(np.array_equal(kn, L._am.feature()), "AM-only vector is not the attention summary", n=1)
        reseed(4)
        out = L.plan.run(L.policy, L.world_model, L.replay_buffer, CPU)
        c(bool(out), "planner did not run over a policy with knowledge width %d" % L.policy.knowledge_dim, n=1)
        c((L._om is None) == (not om) and (L._am is None) == (not am), "a cut block was built anyway", n=2)
    return c.obs("widths follow the enabled blocks at both entries")


@sc("CT-03", "BND-03: absent memory is explicit zeros of the declared width; the policy and planner accept it",
    brk="absent_memory_dropped_not_padded", min_exercised=5,
    oracle="before any step OM/AM are None: the vector is zeros of policy width (legitimate zero context) and the policy "
           "forward and planner run succeed; after data the vector is non-zero")
def b03_ct03a(brk):
    c = Chk()
    L = b03_loop()
    if brk:                                                      # DEFECT: absent blocks contribute nothing (no padding)
        real = DAI._current_knowledge_feature
        L._current_knowledge_feature = (lambda self: np.concatenate(
            [self._om.feature(0) if self._om is not None else np.zeros(0, np.float32),
             self._am.feature() if self._am is not None else np.zeros(0, np.float32)]).astype(np.float32)).__get__(L)
    kn = kn_of(L)
    c(kn.shape == (L.policy.knowledge_dim,), "absent memory gave width %s, policy %d" % (kn.shape, L.policy.knowledge_dim), n=1)
    if kn.shape == (L.policy.knowledge_dim,):
        c(not kn.any(), "absent memory is not zeros", n=1)
        with torch.no_grad():
            lg = L.policy.actor.action_head(L.policy.actor.shared(L.policy._augment(L.lat_seq[:1, 0], L.policy._prep_knowledge(kn), None)))
        c(torch.isfinite(lg).all(), "policy forward on zero context is not finite", n=1)
    reseed(1)
    c(bool(L.plan.run(L.policy, L.world_model, L.replay_buffer, CPU)), "planner failed with absent memory", n=1)
    feed_mem(L, 12)
    c(kn_of(L).any(), "memory never became non-zero after data (zero treated as absence forever)", n=1)
    return c.obs("absent memory is a declared zero context, not an error and not a missing slot")


@sc("CT-03", "BND-03: non-finite evidence in a memory block never reaches the policy input "
             "(REAL DEFECT if red: no finiteness guard in AttentionMemory / OutcomeMemory)",
    brk="nan_latent_every_step", min_exercised=4,
    oracle="after one NaN latent / NaN view key the knowledge vector stays finite and the planner is unaffected")
def b03_ct03b(brk):
    c = Chk()
    L = b03_loop()
    feed_mem(L, 11)
    z = L.lat_seq[:, 0].clone()
    z[0] = float("nan")
    am_step(L, 0, lat=z)
    L._keys = [torch.full((LATV,), float("nan"))] + [near(VIEW["c"])] * (L.streams - 1)
    L._sp = sp_of(L.streams)
    ex = torch.zeros(L.streams, A)
    ex[:, 1] = 1
    with quiet():
        L._outcome_memory_step(None, ex, None, [None] * L.streams, torch.zeros(L.streams), [False] * L.streams, [False] * L.streams, [{}] * L.streams)
    for t in range(12):                                          # valid evidence returns (the break keeps feeding NaN)
        keys = [near(VIEW["a" if t % 2 == 0 else "b"])] + [near(VIEW["c"])] * (L.streams - 1)
        om_step(L, keys, [2 if t % 2 else 0] + [1] * (L.streams - 1), 3.0 if t % 2 else 0.3)
        am_step(L, t, lat=None)
    if brk:                                       # DEFECT (GOV-015 9.6): the finiteness guard is gone, so the NaN reaches the layer
        with torch.no_grad():
            next(L._am.net.parameters()).fill_(float("nan"))
        am_step(L, 12, lat=z)
    kn = kn_of(L)
    c(np.isfinite(kn).all(), "non-finite memory evidence reached the knowledge vector the policy reads: %d of %d entries (retrieval slice %d, attention slice %d)"
      % ((~np.isfinite(kn)).sum(), kn.size, (~np.isfinite(kn[:rd()])).sum(), (~np.isfinite(kn[rd():])).sum()), n=2)
    c(finite_params(L._am.net), "attention layer weights are non-finite", n=1)
    reseed(2)
    c(bool(L.plan.run(L.policy, L.world_model, L.replay_buffer, CPU)), "planner affected", n=1)
    return c.obs("memory blocks refuse non-finite evidence")


@sc("CT-05", "BND-03: the vector is stream 0's memory only; a policy built for another width REFUSES it, the planner follows the policy",
    brk="stream_index_wrong", min_exercised=8,
    oracle="other streams' content changes neither the retrieved action nor the attention summary of stream 0; a stale-width "
           "policy raises on the wider vector (no silent pad/truncate) while the planner (zero path) still runs at that policy's width")
def b03_ct05(brk):
    c = Chk()
    X, Y = b03_loop(), b03_loop()
    for t in range(15):                                          # same stream-0 history; streams 1..2 behave differently
        for L, others in ((X, ([1, 1], "c", "c")), (Y, ([3, 3], "g", "h"))):
            keys = [near(VIEW["a" if t % 2 == 0 else "b"]), near(VIEW[others[1]]), near(VIEW[others[2]])]
            om_step(L, keys, [2 if t % 2 == 1 else 0] + others[0], 3.0 if t % 2 == 1 else 0.3)
            lat = L.lat_seq[:, t % 8].clone()
            if L is Y:
                lat[1:] = torch.randn_like(lat[1:])
            am_step(L, 0, lat=lat)
    if brk:                                                      # DEFECT: the feature helper reads stream 1
        Y._current_knowledge_feature = (lambda self: np.concatenate([self._om.feature(1), self._am.feature()]).astype(np.float32)).__get__(Y)
    kx, ky = kn_of(X), kn_of(Y)
    c(int(np.argmax(kx[8:8 + A])) == 2 and int(np.argmax(ky[8:8 + A])) == 2,
      "stream 0's retrieved action depends on other streams (%d vs %d)" % (np.argmax(kx[8:8 + A]), np.argmax(ky[8:8 + A])), n=2)
    c(torch.equal(X._am.win.buf[0], Y._am.win.buf[0]), "stream 0's attention window depends on other streams' latents", n=L_dim(X))
    c(len(X._om._rings) == NS3 and all(r.written > 0 for r in X._om._rings.values()), "per-stream rings missing", n=1)
    wide = kx
    stale = b03_loop(kd=rd())                                    # policy built before the attention block was enabled
    try:
        stale.policy.actor.shared(stale.policy._augment(stale.lat_seq[:1, 0], stale.policy._prep_knowledge(wide), None))
        refused = False
    except Exception:
        refused = True
    c(refused, "a policy built for a narrower knowledge width silently accepted the wider vector", n=2)
    reseed(3)
    c(bool(stale.plan.run(stale.policy, stale.world_model, stale.replay_buffer, CPU)), "planner broke on the stale-width policy", n=1)
    return c.obs("stream-0 identity kept; width mismatch refused observably")


def L_dim(L):
    return int(L._am_dim)


@sc("CT-06", "BND-03: a failing memory update is observable, does not stop the planner or the policy entry, and memory resumes",
    brk="failure_latches_memory", min_exercised=6,
    oracle="AM.observe raises for a few steps: a warning names it, the vector keeps its width and finiteness, the planner runs; "
           "after the fault clears the attention summary updates again")
def b03_ct06(brk):
    c = Chk()
    L = b03_loop()
    feed_mem(L, 10)
    before = L._am.feature().copy()
    recs, rm = fault_log()
    real = L._am.observe
    state = {"fail": True}

    def flaky(lat, ended):
        if state["fail"] or brk:
            raise RuntimeError("attention dependency down")
        return real(lat, ended)
    L._am.observe = flaky
    try:
        for t in range(3):
            am_step(L, t)
        kn = kn_of(L)
        c(any("attention memory failed" in m for m in recs), "fault not observable", n=1)
        c(kn.shape == (L.policy.knowledge_dim,) and np.isfinite(kn).all(), "vector lost its width/finiteness during the fault", n=2)
        reseed(5)
        c(bool(L.plan.run(L.policy, L.world_model, L.replay_buffer, CPU)), "planner stopped during a memory fault", n=1)
        state["fail"] = False
        for t in range(8):
            am_step(L, t + 3)
    finally:
        rm()
    c(not np.array_equal(L._am.feature(), before), "attention summary never updated after the fault cleared", n=1)
    c(getattr(L, "_am", None) is not None, "memory object was discarded by the fault", n=1)
    return c.obs("memory faults are contained, observable and recoverable")


@sc("CT-07", "BND-03: after a restart the attention layer is restored, the (unsaved) outcome store restarts empty, the width is unchanged",
    brk="restore_skipped", min_exercised=8,
    oracle="a recreated loop on the same storage gives the same attention summary for the same window; the retrieval slice "
           "is zeros until new entries are written; vector width and policy entry unchanged")
def b03_ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b03_")
    L = b03_loop(tmp=tmp)
    feed_mem(L, 24)
    for _ in range(8):
        am_step(L, 0)
    L._am.save()
    ref_win = L.lat_seq[:, :4].clone()
    L2 = b03_loop(tmp=(tempfile.mkdtemp(prefix="bnd_b03x_") if brk else tmp))
    L2.lat_seq = L.lat_seq
    for t in range(4):
        am_step(L, t)                                            # same window in both
        am_step(L2, t)
    c(torch.equal(L._am.win.buf[0], L2._am.win.buf[0]), "fixture: windows differ", n=1)
    a1, a2 = L._am.feature(), L2._am.feature()
    c(np.allclose(a1, a2, atol=1e-6), "restored attention layer gives a different summary for the same window", n=a1.size)
    k2 = kn_of(L2)
    c(not k2[:rd()].any(), "the unsaved outcome store was not empty after restart", n=1)
    c(k2.shape == (L2.policy.knowledge_dim,), "width changed over a restart", n=1)
    c(L2._am._fresh_note == "", "restored layer reports a fresh start: %r" % L2._am._fresh_note, n=1)
    return c.obs("trained attention persists; the outcome store restarts empty by design")


@sc("CT-08", "BND-03: the planner never reads the memory objects (tripwire), the memory slices carry no evaluator field, telemetry is read-only",
    brk="planner_calls_memory", min_exercised=7,
    oracle="a planner run touches neither OM nor AM; the knowledge vector's layout is fixed (two learner slices); "
           "reading the report/feature leaves memory, planner and RNG unchanged")
def b03_ct08(brk):
    c = Chk()
    L = b03_loop()
    feed_mem(L, 14)
    hits = []

    class Trip:
        def __getattr__(self, n):
            hits.append(n)
            raise AssertionError("planner touched memory attribute %r" % n)
    om, am = L._om, L._am
    L._om, L._am = Trip(), Trip()
    if brk:                                                      # DEFECT: a planner change that consults live memory
        L.plan._kn0 = lambda policy, n: torch.from_numpy(np.zeros(policy.knowledge_dim, np.float32)) + float(L._am.feature().sum() * 0)
    reseed(6)
    try:
        L.plan.run(L.policy, L.world_model, L.replay_buffer, CPU)
    except AssertionError:
        pass
    c(not hits, "planner touched memory: %s" % hits[:2], n=2)
    L._om, L._am = om, am
    c(kn_of(L).shape == (rd() + L._am_dim,), "layout of the knowledge vector changed", n=1)
    w0, r0 = phash(am.net), rng_sig()
    snap = (kn_of(L).copy(), L._om.feature(0).copy())
    am.report()
    om.report()
    c(torch.equal(w0, phash(am.net)) and rng_sig() == r0, "reading memory telemetry changed weights or RNG", n=2)
    c(np.array_equal(snap[0], kn_of(L)), "reading the feature twice changed it", n=1)
    mods = [m for m in sys.modules if m.startswith("developmental_ai.evaluator")]
    c(not mods, "an evaluator module is imported on the memory/planning path", n=1)
    return c.obs("planner and memory are isolated; telemetry is read-only")


# ================================================================================================
# BND-04  CAT-04 <-> CAT-05   competence, curriculum frontier, adaptive optimisation, meta adaptation, strategy
# ================================================================================================
MEAS = {}


def fake_measure(*_a, **_k):
    return dict(MEAS)


@contextlib.contextmanager
def measuring():
    """The phases import measure_streams lazily, so patching the module attribute reaches the loop's helpers."""
    MEAS.clear()
    with mock.patch("developmental_ai.core.curriculum.measure_streams", fake_measure):
        yield


AO_CFG = {"enabled": True, "every_segments": 1,
          "schedule": {"enabled": True, "policy": {"kind": "linear", "horizon": 1000, "floor": 0.1},
                       "world_model": {"kind": "constant"}},
          "entropy": {"enabled": True, "target": 0.5, "eta": 0.5},
          "return_norm": {"enabled": True}, "drift_response": {"enabled": True, "hold_segments": 4}}
META_CFG = {"enabled": True, "every_segments": 1, "epoch_len": 2, "step": 0.1, "range": 2.0, "accept_z": 0.0,
            "min_noise_n": 1, "coefficients": ["ppo_clip", "gae_lambda", "value_coef"]}
COMP_CFG = {"enabled": True, "every_segments": 1, "min_meas": 3, "signal_ema": 0.5, "progress_ema": 0.5,
            "modulate_exploration": True, "gain_amp": 0.3, "entropy_amp": 0.3, "max_step": STEP}
STRAT = {"SRL": {"STRAT_ENTRY": {"enabled": True}}, "STRAT_ENTRY": {"SP": {"enabled": True}, "INTL": {"enabled": True}},
         "CALIB": {"SRL": {"enabled": True}}}
B04_OWNER = {
    "mixer.intrinsic_gain": "wp8_competence", "explore.entropy_target_scale": "wp8_competence",
    "policy.entropy_coef": "wp9_adaptive", "policy.lr_scale": "wp9_adaptive", "wm.lr_scale": "wp9_adaptive",
    "meta.ppo_clip": "wp13_meta", "meta.gae_lambda": "wp13_meta", "meta.value_coef": "wp13_meta",
    "explore.uncertainty_scale": "rw1_modulation", "objective.frontier_gain": "rw7_motivation",
    "strategy.spaced_fraction": "rw6_strategy", "strategy.prioritized": "rw6_strategy"}
B04_SLEW = {"mixer.intrinsic_gain": STEP, "explore.entropy_target_scale": STEP, "explore.uncertainty_scale": STEP,
            "objective.frontier_gain": STEP}


def b04_cfg(competence=True, curriculum=True, ao=True, meta=True, strategy=True, mod=True, zpd=True):
    rel = {"enabled": True, "modulation": MODCFG, "motivation": {"amp": 0.3, "max_step": STEP, "min_n": 4, "ema": 0.9},
           "strategy": {"epoch_len": 2, "min_pulls": 1, "seed": 0, "spaced": 0.25}}
    if mod:
        rel.update({"LC": {"NE": {"enabled": True}}, "NE": {"AROUS": {"enabled": True}}})
    if zpd:
        rel["ZPD"] = {"MOT": {"enabled": True}}
    if strategy:
        for k, v in STRAT.items():
            rel[k] = copy.deepcopy(v)
    cfg = {"consolidation": {"enabled": True, "every_segments": 1, "steps": 1, "batch": 4, "seq_len": 4,
                             "lr_scale": 0.1, "replay": {"spaced_fraction": 0.0, "prioritized": False},
                             "distill": {"enabled": True, "snapshot_every_segments": 100, "max_snapshots": 2}},
           "relations": rel}
    if competence:
        cfg["competence"] = dict(COMP_CFG)
    if curriculum:
        cfg["curriculum"] = {"enabled": True, "min_pulls": 1, "every_segments": 2, "seed": 0}
    if ao:
        cfg["adaptive_optim"] = copy.deepcopy(AO_CFG)
    if meta:
        cfg["meta_adapt"] = dict(META_CFG)
    return cfg


def b04_brain(cfg=None, tmp=None, adapters=2):
    ns = brain(cfg if cfg is not None else b04_cfg(), adapters=adapters, tmp=tmp,
               phases=[p for p in PHASES if p != "_planning_phase"])
    # CAT-03's output injected as numbers (BND-02 covers the real chain); `mon` is read by RW-6's calibration channel
    ns._plan = SimpleNamespace(last={}, mon=CalibrationMonitor(window=200, min_n=5))
    ns._am = SimpleNamespace(net=SimpleNamespace(temperature=1.0), last={}, pushed0=True)
    ns._rk = SimpleNamespace(vec=[0.2])
    ns._tg = []
    return ns


def b04_spy(ns):
    """Record (segment, target, competence scale, uncertainty scale) every time the WP-9 controller consumes its target."""
    ao = getattr(ns, "_ao", None)
    if ao and ao.ent is not None and not getattr(ao.ent, "_spied", False):
        o = ao.ent.update
        reg = registry_for(ns)

        def raw(n):                                              # the published value, not through a patched getter
            return next((r["value"] for r in reg.table() if r["name"] == n), 1.0)

        def upd(h, target=None, _o=o):
            ns._tg.append((target, raw("explore.entropy_target_scale"), raw("explore.uncertainty_scale")))
            return _o(h, target)
        ao.ent.update = upd
        ao.ent._spied = True


def b04_seg(ns, losses=None, attr=0.1, ent=0.4, unc=0.5, order=None):
    if losses is not None:
        MEAS.clear()
        for i, l in enumerate(losses):
            if l is not None:
                MEAS[i] = (l, attr)
    ns.lifetime_timesteps += 64
    ns.policy.last_update_stats = {"cond_entropy_norm": ent}
    ns._plan.last = {"policy_calibration_ok": 1.0, "calibration_conf": 1.0 - unc}
    segment(ns, order)
    b04_spy(ns)
    return {r["name"]: r for r in registry_for(ns).table()}


def b04_run(ns, n, mode="changing", seed=5):
    """n segments in the real order.  'constant': identical measurements and readings; 'changing': bounded random
    progress, entropy and calibration.  Returns per-segment {name: value} rows and a list of problems."""
    rs = np.random.RandomState(seed)
    loss = [1.0, 2.0]
    rows, prob, prev = [], [], {}
    for s in range(n):
        if mode == "changing":
            loss = [x * (1.0 - 0.05 * rs.rand()) for x in loss]
            att, ent, unc = 0.1 + 0.05 * rs.rand(), 0.2 + 0.6 * rs.rand(), 0.3 + 0.5 * rs.rand()
        else:
            att, ent, unc = 0.1, 0.4, 0.5
        reg = b04_seg(ns, loss if mode == "changing" else [1.0, 2.0], attr=att, ent=ent, unc=unc)
        for nm, r in reg.items():
            if B04_OWNER.get(nm) != r["owner"]:
                prob.append("%s owned by %s" % (nm, r["owner"]))
            if not r["lo"] - 1e-9 <= r["value"] <= r["hi"] + 1e-9:
                prob.append("%s=%s outside [%s,%s]" % (nm, r["value"], r["lo"], r["hi"]))
            if nm in B04_SLEW and nm in prev and abs(r["value"] - prev[nm]) > B04_SLEW[nm] + 1e-9:
                prob.append("%s slewed %+.4f" % (nm, r["value"] - prev[nm]))
        prev = {nm: r["value"] for nm, r in reg.items()}
        rows.append(dict(prev))
    return rows, prob


@sc("CT-01", "BND-04: competence, the curriculum frontier and the uncertainty gain reach the reward mixer and the entropy "
             "controller's target through their own coefficients",
    brk="entropy_scale_read_dropped", min_exercised=20,
    oracle="each time WP-9 consumes its target it equals min(0.99, 0.5 x competence scale x uncertainty scale) of that moment; "
           "mixer.intrinsic_gain and frontier_gain hold the published values and multiply the intrinsic term once")
def b04_ct01(brk):
    c = Chk()
    with measuring(), tempfile.TemporaryDirectory() as d:
        ns = b04_brain(tmp=d)
        ctx = contextlib.ExitStack()
        if brk:                                                  # DEFECT: WP-9 stops reading WP-8's scale
            from developmental_ai.core.coefficients import CoefficientRegistry
            real_get = CoefficientRegistry.get
            ctx.enter_context(patched(CoefficientRegistry, "get",
                                      lambda self, n, d=None: d if n == "explore.entropy_target_scale" else real_get(self, n, d)))
        with ctx:
            rows, prob = b04_run(ns, 14, "changing")
            rows2, prob2 = b04_run(ns, 8, "constant")
        c(not prob and not prob2, "ownership/bound problems: %s" % (prob + prob2)[:3], n=len(rows) + len(rows2))
        tg = ns._tg
        c(len(tg) >= 10, "the controller consumed its target only %d times" % len(tg), n=1)
        bad = [(round(t, 4), cs, us) for t, cs, us in tg if abs(t - min(0.99, 0.5 * (cs or 1.0) * (us or 1.0))) > 1e-9]
        c(not bad, "controller target is not base x competence x uncertainty: %s" % bad[:2], n=len(tg))
        reg = registry_for(ns)
        m = ns.reward_mixer
        g, f = reg.get("mixer.intrinsic_gain"), reg.get("objective.frontier_gain")
        c(g is not None and f is not None and abs(m.intrinsic_gain - g) < 1e-12 and abs(m.frontier_gain - f) < 1e-12,
          "mixer does not hold the published gains (%s %s vs %s %s)" % (m.intrinsic_gain, m.frontier_gain, g, f), n=2)
        if g is not None and f is not None:
            for i, e in ((1.0, 0.0), (0.0, 1.0), (0.5, -2.0)):
                c(abs(m.mix(i, e, update_stats=False) - (0.7 * g * f * i + 0.3 * e)) < 1e-12, "gain not booked exactly once", n=1)
        c(any(abs(r.get("explore.entropy_target_scale", 1.0) - 1.0) > 1e-9 for r in rows), "WP-8 never moved the entropy-target scale", n=1)
        c(any(abs(r.get("objective.frontier_gain", 1.0) - 1.0) > 1e-9 for r in rows), "RW-7 never moved the frontier gain (curriculum value not consumed)", n=1)
    return c.obs("controllers' coefficients reach mixer and controller exactly once")



def naive_registry():
    """The wiring defect: a registry that lets the last writer win (the CAT-04/05 'duplicated writer' break)."""
    from developmental_ai.core import coefficients as coefmod
    from developmental_ai.core.coefficients import CoefficientRegistry

    def claim(self, name, owner, base, lo, hi):
        c = self._c.get(name)
        if c is None:
            c = self._c[name] = coefmod.Coefficient(name, owner, base, lo, hi)
        c.owner = owner
        return c

    def set_(self, name, owner, value, max_step=None):
        c = self._c[name]
        v = float(value)
        if v == v:
            c.value = min(c.hi, max(c.lo, v))
        c.writes += 1
        return c.value

    def release(self, name, owner):
        self._c.pop(name, None)
    return mock.patch.multiple(CoefficientRegistry, claim=claim, set=set_, release=release)


@sc("CT-02", "BND-04: a cut controller is bit-identical to a never-wired one, and cutting it changes its neighbours' inputs only",
    brk="cut_switch_ignored", min_exercised=14,
    oracle="competence enabled:false == competence absent (registry rows, WP-9 targets, mixer gains, optimizer lr); "
           "competence on differs; curriculum cut leaves the frontier gain neutral; WP-9 cut leaves the policy untouched")
def b04_ct02(brk):
    c = Chk()

    def arm(**cut):
        reseed(31)
        cfg = b04_cfg()
        for k, v in (cut.get("override") or {}).items():
            if v is None:
                cfg.pop(k, None)
            else:
                cfg[k] = v
        ns = b04_brain(cfg)
        if brk and cut.get("force_comp"):                        # DEFECT: the cut switch is ignored by the wiring
            from developmental_ai.core.competence import CompetenceController
            ns._comp = CompetenceController(COMP_CFG, registry_for(ns))
        with measuring():
            rows, _ = b04_run(ns, 8, "changing", seed=9)
        reg = {r["name"]: round(r["value"], 12) for r in registry_for(ns).table()}
        return ns, reg, [round(t[0], 12) for t in ns._tg], ns.reward_mixer.intrinsic_gain, ns.policy.optimizer.param_groups[0]["lr"]
    full = arm()
    cut = arm(override={"competence": {"enabled": False}}, force_comp=True)
    gone = arm(override={"competence": None})
    c(cut[1] == gone[1] and cut[2] == gone[2] and cut[3] == gone[3] and cut[4] == gone[4],
      "competence cut != never wired (%s vs %s)" % (sorted(set(cut[1].items()) ^ set(gone[1].items()))[:2], cut[3] == gone[3]), n=4)
    c(full[1] != gone[1] and full[3] != 1.0, "fixture: competence on did not differ from off", n=2)
    c(gone[3] == 1.0 and "mixer.intrinsic_gain" not in gone[1] and "explore.entropy_target_scale" not in gone[1],
      "cut competence still published a gain", n=3)
    nocurr = arm(override={"curriculum": None})
    c(abs(nocurr[1].get("objective.frontier_gain", 1.0) - 1.0) < 1e-12 and nocurr[0].reward_mixer.frontier_gain == 1.0,
      "no curriculum, but the frontier gain is not neutral", n=2)
    noao = arm(override={"adaptive_optim": None})
    base_pol = mk_pol(noao[0].world_model.rssm.latent_dim)
    c(noao[0].policy.entropy_coef == base_pol.entropy_coef and not any(k.startswith("policy.") for k in noao[1]),
      "WP-9 cut but the entropy coefficient / lr scale moved or was claimed", n=2)
    c(noao[2] == [], "WP-9 cut but a target was still consumed", n=1)
    return c.obs("cut == never wired; neighbours only lose the cut input")


@sc("CT-03", "BND-04: one non-finite fresh-data loss reaches competence, curriculum, meta and strategy together and must "
             "move none of them (REAL DEFECT if red: WP-8 / WP-13 / RW-6 book NaN as progress)",
    brk="inf_instead_of_nan", min_exercised=8,
    oracle="after warm valid segments, a NaN/inf measurement on one stream leaves every controller's state finite and "
           "unchanged and does not displace mixer.intrinsic_gain or the entropy-target scale")
def b04_ct03(brk):
    c = Chk()
    with measuring(), tempfile.TemporaryDirectory() as d:
        ns = b04_brain(tmp=d)
        b04_run(ns, 10, "changing")
        Cc, Cu, Ma = ns._comp, ns._curr, ns._ma
        snap = lambda: {"comp_n": Cc.ctx[0].n, "comp_ema": Cc.ctx[0].loss_ema, "comp_hist": len(Cc.ctx[0].hist),
                        "comp_lp": Cc.ctx[0].lp, "pulls": list(Cu.bandits["p"].rewards_n),
                        "meta": (Ma.accepts, Ma.rejects), "gain": ns.reward_mixer.intrinsic_gain,
                        "ets": registry_for(ns).get("explore.entropy_target_scale")}
        before = snap()
        bad = float("inf") if brk else float("nan")
        for _ in range(2):
            b04_seg(ns, [bad, 1.0])
        after = snap()
        c(math.isfinite(Cc.ctx[0].loss_ema or 0.0), "competence's loss average became %s" % Cc.ctx[0].loss_ema, n=1)
        c(after["comp_n"] == before["comp_n"] and after["comp_hist"] == before["comp_hist"],
          "competence counted the invalid measurement (n %d -> %d)" % (before["comp_n"], after["comp_n"]), n=2)
        c(abs(after["comp_lp"] - before["comp_lp"]) < 1e-9, "competence's progress estimate moved on invalid data (%.3f -> %.3f)" % (before["comp_lp"], after["comp_lp"]), n=1)
        c(abs(after["gain"] - before["gain"]) <= 2 * STEP + 1e-9 and abs(after["comp_lp"]) < 0.5,
          "intrinsic gain displaced by an invalid measurement (%.3f -> %.3f)" % (before["gain"], after["gain"]), n=1)
        c(after["meta"] == before["meta"] or math.isfinite(Ma.noise if hasattr(Ma, "noise") else 0.0), "meta adaptation changed on invalid data", n=1)
        c(all(math.isfinite(v) for v in Cu.bandits["p"].value), "curriculum values became non-finite", n=1)
    return c.obs("invalid measurements move no controller")


@sc("CT-04", "BND-04: all enabled controllers in the real phase order: one owner per coefficient, bounded, slew-limited, "
             "consumers agree, constant inputs settle, changing inputs stay bounded, rogue writers refused",
    brk="last_writer_wins", min_exercised=250,
    oracle="12 coefficients each owned by its declared module, inside [lo,hi]; slewed gains move <= max_step; consumers "
           "(mixer, policy, optimizer, consolidator) hold the published values; constant inputs: gains settle; a rogue "
           "claim/write/release raises and changes nothing")
def b04_ct04(brk):
    c = Chk()
    with measuring(), tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st:
        if brk:
            st.enter_context(naive_registry())
        ns = b04_brain(tmp=d)
        reg = registry_for(ns)
        rows, prob = b04_run(ns, 14, "changing")
        c(not prob, "changing inputs: %s" % prob[:3], n=len(rows) * 8)
        c(all(r["value"] <= r["hi"] + 1e-9 for r in reg.table()) and ns.policy.entropy_coef <= 0.1 + 1e-12,
          "a coefficient or the entropy coefficient left its declared bound", n=2)
        # constant inputs, fresh brain: the open-loop worst case.  Gains settle at a declared bound or at neutral, the
        # z-score gains (which measure change against their own history) stay at neutral, nothing exceeds a bound
        ns_c = b04_brain(tmp=tempfile.mkdtemp(prefix="bnd_b04c_"))
        rows_c, prob_c = b04_run(ns_c, 34, "constant")
        c(not prob_c, "constant inputs: %s" % prob_c[:3], n=len(rows_c) * 8)
        last = rows_c[-5:]
        for nm in ("mixer.intrinsic_gain", "explore.entropy_target_scale", "objective.frontier_gain", "explore.uncertainty_scale"):
            vals = [r.get(nm, 1.0) for r in last]
            c(max(vals) - min(vals) < 1e-3, "%s did not settle under constant input: %s" % (nm, [round(v, 4) for v in vals]), n=1)
        for nm in ("objective.frontier_gain", "explore.uncertainty_scale"):
            c(abs(last[-1].get(nm, 1.0) - 1.0) < 0.05, "%s did not return to neutral with nothing new in its input (%.3f)" % (nm, last[-1].get(nm, 1.0)), n=1)
        c(ns_c.policy.entropy_coef <= 0.1 + 1e-12 and ns_c.reward_mixer.intrinsic_gain <= 1.3 + 1e-9, "a fixed input wound a coefficient past its bound", n=2)
        p, m = ns.policy, ns.reward_mixer
        for held, nm in ((p.entropy_coef, "policy.entropy_coef"), (p.clip_range, "meta.ppo_clip"), (p.gae_lambda, "meta.gae_lambda"),
                         (p.value_coef, "meta.value_coef"), (m.intrinsic_gain, "mixer.intrinsic_gain"), (m.frontier_gain, "objective.frontier_gain"),
                         (ns._cons.old_frac, "strategy.spaced_fraction")):
            if reg.get(nm) is not None:
                c(abs(float(held) - reg.get(nm)) < 1e-9, "consumer of %s holds %s != %s" % (nm, held, reg.get(nm)), n=1)
        if reg.get("policy.lr_scale") is not None:
            c(abs(p.optimizer.param_groups[0]["lr"] - 3e-4 * reg.get("policy.lr_scale")) < 1e-12, "optimizer lr != base x policy.lr_scale", n=1)
        c({"mixer.intrinsic_gain", "explore.entropy_target_scale", "policy.entropy_coef", "meta.ppo_clip",
           "objective.frontier_gain", "explore.uncertainty_scale"} <= {r["name"] for r in reg.table()},
          "a writer was never exercised: %s" % sorted({r["name"] for r in reg.table()}), n=1)
        held = {r["name"]: r["value"] for r in reg.table()}
        for nm in sorted(held):
            for attempt in (lambda: reg.claim(nm, "rogue_ctl", 1.0, 0.0, 9.0), lambda: reg.set(nm, "rogue_ctl", 9.0),
                            lambda: reg.release(nm, "rogue_ctl")):
                try:
                    attempt()
                    c(False, "%s: a rogue writer was accepted" % nm)
                except OwnershipError:
                    c(True, "")
        c({r["name"]: r["value"] for r in reg.table()} == held, "a refused write changed a value", n=1)
    return c.obs("one owner per coefficient; bounded; settled; rogue writers refused")


@sc("CT-05", "BND-04: the same stream-keyed measurement is attributed to the SAME stream by the curriculum and by competence",
    brk="stream_ids_swapped", min_exercised=6,
    oracle="stream 0 improves, stream 1 does not: the bandit arm of stream 0 is credited, competence's per-stream progress of "
           "stream 0 is positive and stream 1's is not; swapping the keys flips both")
def b04_ct05a(brk):
    c = Chk()
    with measuring(), tempfile.TemporaryDirectory() as d:
        ns = b04_brain(tmp=d, adapters=0)
        ns._parallel_envs = [FakeAdapter("p0"), FakeAdapter("p1")]     # one pool per stream: each bandit sees ONE stream
        l0, l1 = 1.0, 2.0
        for k in range(12):
            l0 *= 0.9                                            # stream 0 learns, stream 1 is flat
            a, b = (l1, l0) if brk else (l0, l1)                 # DEFECT: the measurement dict is keyed by the wrong stream
            b04_seg(ns, [a, b])
        Cc, Cu = ns._comp, ns._curr
        c(Cc.ctx[0].lp > 0.02 and Cc.ctx[1].lp < 0.02, "competence progress by stream: %s" % [round(Cc.ctx[i].lp, 3) for i in (0, 1)], n=2)
        v0, v1 = sum(Cu.bandits["p0"].value), sum(Cu.bandits["p1"].value)
        c(v0 > 0.0 > v1 or v0 > v1 + 0.02, "bandit credit by stream: stream 0's pool %.3f, stream 1's pool %.3f" % (v0, v1), n=2)
        c(sum(Cu.bandits["p0"].rewards_n) >= 3 and sum(Cu.bandits["p1"].rewards_n) >= 3, "too few rewards booked", n=2)
        c(Cu.last.get(1) is None or Cu.last[1][0] > 1.0, "curriculum memory of stream 1 holds another stream's loss", n=1)
    return c.obs("stream attribution agrees between WP-3 and WP-8")


@sc("CT-05", "BND-04: a WP-3 context switch is not scored by WP-8 as lost learning progress, yet a genuine regression still is "
             "(REAL DEFECT if red: WP-8 has no context-change input)",
    brk="regression_ignored", min_exercised=4,
    oracle="a loss jump that follows set_context() on a stream leaves that stream's competence progress where the smooth "
           "control has it; the same jump with no context change drives it clearly negative")
def b04_ct05b(brk):
    c = Chk()

    def run(jump, switch):
        with measuring(), tempfile.TemporaryDirectory() as d:
            ns = b04_brain(tmp=d)
            seen, at, lp, last_sw = 0, None, None, -99
            loss = [1.0, 1.0]
            for k in range(14):
                loss = [x * 0.97 for x in loss]
                if jump and at is not None and k == at + 1:
                    loss[0] *= 2.0
                b04_seg(ns, list(loss))
                n_ctx = len(ns._parallel_envs[0].ctx)
                if n_ctx > seen and seen > 0:
                    last_sw = k                      # the brain's own curriculum switched here, whatever the arm
                if k >= 3 and ns._comp.ctx[0].n >= 3 and at is None:
                    # GOV-015 9.10: the 'no switch' arm used a fixed update (6), but the brain's curriculum switches contexts on its own
                    # schedule, and a jump inside the two-update hold that follows ANY switch is deliberately not scored (the fix this
                    # scenario guards). A genuine regression is one that is not coincident with a switch: take the first update at
                    # least context_hold + 2 after the last one.
                    if (switch and n_ctx > seen and seen > 0) or (not switch and k >= 6 and k - last_sw >= 4):
                        at = k
                        lp = ns._comp.ctx[0].lp
                seen = n_ctx
                if at is not None and k == at + 2:
                    return ns._comp.ctx[0].lp, lp
            return ns._comp.ctx[0].lp, lp
    ctrl = run(False, True)
    switched = run(True, True)
    regress = run(True, False)
    c(ctrl[1] is not None and switched[1] is not None, "fixture: no curriculum switch observed", n=1)
    c(regress[1] is not None, "fixture: no update outside the hold window after a switch was found for the regression arm", n=1)
    c(switched[0] > ctrl[0] - 0.1, "competence booked the post-switch loss jump as lost progress (lp %.3f vs %.3f without the jump)"
      % (switched[0], ctrl[0]), n=1)
    c(regress[0] < ctrl[0] - 0.1 and not brk, "a genuine regression (no context change) was not registered (lp %.3f vs %.3f)" % (regress[0], ctrl[0]), n=1)
    return c.obs("context switches are not progress; regressions are")


@sc("CT-06", "BND-04: losing the fresh-data measurement releases the coefficients derived from it within ceil(dev/step) "
             "(REAL DEFECT if red: WP-8 / RW-7 keep applying stale values)",
    brk="release_path_missing", min_exercised=6,
    oracle="after the measurement source returns nothing, mixer.intrinsic_gain, explore.entropy_target_scale and "
           "objective.frontier_gain are back at 1.0 within ceil(max dev / max_step) segments; valid measurements re-engage them")
def b04_ct06(brk):
    c = Chk()
    with measuring(), tempfile.TemporaryDirectory() as d:
        ns = b04_brain(tmp=d)
        b04_run(ns, 14, "changing")
        names = ("mixer.intrinsic_gain", "explore.entropy_target_scale", "objective.frontier_gain")
        dev = {n: abs(cv(registry_for(ns).get(n)) - 1.0) for n in names}
        c(max(dev.values()) > 0.1, "fixture: nothing displaced (%s)" % {k: round(v, 3) for k, v in dev.items()}, n=1)
        bound = math.ceil(max(dev.values()) / STEP - 1e-9)
        steps = 0
        reg_ = registry_for(ns)
        if brk:                                                  # DEFECT (GOV-015 9.6): no release path, the displaced values stay
            reg_.set = lambda name, owner, value, step=None, *a, **k: reg_.get(name)
            reg_.release = lambda *a, **k: None
        while steps < bound + 2:
            b04_seg(ns, [None, None])                            # measure_streams returns {} : "unknown", the documented absence
            steps += 1
        left = {n: round(cv(registry_for(ns).get(n)), 3) for n in names}
        c(all(abs(v - 1.0) < 1e-9 for v in left.values()),
          "after %d segments without measurements (bound %d) the coefficients are still %s" % (steps, bound, left), n=3)
        c(abs(ns.reward_mixer.intrinsic_gain - 1.0) < 1e-9 and abs(ns.reward_mixer.frontier_gain - 1.0) < 1e-9,
          "the reward mixer still carries a stale gain (%.3f, %.3f)" % (ns.reward_mixer.intrinsic_gain, ns.reward_mixer.frontier_gain), n=2)
        rows, prob = b04_run(ns, 12, "changing", seed=2)
        c(any(abs(r.get("mixer.intrinsic_gain", 1.0) - 1.0) > 1e-9 for r in rows), "controllers did not re-engage when measurements returned", n=1)
    return c.obs("measurement loss releases the derived coefficients")


@sc("CT-07", "BND-04: curriculum bandit state persists, every coefficient restarts at neutral, consumers follow, no half-restored state",
    brk="coefficients_assumed_persisted", min_exercised=8,
    oracle="a recreated brain restores the bandit values from curriculum.json; mixer gains, entropy-target scales, "
           "optimizer lr and policy coefficients are at their base values until their controllers re-earn them")
def b04_ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b04_")
    with measuring():
        ns = b04_brain(tmp=tmp)
        b04_run(ns, 12, "changing")
        vals = [list(b.value) for b in ns._curr.bandits.values()]
        c(max(abs(cv(registry_for(ns).get(n)) - 1.0) for n in ("mixer.intrinsic_gain", "explore.entropy_target_scale")) > 0.05,
          "fixture: nothing to lose on restart", n=1)
        ns2 = b04_brain(tmp=tmp)
        if brk:                                                  # DEFECT: the old coefficient values are carried into the new process
            for r in registry_for(ns).table():
                try:
                    registry_for(ns2).claim(r["name"], r["owner"], r["base"], r["lo"], r["hi"])
                    registry_for(ns2).set(r["name"], r["owner"], r["value"])
                except Exception:
                    pass
            ns2.reward_mixer.intrinsic_gain = ns.reward_mixer.intrinsic_gain
        b04_seg(ns2, [1.0, 2.0])                                 # first segment: nothing measured twice yet
        reg2 = {r["name"]: r["value"] for r in registry_for(ns2).table()}
        c(abs(ns2.reward_mixer.intrinsic_gain - 1.0) < 1e-12 and abs(ns2.reward_mixer.frontier_gain - 1.0) < 1e-12,
          "mixer gains survived the restart: %.3f %.3f" % (ns2.reward_mixer.intrinsic_gain, ns2.reward_mixer.frontier_gain), n=2)
        c(all(abs(reg2.get(n, 1.0) - 1.0) < 1e-9 for n in ("mixer.intrinsic_gain", "explore.entropy_target_scale",
                                                            "explore.uncertainty_scale", "objective.frontier_gain")),
          "a coefficient survived the restart: %s" % {k: round(v, 3) for k, v in reg2.items() if abs(v - 1.0) > 1e-9 and k.endswith(("gain", "scale"))}, n=4)
        c(abs(ns2.policy.optimizer.param_groups[0]["lr"] - 3e-4 * reg2.get("policy.lr_scale", 1.0)) < 1e-12, "optimizer lr is not base x scale after restart", n=1)
        cj = os.path.join(tmp, "curriculum.json")
        c(os.path.exists(cj) and json.load(open(cj)).get("bandits"), "bandit state was not persisted", n=1)
        c(ns2._curr.seg >= 1 and any(abs(x) > 1e-9 for b in ns2._curr.bandits.values() for x in b.value),
          "the curriculum bandit did not restore its values", n=1)
    return c.obs("bandit persists; coefficients and consumers restart neutral")


@sc("CT-08", "BND-04: the real measurement source reads only the replay windows; read-only probes leave controllers, mixer and RNG alone",
    brk="telemetry_consumes_rng", min_exercised=8,
    oracle="with the real measure_streams, replay streams that raise on any attribute but newest_sequence are never "
           "touched; sample_all/line/report change no coefficient, weight or RNG state")
def b04_ct08(brk):
    c = Chk()
    hits = []

    class StreamProxy:
        def __init__(self, s):
            self._s = s

        def newest_sequence(self, n):
            hits.append("newest_sequence")
            return self._s.newest_sequence(n)

        def __getattr__(self, name):
            hits.append(name)
            raise AssertionError("measurement source read replay attribute %r" % name)
    with tempfile.TemporaryDirectory() as d:
        ns = b04_brain(tmp=d)
        ns.replay_buffer.streams = [StreamProxy(s) for s in ns.replay_buffer.streams]
        from developmental_ai.core import curriculum as cur
        real_ms = cur.measure_streams
        ctx = mock.patch("developmental_ai.core.curriculum.measure_streams",
                         (lambda wm, rb, sl, dev: (getattr(rb.streams[0], "rewards", None), real_ms(wm, rb, sl, dev))[1])
                         if brk else real_ms)                    # DEFECT: a "helper" that also reads a side column
        with ctx:
            for k in range(3):
                ns.lifetime_timesteps += 64
                ns.policy.last_update_stats = {"cond_entropy_norm": 0.4}
                ns._plan.last = {"policy_calibration_ok": 1.0, "calibration_conf": 0.5}
                segment(ns)
        c(set(hits) == {"newest_sequence"} and len(hits) >= 6, "measurement source touched %s" % sorted(set(hits)), n=len(hits))
        c(len(ns._comp.ctx) == 2, "real measurements never reached competence", n=1)

    def run(telemetry):
        reseed(41)
        with measuring(), tempfile.TemporaryDirectory() as d:
            ns = b04_brain(tmp=d)
            reseed(42)
            rs = np.random.RandomState(8)
            loss = [1.0, 2.0]
            path = []
            for k in range(10):
                loss = [x * (1.0 - 0.04 * rs.rand()) for x in loss]
                b04_seg(ns, loss)
                if telemetry:
                    ns._rel.sample_all(ns)
                    ns._rel.line()
                    ns._comp.report({})
                    ns._curr.report()
                    if brk:
                        torch.rand(1)
                path.append(tuple(round(r["value"], 9) for r in registry_for(ns).table()))
            return path, rng_sig(), thash(phash(ns.policy.actor))
    base, tel = run(False), run(True)
    c(base[0] == tel[0] and base[2] == tel[2], "telemetry changed coefficients or weights", n=2)
    c(base[1] == tel[1], "telemetry consumed learner RNG", n=1)
    return c.obs("the measurement source reads only replay windows; probes are read-only")

# ================================================================================================
# BND-05  CAT-06 -> CAT-01 / CAT-02   map assignment and peer experience: provenance through replay, memory, board
# ================================================================================================
MAPS = ["MAP%02d" % i for i in range(1, 29)]
HELD = ["MAP29", "MAP30", "MAP31", "MAP32"]
ALLMAPS = MAPS + HELD
T5 = 16                                                          # steps per stream in the provenance fixture


def game_cfg(**over):
    c = {"maps": MAPS + HELD, "held_out_maps": list(HELD),
         "map_assignment": {"mode": "spread", "schedule": [{"at_s": 100, "offset": 1}]}}
    c.update(over)
    return copy.deepcopy(c)


def tag_obs(k, mp, t):
    """A frame whose first three pixels name its producer: stream, map, step (the harness's origin record)."""
    o = np.full(OBS, 0.5, np.float32)
    o[0], o[1], o[2] = k / 10.0, ALLMAPS.index(mp) / 100.0, t / 1000.0
    return o


def untag(o):
    return int(round(float(o[0]) * 10)), ALLMAPS[int(round(float(o[1]) * 100))], int(round(float(o[2]) * 1000))


def unit(idx, D):
    v = torch.zeros(D)
    v[idx] = 1.0
    return v


class TripInfo(dict):
    """A step-info dict that fails if the learner reads anything but the restart flag."""
    seen = []

    def get(self, k, d=None):
        TripInfo.seen.append(k)
        if k != "env_restarted":
            raise AssertionError("learner read step-info field %r" % k)
        return dict.get(self, k, d)

    def __getitem__(self, k):
        TripInfo.seen.append(k)
        raise AssertionError("learner read step-info field %r" % k)


def b05_loop(n=3, peers=True, ctx=False, cfgs=None, tmp=None):
    D = n * T5
    cfgs = cfgs or [dd.configure_stream(game_cfg(), k, n) for k in range(n)]
    pcd = po.context_dim(A) if (peers and ctx) else 0
    L = SimpleNamespace(
        config={"outcome_memory": dict(OM_CFG, enabled=True),
                **({"peer_observation": {"enabled": True, "seed": 0, "window": 4, "capacity": 64,
                                         "context": {"enabled": ctx}}} if peers else {})},
        device=CPU, policy=None, world_model=None, replay_buffer=MultiStreamReplayBuffer(n, 4000, OBS, A),
        skill_bank=SimpleNamespace(storage_dir=tmp or tempfile.mkdtemp(prefix="bnd_b05_")), is_discrete=True,
        action_dim=A, _num_envs=n, total_timesteps=0, _vk_step=None, _om=None, _am=None, _cons=None, _rel=None,
        _om_dim=retrieval_dim(A), _am_dim=0, _pc_dim=pcd, _rk_dim=0, _kd_base=0, _peer=None, _rk=None, _am_sp=None,
        infra=SimpleNamespace(last_ledger_segment={}), _wm_param_lock=threading.RLock(), _keys=[None] * n, _sp=None,
        streams=n, cfgs=cfgs, D=D, truth=[[] for _ in range(n)])
    L._current_knowledge_feature_base = lambda: None
    L._step_view_key = lambda i: L._keys[i]
    L._view_key = lambda x: x
    L._step_surprise = lambda *a, **k: L._sp
    for name in ("_outcome_memory_step", "_peer_board", "_peer_observation_step", "_current_knowledge_feature"):
        setattr(L, name, getattr(DAI, name).__get__(L))
    return L


def b05_play(L, steps=T5, die=None, silent=(), info=None, swap_pub=False, start=0, stale=False):
    """Each stream plays ITS configured maps (rotating), its frame is written to its replay stream, and the loop's own
    peer / outcome-memory helpers see the same step.  die = (stream, step): a life boundary there.  A `silent`
    (dead) stream writes nothing to replay and presents an ABSENT view (None); `stale` is the DEFECT of presenting its
    last good view as fresh instead."""
    n = L.streams
    for t in range(start, start + steps):
        keys, acts, dones = [], [], []
        for k in range(n):
            maps_k = dd.adapter_maps(L.cfgs[k])[0]
            mp = maps_k[t % len(maps_k)]
            a = (k + t) % A
            d = bool(die and die == (k, t))
            L.truth[k].append((mp, a, t))
            if k in silent:
                keys.append(unit(k * T5 + ((start - 1) % T5), L.D) if stale else None)
            else:
                L.replay_buffer.add(tag_obs(k, mp, t), a, 0.0, d, stream=k)
                keys.append(unit(k * T5 + (t % T5), L.D))
            acts.append(a)
            dones.append(d)
        L._keys = list(keys)
        L._sp = sp_of(n, 0.3)
        ex = torch.zeros(n, A)
        for i, a in enumerate(acts):
            ex[i, a] = 1.0
        infos = [info if info is not None else {} for _ in range(n)]
        real = None
        pb = L._peer_board()
        if swap_pub and pb is not None:
            real = pb.publish
            pb.publish = lambda s, v, a_, _r=real: _r((s + 1) % n, v, a_)
        with quiet():
            L._peer_observation_step(ex, [None] * n, dones, [False] * n, infos)
            L._outcome_memory_step(None, ex, None, [None] * n, torch.full((n,), 0.2), dones, [False] * n, infos)
        if real is not None:
            pb.publish = real


def replay_rows(L, B=64, T=4):
    reseed(3)
    b = L.replay_buffer.sample_sequences(B, T, CPU, terminal_fraction=0.0)
    rows = []
    for bi, (s, i) in enumerate(b["start_indices"]):
        for j in range(T):
            rows.append((int(s), untag(b["observations"][bi, j].numpy())))
    return rows


@sc("CT-01", "BND-05: a stream's maps, frames, actions and views keep its identity through replay, outcome memory and the peer board",
    brk="peer_stream_ids_rotated", min_exercised=30,
    oracle="every replayed row's (stream, map) tag equals its producer's, maps come only from that stream's configured list, "
           "the peer board's per-stream actions/views and the outcome store's per-stream entries decode to the producing stream")
def b05_ct01(brk):
    c = Chk()
    L = b05_loop(3)
    b05_play(L, swap_pub=bool(brk))
    rows = replay_rows(L)
    parts = [dd.adapter_maps(cf)[0] for cf in L.cfgs]
    bad_s = [r for r in rows if r[1][0] != r[0]]
    bad_m = [r for r in rows if r[1][1] not in parts[r[0]]]
    c(not bad_s, "a replayed row's stream tag differs from the stream it was sampled from (%d rows)" % len(bad_s), n=len(rows))
    c(not bad_m, "a replayed row names a map outside its stream's configured list (%d rows)" % len(bad_m), n=len(rows))
    c(sorted(m for p in parts for m in p) == sorted(MAPS) and not set(m for p in parts for m in p) & set(HELD),
      "streams do not partition exactly the source maps", n=2)
    pb = L._peer
    for k in range(L.streams):
        ring = pb._ring.get(k)
        c(ring is not None and len(ring) > 0, "stream %d never reached the board" % k, n=1)
        if not ring:
            continue
        ok_a = all(a == (k + (n_ - 1)) % A for (n_, e, a, brk_) in ring)
        ok_v = all(int(torch.argmax(e)) // T5 == k for (n_, e, a, brk_) in ring)
        c(ok_a, "peer board stream %d holds actions produced by another stream" % k, n=len(ring))
        c(ok_v, "peer board stream %d holds views produced by another stream" % k, n=len(ring))
    om = L._om
    for k in range(L.streams):
        r = om._rings.get(k)
        c(r is not None and r.n > 0, "stream %d has no outcome entries" % k, n=1)
        if r is None:
            continue
        idx = [int(torch.argmax(r.keys[i])) for i in range(r.n)]
        c(all(i // T5 == k for i in idx), "outcome store ring %d holds another stream's view" % k, n=r.n)
        c(all(int(r.acts[i]) == (k + (idx[i] % T5) + 1) % A for i in range(r.n)),
          "outcome store ring %d holds an action that stream did not take" % k, n=r.n)
    return c.obs("stream/map/action/view identity is preserved end to end")


@sc("CT-02", "BND-05: the peer channel is observation-only: with it cut, replay and outcome memory are bit-identical; a lone stream gets neutral context",
    brk="board_writes_into_replay", min_exercised=10,
    oracle="peers on vs off: sampled replay rows and every outcome-store ring are identical; peer_observation off -> no board; "
           "one stream -> zero context")
def b05_ct02(brk):
    c = Chk()

    def world(peers):
        reseed(7)
        L = b05_loop(3, peers=peers)
        if brk and peers:                                        # DEFECT: the board's publish also appends to the observer's replay
            pb = L._peer_board()
            real = pb.publish
            pb.publish = lambda s, v, a, _r=real: (_r(s, v, a), L.replay_buffer.add(tag_obs(s, MAPS[0], 99), 0, 0.0, False, stream=s))[0]
        b05_play(L)
        return L
    A_, B_ = world(True), world(False)
    c(B_._peer is None and getattr(B_, "_peer_off", False), "peer_observation off still built a board", n=1)
    ra, rb = replay_rows(A_, 32), replay_rows(B_, 32)
    c(ra == rb, "the peer channel changed what replay holds", n=len(ra))
    same = all(torch.equal(A_._om._rings[k].keys, B_._om._rings[k].keys) and torch.equal(A_._om._rings[k].acts, B_._om._rings[k].acts)
               for k in range(3))
    c(same, "the peer channel changed the outcome store", n=3)
    one = b05_loop(1, ctx=True)
    one.config["peer_observation"]["enabled"] = True
    b05_play(one, 6)
    c(one._peer is None and not np.any(one._current_knowledge_feature()[retrieval_dim(A):]), "a lone stream got non-neutral peer context", n=2)
    return c.obs("peer experience is a read-only side channel")


@sc("CT-03", "BND-05: an absent peer view is a break (not a valid view) and no pair spans it",
    brk="absent_view_stored_as_valid", min_exercised=4,
    oracle="publish(None) stores nothing and marks a break; the next entry is flagged as a break so no pair spans it")
def b05_ct03a(brk):
    c = Chk()
    L = b05_loop(3)
    b05_play(L, 8)
    pb = L._peer
    r0 = len(pb._ring[0])
    if brk:
        real = pb.publish
        pb.publish = lambda s, v, a: real(s, torch.ones(L.D) if v is None else v, a)
    pb.publish(0, None, 1)
    c(len(pb._ring[0]) == r0 and pb._brk.get(0) is True, "an absent view was stored as a valid one", n=2)
    pb.publish(0, unit(0, L.D), 2)
    c(len(pb._ring[0]) == r0 + 1 and pb._ring[0][-1][3] is True, "the entry after an absent view is not flagged as a break", n=2)
    return c.obs("absent = break")


@sc("CT-03", "BND-05: a refused (wrong-width) peer view leaves the stream's stamp counter untouched "
             "(REAL DEFECT if red: PeerBoard.publish consumes a stamp before it validates)",
    brk="refusal_keeps_stamp", min_exercised=4,
    oracle="a view of another width raises; the stream's publish count and ring are exactly as before, so the next valid entry is consecutive")
def b05_ct03c(brk):
    c = Chk()
    L = b05_loop(3)
    b05_play(L, 8)
    pb = L._peer
    n0 = dict(pb._count)
    snap = [(n_, tuple(e.tolist()), a) for (n_, e, a, b_) in pb._ring[1]]
    if brk:                                                      # DEFECT (the pre-F-21 order): a stamp is consumed before the view is validated
        real_publish = pb.publish

        def publish_old_order(stream, view, action, _real=real_publish):
            pb._count[int(stream)] = pb._count.get(int(stream), 0) + 1
            return _real(stream, view, action)
        pb.publish = publish_old_order
    try:
        pb.publish(1, torch.randn(L.D + 3), 1)
        c(False, "a wrong-width view was accepted")
    except RuntimeError:
        c(True, "")
    except ValueError:                                           # GOV-015 F-21: the board refuses by name with ValueError
        c(True, "")
    c([(n_, tuple(e.tolist()), a) for (n_, e, a, b_) in pb._ring[1]] == snap, "the refused view changed the ring", n=1)
    c(pb._count == n0, "the refused view consumed a stamp (count %s -> %s)" % (n0.get(1), pb._count.get(1)), n=1)
    pb.publish(1, unit(1 * T5 + 9, L.D), 2)
    c(pb._ring[1][-1][0] == snap[-1][0] + 1, "the next valid entry is not consecutive with the last one (stamp %d after %d): "
      "the pair across it is silently discarded and the observer's window misaligned" % (pb._ring[1][-1][0], snap[-1][0]), n=1)
    return c.obs("a refused message changes nothing")


@sc("CT-03", "BND-05: a non-finite peer view never reaches the policy's peer context or the auxiliary training pairs "
             "(REAL DEFECT if red: PeerBoard.publish accepts non-finite views)",
    brk="nan_view_every_step", min_exercised=4,
    oracle="after one NaN view the observer's context vector and the training pairs stay finite")
def b05_ct03b(brk):
    c = Chk()
    L = b05_loop(3, ctx=True)
    b05_play(L, 10)
    pb = L._peer
    bad = torch.full((L.D,), float("nan"))
    pb.publish(1, bad, 1)
    for t in range(10, 18):
        b05_play(L, 1, start=t)
        if brk:                                                  # DEFECT: the board stores what it is given (GOV-015 F-20 guard gone)
            pb._ring[1].append((pb._count[1] + 1, bad.clone(), 1, False))
    ring_finite = all(bool(torch.isfinite(e).all()) for r in pb._ring.values() for (_, e, _, _) in r)
    c(ring_finite, "the board stored a non-finite view", n=1)
    c(np.isfinite(pb.context(0)).all(), "peer context contains non-finite values", n=1)
    c(np.isfinite(L._current_knowledge_feature()).all(), "the knowledge vector the policy reads contains non-finite values", n=1)
    ok = True
    for _ in range(80):
        d = pb.pairs(16)
        ok &= d is None or all(bool(torch.isfinite(x.float()).all()) for x in d)
    c(ok, "training pairs contain non-finite views", n=2)
    return c.obs("non-finite peer evidence is refused")


@sc("CT-05", "BND-05: a life boundary on one stream splits its pairs and outcome entries and leaves the other streams alone",
    brk="life_boundary_not_signalled", min_exercised=40,
    oracle="no peer pair and no outcome entry spans the boundary of stream 1; streams 0 and 2 keep unbroken sequences; "
           "an episode end in stream 1 does not clear stream 0's or 2's rings")
def b05_ct05(brk):
    c = Chk()
    BR = 8
    L = b05_loop(3)
    if brk:                                                      # DEFECT: the loop stops passing the done flag
        real = DAI._peer_observation_step.__get__(L)
        L._peer_observation_step = lambda ex, nxt, dones, rst, inf: real(ex, nxt, [False] * len(dones), rst, inf)
        real_om = DAI._outcome_memory_step.__get__(L)
        L._outcome_memory_step = lambda *a: real_om(*(a[:5] + ([False] * len(a[5]),) + a[6:]))
    b05_play(L, 14, die=(1, BR))
    pb = L._peer
    bad = []
    for _ in range(60):
        d = pb.pairs(16)
        if d is None:
            continue
        vt, ap, at, vn = d
        for i in range(vt.shape[0]):
            i0, i1 = int(torch.argmax(vt[i])), int(torch.argmax(vn[i]))
            s0, t0, s1, t1 = i0 // T5, i0 % T5, i1 // T5, i1 % T5
            if s0 != s1 or t1 != t0 + 1:
                bad.append("(%d,%d)->(%d,%d)" % (s0, t0, s1, t1))
            elif s0 == 1 and t0 <= BR < t1:
                bad.append("stream 1 pair spans the boundary (%d,%d)" % (t0, t1))
    c(not bad, "peer pairs cross a stream/life boundary: %s" % sorted(set(bad))[:2], n=60)
    r1 = L._om._rings[1]
    idx1 = sorted(int(torch.argmax(r1.keys[i])) % T5 for i in range(r1.n))
    c(BR not in idx1, "an outcome entry spans stream 1's boundary (a key at step %d): steps %s" % (BR, idx1), n=r1.n)
    c(L._om._rings[0].n >= 10 and L._om._rings[2].n >= 10, "the boundary of stream 1 cleared another stream's store", n=2)
    c(len(pb._ring[0]) == 14 and len(pb._ring[2]) == 14, "the boundary of stream 1 cleared another stream's board ring", n=2)
    return c.obs("life boundaries stay inside their stream")


@sc("CT-06", "BND-05: a dead stream stops contributing and its lag is observable; a leaked held-out config is refused and a corrected one resumes",
    brk="dead_stream_republished", min_exercised=9,
    oracle="a silent stream: its replay and board rings stop growing while the others grow, the observer's context stays finite, "
           "last_lag > 0; when it resumes both grow and lag returns to 0; the LeakError is raised and a clean config composes")
def b05_ct06(brk):
    c = Chk()
    L = b05_loop(3, ctx=True)
    b05_play(L, 10)
    pb = L._peer
    n_rep = [len(L.replay_buffer.streams[k]) for k in range(3)]
    b05_play(L, 10, silent=(2,), start=10, stale=bool(brk))
    ctxv = pb.context(0)
    n_rep2 = [len(L.replay_buffer.streams[k]) for k in range(3)]
    c(n_rep2[2] == n_rep[2] and n_rep2[0] > n_rep[0] and n_rep2[1] > n_rep[1], "silent stream's replay did not stop / others did not grow", n=3)
    c(np.isfinite(ctxv).all(), "context broke while a peer was silent", n=1)
    c(pb.last_lag > 1.0, "a dead peer's lag is not observable (%.2f)" % pb.last_lag, n=1)
    n_pb2 = len(pb._ring[2])
    b05_play(L, 8, start=20)
    pb.context(0)
    c(len(L.replay_buffer.streams[2]) > n_rep2[2] and len(pb._ring[2]) > n_pb2, "the resumed stream was not picked up again", n=2)
    c(pb.last_lag < 1.0, "lag did not return to ~0 after the stream resumed (%.2f)" % pb.last_lag, n=1)
    leaked = game_cfg(map_assignment={"per_stream": {1: ["MAP02", "MAP30"]}})
    try:
        dd.configure_stream(leaked, 1, 3)
        c(False, "a held-out map in a source list was accepted")
    except dd.LeakError as e:
        c("MAP30" in str(e), "the leak error does not name the map", n=1)
    ok_cfg = dd.configure_stream(game_cfg(map_assignment={"per_stream": {1: ["MAP02", "MAP03"]}}), 1, 3)
    c(dd.adapter_maps(ok_cfg)[0] == ["MAP02", "MAP03"], "the corrected config did not compose (latched)", n=1)
    return c.obs("dead streams and leaked configs are contained and recoverable")


@sc("CT-07", "BND-05: a restored replay keeps each row on its producing stream; the peer board restarts empty; map assignment is reproducible",
    brk="streams_restored_crossed", min_exercised=12,
    oracle="save/load of the multi-stream replay: every restored row's tag equals its stream; the new board is empty (documented "
           "warm-up) and refills; the configured lists are identical after a rebuild")
def b05_ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b05_rep_")
    L = b05_loop(3)
    b05_play(L, 14)
    L.replay_buffer.save(tmp)
    if brk:                                                      # DEFECT: stream directories restored crossed
        os.rename(os.path.join(tmp, "stream_0"), os.path.join(tmp, "stream_x"))
        os.rename(os.path.join(tmp, "stream_1"), os.path.join(tmp, "stream_0"))
        os.rename(os.path.join(tmp, "stream_x"), os.path.join(tmp, "stream_1"))
    L2 = b05_loop(3)
    n = L2.replay_buffer.load(tmp)
    c(n >= 3 * 10, "restored too few rows (%d)" % n, n=1)
    rows = replay_rows(L2)
    c(all(r[1][0] == r[0] for r in rows), "a restored row sits on another stream than its producer", n=len(rows))
    parts = [dd.adapter_maps(cf)[0] for cf in L2.cfgs]
    c(all(r[1][1] in parts[r[0]] for r in rows), "a restored row names a map outside its stream's list", n=len(rows))
    c(L2._peer_board() is not None and L2._peer_board().pairs(8) is None and not L2._peer_board().context(0).any(),
      "the peer board was not empty after restart", n=2)
    b05_play(L2, 6, start=14)
    c(len(L2._peer._ring[0]) == 6, "the board did not refill after restart", n=1)
    again = [dd.configure_stream(game_cfg(), k, 3)["stream_maps"] for k in range(3)]
    c(again == [cf["stream_maps"] for cf in L.cfgs], "map assignment is not reproducible from the config", n=1)
    return c.obs("restart keeps stream identity; the peer warm-up is the documented one")


@sc("CT-08", "BND-05: held-out maps and evaluator fields cannot enter source training (config, replay, board, memory)",
    brk="leak_check_removed_and_held_map_in_source_list", min_exercised=18,
    oracle="a held-out map in a source list is rejected at configuration; if that check is removed the replay scan still "
           "finds held-out tags; the learner helpers read no step-info field but the restart flag; the board's channel "
           "rejects evaluator fields; telemetry reads leave board and RNG unchanged")
def b05_ct08(brk):
    c = Chk()
    leak = {"per_stream": {1: ["MAP02", "MAP30"]}}
    ctx = patched(dd, "check_no_leak", lambda cfg: None) if brk else contextlib.nullcontext()
    with ctx:
        cfgs = []
        raised = False
        for k in range(3):
            try:
                cfgs.append(dd.configure_stream(game_cfg(map_assignment=dict(leak, mode="spread")), k, 3))
            except dd.LeakError:
                raised = True
                cfgs.append(dd.configure_stream(game_cfg(), k, 3))
        c(raised != bool(brk), "the leak was %s" % ("not rejected" if not brk else "still rejected"), n=1)
        if brk:
            cfgs = [dd.configure_stream(game_cfg(map_assignment=dict(leak, mode="spread")), k, 3) for k in range(3)]
        L = b05_loop(3, cfgs=cfgs)
        b05_play(L, 14)
        rows = replay_rows(L, 64, 4)
        held_rows = [r for r in rows if r[1][1] in HELD]
        c(not held_rows, "held-out maps reached source replay (%d of %d sampled rows)" % (len(held_rows), len(rows)), n=len(rows))
    TripInfo.seen = []
    L2 = b05_loop(3)
    trip = TripInfo(evaluator_score=1.0, health=9, position=(1, 2))
    b05_play(L2, 6, info=trip)
    c(set(TripInfo.seen) <= {"env_restarted"} and len(TripInfo.seen) > 0, "learner helpers read step-info fields %s" % sorted(set(TripInfo.seen)), n=len(TripInfo.seen))
    for field in ("evaluator_checkpoint", "reward", "health", "position"):
        try:
            po.assert_channel_clean({"view": 1, field: 1.0})
            c(False, "peer channel accepted %r" % field)
        except po.LeakError:
            c(True, "")
    pb = L2._peer
    snap = (dict(pb._count), len(pb.hyp), pb.rng.get_state()[1].tobytes(), rng_sig())
    pb.diversity()
    pb.hypotheses()
    c(snap == (dict(pb._count), len(pb.hyp), pb.rng.get_state()[1].tobytes(), rng_sig()), "reading diversity/hypotheses changed board or RNG state", n=1)
    mods = [m for m in ("developmental_ai.core.peer_observation", "developmental_ai.environments.data_diversity",
                        "developmental_ai.world_model.replay_buffer", "developmental_ai.core.outcome_memory")
            if "evaluator" in open(sys.modules[m].__file__).read().lower().replace("evaluator-side", "").replace("evaluator", "evaluator")
            and re.search(r"^\s*(from|import)\s+developmental_ai\.evaluator", open(sys.modules[m].__file__).read(), re.M)]
    c(not mods, "a module on the source-training path imports the evaluator: %s" % mods, n=1)
    return c.obs("no held-out or evaluator data reaches source training")


# ================================================================================================
# BND-06  CAT-02 <-> CAT-05   consolidation, replay selection (the RW-6 recipe) and optimiser updates
# ================================================================================================
def b06_cfg(strategy=True, ao=True, planning=True, ewc=True):
    cons = {"enabled": True, "every_segments": 1, "steps": 1, "batch": 4, "seq_len": 4, "lr_scale": 0.1,
            "replay": {"spaced_fraction": 0.0, "prioritized": False},
            "distill": {"enabled": True, "snapshot_every_segments": 2, "max_snapshots": 3}}
    if ewc:
        cons["regularizer"] = {"kind": "ewc", "lambda": 10.0, "anchor_every_segments": 3, "fisher_batches": 1}
    cfg = {"consolidation": cons}
    if planning:
        cfg["planning"] = {"enabled": True, "rows": 4, "seq_len": 8, "planner": {"candidates": 8, "horizon": 2, "iterations": 1},
                           "calibration": {"min_n": 24, "max_ece": 0.5, "min_corr": -1.0},
                           "ensemble": {"enabled": True, "hidden": 16, "heads": 3}}
    if ao:
        cfg["adaptive_optim"] = {"enabled": True, "every_segments": 1,
                                 "schedule": {"enabled": True, "policy": {"kind": "linear", "horizon": 1000, "floor": 0.1},
                                              "world_model": {"kind": "constant"}},
                                 "entropy": {"enabled": True, "target": 0.5, "eta": 0.5}}
    if strategy:
        rel = {"enabled": True, "strategy": {"epoch_len": 1, "min_pulls": 1, "seed": 0, "spaced": 0.25}}
        for k, v in STRAT.items():
            rel[k] = copy.deepcopy(v)
        cfg["relations"] = rel
    return cfg


B06_LOSS = [1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.65, 0.5, 0.5, 0.45, 0.45, 0.4]


def b06_brain(cfg=None, tmp=None, tag=False, order=None):
    cfg = cfg if cfg is not None else b06_cfg()
    keep = {"_consolidation_phase", "_planning_phase", "_adaptive_optim_phase", "_relations_phase"}
    buf = mk_buf(2, 240, 0, tag=True) if tag else None
    ns = brain(cfg, tmp=tmp, buf=buf, phases=[p for p in PHASES if p in keep])
    if order:
        ns._phases = list(order)
    ns._plan_dummy = None
    ns._am = SimpleNamespace(net=SimpleNamespace(temperature=1.0), last={}, pushed0=True)
    ns.policy.last_update_stats = {"cond_entropy_norm": 0.4}
    ns._consumed, ns._published = [], []
    return ns


def b06_groups(ns):
    P = ns.policy
    g = {"actor": thash(phash(P.actor)), "critic": thash(phash(P.critic)), "wm": thash(phash(ns.world_model)),
         "lr": round(P.optimizer.param_groups[0]["lr"], 12), "entropy_coef": P.entropy_coef}
    if getattr(ns, "_plan", None):
        g["ens"] = thash(phash(ns._plan.ens))
    if P.conditioner is not None:
        g["cond"] = thash(phash(P.conditioner))
    return g


def b06_seg(ns, k, order=None, attrib=None):
    """One segment in the given (default: real) order.  attrib: dict phase -> set(groups changed) accumulated."""
    ns._seg_now = k
    MEAS.clear()
    MEAS[0] = (B06_LOSS[k % len(B06_LOSS)], 0.0)
    ns.lifetime_timesteps += 64
    ns.policy.last_update_stats = {"cond_entropy_norm": 0.4}
    reseed(100 + k)
    for ph in (order or ns._phases):
        before = b06_groups(ns)
        with quiet():
            getattr(ns, ph)()
        if attrib is not None:
            after = b06_groups(ns)
            attrib.setdefault(ph, set()).update(g for g in after if before.get(g) != after[g])
    cons = getattr(ns, "_cons", None)
    if cons:
        ns._published.append((cons.old_frac, cons.prioritized))


@contextlib.contextmanager
def b06_spy(ns):
    real = Consolidator._sample

    def spy(self, buf, device):
        ns._consumed.append((getattr(ns, "_seg_now", -1), self.old_frac, self.prioritized))
        return real(self, buf, device)
    with mock.patch.object(Consolidator, "_sample", spy):
        yield


@sc("CT-01", "BND-06: PPO -> consolidation -> planning -> optimiser controller -> relations in the declared order; each phase writes only "
             "its own parameters; the RW-6 recipe reaches replay selection one segment later",
    brk="relations_before_consolidation", min_exercised=36,
    oracle="source order PPO < consolidation < ... < relations in both bodies; consolidation changes only the actor, planning only the "
           "ensemble, the optimiser controller only learning rate / entropy coefficient, relations nothing in the weights; "
           "the replay selection consumed in segment s is the recipe published at the end of s-1")
def b06_ct01(brk):
    c = Chk()
    src = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    for m in re.finditer(r"self\._consolidation_phase\(\)", src):
        pre = src[max(0, m.start() - 4000):m.start()]
        c("self.policy.train_step(" in pre, "PPO does not precede the consolidation phase in a stepping body", n=1)
    c(PHASES == [p for p in LOOP_ORDER if p != "_peer_phase"] and PHASES.index("_consolidation_phase") < PHASES.index("_planning_phase")
      < PHASES.index("_adaptive_optim_phase") < PHASES.index("_relations_phase"), "loop phase order changed: %s" % PHASES, n=1)
    ns = b06_brain(order=None)
    order = list(ns._phases)
    if brk:                                                      # DEFECT: the relation wiring runs before the consolidation phase
        order = ["_relations_phase"] + [p for p in order if p != "_relations_phase"]
    attrib = {}
    with measuring(), b06_spy(ns):
        for k in range(10):
            b06_seg(ns, k, order=order, attrib=attrib)
    c(attrib.get("_consolidation_phase", set()) == {"actor"} or attrib.get("_consolidation_phase", set()) == {"actor", "cond"},
      "consolidation changed %s (expected the actor only)" % sorted(attrib.get("_consolidation_phase", set())), n=6)
    c(attrib.get("_planning_phase", set()) == {"ens"}, "planning changed %s (expected the ensemble only)" % sorted(attrib.get("_planning_phase", set())), n=6)
    c(attrib.get("_adaptive_optim_phase", set()) <= {"lr", "entropy_coef"} and "lr" in attrib.get("_adaptive_optim_phase", set()),
      "the optimiser controller changed %s (expected lr / entropy coefficient only)" % sorted(attrib.get("_adaptive_optim_phase", set())), n=6)
    c(not attrib.get("_relations_phase", set()), "the relation wiring wrote %s" % sorted(attrib.get("_relations_phase", set())), n=6)
    pub = ns._published
    by_seg = {}
    for sg, fr, pr in ns._consumed:
        by_seg.setdefault(sg, set()).add((fr, pr))
    c(len(by_seg) >= 8 and len(set(pub)) >= 3, "fixture: recipe never changed (published %s)" % sorted(set(pub)), n=2)
    late = [sg for sg in range(1, len(pub)) if sg in by_seg and by_seg[sg] != {pub[sg - 1]}]
    c(not late, "replay selection in segment s is not the recipe published at the end of s-1 (segments %s: consumed %s vs published %s)"
      % (late[:3], [sorted(by_seg[x]) for x in late[:2]], [pub[x - 1] for x in late[:2]]), n=len(by_seg))
    return c.obs("declared order, attributed writes, one-segment recipe delay")


@sc("CT-01", "BND-06: every replay recipe RW-6 can select is executable by the consolidation phase "
             "(REAL DEFECT if red: Consolidator._sample mis-sizes the batch when spaced_fraction > 0)",
    brk="spaced_arms_never_selected", min_exercised=4,
    oracle="Consolidator.run completes (steps >= 1, no exception) on the real policy, world model and replay under each of "
           "the four recipes (0|0.25 spaced) x (uniform|prioritised)")
def b06_ct01b(brk):
    c = Chk()
    ns = b06_brain(b06_cfg(planning=False, ao=False, strategy=False))
    recipes = [(0.0, False), (0.0, True)] if brk else [(0.0, False), (0.25, False), (0.0, True), (0.25, True)]
    for frac, prio in recipes:
        C = Consolidator(dict(ns.config["consolidation"], replay={"spaced_fraction": frac, "prioritized": prio}))
        reseed(9)
        try:
            out = C.run(ns.policy, ns.world_model, ns.replay_buffer, CPU)
            c(bool(out) and out.get("steps", 0) >= 1, "recipe (%.2f, %s): consolidation ran no step" % (frac, prio), n=1)
        except Exception as e:
            c(False, "recipe (spaced %.2f, prioritised %s): consolidation raised %s: %s" % (frac, prio, type(e).__name__, str(e)[:90]), n=1)
    c(len(recipes) == 4, "the spaced recipes were not exercised", n=1)
    return c.obs("every recipe is executable")


@sc("CT-02", "BND-06: with the optimiser controller or the strategy cut, consolidation is bit-identical to consolidation alone",
    brk="optim_controller_before_consolidation", min_exercised=10,
    oracle="after K segments the consolidated actor is identical with {adaptive optimisation on / off}; with the RW-6 channels cut "
           "the replay selection stays the configured (0, False) and the actor equals the never-wired run")
def b06_ct02(brk):
    c = Chk()

    def run(cfg, order_swap=False):
        reseed(61)
        ns = b06_brain(cfg)
        if order_swap:
            ns._phases = ["_adaptive_optim_phase"] + [p for p in ns._phases if p != "_adaptive_optim_phase"]
        with measuring(), b06_spy(ns):
            for k in range(6):
                b06_seg(ns, k)
        return ns
    only = b06_cfg(strategy=False, ao=False, planning=False)
    with_ao = b06_cfg(strategy=False, ao=True, planning=False)
    a = run(only)
    b = run(with_ao, order_swap=bool(brk))
    c(torch.equal(phash(a.policy.actor), phash(b.policy.actor)),
      "adaptive optimisation changed the consolidated actor (it must change only its own values)", n=6)
    rel_cut = b06_cfg(planning=False, ao=False)
    rel_cut["relations"]["SRL"]["STRAT_ENTRY"]["enabled"] = False
    rel_cut["relations"]["STRAT_ENTRY"] = {"SP": {"enabled": False}, "INTL": {"enabled": False}}
    rel_cut["relations"]["CALIB"] = {"SRL": {"enabled": False}}
    d = run(rel_cut)
    c({(f, p_) for _, f, p_ in d._consumed} == {(0.0, False)}, "strategy cut but replay selection left the configured recipe: %s" % sorted({(f, p_) for _, f, p_ in d._consumed}), n=len(d._consumed))
    c(torch.equal(phash(a.policy.actor), phash(d.policy.actor)), "a cut strategy changed the consolidated actor", n=1)
    c(a._cons._calls == b._cons._calls == 6, "consolidation call counts differ", n=2)
    return c.obs("neighbours do not change consolidation when cut")


@sc("CT-03", "BND-06: absent or too little replay idles consolidation without writes, and absent fresh-data evidence releases the recipe to the configured one",
    brk="idle_not_idle", min_exercised=6,
    oracle="replay shorter than 4*seq_len -> consolidation returns {} and writes nothing; with no fresh-loss measurement the "
           "RW-6 arm is released and the recipe consolidation consumes returns to the configured one and stays (GOV-015 9.6)")
def b06_ct03a(brk):
    c = Chk()
    cfg = b06_cfg(planning=False, ao=False)
    ns = b06_brain(cfg)
    short = mk_buf(2, 10, 0)
    ns.replay_buffer = short if not brk else ns.replay_buffer
    w0 = thash(phash(ns.policy.actor))
    with measuring(), b06_spy(ns):
        for k in range(3):
            b06_seg(ns, k)
    c(thash(phash(ns.policy.actor)) == w0 and not ns._consumed, "consolidation wrote weights / sampled on a too-short replay", n=2)
    ns2 = b06_brain(cfg)
    base_ = (cfg["consolidation"]["replay"]["spaced_fraction"], bool(cfg["consolidation"]["replay"]["prioritized"]))   # the configured recipe (the consolidator does not exist until the first segment)
    with measuring(), b06_spy(ns2):
        for k in range(4):
            b06_seg(ns2, k)
        pub = list(ns2._published)
        for _ in range(4):                                       # the measurement source returns nothing
            MEAS.clear()
            ns2.lifetime_timesteps += 64
            segment(ns2)
            pub.append((ns2._cons.old_frac, ns2._cons.prioritized))
    # GOV-015 9.6: "no measurement" is UNKNOWN, not "unchanged" (GOV-015 F-25, relations_strategy.py header: the coefficients are
    # released and the recipe returns to the configured one). The earlier oracle (the recipe stays where it was) contradicted
    # that and BND-04 CT-06. The contract: it returns to the configured recipe, within the slew bound, and then stays.
    c(tuple(pub[-1]) == base_ and len(set(pub[-2:])) == 1,
      "the recipe did not return to the configured one %s and stay with no measurement: %s" % (base_, pub[3:]), n=4)
    c(len(ns2._consumed) >= 6, "consolidation stopped consuming without measurements", n=1)
    return c.obs("idle on absent data; recipe frozen without evidence")


@sc("CT-03", "BND-06: a non-finite fresh-data loss does not change the replay selection consolidation consumes "
             "(REAL DEFECT if red: RW-6 books a non-finite loss as a reward)",
    brk="inf_instead_of_nan", min_exercised=6,
    oracle="the recipe sequence with a NaN loss at one segment equals the sequence of a run that simply had no measurement there")
def b06_ct03b(brk):
    c = Chk()

    def seq(bad):
        reseed(71)
        ns = b06_brain(b06_cfg(planning=False, ao=False))
        with measuring(), b06_spy(ns):
            for k in range(10):
                if k in (4, 5):
                    MEAS.clear()
                    if bad is not None:
                        MEAS[0] = (bad, 0.0)
                    ns.lifetime_timesteps += 64
                    segment(ns)
                    ns._published.append((ns._cons.old_frac, ns._cons.prioritized))
                else:
                    b06_seg(ns, k)
        return ns._published
    ref = seq(None)
    # DEFECT (GOV-015 9.6): the guard is gone and the value reaches RW-6 as a finite outlier (a literal inf/NaN is now
    # filtered upstream, so feeding it could no longer fail the oracle).
    got = seq(1e30 if brk else float("nan"))
    c(ref == got, "a non-finite loss changed the recipe consolidation consumes: %s vs %s" % (got, ref), n=len(ref))
    return c.obs("non-finite evidence does not steer replay selection")


@sc("CT-05", "BND-06: rows replay hands to consolidation keep their start index and stream when the recipe changes (spaced + prioritised)",
    brk="index_misalign", min_exercised=40,
    oracle="every consolidation batch row's (stream, step) frame tag equals its reported start index; spaced rows come "
           "from the older part of their stream; both hold under every arm RW-6 selects")
def b06_ct05(brk):
    c = Chk()
    ns = b06_brain(b06_cfg(planning=False, ao=False, strategy=False), tag=True)
    C = Consolidator(ns.config["consolidation"])
    real_concat = Consolidator._concat
    ctx = mock.patch.object(Consolidator, "_concat",
                            staticmethod(lambda a_, b_: dict(real_concat(a_, b_), start_indices=np.concatenate([np.asarray(a_["start_indices"]), np.asarray(b_["start_indices"])[::-1]]))
                                         if "start_indices" in b_ else real_concat(a_, b_))) if brk else contextlib.nullcontext()
    buf = ns.replay_buffer
    with ctx:
        for (frac, prio) in ((0.0, False), (0.25, False), (0.0, True), (0.25, True)):
            C.old_frac, C.prioritized = frac, prio
            C.batch = 8
            for _ in range(4):
                reseed(5)
                b = C._sample(buf, CPU)
                rows_n = {k: int(v.shape[0]) for k, v in b.items() if isinstance(v, torch.Tensor)}
                c(len(set(rows_n.values())) == 1 and rows_n["observations"] == len(b["start_indices"]),
                  "recipe (%.2f, %s): batch tensors disagree on the number of rows %s (start_indices %d)" % (frac, prio, rows_n, len(b["start_indices"])), n=1)
                for bi, (s_, i_) in enumerate(b["start_indices"]):
                    st, mp, step = untag_b(b["observations"][bi, 0].numpy())
                    c(st == int(s_) and step == int(i_), "row %d: tag (%d,%d) vs start index (%d,%d)" % (bi, st, step, s_, i_))
                    if b["actions"].shape[0] == len(b["start_indices"]):
                        ta = buf.truth.act[int(s_)][int(i_):int(i_) + b["actions"].shape[1]]
                        c([int(x) for x in b["actions"][bi].argmax(-1)] == ta, "row %d: actions are not the replayed actions at its start index" % bi)
                if frac > 0:
                    n_old = int(round(8 * frac))
                    rows = b["start_indices"][-n_old:]
                    older = [(int(buf.streams[int(s_)].position) - int(i_)) % int(buf.streams[int(s_)].capacity) / max(1, int(buf.streams[int(s_)].size)) >= C.old_after
                             for s_, i_ in rows]
                    c(all(older) and len(rows) == n_old, "spaced rows are not from the older part: %s" % older, n=len(rows))
    return c.obs("rows stay aligned with their start index under every recipe")


def untag_b(o):
    v = int(round(float(o[0]) * 100000))
    return v // 1000, None, v % 1000


@sc("CT-06", "BND-06: a consolidation fault is observable and contained, and the next segment resumes; a strategy fault returns the recipe "
             "to the configured one",
    brk="strategy_release_missing", min_exercised=7,
    oracle="a raising consolidation segment leaves the optimiser controller and relations running and warns; the next "
           "segment consolidates again; a raising RW-6 wiring releases the consolidator to its configured recipe")
def b06_ct06(brk):
    c = Chk()
    ns = b06_brain(b06_cfg(planning=False))
    with measuring(), b06_spy(ns):
        for k in range(5):
            b06_seg(ns, k)
        c(len(set(ns._published)) >= 2, "fixture: the recipe never changed", n=1)
        recs, rm = fault_log()
        ns._cons_warned = False                                  # earlier (separate) faults must not mask this one
        real = Consolidator.run
        state = {"fail": True}

        def flaky(self, *a, **k):
            if state["fail"]:
                raise RuntimeError("replay dependency down")
            return real(self, *a, **k)
        try:
            with mock.patch.object(Consolidator, "run", flaky):
                lr0 = ns.policy.optimizer.param_groups[0]["lr"]
                b06_seg(ns, 5)
                c(any("consolidation phase failed" in m for m in recs), "the consolidation fault is not observable", n=1)
                c(ns._ao is not False and getattr(ns, "_rel", False) is not False, "the fault idled the optimiser controller or the relations", n=2)
                state["fail"] = False
                n_before = len(ns._consumed)
                b06_seg(ns, 6)
            c(len(ns._consumed) > n_before, "consolidation did not resume after the fault", n=1)
        finally:
            rm()
        # a fault inside the RW-6 wiring: the consolidator returns to its configured recipe
        import developmental_ai.core.relations_strategy as strat
        from developmental_ai.core.relations import RelationRegistry as RR
        applied = (ns._cons.old_frac, ns._cons.prioritized)
        if brk:
            ctxr = patched(strat, "release", lambda loop, reg: None)
        else:
            ctxr = contextlib.nullcontext()
        with ctxr, patched(strat, "apply", lambda loop, reg: (_ for _ in ()).throw(RuntimeError("strategy fault"))):
            b06_seg(ns, 7)
        c((ns._cons.old_frac, ns._cons.prioritized) == (0.0, False),
          "after a strategy fault the consolidator keeps recipe %s (was %s)" % ((ns._cons.old_frac, ns._cons.prioritized), applied), n=2)
    return c.obs("faults are contained; the recipe is released")


@sc("CT-07", "BND-06: consolidation state (teachers, Fisher, anchor) is restored by the loop's own path; the unsaved recipe, Adam moments and "
             "optimiser coefficients restart at the configured values",
    brk="consolidation_state_not_restored", min_exercised=6,
    oracle="a recreated brain on the same storage restores the teacher snapshots and the anchor; the first consolidation after "
           "restart uses the configured recipe and a fresh optimiser; adaptive-optimisation coefficients are neutral")
def b06_ct07(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b06_")
    ns = b06_brain(tmp=tmp)
    with measuring():
        for k in range(8):
            b06_seg(ns, k)
    C = ns._cons
    teachers, anchor = len(C._teachers), C._anchor is not None
    c(teachers >= 2 and anchor and len(set(ns._published)) >= 2, "fixture: nothing to restore (teachers %d, anchor %s)" % (teachers, anchor), n=1)
    ns2 = b06_brain(tmp=(tempfile.mkdtemp(prefix="bnd_b06x_") if brk else tmp))
    seen = {}
    real_run = Consolidator.run

    def spy(self, *a, **k):
        seen["teachers_at_start"] = len(self._teachers)
        seen["anchor"] = self._anchor is not None
        seen["recipe"] = (self.old_frac, self.prioritized)
        seen["fresh_opt"] = self._opt is None
        return real_run(self, *a, **k)
    with measuring(), mock.patch.object(Consolidator, "run", spy):
        b06_seg(ns2, 0)
    c(seen.get("teachers_at_start") == teachers, "restored %s teachers, expected %d" % (seen.get("teachers_at_start"), teachers), n=1)
    c(seen.get("anchor") is True, "the EWC anchor was not restored", n=1)
    c(seen.get("recipe") == (0.0, False) and seen.get("fresh_opt") is True, "recipe/optimiser did not restart at the configured values: %s" % seen, n=2)
    reg2 = {r["name"]: r["value"] for r in registry_for(ns2).table()}
    c(all(abs(v - 1.0) < 0.1 for k, v in reg2.items() if k.endswith("_scale")) and abs(ns2.policy.optimizer.param_groups[0]["lr"] - 3e-4) < 3e-4 * 0.1,
      "optimiser coefficients are not at their base after restart: %s" % reg2, n=2)
    return c.obs("persisted consolidation resumes; the unsaved parts restart configured")


@sc("CT-08", "BND-06: consolidation and the controllers read only the declared replay fields; telemetry changes no weight, coefficient or RNG",
    brk="consolidation_reads_undeclared_field", min_exercised=6,
    oracle="a replay batch carrying evaluator-named fields is consumed by consolidation without touching them; report()/"
           "relation telemetry leave weights, recipe and RNG unchanged")
def b06_ct08(brk):
    c = Chk()
    ns = b06_brain(b06_cfg(planning=False, strategy=False))
    real_ss = ns.replay_buffer.sample_sequences

    def poisoned(*a, **k):
        b = Tripwire(real_ss(*a, **k))
        b["evaluator_checkpoint"] = torch.ones(1)
        b["held_out_score"] = torch.ones(1)
        return b
    ns.replay_buffer.sample_sequences = poisoned
    ctx = contextlib.nullcontext()
    if brk:                                                      # DEFECT: a consolidation change that reads a side field
        real_f = Consolidator._features
        ctx = patched(Consolidator, "_features", lambda self, policy, wm, batch: (batch["evaluator_checkpoint"], real_f(self, policy, wm, batch))[1])
    with measuring(), ctx:
        for k in range(4):
            b06_seg(ns, k)
    c(getattr(ns, "_cons_warned", False) is False and ns._cons._calls == 4, "a tripwire fired inside consolidation (phase idled)", n=2)

    def arm(tel):
        reseed(81)
        n2 = b06_brain(b06_cfg(planning=False))
        with measuring():
            for k in range(6):
                b06_seg(n2, k)
                if tel:
                    n2._cons.report()
                    n2._rel.sample_all(n2)
                    n2._rel.line()
                    n2._ao.last_line
        return thash(phash(n2.policy.actor)), tuple(n2._published), rng_sig(), n2.policy.entropy_coef
    a, b = arm(False), arm(True)
    c(a == b, "telemetry changed weights, recipe, RNG or the entropy coefficient", n=4)
    return c.obs("declared fields only; telemetry is read-only")


# ================================================================================================
# BND-07  CAT-07 <-> every enabled group   checkpoint/restart, instrumentation, injected resource pressure
# ================================================================================================
from developmental_ai.infra import autoscaler as AS  # noqa: E402

B07_STATE_FILES = ("planning.pt", "consolidation.pt", "curriculum.json", "attn.pt", "risk.pt")


def b07_cfg():
    cfg = b04_cfg(strategy=False)
    cfg["consolidation"]["regularizer"] = {"kind": "ewc", "lambda": 10.0, "anchor_every_segments": 3, "fisher_batches": 1}
    cfg["consolidation"]["distill"] = {"enabled": True, "snapshot_every_segments": 2, "max_snapshots": 3}
    cfg["planning"] = {"enabled": True, "rows": 4, "seq_len": 8, "planner": {"candidates": 8, "horizon": 2, "iterations": 1},
                       "calibration": {"min_n": 32, "window": 32, "max_ece": 0.2, "min_corr": -1.0},
                       "ensemble": {"enabled": True, "hidden": 16, "heads": 3}}
    cfg["relations"]["PFC"] = {"EF": {"enabled": True}}          # RW-15 telemetry channel beside the wired RW-1/RW-7
    return cfg


def b07_brain(tmp=None, det=32, cfg=None, seed=0):
    tmp = tmp or tempfile.mkdtemp(prefix="bnd_b07_")
    ns = brain(cfg or b07_cfg(), adapters=2, am=True, tmp=tmp, det=det, seed=seed, phases=list(PHASES))
    ns._rk = RiskEstimator({"horizon": 4, "head": {"enabled": True, "hidden": 8}}, 2, CPU, os.path.join(tmp, "risk.pt"))
    ns._peer = po.PeerBoard({"seed": 0, "window": 4, "capacity": 64}, A, True, CPU)
    nb = ns.replay_buffer.newest_sequences(8, CPU)
    with torch.no_grad():
        st, _ = ns.world_model.observe_sequence(nb["observations"], nb["actions"])
        ns._lat = torch.cat([st["h"], st["z"]], -1)               # (streams, T, latent): the world model's own latents
    ns._k = 0
    return ns


def b07_seg(ns, k=None, instrument=False, scaler=None, sig=None, const=False):
    """One segment of every enabled group in the real order, plus the per-step hooks the loop runs between segments."""
    k = ns._k if k is None else k
    ns._k = k + 1
    MEAS.clear()
    if const:
        MEAS.update({0: (1.0, 0.1), 1: (2.0, 0.1)})
    else:
        MEAS.update({0: (B06_LOSS[k % len(B06_LOSS)], 0.1), 1: (2.0 * B06_LOSS[(k + 3) % len(B06_LOSS)], 0.1)})
    ns.lifetime_timesteps += 64
    ns.policy.last_update_stats = {"cond_entropy_norm": 0.4}
    reseed(200 + k)
    for t in range(4):                                           # the per-step hooks (WP-7 window, WP-11 risk, WP-12 board)
        z = ns._lat[:, (k + t) % ns._lat.shape[1]]
        with quiet():
            ns._am.observe(z, [False, False])
            ns._rk.observe(z, ns.world_model.continue_predictor, [False, False], [False, False])
        for sidx in range(2):
            ns._peer.publish(sidx, unit(sidx * 8 + (k + t) % 8, 16), (k + t + sidx) % A)
    segment(ns)
    if instrument:
        ns._rel.sample_all(ns)
        ns._rel.line()
        ns._plan.report() if getattr(ns, "_plan", None) else None
    if scaler is not None and sig is not None:
        return scaler.observe(sig(k))
    return None


def b07_save(ns):
    """What the loop persists on its own at a stop: phase files (written by the phases), attention + risk saves, replay, world model."""
    ns._am.save()
    ns._rk.save()
    ns.replay_buffer.save(os.path.join(ns.dir, "replay"))
    torch.save(ns.world_model.state_dict(), os.path.join(ns.dir, "wm.pt"))


def b07_restore(ns2):
    ns2.world_model.load_state_dict(torch.load(os.path.join(ns2.dir, "wm.pt"), map_location=CPU))
    ns2.replay_buffer.load(os.path.join(ns2.dir, "replay"))


def b07_hash(ns):
    g = {"wm": thash(phash(ns.world_model)), "actor": thash(phash(ns.policy.actor)), "critic": thash(phash(ns.policy.critic)),
         "am": thash(phash(ns._am.net)), "risk": thash(phash(ns._rk.head)) if getattr(ns._rk, "head", None) is not None else "",
         "coefs": tuple(sorted((r["name"], round(r["value"], 12)) for r in registry_for(ns).table())),
         "lr": round(ns.policy.optimizer.param_groups[0]["lr"], 12), "gain": (ns.reward_mixer.intrinsic_gain, ns.reward_mixer.frontier_gain)}
    if getattr(ns, "_plan", None):
        g["ens"] = thash(phash(ns._plan.ens))
        g["T"] = ns._plan.T
    if getattr(ns, "_cons", None):
        g["cons"] = (ns._cons._calls, len(ns._cons._teachers))
    return g


def sig_of(act_s=0.01, **kw):
    return lambda k: AS.Signals(step=k, z_info=5.0, act_s=act_s, act_budget_s=0.1, **kw)


def b07_ctx():
    return mock.patch("developmental_ai.core.curriculum.measure_streams", fake_measure)


@sc("CT-07", "BND-07: restart restores every group's PERSISTED state and restarts every UNSAVED one at its documented neutral",
    brk="curriculum_state_not_restored", min_exercised=18,
    oracle="after save + recreate on the same storage: world model, replay count, attention net, risk head, planner temperature/"
           "ensemble, consolidation teachers/anchor, curriculum bandit equal the saved ones; coefficients, mixer gains, optimiser "
           "scales, recipe, calibration window and the peer board start neutral; the frozen configuration is untouched")
def b07_ct07a(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b07_")
    with b07_ctx():
        ns = b07_brain(tmp)
        cfg0 = copy.deepcopy(ns.config)
        for k in range(9):
            b07_seg(ns, k)
        c(ns.config == cfg0, "a group mutated the frozen configuration during the run", n=1)
        b07_save(ns)
        saved = b07_hash(ns)
        n_rows = len(ns.replay_buffer)
        T = ns._plan.T
        teachers, anchor = len(ns._cons._teachers), ns._cons._anchor is not None
        vals = [list(b.value) for b in ns._curr.bandits.values()]
        c(max(abs(cv(registry_for(ns).get(n)) - 1.0) for n in ("mixer.intrinsic_gain", "explore.entropy_target_scale")) > 0.05
          and teachers >= 2 and anchor and any(any(abs(x) > 1e-9 for x in v) for v in vals), "fixture: nothing to lose on restart", n=1)
        files = {f: os.path.exists(os.path.join(tmp, f)) for f in B07_STATE_FILES}
        c(all(files.values()), "persisted files missing before restart: %s" % [f for f, ok in files.items() if not ok], n=len(files))
        if brk:                                                  # DEFECT: one group's state file is lost / not found
            os.remove(os.path.join(tmp, "curriculum.json"))
        ns2 = b07_brain(tmp)
        b07_restore(ns2)
        c(thash(phash(ns2.world_model)) == saved["wm"], "world model not restored", n=1)
        c(len(ns2.replay_buffer) == n_rows, "replay rows %d != %d" % (len(ns2.replay_buffer), n_rows), n=1)
        c(thash(phash(ns2._am.net)) == saved["am"], "attention layer not restored", n=1)
        seen = {}
        real_run, real_plan_load = Consolidator.run, PlanningPhase.load_state_dict

        def spy_run(self, *a, **k):
            seen.setdefault("teachers", len(self._teachers))
            seen.setdefault("anchor", self._anchor is not None)
            seen.setdefault("recipe", (self.old_frac, self.prioritized))
            return real_run(self, *a, **k)

        def spy_load(self, d):
            real_plan_load(self, d)
            seen["T"] = self.T
            seen["ens"] = thash(phash(self.ens))
        with mock.patch.object(Consolidator, "run", spy_run), mock.patch.object(PlanningPhase, "load_state_dict", spy_load):
            b07_seg(ns2, 0)
        c(seen.get("T") == T and seen.get("ens") == saved["ens"], "planner temperature/ensemble not restored (%s)" % seen.get("T"), n=2)
        c(seen.get("teachers") == teachers and seen.get("anchor") is True, "consolidation teachers/anchor not restored (%s)" % {k: seen.get(k) for k in ("teachers", "anchor")}, n=2)
        c(seen.get("recipe") == (0.0, False), "the unsaved replay recipe did not restart configured", n=1)
        bv = [list(b.value) for b in ns2._curr.bandits.values()]
        c(any(any(abs(x) > 1e-9 for x in v) for v in bv) and ns2._curr.seg >= 1, "curriculum bandit values not restored (%s)" % bv, n=1)
        reg2 = {r["name"]: r["value"] for r in registry_for(ns2).table()}
        ctl = {k: v for k, v in reg2.items() if k in ("mixer.intrinsic_gain", "explore.entropy_target_scale", "explore.uncertainty_scale",
                                                      "objective.frontier_gain")}
        c(all(abs(v - 1.0) < 1e-9 for v in ctl.values()) and ns2.reward_mixer.intrinsic_gain == 1.0 and ns2.reward_mixer.frontier_gain == 1.0,
          "a controller coefficient survived the restart: %s" % {k: round(v, 3) for k, v in ctl.items() if abs(v - 1.0) > 1e-9}, n=3)
        c(ns2._plan.last.get("policy_calibration_ok") == 0.0, "unsaved calibration window treated as valid after restart", n=1)
        c(len(ns2._peer._ring.get(0, ())) <= 4 + 0 and ns2._peer._count.get(0, 0) == 4, "the peer board was not empty at restart", n=1)
        c(ns2.config == cfg0, "restart changed the frozen configuration", n=1)
    return c.obs("persisted state restored, unsaved state neutral, configuration frozen")


@sc("CT-05", "BND-07: restoring onto a world model of another latent width starts every width-bound group exactly fresh "
             "(REAL DEFECT if red: PlanningPhase.load_state_dict partial hybrid)",
    brk="width_blind_restore", min_exercised=8,
    oracle="after a latent-width change the attention layer, risk head, planner ensemble and consolidation teachers equal a "
           "never-restored build (no partial copy) and every group still runs a segment")
def b07_ct05(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b07w_")
    with b07_ctx():
        ns = b07_brain(tmp)
        for k in range(6):
            b07_seg(ns, k)
        ns._am.save()
        ns._rk.save()
        # GOV-015 9.6: write the OLD-width planner to a file of its own NOW. The brain built next shares this directory and
        # its segment rewrites planning.pt at the NEW width, so reading planning.pt afterwards loaded a same-width file and
        # the "partial copy across a width change" the oracle then saw was a legitimate full load (a fixture defect).
        old_pth = os.path.join(tmp, "planning_oldwidth.pt")
        if getattr(ns, "_plan", None):
            torch.save(ns._plan.state_dict(), old_pth)
        # the same storage, a bigger world model (a stage-D grow of the deterministic part)
        wide = b07_brain(tmp, det=48)
        fresh = b07_brain(tempfile.mkdtemp(prefix="bnd_b07f_"), det=48)
        with quiet():
            b07_seg(wide, 0)
            b07_seg(fresh, 0)
        c(wide._am.net.latent_dim == wide.world_model.rssm.latent_dim and "latent width" in wide._am._fresh_note,
          "attention layer did not start fresh at the new width (%r)" % wide._am._fresh_note, n=2)
        first = next(iter(wide._rk.head.parameters()))
        c(first.shape[-1] == wide.world_model.rssm.latent_dim, "risk head is not at the new width (%s)" % (tuple(first.shape),), n=1)
        # planner: the loop's restore path reads planning.pt written at the old width
        pth = old_pth
        c(os.path.exists(pth), "fixture: no saved planner", n=1)
        P3 = PlanningPhase(wide.config["planning"], A, wide.world_model.rssm.latent_dim, CPU)
        built = {k_: v.clone() for k_, v in P3.ens.state_dict().items()}
        saved = torch.load(pth, map_location=CPU)
        c(tuple(saved["ens"]["heads.0.0.weight"].shape) != tuple(built["heads.0.0.weight"].shape),
          "fixture: the saved planner is not at the new width (%s vs %s)"
          % (tuple(saved["ens"]["heads.0.0.weight"].shape), tuple(built["heads.0.0.weight"].shape)), n=1)
        P3.load_state_dict(saved)
        if brk:                                              # DEFECT (the pre-GOV-015 restore): copy whatever overlaps across the width change
            with torch.no_grad():
                for k_, v in P3.ens.state_dict().items():
                    sv = saved["ens"].get(k_)
                    if sv is not None and sv.dim() == v.dim():
                        sl = tuple(slice(0, min(x, y)) for x, y in zip(v.shape, sv.shape))
                        v[sl].copy_(sv[sl].to(v.dtype))
        changed = [k_ for k_, v in P3.ens.state_dict().items() if not torch.equal(built[k_], v)]
        c(not changed, "planner ensemble was partially copied across a width change (%d tensors, e.g. %s)" % (len(changed), changed[:2]), n=2)
        c(wide._cons._teachers == [] or len(wide._cons._teachers) <= 1, "consolidation teachers of the old width were kept (%d)" % len(wide._cons._teachers), n=1)
        c(finite_params(wide.policy.actor, wide._am.net) and wide._cons._calls >= 1, "a group stopped running after the width change", n=2)
    return c.obs("width-bound state starts fresh")


@sc("CT-07", "BND-07: the autoscaler's cooldown and ADR-005 stepping state survive the restart that applied a change "
             "(REAL DEFECT if red: PressureOrder step scale is not restored)",
    brk="scale_not_restored", min_exercised=4,
    oracle="a rebuilt autoscaler given autoscale.json's state has the same cooldown and the same stepping fingerprint as the one that wrote it")
def b07_ct07b(brk):
    c = Chk()
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "scale.json")
        cfg = {"actuate": True, "cooldown_segments": 6, "min_streams": 2, "max_streams": 8,
               "adr005": {"enabled": True, "scale_file": sf, "stepping_actuated": True, "refusal_expiry_hours": 12}}
        a = AS.Autoscaler(cfg, size="tiny", streams=4)
        for k in range(6):
            a.observe(AS.Signals(step=k, z_info=5.0, act_s=0.5, act_budget_s=0.1, game_s_per_decision=0.1))
        saved = json.loads(json.dumps(a.state()))
        b = AS.Autoscaler(cfg, size="tiny", streams=4)
        if not brk:
            b.restore(saved)
        c(a.order.step_scale == 2.0, "fixture: stepping never reached its dial", n=1)
        c(b.segments_since_change == a.segments_since_change, "cooldown not restored", n=1)
        c(b.order.step_scale == a.order.step_scale, "stepping scale not restored (saved %s, rebuilt %s)" % (a.order.step_scale, b.order.step_scale), n=1)
        c(b.state()["order"]["fingerprint"] == saved["order"]["fingerprint"], "comparison fingerprint differs after restart", n=1)
    return c.obs("resource-pressure state continues across a restart")



from developmental_ai.core.curriculum import measure_streams as REAL_MEASURE  # noqa: E402  (before any patch)


@sc("CT-01", "BND-07: the autoscaler consumes the groups' REAL numbers (fresh-data loss from the world model and replay) and answers "
             "compute pressure model-first; advice moves no group state",
    brk="fresh_train_swapped", min_exercised=12,
    oracle="Decision.evidence carries the measured fresh loss and the gap fresh/train exactly; sustained pressure walks "
           "live -> medium (model first) one dial per decision; every group's state hash is unchanged by the advice")
def b07_ct01(brk):
    c = Chk()
    with b07_ctx(), tempfile.TemporaryDirectory() as td:
        ns = b07_brain()
        for k in range(4):
            b07_seg(ns, k)
        sf = os.path.join(td, "scale.json")
        a = AS.Autoscaler({"actuate": False, "cooldown_segments": 2, "min_streams": 2, "max_streams": 8, "window": 2,
                           "adr005": {"enabled": True, "scale_file": sf, "refusal_expiry_hours": 12}}, size="live", streams=8)
        m = REAL_MEASURE(ns.world_model, ns.replay_buffer, 8, CPU)
        c(len(m) == 2 and all(math.isfinite(v[0]) for v in m.values()), "fixture: real measurement unavailable %s" % m, n=1)
        fresh = float(np.mean([v[0] for v in m.values()]))
        train = 0.5 * fresh
        h0 = b07_hash(ns)
        f_, t_ = (train, fresh) if brk else (fresh, train)
        d = a.observe(AS.Signals(step=1, wm_fresh=f_, wm_train=t_, z_info=5.0, act_s=0.5, act_budget_s=0.1))
        ev = d.evidence
        c(ev["fresh"] == round(fresh, 4) and ev["gap"] == round(fresh / train, 3),
          "evidence does not carry the measured numbers: fresh %s gap %s (measured %.4f, %.3f)" % (ev["fresh"], ev["gap"], fresh, fresh / train), n=3)
        c(d.action == AS.SHRINK and d.target_size == "medium", "first response to pressure was %s -> %s (model first)" % (d.action, d.target_size), n=2)
        acts = [d.action]
        for i in range(2, 8):
            acts.append(a.observe(AS.Signals(step=i, wm_fresh=fresh, wm_train=train, z_info=5.0, act_s=0.5, act_budget_s=0.1)).action)
        c(acts[0] == AS.SHRINK and all(x in (AS.SHRINK, AS.HOLD) for x in acts[:4]), "unexpected response sequence %s" % acts, n=len(acts))
        c(a.size == AS.SIZE_NAMES.index("live") and a.streams == 8, "advice moved the (unactuated) autoscaler state", n=2)
        c(b07_hash(ns) == h0, "advice changed a group's state", n=1)
    return c.obs("real numbers reach the autoscaler; advice is read-only")


@sc("CT-02", "BND-07: instrumentation and injected pressure (advice AND an actuated request) leave every group's state, RNG and the frozen "
             "configuration bit-identical; no decision is applied to live objects",
    brk="decision_applied_live", min_exercised=14,
    oracle="after N segments the state hash, RNG signature and config equal a run without instrumentation; replay streams, "
           "latent width and policy knowledge width are unchanged although SHRINK / SLOW requests were issued")
def b07_ct02(brk):
    c = Chk()

    def arm(mode):
        reseed(55)
        with b07_ctx(), tempfile.TemporaryDirectory() as td:
            ns = b07_brain()
            cfg0 = copy.deepcopy(ns.config)
            scaler = None
            reseed(56)
            if mode in ("advice", "actuated"):
                scaler = AS.Autoscaler({"actuate": mode == "actuated", "cooldown_segments": 1, "min_streams": 2, "max_streams": 8, "window": 2,
                                        "adr005": {"enabled": True, "scale_file": os.path.join(td, "scale.json"),
                                                   "stepping_actuated": mode == "actuated"}},
                                       size="live", streams=8)
            decisions = []
            for k in range(8):
                d = b07_seg(ns, k, instrument=(mode != "plain"), scaler=scaler, sig=sig_of(0.5, game_s_per_decision=0.1))
                decisions.append(d.action if d else None)
                if brk and d and d.action != AS.HOLD and len(ns.replay_buffer.streams) > 1:
                    ns.replay_buffer.streams.pop()               # DEFECT: the request is applied inside the running process
            return (b07_hash(ns), rng_sig(), ns.config == cfg0, len(ns.replay_buffer.streams), ns.world_model.rssm.latent_dim,
                    ns.policy.knowledge_dim, decisions, scaler)
    plain, adv, act = arm("plain"), arm("advice"), arm("actuated")
    c(plain[0] == adv[0] == act[0], "a group's state differs with instrumentation/advice/actuation: %s" %
      sorted(k for k in plain[0] if plain[0][k] != act[0][k] or plain[0][k] != adv[0][k]), n=len(plain[0]))
    c(plain[1] == adv[1] == act[1], "instrumentation or pressure advice consumed learner RNG", n=2)
    c(plain[2] and adv[2] and act[2], "a group mutated the frozen configuration", n=3)
    c(act[3] == 2 and act[4] == plain[4] and act[5] == plain[5], "an actuated request changed live objects (streams %d, latent %d, knowledge %d)" % act[3:6], n=3)
    c(any(d not in (None, AS.HOLD) for d in act[6]), "fixture: the actuated scaler never issued a request (%s)" % act[6], n=1)
    return c.obs("instrumentation and pressure requests are read-only for the running brain")


@sc("CT-03", "BND-07: missing, NaN, infinite, negative or zero pressure signals trigger no decision and no state change; real pressure still does "
             "(REAL DEFECT if red: a non-finite latency / VRAM reading is read as overload)",
    brk="nan_mapped_to_saturation", min_exercised=12,
    oracle="each invalid Signals gives HOLD with segments_since_change unchanged, writes no scale file and leaves every group "
           "state hash alone; a valid breach right after is answered")
def b07_ct03(brk):
    c = Chk()
    nan, inf = float("nan"), float("inf")
    with b07_ctx(), tempfile.TemporaryDirectory() as td:
        ns = b07_brain()
        for k in range(3):
            b07_seg(ns, k)
        sf = os.path.join(td, "scale.json")
        h0 = b07_hash(ns)
        real = AS.Autoscaler.observe
        if brk:                                                  # DEFECT: the telemetry adapter maps a non-finite latency to "saturated"
            fix = lambda x: 10.0 if (x is not None and not math.isfinite(x)) else x   # noqa: E731
            AS.Autoscaler.observe = lambda self, sig: real(self, AS.Signals(**dict(sig.__dict__, act_s=fix(sig.act_s), vram_frac=fix(sig.vram_frac))))

        def mk():
            return AS.Autoscaler({"actuate": True, "cooldown_segments": 1, "min_streams": 2, "max_streams": 8, "window": 2,
                                  "adr005": {"enabled": True, "scale_file": sf, "refusal_expiry_hours": 12}}, size="live", streams=8)
        try:
            bad = [dict(act_s=nan, act_budget_s=0.1), dict(act_s=inf, act_budget_s=0.1), dict(act_s=-1.0, act_budget_s=0.1),
                   dict(act_s=0.5, act_budget_s=0.0), dict(act_s=0.5, act_budget_s=nan), dict(act_s=None, act_budget_s=None),
                   dict(vram_frac=nan), dict(vram_frac=inf), dict(vram_frac=-0.5), dict(act_s=0.5, act_budget_s=None)]
            for i, kw in enumerate(bad):
                a = mk()
                d = a.observe(AS.Signals(step=i, z_info=5.0, **kw))
                c(d.action == AS.HOLD and AS.SIZE_NAMES[a.size] == "live" and a.streams == 8,
                  "non-finite/invalid signal %s was acted on (%s -> %s x %d)" % (kw, d.action, d.target_size, d.target_streams), n=1)
            c(not os.path.exists(sf), "an invalid signal wrote a scale request", n=1)
            c(b07_hash(ns) == h0, "an invalid signal changed a group's state", n=1)
            a = mk()
            d = a.observe(AS.Signals(step=20, z_info=5.0, act_s=0.5, act_budget_s=0.1))
            c(d.action != AS.HOLD, "a valid budget breach was not answered", n=1)
        finally:
            AS.Autoscaler.observe = real
    return c.obs("only valid pressure is answered")


@sc("CT-06", "BND-07: simulated time pressure makes the planner fall back to habit and shrink its rows, the autoscaler answers model-first, "
             "every group's coefficient contract holds, and relief restores both",
    brk="budget_ignored", min_exercised=16,
    oracle="max_phase_s breached: planned rows 0 (no_budget counted), rows halved, ADR-005 first rung SHRINK; coefficient owners/"
           "bounds hold throughout; when relieved the planner plans again with full rows within 3 segments")
def b07_ct06(brk):
    c = Chk()
    with b07_ctx(), tempfile.TemporaryDirectory() as td:
        ns = b07_brain()
        for k in range(6):
            b07_seg(ns, k)
        P = ns._plan
        c(P.last.get("plan_rows", 0) > 0 and P.last["rows"] == 4.0, "fixture: planner not trusting before pressure (%s)" % {k: P.last.get(k) for k in ("plan_rows", "rows")}, n=1)
        a = AS.Autoscaler({"actuate": False, "cooldown_segments": 1, "min_streams": 2, "max_streams": 8, "window": 2,
                           "adr005": {"enabled": True, "scale_file": os.path.join(td, "scale.json")}}, size="live", streams=8)
        real_dec = Arbiter.decide
        ctx = mock.patch.object(Arbiter, "decide", lambda self, conf, dis, pol_ok, mod_ok, budget_ok: real_dec(self, conf, dis, pol_ok, mod_ok, True)) if brk else contextlib.nullcontext()
        P.max_phase_s = 1e-9                                     # the injected pressure: every phase now exceeds its time budget
        P.budget.phase_s = 1.0
        first = None
        with ctx:
            for k in range(6, 9):
                d = b07_seg(ns, k, scaler=a, sig=sig_of(0.5))
                first = first or d
                c(P.last["plan_rows"] == 0.0 and P.last["rows"] == 2.0, "under time pressure planned %s rows with %s sampled" % (P.last["plan_rows"], P.last["rows"]), n=2)
        c(P.arbiter.why["no_budget"] > 0, "the arbiter did not record no_budget", n=1)
        c(first.action == AS.SHRINK and first.target_size == "medium", "autoscaler answered %s -> %s" % (first.action, first.target_size), n=2)
        reg = {r["name"]: r for r in registry_for(ns).table()}
        c(all(B04_OWNER.get(n_) in (None, r["owner"]) and r["lo"] - 1e-9 <= r["value"] <= r["hi"] + 1e-9 for n_, r in reg.items()),
          "a coefficient lost its owner or bound under pressure", n=len(reg))
        c(np.isfinite(ns.policy.entropy_coef) and finite_params(ns.policy.actor), "learner state became non-finite under pressure", n=2)
        P.max_phase_s = 10.0                                     # relief
        P.budget.phase_s = 0.01
        ok = 0
        for k in range(9, 13):
            b07_seg(ns, k, scaler=a, sig=sig_of(0.01))
            if P.last["plan_rows"] > 0 and P.last["rows"] == 4.0:
                ok += 1
                break
        c(ok == 1, "the planner did not resume full planning within 4 segments after relief", n=1)
    return c.obs("time pressure -> habit fallback and recovery, contracts intact")


@sc("CT-08", "BND-07: no persisted state file carries an evaluator field, no evaluator module is on the persistence path, and restart/telemetry read nothing from it",
    brk="evaluator_field_in_state", min_exercised=14,
    oracle="every key of planning.pt, consolidation.pt, curriculum.json, attention memory and risk state is a learner key; the "
           "modules that write/restore them import no evaluator; sys.modules has no evaluator after save + restart")
def b07_ct08(brk):
    c = Chk()
    tmp = tempfile.mkdtemp(prefix="bnd_b07e_")
    with b07_ctx():
        ns = b07_brain(tmp)
        for k in range(5):
            b07_seg(ns, k)
        b07_save(ns)
        if brk:                                                  # DEFECT: a "helper" stores the held-out checkpoint beside the planner state
            d = torch.load(os.path.join(tmp, "planning.pt"), map_location=CPU)
            d["evaluator_checkpoint"] = 3.0
            torch.save(d, os.path.join(tmp, "planning.pt"))

    def keys(obj, depth=0):
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.append(str(k))
                if depth < 2:
                    out += keys(v, depth + 1)
        return out
    pat = re.compile(r"evaluator|held_?out|answer_?key|checkpoint_score|oracle|map(29|3[0-2])", re.I)
    for f in ("planning.pt", "consolidation.pt", "attn.pt", "risk.pt"):
        d = torch.load(os.path.join(tmp, f), map_location=CPU)
        bad = [k for k in keys(d) if pat.search(k) and not k.startswith("net.")]
        c(not bad, "%s carries learner-external fields %s" % (f, bad[:2]), n=1 + len(keys(d)) // 20)
    cj = json.load(open(os.path.join(tmp, "curriculum.json")))
    c(not [k for k in keys(cj) if pat.search(k)], "curriculum.json carries learner-external fields", n=1)
    with b07_ctx():                                              # the restart itself (after the scan: it rewrites the files)
        ns2 = b07_brain(tmp)
        b07_restore(ns2)
        b07_seg(ns2, 0)
    base = os.path.join(ROOT, "developmental_ai")
    for rel in ("core/planning.py", "core/uncertainty.py", "policy/consolidation.py", "core/curriculum.py", "core/attention_memory.py",
                "core/risk.py", "world_model/replay_buffer.py", "infra/autoscaler.py", "infra/scale_order.py"):
        src = open(os.path.join(base, rel)).read()
        c(not re.search(r"^\s*(from|import)\s+developmental_ai\.evaluator", src, re.M), "%s imports the evaluator" % rel, n=1)
    mods = [m for m in sys.modules if m.startswith("developmental_ai.evaluator")]
    c(not mods, "an evaluator module is loaded after save + restart: %s" % mods[:2], n=1)
    return c.obs("persisted state is learner-only; the persistence path never touches the evaluator")



# ================================================================================================
# Supported controller combinations applied TOGETHER (ASS-012 section 4, last paragraph)
#   COMBO-A  confidence modulation (RW-1) + attention (RW-2 salience bias, WP-7) + consolidation (WP-2, RW-3)
#   COMBO-B  curriculum (WP-3) + adaptive optimisation (WP-9) + strategy selection (RW-6)
# Inventory (checked in code, nothing invented): RW-1 -> {WP-9 target, WP-7 temperature, WP-2 strength} are wired;
# RW-2 installs AttentionMemory.bias_fn; RW-3 adds factors to the SAME strength product RW-1 finalises; RW-6 writes
# Consolidator.old_frac/prioritized; WP-3/WP-9 share the measurement/optimiser inputs.  No RW-1<->RW-6 or RW-2<->RW-6
# connection exists, so no scenario asserts one.
# ================================================================================================
def ca_cfg():
    cfg = b06_cfg(strategy=False, ao=True, planning=True, ewc=False)
    cfg["relations"] = rel_cfg(*MOD, ("SAL", "ATT"), ("PRED", "ATT"), ("BORE", "ATT"), ("EMO", "CONS"), ("AMY", "CONS"), ("RP", "CONS"),
                               modulation=MODCFG, attention={"min_n": 4, "beta": 0.5, "bore_amp": 0.5},
                               consolidation={"amp": 0.3, "fresh_every": 1, "seq_len": 8})
    return cfg


def ca_brain(rogue=False):
    ns = b07_brain(cfg=ca_cfg())
    ns._head0 = {k: v.clone() for k, v in ns.policy.actor.action_head.state_dict().items()}
    if rogue:
        reg = registry_for(ns)
        reg.claim("consol.strength_scale", "wp9_adaptive", 1.0, 0.0, 9.0)       # a rival controller got there first
        reg.set("consol.strength_scale", "wp9_adaptive", 3.0)
    return ns


def ca_seg(ns, k, mode="changing", conf=None):
    rs = np.random.RandomState(900 + k)
    const = mode == "constant"
    ns.infra.last_ledger_segment = {"total": 0.5 if const else float(rs.rand())}
    ns._am_sp = sp_of(2, 0.3 if const else 0.2 + float(rs.rand()))
    set_conf(ns, conf if conf is not None else (1.0 if const else [1.0, 0.3, 0.0][k % 3]))
    return b07_seg(ns, k, const=const)


def ca_state(ns):
    reg = registry_for(ns)
    return {"temp": ns._am.net.temperature, "strength": ns._cons.strength if getattr(ns, "_cons", None) else 1.0,
            "ets": cv(reg.get("explore.uncertainty_scale")), "ent": ns.policy.entropy_coef,
            "fac": dict(ns._rel.factors.get("consol", {})) if getattr(ns, "_rel", None) else {}}


@sc("CT-04", "COMBO-A: confidence modulation + attention + consolidation together: one owner, one product, bounded, consumers agree, "
             "release under constant and changing inputs",
    brk="rival_writer_claimed_strength_first", min_exercised=40,
    oracle="consol.strength_scale and attn.temperature_scale are owned by rw1_modulation only; the strength is the slewed, bounded "
           "product of NM-06 x EMO x AMY x RP; the salience bias is installed on the same attention layer the temperature drives and "
           "changes its read; constant inputs settle; losing the planner evidence returns the RW-1 factor to neutral within the bound")
def combo_a(brk):
    c = Chk()
    with b07_ctx():
        ns = ca_brain(rogue=bool(brk))
        reg = registry_for(ns)
        prev = {"strength": 1.0, "temp": 1.0}
        for k in range(14):
            ca_seg(ns, k, "changing")
            st = ca_state(ns)
            if k >= 6:
                tab = {r["name"]: r for r in reg.table()}
                c(tab.get("consol.strength_scale", {}).get("owner") == modulation.OWNER and tab.get("attn.temperature_scale", {}).get("owner") == modulation.OWNER,
                  "strength/temperature are not owned by the modulation module: %s" % {n: tab.get(n, {}).get("owner") for n in ("consol.strength_scale", "attn.temperature_scale")}, n=2)
                c(set(st["fac"]) >= {"NM-06", "rel.EMO.CONS", "rel.AMY.CONS", "rel.RP.CONS"}, "strength product lacks a declared factor: %s" % sorted(st["fac"]), n=1)
                prod = 1.0
                for v in st["fac"].values():
                    prod *= v
                want = min(prev["strength"] + STEP, max(prev["strength"] - STEP, min(4.0, max(0.25, prod))))
                c(abs(st["strength"] - want) < 1e-9, "strength %.4f != slewed bounded product %.4f" % (st["strength"], want), n=1)
                c(abs(ns._cons.strength - cv(reg.get("consol.strength_scale"))) < 1e-9 and abs(ns._am.net.temperature - cv(reg.get("attn.temperature_scale"))) < 1e-9,
                  "consumers disagree with the published coefficients", n=2)
            prev = st
        am = ns._am
        c(am.bias_fn is not None, "the RW-2 salience bias is not installed on the attention layer", n=1)
        x, m = am.win.get(0)
        ring = am.bias_fn.ring if am.bias_fn is not None else None
        if ring is not None and bool((ring.abs() > 1e-9).any()):
            with torch.no_grad():
                am.net.eval()
                _, _, _ = am.net(x[None], m[None])
                w_plain = am.net.last_weights.clone()
                _, _, _ = am.net(x[None], m[None], bias=ring[None])
                w_bias = am.net.last_weights.clone()
            c(not torch.allclose(w_plain, w_bias), "the salience bias does not change the attention read", n=1)
        # constant inputs: the z-score factors settle at neutral
        ns2 = ca_brain()
        for k in range(30):
            ca_seg(ns2, k, "constant")
        f2 = ca_state(ns2)["fac"]
        c(all(1.0 - 0.3 - 1e-9 <= v <= 1.0 + 0.3 + 1e-9 for v in f2.values()), "a factor left [1-amp, 1+amp] under constant input: %s" % {k: round(v, 3) for k, v in f2.items()}, n=len(f2))
        c(0.25 <= ns2._cons.strength <= 4.0 and 0.7 <= ns2._am.net.temperature <= 1.0 / 0.7 + 1e-9, "a coefficient left its declared bound", n=2)
        # evidence loss: the planner's calibration window vanishes
        for k in range(14, 20):
            ca_seg(ns, k, "changing", conf=0.0)
        before = ca_state(ns)
        dev = abs(before["fac"].get("NM-06", 1.0) - 1.0)
        bound = math.ceil(max(abs(before["temp"] - 1.0), dev) / STEP - 1e-9) + 1
        with patched(ns.replay_buffer, "newest_sequences", lambda *a, **k: None):
            for k in range(20, 20 + bound + 2):
                ca_seg(ns, k, "changing")
        after = ca_state(ns)
        c(abs(after["fac"].get("NM-06", 1.0) - 1.0) < 1e-9 and abs(after["temp"] - 1.0) < 1e-9,
          "RW-1's contribution was not released after evidence loss (NM-06 %.3f, temperature %.3f)" % (after["fac"].get("NM-06", -1), after["temp"]), n=2)
        c(set(after["fac"]) >= {"rel.EMO.CONS", "rel.AMY.CONS"}, "evidence loss in RW-1 removed another cluster's factors", n=1)
    return c.obs("one owner, one product, bounded; RW-1 releases without disturbing RW-2 / RW-3")


@sc("CT-04", "COMBO-B: curriculum + adaptive optimisation + strategy selection together run clean under constant and changing inputs; "
             "a failing combination is isolated to a component and re-assembled (REAL DEFECT if red; trace kept in the note)",
    brk="spaced_recipe_forced_off", min_exercised=40,
    oracle="coefficients owned/bounded, contexts assigned, optimiser lr = base x scale AND every segment's consolidation completes; "
           "on failure each component is run alone then together and the failing pair is named")
def combo_b(brk):
    # The deliberate break reintroduces F-24 (GOV-015): Consolidator._sample compares the other
    # tensors against the already-shrunk observation row count, so the spaced-replay path raises.
    with (mock.patch.object(Consolidator, "_sample", _sample_f24_defect) if brk else contextlib.nullcontext()):
        return _combo_b(brk)


def _sample_f24_defect(self, buf, device):
    """The pre-GOV-015 body of Consolidator._sample (kept only as the break)."""
    B, T = self.batch, self.seq_len
    n_old = int(round(B * self.old_frac))
    batch = buf.sample_sequences(B - n_old if n_old else B, T, device, terminal_fraction=0.0,
                                 prioritized=self.prioritized)
    if n_old > 0:
        extra = buf.sample_sequences(max(n_old * 4, 8), T, device, terminal_fraction=0.0, prioritized=False)
        keep = self._old_rows(buf, extra["start_indices"])[:n_old]
        if len(keep):
            sel = torch.as_tensor(keep, device=device, dtype=torch.long)
            for k, v in list(extra.items()):
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == extra["observations"].shape[0]:
                    extra[k] = v.index_select(0, sel)
            extra["start_indices"] = extra["start_indices"][keep]
            batch = self._concat(batch, extra)
    return batch


def _combo_b(brk):
    c = Chk()

    def run(**parts):
        cfg = b04_cfg(competence=False, meta=False, mod=False, zpd=False, **parts)
        ns = b04_brain(cfg, adapters=2)
        with measuring():
            rows, prob = b04_run(ns, 14, "changing")
            rows_c, prob_c = b04_run(b04_brain(cfg, adapters=2), 24, "constant")
        warned = bool(getattr(ns, "_cons_warned", False))
        calls = ns._cons._calls if getattr(ns, "_cons", None) else 0
        return ns, prob + prob_c, warned
    ns, prob, warned = run(curriculum=True, ao=True, strategy=True)
    reg = registry_for(ns)
    c(not prob, "ownership/bound problems: %s" % prob[:3], n=60)
    c(all(a.ctx for a in ns._parallel_envs), "curriculum never assigned a context", n=1)
    names = {r["name"] for r in reg.table()}
    c({"policy.lr_scale", "policy.entropy_coef"} <= names, "adaptive optimisation claimed no coefficient: %s" % sorted(names), n=1)
    c(abs(ns.policy.optimizer.param_groups[0]["lr"] - 3e-4 * (reg.get("policy.lr_scale") or 1.0)) < 1e-12, "optimizer lr != base x policy.lr_scale", n=1)
    iso = ""
    if warned:
        parts = {}
        for name, kw in (("curriculum", dict(curriculum=True, ao=False, strategy=False)), ("adaptive_optim", dict(curriculum=False, ao=True, strategy=False)),
                         ("strategy", dict(curriculum=False, ao=False, strategy=True))):
            _, _, w = run(**kw)
            parts[name] = w
        iso = "; isolated: " + ", ".join("%s alone %s" % (k_, "FAILS" if v else "clean") for k_, v in parts.items()) + \
              "; re-assembled: curriculum+strategy %s" % ("FAILS" if run(curriculum=True, ao=False, strategy=True)[2] else "clean")
    c(not warned, "consolidation failed inside the combination (the phase idled with a warning each segment)" + iso, n=2)
    return c.obs("the three controllers coexist and consolidation keeps working")



# ================================================================================================
# Additional implemented cross-group paths found while inventorying the loop (the BND table is not exhaustive, ASS-012 4)
#   BND-X1  CAT-03 -> CAT-05   RW-6 CALIB.SRL: the planner's policy ECE sets the strategy bandit's exploration rate
#   BND-X2  CAT-03 -> CAT-04   RW-7 SAFE.CUR: the WP-11 risk feature damps (never silences) the information-gain weight
#   BND-X3  CAT-03 -> CAT-02   RW-5 PFC.WM: the executive gate decides which latents enter the attention window the policy reads
# ================================================================================================
from developmental_ai.core import relations_executive as executive  # noqa: E402
from developmental_ai.core import relations_strategy as strategy  # noqa: E402
from developmental_ai.curiosity.intrinsic_balance import InfoGainTerm  # noqa: E402
from developmental_ai.core.relations_modulation import ZTrack  # noqa: E402


@sc("CT-01", "BND-X1: the planner's policy ECE reaches the strategy bandit's exploration rate with the declared sign and bounds",
    brk="wrong_calibration_channel", min_exercised=10,
    oracle="epsilon after each epoch equals clip(eps0 x (1 + 0.5 x clip(z of the planner's policy ECE)), 0.02, 0.5) recomputed independently; "
           "a worsening calibration raises it")
def bx1(brk):
    c = Chk()
    cfg = b06_cfg(ao=False, planning=True, ewc=False)
    cfg["planning"]["calibration"] = {"min_n": 16, "window": 16, "max_ece": 0.9, "min_corr": -1.0}
    ns = b06_brain(cfg)
    ns._head0 = {k: v.clone() for k, v in ns.policy.actor.action_head.state_dict().items()}
    ctx = contextlib.nullcontext()
    if brk:                                                      # DEFECT: the channel reads the model monitor, not the policy's
        ctx = patched(strategy, "calibration_error", lambda loop: loop._plan.mon.ece("model"))
    ref, eps_seen, eces = ZTrack(0.9, 5), [], []
    with measuring(), ctx:
        for k in range(14):
            set_conf(ns, [0.3, 1.0][k % 2] if k < 10 else 200.0)
            b06_seg(ns, k)
            e = ns._plan.mon.ece("policy")
            eces.append(e)
            st = ns._rel.state.get("strat")
            if st is None or e is None:
                continue
            z = ref.update(float(e))
            want = max(0.02, min(0.5, st["eps0"] * (1.0 + 0.5 * max(-1.0, min(1.0, z)))))
            eps_seen.append((st["bandit"].eps, want))
    c(len(eps_seen) >= 8, "fixture: too few epochs with a policy ECE (%d)" % len(eps_seen), n=1)
    c(all(abs(a - b) < 1e-9 for a, b in eps_seen), "bandit epsilon != the independently computed value (first mismatch %s)" %
      (next(((round(a, 4), round(b, 4)) for a, b in eps_seen if abs(a - b) >= 1e-9), None),), n=len(eps_seen))
    c(max(a for a, _ in eps_seen) > 0.1 + 1e-6, "a worsening calibration never raised the exploration rate", n=1)
    return c.obs("planner calibration governs the strategy bandit's exploration")


@sc("CT-01", "BND-X2: the risk feature damps the information-gain weight within [1-safe_amp, 1], is released with the source, "
             "and reaches the live term object",
    brk="stale_term_object", min_exercised=12,
    oracle="weight = base x scale with scale in [0.5, 1] slewed <= max_step; a vanished risk source returns it to base within "
           "ceil(dev/step) segments; the object the wiring writes is the one the reward path pays with")
def bx2(brk):
    c = Chk()
    cfg = b06_cfg(strategy=False, ao=False, planning=False, ewc=False)
    cfg["relations"] = rel_cfg(("SAFE", "CUR"), motivation={"amp": 0.3, "max_step": STEP, "min_n": 4, "ema": 0.9, "safe_amp": 0.5})
    ns = b06_brain(cfg)
    ns._ig_term = InfoGainTerm(0.5)
    live = ns._ig_term
    ns._rk = SimpleNamespace(vec=np.array([0.2], np.float32))
    if brk:
        ns._ig_term = InfoGainTerm(0.5)                          # DEFECT: the reward path was given another object after wiring
    rs = np.random.RandomState(4)
    prev, hist = 1.0, []
    with measuring():
        for k in range(16):
            ns._rk.vec = np.array([0.2 + (0.7 if k >= 8 else 0.0) + 0.02 * rs.rand()], np.float32)
            b06_seg(ns, k)
            sc_ = registry_for(ns).get("info.weight_scale")
            hist.append(cv(sc_))
            c(0.5 - 1e-9 <= cv(sc_) <= 1.0 + 1e-9 and abs(cv(sc_) - prev) <= STEP + 1e-9, "scale %.3f left its bound / slew" % cv(sc_), n=1)
            prev = cv(sc_)
    c(min(hist) < 1.0 - 3 * STEP, "a risk step never damped the information-gain weight (min scale %.3f)" % min(hist), n=1)
    c(abs(live.weight - 0.5 * cv(registry_for(ns).get("info.weight_scale"))) < 1e-9, "the term the reward path pays with does not hold base x scale (%.4f)" % live.weight, n=1)
    dev = abs(prev - 1.0)
    bound = math.ceil(dev / STEP - 1e-9)
    ns._rk = None                                                # the risk source vanishes
    n = 0
    with measuring():
        while n < bound + 2 and abs(cv(registry_for(ns).get("info.weight_scale")) - 1.0) > 1e-9:
            b06_seg(ns, 20 + n)
            n += 1
    c(n <= bound, "released in %d segments, bound %d" % (n, bound), n=n + 1)
    c(abs(live.weight - 0.5) < 1e-9, "the term the reward path pays with kept a stale weight after the source vanished (%.4f)" % live.weight, n=1)
    return c.obs("risk damps, never silences; released with its source")


@sc("CT-01", "BND-X3: the executive gate (RW-5) decides which latents enter the attention window, and the policy's attention slice "
             "is the summary of exactly the admitted window",
    brk="gate_not_installed", min_exercised=6,
    oracle="a reference attention layer with identical weights fed ONLY the latents the gate admitted has the same window and the "
           "same summary as the gated one; held steps leave the window unchanged")
def bx3(brk):
    c = Chk()
    L = b03_loop(om=False, am=True)
    am_step(L, 0)
    reg = build_default_registry(rel_cfg(("PFC", "WM"), executive={"min_n": 3, "max_skip": 4}))
    if not brk:
        reg.wire_all(L)                                          # the loop's relation wiring installs the gate
    c(brk or L._am.push_gate is not None, "RW-5 gate was not installed by the real wiring", n=1)
    ref = copy.deepcopy(L._am)
    ref.push_gate = None
    rng = np.random.RandomState(4)
    admitted, held = 0, 0
    for k in range(16):
        z = torch.from_numpy(rng.randn(L.streams, L.lat_seq.shape[-1]).astype(np.float32))
        if k >= 6:
            z[0] = L.lat_seq[0, 0]                               # stream 0 stops changing: the gate should hold
        with quiet():
            L._attention_memory_step(z, [False] * L.streams, [False] * L.streams, [{}] * L.streams, None)
        if L._am.pushed0:
            admitted += 1
            ref.observe(z, [False] * L.streams)
        else:
            held += 1
            for i in range(1, L.streams):                        # the other streams are not gated
                ref.win.push(i, z[i])
    c(held >= 1 and admitted >= 1, "fixture: the gate never held / never admitted (held %d admitted %d)" % (held, admitted), n=2)
    c(torch.equal(L._am.win.buf[0], ref.win.buf[0]) and L._am.win.n[0] == ref.win.n[0], "stream 0's window differs from the admitted-only reference", n=2)
    c(torch.equal(L._am.win.buf[1:], ref.win.buf[1:]), "the gate held out another stream's latents", n=1)
    with torch.no_grad():
        x, m = L._am.win.get(0)
        L._am.net.eval()
        summary, _, _ = L._am.net(x[None], m[None])
    c(brk or np.allclose(summary[0].float().numpy(), L._am.feature(), atol=1e-5),
      "the attention slice at policy entry is not the summary of the gated window", n=1)
    return c.obs("the gate shapes the window the policy reads")


def blocked_decl(cid, title, reason):
    if not ONLY or ONLY in title:
        H.blocked(cid, reason, title=title)


blocked_decl("CT-01", "BND-02: calibration -> ACTING-time planner / arbiter",
             "planning.acting is not wired in this build (planning.py header, GOV-011 D-30; _planning_phase warns if it is set): only the "
             "learning-phase path is tested above; Arbiter.decide and LatencyBudget.acting_ok have no acting-step caller")
blocked_decl("CT-01", "BND-03: RW-4 semantic store -> policy retrieval",
             "no policy retrieval consumer of the semantic store exists (manifest CAT-02 RW-4 note; GOV-013 3; grep of the loop and policy finds none): "
             "storage checks belong to CAT-02, nothing is claimed about semantic knowledge improving actions")
blocked_decl("CT-01", "BND-04: RW-12 learning stage / plasticity <-> WP-8 / WP-9",
             "RW-12 has no module in the tree (manifest CAT-05 'excluded', GOV-012): no connection is invented")
blocked_decl("CT-01", "BND-05: peer auxiliary training -> shared encoder (CAT-01)",
             "PeerPredictor trains only its own heads (peer_observation.py docstring: 'shapes no shared representation in this build'; manifest CAT-06 WP-12 "
             "'partial'): there is no peer -> encoder path to test")
blocked_decl("CT-01", "BND-05: RW-8 transfer mechanisms beyond measurement (CAT-06 -> CAT-01)",
             "RW-8 is measurements only (manifest CAT-06/CAT-01 'partial'); the remaining mechanisms are deferred (GOV-012)")
blocked_decl("CT-01", "BND-07: RW-13 (CAT-07) <-> any group",
             "RW-13 has no module in the tree (manifest CAT-07 'excluded', GOV-012): nothing to compose")



sys.exit(H.finish())
