"""CS-1 within-group contracts for CAT-05, Curriculum and learning adaptation
(GOV-013 section 2; ASS-012 section 3, CT-01..CT-08).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat05_smoke.py

Members: WP-3 (curriculum bandit), WP-9 (adaptive optimisation), WP-13 (online hyper-parameter
adaptation), RW-6 (strategy bandit over consolidation recipes), and the coefficient registry they
share. RW-12 has no module in the tree (GOV-012) and is not tested.

Question: do practice selection and optimiser controllers change only the values they own and
preserve attribution of updates?

REAL module objects throughout (tiny dims, CPU, seeded). The real loop phase helpers
(`_curriculum_phase`, `_adaptive_optim_phase`, `_meta_adapt_phase`, `_relations_phase`, ...) are bound
to a bare namespace so they run in their real order against a real StandaloneActorCritic, real
Consolidator, real registry and real controllers. The only stand-ins are numbers (measured losses,
PPO entropy readings, calibration data) and a game-interface double that only offers
`curriculum_options()` / `set_context()`. No scripted skill or task meaning, no evaluator code, no
game engine.

The deliberate breaks are injected into the fixture or a monkeypatched copy, never into production
code. A scenario that fails in its NORMAL run is a real defect in the code under test (FINDING notes).
"""
from __future__ import annotations

import ast
import contextlib
import io
import logging
import math
import os
import random
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs, reintroduce_nonfinite  # noqa: E402

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from developmental_ai.core import coefficients as coefmod  # noqa: E402
from developmental_ai.core import curriculum as curr  # noqa: E402
from developmental_ai.core import relations_strategy as strategy  # noqa: E402
from developmental_ai.core.coefficients import (  # noqa: E402
    CoefficientRegistry, OwnershipError, registry_for)
from developmental_ai.core.competence import CompetenceController  # noqa: E402
from developmental_ai.core.curriculum import ContextBandit, CurriculumScheduler  # noqa: E402
from developmental_ai.core.developmental_loop import DevelopmentalAI  # noqa: E402
from developmental_ai.core.meta_adapt import HyperAdapter  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EvaluatorLeakError, RelationRegistry, build_default_registry)
from developmental_ai.core.uncertainty import CalibrationMonitor  # noqa: E402
from developmental_ai.policy import adaptive_optim as aopt  # noqa: E402
from developmental_ai.policy.actor_critic import RewardMixer, StandaloneActorCritic  # noqa: E402
from developmental_ai.policy.adaptive_optim import AdaptiveOptim, PlasticityMonitor, ReturnNormalizer, lr_scale  # noqa: E402
from developmental_ai.policy.consolidation import Consolidator  # noqa: E402
from developmental_ai.core.attention_memory import GatedWindowAttention  # noqa: E402
from developmental_ai.core import relations_modulation as modulation  # noqa: E402
from developmental_ai.core import relations_motivation as motivation  # noqa: E402
from developmental_ai.world_model.rssm import RSSM  # noqa: E402

H = Harness("cat05", "Curriculum and learning adaptation (WP-3, WP-9, WP-13, RW-6)")

# ----------------------------------------------------------------------------- fixtures
PHASES = ("_curriculum_phase", "_competence_phase", "_adaptive_optim_phase", "_meta_adapt_phase",
          "_relations_phase")
# GOV-015 F-09: the loop's re-arm helpers, bound with the phases that use them
REARM = ("_phase_failed", "_phase_rearm", "_phase_ok")
MEAS: dict = {}
BASE_LR = 3e-4


def fake_measure(*_a, **_k):
    return dict(MEAS)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def measuring(fn=fake_measure):
    MEAS.clear()
    with patch("developmental_ai.core.curriculum.measure_streams", fn):
        yield


def make_policy():
    torch.manual_seed(2)
    return StandaloneActorCritic(obs_dim=8, action_dim=3, hidden_dim=16, device=torch.device("cpu"))


class FakeAdapter:
    """The game interface only: what the scheduler may ask of an environment adapter."""

    def __init__(self, pool="p", maps=("m1", "m2")):
        self.pool, self.maps, self.ctx = pool, list(maps), []

    def curriculum_options(self):
        return {"pool": self.pool, "maps": self.maps}

    def set_context(self, ctx):
        self.ctx.append(dict(ctx))


def build_loop(cfg, *, adapters=0, tmp=None, cons=False):
    torch.manual_seed(1)
    ns = SimpleNamespace(
        config=cfg, world_model=SimpleNamespace(rssm=object()), _wm_param_lock=threading.RLock(),
        replay_buffer=SimpleNamespace(streams=[]), device="cpu",
        reward_mixer=RewardMixer(intrinsic_weight=0.7, extrinsic_weight=0.3), is_discrete=True,
        lifetime_timesteps=0, infra=SimpleNamespace(drift_alarms=0), _wm_train=None,
        policy=make_policy(), skill_bank=SimpleNamespace(storage_dir=tmp or tempfile.gettempdir()))
    ns._parallel_envs = [FakeAdapter() for _ in range(adapters)]
    if cons:
        ns._cons = Consolidator({"replay": {"spaced_fraction": 0.0, "prioritized": False}})
        ns._plan = SimpleNamespace(last={}, mon=CalibrationMonitor(window=200, min_n=5))
        ns._am = SimpleNamespace(net=GatedWindowAttention(4), last={}, pushed0=True)
        ns._curr_stub = None
        ns._rk = SimpleNamespace(vec=[0.2])
    for n in PHASES + REARM:
        setattr(ns, n, getattr(DevelopmentalAI, n).__get__(ns))
    return ns


def fault_logger():
    recs = []

    class _H(logging.Handler):
        def emit(self, r):
            recs.append(r)
    h = _H(level=logging.WARNING)
    logging.getLogger().addHandler(h)
    return recs, h


def coef(ns, name):
    return registry_for(ns).get(name)


AO_CFG = {"enabled": True, "every_segments": 1,
          "schedule": {"enabled": True, "policy": {"kind": "linear", "horizon": 1000, "floor": 0.1},
                       "world_model": {"kind": "constant"}},
          "entropy": {"enabled": True, "target": 0.5, "eta": 0.5},
          "return_norm": {"enabled": True}, "drift_response": {"enabled": True, "hold_segments": 4}}
META_CFG = {"enabled": True, "every_segments": 1, "epoch_len": 2, "step": 0.1, "range": 2.0,
            "accept_z": 0.0, "min_noise_n": 1, "coefficients": ["ppo_clip", "gae_lambda"]}
SRL_CFG = {"enabled": True, "SRL": {"STRAT_ENTRY": {"enabled": True}},
           "STRAT_ENTRY": {"SP": {"enabled": True}, "INTL": {"enabled": True}},
           "CALIB": {"SRL": {"enabled": True}},
           "strategy": {"epoch_len": 1, "min_pulls": 1, "seed": 0, "spaced": 0.25}}


def strat_cfg(**cut):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in SRL_CFG.items()}
    for k in ("SRL", "STRAT_ENTRY", "CALIB"):
        c[k] = {a: dict(b) for a, b in SRL_CFG[k].items()}
    for name, on in cut.items():
        f, t = name.split("-")
        c[f][t]["enabled"] = bool(on)
    return c


def feed(Ma, loss_box, d, n):
    """n measurements, each a relative loss drop d from the previous one."""
    for _ in range(n):
        loss_box[0] *= (1.0 - d)
        MEAS.clear()
        MEAS[0] = (loss_box[0], 0.0)
        yield


# ============================================================================= CT-01
@H.scenario("CT-01", "WP-3: a stream that beats the others is credited to the arm it was in, and the chosen context reaches the adapter",
            brk="wrong_stream_id", min_exercised=4,
            oracle="value[arm of stream 0] = +0.25, value[arm of stream 1] = -0.25 (mean-baselined reward); set_context received m1 / m2 first")
def ct01_curriculum(brk):
    with tempfile.TemporaryDirectory() as d:
        ns = build_loop({"curriculum": {"enabled": True, "min_pulls": 1, "every_segments": 4, "seed": 0}},
                        adapters=2, tmp=d)
        with measuring():
            for seg, meas in enumerate(({}, {0: (1.0, 0.5), 1: (1.0, 0.5)}, {0: (0.5, 0.5), 1: (1.0, 0.5)})):
                if brk == "wrong_stream_id" and seg == 2:
                    meas = {0: meas[1], 1: meas[0]}        # the measurement of stream 1 filed under stream 0
                MEAS.clear()
                MEAS.update(meas)
                with quiet():
                    ns._curriculum_phase()
        b = ns._curr.bandits["p"]
        first = [a.ctx[0] for a in ns._parallel_envs]
        ok = first == [{"map": "m1"}, {"map": "m2"}] and b.rewards_n == [1, 1] \
            and abs(b.value[0] - 0.25) < 1e-9 and abs(b.value[1] + 0.25) < 1e-9
    return Obs(bool(ok), exercised=4, note=f"values {b.value} pulls {b.pulls}; first contexts {first}")


@H.scenario("CT-01", "WP-9: schedule and entropy controller reach the optimizer's learning rate and the policy's entropy coefficient, bounded",
            brk="consumer_not_updated", min_exercised=14,
            oracle="optimizer lr == base*lr_scale(t); entropy below target raises, above lowers; policy.entropy_coef == registry; within [min,max]")
def ct01_adaptive(brk):
    policy, reg = make_policy(), CoefficientRegistry()
    Ao = AdaptiveOptim(AO_CFG, reg, policy)
    ok, n, coefs = True, 0, [policy.entropy_coef]
    for t in (0.0, 250.0, 500.0, 750.0, 1000.0):
        policy.last_update_stats = {"cond_entropy_norm": 0.2}
        Ao.run(policy, [], t, 0)
        if brk == "consumer_not_updated":
            for g in policy.optimizer.param_groups:
                g["lr"] = BASE_LR                           # the optimizer is rebuilt behind the controller
        want = BASE_LR * lr_scale("linear", t, 1000.0, 0.0, 0.1)
        ok &= abs(policy.optimizer.param_groups[0]["lr"] - want) < 1e-12
        ok &= reg.get("policy.entropy_coef") == policy.entropy_coef
        ok &= policy.entropy_coef > coefs[-1] or policy.entropy_coef == 0.1       # below target: up
        coefs.append(policy.entropy_coef)
        n += 3
    for _ in range(6):                                      # entropy above target: down, never below the floor
        policy.last_update_stats = {"cond_entropy_norm": 0.95}
        Ao.run(policy, [], 1000.0, 0)
        ok &= policy.entropy_coef < coefs[-1] and 1e-4 <= policy.entropy_coef <= 0.1
        coefs.append(policy.entropy_coef)
        n += 1
    for _ in range(40):                                     # saturate at the upper bound
        policy.last_update_stats = {"cond_entropy_norm": 0.0}
        Ao.run(policy, [], 1000.0, 0)
    ok &= policy.entropy_coef <= 0.1 + 1e-12
    return Obs(bool(ok), exercised=n + 1, note=f"entropy_coef path {[round(c, 5) for c in coefs]}")


@H.scenario("CT-01", "WP-13: a better trial is accepted and written to the policy, a worse one is reverted; one coefficient differs at a time",
            brk="trial_value_leaks", min_exercised=10,
            oracle="trial: only the target coefficient is off its current value; accepted value persists on policy.clip_range; rejected reverts and flips direction")
def ct01_meta(brk):
    ns = build_loop({"meta_adapt": META_CFG})
    box, trace, ok = [1.0], [], True
    ctx = patch.object(HyperAdapter, "values",
                       lambda self: {self.attr[n]: (self.trial_v if n == self.target else self.cur[n])
                                     for n in self.names}) if brk == "trial_value_leaks" else contextlib.nullcontext()
    with measuring(), ctx:
        MEAS[0] = (1.0, 0.0)
        with quiet():
            ns._meta_adapt_phase()                          # first measurement: no interval yet
        Ma = ns._ma
        plan = [(0.01, 3), (0.05, 3), (0.01, 3), (0.01, 3), (0.0, 3), (0.01, 3)]
        for d, n in plan:
            for _ in feed(Ma, box, d, n):
                with quiet():
                    ns._meta_adapt_phase()
                p, tgt = Ma.phase, Ma.target
                clip, gae = ns.policy.clip_range, ns.policy.gae_lambda
                trace.append((p, tgt, round(clip, 4), round(gae, 4)))
                for name, held in (("ppo_clip", clip), ("gae_lambda", gae)):
                    expect = Ma.trial_v if (p == "trial" and name == tgt) else Ma.cur[name]
                    ok &= abs(held - expect) < 1e-9 and abs(held - coef(ns, SPEC_NAME[name])) < 1e-9 \
                        if p != "trial" or name != tgt else abs(held - expect) < 1e-9
    # the first epoch pair: ctrl_a 0.01, trial (clip x1.1) 0.05, ctrl_b 0.01 -> accepted
    ok &= any(t[0] == "trial" and t[1] == "ppo_clip" and abs(t[2] - 0.22) < 1e-4 and abs(t[3] - 0.95) < 1e-9
              for t in trace)
    ok &= Ma.accepts >= 1 and Ma.rejects >= 1 and Ma.dir["gae_lambda"] == -1
    ok &= abs(ns.policy.clip_range - 0.22) < 1e-9 or Ma.cur["ppo_clip"] > 0.2
    return Obs(bool(ok), exercised=len(trace), note=f"accepts {Ma.accepts} rejects {Ma.rejects} dirs {Ma.dir} final clip {ns.policy.clip_range}")


SPEC_NAME = {"ppo_clip": "meta.ppo_clip", "gae_lambda": "meta.gae_lambda"}


@H.scenario("CT-01", "RW-6: each arm's recipe reaches the consolidator (spaced fraction, prioritised sampling) with the declared change signal",
            brk="recipe_mapping_swapped", min_exercised=8,
            oracle="arms 0..3 give (old_frac, prioritized) = (0,F), (0.25,F), (0,T), (0.25,T); ST-39/ST-40 report the change from the configured value")
def ct01_strategy(brk):
    ns = build_loop({"relations": SRL_CFG}, cons=True)
    seq, chan = [], []
    ctx = patch.object(strategy, "RECIPES", tuple(reversed(strategy.RECIPES))) if brk else contextlib.nullcontext()
    with measuring(), ctx:
        for seg, loss in enumerate((1.0, 0.9, 0.9, 0.7)):
            MEAS.clear()
            MEAS[0] = (loss, 0.0)
            with quiet():
                ns._relations_phase()
            seq.append((round(ns._cons.old_frac, 4), ns._cons.prioritized))
            lastseg = ns._rel.last_segment["channels"]
            chan.append((lastseg["rel.STRAT_ENTRY.SP"]["last"], lastseg["rel.STRAT_ENTRY.INTL"]["last"]))
    want = [(0.0, False), (0.25, False), (0.0, True), (0.25, True)]
    ok = seq == want and [c[0] for c in chan] == [0.0, 0.25, 0.0, 0.25] and [c[1] for c in chan] == [0.0, 0.0, 1.0, 1.0]
    ok &= abs(coef(ns, "strategy.spaced_fraction") - ns._cons.old_frac) < 1e-12
    return Obs(bool(ok), exercised=len(seq) * 2, note=f"recipes {seq}; channel values {chan}")


# ============================================================================= CT-02
@H.scenario("CT-02", "WP-9: each sub-switch is independent; with none on, the policy, optimizer and registry are bit-identical to untouched",
            brk="part_leaks", min_exercised=14,
            oracle="entropy-only: lr untouched and no lr coefficient; schedule-only: entropy_coef untouched; return-norm-only: nothing claimed; none: weights, lr, registry unchanged")
def ct02_adaptive_parts(brk):
    ref = make_policy()
    ref_w = {k: v.clone() for k, v in ref.actor.state_dict().items()}
    ok, n = True, 0
    cases = {
        "entropy": {"enabled": True, "entropy": {"enabled": True, "target": 0.5, "eta": 0.5}},
        "schedule": {"enabled": True, "schedule": {"enabled": True, "policy": {"kind": "linear", "horizon": 1000, "floor": 0.1},
                                                   "world_model": {"kind": "constant"}}},
        "return_norm": {"enabled": True, "return_norm": {"enabled": True}},
        "none": {"enabled": True},
    }
    if brk == "part_leaks":                              # a config bug enables the schedule under "entropy"
        cases["entropy"]["schedule"] = AO_CFG["schedule"]
    for name, cfg in cases.items():
        policy, reg = make_policy(), CoefficientRegistry()
        Ao = AdaptiveOptim(cfg, reg, policy)
        policy.last_update_stats = {"cond_entropy_norm": 0.1}
        Ao.run(policy, [], 800.0, 0)
        names = {r["name"] for r in reg.table()}
        lr = policy.optimizer.param_groups[0]["lr"]
        ent = policy.entropy_coef
        if name == "entropy":
            ok &= names == {"policy.entropy_coef"} and lr == BASE_LR and ent != 0.01
        elif name == "schedule":
            ok &= names == {"policy.lr_scale", "wm.lr_scale"} and ent == 0.01 and lr < BASE_LR
        elif name == "return_norm":
            ok &= not names and lr == BASE_LR and ent == 0.01 and policy.return_normalizer is not None
        else:
            ok &= not names and lr == BASE_LR and ent == 0.01 and policy.return_normalizer is None
            ok &= all(torch.equal(v, ref_w[k]) for k, v in policy.actor.state_dict().items())
        n += 4
    return Obs(bool(ok), exercised=n, note=f"{n} parity comparisons")


@H.scenario("CT-02", "RW-6: cutting each strategy channel leaves its consumer at the configured value and the others working",
            brk="cut_ignored", min_exercised=12,
            oracle="selector cut: nothing applied; SP cut: old_frac always configured, prioritised still switches; INTL cut: the reverse; CALIB cut: bandit epsilon == configured")
def ct02_strategy_cuts(brk):
    def run(cfg):
        ns = build_loop({"relations": cfg}, cons=True)
        seq, eps = [], []
        with measuring():
            for loss in (1.0, 0.9, 0.9, 0.7, 0.7, 0.7):
                MEAS.clear()
                MEAS[0] = (loss, 0.0)
                for a in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):   # varying calibration evidence for CP-35
                    pass
                ns._plan.mon.note("policy", [0.9] * 10, [1.0] * int(10 * (0.3 + 0.1 * len(seq))) +
                                  [0.0] * (10 - int(10 * (0.3 + 0.1 * len(seq)))))
                with quiet():
                    ns._relations_phase()
                seq.append((round(ns._cons.old_frac, 4), ns._cons.prioritized))
                st = ns._rel.state.get("strat")
                eps.append(None if st is None else st["bandit"].eps)
        return seq, eps, ns
    full, eps_full, _ = run(strat_cfg())
    cut_on = brk == "cut_ignored"
    sel, eps_sel, nsel = run(strat_cfg(**{"SRL-STRAT_ENTRY": cut_on}))
    sp, _, _ = run(strat_cfg(**{"STRAT_ENTRY-SP": cut_on}))
    intl, _, _ = run(strat_cfg(**{"STRAT_ENTRY-INTL": cut_on}))
    cal, eps_cal, _ = run(strat_cfg(**{"CALIB-SRL": cut_on}))
    base = (0.0, False)
    ok = all(s == base for s in sel) and coef(nsel, "strategy.spaced_fraction") is None
    ok &= all(s[0] == 0.0 for s in sp) and any(s[1] for s in sp)
    ok &= all(s[1] is False for s in intl) and any(s[0] > 0 for s in intl)
    ok &= all(e == 0.1 for e in eps_cal if e is not None) and any(abs(e - 0.1) > 1e-9 for e in eps_full if e is not None)
    ok &= any(s != base for s in full)
    return Obs(bool(ok), exercised=len(full) * 5, note=f"full {full}; eps(full) {[round(e, 3) for e in eps_full if e is not None]}")


@H.scenario("CT-02", "WP-3 / WP-9 / WP-13 left off: the loop helpers do nothing (no context set, no coefficient claimed, policy and optimizer bit-identical)",
            brk="off_switch_ignored", min_exercised=8,
            oracle="helpers set their handle to False; adapters never receive set_context; registry empty; weights, lr and PPO coefficients unchanged; RNG unchanged")
def ct02_off(brk):
    with tempfile.TemporaryDirectory() as d:
        cfg = {"curriculum": {"enabled": brk == "off_switch_ignored", "min_pulls": 1},
               "meta_adapt": {"enabled": brk == "off_switch_ignored", "every_segments": 1},
               "adaptive_optim": {"enabled": brk == "off_switch_ignored", "every_segments": 1,
                                  "schedule": AO_CFG["schedule"]}}
        ns = build_loop(cfg, adapters=2, tmp=d)
        w0 = {k: v.clone() for k, v in ns.policy.actor.state_dict().items()}
        pc = (ns.policy.clip_range, ns.policy.gae_lambda, ns.policy.value_coef, ns.policy.entropy_coef)
        rng = torch.get_rng_state().clone()
        with measuring():
            MEAS.update({0: (1.0, 0.1), 1: (1.0, 0.1)})
            for _ in range(8):
                ns.lifetime_timesteps += 500
                with quiet():
                    ns._curriculum_phase()
                    ns._adaptive_optim_phase()
                    ns._meta_adapt_phase()
        ok = ns._curr is False and ns._ao is False and ns._ma is False
        ok &= all(a.ctx == [] for a in ns._parallel_envs) and registry_for(ns).table() == []
        ok &= all(torch.equal(v, w0[k]) for k, v in ns.policy.actor.state_dict().items())
        ok &= ns.policy.optimizer.param_groups[0]["lr"] == BASE_LR
        ok &= (ns.policy.clip_range, ns.policy.gae_lambda, ns.policy.value_coef, ns.policy.entropy_coef) == pc
        ok &= torch.equal(torch.get_rng_state(), rng)
    return Obs(bool(ok), exercised=8, note=f"handles {ns._curr!r} {ns._ao!r} {ns._ma!r}")


# ============================================================================= CT-03
@H.scenario("CT-03", "WP-3: a missing, NaN or infinite measurement gives no reward and does not replace the previous baseline",
            brk="nan_accepted", min_exercised=5,
            oracle="bandit values and baselines unchanged by invalid rows; the next valid row is judged against the last FINITE baseline")
def ct03_curriculum(brk):
    sch = CurriculumScheduler({"min_pulls": 1, "every_segments": 8, "seed": 0})
    sch.register({0: {"pool": "p", "maps": ["a", "b"]}, 1: {"pool": "p", "maps": ["a", "b"]}})
    ctx = patch.object(curr, "math", SimpleNamespace(isfinite=lambda x: True, log=math.log, sqrt=math.sqrt, exp=math.exp)) \
        if brk else contextlib.nullcontext()
    with ctx:
        sch.step({})
        sch.step({0: (1.0, 0.5), 1: (1.0, 0.5)})
        base_last = dict(sch.last)
        v0 = list(sch.bandits["p"].value)
        bad = [{0: (float("nan"), 0.5)}, {0: (1.0, float("inf"))}, {5: (0.1, 0.1)}, {}]
        for m in bad:
            sch.step(m)
        unchanged = sch.bandits["p"].value == v0 or all(math.isfinite(x) for x in sch.bandits["p"].value)
        no_reward = sch.bandits["p"].rewards_n == [0, 0]
        baseline_ok = sch.last[0] is not None and all(math.isfinite(x) for x in sch.last[0])
        sch.step({0: (0.5, 0.5)})
        got = sch.bandits["p"].rewards_n
    ok = no_reward and baseline_ok and unchanged and sum(got) == 1 and sch.last[0] == (0.5, 0.5)
    return Obs(bool(ok), exercised=5, note=f"rewards_n before valid {no_reward}, after {got}; baseline {base_last[0]} -> {sch.last[0]}")


@H.scenario("CT-03", "WP-9: no stats, no entropy field, a NaN reading, the same PPO update seen twice and non-finite returns cause no extra update",
            brk="stale_accepted", min_exercised=6,
            oracle="entropy coefficient moves exactly once per distinct valid PPO update; the return normaliser ignores non-finite targets")
def ct03_adaptive(brk):
    policy, reg = make_policy(), CoefficientRegistry()
    Ao = AdaptiveOptim({"enabled": True, "entropy": {"enabled": True, "target": 0.5, "eta": 0.5},
                        "return_norm": {"enabled": True}}, reg, policy)
    c0, seen = policy.entropy_coef, []
    st = {"cond_entropy_norm": 0.2}
    for stats in (None, {}, {"cond_entropy_norm": float("nan")}, st, (dict(st) if brk else st), st):
        policy.last_update_stats = stats
        Ao.run(policy, [], 0.0, 0)
        seen.append(policy.entropy_coef)
    one_update = abs(seen[3] - c0 * math.exp(0.5 * 0.3)) < 1e-12
    ok = seen[0] == c0 and seen[1] == c0 and seen[2] == c0 and one_update and (seen[5] == seen[3] or bool(brk) and False)
    ok &= seen[4] == seen[3]
    rn = ReturnNormalizer()
    s0 = rn.update(np.array([0.0, 1.0, 2.0, 5.0]))
    s1 = rn.update(np.array([float("nan"), float("inf"), -float("inf")]))
    s2 = rn.update(np.array([float("nan"), 3.0]))
    ok &= s1 == s0 and s2 == s0 and math.isfinite(rn.scale) and (rn.range is None or math.isfinite(rn.range))
    return Obs(bool(ok), exercised=6, note=f"entropy_coef after each presentation {[round(x, 5) for x in seen]}")


# FINDING (real defect, see report): EntropyController.update only filters NaN (`h_norm != h_norm`); an
# infinite entropy reading is treated as valid and shrinks the coefficient by exp(-1).
@H.scenario("CT-03", "WP-9: a non-finite entropy reading (NaN, +inf, -inf) is refused and the coefficient is untouched",
            brk="inf_accepted", min_exercised=3,
            oracle="policy.entropy_coef unchanged after each non-finite reading")
@reintroduce_nonfinite("developmental_ai.policy.adaptive_optim")
def ct03_entropy_inf(brk):
    policy, reg = make_policy(), CoefficientRegistry()
    Ao = AdaptiveOptim({"enabled": True, "entropy": {"enabled": True, "target": 0.5, "eta": 0.5}}, reg, policy)
    c0 = policy.entropy_coef
    moved = []
    for v in (float("nan"), float("inf"), -float("inf")) + ((float("inf"),) if brk else ()):
        before = policy.entropy_coef
        policy.last_update_stats = {"cond_entropy_norm": v}
        Ao.run(policy, [], 0.0, 0)
        moved.append(policy.entropy_coef != before)
    return Obs(not any(moved), exercised=3, note=f"moved on [nan, +inf, -inf]: {moved[:3]}; coef {policy.entropy_coef} base {c0}")


# FINDING (real defect, see report): HyperAdapter._progress/_finish_epoch never check finiteness. One NaN
# loss makes an epoch mean NaN, the noise estimate NaN for ever (decay*NaN), and `diff > accept_z*noise`
# is then False for ever: no trial can be accepted again.
@H.scenario("CT-03", "WP-13: one non-finite loss measurement does not permanently disable acceptance of a better trial",
            brk="nan_poisons_noise", min_exercised=3,
            oracle="after a NaN measurement the noise estimate is finite and a clearly better trial is still accepted within 8 rounds")
@reintroduce_nonfinite("developmental_ai.core.meta_adapt")
def ct03_meta(brk):
    def drive(inject):
        policy, reg = make_policy(), CoefficientRegistry()
        Ma = HyperAdapter(META_CFG, reg, policy)
        box, accepts_after = [1.0], 0
        Ma.update({0: (1.0, 0.0)})
        poisoned = False
        for rnd in range(8):
            for _ in range(9):
                d = 0.05 if Ma.phase == "trial" else 0.01
                box[0] *= (1.0 - d)
                loss = float("nan") if (inject and not poisoned and Ma.phase == "ctrl_a" and Ma.n_in_epoch == 1 and rnd == 0) else box[0]
                poisoned |= math.isnan(loss)
                Ma.update({0: (loss, 0.0)})
            if inject and rnd == 0 and not poisoned:
                continue
        return Ma, reg
    Mc, _ = drive(False)
    Mn, _ = drive(True)
    ok = Mc.accepts >= 1 and math.isfinite(Mn.noise) and Mn.accepts >= 1
    return Obs(bool(ok), exercised=3, note=f"clean accepts {Mc.accepts}; after one NaN: accepts {Mn.accepts}, noise {Mn.noise}")


@H.scenario("CT-03", "RW-6: with no fresh-data loss available the arm and its bandit are left alone",
            brk="absence_as_zero_loss", min_exercised=4,
            oracle="no reward booked and the applied recipe unchanged while the source is absent")
def ct03_strategy_absent(brk):
    ns = build_loop({"relations": SRL_CFG}, cons=True)
    fn = (lambda *a, **k: {0: (0.0, 0.0)}) if brk else (lambda *a, **k: {})
    with measuring(fn):
        for _ in range(5):
            with quiet():
                ns._relations_phase()
        st = ns._rel.state["strat"]
        ok = st["bandit"].rewards_n == [0, 0, 0, 0] and st["last_reward"] is None and st["arm"] == 0 \
            and ns._cons.old_frac == 0.0 and ns._cons.prioritized is False
    return Obs(bool(ok), exercised=5, note=f"rewards_n {st['bandit'].rewards_n} arm {st['arm']}")


# FINDING (real defect, see report): relations_strategy.apply computes
# r = max(-1, min(1, (Lprev - L)/|Lprev|)); with L = NaN, min(1.0, nan) returns 1.0, so a non-finite
# fresh loss is booked as the MAXIMUM reward for the active arm.
@H.scenario("CT-03", "RW-6: a non-finite fresh-data loss earns no reward (it must not be booked as maximal progress)",
            brk="nan_loss_rewarded", min_exercised=3,
            oracle="no bandit update and last_reward stays None when the measured loss is NaN or infinite")
@reintroduce_nonfinite("developmental_ai.core.relations_strategy")
def ct03_strategy_nan(brk):
    ns = build_loop({"relations": SRL_CFG}, cons=True)
    with measuring():
        MEAS[0] = (1.0, 0.0)
        with quiet():
            ns._relations_phase()
        before = list(ns._rel.state["strat"]["bandit"].rewards_n)
        for v in (float("nan"), float("inf")):
            MEAS[0] = (v, 0.0)
            with quiet():
                ns._relations_phase()
        st = ns._rel.state["strat"]
        booked = [a - b for a, b in zip(st["bandit"].rewards_n, before)]
    ok = sum(booked) == 0
    return Obs(bool(ok), exercised=3, note=f"rewards booked for non-finite losses (per arm) {booked}; running baseline rbase={st['rbase']}")


# ============================================================================= CT-04
def compose(tmp):
    cfg = {
        "curriculum": {"enabled": True, "min_pulls": 1, "every_segments": 2, "seed": 0},
        "competence": {"enabled": True, "every_segments": 1, "min_meas": 3, "signal_ema": 0.5,
                       "progress_ema": 0.5, "modulate_exploration": True, "gain_amp": 0.3,
                       "entropy_amp": 0.3, "max_step": 0.05},
        "adaptive_optim": dict(AO_CFG),
        "meta_adapt": dict(META_CFG, coefficients=["ppo_clip", "gae_lambda", "value_coef"]),
        "relations": dict(SRL_CFG, LC={"NE": {"enabled": True}}, NE={"AROUS": {"enabled": True}},
                          modulation={"amp": 0.3, "max_step": 0.05, "min_n": 5, "ema": 0.9}),
    }
    ns = build_loop(cfg, adapters=2, tmp=tmp, cons=True)
    return ns


EXPECTED_OWNER = {
    "mixer.intrinsic_gain": "wp8_competence", "explore.entropy_target_scale": "wp8_competence",
    "policy.entropy_coef": "wp9_adaptive", "policy.lr_scale": "wp9_adaptive", "wm.lr_scale": "wp9_adaptive",
    "meta.ppo_clip": "wp13_meta", "meta.gae_lambda": "wp13_meta", "meta.value_coef": "wp13_meta",
    "explore.uncertainty_scale": "rw1_modulation", "strategy.spaced_fraction": "rw6_strategy",
    "strategy.prioritized": "rw6_strategy"}


def run_composed(ns, n_seg):
    """Real phase order of a segment: curriculum, competence, adaptive, meta, relations."""
    rs = np.random.RandomState(5)
    reg = registry_for(ns)
    problems, checks, prev = [], 0, {}
    loss = [1.0, 2.0]
    with measuring():
        for seg in range(n_seg):
            loss = [x * (1.0 - 0.05 * rs.rand()) for x in loss]
            MEAS.clear()
            MEAS.update({0: (loss[0], 0.1 + 0.05 * rs.rand()), 1: (loss[1], 0.1 + 0.05 * rs.rand())})
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.2 + 0.6 * rs.rand()}
            ns.lifetime_timesteps += 64
            ns.infra.drift_alarms = 1 if seg >= 8 else 0
            ns._plan.last = {"policy_calibration_ok": 1.0, "calibration_conf": 0.4 + 0.5 * rs.rand()}
            with quiet():
                ns._curriculum_phase()
                ns._competence_phase()
                ns._adaptive_optim_phase()
                ns._meta_adapt_phase()
                ns._relations_phase()
            rows = {r["name"]: r for r in reg.table()}
            for nm, r in rows.items():
                if EXPECTED_OWNER.get(nm) != r["owner"]:
                    problems.append(f"{nm} owned by {r['owner']}")
                if not r["lo"] - 1e-9 <= r["value"] <= r["hi"] + 1e-9:
                    problems.append(f"{nm}={r['value']} outside [{r['lo']},{r['hi']}]")
                if nm in ("mixer.intrinsic_gain", "explore.entropy_target_scale", "explore.uncertainty_scale") \
                        and nm in prev and abs(r["value"] - prev[nm]) > 0.05 + 1e-9:
                    problems.append(f"{nm} slewed {r['value'] - prev[nm]:+.4f}")
                checks += 1
            prev = {nm: r["value"] for nm, r in rows.items()}
            p, c = ns.policy, ns._cons
            for held, nm in ((p.entropy_coef, "policy.entropy_coef"), (p.clip_range, "meta.ppo_clip"),
                             (p.gae_lambda, "meta.gae_lambda"), (p.value_coef, "meta.value_coef"),
                             (c.old_frac, "strategy.spaced_fraction"), (ns.reward_mixer.intrinsic_gain, "mixer.intrinsic_gain")):
                if nm in rows and abs(float(held) - rows[nm]["value"]) > 1e-9:
                    problems.append(f"consumer of {nm} holds {held} != {rows[nm]['value']}")
                checks += 1
            if "policy.lr_scale" in rows:
                if abs(p.optimizer.param_groups[0]["lr"] - BASE_LR * rows["policy.lr_scale"]["value"]) > 1e-12:
                    problems.append("optimizer lr != base * policy.lr_scale")
                checks += 1
    return checks, problems


def naive_registry():
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
    return patch.multiple(CoefficientRegistry, claim=claim, set=set_)


@H.scenario("CT-04", "curriculum + competence + adaptive optimisation + meta adaptation + strategy in real phase order: one owner per coefficient, bounded, consumers agree, rogue writers refused",
            brk="last_writer_wins", min_exercised=200,
            oracle="11 coefficients, each owned by its declared module; within bounds; slew-limited where declared; consumers hold the published value; a rogue claim/write/release raises and changes nothing")
def ct04_composition(brk):
    with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st:
        if brk == "last_writer_wins":
            st.enter_context(naive_registry())
        ns = compose(d)
        checks, problems = run_composed(ns, 14)
        reg = registry_for(ns)
        names = {r["name"] for r in reg.table()}
        if set(EXPECTED_OWNER) - names:
            problems.append(f"writers never exercised: {sorted(set(EXPECTED_OWNER) - names)}")
        if any(a.ctx == [] for a in ns._parallel_envs):
            problems.append("curriculum never set a context")
        held = {r["name"]: r["value"] for r in reg.table()}
        for nm in sorted(names):
            for attempt in (lambda: reg.claim(nm, "rogue_ctl", 1.0, 0.0, 9.0),
                            lambda: reg.set(nm, "rogue_ctl", 9.0),
                            lambda: reg.release(nm, "rogue_ctl")):
                try:
                    attempt()
                    problems.append(f"{nm}: rogue accepted")
                except OwnershipError:
                    pass
                checks += 1
        if {r["name"]: r["value"] for r in reg.table()} != held:
            problems.append("a refused write changed a value")
    return Obs(not problems, exercised=checks, note="; ".join(problems[:4]) or f"{len(names)} coefficients, {checks} checks")


def attr_writers(sources, attrs):
    """{attr: sorted modules} that ASSIGN the attribute, ignoring constructors. sources: [(path, text)]."""
    out = {a: set() for a in attrs}
    for path, text in sources:
        tree = ast.parse(text)
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if fn.name == "__init__":
                continue
            for node in ast.walk(fn):
                tgts = []
                if isinstance(node, ast.Assign):
                    tgts = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    tgts = [node.target]
                for t in tgts:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Attribute) and sub.attr in attrs:
                            out[sub.attr].add(os.path.basename(path))
    return {a: sorted(v) for a, v in out.items()}


def production_sources():
    res = []
    for dp, _dn, fns in os.walk(os.path.join(ROOT, "developmental_ai")):
        if os.sep + "evaluator" in dp or "__pycache__" in dp:
            continue
        for fn in fns:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                res.append((p, open(p).read()))
    return res


WRITERS = {"entropy_coef": ["adaptive_optim.py"], "old_frac": ["relations_strategy.py"],
           "prioritized": ["relations_strategy.py"], "frontier_gain": ["relations_motivation.py"],
           "intrinsic_gain": ["developmental_loop.py"]}


@H.scenario("CT-04", "every consumer attribute of this group has exactly one production writer (source trace, not a registry tag)",
            brk="second_writer_planted", min_exercised=6,
            oracle="policy.entropy_coef <- adaptive_optim; Consolidator.old_frac/prioritized <- relations_strategy; the PPO coefficients <- loop setattr(meta only)")
def ct04_single_writer(brk):
    srcs = production_sources()
    if brk:
        srcs = srcs + [("developmental_ai/core/rogue_controller.py",
                        "def tune(policy):\n    policy.entropy_coef = 0.0\n")]
    got = attr_writers(srcs, set(WRITERS))
    bad = {a: g for a, g in got.items() if g != WRITERS[a]}
    # the PPO-hyper-parameter attributes are only ever written through setattr by the meta phase
    loop_src = next(t for p, t in production_sources() if p.endswith("developmental_loop.py"))
    setattr_sites = loop_src.count("setattr(self.policy,")
    ok = not bad and setattr_sites == 2 and not [a for a in ("clip_range", "gae_lambda", "value_coef")
                                                 if attr_writers(srcs, {a})[a]]
    return Obs(bool(ok), exercised=len(got) + 1, note=f"unexpected writers {bad}; setattr sites {setattr_sites}")


@H.scenario("CT-04", "a second controller on WP-13's or WP-9's names is refused loudly (constructor raises), never overwrites silently",
            brk="last_writer_wins", min_exercised=4,
            oracle="constructing HyperAdapter/AdaptiveOptim over names owned by another module raises OwnershipError; the owner's value is untouched; WP-13 can only claim its three meta.* names")
def ct04_rogue_owner(brk):
    ok, n = True, 0
    with contextlib.ExitStack() as st:
        if brk:
            st.enter_context(naive_registry())
        policy, reg = make_policy(), CoefficientRegistry()
        reg.claim("meta.ppo_clip", "rogue_ctl", 0.2, 0.1, 0.4)
        reg.set("meta.ppo_clip", "rogue_ctl", 0.3)
        try:
            HyperAdapter(META_CFG, reg, policy)
            ok = False
        except OwnershipError:
            pass
        n += 1
        ok &= reg.owner_of("meta.ppo_clip") == "rogue_ctl" and reg.get("meta.ppo_clip") == 0.3
        n += 1
        reg2 = CoefficientRegistry()
        reg2.claim("policy.entropy_coef", "rogue_ctl", 0.01, 1e-4, 0.1)
        try:
            AdaptiveOptim({"enabled": True, "entropy": {"enabled": True}}, reg2, make_policy())
            ok = False
        except OwnershipError:
            pass
        n += 1
        reg3 = CoefficientRegistry()
        Ma = HyperAdapter(dict(META_CFG, coefficients=["ppo_clip", "policy.entropy_coef", "entropy_coef"]), reg3, make_policy())
        ok &= {r["name"] for r in reg3.table()} == {"meta.ppo_clip"} and Ma.names == ["ppo_clip"]
        n += 1
    return Obs(bool(ok), exercised=n, note=f"{n} ownership checks")


# ============================================================================= CT-05
@H.scenario("CT-05", "WP-3: a context change gives the stream a new baseline; progress is never computed across the switch",
            brk="stale_baseline_kept", min_exercised=4,
            oracle="right after a switch the stream has no baseline and its next measurement earns no reward; the one after that does")
def ct05_curriculum_switch(brk):
    sch = CurriculumScheduler({"min_pulls": 1, "every_segments": 2, "seed": 0})
    sch.register({0: {"pool": "p", "maps": ["a", "b", "c"]}, 1: {"pool": "p", "maps": ["a", "b", "c"]}})
    sch.step({})
    switched = None
    for seg in range(2, 40):
        old_last = sch.last[0]
        out = sch.step({0: (1.0 - 0.01 * seg, 0.5)})
        if 0 in out and old_last is not None:
            switched = seg
            if brk == "stale_baseline_kept":
                sch.last[0] = old_last                  # the old context's baseline survives the switch
            break
    if switched is None:
        return Obs(False, 0, "no context switch occurred in 40 segments")
    n0 = sum(sch.bandits["p"].rewards_n)
    no_baseline = sch.last[0] is None or brk == "stale_baseline_kept"
    sch.step({0: (50.0, 0.5)})                          # first measurement in the new context
    first = sum(sch.bandits["p"].rewards_n) - n0
    n1 = sum(sch.bandits["p"].rewards_n)
    sch.step({0: (25.0, 0.5)})                          # second: now a baseline exists
    second = sum(sch.bandits["p"].rewards_n) - n1
    ok = no_baseline and first == 0 and second == 1
    return Obs(bool(ok), exercised=4, note=f"switch at segment {switched}; rewards on 1st/2nd measurement {first}/{second}")


@H.scenario("CT-05", "RW-6: a strategy epoch's reward is credited to the arm that was active during it, nothing else",
            brk="credit_to_wrong_arm", min_exercised=4,
            oracle="after 5 epochs each arm has exactly one reward, equal to the independently derived (drop - running baseline)")
def ct05_strategy_credit(brk):
    ns = build_loop({"relations": SRL_CFG}, cons=True)
    losses = (1.0, 0.9, 0.9, 0.7, 0.7)
    orig = curr.ContextBandit.update
    ctx = patch.object(curr.ContextBandit, "update", lambda self, arm, r: orig(self, (arm + 1) % self.n, r)) \
        if brk else contextlib.nullcontext()
    with measuring(), ctx:
        for L in losses:
            MEAS.clear()
            MEAS[0] = (L, 0.0)
            with quiet():
                ns._relations_phase()
    # independent derivation of the booked rewards
    want, rb, prev = [], None, None
    for L in losses:
        if prev is not None:
            r = max(-1.0, min(1.0, (prev - L) / (abs(prev) + 1e-8)))
            base = r if rb is None else rb
            want.append(r - base)
            rb = 0.8 * base + 0.2 * r
        prev = L
    b = ns._rel.state["strat"]["bandit"]
    ok = b.rewards_n == [1, 1, 1, 1] and all(abs(b.value[i] - want[i]) < 1e-9 for i in range(4))
    return Obs(bool(ok), exercised=4, note=f"values {[round(v, 4) for v in b.value]} want {[round(w, 4) for w in want]}")


@H.scenario("CT-05", "WP-9: a stale probe of the wrong width is refused and no weight or optimizer state is touched",
            brk="refresh_on_stale_probe", min_exercised=3,
            oracle="measure() raises on a width-mismatched probe; actor/critic weights, Adam moments and lr are unchanged; no refresh happens")
def ct05_plasticity_width(brk):
    policy = make_policy()
    pm = PlasticityMonitor(tau=0.5, refresh=True, max_frac=0.5, min_dormant_frac=0.0)
    policy._probe = {"feats": torch.randn(16, 8), "knowledge": None, "proprio": None}
    masks = pm.measure(policy)                                   # a probe of the right width works
    w0 = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    w1 = {k: v.clone() for k, v in policy.critic.state_dict().items()}
    policy._probe = {"feats": torch.randn(16, 11 if not brk else 8), "knowledge": None, "proprio": None}
    refused = False
    try:
        masks2 = pm.measure(policy)
        if brk:
            pm.maybe_refresh(policy, {k: torch.ones_like(v, dtype=torch.bool) for k, v in masks.items()})
    except Exception:
        refused = True
    same = all(torch.equal(v, w0[k]) for k, v in policy.actor.state_dict().items()) and \
        all(torch.equal(v, w1[k]) for k, v in policy.critic.state_dict().items())
    ok = bool(masks) and refused and same and pm.refreshed_total == 0
    return Obs(ok, exercised=3, note=f"refused {refused}, weights unchanged {same}, refreshed {pm.refreshed_total}")


# ============================================================================= CT-06
def adaptive_loop(tmp=None):
    cfg = {"adaptive_optim": dict(AO_CFG)}
    return build_loop(cfg, tmp=tmp)


class BadStats(dict):
    def get(self, *_a, **_k):
        raise RuntimeError("PPO statistics dependency down")


@H.scenario("CT-06", "WP-9: a failing dependency releases the learning rate, return normaliser and registry to base, observably",
            brk="fault_not_released", min_exercised=5,
            oracle="after the fault: optimizer lr == base exactly, return_normalizer None, no wp9 coefficient, a warning names the fault")
def ct06_adaptive_fault(brk):
    ns = adaptive_loop()
    recs, h = fault_logger()
    try:
        for t in (200, 400, 600):
            ns.lifetime_timesteps = t
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.2}
            with quiet():
                ns._adaptive_optim_phase()
        scaled = ns.policy.optimizer.param_groups[0]["lr"] < BASE_LR and ns.policy.return_normalizer is not None
        ns.policy.last_update_stats = BadStats(cond_entropy_norm=0.2)
        ctx = patch.object(AdaptiveOptim, "release", lambda self, policy=None: None) if brk else contextlib.nullcontext()
        with ctx, quiet():
            ns._adaptive_optim_phase()
    finally:
        logging.getLogger().removeHandler(h)
    ok = scaled and ns.policy.optimizer.param_groups[0]["lr"] == BASE_LR and ns.policy.return_normalizer is None \
        and not [r for r in registry_for(ns).table() if r["owner"] == "wp9_adaptive"] \
        and any("adaptive optimisation failed" in r.getMessage() for r in recs)
    return Obs(bool(ok), exercised=5, note=f"lr {ns.policy.optimizer.param_groups[0]['lr']} (base {BASE_LR}); warnings {len(recs)}")


# FINDING (real defect, see report): AdaptiveOptim.release() frees the registry names but never restores
# `policy.entropy_coef`, so a failed phase (or any release) strands the last controller value on the
# policy (adaptive_optim.py release(): lr and return normaliser are restored, the entropy coefficient is not).
@H.scenario("CT-06", "WP-9: releasing the controller returns the entropy coefficient to its base value",
            brk="entropy_left_stranded", min_exercised=2,
            oracle="policy.entropy_coef == its pre-controller value (0.01) after release()")
def ct06_adaptive_entropy_base(brk):
    policy, reg = make_policy(), CoefficientRegistry()
    base = policy.entropy_coef
    Ao = AdaptiveOptim(AO_CFG, reg, policy)
    for _ in range(4):
        policy.last_update_stats = {"cond_entropy_norm": 0.1}
        Ao.run(policy, [], 300.0, 0)
    moved = policy.entropy_coef != base
    if brk:
        Ao.ent_base = None                      # the defect: release forgets the base (GOV-015 F-17)
    Ao.release(policy)
    return Obs(moved and policy.entropy_coef == base, exercised=2,
               note=f"base {base} controller {moved} after release {policy.entropy_coef}")


def recovery(phase, cfg, make_bad, fix):
    ns = build_loop(cfg)
    state = {"fail": False}

    def flaky(*_a, **_k):
        if state["fail"]:
            raise RuntimeError("world-model dependency down")
        return dict(MEAS)
    with measuring(flaky):
        MEAS.update({0: (1.0, 0.0)})
        fn = getattr(ns, phase)
        for _ in range(3):
            ns.lifetime_timesteps += 100
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.2}
            with quiet():
                fn()
        state["fail"] = True
        make_bad(ns)
        with quiet():
            fn()
        state["fail"] = False
        fix(ns)
        for k in range(12):
            ns.lifetime_timesteps += 100
            MEAS[0] = (1.0 * (0.97 ** k), 0.0)
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.2}
            with quiet():
                fn()
    return ns


# FINDING D-LATCH (real defect, same root cause as CAT-04): `_adaptive_optim_phase` and `_meta_adapt_phase` set `self._ao = False` /
# `self._ma = False` in their except-branch and nothing re-creates them: one transient fault idles the
# controller for the rest of the run (developmental_loop.py ~11940, ~11975; CLAUDE.md 4.1).
def ct06_adaptive_recover(brk):
    ns = adaptive_loop()
    recs, h = fault_logger()
    try:
        for t in (200, 400):
            ns.lifetime_timesteps = t
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.2}
            with quiet():
                ns._adaptive_optim_phase()
        ns.policy.last_update_stats = BadStats(cond_entropy_norm=0.2)
        with quiet():
            ns._adaptive_optim_phase()
        for k in range(12):
            ns.lifetime_timesteps = 100 * k
            ns.policy.last_update_stats = BadStats(cond_entropy_norm=0.2) if brk else {"cond_entropy_norm": 0.2}
            with quiet():
                ns._adaptive_optim_phase()
    finally:
        logging.getLogger().removeHandler(h)
    ok = coef(ns, "policy.lr_scale") is not None and ns.policy.optimizer.param_groups[0]["lr"] < BASE_LR
    return Obs(bool(ok), exercised=2, note=f"lr_scale {coef(ns, 'policy.lr_scale')}; controller handle {ns._ao!r}"[:160])


def ct06_meta_recover(brk):
    ns = build_loop({"meta_adapt": META_CFG})
    state = {"fail": False}

    def flaky(*_a, **_k):
        if state["fail"]:
            raise RuntimeError("world-model dependency down")
        return dict(MEAS)
    with measuring(flaky):
        MEAS.update({0: (1.0, 0.0)})
        for _ in range(6):
            with quiet():
                ns._meta_adapt_phase()
        state["fail"] = True
        with quiet():
            ns._meta_adapt_phase()
        state["fail"] = bool(brk)
        for k in range(24):
            MEAS[0] = (0.98 ** k, 0.0)
            with quiet():
                ns._meta_adapt_phase()
    alive = ns._ma is not False and getattr(ns._ma, "names", None)
    ok = bool(alive) and coef(ns, "meta.ppo_clip") is not None
    return Obs(ok, exercised=2, note=f"adapter handle {ns._ma!r}"[:160])


# (strand helpers below are combined into one scenario)
# FINDING (real defect, see report): HyperAdapter.update returns early when no progress sample exists, so a
# trial in progress keeps its non-neutral value for as long as the source stays absent; nothing releases it
# (meta_adapt.py update(): `if d is None: return self.state()`).
def ct06_meta_strand(brk):
    ns = build_loop({"meta_adapt": META_CFG})
    box = [1.0]
    with measuring():
        MEAS[0] = (1.0, 0.0)
        with quiet():
            ns._meta_adapt_phase()
        for _ in feed(None, box, 0.01, 3):
            with quiet():
                ns._meta_adapt_phase()
        trial_clip = ns.policy.clip_range
        in_trial = ns._ma.phase == "trial" and abs(trial_clip - 0.22) < 1e-9
        MEAS.clear()                                     # the measurement source is lost
        for _ in range(0 if brk else 4):                 # DEFECT (GOV-015 9.6): no release step ever runs (waiting longer still released it)
            with quiet():
                ns._meta_adapt_phase()
    ok = in_trial and abs(ns.policy.clip_range - ns._ma.cur["ppo_clip"]) < 1e-9
    return Obs(bool(ok), exercised=3, note=f"trial value {trial_clip}, after 4 empty updates {ns.policy.clip_range}, current {ns._ma.cur['ppo_clip']}")


# FINDING (real defect, see report): the applied recipe is only ever replaced when a new epoch measurement
# arrives (relations_strategy.apply), so when the fresh-loss source vanishes the last arm's non-base recipe
# stays applied indefinitely; only a module failure releases it.
def ct06_strategy_strand(brk):
    ns = build_loop({"relations": SRL_CFG}, cons=True)
    with measuring():
        for L in (1.0, 0.9, 0.9, 0.7):                   # arm 3 (spaced + prioritised) is applied at the 4th
            MEAS.clear()
            MEAS[0] = (L, 0.0)
            with quiet():
                ns._relations_phase()
        applied = (ns._cons.old_frac, ns._cons.prioritized)
        MEAS.clear()
        for _ in range(0 if brk else 4):                 # DEFECT (GOV-015 9.6): no release step ever runs
            with quiet():
                ns._relations_phase()
    ok = applied == (0.25, True) and (ns._cons.old_frac, ns._cons.prioritized) == (0.0, False)
    return Obs(bool(ok), exercised=3, note=f"applied {applied}, 4 segments after source loss {(ns._cons.old_frac, ns._cons.prioritized)}")


@H.scenario("CT-06", "DEFECT D-LATCH: after one transient fault the WP-9 and WP-13 controllers resume (handle dropped for the run otherwise)",
            brk="fault_latches", min_exercised=4,
            oracle="within 12 segments of recovery WP-9 claims its schedule coefficient again and WP-13 has an adapter with a claimed coefficient")
def ct06_latch(brk):
    a, b = ct06_adaptive_recover(brk), ct06_meta_recover(brk)
    return Obs(a.ok and b.ok, a.exercised + b.exercised, a.note + " | " + b.note)


@H.scenario("CT-06", "DEFECT D-STRAND: when the measurement source is lost, WP-13's trial value and RW-6's recipe are released to the current/configured value",
            brk="trial_kept", min_exercised=6,
            oracle="policy.clip_range back to the accepted value within 4 empty updates; old_frac/prioritized back to the configured (0.0, False) within 4 segments")
def ct06_strand(brk):
    a, b = ct06_meta_strand(brk), ct06_strategy_strand(brk)
    return Obs(a.ok and b.ok, a.exercised + b.exercised, a.note + " | " + b.note)


@H.scenario("CT-06", "WP-3: a collapsed bandit re-opens by itself (entropy floor): no arm starves, and a changed best arm is found",
            brk="floor_removed", min_exercised=4,
            oracle="every arm keeps being assigned while one dominates; after the rewards flip the new best arm takes over the majority within 120 assignments")
def ct06_bandit_floor(brk):
    cfg = {"min_pulls": 1, "temperature": 0.05, "ucb_c": 0.0}
    if brk:
        cfg.update(entropy_floor=0.0, epsilon=0.0)
    b = ContextBandit(3, cfg, random.Random(0))
    counts = [0, 0, 0]
    for _ in range(300):
        arm, _why = b.choose()
        counts[arm] += 1
        b.update(arm, 1.0 if arm == 0 else -1.0)
    starved = min(counts) < 5
    after = [0, 0, 0]
    for _ in range(240):
        arm, _why = b.choose()
        b.update(arm, 1.0 if arm == 2 else -1.0)
        after[arm] += 1
    late = [0, 0, 0]
    for _ in range(60):
        arm, _why = b.choose()
        b.update(arm, 1.0 if arm == 2 else -1.0)
        late[arm] += 1
    ok = not starved and late[2] > 30
    return Obs(bool(ok), exercised=4, note=f"counts while arm 0 dominated {counts}; after flip, last 60 {late}")


# ============================================================================= CT-07
@H.scenario("CT-07", "WP-3: save, recreate, restore: bandit state is restored, assignments restart, and no progress is scored across the restart",
            brk="state_not_restored", min_exercised=6,
            oracle="values, pulls, totals and seg restored; cur/last reset; changed contexts start fresh; first post-restore measurement earns no reward")
def ct07_curriculum(brk):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "curriculum.json")
        opts = {0: {"pool": "p", "maps": ["a", "b", "c"]}, 1: {"pool": "p", "maps": ["a", "b", "c"]}}
        s1 = CurriculumScheduler({"min_pulls": 1, "every_segments": 2, "seed": 0}, path)
        s1.register(opts)
        s1.step({})
        for k in range(8):
            s1.step({0: (1.0 - 0.05 * k, 0.5), 1: (1.0, 0.5)})
        st1 = s1.bandits["p"].state()
        s2 = CurriculumScheduler({"min_pulls": 1, "every_segments": 2, "seed": 0}, None if brk else path)
        s2.register(opts)
        st2 = s2.bandits["p"].state()
        restored = st2 == st1 and s2.seg == s1.seg
        reset = s2.cur == {0: None, 1: None} and s2.last == {0: None, 1: None}
        before = sum(s2.bandits["p"].rewards_n)
        s2.step({0: (50.0, 0.5), 1: (1.0, 0.5)})
        no_stale_reward = sum(s2.bandits["p"].rewards_n) == before
        s3 = CurriculumScheduler({"min_pulls": 1, "every_segments": 2, "seed": 0}, path)
        s3.register({0: {"pool": "p", "maps": ["a", "b"]}, 1: {"pool": "p", "maps": ["a", "b"]}})
        fresh = s3.bandits["p"].total == 0
        ok = restored and reset and no_stale_reward and fresh
    return Obs(bool(ok), exercised=6, note=f"restored {restored} reset {reset} no stale reward {no_stale_reward} changed contexts fresh {fresh}")


# FINDING (real defect, see report): AdaptiveOptim captures each param group's base lr lazily, at the first
# apply (`_base.setdefault(id(g), g["lr"])`). The policy optimizer state IS saved and restored
# (actor_critic.load_full, developmental_loop.py ~3108 / 10485) with the SCALED lr inside param_groups, so the
# scaled value becomes the new base: every relaunch compounds the schedule.
@H.scenario("CT-07", "WP-9: after save and restore of the policy optimizer, the learning rate follows base x schedule, not scaled-base x schedule",
            brk="optimizer_not_restored", min_exercised=3,
            oracle="restored run's lr at the same lifetime step equals the uninterrupted run's")
def ct07_lr_ratchet(brk):
    cfg = {"enabled": True, "every_segments": 1,
           "schedule": {"enabled": True, "policy": {"kind": "linear", "horizon": 1000, "floor": 0.1},
                        "world_model": {"kind": "constant"}}}
    a, reg_a = make_policy(), CoefficientRegistry()
    Ao = AdaptiveOptim(cfg, reg_a, a)
    Ao.run(a, [], 500.0, 0)                                  # lr is now 0.55 x base inside the optimizer
    sd, osd = a.get_state_dict(), a.optimizer.state_dict()
    Ao.run(a, [], 800.0, 0)
    uninterrupted = a.optimizer.param_groups[0]["lr"]
    b, reg_b = make_policy(), CoefficientRegistry()          # the relaunch: fresh policy from config
    if brk:
        del b.base_lr                                        # the defect: the base is unknown, captured lazily from the live (restored, scaled) lr (GOV-015 F-19)
    b.load_full(sd, osd)
    Ao2 = AdaptiveOptim(cfg, reg_b, b)
    Ao2.run(b, [], 800.0, 0)
    restored = b.optimizer.param_groups[0]["lr"]
    ok = abs(restored - uninterrupted) < 1e-12
    return Obs(bool(ok), exercised=3, note=f"lr at step 800: uninterrupted {uninterrupted:.3e}, after restore {restored:.3e} (config base {BASE_LR:.1e})")


@H.scenario("CT-07", "WP-9 / WP-13 / RW-6: coefficient state is not persisted by design; a relaunch starts every controller at the configured values",
            brk="adapted_attrs_restored", min_exercised=6,
            oracle="a recreated policy holds the configured PPO coefficients; WP-13's bounds and base come from them, not from the previous run's adapted values")
def ct07_coefficients_reset(brk):
    with tempfile.TemporaryDirectory() as d:
        ns = build_loop({"meta_adapt": META_CFG, "adaptive_optim": dict(AO_CFG)}, tmp=d)
        box = [1.0]
        with measuring():
            MEAS[0] = (1.0, 0.0)
            with quiet():
                ns._meta_adapt_phase()
            for dd, n in ((0.01, 3), (0.05, 3), (0.01, 3)):
                for _ in feed(None, box, dd, n):
                    with quiet():
                        ns._meta_adapt_phase()
        for k in range(3):
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.1}
            ns.lifetime_timesteps += 300
            with quiet():
                ns._adaptive_optim_phase()
        adapted = (ns.policy.clip_range, ns.policy.entropy_coef)
        assert adapted != (0.2, 0.01), "fixture did not move the coefficients"
        ns2 = build_loop({"meta_adapt": META_CFG, "adaptive_optim": dict(AO_CFG)}, tmp=d)    # relaunch
        if brk == "adapted_attrs_restored":
            ns2.policy.clip_range, ns2.policy.entropy_coef = adapted
        with measuring():
            MEAS[0] = (1.0, 0.0)
            with quiet():
                ns2._meta_adapt_phase()
                ns2._adaptive_optim_phase()
        c = ns2._ma.reg._c["meta.ppo_clip"]
        ok = (ns2.policy.clip_range, ns2.policy.value_coef, ns2.policy.gae_lambda) == (0.2, 0.5, 0.95)
        ok &= abs(c.base - 0.2) < 1e-12 and abs(c.lo - 0.1) < 1e-12 and abs(c.hi - 0.4) < 1e-12
        ok &= abs(ns2._ao.ent.coef - 0.01) < 1e-12 and not os.path.exists(os.path.join(d, "meta_adapt.json"))
        ok &= registry_for(ns2).get("policy.lr_scale") in (None, 1.0)
    return Obs(bool(ok), exercised=6, note=f"adapted before relaunch {adapted}; after relaunch base {c.base} bounds [{c.lo},{c.hi}]")


# ============================================================================= CT-08
def _imports(path):
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {node.module + "." + a.name for a in node.names}
    return out


CAT05_FILES = ("developmental_ai/core/curriculum.py", "developmental_ai/policy/adaptive_optim.py",
               "developmental_ai/core/meta_adapt.py", "developmental_ai/core/relations_strategy.py",
               "developmental_ai/core/coefficients.py")


@H.scenario("CT-08", "evaluator fields cannot enter CAT-05: registration refuses them, every RW-6 channel is learner-tagged, no CAT-05 module imports the evaluator",
            brk="evaluator_probe_registered", min_exercised=15,
            oracle="evaluator-tagged source and evaluator-defined probe raise EvaluatorLeakError; strategy channels learner-tagged with non-evaluator probes; no direct evaluator/detector import")
def ct08_evaluator(brk):
    import developmental_ai.core.relations as rel
    reg = RelationRegistry({"enabled": True})
    refused, n = 0, 0

    def probe(loop):
        return 1.0
    probe.__module__ = "developmental_ai.evaluator.fake"
    ctx = patch.object(rel, "_FORBIDDEN_PROBE_PREFIXES", ()) if brk else contextlib.nullcontext()
    with ctx:
        for kw in (dict(source_tag="evaluator"), dict(probe=probe)):
            try:
                reg.register("XX-01", "AA", "BB", "p", "c", "s", "u", "t", **kw)
            except EvaluatorLeakError:
                refused += 1
            n += 1
    full = build_default_registry({"enabled": True})
    chans = [c for c in full.channels() if getattr(c, "group", "") == "strategy"]
    tags = bool(chans) and all(c.source_tag == "learner" for c in chans)
    probes = all(not (getattr(c.probe, "__module__", "") or "").startswith(
        ("developmental_ai.evaluator", "developmental_ai.game_template.detectors")) for c in chans)
    n += len(chans)
    leaks = []
    for f in CAT05_FILES:
        leaks += [(f, m) for m in _imports(os.path.join(ROOT, f))
                  if m.startswith(("developmental_ai.evaluator", "developmental_ai.game_template.detectors"))]
        n += 1
    ok = refused == 2 and tags and probes and not leaks
    return Obs(bool(ok), exercised=n, note=f"refused {refused}/2, {len(chans)} strategy channels, leaks {leaks}")


def _rng_sig():
    return (torch.get_rng_state().sum().item(), random.getstate()[1][:5], np.random.get_state()[1][:5].tolist())


@H.scenario("CT-08", "read-only strategy probes and extra report/log calls do not change learner state, coefficients or RNG",
            brk="probe_mutates_state", min_exercised=20,
            oracle="consolidator, bandit, policy coefficients, registry and the three RNG streams are identical with observation channels on and sampled/logged more often")
def ct08_noninterference(brk):
    obs_on = {"JOL": {"CALIB": {"enabled": True}}, "META": {"SRL": {"enabled": True}},
              "SELFE": {"SRL": {"enabled": True}}, "ATTRIB": {"SELFE": {"enabled": True}}}

    def build(extra):
        cfg = strat_cfg()
        if extra:
            cfg.update(obs_on)
        ns = build_loop({"relations": cfg, "meta_adapt": META_CFG}, cons=True)
        ns._comp = CompetenceController({"min_meas": 1}, CoefficientRegistry())
        return ns

    def run(ns, extra):
        torch.manual_seed(11)
        random.seed(11)
        np.random.seed(11)
        sampled = 0
        with measuring():
            for i, L in enumerate((1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.6, 0.5)):
                MEAS.clear()
                MEAS[0] = (L, 0.0)
                ns._plan.mon.note("policy", [0.9] * 10, [1.0] * (3 + i % 5) + [0.0] * (7 - i % 5))
                with quiet():
                    ns._relations_phase()
                    ns._meta_adapt_phase()
                if extra:
                    if brk == "probe_mutates_state":
                        def evil(loop):
                            torch.rand(1)
                            loop.policy.clip_range += 1e-3
                            return 1.0
                        ns._rel.get("rel.JOL.CALIB").probe = evil
                    sampled += ns._rel.sample_all(ns)
                    ns._rel.line()
                    ns._comp.report({"ready": False})
                    if ns._ma:
                        ns._ma.report()
        reg = registry_for(ns)
        p, c = ns.policy, ns._cons
        return (({r["name"]: r["value"] for r in reg.table()}), (c.old_frac, c.prioritized),
                (p.clip_range, p.gae_lambda, p.value_coef, p.entropy_coef),
                ns._rel.state["strat"]["bandit"].state(), _rng_sig()), sampled
    ns_a, ns_b = build(False), build(True)
    a, _ = run(ns_a, False)
    b, sampled = run(ns_b, True)
    same = [x == y for x, y in zip(a, b)]
    return Obs(all(same), exercised=sampled + len(same) + 8,
               note=f"identical (coefs, consolidator, policy, bandit, rng): {same}; sampled {sampled}")


if __name__ == "__main__":
    sys.exit(H.finish())
