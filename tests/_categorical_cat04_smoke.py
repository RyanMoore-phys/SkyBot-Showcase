"""CS-1 within-group contracts for CAT-04, Motivation, exploration and skills
(GOV-013 section 2; ASS-012 section 3, CT-01..CT-08).

    PYTHONPATH=. ./venv/bin/python tests/_categorical_cat04_smoke.py

Members: WP-5 (information gain), WP-1 (effect discovery, BUILT-UNMEASURED: only its WIRING is
tested here, never that it finds useful skills), WP-8 (competence), RW-1 (modulation), RW-7
(motivation), RW-11 (read-only skill channels), the coefficient registry and the reward mixer.

Question: are intrinsic income, competence, skill discovery and control modulation consistent without
double payment or false confidence?

Everything runs on the REAL module objects (tiny dims, CPU, seeded). The real loop phase helpers
(`DevelopmentalAI._competence_phase`, `_relations_phase`, `_info_gain_step`, ...) are bound to a bare
namespace so they run in their real order against real controllers; the only stand-ins are numbers
(measured losses, calibration confidence, risk, bandit values, surprise/agency) injected into them.
No scripted skill, no task meaning, no evaluator code, no game engine.

THE DELIBERATE BREAKS are injected here, into the fixture or a monkeypatched copy, never into
production code. A scenario that FAILS in its normal run (not its break) is a real defect in the
code under test; see the FINDING comments.
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
from developmental_ai.core import relations_modulation as modulation  # noqa: E402
from developmental_ai.core import relations_motivation as motivation  # noqa: E402
from developmental_ai.core.coefficients import (  # noqa: E402
    CoefficientRegistry, OwnershipError, registry_for)
from developmental_ai.core.competence import CompetenceController  # noqa: E402
from developmental_ai.core.curriculum import ContextBandit  # noqa: E402
from developmental_ai.core.developmental_loop import DevelopmentalAI  # noqa: E402
from developmental_ai.core.relations import (  # noqa: E402
    EvaluatorLeakError, RelationRegistry, build_default_registry)
from developmental_ai.core.uncertainty import CalibrationMonitor  # noqa: E402
from developmental_ai.curiosity.action_surprise import (  # noqa: E402
    GuardClock, StepSurprise, prior_logits_all_actions)
from developmental_ai.curiosity.intrinsic_balance import InfoGainTerm, ShareCap  # noqa: E402
from developmental_ai.policy.actor_critic import RewardMixer, StandaloneActorCritic  # noqa: E402
from developmental_ai.policy.adaptive_optim import AdaptiveOptim  # noqa: E402
from developmental_ai.policy.consolidation import Consolidator  # noqa: E402
from developmental_ai.core.attention_memory import GatedWindowAttention  # noqa: E402
from developmental_ai.skill_bank.effect_discovery import (  # noqa: E402
    EffectDiscovery, EffectKeyClusterer, Probation)
from developmental_ai.skill_bank.skill_bank import SkillBank  # noqa: E402
from developmental_ai.world_model.rssm import RSSM  # noqa: E402

H = Harness("cat04", "Motivation, exploration and skills (WP-5, WP-1, WP-8, RW-1, RW-7, RW-11)")

# ----------------------------------------------------------------------------- fixtures
RSSM_KW = dict(obs_dim=8, action_dim=3, stochastic_size=4, stochastic_classes=5,
               deterministic_size=16, hidden_size=16)
PHASES = ("_step_surprise", "_info_gain_step", "_competence_phase", "_adaptive_optim_phase",
          "_meta_adapt_phase", "_relations_phase")
# GOV-015 F-09: the loop's re-arm helpers, bound with the phases that use them
REARM = ("_phase_failed", "_phase_rearm", "_phase_ok")
MEAS: dict = {}


def fake_measure(*_a, **_k):
    return dict(MEAS)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def measuring(fn=fake_measure):
    """The phases import measure_streams lazily, so patching the module attribute reaches them."""
    MEAS.clear()
    with patch("developmental_ai.core.curriculum.measure_streams", fn):
        yield


def make_rssm():
    torch.manual_seed(1)
    return RSSM(**RSSM_KW)


def observe(rssm, B=3, seed=0, steps=2):
    """Two real observe steps; returns the state BEFORE the last step, its action and model_info."""
    g = torch.Generator().manual_seed(seed)
    st = rssm.initial_state(B, torch.device("cpu"))
    prev = a = mi = None
    with torch.no_grad():
        for _ in range(steps):
            a = torch.nn.functional.one_hot(torch.randint(0, 3, (B,), generator=g), 3).float()
            o = torch.randn(B, 8, generator=g)
            prev = st
            st, mi = rssm.observe_step(prev, a, o)
    return prev, a, mi


def build_loop(cfg, rssm=None, policy=False, cons=False, relations_fixture=False):
    ns = SimpleNamespace(
        config=cfg, world_model=SimpleNamespace(rssm=rssm or make_rssm()),
        _wm_param_lock=threading.RLock(), replay_buffer=SimpleNamespace(streams=[]),
        device="cpu", reward_mixer=RewardMixer(intrinsic_weight=0.7, extrinsic_weight=0.3),
        is_discrete=True, lifetime_timesteps=0, infra=SimpleNamespace(drift_alarms=0),
        _wm_train=None)
    if policy:
        torch.manual_seed(2)
        ns.policy = StandaloneActorCritic(obs_dim=8, action_dim=3, hidden_dim=16,
                                          device=torch.device("cpu"))
    if cons or relations_fixture:
        ns._cons = Consolidator({})
    if relations_fixture:
        ns._am = SimpleNamespace(net=GatedWindowAttention(4), last={}, pushed0=True)
        ns._curr = SimpleNamespace(bandits={"p": ContextBandit(3, {}, random.Random(0))})
        ns._rk = SimpleNamespace(vec=[0.2])
        ns._plan = SimpleNamespace(last={}, mon=CalibrationMonitor())
        ns._ig_term = InfoGainTerm(0.5)
    for n in PHASES + REARM:
        setattr(ns, n, getattr(DevelopmentalAI, n).__get__(ns))
    return ns


def valid_plan(unc):
    return {"policy_calibration_ok": 1.0, "calibration_conf": 1.0 - float(unc)}


RCFG = {"enabled": True,
        "modulation": {"amp": 0.3, "max_step": 0.05, "min_n": 5, "ema": 0.9, "cons_sign": -1},
        "motivation": {"amp": 0.3, "max_step": 0.05, "min_n": 5, "ema": 0.9, "safe_amp": 0.5},
        "LC": {"NE": {"enabled": True}},
        "NE": {"AROUS": {"enabled": True}, "ATT": {"enabled": True}, "CONS": {"enabled": True}},
        "ZPD": {"MOT": {"enabled": True}}, "SAFE": {"CUR": {"enabled": True}}}


def rcfg(**cut):
    """RCFG with channels switched off by name, e.g. rcfg(NE_AROUS=False)."""
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in RCFG.items()}
    for k in ("LC", "NE", "ZPD", "SAFE"):
        c[k] = {a: dict(b) for a, b in RCFG[k].items()}
    for name, on in cut.items():
        f, t = name.split("_")
        c[f][t]["enabled"] = bool(on)
    return c


def rel_loop(cfg):
    ns = build_loop({"relations": cfg}, relations_fixture=True)
    ns._ig_base = float(ns._ig_term.weight)
    return ns


def rel_segment(ns, unc, val, risk):
    ns._plan.last = valid_plan(unc)
    ns._curr.bandits["p"].value = [val, 0.0, 0.0]
    ns._rk.vec = [risk]
    with quiet():
        ns._relations_phase()


def jitter(i):
    return 0.30 + 0.04 * (i % 2), 0.10 + 0.02 * (i % 2), 0.20 + 0.05 * (i % 2)


def coef(ns, name):
    return registry_for(ns).get(name)


# ============================================================================= CT-01
@H.scenario("CT-01", "WP-5: agency from the real RSSM reaches the intrinsic stream with the right sign and size",
            brk="consumer_call_omitted", min_exercised=6,
            oracle="prior over the taken action equals observe_step's prior; intrinsic gains weight*min(clip,max(0,agency-floor)) per row")
def ct01_info_gain(brk):
    rssm = make_rssm()
    B = 3
    prev, act, mi = observe(rssm, B)
    ns = build_loop({"curiosity": {"info_gain": {"weight": 0.5, "floor": 0.1, "clip": 2.0}}}, rssm=rssm)
    # the batched all-actions prior must contain, at the executed action, the prior observe_step used
    allp = prior_logits_all_actions(rssm, prev, 3)
    taken = act.argmax(-1)
    err = max(float((allp[b, int(taken[b])] - mi["prior_logits"][b]).abs().max()) for b in range(B))
    # producer perturbation: three agency levels (above clip, below floor, in range)
    sp = StepSurprise(rssm, prev, act, mi, [False] * B, True)     # no row near a life boundary (guard True = ignore; see CT-05)
    sp._agency = {0: 3.0, 1: 0.05, 2: 0.6}
    ns._as_cache = (mi, sp)
    intrinsic = torch.zeros(B)
    if brk != "consumer_call_omitted":
        with quiet():
            ns._info_gain_step(prev, act, mi, intrinsic, [False] * B, [False] * B, [{}] * B)
    want = [0.5 * 2.0, 0.0, 0.5 * 0.5]
    got = [float(x) for x in intrinsic]
    ok = err < 1e-5 and all(abs(g - w) < 1e-6 for g, w in zip(got, want)) and all(g >= 0 for g in got)
    return Obs(ok, exercised=B + B, note=f"mirror err {err:.1e}; paid {got} want {want}")


@H.scenario("CT-01", "WP-8: flat progress raises and rising progress lowers the mixer's intrinsic gain, slew-limited",
            brk="gain_not_applied", min_exercised=12,
            oracle="mixer.intrinsic_gain == registry value; mix(1,0) == w_int*gain; flat: up, rising: down; |step| <= max_step")
def ct01_competence(brk):
    cfg = {"competence": {"enabled": True, "every_segments": 1, "min_meas": 3, "signal_ema": 0.5,
                          "progress_ema": 0.5, "gain_amp": 0.3, "max_step": 0.05}}
    ns = build_loop(cfg)
    gains, checks, ok = [], 0, True
    with measuring():
        loss = 1.0
        for seg in range(18):
            if seg >= 8:
                loss *= 0.7                       # progress is now rising
            MEAS.update({0: (loss, 0.1), 1: (loss * 2.0, 0.1)})
            with quiet():
                ns._competence_phase()
            if brk == "gain_not_applied":
                ns.reward_mixer.intrinsic_gain = 1.0
            reg_g = coef(ns, "mixer.intrinsic_gain")
            if reg_g is None:                     # warm-up: nothing written, mixer neutral
                ok &= ns.reward_mixer.intrinsic_gain == 1.0
                continue
            mixed = ns.reward_mixer.mix(1.0, 0.0, update_stats=False)
            ok &= abs(mixed - 0.7 * reg_g) < 1e-12 and ns.reward_mixer.intrinsic_gain == reg_g
            if gains:
                ok &= abs(reg_g - gains[-1]) <= 0.05 + 1e-9
            gains.append(reg_g)
            checks += 1
    peak = max(gains) if gains else 1.0
    ok &= peak >= 1.2 and gains[-1] < peak - 0.1
    return Obs(bool(ok), exercised=checks, note=f"peak {peak:.3f} end {gains[-1] if gains else None}")


@H.scenario("CT-01", "RW-1 / RW-7: uncertainty, frontier and risk spikes move their consumers in the declared direction",
            brk="consumer_call_omitted", min_exercised=6,
            oracle="spike: explore scale up, attention temperature down, consolidation strength down, frontier gain up, info weight down; each |step| <= 0.05")
def ct01_relations(brk):
    ns = rel_loop(rcfg())
    for i in range(8):
        rel_segment(ns, *jitter(i))
    pre = {"ex": coef(ns, "explore.uncertainty_scale"), "tmp": ns._am.net.temperature,
           "cons": ns._cons.strength, "fr": ns.reward_mixer.frontier_gain, "ig": ns._ig_term.weight}
    if brk == "consumer_call_omitted":
        ns._plan.last = valid_plan(0.95)         # inputs spike, but the wiring step never runs
    else:
        rel_segment(ns, 0.95, 5.0, 0.95)
    post = {"ex": coef(ns, "explore.uncertainty_scale"), "tmp": ns._am.net.temperature,
            "cons": ns._cons.strength, "fr": ns.reward_mixer.frontier_gain, "ig": ns._ig_term.weight}
    d = {k: post[k] - pre[k] for k in pre}
    eps = 1e-9
    ok = (0 < d["ex"] <= 0.05 + eps and -0.05 - eps <= d["tmp"] < 0 and -0.05 - eps <= d["cons"] < 0
          and 0 < d["fr"] <= 0.05 + eps and d["ig"] < 0
          and abs(post["ig"] / ns._ig_base - pre["ig"] / ns._ig_base) <= 0.05 + eps
          and post["ig"] > 0)
    # the registry and the consumer must agree (the consumer's own state, not a counter)
    ok &= abs(coef(ns, "attn.temperature_scale") - post["tmp"]) < 1e-12
    ok &= abs(coef(ns, "consol.strength_scale") - post["cons"]) < 1e-12
    return Obs(bool(ok), exercised=6, note="deltas " + " ".join(f"{k}={v:+.4f}" for k, v in d.items()))


# ---- WP-1: the trigger -> effect key -> probation chain
EFF_CFG = dict(effect_delay=2, warmup=5, z_thr=2.0, s_floor=0.5, agency_frac=0.25, tau=0.8,
               min_change=0.05, probation_K=3, probation_min_successes=2, probation_alpha=0.05,
               min_base_steps=10, backoff_start=2, exemplars=2)
V = [torch.eye(6)[i] for i in range(4)]            # frozen-encoder view embeddings (numbers only)
SID = "ach_00_effect_0"                             # a skill NAMED by the minting convention, bound to effect_0


class World:
    """One real RSSM step reused as the template of every StepSurprise; only the numbers are injected."""

    def __init__(self, B=1):
        self.B = B
        self.rssm = make_rssm()
        self.prev, self.act, self.mi = observe(self.rssm, B)

    def sp(self, s, ag, guard=None):
        # guard True = the row is within k steps of a life boundary (ignored). The neutral default is
        # "no row near a boundary". It was [True] * B while the consumers read the flag backwards
        # (ASS-014 F-13); corrected with the fix (GOV-014 B1, section 10 deviation).
        sp = StepSurprise(self.rssm, self.prev, self.act, self.mi, guard if guard is not None else [False] * self.B, True)
        sp.s = torch.tensor([float(x) for x in s])
        sp._agency = {i: float(a) for i, a in enumerate(ag)}
        return sp

    def step(self, D, t, s, ag, keys, running=None, ended=None, guard=None, nxt=None):
        B = self.B
        obs = torch.zeros(B, 4)
        return D.step(self.sp(s, ag, guard), keys, obs, obs if nxt is None else nxt,
                      ended or [False] * B, t, running or [None] * B)


def quiet_steps(W, D, t0, n, view=0):
    for t in range(t0, t0 + n):
        W.step(D, t, [0.2], [0.0], [V[view]])
    return t0 + n


def invoke(W, D, t0, effect_view=1, sid=SID, hit=True):
    """One skill invocation of 3 steps with a spike (surprise+agency) on its 2nd step; the view changes
    `effect_delay` steps later. Returns (next t, resolved list)."""
    res = []
    for k in range(7):
        t = t0 + k
        spike = k == 1
        view = effect_view if (hit and 3 <= k < 4) else 0
        out = W.step(D, t, [4.0 if spike else 0.2], [2.0 if spike else 0.0], [V[view]],
                     running=[sid if k < 3 else None])
        res += out["resolved"]
        res += [("verdict", v) for v in out["verdicts"]]
    return t0 + 7, res


def centroid_cos(D, key, a, b):
    d = (V[b] - V[a])
    d = d / d.norm()
    return float(D.keys.centroid(key) @ d)


@H.scenario("CT-01", "WP-1 wiring: surprise x agency -> effect key -> probation hit -> keep verdict",
            brk="view_key_feed_missing", min_exercised=5,
            oracle="spike with agency fires; key centroid points along the real view change; same effect joins the same key; 3/3 hits keep; agency-less spike fires nothing")
def ct01_effect_chain(brk):
    W, D = World(), EffectDiscovery(EFF_CFG)
    t = quiet_steps(W, D, 0, 14)
    fires0 = D.stats["fires"]
    # negative control: a surprise spike the action does NOT explain must not fire
    W.step(D, t, [4.0], [0.1], [V[0]])
    t += 1
    nofire = D.stats["fires"] == fires0
    keys_seen, resolved_all, verdicts = [], [], []
    for r in range(3):
        if brk == "view_key_feed_missing":
            # the producer->consumer link is cut: the view embedding never reaches the clusterer
            for k in range(7):
                out = W.step(D, t + k, [4.0 if k == 1 else 0.2], [2.0 if k == 1 else 0.0], [None],
                             running=[SID if k < 3 else None])
                resolved_all += out["resolved"]
            t += 7
            continue
        t, res = invoke(W, D, t)
        resolved_all += [x for x in res if x[0] != "verdict"]
        verdicts += [x[1] for x in res if x[0] == "verdict"]
    names = {k for (_, k, _) in resolved_all}
    ok = nofire and names == {"effect_0"} and len(resolved_all) == 3
    ok &= bool(len(verdicts) == 1 and verdicts[0]["verdict"] == "keep" and verdicts[0]["succ"] == 3)
    if "effect_0" in D.keys.keys:
        ok &= centroid_cos(D, "effect_0", 0, 1) > 0.99 and D.keys.keys["effect_0"]["n"] == 3
        cue = resolved_all[0][2]
        ok &= cue.shape == (192,) and abs(float(np.linalg.norm(cue)) - 1.0) < 1e-4
    else:
        ok = False
    return Obs(bool(ok), exercised=len(resolved_all) + len(verdicts) + int(nofire) + 1,
               note=f"keys {sorted(D.keys.keys)} resolved {len(resolved_all)} verdicts {[v['verdict'] for v in verdicts]}")


# ============================================================================= CT-02
def run_relations(cfg):
    ns = rel_loop(cfg)
    for i in range(8):
        rel_segment(ns, *jitter(i))
    for _ in range(3):
        rel_segment(ns, 0.95, 5.0, 0.95)
    reg = registry_for(ns)
    snap = {r["name"]: r["value"] for r in reg.table()}
    snap.update(_tmp=ns._am.net.temperature, _cons=ns._cons.strength,
                _fr=ns.reward_mixer.frontier_gain, _ig=ns._ig_term.weight)
    return snap, ns


CUTS = {  # switch -> (registry name, consumer key, neutral)
    "NE_AROUS": ("explore.uncertainty_scale", None, None),
    "NE_ATT": ("attn.temperature_scale", "_tmp", 1.0),
    "NE_CONS": ("consol.strength_scale", "_cons", 1.0),
    "ZPD_MOT": ("objective.frontier_gain", "_fr", 1.0),
    "SAFE_CUR": ("info.weight_scale", "_ig", 0.5),
}


@H.scenario("CT-02", "RW-1/RW-7: cutting each channel, then the whole group, restores neutral and leaves the rest identical",
            brk="cut_ignored", min_exercised=14,
            oracle="cut channel: coefficient absent and consumer exactly neutral; every other value bit-identical to the all-on run")
def ct02_cuts(brk):
    full, _ = run_relations(rcfg())
    ok, n = True, 0
    for sw, (name, ckey, neutral) in CUTS.items():
        cfg = rcfg(**{sw: brk == "cut_ignored"})        # broken: the switch does not cut
        snap, _ns = run_relations(cfg)
        ok &= name not in snap
        n += 1
        if ckey is not None:
            ok &= snap[ckey] == neutral
            n += 1
        for k, v in full.items():
            if k != name and k != ckey:
                ok &= snap.get(k) == v
                n += 1
    for sw in ("LC_NE",):          # the gain source cut: all three modulation outputs sit exactly at neutral
        snap, _ns = run_relations(rcfg(**{sw: brk == "cut_ignored"}))
        for nm in ("explore.uncertainty_scale", "attn.temperature_scale", "consol.strength_scale"):
            ok &= snap.get(nm, 1.0) == 1.0
            n += 1
    off, _ns = run_relations(dict(RCFG, enabled=False))        # the group: master off
    ok &= not [k for k in off if not k.startswith("_")] and off["_tmp"] == 1.0 and off["_cons"] == 1.0 \
        and off["_fr"] == 1.0 and off["_ig"] == 0.5
    n += 1
    return Obs(bool(ok), exercised=n, note=f"{n} neutral/parity comparisons")


@H.scenario("CT-02", "WP-5 / WP-8 switched off: intrinsic stream, mixer and registry are bit-identical to never having them",
            brk="off_switch_ignored", min_exercised=6,
            oracle="weight 0: no term, intrinsic tensor equal; competence disabled: nothing claimed, mixer.mix unchanged")
def ct02_off(brk):
    rssm = make_rssm()
    prev, act, mi = observe(rssm, 3)
    w = 0.5 if brk == "off_switch_ignored" else 0.0
    ns = build_loop({"curiosity": {"info_gain": {"weight": w, "floor": -100.0}},
                     "competence": {"enabled": brk == "off_switch_ignored", "every_segments": 1, "min_meas": 1}},
                    rssm=rssm)
    base = torch.tensor([0.1, -0.2, 0.3])
    intrinsic = base.clone()
    ref = RewardMixer(intrinsic_weight=0.7, extrinsic_weight=0.3)
    with measuring():
        MEAS.update({0: (1.0, 0.1)})
        with quiet():
            for _ in range(3):
                ns._info_gain_step(prev, act, mi, intrinsic, [False] * 3, [False] * 3, [{}] * 3)
                ns._competence_phase()
    ok = torch.equal(intrinsic, base) and getattr(ns, "_ig_term", None) is False
    ok &= getattr(ns, "_comp", None) is False and registry_for(ns).table() == []
    ok &= all(ns.reward_mixer.mix(i, e, update_stats=False) == ref.mix(i, e, update_stats=False)
              for i, e in ((1.0, 0.0), (0.3, 2.0), (-1.0, 0.5)))
    return Obs(bool(ok), exercised=6, note=f"intrinsic {intrinsic.tolist()} comp {getattr(ns, '_comp', None)}")


# ============================================================================= CT-03
@H.scenario("CT-03", "RW-1: missing, NaN, inf, out-of-range or unflagged calibration gives no sample and no state update",
            brk="invalid_accepted", min_exercised=8,
            oracle="ZTrack.n stays 0 and coefficients neutral for every invalid source; a legitimate constant source is a sample (n grows, scale stays 1.0)")
def ct03_modulation(brk):
    inf, nan = float("inf"), float("nan")
    cases = [None, {}, {"calibration_conf": 0.5}, {"calibration_conf": 0.5, "policy_calibration_ok": 0.0},
             *({"calibration_conf": v, "policy_calibration_ok": 1.0} for v in (nan, inf, -0.1, 1.1, None))]

    def naive(loop):                       # the broken producer: trusts the confidence field blindly
        last = getattr(getattr(loop, "_plan", None), "last", None) or {}
        c = last.get("calibration_conf")
        return None if c is None else 1.0 - float(c)

    ok, n = True, 0
    with contextlib.ExitStack() as st:
        if brk == "invalid_accepted":
            st.enter_context(patch.object(modulation, "policy_uncertainty", naive))
        for last in cases:
            ns = rel_loop(rcfg())
            ns._plan.last = last
            with quiet():
                ns._relations_phase()
            z = ns._rel.state["mod"]["z"]
            ok &= z.n == 0 and not math.isnan(z.m) and coef(ns, "explore.uncertainty_scale") == 1.0
            n += 1
        ns = rel_loop(rcfg())              # a legitimate constant source is NOT absence
        for _ in range(7):
            rel_segment(ns, 0.4, 0.1, 0.2)
        z = ns._rel.state["mod"]["z"]
        ok &= z.n == 7 and coef(ns, "explore.uncertainty_scale") == 1.0
        n += 1
        ns._plan.last = valid_plan(0.4)    # recovery from invalid back to valid
    return Obs(bool(ok), exercised=n, note=f"{n} source cases")


# FINDING (real defect, see report): ShareCap.limit stores abs(rest) without a finiteness check, so one
# NaN/inf in the stream's other income poisons the per-stream average for ever and the payment itself.
@H.scenario("CT-03", "WP-5: a NaN or infinite income on a stream cannot poison the share cap or the payment",
            brk="nan_poisons_cap_state", min_exercised=4,
            oracle="payments stay finite and non-negative, and later valid steps are capped normally again")
@reintroduce_nonfinite("developmental_ai.curiosity.intrinsic_balance")     # GOV-015 9.6: the break must remove the guard
def ct03_sharecap(brk):
    term = InfoGainTerm(weight=0.5, floor=-100.0, clip=1.0, cap_share=0.5, cap_floor=0.002)
    W = World(B=2)
    paid, ok, n = [], True, 0
    seq = [(0.1, 0.1), (float("nan"), 0.1), (0.1, 0.1), (float("inf"), 0.1), (0.1, 0.1)]
    if brk:
        seq = seq + [(float("nan"), 0.1)] * 2
    for r0, r1 in seq:
        p = term.pay(W.sp([0.2, 0.2], [1.0, 1.0]), [r0, r1])
        paid.append(p)
        finite = all(math.isfinite(x) and x >= 0.0 for x in p)
        ok &= finite
        n += 1
    ok &= all(math.isfinite(v) for v in term.cap._rest.values())
    # after the faulty income the cap must again bind: payment <= rest*share/(1-share)+floor (+ slack)
    last = term.pay(W.sp([0.2, 0.2], [1.0, 1.0]), [0.1, 0.1])
    ok &= all(0.0 <= x <= 0.1 + 0.002 + 1e-6 for x in last)
    return Obs(bool(ok), exercised=n + 1, note=f"payments {[[round(x, 3) if math.isfinite(x) else x for x in p] for p in paid]}")


# FINDING (real defect, see report): CompetenceController.update takes a NaN fresh loss into loss_ema,
# the history and the self-model features without checking, and nothing ever clears it.
@H.scenario("CT-03", "WP-8: a non-finite fresh-data loss is refused and leaves the tracker's state untouched",
            brk="nan_loss_accepted", min_exercised=4,
            oracle="loss_ema, history and gain stay finite and equal to their pre-fault values; valid input afterwards still moves the gain")
@reintroduce_nonfinite("developmental_ai.core.competence")
def ct03_competence(brk):
    reg = CoefficientRegistry()
    C = CompetenceController({"min_meas": 3, "signal_ema": 0.5, "progress_ema": 0.5}, reg)
    for _ in range(5):
        C.update({0: (1.0, 0.1), 1: (2.0, 0.1)})
    pre = (C.ctx[0].loss_ema, len(C.ctx[0].hist), C.ctx[0].n, reg.get("mixer.intrinsic_gain"))
    bad = float("nan") if brk == "nan_loss_accepted" else float("inf")
    for v in (float("nan"), float("inf")):
        C.update({0: (v, 0.1), 1: (2.0, 0.1)})
    s = C.ctx[0]
    finite = math.isfinite(s.loss_ema) and all(math.isfinite(h) for h in s.hist) and math.isfinite(s.lp) \
        and math.isfinite(C.flat)
    same = (s.loss_ema, len(s.hist), s.n) == pre[:3]
    g = reg.get("mixer.intrinsic_gain")
    ok = finite and same and g is not None and math.isfinite(g)
    return Obs(bool(ok), exercised=4, note=f"loss_ema {s.loss_ema} hist {len(s.hist)} (was {pre[1]}) flat {C.flat}")


# FINDING (real defect, see report): EffectKeyClusterer.assign only tests `mag < min_change`, which is
# False for NaN, so a NaN view change mints a NaN-centroid key (and a NaN cue goes to the goal store).
@H.scenario("CT-03", "WP-1: a missing, NaN or infinite view embedding mints no effect key; an unchanged view is a counted zero",
            brk="nan_embedding_keyed", min_exercised=5,
            oracle="no key and no resolved effect for None/NaN/inf embeddings; unchanged view counted as 'nochange', not absent")
@reintroduce_nonfinite("developmental_ai.skill_bank.effect_discovery")
def ct03_effect(brk):
    W, D = World(), EffectDiscovery(EFF_CFG)
    t = quiet_steps(W, D, 0, 14)
    bad_views = [None, torch.full((6,), float("nan")), torch.full((6,), float("inf"))]
    bad_views = bad_views + ([torch.full((6,), float("nan"))] if brk else [])
    resolved = []
    for bv in bad_views:
        for k in range(5):
            out = W.step(D, t, [4.0 if k == 1 else 0.2], [2.0 if k == 1 else 0.0],
                         [bv if k >= 3 else V[0]])
            resolved += out["resolved"]
            t += 1
        t = quiet_steps(W, D, t, 3)
    n0 = D.keys.stats["nochange"]
    for k in range(5):                            # same view again: a legitimate "nothing changed"
        W.step(D, t + k, [4.0 if k == 1 else 0.2], [2.0 if k == 1 else 0.0], [V[0]])
    ok = not D.keys.keys and not resolved and D.keys.stats["nochange"] > n0
    if D.keys.keys:                               # a minted key must at least be finite
        ok = False
    return Obs(bool(ok), exercised=len(bad_views) + 2,
               note=f"keys {list(D.keys.keys)} resolved {len(resolved)} nochange {D.keys.stats['nochange']}")


@H.scenario("CT-03", "RW-11: no skill bank is an absent sample, an empty bank is a legitimate zero",
            brk="absence_read_as_zero", min_exercised=4,
            oracle="OU-03/OU-04 skipped with no bank; sampled as 0.0 with an empty bank; no write to the bank")
def ct03_skills(brk):
    cfg = {"enabled": True, "PROC": {"SKILL": {"enabled": True}}, "AUTO": {"SKILL": {"enabled": True}}}
    reg = build_default_registry(cfg)
    ns = SimpleNamespace()
    got_absent = reg.sample_all(ns)
    skipped = reg.get("rel.PROC.SKILL").skipped
    if brk == "absence_read_as_zero":
        ns.skill_bank = None                 # a probe that coerced absence to 0 would count here
        from developmental_ai.core import relations_skills as rs
        with patch.object(rs, "_summary", lambda loop: {"falling_skill_fraction": 0.0, "bound_option_count": 0.0}):
            got_absent = reg.sample_all(ns)
    with tempfile.TemporaryDirectory() as d:
        ns2 = SimpleNamespace(skill_bank=SkillBank(storage_dir=d), option_executor=None)
        before = dict(ns2.skill_bank.skills)
        reg2 = build_default_registry(cfg)
        got_zero = reg2.sample_all(ns2)
        reg2.end_segment()
        ch = reg2.get("rel.PROC.SKILL")
        ok = got_absent == 0 and skipped >= 1 and got_zero == 2 and ch.n == 0 and ch.segments == 1 \
            and reg2.last_segment["channels"]["rel.PROC.SKILL"]["last"] == 0.0 \
            and dict(ns2.skill_bank.skills) == before
    return Obs(bool(ok), exercised=4, note=f"absent sampled {got_absent} skipped {skipped}; empty-bank sampled {got_zero}")


# ============================================================================= CT-04
def compose(policy=True):
    cfg = {
        "curiosity": {"info_gain": {"weight": 0.4, "floor": 0.0, "clip": 2.0}},
        "competence": {"enabled": True, "every_segments": 1, "min_meas": 3, "signal_ema": 0.5,
                       "progress_ema": 0.5, "modulate_exploration": True, "gain_amp": 0.3,
                       "entropy_amp": 0.3, "max_step": 0.05},
        "adaptive_optim": {"enabled": True, "every_segments": 1,
                           "schedule": {"enabled": True, "policy": {"kind": "linear", "horizon": 1000, "floor": 0.1},
                                        "world_model": {"kind": "constant"}},
                           "entropy": {"enabled": True, "target": 0.5, "eta": 0.5},
                           "return_norm": {"enabled": True},
                           "drift_response": {"enabled": True, "hold_segments": 4}},
        "meta_adapt": {"enabled": True, "every_segments": 1, "epoch_len": 2, "step": 0.1, "range": 2.0,
                       "accept_z": 0.0, "min_noise_n": 1,
                       "coefficients": ["ppo_clip", "gae_lambda", "value_coef"]},
        "relations": dict(rcfg(), SRL={"STRAT_ENTRY": {"enabled": True}},
                          STRAT_ENTRY={"SP": {"enabled": True}, "INTL": {"enabled": True}},
                          CALIB={"SRL": {"enabled": True}},
                          strategy={"epoch_len": 2, "min_pulls": 1, "seed": 0}),
    }
    ns = build_loop(cfg, policy=True, relations_fixture=True)
    ns._rk = SimpleNamespace(vec=[0.2])
    return ns


EXPECTED_OWNER = {
    "mixer.intrinsic_gain": "wp8_competence", "explore.entropy_target_scale": "wp8_competence",
    "policy.entropy_coef": "wp9_adaptive", "policy.lr_scale": "wp9_adaptive", "wm.lr_scale": "wp9_adaptive",
    "meta.ppo_clip": "wp13_meta", "meta.gae_lambda": "wp13_meta", "meta.value_coef": "wp13_meta",
    "explore.uncertainty_scale": "rw1_modulation", "attn.temperature_scale": "rw1_modulation",
    "consol.strength_scale": "rw1_modulation", "objective.frontier_gain": "rw7_motivation",
    "info.weight_scale": "rw7_motivation", "strategy.spaced_fraction": "rw6_strategy",
    "strategy.prioritized": "rw6_strategy"}
SLEWED = {"mixer.intrinsic_gain": 0.05, "explore.entropy_target_scale": 0.05, "explore.uncertainty_scale": 0.05,
          "attn.temperature_scale": 0.05, "consol.strength_scale": 0.05, "objective.frontier_gain": 0.05,
          "info.weight_scale": 0.05}


def run_composed(ns, n_seg, check=True):
    """The real phase order of a segment: competence, adaptive optimisation, meta adaptation, relations.
    Returns (checks verified, problems)."""
    rs = np.random.RandomState(3)
    reg = registry_for(ns)
    prev, problems, checks = {}, [], 0
    loss = [1.0, 2.0]
    with measuring():
        for seg in range(n_seg):
            loss = [x * (1.0 - 0.04 * rs.rand()) for x in loss]
            MEAS.clear()
            MEAS.update({0: (loss[0], 0.1 + 0.05 * rs.rand()), 1: (loss[1], 0.1 + 0.05 * rs.rand())})
            ns.policy.last_update_stats = {"cond_entropy_norm": 0.2 + 0.6 * rs.rand()}
            ns.lifetime_timesteps += 64
            ns.infra.drift_alarms = 1 if seg >= 9 else 0
            ns._plan.last = valid_plan(0.3 + 0.6 * rs.rand())
            ns._curr.bandits["p"].value = [0.1 + rs.rand(), 0.0, 0.0]
            ns._rk.vec = [0.1 + rs.rand()]
            with quiet():
                ns._competence_phase()
                ns._adaptive_optim_phase()
                ns._meta_adapt_phase()
                ns._relations_phase()
            if not check:
                continue
            rows = {r["name"]: r for r in reg.table()}
            for nm, r in rows.items():
                if EXPECTED_OWNER.get(nm) != r["owner"]:
                    problems.append(f"{nm} owned by {r['owner']}")
                if not r["lo"] - 1e-9 <= r["value"] <= r["hi"] + 1e-9:
                    problems.append(f"{nm}={r['value']} outside [{r['lo']},{r['hi']}]")
                if nm in SLEWED and nm in prev and abs(r["value"] - prev[nm]) > SLEWED[nm] + 1e-9:
                    problems.append(f"{nm} slewed {r['value'] - prev[nm]:+.4f}")
                checks += 1
            prev = {nm: r["value"] for nm, r in rows.items()}
            # every consumer holds exactly what its owner published
            m, p, c = ns.reward_mixer, ns.policy, ns._cons
            pairs = [(m.intrinsic_gain, "mixer.intrinsic_gain"), (m.frontier_gain, "objective.frontier_gain"),
                     (p.entropy_coef, "policy.entropy_coef"), (p.clip_range, "meta.ppo_clip"),
                     (p.gae_lambda, "meta.gae_lambda"), (p.value_coef, "meta.value_coef"),
                     (c.old_frac, "strategy.spaced_fraction"), (c.strength, "consol.strength_scale"),
                     (ns._am.net.temperature, "attn.temperature_scale")]
            for held, nm in pairs:
                if nm in rows and abs(float(held) - rows[nm]["value"]) > 1e-9:
                    problems.append(f"consumer of {nm} holds {held} != {rows[nm]['value']}")
                checks += 1
            if "policy.lr_scale" in rows:
                lr = p.optimizer.param_groups[0]["lr"]
                if abs(lr - 3e-4 * rows["policy.lr_scale"]["value"]) > 1e-12:
                    problems.append(f"policy lr {lr} vs scale {rows['policy.lr_scale']['value']}")
                checks += 1
    return checks, problems


def naive_registry():
    """The wiring defect: a registry that lets the last writer win."""
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


@H.scenario("CT-04", "all enabled coefficient writers in real phase order: one owner each, bounded, slew-limited, consumers agree; a rogue second writer is refused",
            brk="last_writer_wins", min_exercised=300,
            oracle="15 coefficients each owned by exactly the declared module; values within bounds; a rogue claim/write/release raises OwnershipError and changes nothing")
def ct04_composition(brk):
    with contextlib.ExitStack() as st:
        if brk == "last_writer_wins":
            st.enter_context(naive_registry())
        ns = compose()
        checks, problems = run_composed(ns, 16)
        reg = registry_for(ns)
        names = {r["name"] for r in reg.table()}
        missing = set(EXPECTED_OWNER) - names
        if missing:
            problems.append(f"writers never exercised: {sorted(missing)}")
        refused0, held = len(reg.refused), {r["name"]: r["value"] for r in reg.table()}
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
        after = {r["name"]: r["value"] for r in reg.table()}
        if after != held:
            problems.append("a refused write changed a value")
        if not brk and len(reg.refused) < refused0 + 2 * len(names):
            problems.append("refusals not recorded")
    return Obs(not problems, exercised=checks, note="; ".join(problems[:4]) or f"{len(names)} coefficients, {checks} checks")


@H.scenario("CT-04", "no duplicate reward booking: gains multiply the intrinsic term once, never the extrinsic, never the weight; information gain is booked once",
            brk="gain_applied_twice", min_exercised=9,
            oracle="mix(i,e) == w_int*gain_wp8*gain_rw7*i + w_ext*e with the INITIAL weights; info gain adds exactly weight*min(clip,agency-floor) once")
def ct04_booking(brk):
    ns = compose()
    run_composed(ns, 12, check=False)
    m = ns.reward_mixer
    if brk == "gain_applied_twice":
        m.intrinsic_weight *= m.intrinsic_gain         # a second booking of the same gain
    g, f = coef(ns, "mixer.intrinsic_gain"), coef(ns, "objective.frontier_gain")
    ok = g is not None and f is not None and (g != 1.0 or f != 1.0)
    n = 0
    for i, e in ((1.0, 0.0), (0.0, 1.0), (0.5, -2.0), (-0.4, 3.0), (2.0, 2.0)):
        want = 0.7 * g * f * i + 0.3 * e
        ok &= abs(m.mix(i, e, update_stats=False) - want) < 1e-12
        n += 1
    ok &= m.intrinsic_weight == 0.7 and m.extrinsic_weight == 0.3
    mult = motivation.intrinsic_multiplier(ns)          # the MO-01 channel reports the same product, once
    ok &= abs(mult - 0.7 * g * f) < 1e-12
    n += 2
    # information gain: exactly one booking per step
    rssm = ns.world_model.rssm
    prev, act, mi = observe(rssm, 2, seed=5)
    sp = StepSurprise(rssm, prev, act, mi, [False, False], True)  # neither row near a life boundary
    sp._agency = {0: 1.2, 1: 0.7}
    ns._as_cache = (mi, sp)
    ns.config["curiosity"]["info_gain"]["weight"] = 0.4
    ns._ig_term = None                                  # rebuilt by the helper from the config
    intrinsic = torch.zeros(2)
    with quiet():
        ns._info_gain_step(prev, act, mi, intrinsic, [False] * 2, [False] * 2, [{}] * 2)
        if brk == "gain_applied_twice":
            ns._info_gain_step(prev, act, mi, intrinsic, [False] * 2, [False] * 2, [{}] * 2)
    want = [0.4 * min(2.0, 1.2), 0.4 * min(2.0, 0.7)]
    ok &= all(abs(float(a) - b) < 1e-6 for a, b in zip(intrinsic, want))
    n += 2
    return Obs(bool(ok), exercised=n, note=f"gain {g:.3f} frontier {f:.3f} booked {intrinsic.tolist()} want {want}")


@H.scenario("CT-04", "WP-8 and RW-1 both scale WP-9's entropy target: an explicit product, each by its own owner",
            brk="second_scale_dropped", min_exercised=3,
            oracle="entropy coefficient update uses target = base * entropy_target_scale * uncertainty_scale (capped 0.99)")
def ct04_target_combiner(brk):
    torch.manual_seed(2)
    policy = StandaloneActorCritic(obs_dim=8, action_dim=3, hidden_dim=16, device=torch.device("cpu"))
    reg = CoefficientRegistry()
    Ao = AdaptiveOptim({"enabled": True, "entropy": {"enabled": True, "target": 0.5, "eta": 0.5}}, reg, policy)
    reg.claim("explore.entropy_target_scale", "wp8_competence", 1.0, 0.7, 1.3)
    reg.claim("explore.uncertainty_scale", modulation.OWNER, 1.0, 0.7, 1.3)
    reg.set("explore.entropy_target_scale", "wp8_competence", 1.2)
    reg.set("explore.uncertainty_scale", modulation.OWNER, 1.1)
    c0, h = policy.entropy_coef, 0.30
    tgt = min(0.99, 0.5 * 1.2 * 1.1)
    want = min(0.1, max(1e-4, c0 * math.exp(max(-1.0, min(1.0, 0.5 * (tgt - h))))))
    policy.last_update_stats = {"cond_entropy_norm": h}
    ctx = patch.object(reg, "get", lambda n, d=None: 1.0 if (brk and n == "explore.uncertainty_scale")
                       else CoefficientRegistry.get(reg, n, d))
    with ctx:
        Ao.run(policy, [], 0.0, 0)
    ok = abs(policy.entropy_coef - want) < 1e-12 and reg.get("policy.entropy_coef") == policy.entropy_coef
    try:                                                # neither controller may write the other's scale
        reg.set("explore.uncertainty_scale", "wp8_competence", 1.3)
        ok = False
    except OwnershipError:
        pass
    try:
        reg.set("explore.entropy_target_scale", modulation.OWNER, 1.3)
        ok = False
    except OwnershipError:
        pass
    return Obs(bool(ok), exercised=3, note=f"coef {policy.entropy_coef:.6f} want {want:.6f}")


# ============================================================================= CT-05
# FINDING D-GUARD: GuardClock.update returns True for rows WITHIN k_guard steps of a
# life boundary ("ignore"), but StepSurprise consumers in intrinsic_balance.py:73 and
# effect_discovery.py:98 read True as "eligible". The polarity is inverted for those two consumers.
def ct05_guard_pay(brk):
    rssm = make_rssm()
    ns = build_loop({"curiosity": {"info_gain": {"weight": 0.5, "floor": -100.0, "clip": 1.0}}}, rssm=rssm)
    pays, n = {}, 0
    ctx = patch.object(GuardClock, "update", lambda self, n_, ended: [False] * n_) if brk else contextlib.nullcontext()
    with ctx, quiet():
        for t in range(13):
            prev, act, mi = observe(rssm, 2, seed=10 + t)
            intrinsic = torch.zeros(2)
            ns._info_gain_step(prev, act, mi, intrinsic, [t == 0, False], [False, False], [{}, {}])
            pays[t] = intrinsic.tolist()
    ok = all(pays[t][0] == 0.0 for t in (1, 2, 3)) and pays[12][0] > 0.0 and all(pays[t][1] > 0.0 for t in (1, 2, 3, 12))
    n = 8
    return Obs(bool(ok), exercised=n, note=f"stream0 pays t1..3 {[pays[t][0] for t in (1, 2, 3)]} t12 {pays[12][0]}; stream1 t1 {pays[1][1]}")


def ct05_guard_trigger(brk):
    W = World(B=2)
    D = EffectDiscovery(EFF_CFG)
    clock = GuardClock(8)
    ctx = patch.object(GuardClock, "update", lambda self, n_, ended: [False] * n_) if brk else contextlib.nullcontext()
    fired = {0: 0, 1: 0}
    with ctx:
        for t in range(24):
            ended = [t in (19,), False]
            guard = clock.update(2, ended)
            spike = t == 21
            sp = W.sp([4.0 if spike else 0.2] * 2, [2.0 if spike else 0.0] * 2, guard=guard)
            for f in D.trigger.step(sp):
                fired[f["stream"]] += 1
    ok = fired[1] == 1 and fired[0] == 0
    return Obs(bool(ok), exercised=4, note=f"fires per stream {fired}; want {{0: 0, 1: 1}} (stream 1 mid-life fires, stream 0 post-boundary silent)")


@H.scenario("CT-05", "DEFECT D-GUARD: GuardClock polarity agrees with both StepSurprise consumers (information gain and the skill trigger)",
            brk="boundary_guard_dropped", min_exercised=12,
            oracle="mid-life rows are paid / fire; rows within k_guard of a life boundary are not; GuardClock marks boundary rows True but intrinsic_balance.py:73 and effect_discovery.py:98 read True as eligible")
def ct05_guard(brk):
    a, b = ct05_guard_pay(brk), ct05_guard_trigger(brk)
    return Obs(a.ok and b.ok, a.exercised + b.exercised, a.note + " | " + b.note)


@H.scenario("CT-05", "WP-1: streams stay separate (key, probation hit, base fire) and a life boundary cancels the pending effect",
            brk="wrong_stream_id", min_exercised=6,
            oracle="each stream's key points along ITS OWN view change; the skill running on stream 0 is credited only with stream 0's effect; an ended stream resolves nothing")
def ct05_streams(brk):
    W, D = World(B=2), EffectDiscovery(EFF_CFG)
    swap = (lambda ks: ks[::-1]) if brk == "wrong_stream_id" else (lambda ks: ks)
    t = 0
    for _ in range(14):
        W.step(D, t, [0.2, 0.2], [0.0, 0.0], swap([V[0], V[0]]), running=[SID, None])
        t += 1
    out_all = []
    for k in range(5):                      # both streams spike at k=1; their views change differently at k=3
        sp_s = [4.0, 4.0] if k == 1 else [0.2, 0.2]
        sp_a = [2.0, 2.0] if k == 1 else [0.0, 0.0]
        keys = [V[1], V[2]] if k >= 3 else [V[0], V[0]]
        out = W.step(D, t, sp_s, sp_a, swap(keys), running=[SID, None] if k < 3 else [None, None])
        out_all += out["resolved"]
        t += 1
    by_stream = {i: key for (i, key, _) in out_all}
    ok = set(by_stream) == {0, 1} and by_stream.get(0) != by_stream.get(1)
    if ok:
        ok &= centroid_cos(D, by_stream[0], 0, 1) > 0.99 and centroid_cos(D, by_stream[1], 0, 2) > 0.99
    inv = [v for v in D.probation._by_id.values()]
    ok &= any(v["sid"] == SID and v["hit"] for v in inv) if by_stream.get(0) == "effect_0" else False
    ok &= D.probation.base_fires.get(by_stream.get(1), 0) == 1
    # boundary: both spike again, stream 0 ends before the effect lands -> only stream 1 resolves
    res2 = []
    t = quiet_steps_b(W, D, t, 3)
    for k in range(5):
        sp_s = [4.0, 4.0] if k == 1 else [0.2, 0.2]
        sp_a = [2.0, 2.0] if k == 1 else [0.0, 0.0]
        keys = [V[1], V[2]] if k >= 3 else [V[0], V[0]]
        out = W.step(D, t, sp_s, sp_a, swap(keys), ended=[k == 2, False])
        res2 += out["resolved"]
        t += 1
    ok &= [i for (i, _, _) in res2] == [1]
    return Obs(bool(ok), exercised=6, note=f"keys by stream {by_stream}; after boundary {[i for (i, _, _) in res2]}")


def quiet_steps_b(W, D, t0, n):
    for t in range(t0, t0 + n):
        W.step(D, t, [0.2, 0.2], [0.0, 0.0], [V[0], V[0]])
    return t0 + n


@H.scenario("CT-05", "WP-8: measurements are kept per stream; a call that names one stream leaves the others' history alone",
            brk="wrong_stream_id", min_exercised=4,
            oracle="ctx[i].loss_ema tracks only stream i's own series; a partial update leaves the other context untouched")
def ct05_competence_streams(brk):
    C = CompetenceController({"min_meas": 2}, CoefficientRegistry())
    for _ in range(4):
        C.update({0: (1.0, 0.1), 1: (5.0, 0.1)})
    snap0 = (C.ctx[0].loss_ema, C.ctx[0].n, len(C.ctx[0].hist))
    for _ in range(3):
        C.update({1: (4.0, 0.1)} if brk != "wrong_stream_id" else {0: (4.0, 0.1)})
    ok = (C.ctx[0].loss_ema, C.ctx[0].n, len(C.ctx[0].hist)) == snap0
    ok &= abs(C.ctx[0].loss_ema - 1.0) < 1e-9 and C.ctx[1].loss_ema < 5.0 and C.ctx[1].n == 7
    return Obs(bool(ok), exercised=4, note=f"ctx0 {snap0} -> {(C.ctx[0].loss_ema, C.ctx[0].n)}; ctx1.n {C.ctx[1].n}")


@H.scenario("CT-05", "WP-1: an embedding of the wrong width is refused and leaves every key untouched",
            brk="silent_pad", min_exercised=3,
            oracle="assign() raises on a width mismatch; centroids, counts and key names are unchanged")
def ct05_width(brk):
    K = EffectKeyClusterer(tau=0.8, min_change=0.05)
    K.assign(V[1] - V[0])
    K.assign(V[2] - V[0])
    snap = {k: (v["c"].clone(), v["n"]) for k, v in K.keys.items()}
    refused = 0
    wrong = torch.ones(9)
    for _ in range(3):
        try:
            e = torch.cat([wrong, torch.zeros(0)])
            if brk == "silent_pad":                   # an undeclared migration: zero-pad to the old width
                e = e[:6]
            K.assign(e)
        except RuntimeError:
            refused += 1
    same = set(K.keys) == set(snap) and all(
        torch.equal(K.keys[k]["c"], snap[k][0]) and K.keys[k]["n"] == snap[k][1] for k in snap)
    ok = refused == 3 and same
    return Obs(bool(ok), exercised=3, note=f"refused {refused}/3, state unchanged {same}")


# ============================================================================= CT-06
@H.scenario("CT-06", "RW-1: losing the calibration source releases every modulated value within ceil(|v-1|/step) updates; recovery re-engages",
            brk="stale_source_kept", min_exercised=8,
            oracle="neutral reached within the release bound for the 3 coefficients and their consumers; valid input afterwards moves them again")
def ct06_modulation(brk):
    ns = rel_loop(rcfg())
    for i in range(8):
        rel_segment(ns, *jitter(i))
    for _ in range(10):
        rel_segment(ns, 0.95, 0.4, 0.2)
    names = ("explore.uncertainty_scale", "attn.temperature_scale", "consol.strength_scale")
    start = {n: coef(ns, n) for n in names}
    bound = {n: math.ceil(abs(start[n] - 1.0) / 0.05 - 1e-9) for n in names}
    assert max(bound.values()) >= 4, "fixture did not drive the coefficients away from neutral"
    steps_to_neutral, ok = {}, True
    last_valid = dict(ns._plan.last)
    for u in range(1, max(bound.values()) + 3):
        ns._plan.last = last_valid if brk == "stale_source_kept" else {"policy_calibration_ok": 0.0}
        ns._curr.bandits["p"].value = [0.4, 0.0, 0.0]
        with quiet():
            ns._relations_phase()
        for n in names:
            if n not in steps_to_neutral and abs(coef(ns, n) - 1.0) < 1e-9:
                steps_to_neutral[n] = u
    ok &= all(n in steps_to_neutral and steps_to_neutral[n] <= bound[n] for n in names)
    ok &= abs(ns._cons.strength - 1.0) < 1e-9 and abs(ns._am.net.temperature - 1.0) < 1e-9
    n_before = ns._rel.state["mod"]["z"].n
    for i in range(6):                                       # recovery: valid input again
        rel_segment(ns, *jitter(i))
    rel_segment(ns, 0.95, 0.4, 0.2)
    ok &= ns._rel.state["mod"]["z"].n > n_before and abs(coef(ns, "explore.uncertainty_scale") - 1.0) > 0.04
    return Obs(bool(ok), exercised=len(names) * 2 + 2, note=f"bound {bound} reached {steps_to_neutral}")


def fault_logger():
    recs = []

    class _H(logging.Handler):
        def emit(self, r):
            recs.append(r)
    h = _H(level=logging.WARNING)
    logging.getLogger().addHandler(h)
    return recs, h


@H.scenario("CT-06", "WP-8: a failing dependency releases the coefficient to base, observably",
            brk="fault_not_released", min_exercised=4,
            oracle="after the failure the mixer gain is exactly 1.0, the registry holds no competence name, and a warning names the fault")
def ct06_comp_fault(brk):
    cfg = {"competence": {"enabled": True, "every_segments": 1, "min_meas": 3, "signal_ema": 0.5, "gain_amp": 0.3}}
    ns = build_loop(cfg)
    state = {"fail": False}

    def flaky(*_a, **_k):
        if state["fail"]:
            raise RuntimeError("world-model dependency down")
        return dict(MEAS)
    recs, h = fault_logger()
    try:
        with measuring(flaky):
            MEAS.update({0: (1.0, 0.1)})
            for _ in range(8):
                with quiet():
                    ns._competence_phase()
            lifted = ns.reward_mixer.intrinsic_gain
            state["fail"] = True
            if brk == "fault_not_released":
                with patch.object(CompetenceController, "release", lambda self: None):
                    with quiet():
                        ns._competence_phase()
            else:
                with quiet():
                    ns._competence_phase()
    finally:
        logging.getLogger().removeHandler(h)
    ok = lifted > 1.1 and ns.reward_mixer.intrinsic_gain == 1.0 and coef(ns, "mixer.intrinsic_gain") is None \
        and any("competence phase failed" in r.getMessage() for r in recs)
    return Obs(bool(ok), exercised=4, note=f"gain before fault {lifted:.3f}, after {ns.reward_mixer.intrinsic_gain}")


# FINDING (real defect, see report): after one transient fault `_competence_phase` sets `self._comp = False`,
# which nothing ever resets: the controller never comes back although the dependency recovers
# (developmental_loop.py, _competence_phase except-branch). Same shape in _adaptive_optim_phase,
# _meta_adapt_phase and (via _wire_off) the relation wiring. CLAUDE.md 4.1: guard-becomes-latch.
def ct06_comp_recover(brk):
    cfg = {"competence": {"enabled": True, "every_segments": 1, "min_meas": 3, "signal_ema": 0.5, "gain_amp": 0.3}}
    ns = build_loop(cfg)
    state = {"fail": False}

    def flaky(*_a, **_k):
        if state["fail"]:
            raise RuntimeError("world-model dependency down")
        return dict(MEAS)
    recs, h = fault_logger()
    try:
        with measuring(flaky):
            MEAS.update({0: (1.0, 0.1)})
            for _ in range(8):
                with quiet():
                    ns._competence_phase()
            state["fail"] = True
            with quiet():
                ns._competence_phase()
            state["fail"] = False
            if brk == "fault_latches":
                state["fail"] = True               # the dependency never comes back
            for _ in range(12):
                with quiet():
                    ns._competence_phase()
    finally:
        logging.getLogger().removeHandler(h)
    g = ns.reward_mixer.intrinsic_gain
    ok = g > 1.0 and coef(ns, "mixer.intrinsic_gain") is not None
    return Obs(bool(ok), exercised=3, note=f"gain 12 segments after recovery {g}; controller state {ns._comp!r}"[:160])


@H.scenario("CT-06", "RW-1: a fault inside the relation wiring releases every coefficient to neutral, observably",
            brk="fault_not_released", min_exercised=5,
            oracle="all RW-1 coefficients absent, consolidation strength and attention temperature exactly 1.0, a warning names the fault")
def ct06_rel_fault(brk):
    ns = rel_loop(rcfg())
    for i in range(8):
        rel_segment(ns, *jitter(i))
    for _ in range(6):
        rel_segment(ns, 0.95, 0.4, 0.2)
    pre_off_neutral = ns._cons.strength != 1.0 and ns._am.net.temperature != 1.0

    class Dead:
        @property
        def last(self):
            raise RuntimeError("planning dependency down")
    recs, h = fault_logger()
    try:
        ns._plan = Dead()
        ctx = patch.object(modulation, "release", lambda loop, reg: None) if brk else contextlib.nullcontext()
        with ctx, quiet():
            ns._relations_phase()
    finally:
        logging.getLogger().removeHandler(h)
    reg = ns._rel if ns._rel is not False else None
    names = ("explore.uncertainty_scale", "attn.temperature_scale", "consol.strength_scale")
    ok = pre_off_neutral and all(coef(ns, n) is None for n in names) and ns._cons.strength == 1.0 \
        and ns._am.net.temperature == 1.0 and any("relation wiring" in r.getMessage() for r in recs)
    return Obs(bool(ok), exercised=5, note=f"strength {ns._cons.strength} temp {ns._am.net.temperature} warned {len(recs)}")


# FINDING (real defect, see report): RelationRegistry.wire_all adds a failed module to `_wire_off`
# for the whole run (relations.py wire_all, "idled for the run"); the dependency recovering does not
# re-open it.
def ct06_rel_recover(brk):
    ns = rel_loop(rcfg())
    for i in range(8):
        rel_segment(ns, *jitter(i))

    class Flaky:
        broken = False

        @property
        def last(self):
            if self.broken:
                raise RuntimeError("planning dependency down")
            return self._last

        @last.setter
        def last(self, v):
            self._last = v
    f = Flaky()
    f.last = valid_plan(0.3)
    ns._plan = f
    f.broken = True
    with quiet():
        ns._relations_phase()
    f.broken = brk == "fault_latches"                 # the break: the dependency never comes back
    for i in range(12):
        f.last = valid_plan(0.30 + 0.6 * (i >= 6))
        with quiet():
            ns._relations_phase()
    moved = coef(ns, "explore.uncertainty_scale")
    ok = moved is not None and moved > 1.0
    return Obs(bool(ok), exercised=3, note=f"explore.uncertainty_scale 12 segments after recovery: {moved}; idled {sorted(ns._rel._wire_off) if ns._rel else None}"[:200])


@H.scenario("CT-06", "DEFECT D-LATCH: after one transient fault the WP-8 controller and the RW-1 wiring resume (they are idled for the run otherwise)",
            brk="fault_latches", min_exercised=6,
            oracle="within 12 segments of the dependency recovering, competence moves the mixer gain again and modulation moves explore.uncertainty_scale again")
def ct06_latch(brk):
    a, b = ct06_comp_recover(brk), ct06_rel_recover(brk)
    return Obs(a.ok and b.ok, a.exercised + b.exercised, a.note + " | " + b.note)


@H.scenario("CT-06", "WP-5: a quiet stream cannot starve the information-gain term, and a burst of other income re-opens to normal",
            brk="floor_removed", min_exercised=4,
            oracle="with ~0 other income the capped term still pays > 0 (floor); after a burst the allowance decays back")
def ct06_sharecap(brk):
    W = World(B=1)
    term = InfoGainTerm(weight=0.5, floor=-100.0, clip=1.0, cap_share=0.5,
                        cap_floor=0.0 if brk == "floor_removed" else 0.002)
    quiet_pays = [term.pay(W.sp([0.2], [1.0]), [0.0])[0] for _ in range(5)]
    for _ in range(30):
        term.pay(W.sp([0.2], [1.0]), [10.0])          # a burst of other income raises the allowance
    burst = term.pay(W.sp([0.2], [1.0]), [10.0])[0]
    for _ in range(1500):
        term.pay(W.sp([0.2], [1.0]), [0.0])
    after = term.pay(W.sp([0.2], [1.0]), [0.0])[0]
    ok = all(p > 0.0 for p in quiet_pays) and burst > 0.4 and 0.0 < after < 0.05
    return Obs(bool(ok), exercised=4, note=f"quiet {quiet_pays[0]:.4f} burst {burst:.3f} after decay {after:.4f}")


@H.scenario("CT-06", "WP-1: a retired effect skill backs off by doubling, restarts with a fresh record, and a later skill can be kept",
            brk="backoff_not_doubled", min_exercised=5,
            oracle="holds 2 then 4 after two failed probations (record dropped each time); the third attempt, with real hits, is kept")
def ct06_probation(brk):
    cfg = dict(EFF_CFG, backoff_start=0 if brk == "backoff_not_doubled" else 2)
    W, D = World(), EffectDiscovery(cfg)
    t = quiet_steps(W, D, 0, 14)
    for k in range(5):                                       # a base-policy effect makes effect_0 exist
        W.step(D, t, [4.0 if k == 1 else 0.2], [2.0 if k == 1 else 0.0], [V[1] if k == 3 else V[0]])
        t += 1
    t = quiet_steps(W, D, t, 20)
    holds, kept = [], []
    for rnd in range(3):
        for _ in range(3):
            t, res = invoke(W, D, t, hit=(rnd == 2))
            for tag, v in [x for x in res if x[0] == "verdict"]:
                (holds if v["verdict"] == "retire" else kept).append(v)
    ok = [v["hold"] for v in holds] == [2, 4] and len(kept) == 1 and kept[0]["succ"] == 3
    return Obs(bool(ok), exercised=len(holds) + len(kept) + 3,
               note=f"holds {[v['hold'] for v in holds]} kept {[(v['succ'], v['n']) for v in kept]}")


# ============================================================================= CT-07
def run_phase2(W, D, t0):
    """Two more invocations with the same effect, after three quiet steps (the restore resets _prev)."""
    t = quiet_steps(W, D, t0, 3)
    out = []
    for _ in range(2):
        t, res = invoke(W, D, t)
        out += res
    return t, out


@H.scenario("CT-07", "WP-1: save, recreate, restore and continue the same fixture gives the same keys, baselines and verdict",
            brk="state_not_restored", min_exercised=5,
            oracle="restored object reproduces the uninterrupted run's resolved keys, centroids, probation record and keep verdict")
def ct07_effect_persist(brk):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "effect_discovery.pt")
        W, D = World(), EffectDiscovery(EFF_CFG, path)
        t = quiet_steps(W, D, 0, 14)
        t, _ = invoke(W, D, t)
        t = quiet_steps(W, D, t, 6)                       # the first verdict-delay has elapsed
        D.save()
        saved = os.path.exists(path)
        W1, W2 = World(), World()
        _, ref = run_phase2(W1, D, t)                     # uninterrupted continuation
        D2 = EffectDiscovery(EFF_CFG, None if brk == "state_not_restored" else path)
        _, got = run_phase2(W2, D2, t)

        def sig(res):
            return [(x[1], round(float(x[2].sum()), 4)) if x[0] != "verdict"
                    else ("verdict", x[1]["verdict"], x[1]["succ"], x[1]["n"]) for x in res]
        same_out = sig(ref) == sig(got) and any(x[0] == "verdict" for x in ref)
        same_keys = sorted(D.keys.keys) == sorted(D2.keys.keys) and all(
            float(D.keys.centroid(k) @ D2.keys.centroid(k)) > 0.999 for k in D.keys.keys if k in D2.keys.keys)
        same_state = D.trigger.state()["n"] == D2.trigger.state()["n"] \
            and D.probation.state()["rec"] == D2.probation.state()["rec"] \
            and D.probation.state()["retirements"] == D2.probation.state()["retirements"]
        ok = saved and same_out and same_keys and same_state
    return Obs(bool(ok), exercised=6, note=f"saved {saved} same outputs {same_out} keys {same_keys} state {same_state}")


@H.scenario("CT-07", "WP-8 / RW-1: coefficient values are not persisted by design; a recreated brain starts exactly neutral and warms up",
            brk="stale_value_after_restore", min_exercised=4,
            oracle="RewardMixer.state_dict carries no gain; after recreation gains are 1.0 and nothing is written before min_meas measurements")
def ct07_reset(brk):
    ns = compose()
    run_composed(ns, 12, check=False)
    sd = ns.reward_mixer.state_dict()
    in_state = [k for k in sd if "gain" in k]
    lifted = (ns.reward_mixer.intrinsic_gain, ns.reward_mixer.frontier_gain)
    ns2 = compose()
    ns2.reward_mixer.load_state_dict(sd)
    if brk == "stale_value_after_restore":
        ns2.reward_mixer.intrinsic_gain = lifted[0]     # a restore that smuggled the gain back in
    ok = not in_state and (lifted[0] != 1.0 or lifted[1] != 1.0)
    ok &= ns2.reward_mixer.intrinsic_gain == 1.0 and ns2.reward_mixer.frontier_gain == 1.0
    with measuring():
        for seg in range(2):                            # fewer than min_meas (3): nothing may be written
            MEAS.clear()
            MEAS.update({0: (1.0, 0.1)})
            with quiet():
                ns2._competence_phase()
            ok &= coef(ns2, "mixer.intrinsic_gain") in (None, 1.0) and ns2.reward_mixer.intrinsic_gain == 1.0
    return Obs(bool(ok), exercised=4, note=f"gain in state {in_state}; lifted before {lifted}")


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


CAT04_FILES = ("developmental_ai/curiosity/intrinsic_balance.py", "developmental_ai/curiosity/action_surprise.py",
               "developmental_ai/skill_bank/effect_discovery.py", "developmental_ai/core/competence.py",
               "developmental_ai/core/coefficients.py", "developmental_ai/core/relations_modulation.py",
               "developmental_ai/core/relations_motivation.py", "developmental_ai/core/relations_skills.py")


@H.scenario("CT-08", "evaluator fields cannot enter CAT-04: registration refuses them and no CAT-04 module imports the evaluator",
            brk="evaluator_probe_registered", min_exercised=20,
            oracle="evaluator-tagged source and evaluator-defined probe raise EvaluatorLeakError; every CAT-04 channel is learner-tagged; no direct import of the evaluator or detector packages")
def ct08_evaluator(brk):
    import developmental_ai.core.relations as rel
    reg = RelationRegistry({"enabled": True})
    refused, n = 0, 0

    def probe(loop):                       # a dummy defined "inside" the evaluator package by name only
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
    cat04 = [c for c in full.channels() if getattr(c, "group", "") in ("modulation", "motivation", "skills")]
    tags_ok = bool(cat04) and all(c.source_tag == "learner" for c in cat04)
    probes_ok = all(not (getattr(c.probe, "__module__", "") or "").startswith(
        ("developmental_ai.evaluator", "developmental_ai.game_template.detectors")) for c in cat04)
    n += len(cat04)
    leaks = []
    for f in CAT04_FILES:
        bad = [m for m in _imports(os.path.join(ROOT, f))
               if m.startswith(("developmental_ai.evaluator", "developmental_ai.game_template.detectors"))
               or m == "developmental_ai.evaluator"]
        leaks += [(f, m) for m in bad]
        n += 1
    ok = refused == 2 and tags_ok and probes_ok and not leaks
    return Obs(bool(ok), exercised=n, note=f"refused {refused}/2, {len(cat04)} CAT-04 channels learner-tagged, leaks {leaks}")


def _rng_sig():
    return (torch.get_rng_state().sum().item(), random.getstate()[1][:5], np.random.get_state()[1][:5].tolist())


OBS_ON = {"MOT": {"INTR": {"enabled": True}, "EXTR": {"enabled": True}},
          "SCAF": {"ZPD": {"enabled": True}}, "FLOW": {"MOT": {"enabled": True}},
          "SDT": {"INTR": {"enabled": True}}, "SENSE": {"SDT": {"enabled": True}},
          "PROC": {"SKILL": {"enabled": True}}, "AUTO": {"SKILL": {"enabled": True}}}


@H.scenario("CT-08", "read-only probes and extra logging calls do not change learner state, reward or RNG",
            brk="probe_mutates_state", min_exercised=20,
            oracle="coefficients, mixer, info weight, consumers, skill bank and all three RNG streams are identical with the observation channels on and sampled/logged more often")
def ct08_noninterference(brk):
    with tempfile.TemporaryDirectory() as d:
        def build(extra):
            cfg = rcfg()
            if extra:
                cfg.update(OBS_ON)
            ns = rel_loop(cfg)
            if extra:
                ns.skill_bank, ns.option_executor = SkillBank(storage_dir=d), None
                ns._comp = CompetenceController({"min_meas": 1}, CoefficientRegistry())
            return ns

        def run(ns, extra):
            seed_state = (torch.manual_seed(7).get_state(), random.seed(7), np.random.seed(7))   # same RNG start
            rs = np.random.RandomState(0)
            sampled = 0
            for i in range(10):
                rel_segment(ns, 0.3 + 0.6 * rs.rand(), 0.1 + rs.rand(), 0.1 + rs.rand())
                if extra:
                    if brk == "probe_mutates_state" and i >= 1:
                        def evil(loop):
                            torch.rand(1)                                  # a "read-only" probe that draws RNG
                            loop.reward_mixer.intrinsic_weight += 1e-3     # and writes the reward weight
                            return 1.0
                        ns._rel.get("rel.MOT.EXTR").probe = evil
                    sampled += ns._rel.sample_all(ns)                      # sampled more often than once a segment
                    ns._rel.line()
                    ns._comp.report({"ready": False})                     # extra logging calls
            reg = registry_for(ns)
            if extra:
                sampled += sum(c.segments for c in ns._rel.channels() if c.enabled and c.probe is not None)
            return ({r["name"]: r["value"] for r in reg.table()}, ns.reward_mixer.state_dict(),
                    ns._ig_term.weight, ns._cons.strength, ns._am.net.temperature,
                    ns.reward_mixer.frontier_gain, _rng_sig()), sampled
        ns_a, ns_b = build(False), build(True)
        a, _ = run(ns_a, False)
        b, sampled = run(ns_b, True)
        bank_untouched = not ns_b.skill_bank.skills
    same = [x == y for x, y in zip(a, b)]
    return Obs(all(same) and bank_untouched, exercised=sampled + len(same),
               note=f"identical (coef, mixer, info, cons, temp, frontier, rng): {same}; sampled {sampled}")


if __name__ == "__main__":
    sys.exit(H.finish())
