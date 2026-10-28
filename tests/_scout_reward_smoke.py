"""Scout reward smoke (2026-10-18): ONE reward function for every stream.

THE INCIDENT THIS ENCODES

    At tiny x 8 Doom streams, 7 of every 8 PPO rows are SCOUT rows, and
    `_scout_mixed_reward` paid them a reduced reward: learning progress +
    episodic novelty + persistence/gaze potentials + GUI terms. Stream 0
    additionally got view novelty, imagination curiosity, habituation,
    empowerment and the farm damp. One critic fitted both reward functions
    (diagnosis of the near-uniform policy, docs/assurance/
    ass-003-policy-learning.md, ASS-003). `policy.scouts_same_reward` pays
    every scout by each GAME-AGNOSTIC term stream 0 is paid by, in stream
    0's order, with that stream's OWN state. VLM / fovea / symbolizer terms
    stay stream-0-only (documented in _scout_mixed_reward).

Contracts:
    A. LEGACY IDENTITY: with the switch off, _scout_mixed_reward returns
       exactly what the pre-2026-10-18 function returned (a frozen copy of
       it lives below) over 300 random steps — persistence, gaze level, GUI
       zeroing and dwell cost all exercised.
    B. SAME RULE AS STREAM 0, per term, for identical inputs: view novelty
       (count rule + stream 0's no-fovea/no-symbol pricing), habituation,
       empowerment (same score fn, cadence, potential) and the farm damp
       (same detector rule on the same cells / incomes) pay a scout exactly
       what they pay stream 0.
    C. ISOLATION: scout calls never move stream 0's state (view counts, farm
       detector cells, empowerment phi/last step, habituation memory/tally).
    D. LIVE: the real lifelong body (_collect_segment, 3 streams) builds
       per-stream view counts, farm detectors, empowerment state and
       imagination instances for EVERY scout; switch off builds none.
    E. IMAGINATION CADENCE: each scout probes once per lcm(n, every)/n of its
       own steps (stream 0's per-stream period), staggered across streams.
    F. WIRED: the scout call site passes action= and rssm_state=; stream 0's
       view novelty and imagination use the same shared helpers.

Run: PYTHONPATH=. python tests/_scout_reward_smoke.py
     SCOUT_FALSIFY=1 ... runs B, D, E with the switch OFF (they must FAIL).
"""
import math
import os
import random
import shutil
import sys
import tempfile
import types

sys.path.insert(0, ".")

import logging

import numpy as np
import torch
import yaml

logging.disable(logging.WARNING)

NAME = "scout-reward"
FALSIFY = bool(os.environ.get("SCOUT_FALSIFY"))
TMP = tempfile.mkdtemp(prefix="scout_reward_smoke_")
FAILS = []
N = [0]


def check(cond, msg):
    N[0] += 1
    print(f"  {N[0]}. {'ok   ' if cond else 'FAIL '} {msg}")
    if not cond:
        FAILS.append(msg)


def _env_cls():
    import gymnasium as gym

    class FakeEnv(gym.Env):
        metadata = {}

        def __init__(self, idx):
            self.observation_space = gym.spaces.Box(-5, 5, (8,), np.float32)
            self.action_space = gym.spaces.Discrete(4)
            self._rng = np.random.default_rng(10 + idx)
            self._t = 0

        def reset(self, **kw):
            return self._rng.normal(0, 1, 8).astype(np.float32), {}

        def step(self, a):
            self._t += 1
            return (self._rng.normal(0, 1, 8).astype(np.float32), 0.0,
                    self._t % 41 == 0, False, {})

    return FakeEnv


def _agent(same=True, n=3):
    import developmental_ai.core.developmental_loop as dl
    from developmental_ai.environments.wrappers import DevelopmentalEnvWrapper
    FakeEnv = _env_cls()
    state = {"n": 0}

    def fake(*a, **k):
        i = state["n"]
        state["n"] += 1
        return DevelopmentalEnvWrapper(FakeEnv(i), normalize_obs=False), None
    dl.make_env = fake
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 32, "encoder_hidden": 32,
        "decoder_hidden": 32, "batch_size": 4, "sequence_length": 8,
        "train_iters": 1, "buffer_capacity": 4000})
    cfg["parallel_envs"] = {"enabled": True, "num_envs": n}
    cfg["policy"].update({"n_steps": 10 ** 9, "minibatch_size": 16,
                          "n_epochs": 1, "arch": "rssm",
                          "scouts_in_ppo": True, "scouts_same_reward": same})
    cfg["lifelong"] = {"enabled": True, "forever": False, "segment_len": 60,
                       "wm_train_every": 10 ** 6, "goal_horizon": 10 ** 9,
                       "state_decay": 1.0, "reset_on_death_only": True,
                       "stop_file": os.path.join(TMP, "STOP")}
    cfg.setdefault("curiosity", {}).update({
        "novelty_weight": 0.08, "novelty_semantic": True,
        "habituation_scale": 30.0})
    cfg["imagination_curiosity"] = {"enabled": True, "every": 4,
                                    "rollouts": 2, "horizon": 2}
    cfg["infra"] = {"empowerment_weight": 0.05, "empowerment_interval": 2,
                    "farm_key": "view"}
    cfg.setdefault("loop", {})["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = os.path.join(TMP, "sb")
    cfg["metrics"] = {"enabled": False}
    cfg.pop("diagnostics", None)
    ai = dl.DevelopmentalAI(config=cfg)
    ai._ensure_parallel_envs()
    return ai


def _close(ai):
    for e in getattr(ai, "envs", []) or []:
        try:
            e.close()
        except Exception:
            pass


# ---- the scout reward as it was before 2026-10-18 (frozen reference) -----
def _old_scout(self, e_i, info, raw_ext, intrinsic_e):
    _w = (info or {}).get("world") or {}
    _st = self._sc_phi.setdefault(e_i, {})
    _g = float(self.config.get("policy", {}).get("gamma", 0.99))
    _ext = float(raw_ext)
    _int = float(intrinsic_e)
    _pw = float(self._persist_weight)
    if _pw > 0.0:
        _phi = min(1.0, float(info.get("attack_run", 0.0) or 0.0)
                   / max(1.0, float(self._persist_ticks)))
        _int += _pw * (_g * _phi - float(_st.get("persist", 0.0)))
        _st["persist"] = _phi
    _plw = float(self._pitch_level_weight)
    _pv = _w.get("pitch")
    if _plw > 0.0 and _pv is not None:
        _php = -((min(90.0, abs(float(_pv))) / 90.0) ** 4)
        if "pitch" in _st:
            _int += _plw * (_php - float(_st["pitch"]))
        _st["pitch"] = _php
    if bool(info.get("gui_open")):
        _int = 0.0
    _ext += self._gui_reward(e_i, info)
    return self.reward_mixer.mix(_int, _ext, update_stats=False)


class _Rec:
    """learning-telemetry stand-in: records each scout row's term parts."""
    def __init__(self):
        self.rows = []

    def book_parts(self, e, parts_i, parts_e, mixed, mixer, after_damp=None,
                   mults=None):
        self.rows.append((e, dict(parts_i), dict(mults or {})))


def test_legacy(ai):
    print("A. legacy identity (switch off)")
    ai._scouts_same_reward = False
    ai._persist_weight, ai._persist_ticks = 0.05, 20
    ai._pitch_level_weight = 0.02
    ai._gui_dwell_weight, ai._gui_dwell_step_cost = 0.3, 0.001
    rng = np.random.default_rng(5)
    steps = []
    for _ in range(300):
        steps.append((int(rng.integers(1, 3)),
                      {"attack_run": int(rng.integers(0, 30)),
                       "gui_open": bool(rng.random() < 0.15),
                       "world": {"pitch": float(rng.uniform(-90, 90))}},
                      float(rng.normal(0, 0.1)), float(rng.normal(0, 0.05))))
    ai._sc_phi = {}
    new = [ai._scout_mixed_reward(e, dict(i), r, x) for e, i, r, x in steps]
    ai._sc_phi = {}
    old = [_old_scout(ai, e, dict(i), r, x) for e, i, r, x in steps]
    check(np.allclose(new, old, rtol=0, atol=1e-12),
          f"switch off == the pre-change scout reward on 300 steps "
          f"(max |d| {max(abs(a - b) for a, b in zip(new, old)):.1e})")


def test_same_rule(ai):
    from developmental_ai.infra.stack import InfraStack
    print("B/C. same rule as stream 0, and stream 0's state untouched")
    ai._scouts_same_reward = not FALSIFY
    ai._persist_weight = ai._pitch_level_weight = ai._gui_dwell_weight = 0.0
    ai.imagination_curiosity.enabled = False      # isolate the other terms
    ai._ltel = rec = _Rec()
    nw = float(ai._novelty_weight)
    floor = float(ai._nov_unnamed_floor) if ai._nov_semantic else 1.0
    # state snapshots for C
    s0 = (dict(ai._nov_counts), dict(ai.infra.farm._cells),
          ai.infra._emp_phi_prev, ai.infra._emp_last_step,
          dict(ai._habit_prev), getattr(ai, "_habit_hits", 0))
    # deterministic empowerment score (a function of the latent row)
    ai.infra._empowerment_fn = lambda wm, start, A: float(
        start["h"].float().sum().item())
    ref = InfraStack({"empowerment_weight": 0.05, "empowerment_interval": 2},
                     action_dim=int(ai.action_dim))
    ref._empowerment_fn = ai.infra._empowerment_fn
    H = int(ai.world_model.rssm.deterministic_size)
    keys = [torch.full((1, 6), v) for v in (1.0, 1.0, 1.0, -1.0, 2.0, 1.0)]
    infos = [{"achievements": {"mine_dirt": k}, "breaks_by_type": {"dirt": 40}}
             for k in (0, 1, 2, 2, 3, 4)]
    exp_nov, exp_hab, exp_emp, exp_farm = [], [], [], []
    ref_counts, ref_prev = {}, {}
    for t, (vk, inf) in enumerate(zip(keys, infos)):
        ai.total_timesteps = 10 * (t + 1)
        ai._vk_step = (ai.total_timesteps, [vk, vk, vk])
        h = torch.full((3, H), 0.1 * (t + 1))
        z = torch.zeros(3, int(ai.world_model.rssm.stochastic_size
                               * ai.world_model.rssm.stochastic_classes))
        rs = {"h": h, "z": z}
        rec.rows.clear()
        ai._scout_mixed_reward(1, dict(inf), 0.0, 0.01, action=1,
                               rssm_state=rs)
        parts, mults = rec.rows[-1][1], rec.rows[-1][2]
        # what STREAM 0's helpers pay for the same inputs
        pc = ai._perceptual_cell(vk)
        exp_nov.append((ai._view_novelty_count(ref_counts, pc, nw) * floor,
                        parts.get("novelty")))
        exp_hab.append((ai._habituation_factor(dict(inf), prev=ref_prev,
                                               tally=False),
                        mults.get("habituation", 1.0)))
        e0 = ref.empowerment_shaping(ai.world_model, rs, int(ai.action_dim),
                                     ai.total_timesteps,
                                     float(ai.config["policy"].get("gamma", 0.99)),
                                     stream=0)
        exp_emp.append((e0, parts.get("empowerment", 0.0)))
    check(all(a is not None and b is not None and abs(a - b) < 1e-12
              for a, b in exp_nov),
          "B. view novelty pays a scout exactly stream 0's rule %s"
          % [round(b, 4) if b is not None else None for _, b in exp_nov])
    check(all(abs(a - b) < 1e-12 for a, b in exp_hab)
          and any(a < 1.0 for a, _ in exp_hab),
          "B. habituation: same factor on the same break counters %s"
          % [round(b, 3) for _, b in exp_hab])
    check(all(abs(a - b) < 1e-12 for a, b in exp_emp)
          and any(abs(a) > 0 for a, _ in exp_emp),
          "B. empowerment: same score, cadence and potential as stream 0 %s"
          % [round(b, 5) for _, b in exp_emp])
    # FARM: the same cell + income sequence, scout detector vs stream 0's
    ref2 = InfraStack({}, action_dim=int(ai.action_dim))
    ai.infra._stream_farms.clear()
    d_sc, d_0 = [], []
    for t in range(60):
        cell = ("view", t % 3)                      # a 3-cell loop
        d_sc.append(ai.infra.stream_farm_damp(1, cell, 0, 0.05, 1000 + t))
        ref2.on_step(step=1000 + t, action=0, events=[], cell=cell,
                     position=None, pitch=None, fovea_probs=None,
                     fovea_counts=None, step_reward=0.05)
        d_0.append(ref2.last_loop_damp)
    check(d_sc == d_0 and min(d_sc) < 1.0,
          "B. farm damp: a scout's detector damps the same loop exactly as "
          "stream 0's (min %.3f)" % min(d_sc))
    s1 = (dict(ai._nov_counts), dict(ai.infra.farm._cells),
          ai.infra._emp_phi_prev, ai.infra._emp_last_step,
          dict(ai._habit_prev), getattr(ai, "_habit_hits", 0))
    check(s0 == s1, "C. stream 0's view counts, farm cells, empowerment "
          "phi/last step and habituation memory/tally are untouched")
    ai._ltel = None


def test_cadence(ai):
    print("E. imagination cadence per scout")
    ai._scouts_same_reward = not FALSIFY
    ai.imagination_curiosity.enabled = True
    ai._last_wm_metrics = {"loss": 1.0}
    calls = {}

    class _Inst:
        def __init__(self, e):
            self.e = e

        def bonus(self, *a, **k):
            calls.setdefault(self.e, []).append(ai._sc_imag_steps[self.e] - 1)
            return 0.0
    n = int(ai._num_envs)
    ai._sc_imag = {e: _Inst(e) for e in range(1, n)}
    ai._sc_imag_steps = {}
    H = int(ai.world_model.rssm.deterministic_size)
    rs = {"h": torch.zeros(n, H), "z": torch.zeros(n, 4)}
    L = 120
    for _ in range(L):
        for e in range(1, n):
            if ai._scouts_same_reward:
                ai._scout_imagination_bonus(e, rs)
    period = math.lcm(n, int(ai.imagination_curiosity.every)) // n
    ok = all(len(calls.get(e, [])) == L // period for e in range(1, n))
    firsts = [calls.get(e, [None])[0] for e in range(1, n)]
    check(ok and len(set(firsts)) == len(firsts),
          f"E. each scout probes every {period} own steps "
          f"({[len(calls.get(e, [])) for e in range(1, n)]} of {L}), "
          f"staggered (first probes at {firsts})")


def test_live():
    print("D. live body builds per-stream state for every scout")
    ai = _agent(same=not FALSIFY, n=3)
    ai._last_wm_metrics = {"loss": 1.0}      # imagination is gated on a WM
    try:
        ai._collect_segment()
    finally:
        _close(ai)
    sc = (1, 2)
    check(all(len(ai._sc_nov_counts.get(e, {})) > 0 for e in sc),
          "D. view counts per scout %s" % {e: len(ai._sc_nov_counts.get(e, {}))
                                           for e in sc})
    check(set(ai.infra._stream_farms) == set(sc),
          "D. a farm detector per scout %s" % sorted(ai.infra._stream_farms))
    check(set(ai.infra._emp_streams) == set(sc),
          "D. empowerment state per scout %s" % sorted(ai.infra._emp_streams))
    check(set(ai._sc_imag) == set(sc),
          "D. an imagination instance per scout %s" % sorted(ai._sc_imag))
    off = _agent(same=False, n=3)
    off._last_wm_metrics = {"loss": 1.0}
    try:
        off._collect_segment()
    finally:
        _close(off)
    check(not off._sc_nov_counts and not off.infra._stream_farms
          and not off.infra._emp_streams and not off._sc_imag,
          "D. switch off (legacy): no scout term state is built")
    return ai


def test_wiring():
    print("F. wiring")
    src = open("developmental_ai/core/developmental_loop.py").read()
    check("action=env_actions[_e], rssm_state=rssm_state)" in src,
          "F. the scout call site passes action= and rssm_state=")
    check(src.count("self._view_novelty_count(") == 2
          and "_imag_policy = self._imagination_policy_fn(_ikv, 0)" in src,
          "F. stream 0's view novelty and imagination use the shared helpers")


if __name__ == "__main__":
    try:
        ai = _agent(same=True, n=3)
        try:
            test_legacy(ai)
            test_same_rule(ai)
            test_cadence(ai)
        finally:
            _close(ai)
        test_live()
        test_wiring()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print(f"[{NAME}] FAILED ({len(FAILS)})")
        sys.exit(1)
    print(f"[{NAME}] ALL PASS")
