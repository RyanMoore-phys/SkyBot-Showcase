"""Real-time credit smoke (2026-10-18): every transition is labelled with the
action that DROVE it, and PPO pays each decision for its own interval.

THE LIVE INCIDENT THIS ENCODES

    Doom (ViZDoom real-time) and every Player game never wait for SkyBot.
    `vizdoom_env._step_realtime(a)` swapped the held buttons to `a` and
    returned the newest frame at once; the clock thread had played every tic
    of that interval with the PREVIOUS decision's buttons (the loop thinks
    ~165 ms, ~5.7 tics, per decision). So step(a_t) returned the outcome of
    a_{t-1}, and every learner paired it with a_t: the world model and its
    inverse head, the curiosity forward model and its action-attribution
    gate (measured on main: 0.0076, "very low"), empowerment and PPO. After
    40 minutes the policy was 97-98 % of uniform (ln 16) in every run.

    The fix (developmental_ai/core/realtime_credit.py): executed-action
    labels, a one-prior-step policy lookahead through the HELD action, and a
    one-step PPO credit delay. Synchronous games are untouched.

    The harness is the census smoke's real DevelopmentalAI running the live
    body (`_collect_segment`), on a FakeLagEnv that behaves like the Doom
    clock: step(a) returns a frame produced by the PREVIOUS held action, and
    that frame ENCODES that action, so every check below is exact.

Contracts:
    A. REPLAY LABELS: on every contiguous row of a real-time stream, the
       stored action == the action decoded from the NEXT row's frame (the
       interval's real cause). Today's code matches only by chance (~1/K).
    B. CURIOSITY + BELIEF: the action handed to the curiosity reward is the
       executed one on real-time streams, and observe_step gets the same.
    C. PPO CREDIT: with a reward of 1.0 paid iff the interval's executed
       action is TARGET, every stored tau=1 row has reward 1.0 exactly when
       its OWN action is TARGET.
    D. LOOKAHEAD: the policy input's h (one prior step through the held
       action) equals the next posterior's h on every purity-1 interval.
    E. SYNCHRONOUS UNCHANGED: a fleet of synchronous envs stores identical
       replay arrays and identical PPO rows with the switch on and off.
    F. WIRED ONCE PER BODY: _rt_exec / _acting_latent / _rt_check_h /
       _rt_credit_step / _rt_flush each called in both stepping bodies, and
       neither body calls policy.store_transition directly any more.
    G. CreditDelay unit cases: primitives, an option of tau 3, a death
       mid-option and after a primitive, a restart, a gap, a flush.
    H. LEGACY REPLAY: a lagged real-time buffer is relabelled on load
       (new[t] = old[t-1], no-op after a done, restart carried, row 0 no-op);
       an executed-stamped buffer and a synchronous buffer load unchanged.

Run: PYTHONPATH=. python tests/_realtime_credit_smoke.py
     RTC_FALSIFY=1 ... runs the same checks with the fix OFF (A-D must FAIL).
"""
import inspect
import logging
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml

logging.disable(logging.WARNING)

NAME = "realtime-credit"
FALSIFY = bool(os.environ.get("RTC_FALSIFY"))
TMP = tempfile.mkdtemp(prefix="rtcredit_smoke_")
K = 4                        # actions
TARGET = 2
FAILS = []
N = [0]


def check(cond, msg):
    N[0] += 1
    print(f"  {N[0]}. {'ok   ' if cond else 'FAIL '} {msg}")
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------ fake envs
def _fake_env_cls():
    import gymnasium as gym

    class FakeLagEnv(gym.Env):
        """step(a) returns a frame made by the PREVIOUS held action, encoded
        one-hot in obs[:K] (the Doom clock's semantics). `realtime=False`
        gives the synchronous twin: the frame shows `a` itself."""
        metadata = {}

        def __init__(self, idx, realtime=True):
            self.idx = idx
            self.realtime = bool(realtime)
            self.observation_space = gym.spaces.Box(-5, 5, (8,), np.float32)
            self.action_space = gym.spaces.Discrete(K)
            self._held = 0
            self._t = 0
            self._rng = np.random.default_rng(100 + idx)

        @property
        def held_action(self):
            return int(self._held)

        def _obs(self, shown):
            o = np.zeros(8, np.float32)
            o[shown] = 1.0
            o[K:] = self._rng.normal(0, 0.1, 8 - K).astype(np.float32)
            return o

        def reset(self, **kw):
            self._held = 0
            return self._obs(0), {"held_action": 0}

        def step(self, a):
            a = int(a)
            self._t += 1
            if self.realtime:
                ex, self._held = self._held, a      # the PREVIOUS action ran
            else:
                ex = a
            term = (self._t % 53 == 0)
            info = {"executed_action": ex, "executed_purity": 1.0,
                    "held_action": a}
            LAST_EX[self.idx] = ex
            return self._obs(ex), 0.0, term, False, info

    return FakeLagEnv


LAST_EX = {}


def _agent(realtime=True, rtc=True, seed=0):
    import developmental_ai.core.developmental_loop as dl
    from developmental_ai.environments.wrappers import DevelopmentalEnvWrapper
    FakeLagEnv = _fake_env_cls()
    state = {"n": 0}

    def fake(*a, **k):
        idx = state["n"]
        state["n"] += 1
        return DevelopmentalEnvWrapper(FakeLagEnv(idx, realtime),
                                       normalize_obs=False), None
    dl.make_env = fake
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["seed"] = seed
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 32, "encoder_hidden": 32,
        "decoder_hidden": 32, "batch_size": 4, "sequence_length": 8,
        "train_iters": 1, "buffer_capacity": 4000})
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["policy"].update({"n_steps": 40, "minibatch_size": 16, "n_epochs": 1,
                          "arch": "rssm", "scouts_in_ppo": True})
    cfg["lifelong"] = {"enabled": True, "forever": False, "segment_len": 120,
                       "wm_train_every": 10 ** 6, "goal_horizon": 10 ** 9,
                       "state_decay": 1.0, "reset_on_death_only": True,
                       "stop_file": os.path.join(TMP, "STOP")}
    cfg.setdefault("loop", {})["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = os.path.join(TMP, f"sb{seed}")
    cfg["metrics"] = {"enabled": False}
    cfg.pop("diagnostics", None)
    cfg["realtime_credit"] = {"enabled": bool(rtc), "policy_lookahead": True,
                              "ppo_delay": True, "legacy_replay": "relabel",
                              "bootstrap_proprio": True}
    ai = dl.DevelopmentalAI(config=cfg)
    ai._ensure_parallel_envs()
    return ai


def _instrument(ai):
    """Record PPO rows, curiosity/observe actions; pay 1.0 iff the executed
    action of the interval just observed is TARGET (both channels)."""
    rows, cur, obs_act = [], [], []
    real_store = ai.policy.store_transition

    def store(*a, **kw):
        if a:
            keys = ["obs", "action", "reward", "done", "log_prob", "value"]
            kw = dict(zip(keys, a), **kw)
        rows.append({"stream": int(kw.get("stream", 0)),
                     "action": int(kw["action"]), "reward": float(kw["reward"]),
                     "tau": int(kw.get("tau", 1) or 1),
                     "done": bool(kw["done"])})
        return real_store(**kw)
    ai.policy.store_transition = store
    ai.policy.train_step = lambda *a, **k: {}          # keep rows for checks
    ai.policy.should_update = lambda *a, **k: False

    real_cur = ai._curiosity_reward

    def cur_reward(obs_t, action, next_obs_t, **kw):
        cur.append((action.detach().cpu().numpy().argmax(1).copy(),
                    next_obs_t.detach().cpu().numpy()[:, :K].argmax(1).copy()))
        return real_cur(obs_t, action, next_obs_t, **kw)
    ai._curiosity_reward = cur_reward

    rssm = ai.world_model.rssm
    real_obs = rssm.observe_step

    def observe(prev, action, emb):
        obs_act.append(action.detach().cpu().numpy().argmax(1).copy())
        return real_obs(prev, action, emb)
    rssm.observe_step = observe

    ai.reward_mixer.mix = lambda i, e: 1.0 if LAST_EX.get(0) == TARGET else 0.0
    ai._scout_mixed_reward = (lambda e, info, r, i, **kw:
                              1.0 if (info or {}).get("executed_action") == TARGET
                              else 0.0)
    return rows, cur, obs_act


def _replay_rows(ai, e):
    s = ai.replay_buffer.streams[e]
    n = s.size
    acts = np.asarray(s.actions[:n]).argmax(1)
    obs = np.asarray(s.observations[:n], np.float32)[:, :K].argmax(1)
    dn = np.asarray(s.dones[:n]) > 0.5
    rs = np.asarray(s.restarts[:n]).astype(bool)
    return acts, obs, dn, rs


# ------------------------------------------------------------- A..D
def test_live():
    print(f"A-D. live body on a lagging real-time fleet (fix {'OFF' if FALSIFY else 'ON'})")
    ai = _agent(realtime=True, rtc=not FALSIFY)
    rows, cur, obs_act = _instrument(ai)
    try:
        ai._collect_segment()
    finally:
        _close(ai)
    for e in (0, 1):
        acts, nxt, dn, rs = _replay_rows(ai, e)
        ok = [t for t in range(len(acts) - 1)
              if not dn[t] and not rs[t] and not rs[t + 1]]
        m = np.mean([acts[t] == nxt[t + 1] for t in ok]) if ok else 0.0
        check(len(ok) > 50 and m >= 0.99,
              f"A. stream {e}: stored action == the interval's cause on "
              f"{100 * m:.1f}% of {len(ok)} contiguous rows")
    hits = np.concatenate([a == b for a, b in cur]) if cur else np.zeros(1)
    check(cur and hits.mean() >= 0.99,
          f"B. curiosity got the executed action on {100 * hits.mean():.1f}% "
          f"of {hits.size} stream-steps")
    # one extra observe_step precedes the first step (the stream's start),
    # so align on the last len(cur) calls
    tail = obs_act[-len(cur):] if cur else []
    same = (len(tail) == len(cur) and all(
        np.array_equal(cur[i][0], tail[i]) for i in range(len(cur))))
    check(same, "B. observe_step got the same (executed) actions as curiosity")
    r1 = [r for r in rows if r["tau"] == 1]
    agree = np.mean([(r["reward"] == 1.0) == (r["action"] == TARGET)
                     for r in r1]) if r1 else 0.0
    pos = sum(r["reward"] == 1.0 for r in r1)
    check(len(r1) > 100 and agree == 1.0 and pos > 5,
          f"C. reward==1 <=> own action==TARGET on {100 * agree:.1f}% of "
          f"{len(r1)} PPO rows ({pos} paid)")
    hm = ai._rt_hmatch
    check(hm[1] > 50 and hm[0] == hm[1],
          f"D. lookahead h == next posterior h on {hm[0]}/{hm[1]} intervals")
    return ai


def _close(ai):
    try:
        ai.close()
    except Exception:
        pass


# ------------------------------------------------------------------ E
def test_sync_identical():
    print("E. a synchronous fleet is byte-identical with the switch on and off")
    out = []
    for rtc in (True, False):
        ai = _agent(realtime=False, rtc=rtc, seed=7)
        rows, _, _ = _instrument(ai)
        try:
            ai._collect_segment()
        finally:
            _close(ai)
        rep = []
        for e in (0, 1):
            s = ai.replay_buffer.streams[e]
            n = s.size
            rep.append((np.asarray(s.observations[:n]).tobytes(),
                        np.asarray(s.actions[:n]).tobytes(),
                        np.asarray(s.rewards[:n]).tobytes(),
                        np.asarray(s.dones[:n]).tobytes()))
        out.append((rep, rows, ai._rt_credit, ai._rt_lookahead_n))
    (rep_on, rows_on, cred_on, la_on), (rep_off, rows_off, _, _) = out
    check(rep_on == rep_off, "E. replay observations/actions/rewards/dones identical")
    check(rows_on == rows_off and len(rows_on) > 100,
          f"E. PPO rows identical ({len(rows_on)} rows)")
    check(cred_on is None and la_on == 0,
          "E. with the switch on, no credit delay and no lookahead engaged")


# ------------------------------------------------------------------ F
def test_wiring():
    print("F. wired once per stepping body")
    import developmental_ai.core.developmental_loop as dl
    D = dl.DevelopmentalAI
    bodies = {"_collect_segment": inspect.getsource(D._collect_segment),
              "_run_episode_parallel": inspect.getsource(D._run_episode_parallel)}
    for call in ("self._rt_exec(", "self._acting_latent(", "self._rt_check_h(",
                 "self._rt_credit_step(", "self._rt_flush()"):
        c = {k: v.count(call) for k, v in bodies.items()}
        check(all(v == 1 for v in c.values()), f"F. {call} once in each body {c}")
    raw = {k: v.count("self.policy.store_transition(") for k, v in bodies.items()}
    check(all(v == 0 for v in raw.values()),
          f"F. no direct policy.store_transition in either body {raw}")


# ------------------------------------------------------------------ G
def test_credit_delay():
    print("G. CreditDelay cases")
    from developmental_ai.core.realtime_credit import CreditDelay
    g = 0.9
    c = CreditDelay(g)
    # primitives: row at k paid m[k+1]
    c.submit(0, {"action": 1, "reward": 9.0, "done": False, "tau": 1})
    o0 = c.step_end(0, 1.0, False, False)        # k=0 -> pending
    c.submit(0, {"action": 2, "reward": 9.0, "done": False, "tau": 1})
    o1 = c.step_end(0, 2.0, False, False)        # k=1 pays row0 with m1=2
    check(o0 == [] and len(o1) == 1 and o1[0]["reward"] == 2.0
          and o1[0]["action"] == 1 and o1[0]["tau"] == 1,
          "G. a primitive is paid the NEXT interval's reward")
    # option tau 3 emitted at k=4 (decisions k=2..4) -> paid m3,m4,m5
    c = CreditDelay(g)
    for k, m in enumerate([0.0, 0.0, 0.0, 1.0, 2.0]):
        if k == 4:
            c.submit(0, {"action": 9, "reward": -1.0, "done": False, "tau": 3})
        c.step_end(0, m, False, False)
    o = c.step_end(0, 4.0, False, False)
    want = 1.0 + g * 2.0 + g * g * 4.0
    check(len(o) == 1 and abs(o[0]["reward"] - want) < 1e-9 and o[0]["tau"] == 3,
          f"G. option tau 3: R = m3 + g m4 + g^2 m5 = {want:.3f}")
    # death mid-option: option k0=1..3 emitted at the death step k=3
    c = CreditDelay(g)
    for k, m in enumerate([0.0, 0.0, 1.0]):
        c.step_end(0, m, False, False)
    c.submit(0, {"action": 9, "reward": 0.0, "done": True, "tau": 3})
    o = c.step_end(0, 5.0, True, False)
    check(len(o) == 1 and abs(o[0]["reward"] - (1.0 + g * 5.0)) < 1e-9
          and o[0]["done"] and o[0]["tau"] == 2,
          "G. death mid-option: paid its observed intervals, done, tau 2")
    # death right after a primitive: the pending row gets done; the new
    # primitive emitted at the death step never ran -> dropped
    c = CreditDelay(g)
    c.submit(0, {"action": 1, "reward": 0.0, "done": False, "tau": 1})
    c.step_end(0, 0.0, False, False)
    c.submit(0, {"action": 3, "reward": 0.0, "done": True, "tau": 1})
    o = c.step_end(0, -1.0, True, False)
    check(len(o) == 1 and o[0]["action"] == 1 and o[0]["done"]
          and o[0]["reward"] == -1.0 and c.dropped == 1,
          "G. death: the previous action owns it; the unexecuted one is dropped")
    # restart: the interval was never observed -> nobody paid for it
    c = CreditDelay(g)
    c.submit(0, {"action": 1, "reward": 0.0, "done": False, "tau": 1})
    c.step_end(0, 0.0, False, False)
    o = c.step_end(0, 7.0, False, True)
    check(o == [] and c.dropped == 1, "G. restart: the unobserved interval pays no one")
    # gap + flush
    c = CreditDelay(g)
    c.submit(0, {"action": 1, "reward": 0.0, "done": False, "tau": 1})
    c.step_end(0, 0.0, False, False)
    fl = c.flush(0)
    c.gap(0)
    c.submit(0, {"action": 2, "reward": 0.0, "done": False, "tau": 1})
    c.step_end(0, 3.0, False, False)
    o = c.step_end(0, 4.0, False, False)
    check(fl == [] and len(o) == 1 and o[0]["reward"] == 4.0,
          "G. flush drops an unobserved primitive; nothing is paid across a gap")


# ------------------------------------------------------------------ H
def test_legacy_replay():
    print("H. a lagged legacy real-time buffer is relabelled on load")
    from developmental_ai.world_model.replay_buffer import ReplayBuffer
    rng = np.random.default_rng(3)
    T = 300
    chosen = rng.integers(0, K, T)
    dones = np.zeros(T, bool)
    dones[[50, 120]] = True
    src = ReplayBuffer(capacity=1000, obs_dim=8, action_dim=K)
    for t in range(T):
        src.add(np.zeros(8, np.float32), int(chosen[t]), 0.0, bool(dones[t]),
                restart=(t == 200))
    p = os.path.join(TMP, "lagged")
    src.save(p)                                        # unstamped = legacy

    def load(sem, pol):
        b = ReplayBuffer(capacity=1000, obs_dim=8, action_dim=K)
        b.action_semantics, b.legacy_lag_policy = sem, pol
        n = b.load(p)
        return b, n
    b, n = load("executed", "relabel")
    a = np.asarray(b.actions[:n]).argmax(1)
    r = np.asarray(b.restarts[:n]).astype(bool)
    want = np.empty(T, int)
    want[0] = 0
    want[1:] = chosen[:-1]
    want[[51, 121]] = 0                                # after a done: no-op
    check(n == T and np.array_equal(a, want),
          "H. new[t] = old[t-1], no-op after a done, row 0 no-op")
    check(r[0] and r[200] and r[201] and r.sum() == 3,
          "H. restart flags: row 0, the restart row, and the row after it")
    b2, n2 = load(None, "relabel")                     # a synchronous stream
    check(np.array_equal(np.asarray(b2.actions[:n2]).argmax(1), chosen),
          "H. a synchronous stream's buffer loads unchanged")
    src.action_semantics = "executed"
    p2 = os.path.join(TMP, "stamped")
    src.save(p2)
    b3 = ReplayBuffer(capacity=1000, obs_dim=8, action_dim=K)
    b3.action_semantics, b3.legacy_lag_policy = "executed", "relabel"
    n3 = b3.load(p2)
    check(np.array_equal(np.asarray(b3.actions[:n3]).argmax(1), chosen),
          "H. an executed-stamped buffer is never relabelled")
    b4, n4 = load("executed", "drop")
    check(n4 == 0, "H. legacy_replay: drop restores nothing")


def main():
    try:
        test_live()
        if not FALSIFY:
            test_sync_identical()
            test_wiring()
            test_credit_delay()
            test_legacy_replay()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print(f"[{NAME}] FAILED ({len(FAILS)})")
        sys.exit(1)
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
