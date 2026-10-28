"""The autoscaler: world-model size x stream count, recommendations only.

Born 2026-10-17 (docs/research/res-004-rssm-size-experiment.md, "the pyramid"). The user's
design: many small SkyBots early, a bigger brain with fewer streams later,
and back down when a task needs less — one brain throughout. The decision
engine (developmental_ai/infra/autoscaler.py) is encoded here because each
rule is a way the scaler could otherwise do harm:

    A. A COLLAPSED LATENT HOLDS EVERYTHING. The live world model carried
       0.00-0.01 nats per frame (measured on main, 2026-10-17) and every size
       scored alike; scaling such a model measures nothing. The hold names
       its release.
    B. BUDGETS WIN: acting slower than the game's budget, or VRAM over the
       cap, shrinks the model — or, at the smallest model, drops a stream.
    C. PATIENCE: no decision without `window` segments of fresh-data loss,
       none while still improving, none inside the cooldown.
    D. OUT OF ROOM -> GROW: plateaued, fresh ~= train (underfitting), novelty
       still arriving -> grow the model AND drop one stream (the pyramid).
    E. OUT OF DATA -> MORE STREAMS: plateaued while memorising (fresh >> train,
       or each frame reused too often); at the stream maximum, shrink instead.
    F. HYSTERESIS: between gap_up and gap_down nothing moves; gap_up must be
       below gap_down or the config is refused.
    G. NO LATCH (§4.1): every hold names what releases it; the cooldown always
       expires; at both bounds a recommendation falls through instead of
       repeating; a ladder walk down then up returns to where it started.
    G2. ADVICE IS NOT ACTION: with actuate off (the live default) a
       recommendation leaves the state at the brain actually running, and
       repeats after the cooldown. Stage C (2026-10-18) logged "live x 3 ->
       live x 4" from a live x 2 run because the state advanced anyway.
    H. WIRED: the loop's _autoscale_signals measures the fresh-data loss and
       z-info on each stream's NEWEST window (never yet sampled for training),
       the training loss, and the replay ratio from the frames-trained counter;
       the infra stack turns them into an `infra/autoscale:` line, an action,
       and an atomically written autoscale.json.

Run: PYTHONPATH=. python tests/_autoscaler_smoke.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.infra.autoscaler import (FEWER_STREAMS, GROW, HOLD,     # noqa: E402
                                               MORE_STREAMS, SHRINK, Autoscaler,
                                               Signals)

NAME = "autoscaler"
N = [0]
CFG = {"window": 4, "cooldown_segments": 3, "plateau_rel": 0.02,
       "gap_up": 1.15, "gap_down": 1.5, "z_min_nats": 1.0,
       "min_streams": 2, "max_streams": 8}


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def feed(a, n, **kw):
    """Feed n identical segments; return the FIRST non-hold decision (a change
    is followed by its cooldown), else the last hold."""
    d, first = None, None
    for i in range(n):
        base = dict(step=i, wm_fresh=0.03, wm_train=0.03, z_info=8.0, novelty=1.0)
        base.update(kw)
        d = a.observe(Signals(**base))
        if first is None and d.action != HOLD:
            first = d
    return first or d


def main():
    # ---------------------------------------------------------------- A
    a = Autoscaler(CFG, size="medium", streams=4)
    d = feed(a, 10, z_info=0.01, wm_fresh=0.03, wm_train=0.03)
    check(d.action == HOLD and "collapsed" in d.reason and "nats" in d.releases_when,
          "A. collapsed latent (0.01 nats) holds, and names its release")
    d = a.observe(Signals(step=99, z_info=0.01, act_s=1.0, act_budget_s=0.03))
    check(d.action == HOLD, "A. collapse outranks even a blown latency budget")

    # ---------------------------------------------------------------- B
    a = Autoscaler(CFG, size="live", streams=2)
    d = a.observe(Signals(step=1, z_info=8.0, act_s=0.05, act_budget_s=0.0286))
    check(d.action == SHRINK and d.target_size == "medium",
          "B. acting 50 ms vs Doom's 28.6 ms budget -> shrink live -> medium")
    a = Autoscaler(CFG, size="tiny", streams=4)
    d = a.observe(Signals(step=1, z_info=8.0, vram_frac=0.95))
    check(d.action == FEWER_STREAMS and d.target_streams == 3,
          "B. VRAM over the cap at the smallest model -> one stream fewer")
    a = Autoscaler(CFG, size="live", streams=2)
    check(a.observe(Signals(step=1, z_info=8.0, act_s=0.01, act_budget_s=None)).action == HOLD,
          "B. a game with no latency budget never shrinks on latency")

    # ---------------------------------------------------------------- C
    a = Autoscaler(CFG, size="medium", streams=4)
    d = feed(a, 2)
    check(d.action == HOLD and "history" in d.reason, "C. too little history -> hold")
    a = Autoscaler(CFG, size="medium", streams=4)
    for i, f in enumerate([0.05, 0.045, 0.04, 0.035, 0.03, 0.025]):
        d = a.observe(Signals(step=i, wm_fresh=f, wm_train=f, z_info=8.0, novelty=1.0))
    check(d.action == HOLD and "improving" in d.reason, "C. still improving -> hold")

    # ---------------------------------------------------------------- D
    a = Autoscaler(CFG, size="small", streams=8)
    d = feed(a, 6, wm_fresh=0.030, wm_train=0.029)
    check(d.action == GROW and d.target_size == "medium" and d.target_streams == 7,
          "D. plateau + fresh~train + novelty -> grow small->medium, 8->7 streams (pyramid)")
    a = Autoscaler(CFG, size="small", streams=8)
    d = feed(a, 6, wm_fresh=0.030, wm_train=0.029, novelty=0.0)
    check(d.action == HOLD and "no novelty" in d.reason,
          "D. no novelty arriving -> do not grow (nothing new to make room for)")
    a = Autoscaler(CFG, size="live", streams=2)
    d = feed(a, 6, wm_fresh=0.030, wm_train=0.029)
    check(d.action == HOLD and "largest" in d.reason and d.releases_when,
          "D. already the largest -> hold, released by raising max_size")

    # ---------------------------------------------------------------- E
    a = Autoscaler(CFG, size="medium", streams=4)
    d = feed(a, 6, wm_fresh=0.06, wm_train=0.03)
    check(d.action == MORE_STREAMS and d.target_streams == 5,
          "E. memorising (fresh/train 2.0) -> more streams")
    a = Autoscaler(CFG, size="medium", streams=4)
    d = feed(a, 6, wm_fresh=0.03, wm_train=0.03, replay_ratio=200.0)
    check(d.action == MORE_STREAMS, "E. each frame reused 200x -> more streams")
    a = Autoscaler(CFG, size="medium", streams=8)
    d = feed(a, 6, wm_fresh=0.06, wm_train=0.03)
    check(d.action == SHRINK and d.target_size == "small",
          "E. memorising at the stream maximum -> shrink the model instead")
    # main 2026-10-18: memorising at tiny x 8 became "tiny x 8 -> tiny x 7"
    # (the budget fallback). Fewer streams is less data = more memorising.
    a = Autoscaler(CFG, size="tiny", streams=8)
    d = feed(a, 6, wm_fresh=0.06, wm_train=0.03)
    check(d.action == HOLD and "smallest" in d.reason and d.releases_when,
          "E. memorising at the stream maximum AND the smallest model -> hold, "
          "never fewer streams")
    # the gap is the WINDOW's, not one segment's: alternating 0.9 / 1.7
    # (mean ~1.3, inside the band) with a 1.72 last segment does not act
    a = Autoscaler(CFG, size="medium", streams=4)
    for i in range(CFG["window"] - 1):                # gaps 0.9, 1.7, 0.9
        d = a.observe(Signals(step=i, wm_fresh=0.051 if i % 2 else 0.027,
                              wm_train=0.03, z_info=8.0, novelty=1.0))
    d = a.observe(Signals(step=99, wm_fresh=0.0516, wm_train=0.03, z_info=8.0, novelty=1.0))
    check(d.action == HOLD, "E. one noisy segment (gap 1.72) inside a window "
          "averaging ~1.3 does not move anything (%s)" % d.action)

    # ---------------------------------------------------------------- F
    a = Autoscaler(CFG, size="medium", streams=4)
    d = feed(a, 6, wm_fresh=0.036, wm_train=0.03)                 # gap 1.2
    check(d.action == HOLD and "hysteresis" in d.reason,
          "F. fresh/train 1.2 is inside [1.15, 1.5] -> hold")
    try:
        Autoscaler(dict(CFG, gap_up=1.6, gap_down=1.5))
        refused = False
    except ValueError:
        refused = True
    check(refused, "F. gap_up >= gap_down is refused (no hysteresis band)")

    # ---------------------------------------------------------------- G
    a = Autoscaler(CFG, size="small", streams=8)
    g = feed(a, CFG["window"], wm_fresh=0.030, wm_train=0.029)    # the 4th segment grows
    check(g.action == GROW, "G. (setup) the window-th segment recommends a grow")
    d = a.observe(Signals(step=50, wm_fresh=0.03, wm_train=0.029, z_info=8.0, novelty=1.0))
    check(d.action == HOLD and "cooling" in d.reason and "segment" in d.releases_when,
          "G. right after a change: cooling down, with the segments left")
    released = False
    for i in range(CFG["cooldown_segments"] + CFG["window"] + 2):
        d = a.observe(Signals(step=60 + i, wm_fresh=0.03, wm_train=0.029, z_info=8.0,
                              novelty=1.0))
        released = released or d.action != HOLD or "cooling" not in d.reason
    check(released, "G. the cooldown expires")
    a = Autoscaler(dict(CFG, cooldown_segments=0), size="tiny", streams=2)
    d = a.observe(Signals(step=1, z_info=8.0, vram_frac=0.99))
    check(d.action == HOLD and d.releases_when,
          "G. smallest model at the fewest streams: hold with a release, not a loop")
    ups = Autoscaler(dict(CFG, cooldown_segments=0, window=2, actuate=True),
                     size="tiny", streams=8)
    path = []
    for i in range(40):
        d = ups.observe(Signals(step=i, wm_fresh=0.03, wm_train=0.029, z_info=8.0,
                                novelty=1.0))
        if d.action != HOLD:
            path.append(d.target_size)
    check(path[:3] == ["small", "medium", "live"] and ups.state()["size"] == "live",
          "G. sustained 'out of room' walks the ladder up and stops at the top: %s" % path)
    downs = Autoscaler(dict(CFG, cooldown_segments=0, actuate=True), size="live", streams=2)
    for i in range(10):
        downs.observe(Signals(step=i, z_info=8.0, act_s=0.05, act_budget_s=0.0286))
    check(downs.state()["size"] == "tiny",
          "G. sustained over-budget walks it back down to the floor (no stuck state)")
    # ---------------------------------------------------------------- G2
    adv = Autoscaler(dict(CFG, cooldown_segments=2, window=2), size="live", streams=2)
    recs = []
    for i in range(12):
        d = adv.observe(Signals(step=i, wm_fresh=0.03, wm_train=0.01, z_info=80.0,
                                novelty=1.0))
        if d.action != HOLD:
            recs.append((d.streams, d.target_streams))
    check(len(recs) >= 2 and all(r == (2, 3) for r in recs)
          and adv.state()["size"] == "live" and adv.streams == 2,
          "G2. advice mode: state stays live x 2, the same 2 -> 3 advice repeats: %s" % recs)
    wired()
    print(f"[{NAME}] ALL PASS")


def wired():
    import json
    import tempfile
    import threading
    import types

    import numpy as np
    import torch

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.infra.stack import InfraStack
    from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer
    from developmental_ai.world_model.rssm import WorldModel

    torch.manual_seed(0)
    wm = WorldModel(obs_dim=12, action_dim=4, stochastic_size=4, stochastic_classes=4,
                    deterministic_size=32, hidden_dim=32)
    rb = MultiStreamReplayBuffer(num_streams=2, capacity=4000, obs_dim=12, action_dim=4)
    for t in range(60):
        for st in range(2):
            rb.add(np.random.rand(12).astype(np.float32), np.eye(4)[t % 4], 0.0, False,
                   stream=st)
    fake = types.SimpleNamespace(
        config={"world_model": {"sequence_length": 8}}, replay_buffer=rb, world_model=wm,
        device=torch.device("cpu"), _wm_param_lock=threading.RLock(), total_timesteps=1000,
        _wm_frames_trained=0, _last_wm_metrics={"reconstruction": 0.05}, _num_envs=2)
    s1 = DevelopmentalAI._autoscale_signals(fake, intrinsic_per_step=0.01)
    check(s1.wm_fresh is not None and np.isfinite(s1.wm_fresh) and s1.z_info is not None
          and s1.z_info >= 0.0 and s1.wm_train == 0.05 and s1.novelty == 0.01,
          "H. fresh loss %.4f and z-info %.3f nats from the newest windows; train loss read"
          % (s1.wm_fresh, s1.z_info))
    fake.total_timesteps, fake._wm_frames_trained = 1100, 100 * 256
    s2 = DevelopmentalAI._autoscale_signals(fake)
    check(s2.replay_ratio == 256.0 and s2.act_s is not None,
          "H. replay ratio = frames trained / frames collected (%.0f)" % s2.replay_ratio)
    with tempfile.TemporaryDirectory() as d:
        stack = InfraStack({"log_dir": d, "autoscaler": dict(CFG, enabled=True)}, action_dim=4)
        stack.configure_autoscaler(2048, 2)
        check(stack.autoscaler.state()["size"] == "live" and stack.autoscaler.streams == 2,
              "H. the stack places the autoscaler on the built model's rung (2048 -> live)")
        lines, acts = stack.segment({"step": 1, "autoscale": s1})
        rec = json.load(open(f"{d}/autoscale.json"))
        check(any(l.startswith("  infra/autoscale: ") for l in lines)
              and acts.get("autoscale", {}).get("action") and rec["decision"]["action"],
              "H. segment() prints the decision, returns it as an action, writes autoscale.json")
        off = InfraStack({"log_dir": d}, action_dim=4)
        check(off.autoscaler is None, "H. off unless autoscaler.enabled (legacy brain unchanged)")
    # G3. THE COOLDOWN SURVIVES THE RESTART THAT APPLIED THE CHANGE (main,
    # 2026-10-18: every launch reset it, so an actuated change's own restart
    # erased its cooldown).
    with tempfile.TemporaryDirectory() as d:
        acfg = dict(CFG, enabled=True, actuate=True, cooldown_segments=10)
        st = InfraStack({"log_dir": d, "autoscaler": acfg}, action_dim=4)
        st.configure_autoscaler(256, 4)                       # tiny x 4
        for i in range(CFG["window"] + 1):
            _, acts = st.segment({"step": i, "autoscale": Signals(
                step=i, wm_fresh=0.06, wm_train=0.03, z_info=8.0, novelty=1.0)})
            if acts.get("autoscale", {}).get("action") not in (None, HOLD):
                break
        moved = acts["autoscale"]["action"]
        st2 = InfraStack({"log_dir": d, "autoscaler": acfg}, action_dim=4)
        st2.configure_autoscaler(256, 5)                      # relaunched at x 5
        held = []
        for i in range(CFG["window"] + 2):
            _, a2 = st2.segment({"step": 100 + i, "autoscale": Signals(
                step=100 + i, wm_fresh=0.06, wm_train=0.03, z_info=8.0, novelty=1.0)})
            held.append(a2["autoscale"]["action"])
        check(moved == MORE_STREAMS and all(h == HOLD for h in held)
              and "cooling" in a2["autoscale"]["reason"],
              "G3. after a relaunch the cooldown continues (%d holds, still cooling)"
              % len(held))


if __name__ == "__main__":
    main()
