"""PPO fixes (2026-10-18): why the live Doom policy stayed ~97 % uniform.

THE LIVE FINDING THIS ENCODES

    Every Doom run on `main` (Stage C, the mock deploy, the scale test) ended
    with the policy as random as it began: entropy 2.68-2.73 of a printed
    3.87 "max". Code investigation found, in this file's scope:
      * the trust region was a per-32-row-minibatch k3 test against
        target_kl 0.05 — the FIRST noisy 32-row estimate above it froze the
        actor, so a segment moved ~0.02 nats ("KL-STOPPED" every update);
      * the logit-range bound (5.0) measured its spread over all 72 head
        columns, 56 of them masked on every Doom row;
      * the reward mixer annealed weight from curiosity to an extrinsic
        reward Doom never pays (0.695 -> 0.649 in 10 segments, toward 0);
      * dream distillation took an Adam step every segment at eff_w 0.0016
        (Adam cancels the scale) over all 48 primitive slots;
      * "% of max" divided by ln 48 although a Doom row has 16 valid actions;
      * nothing measured whether the policy's action depends on the state.

Contracts:
    A. TRUST REGION ON THE ROLLOUT (kl_check: rollout). Same data, seed and
       minibatch order (the KL trajectories are identical until a freeze):
       minibatch mode freezes the actor while the WHOLE-rollout KL is still
       below target_kl (the measured fault: it stops on an 8-row estimate);
       rollout mode does not freeze when the whole rollout never breaches
       the target, and with a target it does breach it freezes at the first
       check past it, reporting where (kl_trip_epoch / kl_trip_update).
       Without target_kl both modes are identical.
    B. THE BOUND COUNTS VALID ACTIONS ONLY (logit_bound_valid_only). With
       extreme logits on masked columns the old bound compresses the valid
       actions toward uniform; the new one leaves a valid spread under the
       range untouched. Acting and training compute the same log-prob in
       both the masked and the unmasked path.
    C. NO TASK, NO ANNEAL (hold_without_extrinsic). With no extrinsic reward
       ever earned, decay() leaves the weights unchanged (old mode: it
       anneals toward 0); after the first extrinsic reward it anneals as
       before. The mixer state round-trips through state_dict.
    D. DISTILLATION: below distill_min_weight no Adam step is taken (the
       actor is bit-identical); with distill_masked the head rows of
       primitive slots no stream can execute get exactly no update (old
       mode: they move).
    E. I(S;A), MEASURED ONLY. A policy whose logits depend on the state has
       I(S;A) > 0; one with the same (non-uniform) logits everywhere has
       I ~ 0. The fixed probe is captured once and re-read at every update
       (KL from the first policy 0 at capture, > 0 after learning). Learner
       state (weights, Adam, every RNG) is byte-identical with the stats on
       vs off.
    F. The readout uses the valid-action count (16 of 48 -> ln 16), and the
       loop's trust-region line prints the trip point, EV, clip and I(S;A).

Run: PYTHONPATH=. python tests/_ppo_fixes_smoke.py
"""
import copy
import os
import random
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import developmental_ai.policy.actor_critic as AC                       # noqa: E402
from developmental_ai.policy.actor_critic import (                       # noqa: E402
    DreamActorCritic, RewardMixer, StandaloneActorCritic)

NAME = "ppo-fixes"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


OBS, ACT = 16, 8


def pol(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(obs_dim=OBS, action_dim=ACT, hidden_dim=32, device="cpu",
                entropy_coef=0.0, learning_rate=3e-3)
    base.update(kw)
    return StandaloneActorCritic(**base)


def fill(p, n=512, seed=0, mask=None):
    """A rollout whose advantage depends on the action, so PPO moves."""
    g = np.random.default_rng(seed)
    for t in range(n):
        o = g.random(OBS).astype(np.float32)
        a, info = p.select_action(o, action_mask=mask)
        r = 1.0 if a == int(o[0] * ACT) % ACT else 0.0
        p.store_transition(o, a, r, t == n - 1, info["log_prob"], info["value"],
                           action_mask=mask)


def snapshot(p):
    return (np.stack(p.rollout_obs).astype(np.float32),
            np.asarray(p.rollout_actions), np.asarray(p.rollout_log_probs, np.float32))


def rollout_kl(p, snap):
    o, a, lp0 = snap
    lp = p._rollout_logp(torch.from_numpy(o), None, None, None, None)
    nl = lp.gather(-1, torch.as_tensor(a).reshape(-1, 1)).reshape(-1)
    lr = nl - torch.from_numpy(lp0)
    return float(((lr.exp() - 1.0) - lr).mean())


def test_kl_check():
    print("A. trust region on the rollout")
    def run(mode, target):
        p = pol(seed=3, minibatch_size=8, target_kl=target, kl_check=mode,
                kl_check_every=4)
        fill(p, seed=1)
        snap = snapshot(p)
        torch.manual_seed(7)
        m = p.train_step(n_epochs=4)
        return m, rollout_kl(p, snap)
    mb = run("minibatch", 0.02)
    check(mb[0]["kl_stopped"] and mb[1] < 0.02,
          "A. minibatch mode froze on noise: whole-rollout KL at the freeze "
          "%.4f < target 0.02 (tripped at update %s)"
          % (mb[1], mb[0]["kl_trip_update"]))
    check(mb[0]["actor_updates_applied"] == mb[0]["kl_trip_update"]
          and mb[0]["grad_norm_actor_applied_mean"] > 0
          and mb[0]["n_updates"] > mb[0]["actor_updates_applied"],
          "A. the actor grad norm is reduced over the %d steps that updated "
          "the actor (mean %.3f), not the final epoch's critic-only steps"
          % (mb[0]["actor_updates_applied"], mb[0]["grad_norm_actor_applied_mean"]))
    ro = run("rollout", 0.02)
    check(not ro[0]["kl_stopped"] and ro[0]["approx_kl_rollout"] < 0.02,
          "A. rollout mode, same data: the whole rollout never breached 0.02, "
          "so the actor was not frozen (last check %.4f)" % ro[0]["approx_kl_rollout"])
    lo = run("rollout", 0.012)
    check(lo[0]["kl_stopped"] and lo[0]["approx_kl_rollout"] > 0.012
          and lo[0]["kl_trip_update"] is not None and lo[0]["kl_check"] == "rollout",
          "A. with target 0.012 it froze at the first check past it (%.4f at "
          "epoch %s update %s)" % (lo[0]["approx_kl_rollout"],
                                   lo[0]["kl_trip_epoch"], lo[0]["kl_trip_update"]))
    # no target_kl -> the two modes are the same computation
    outs = []
    for mode in ("minibatch", "rollout"):
        p = pol(seed=3, minibatch_size=8, target_kl=0.0, kl_check=mode)
        fill(p, seed=1)
        torch.manual_seed(7)
        p.train_step(n_epochs=2)
        outs.append(copy.deepcopy(p.actor.state_dict()))
    check(all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0]),
          "A. without target_kl both modes produce identical weights")


def test_bound():
    print("B. the bound counts valid actions only")
    mask = np.zeros(ACT, bool)
    mask[:3] = True
    out = {}
    for flag in (False, True):
        p = pol(seed=0, logit_range=2.0, logit_bound_valid_only=flag)
        with torch.no_grad():
            p.actor.action_head.weight.zero_()
            b = torch.tensor([0.0, 0.75, 1.5, 50, -50, 50, -50, 50])
            p.actor.action_head.bias.copy_(b)
        o = np.zeros(OBS, np.float32)
        raw = torch.tensor([[0.0, 0.75, 1.5]])
        want = torch.softmax(raw, -1)
        m_t = torch.as_tensor(mask).reshape(1, -1)
        with torch.no_grad():
            aug = p._augment(p._encode(torch.from_numpy(o).reshape(1, -1)), None)
            lg = p._bound_logits(p.actor.action_head(p.actor.shared(aug)), m_t)
            got = torch.softmax(lg.masked_fill(~m_t, -1e9), -1)[:, :3]
        out[flag] = float((got - want).abs().max())
        # acting log-prob == training log-prob, masked path
        a, info = p.select_action(o, action_mask=mask)
        p.store_transition(o, a, 0.0, False, info["log_prob"], info["value"],
                           action_mask=mask)
        lp = p._rollout_logp(torch.from_numpy(o).reshape(1, -1), None, None, None,
                             torch.as_tensor(mask).reshape(1, -1))
        out[(flag, "lp")] = abs(float(lp[0, a]) - info["log_prob"])
    check(out[False] > 0.05,
          "B. old bound: masked extremes compress the 3 valid actions (|dp| %.3f)"
          % out[False])
    check(out[True] < 1e-6,
          "B. valid-only bound: a valid spread under the range is untouched "
          "(|dp| %.1e)" % out[True])
    check(out[(False, "lp")] < 1e-5 and out[(True, "lp")] < 1e-5,
          "B. acting and training log-probs agree (masked path, both modes)")
    p = pol(seed=0, logit_range=0.5, logit_bound_valid_only=True)
    o = np.random.default_rng(2).random(OBS).astype(np.float32)
    a, info = p.select_action(o)
    lp = p._rollout_logp(torch.from_numpy(o).reshape(1, -1), None, None, None, None)
    check(abs(float(lp[0, a]) - info["log_prob"]) < 1e-5,
          "B. ...and in the unmasked path (it used to skip the bound when acting)")


def test_mixer():
    print("C. no task, no anneal")
    old = RewardMixer(decay_rate=5e-6, min_intrinsic=0.0, adaptive_gating=True)
    new = RewardMixer(decay_rate=5e-6, min_intrinsic=0.0, adaptive_gating=True,
                      hold_without_extrinsic=True)
    for m in (old, new):
        for _ in range(10):
            m.mix(0.02, 0.0)
            m.decay(1024)
    check(old.intrinsic_weight < 0.66,
          "C. old: with extrinsic always 0 the weight anneals (%.3f)"
          % old.intrinsic_weight)
    check(new.intrinsic_weight == 0.7 and new.extrinsic_weight == 0.3,
          "C. new: no extrinsic ever earned -> weights held at 0.7/0.3")
    new.mix(0.02, 1.0)
    new.decay(1024)
    check(new.intrinsic_weight < 0.7, "C. after the first extrinsic reward it anneals")
    st = new.state_dict()
    fresh = RewardMixer(hold_without_extrinsic=True)
    fresh.load_state_dict(st)
    check(fresh.state_dict() == st, "C. mixer state round-trips through state_dict")


class _Stub:
    pass


def distill_stub(masked, min_w, eff=1.0):
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    torch.manual_seed(0)
    lat, P, K = 12, 6, 2
    s = _Stub()
    s.device = torch.device("cpu")
    s.obs_dim = OBS
    s.action_dim = P
    s.meta_action_dim = P + K
    s.policy = StandaloneActorCritic(obs_dim=OBS, action_dim=P + K, hidden_dim=16,
                                     device="cpu", arch="rssm", enc_dim=lat)
    s.dream_actor = DreamActorCritic(lat, P, hidden_dim=16)
    s._distill_opt = torch.optim.Adam(s.policy.actor.parameters(), lr=1e-2)
    s.dream_distill_trust_gate = False
    s.dream_distill_value_gate = False
    s.dream_distill_anneal = False
    s.dream_distill_temp = 0.5
    s.dream_distill_weight = eff
    s.dream_distill_min_weight = min_w
    s.dream_distill_masked = masked
    s._stream_masks = [np.array([1, 1, 1, 0, 0, 0], bool)]
    s._current_knowledge_feature = lambda: None
    s._cur_proprio_t = lambda: None
    s._distill_valid_mask = types.MethodType(DevelopmentalAI._distill_valid_mask, s)
    g = torch.Generator().manual_seed(1)
    B, L = 4, 3

    class _RB:
        def sample_sequences(self, b, l, device=None):
            return {"observations": torch.rand(b, l, OBS, generator=g),
                    "actions": torch.eye(P)[torch.randint(0, P, (b, l), generator=g)]}

    class _RSSM:
        latent_dim = lat

        def get_latent(self, st):
            return torch.cat([st["h"], st["z"]], -1)

    class _WM:
        rssm = _RSSM()

        def observe_sequence(self, o, a, p=None):
            b, l = o.shape[:2]
            return ({"h": torch.randn(b, l, 8, generator=g),
                     "z": torch.randn(b, l, 4, generator=g)}, {})

    s.replay_buffer = _RB()
    s.world_model = _WM()
    return s


def test_distill():
    print("D. distillation")
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    s = distill_stub(masked=True, min_w=0.05, eff=0.0016)
    before = copy.deepcopy(s.policy.actor.state_dict())
    out = DevelopmentalAI._distill_dream_to_real(s, 4, 3)
    check(out.get("dream_distill_skipped") is True
          and all(torch.equal(before[k], s.policy.actor.state_dict()[k]) for k in before),
          "D. eff_w 0.0016 < min 0.05: no Adam step, actor bit-identical")
    moved = {}
    for masked in (False, True):
        s = distill_stub(masked=masked, min_w=0.0, eff=1.0)
        w0 = s.policy.actor.action_head.weight.detach().clone()
        DevelopmentalAI._distill_dream_to_real(s, 4, 3)
        w1 = s.policy.actor.action_head.weight.detach()
        moved[masked] = float((w1[3:6] - w0[3:6]).abs().max())
    check(moved[False] > 0,
          "D. unmasked: the 3 unexecutable primitive rows move (%.1e)" % moved[False])
    check(moved[True] == 0.0,
          "D. masked: unexecutable primitive rows get exactly no update")


def test_info():
    print("E. I(S;A), measured only")
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(256, ACT, generator=g) * 3.0          # state-dependent
    i_dep = AC.StandaloneActorCritic._info_from_probs(torch.softmax(lg, -1), None)
    same = torch.softmax(torch.randn(1, ACT, generator=g) * 3.0, -1).repeat(256, 1)
    i_ind = AC.StandaloneActorCritic._info_from_probs(same, None)
    check(i_dep["mutual_info"] > 0.3,
          "E. state-dependent logits: I(S;A) = %.3f nats > 0" % i_dep["mutual_info"])
    check(abs(i_ind["mutual_info"]) < 1e-5 and i_ind["cond_entropy_norm"] < 0.95,
          "E. same non-uniform logits everywhere: I = %.1e, entropy %.2f of uniform"
          % (i_ind["mutual_info"], i_ind["cond_entropy_norm"]))
    mask = torch.zeros(256, 48, dtype=torch.bool)
    mask[:, :16] = True
    pm = torch.softmax(torch.zeros(256, 48).masked_fill(~mask, -1e9), -1)
    st = AC.StandaloneActorCritic._info_from_probs(pm, mask)
    check(abs(st["n_valid_mean"] - 16) < 1e-9 and abs(st["cond_entropy_norm"] - 1) < 1e-4,
          "F. 16 valid of 48: uniform over them reads 100%% of ln 16 (%.4f)"
          % st["cond_entropy_norm"])

    # the probe: captured once, re-read every update
    p = pol(seed=5, minibatch_size=32)
    fill(p, n=256, seed=2)
    m1 = p.train_step(n_epochs=1)
    probe = p._probe
    check(probe is not None and abs(m1["probe_kl_from_first"]) < 1e-9
          and m1["probe_rows"] > 0,
          "E. probe captured at the first update (%d rows, KL0 = 0 at capture)"
          % m1["probe_rows"])
    lp0 = probe["logp0"].clone()
    fill(p, n=256, seed=3)
    m2 = p.train_step(n_epochs=4)
    check(p._probe is probe and torch.equal(p._probe["logp0"], lp0)
          and m2["probe_kl_from_first"] > 0,
          "E. the probe is fixed across updates and drift from it is measured "
          "(KL0 %.4f)" % m2["probe_kl_from_first"])

    # byte-identical learner state with the stats on vs off
    def run(stats):
        AC.COLLECT_STATS = stats
        try:
            random.seed(0)
            np.random.seed(0)
            q = pol(seed=9, minibatch_size=32, kl_check="rollout", target_kl=0.05)
            fill(q, n=256, seed=4)
            q.train_step(n_epochs=2)
            return (copy.deepcopy(q.actor.state_dict()),
                    copy.deepcopy(q.critic.state_dict()),
                    copy.deepcopy(q.optimizer.state_dict()["state"]),
                    torch.random.get_rng_state().clone(),
                    np.random.get_state()[1].copy(), random.getstate())
        finally:
            AC.COLLECT_STATS = True
    on, off = run(True), run(False)
    same_w = all(torch.equal(on[0][k], off[0][k]) for k in on[0]) and \
        all(torch.equal(on[1][k], off[1][k]) for k in on[1])
    same_opt = all(torch.equal(on[2][i][k], off[2][i][k])
                   for i in on[2] for k in on[2][i] if torch.is_tensor(on[2][i][k]))
    check(same_w and same_opt and torch.equal(on[3], off[3])
          and np.array_equal(on[4], off[4]) and on[5] == off[5],
          "E. weights, Adam state and torch/numpy/python RNGs byte-identical "
          "with the stats on vs off")


def test_wired():
    print("F. wired")
    src = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    pr = src[src.index('"  Policy: entropy "') - 1500:src.index('"  Policy: entropy "')]
    check('_pp0.get("n_valid_mean")' in pr,
          "F. the entropy readout divides by ln(valid actions), not ln(action_dim)")
    tr = src[src.index('print(f"  PPO trust region:') - 2500:
             src.index('print(f"  PPO trust region:') + 1200]
    for k in ("kl_trip_update", "approx_kl_rollout", "explained_variance",
              "clip_frac", "mutual_info", "probe_mutual_info", "probe_kl_from_first",
              "actor_updates_applied"):
        check(k in tr, f"F. the trust-region line prints {k}")


def main():
    test_kl_check()
    test_bound()
    test_mixer()
    test_distill()
    test_info()
    test_wired()
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
