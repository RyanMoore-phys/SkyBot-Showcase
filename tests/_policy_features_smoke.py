"""PPO warm start: resume the feature layers, reset the action head, explore.

Born 2026-10-17 (user request). The policy, dream actor and curiosity are
refused at resume BY NAME, because a converged, near-zero-entropy policy
resumed under a changed reward carried its old habits forward and could not
explore out (CLAUDE.md §10). The user asked for the middle way: keep what the
policy learned about SEEING, discard what it learned about WANTING, and make
it explore while the new head forms.

Contracts:
    A. load_features copies actor.shared (and the conditioner / a conv
       encoder) exactly, and leaves the action head and the critic at their
       fresh initialisation.
    B. A stored policy of another arch or input width is REFUSED before any
       copy (stage D growth or a sensor change changes the width) — the live
       policy is untouched.
    C. The exploration boost multiplies the entropy coefficient by `factor`
       at the first PPO update and decays linearly to exactly 1x after
       `updates`; with no boost the coefficient is the configured one.
    D. A real PPO update uses (and reports) the boosted coefficient.
    E. Wired: `policy_features` is a separate component gated on world_model
       having restored (the features read its latent), ordered after it;
       the whole `policy` is still refused by name; the universal brain
       resumes the full base list plus policy_full (2026-10-18), which
       falls back to this features-only path when the reward fingerprint or
       the shapes changed; the legacy brain is unchanged.
    F. KICKSTARTING (the alternative): the stored actor becomes a frozen
       teacher; PPO updates with the KL term move a fresh student CLOSER to
       the teacher than the same updates without it; the weight decays to 0
       and the teacher is dropped after `updates`; a mismatched teacher is
       refused before anything changes; `policy_kickstart` is gated like
       policy_features.

Run: PYTHONPATH=. python tests/_policy_features_smoke.py
"""
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.policy.actor_critic import StandaloneActorCritic   # noqa: E402

NAME = "policy-features"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def pol(obs_dim=12, seed=0):
    torch.manual_seed(seed)
    return StandaloneActorCritic(obs_dim=obs_dim, action_dim=4, hidden_dim=16,
                                 device="cpu", entropy_coef=0.01)


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def main():
    # ---------------------------------------------------------------- A
    old = pol(seed=1)
    with torch.no_grad():
        for p in old.actor.parameters():
            p.add_(0.5)                              # a "trained" policy
    sd = old.get_state_dict()
    new = pol(seed=2)
    head0 = {k: v.clone() for k, v in new.actor.action_head.state_dict().items()}
    crit0 = {k: v.clone() for k, v in new.critic.state_dict().items()}
    plan = new.load_features(sd)
    check(same(new.actor.shared.state_dict(), old.actor.shared.state_dict()),
          "A. actor.shared copied exactly from the stored policy")
    check(same(new.actor.action_head.state_dict(), head0)
          and same(new.critic.state_dict(), crit0)
          and plan["action_head"] == "fresh" and plan["critic"] == "fresh",
          "A. the action head and the critic stay at their FRESH initialisation")

    # ---------------------------------------------------------------- B
    wide = pol(obs_dim=20, seed=3)
    before = {k: v.clone() for k, v in wide.actor.state_dict().items()}
    try:
        wide.load_features(sd)
        refused = False
    except ValueError as e:
        refused = "refusing before copy" in str(e)
    check(refused and same(wide.actor.state_dict(), before),
          "B. a stored policy of another input width is refused; nothing copied")
    bad = dict(sd, arch="conv")
    try:
        new.load_features(bad)
        refused = False
    except ValueError:
        refused = True
    check(refused, "B. a stored policy of another arch is refused")

    # ---------------------------------------------------------------- C
    q = pol()
    check(q.effective_entropy_coef() == 0.01, "C. no boost: the configured coefficient")
    q.start_entropy_boost(3.0, 4)
    seq = []
    for _ in range(6):
        seq.append(round(q.effective_entropy_coef(), 5))
        q._ent_boost_left = max(0, q._ent_boost_left - 1)       # one update spent
    check(seq == [0.03, 0.025, 0.02, 0.015, 0.01, 0.01],
          "C. x3 at the first update, linear decay, exactly 1x after 4: %s" % seq)

    # ---------------------------------------------------------------- D
    r = pol()
    r.start_entropy_boost(3.0, 10)
    rng = np.random.default_rng(0)
    for t in range(64):
        o = rng.random(12).astype(np.float32)
        a, info = r.select_action(o)
        r.store_transition(o, a, float(rng.random()), t == 63,
                           info.get("log_prob", 0.0), info.get("value", 0.0))
    m = r.train_step(n_epochs=1)
    check(abs(m["entropy_coef_eff"] - 0.03) < 1e-9 and r._ent_boost_left == 9,
          "D. a real PPO update used x3 (%.3f) and spent one boost update"
          % m["entropy_coef_eff"])

    # ---------------------------------------------------------------- E
    src = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    keyed = src[src.index("_RESUME_ENCODER_KEYED = ("):src.index("class _ResumePreflightRefusal")]
    check('"policy_features"' in keyed,
          "E. policy_features is gated on world_model (encoder-keyed)")
    order = src[src.index("names = sorted(names, key=lambda n:"):][:400]
    check('"policy_features"' in order, "E. ...and ordered after world_model")
    check('if _n in ("policy", "dream_actor", "curiosity"):' in src,
          "E. the whole policy is still refused by name")
    uni = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))["loop"]
    leg = yaml.safe_load(open(os.path.join(ROOT, "configs/minecraft_skybot.yaml")))["loop"]
    # 2026-10-18: universal resumes `policy_full` (reward-fingerprint gated),
    # whose fallback IS this features-only path; policy_features stays a
    # loadable component (tests/_resume_smoke.py covers policy_full)
    check(uni["resume_components"][:-1] == leg["resume_components"]
          and uni["resume_components"][-1] == "policy_full"
          and '"policy_features": ("policy.pt", _load_policy_features)' in src
          and "_load_policy_features(sd)            # may raise" in src,
          "E. universal = the full base list + policy_full, whose fallback is "
          "policy_features")
    check("policy_features" not in leg["resume_components"],
          "E. the legacy Minecraft brain is unchanged")
    # ---------------------------------------------------------------- F
    teacher = pol(seed=11)
    with torch.no_grad():
        teacher.actor.action_head.weight.mul_(8.0)        # a decisive teacher
    t_actor = teacher.get_state_dict()["actor"]
    probe = torch.from_numpy(np.random.default_rng(5).random((64, 12)).astype(np.float32))

    def kl_to_teacher(p_):
        with torch.no_grad():
            tl = torch.log_softmax(teacher.actor.action_head(teacher.actor.shared(probe)), -1)
            sl = torch.log_softmax(p_.actor.action_head(p_.actor.shared(probe)), -1)
            return float((tl.exp() * (tl - sl)).sum(-1).mean())

    def train(p_, n_upd):
        g = np.random.default_rng(1)
        for u in range(n_upd):
            for t in range(64):
                o = g.random(12).astype(np.float32)
                a, info = p_.select_action(o)
                p_.store_transition(o, a, 0.0, t == 63, info.get("log_prob", 0.0),
                                    info.get("value", 0.0))
            last = p_.train_step(n_epochs=2)
        return last

    plain, ks = pol(seed=2), pol(seed=2)
    ks.start_kickstart(t_actor, 5.0, 3)
    k0 = kl_to_teacher(ks)
    m = train(ks, 3)
    train(plain, 3)
    check(kl_to_teacher(ks) < kl_to_teacher(plain) and kl_to_teacher(ks) < k0,
          "F. kickstarted student ends nearer the teacher (KL %.3f) than without "
          "(%.3f), from %.3f" % (kl_to_teacher(ks), kl_to_teacher(plain), k0))
    check(ks._teacher is None and ks.kickstart_weight() == 0.0 and m["kickstart_weight"] > 0,
          "F. the weight decays to 0 and the teacher is dropped after 3 updates")
    wide = pol(obs_dim=20, seed=3)
    try:
        wide.start_kickstart(t_actor, 1.0, 5)
        refused = False
    except ValueError:
        refused = True
    check(refused and wide._teacher is None, "F. a mismatched teacher is refused")
    check('"policy_kickstart"' in keyed and '"policy_kickstart"' in order,
          "F. policy_kickstart is gated on world_model and ordered after it")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
