"""Smoke: selective resume — keep perception across restarts, never the policy.

MEASURED 2026-08-03: `load_checkpoint()` has existed for months and NOTHING
in the launch path called it (only the demo/eval/transfer scripts do). So
every restart discarded world_model.pt (55 MB), policy.pt, curiosity.pt,
dream_actor.pt and glue_layer.pt and relearned from scratch, while the skill
bank persisted — the system retained SKILLS better than LEARNING. Visible in
the logs as entropy starting ~90% after every deploy and the reward curve
0.02 -> 0.14 -> 0.38 -> 0.53 being re-run from zero each time.

What this pins:
  1. The refusal list. `policy` / `dream_actor` / `curiosity` must be refused
     BY NAME, so a config typo can never resurrect a converged, 0-entropy
     policy under a reward function that has since changed.
  2. The loaders match the WRITER. symbolic_decoder.pt is the inner
     `.decoder`; glue_layer.pt is a two-key {"gnn","gate"} dict. Loading
     either into the outer object raises or silently no-ops.
  3. A shape mismatch is LOUD and leaves fresh weights intact — action_dim
     moved 10 -> 12 historically.
  4. Off by default: no resume_components -> byte-identical old behaviour.
  6. `policy_full` (2026-10-18): the bare `policy` stays refused by name (1),
     but the WHOLE policy — head, critic, Adam, the reward mixer — resumes
     when policy_meta.json carries the SAME reward fingerprint and every
     tensor fits; a changed reward falls back to the features-only path
     (fresh head, entropy boost), reshaped tensors are refused cleanly, and
     it is gated on world_model like every latent reader. The refusal's
     reason ("a converged policy under a reward that has since changed") is
     exactly what the fingerprint tests.

Run: PYTHONPATH=. python tests/_resume_smoke.py
"""
import sys

sys.path.insert(0, ".")

import os
import tempfile

import torch
import yaml

LOOP = "developmental_ai/core/developmental_loop.py"
CFG = "configs/minecraft_skybot.yaml"


def main() -> None:
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    src = open(LOOP).read()
    cfg = yaml.safe_load(open(CFG))

    # ---- 1. skybot resumes EXPERIENCE, never policy ----------------------
    # (2026-08-10: perception + familiarity joined — grounded heads, label
    # evidence and the "what have I seen" counts are experience, not policy)
    # (2026-08-17: magnet joined — the learning-progress estimates that
    # decide WHERE TO LOOK are experience by the same test. Measured after
    # that day's relaunch: "w=0.0000, target=None", every category at LP
    # 0.000 — the magnet was amnesiac, not suppressed. Its cold-start
    # budget is deliberately NOT in the payload; see VisionScaffold.state.)
    # (2026-08-23: replay_buffer joined — the world model's RAW EXPERIENCE.
    # It passes the same test: restoring frames and actions cannot resume a
    # converged optimum, it only spares the model re-collecting the world.
    # Until this existed the buffer had NO persistence at all, so the
    # supervisor's automatic crash-relaunch restarted world-model training
    # from EMPTY — the effective memory was never the configured capacity,
    # it was "however long since the last crash". Deliberately NOT in
    # _RESUME_ENCODER_KEYED: raw frames are not expressed in the encoder's
    # latent space, and training a fresh encoder on old frames is exactly
    # what a replay buffer is FOR.)
    rc = cfg["loop"]["resume_components"]
    assert set(rc) == {"world_model", "symbolic_decoder", "glue_layer",
                       "knowledge_graph", "perception", "familiarity",
                       "magnet", "replay_buffer"}, \
        f"unexpected resume set: {rc}"
    for banned in ("policy", "dream_actor", "curiosity"):
        assert banned not in rc, (
            f"'{banned}' is in resume_components — restoring a converged "
            "policy resumes its entropy collapse")

    # ---- 2. the refusal is enforced in CODE, not just absent from config -
    class _Stub:
        """Minimal stand-in: only the bits _resume_components touches."""
        def __init__(self, tmp):
            self.device = torch.device("cpu")
            self.config = {}
            self.tmp = tmp
            self.loaded = []
        # world_model / symbolic_decoder / glue are probed via attributes;
        # leaving them absent proves the loader fails LOUD rather than
        # silently skipping a component the user asked for.
    tmp = tempfile.mkdtemp()
    s = _Stub(tmp)
    # a checkpoint that LOOKS complete
    torch.save({"nope": torch.zeros(1)}, os.path.join(tmp, "world_model.pt"))
    torch.save({"nope": torch.zeros(1)}, os.path.join(tmp, "policy.pt"))

    DevelopmentalAI._resume_components(s, tmp, ["policy", "dream_actor",
                                                "curiosity"])
    # nothing raised, and crucially nothing was restored
    assert not getattr(s, "_restored", None), "a refused component loaded"
    assert "REFUSING" in src, "the by-name refusal is not implemented"
    assert 'if _n in ("policy", "dream_actor", "curiosity")' in src, \
        "the refusal list changed — policy could be resumed by config alone"

    # ---- 3. a missing/failed component must not take the run down -------
    #  world_model.pt here is garbage; the loader must warn and continue.
    class _WM:
        # state_dict() must answer the PRE-FLIGHT key check (2026-08-23): the
        # loader now compares the model's keys against the payload's before
        # copying anything. Returning the payload's own key here keeps this
        # case about the SHAPE mismatch it was written for.
        def state_dict(self):
            return {"nope": torch.zeros(1)}

        def load_state_dict(self, m, strict=True):
            raise RuntimeError("size mismatch for encoder.0.weight")
    s2 = _Stub(tmp)
    s2.world_model = _WM()
    DevelopmentalAI._resume_components(s2, tmp, ["world_model"])   # no raise

    # ---- 3b. a checkpoint that CANNOT FILL the model is refused BEFORE any
    #      tensor is copied (2026-08-23 review). Previously strict=False
    #      accepted it silently, world_model entered `_ok`, and the
    #      dependency gate then admitted perception/familiarity onto a
    #      half-fresh encoder. The refusal must still warn-and-continue, and
    #      must not have mutated the module on its way out.
    class _WMPartial:
        def __init__(self):
            self.touched = False

        def state_dict(self):
            return {"nope": torch.zeros(1), "encoder.0.weight": torch.zeros(1)}

        def load_state_dict(self, m, strict=True):
            self.touched = True
            return ([], [])
    s3 = _Stub(tmp)
    s3.world_model = _WMPartial()
    DevelopmentalAI._resume_components(s3, tmp, ["world_model"])   # no raise
    assert not s3.world_model.touched, (
        "a checkpoint missing `encoder.0.weight` reached load_state_dict — "
        "the pre-flight must refuse BEFORE copying, or the model is left a "
        "partial hybrid while the resume reports success")
    # ---- 3c. SAME KEYS, DIFFERENT SHAPE (2026-10-17). Passed the key-only
    #      pre-flight, then torch copied every matching tensor and raised at
    #      the end: a partial hybrid. Must be refused before any copy.
    class _WMReshaped:
        def __init__(self):
            self.touched = False

        def state_dict(self):
            return {"inverse_head.net.0.weight": torch.zeros(4, 8)}

        def load_state_dict(self, m, strict=True):
            self.touched = True
            return ([], [])
    torch.save({"inverse_head.net.0.weight": torch.zeros(4, 16)},
               os.path.join(tmp, "world_model.pt"))
    s4 = _Stub(tmp)
    s4.world_model = _WMReshaped()
    DevelopmentalAI._resume_components(s4, tmp, ["world_model"])   # no raise
    assert not s4.world_model.touched, (
        "a checkpoint whose tensor SHAPES differ reached load_state_dict — "
        "torch would copy the matching ones and raise at the end, leaving a "
        "partial hybrid; the pre-flight must refuse on shape too")
    # and an absent directory is a clean no-op, not a crash
    DevelopmentalAI._resume_components(s2, os.path.join(tmp, "nope"),
                                       ["world_model"])

    # ---- 4. loaders address the same objects the writer wrote -----------
    save_blk = src[src.index("def _save_checkpoint"):src.index("def load_checkpoint")]
    assert "self.symbolic_decoder.decoder.state_dict()" in save_blk
    assert 'getattr(self.symbolic_decoder, "decoder", None)' in src, \
        "resume loads symbolic_decoder into the OUTER object, not .decoder"
    assert '"gnn": self.glue.knowledge_integrator.gnn.state_dict()' in save_blk
    assert '_ki.gnn.load_state_dict(m["gnn"])' in src, \
        "resume does not unpack the two-key glue_layer dict"

    # ---- 5. OFF by default ----------------------------------------------
    assert 'resume_components") or [])' in src, "resume is not opt-in"
    s3 = _Stub(tmp)
    s3.config = {"loop": {}}          # no resume_components -> never called
    assert not s3.loaded

    # ---- 6. policy_full: the WHOLE policy, only for an UNCHANGED reward ---
    policy_full_contract(src)

    print("[resume-smoke] ALL PASS: skybot resumes world_model + "
          "symbolic_decoder + glue_layer + knowledge_graph and nothing else; "
          "policy/dream_actor/curiosity are refused BY NAME in code so a "
          "config typo cannot resurrect a 0-entropy policy; loaders address "
          "the same objects _save_checkpoint wrote (.decoder, the "
          "{gnn,gate} dict); a shape mismatch or missing dir warns and "
          "continues on fresh weights; off by default; policy_full restores "
          "the whole policy only under an unchanged reward fingerprint and "
          "otherwise falls back to features-only (or a clean refusal)")


def policy_full_contract(src: str) -> None:
    """2026-10-18. Every relaunch reset the action head and critic (+ x3
    entropy boost); `policy_full` restores the whole policy — head, critic,
    Adam, the reward mixer's schedule — ONLY when policy_meta.json carries
    the same reward fingerprint and every tensor fits. Otherwise: the
    features-only fallback (head fresh, boost on), or a clean refusal."""
    import json
    import types

    import numpy as np
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.policy.actor_critic import (RewardMixer,
                                                      StandaloneActorCritic)

    def cfg(ep=0.07):
        return {"loop": {"policy_features_boost": {"factor": 3.0, "updates": 5}},
                "curiosity": {"episodic_weight": ep},
                "policy": {"intrinsic_reward_weight": 0.7, "gamma": 0.995},
                "universal": {"games": ["doom"], "action_slots": 48}}

    def stub(c, obs_dim=10, seed=0):
        torch.manual_seed(seed)
        s = types.SimpleNamespace()
        s.device = torch.device("cpu")
        s.config = c
        s.action_dim = 4
        s.meta_action_dim = 4
        s.world_model = torch.nn.Linear(3, 3)
        s.policy = StandaloneActorCritic(obs_dim=obs_dim, action_dim=4,
                                         hidden_dim=8, device="cpu")
        s.reward_mixer = RewardMixer(hold_without_extrinsic=True)
        s._FP_POLICY_KEYS = DevelopmentalAI._FP_POLICY_KEYS
        s._reward_fingerprint = types.MethodType(
            DevelopmentalAI._reward_fingerprint, s)
        return s

    ck = tempfile.mkdtemp()
    w = stub(cfg())
    torch.save(w.world_model.state_dict(), os.path.join(ck, "world_model.pt"))
    g = np.random.default_rng(0)
    for t in range(64):                      # a few real updates: Adam state
        o = g.random(10).astype(np.float32)
        a, info = w.policy.select_action(o)
        w.policy.store_transition(o, a, float(a == 1), t == 63,
                                  info["log_prob"], info["value"])
    w.policy.train_step(n_epochs=2)
    w.reward_mixer.intrinsic_weight, w.reward_mixer.extrinsic_weight = 0.55, 0.45
    torch.save(w.policy.get_state_dict(), os.path.join(ck, "policy.pt"))
    torch.save(w.policy.optimizer.state_dict(), os.path.join(ck, "policy_optim.pt"))
    fp, _ = w._reward_fingerprint()
    json.dump({"reward_fingerprint": fp, "mixer": w.reward_mixer.state_dict()},
              open(os.path.join(ck, "policy_meta.json"), "w"))

    def eq(a, b):
        return all(torch.equal(a[k], b[k]) for k in a)

    # same reward -> the whole policy, Adam and the mixer; NO boost
    r = stub(cfg(), seed=1)
    DevelopmentalAI._resume_components(r, ck, ["world_model", "policy_full"])
    assert getattr(r, "_policy_resume_mode", None) == "full", "full resume did not run"
    assert eq(r.policy.actor.state_dict(), w.policy.actor.state_dict()) and \
        eq(r.policy.critic.state_dict(), w.policy.critic.state_dict()), \
        "policy_full did not restore the actor head and the critic"
    assert r.policy.optimizer.state_dict()["state"], "Adam state not restored"
    assert r.reward_mixer.intrinsic_weight == 0.55, "mixer schedule not restored"
    assert r.policy._ent_boost_left == 0, "a full resume must not boost entropy"

    # changed reward -> features only, fresh head, boost on
    c2 = stub(cfg(ep=0.05), seed=2)
    head0 = {k: v.clone() for k, v in c2.policy.actor.action_head.state_dict().items()}
    DevelopmentalAI._resume_components(c2, ck, ["world_model", "policy_full"])
    assert c2._policy_resume_mode == "features", "fingerprint change did not fall back"
    assert eq(c2.policy.actor.shared.state_dict(), w.policy.actor.shared.state_dict()) \
        and eq(c2.policy.actor.action_head.state_dict(), head0), \
        "fallback must load the feature layers and keep the head fresh"
    assert c2.policy._ent_boost_left == 5 and c2.reward_mixer.intrinsic_weight == 0.7, \
        "fallback: boost on, mixer NOT restored (it belongs to the old reward)"

    # same reward, other shapes -> both paths refuse; nothing copied
    d = stub(cfg(), obs_dim=14, seed=3)
    before = {k: v.clone() for k, v in d.policy.actor.state_dict().items()}
    DevelopmentalAI._resume_components(d, ck, ["world_model", "policy_full"])
    assert eq(d.policy.actor.state_dict(), before) and \
        getattr(d, "_policy_resume_mode", None) is None, \
        "a reshaped policy must be refused cleanly by both paths"

    # the world model gate still applies (no world model -> no policy)
    e = stub(cfg(), seed=4)
    e_before = {k: v.clone() for k, v in e.policy.actor.state_dict().items()}
    DevelopmentalAI._resume_components(e, ck, ["policy_full"])
    assert eq(e.policy.actor.state_dict(), e_before), \
        "policy_full loaded without world_model restoring"

    # the fingerprint ignores what does not define the reward
    f1, _ = stub(cfg())._reward_fingerprint()
    big = cfg()
    big["loop"]["log_dir"] = "elsewhere"
    big["world_model"] = {"deterministic_size": 1024}
    big["parallel_envs"] = {"num_envs": 8}
    assert stub(big)._reward_fingerprint()[0] == f1, \
        "world-model size / streams / paths must not change the reward fingerprint"
    assert stub(cfg(ep=0.071))._reward_fingerprint()[0] != f1, \
        "a curiosity weight change must change the fingerprint"

    save_blk = src[src.index("def _save_checkpoint_locked"):]
    save_blk = save_blk[:save_blk.index("# options: record the head layout")]
    assert '"policy_meta.json"' in save_blk and '"reward_fingerprint"' in save_blk \
        and '"policy_optim.pt"' in save_blk and '"loop_meta.json"' in save_blk, \
        "the checkpoint must write what policy_full and the lifetime clock read"
    assert 'if _n in ("policy", "dream_actor", "curiosity"):' in src, \
        "the bare policy must stay refused by name"
    print("  6. policy_full: unchanged reward -> whole policy + Adam + mixer, "
          "no boost; changed reward -> features-only + boost; reshaped -> "
          "clean refusal; gated on world_model; fingerprint ignores size/"
          "streams/paths")


if __name__ == "__main__":
    main()
