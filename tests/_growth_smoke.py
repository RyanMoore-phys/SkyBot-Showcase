"""Stage D: the brain grows in place — and shrinks only if it loses nothing.

Born 2026-10-17 (docs/research/res-004-rssm-size-experiment.md stage D). User design: SkyBot's
world model starts small and grows as it masters its worlds, staying the SAME
brain; it can also come back down when a task needs less. The autoscaler
(infra/autoscaler.py) decides; this suite covers the machinery that acts:
world_model/growth.py, tools/grow_brain.py, the composer, the loop and the
supervisor.

Contracts:
    A. EXACT GROWTH: a TRAINED world model with every live feature (FiLM, flow
       head, coord channels, latent horizons, summed reconstruction, embedding
       inverse head) grown h x2, h x2 + more categoricals/classes, and h x4
       produces the same reconstruction, reward, continue and imagined frames
       (eval mode) to within 1e-5.
    B. REAL NEW CAPACITY: after a few training steps the duplicated h units
       are no longer equal — the zero-sum split broke the symmetry.
    C. REFUSALS NAME THE PROBLEM: shrinking through the exact path, a
       non-integer h factor, or a tensor with no growth rule raises; nothing is
       guessed.
    D. THE TOOL, END TO END on a real checkpoint directory: world_model.pt is
       grown and verified, the old files are kept, world_model_arch.json and
       the scale file record the new size, a latent-sized symbolic decoder is
       moved aside (a reshaped load would be a partial hybrid), and the grown
       checkpoint loads STRICTLY into a model built from the new arch.
       (2026-10-18) The latent's READERS grow with it: policy.pt and
       dream_actor.pt are grown and verified on the same frames (old reader
       on old latents vs grown on grown, logits/values within tol), the
       grown policy loads WHOLE into a policy built for the new latent,
       Adam's old-width moments are moved aside, and a reader that fails or
       does not read the latent is left byte-identical.
    E. SHRINK IS ACCEPTED ONLY WITHIN TOLERANCE: relearned from the saved
       replay buffer; with an impossible tolerance it is refused and the
       bigger model stays untouched.
    F. --apply-pending removes runlogs/SCALE_RESTART whether it applied or
       refused (no request is retried at every launch — §4.1).
    G. THE NEXT LAUNCH BUILDS THE NEW SIZE: compose applies the scale file's
       dims and stream count; Minecraft stays capped at 2 (the Paper server).
    H. ACTUATION IS OFF BY DEFAULT and, when on, is a graceful stop plus a
       marker; the supervisor relaunches kind=scale without counting a crash.
    I. A NEW BRAIN STARTS SMALL (user, 2026-10-18: tiny x 8). At the launch
       (--record-scale), no scale file and no saved world model -> compose
       builds `universal.initial_scale`
       INCLUDING the encoder width (without it "tiny" had live's 768-wide
       encoder, ~6x the measured tiny), and the launch's --record-scale
       writes it once; an EXISTING brain without a record is never seeded
       (it would be refused at resume and start fresh); grow_brain keeps the
       encoder width in the record it rewrites.

Run: PYTHONPATH=. python tests/_growth_smoke.py
"""
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.world_model.growth import (Dims, GrowthError, build_from_arch,  # noqa
                                                 dims_of, grow_state_dict, load_arch,
                                                 save_arch, verify)
from developmental_ai.world_model.rssm import WorldModel                              # noqa

NAME = "growth"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def build(d: Dims):
    torch.manual_seed(0)
    return WorldModel(obs_dim=3 * 32 * 32, action_dim=6, stochastic_size=d.stoch,
                      stochastic_classes=d.classes, deterministic_size=d.det, hidden_dim=48,
                      pixel_obs=True, image_size=32, inverse_dynamics=True,
                      film_conditioning=True, flow_head=True, flow_size=8,
                      coord_channels=True, readout_channels=8, recon_reduction="sum",
                      inverse_input="embed", latent_horizons=[4])


def data(seed=0, B=2, T=8):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(B, T, 3 * 32 * 32, generator=g)
    a = torch.nn.functional.one_hot(torch.randint(0, 6, (B, T), generator=g), 6).float()
    return o, a, torch.zeros(B, T), torch.ones(B, T)


def grown(old, d):
    g = build(d)
    g.load_state_dict(grow_state_dict(old.state_dict(), g.state_dict(), dims_of(old), d),
                      strict=True)
    return g


def main():
    tmp = tempfile.mkdtemp(prefix="growth_")
    cwd = os.getcwd()
    try:
        run(tmp)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{NAME}] ALL PASS")


def run(tmp):
    old = build(Dims(32, 4, 4))
    o, a, r, c = data()
    for _ in range(3):
        old.train_step(o, a, r, c)

    # ---------------------------------------------------------------- A
    for d in (Dims(64, 4, 4), Dims(64, 6, 6), Dims(128, 8, 4)):
        diff = verify(old, grown(old, d), o, a)
        check(max(diff.values()) < 1e-5,
              f"A. {Dims(32, 4, 4)} -> {d}: outputs identical (max |diff| "
              f"{max(diff.values()):.1e})")

    # ---------------------------------------------------------------- B
    g = grown(old, Dims(64, 4, 4))
    for i in range(4):
        g.train_step(*data(10 + i))
    w = g.rssm.gru.weight_hh.detach()
    rows = w[64:128]                       # update-gate block: copies 0..31 vs 32..63
    gap = float((rows[:32] - rows[32:]).abs().max())
    check(gap > 1e-6, f"B. after training the duplicated h units differ ({gap:.1e}): "
          "new capacity, not a clone")

    # ---------------------------------------------------------------- C
    big = build(Dims(64, 4, 4))
    for bad, why in ((Dims(16, 4, 4), "grow only"), (Dims(48, 4, 4), "integer factor")):
        try:
            grow_state_dict(old.state_dict(), build(bad).state_dict(), dims_of(old), bad)
            msg = ""
        except GrowthError as e:
            msg = str(e)
        check(why in msg, f"C. {Dims(32, 4, 4)} -> {bad} refused ({why})")
    sd = old.state_dict()
    sd["decoder.fc.weight"] = torch.zeros(3, 3)
    try:
        grow_state_dict(sd, big.state_dict(), dims_of(old), Dims(64, 4, 4))
        msg = ""
    except GrowthError as e:
        msg = str(e)
    check("decoder.fc.weight" in msg, "C. a tensor with no growth rule is refused, named")

    # ---------------------------------------------------------------- D
    from tools import grow_brain as G
    ck = os.path.join(tmp, "brain", "logs", "checkpoints")
    os.makedirs(ck)
    tiny = WorldModel(obs_dim=3 * 32 * 32, action_dim=6, deterministic_size=256,
                      stochastic_size=16, stochastic_classes=16, hidden_dim=32,
                      pixel_obs=True, image_size=32, inverse_dynamics=True,
                      inverse_input="embed", recon_reduction="sum")
    torch.save(tiny.state_dict(), os.path.join(ck, "world_model.pt"))
    # the live loop passes NumPy integers (obs_dim, layout indices): a real
    # Doom run failed to write the arch file on exactly this (2026-10-17)
    tiny.init_kwargs["obs_dim"] = np.int64(tiny.init_kwargs["obs_dim"])
    tiny.init_kwargs["proprio_layout"] = {"pitch": np.int64(0)}
    save_arch(tiny, ck)
    check(load_arch(ck)["obs_dim"] == 3 * 32 * 32,
          "D. the arch file is written even with NumPy integers (as the live loop passes)")
    torch.save({"head.weight": torch.zeros(5, 256 + 256)},
               os.path.join(ck, "symbolic_decoder.pt"))       # latent-sized
    # the latent's READERS (2026-10-18): an rssm policy (latent + 3 proprio +
    # gated 4-d knowledge) with Adam state, and a dream actor
    from developmental_ai.policy.actor_critic import (DreamActorCritic,
                                                      StandaloneActorCritic)
    torch.manual_seed(1)
    pol0 = StandaloneActorCritic(obs_dim=3 * 32 * 32, action_dim=8, hidden_dim=16,
                                 device="cpu", arch="rssm", enc_dim=512,
                                 proprio_dim=3, knowledge_dim=4)
    with torch.no_grad():
        for prm in list(pol0.actor.parameters()) + list(pol0.critic.parameters()):
            prm.add_(0.1 * torch.randn_like(prm))         # not an init
    torch.save(pol0.get_state_dict(), os.path.join(ck, "policy.pt"))
    torch.save({"state": {0: {"exp_avg": torch.zeros(16, 512 + 3 + 4)}},
                "param_groups": [{"params": [0]}]},
               os.path.join(ck, "policy_optim.pt"))
    torch.save(DreamActorCritic(512, 6, hidden_dim=16).state_dict(),
               os.path.join(ck, "dream_actor.pt"))
    scale = os.path.join(tmp, "brain", "scale.json")
    rep = G.resize(ck, "small", scale, streams=4)
    check(rep["result"] == "grown (exact)" and rep["written"]
          and max(rep["verify_max_abs"].values()) < 1e-4,
          "D. tiny -> small grown and verified (max |diff| %.1e)"
          % max(rep["verify_max_abs"].values()))
    arch = load_arch(ck)
    m = build_from_arch(arch)
    m.load_state_dict(torch.load(os.path.join(ck, "world_model.pt"), weights_only=False),
                      strict=True)
    check(arch["deterministic_size"] == 512 and arch["stochastic_size"] == 24,
          "D. world_model_arch.json records the new size; the checkpoint loads strictly")
    check(any(f.startswith("world_model.pt.pre_resize_") for f in os.listdir(ck)),
          "D. the old world model is kept (pre_resize backup)")
    sc = json.load(open(scale))
    check(sc["size"] == "small" and sc["dims"]["det"] == 512 and sc["streams"] == 4,
          "D. the scale file records size small, 512/24/24, 4 streams")
    check("symbolic_decoder.pt" in rep["moved_aside"]
          and not os.path.exists(os.path.join(ck, "symbolic_decoder.pt")),
          "D. the latent-sized symbolic decoder is moved aside (fresh, not a hybrid)")
    rd = rep.get("readers", {})
    pd = (rd.get("policy.pt") or {}).get("grown (exact)") if isinstance(
        rd.get("policy.pt"), dict) else None
    dd = (rd.get("dream_actor.pt") or {}).get("grown (exact)") if isinstance(
        rd.get("dream_actor.pt"), dict) else None
    check(pd is not None and max(pd.values()) < 1e-4,
          "D. the POLICY grows with it: logits and value identical on the same "
          "frames (old policy/old latent vs grown/grown, max |diff| %s)" % pd)
    check(dd is not None and max(dd.values()) < 1e-4,
          "D. ...and the dream actor (max |diff| %s)" % dd)
    pol1 = StandaloneActorCritic(obs_dim=3 * 32 * 32, action_dim=8, hidden_dim=16,
                                 device="cpu", arch="rssm", enc_dim=512 + 24 * 24,
                                 proprio_dim=3, knowledge_dim=4)
    plan = pol1.load_full(torch.load(os.path.join(ck, "policy.pt"), weights_only=False))
    check(plan["actor"] == "loaded" and plan["critic"] == "loaded",
          "D. the grown policy.pt loads WHOLE into a policy built for the new "
          "latent (it is not refused and restarted any more)")
    check(not os.path.exists(os.path.join(ck, "policy_optim.pt"))
          and any(f.startswith("policy_optim.pt.pre_resize_") for f in os.listdir(ck)),
          "D. Adam's old-width moments are moved aside (Adam restarts, weights do not)")
    stale = {"state": {0: {"exp_avg": torch.zeros(16, 512 + 3 + 4)}},
             "param_groups": [{"params": list(range(
                 len(pol1.optimizer.state_dict()["param_groups"][0]["params"])))}]}
    check(pol1.load_full(torch.load(os.path.join(ck, "policy.pt"), weights_only=False),
                         stale)["optimizer"] == "fresh",
          "D. a stale-width Adam state is refused even if handed over")
    # a reader that does not read the latent is reported and left untouched
    ck2 = os.path.join(tmp, "brain2", "checkpoints")
    os.makedirs(ck2)
    torch.save(tiny.state_dict(), os.path.join(ck2, "world_model.pt"))
    save_arch(tiny, ck2)
    flat = StandaloneActorCritic(obs_dim=12, action_dim=4, hidden_dim=8, device="cpu")
    torch.save(flat.get_state_dict(), os.path.join(ck2, "policy.pt"))
    raw = open(os.path.join(ck2, "policy.pt"), "rb").read()
    rep2 = G.resize(ck2, "small", os.path.join(tmp, "brain2", "scale.json"))
    check(str(rep2["readers"].get("policy.pt", "")).startswith("left as is")
          and open(os.path.join(ck2, "policy.pt"), "rb").read() == raw,
          "D. a policy that does not read the latent is left byte-identical "
          "(%s)" % str(rep2["readers"].get("policy.pt"))[:60])

    # ---------------------------------------------------------------- E
    from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer
    rb = MultiStreamReplayBuffer(num_streams=1, capacity=2000, obs_dim=3 * 32 * 32,
                                 action_dim=6, obs_uint8=True)
    rng = np.random.default_rng(0)
    for t in range(400):
        rb.add((rng.random(3 * 32 * 32) * 255).astype(np.uint8).astype(np.float32) / 255.0,
               np.eye(6)[t % 6], 0.0, False, stream=0)
    rb.save(os.path.join(ck, "replay_buffer"))
    before = open(os.path.join(ck, "world_model.pt"), "rb").read()
    rep = G.resize(ck, "tiny", scale, shrink_tol=0.0, shrink_steps=3)
    check(rep["result"].startswith("shrink REFUSED") and not rep["written"]
          and open(os.path.join(ck, "world_model.pt"), "rb").read() == before,
          "E. a shrink that cannot match the bigger model is refused; nothing written")
    rep = G.resize(ck, "tiny", scale, shrink_tol=1e9, shrink_steps=3)
    check(rep["result"].startswith("shrunk") and rep["written"]
          and load_arch(ck)["deterministic_size"] == 256,
          "E. within tolerance it is accepted and the arch records the smaller size")

    # ---------------------------------------------------------------- F
    os.chdir(tmp)
    os.makedirs("runlogs", exist_ok=True)
    json.dump({"action": "grow_model", "target_size": "medium"}, open(G.PENDING, "w"))
    G.main(["--apply-pending", "--checkpoints", "/nonexistent", "--scale-file", scale])
    check(not os.path.exists(G.PENDING),
          "F. a refused pending resize still removes runlogs/SCALE_RESTART")

    # ---------------------------------------------------------------- G
    from tools.compose_run_config import compose
    uni = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    uni["universal"]["scale_file"] = scale
    up = os.path.join(tmp, "universal.yaml")
    yaml.safe_dump(uni, open(up, "w"))
    json.dump({"size": "medium", "dims": {"det": 1024, "stoch": 32, "classes": 32},
               "streams": 4}, open(scale, "w"))
    base = os.path.join(ROOT, "configs/minecraft_skybot.yaml")
    games = os.path.join(ROOT, "configs/games")
    cfg = compose(["doom"], base, up, games)
    wm = cfg["world_model"]
    check(wm["deterministic_size"] == 1024 and wm["stochastic_size"] == 32
          and len(cfg["parallel_envs"]["streams"]) == 4,
          "G. compose builds the recorded size (1024/32/32) and 4 Doom streams")
    json.dump({"streams": 6}, open(scale, "w"))
    cfg = compose(["minecraft", "doom"], base, up, games)
    per = [s["game"] for s in cfg["parallel_envs"]["streams"]]
    check(per.count("minecraft") == 2 and per.count("doom") == 4,
          "G. 6 streams over Minecraft + Doom: Minecraft stays at 2, Doom takes 4 (%s)" % per)

    # ---------------------------------------------------------------- I
    import tools.compose_run_config as C
    nb = os.path.join(tmp, "newbrain")
    uni2 = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    check(uni2["universal"].get("initial_scale") == {"size": "tiny", "streams": 8},
          "I. the shipped initial scale is tiny x 8")
    uni2["universal"]["scale_file"] = os.path.join(nb, "scale.json")
    uni2["loop"]["log_dir"] = os.path.join(nb, "logs")
    up2 = os.path.join(tmp, "universal2.yaml")
    yaml.safe_dump(uni2, open(up2, "w"))
    plain = compose(["doom"], base, up2, games)
    check(plain["world_model"]["deterministic_size"] != 256
          and len(plain["parallel_envs"]["streams"]) == 2,
          "I. a plain compose (tests, experiment arms) is unchanged: no seeding")
    cfg = compose(["doom"], base, up2, games, seed_new_brain=True)
    wm = cfg["world_model"]
    check(wm["deterministic_size"] == 256 and wm["stochastic_size"] == 16
          and wm["encoder_hidden"] == 128 and wm["decoder_hidden"] == 128
          and len(cfg["parallel_envs"]["streams"]) == 8,
          "I. a new brain composes tiny (256/16/16, encoder 128) x 8 Doom streams")
    out = os.path.join(tmp, "run_config.yaml")
    rc = C.main(["--games", "doom", "--universal", up2, "--out", out])
    check(rc == 0 and not os.path.exists(uni2["universal"]["scale_file"]),
          "I. without --record-scale nothing is seeded or written")
    rc = C.main(["--games", "doom", "--universal", up2, "--out", out, "--record-scale"])
    rec = json.load(open(uni2["universal"]["scale_file"]))
    check(rc == 0 and rec["size"] == "tiny" and rec["streams"] == 8
          and rec["dims"]["encoder"] == 128 and "_scale_seeded" not in open(out).read(),
          "I. the launch records the seed once (size, streams, encoder width)")
    json.dump(dict(rec, streams=5), open(uni2["universal"]["scale_file"], "w"))
    C.main(["--games", "doom", "--universal", up2, "--out", out, "--record-scale"])
    check(json.load(open(uni2["universal"]["scale_file"]))["streams"] == 5,
          "I. an existing record is never overwritten by the seed")
    os.remove(uni2["universal"]["scale_file"])
    os.makedirs(os.path.join(nb, "logs", "checkpoints"), exist_ok=True)
    open(os.path.join(nb, "logs", "checkpoints", "world_model.pt"), "w").close()
    cfg = compose(["doom"], base, up2, games, seed_new_brain=True)
    check(cfg["world_model"]["deterministic_size"] != 256
          and len(cfg["parallel_envs"]["streams"]) == 2,
          "I. an existing brain with no record is NOT seeded (keeps its saved size)")
    sf = os.path.join(tmp, "enc_scale.json")
    json.dump({"size": "tiny", "dims": {"det": 256, "stoch": 16, "classes": 16,
                                        "encoder": 128}}, open(sf, "w"))
    G._write_scale(sf, "medium", None, {"from": "tiny", "to": "medium"}, False)
    d = json.load(open(sf))["dims"]
    check(d["det"] == 1024 and d["encoder"] == 128,
          "I. a resize keeps the brain's encoder width in the record (%s)" % d)

    # ---------------------------------------------------------------- H
    check(uni["infra"]["autoscaler"].get("actuate") is False,
          "H. actuation ships OFF (recommendations only until host-proven)")
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    ctrl = types.SimpleNamespace(_stop_requested=False)
    rf = os.path.join(tmp, "runlogs", "SCALE_RESTART")
    fake = types.SimpleNamespace(config={"infra": {"autoscaler": {"actuate": False,
                                                                   "restart_file": rf}}},
                                 _ll_ctrl=ctrl)
    d = {"action": "grow_model", "target_size": "medium", "target_streams": 3}
    check(not DevelopmentalAI._maybe_scale_restart(fake, d) and not os.path.exists(rf)
          and not ctrl._stop_requested, "H. actuate off: no marker, no stop")
    fake.config["infra"]["autoscaler"]["actuate"] = True
    check(DevelopmentalAI._maybe_scale_restart(fake, d) and ctrl._stop_requested
          and json.load(open(rf))["target_size"] == "medium",
          "H. actuate on: marker written and a GRACEFUL stop requested")
    ctrl._stop_requested = False
    check(not DevelopmentalAI._maybe_scale_restart(fake, {"action": "hold"})
          and not ctrl._stop_requested, "H. a hold never restarts")
    sup = open(os.path.join(ROOT, "scripts/supervise_skybot.sh")).read()
    i_scale = sup.index('if [ "$KIND" = "scale" ]')
    crashlog_write = sup.index('} >> "$CRASHLOG"')
    check(i_scale < crashlog_write and i_scale < sup.index("crash-loop guards"),
          "H. the supervisor relaunches kind=scale BEFORE the crash log and crash-loop guard")
    lau = open(os.path.join(ROOT, "scripts/launch_skybot.sh")).read()
    check(lau.index("--apply-pending") < lau.index("compose_run_config.py"),
          "H. the launcher resizes the brain BEFORE composing the run config")


if __name__ == "__main__":
    main()
