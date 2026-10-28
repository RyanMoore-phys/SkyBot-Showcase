"""WP-20: the predictive-coding core runs inside the REAL loop, not only as a model (GOV-016 section 11g).

Born 2026-10-28. Every PC test so far builds a `WorldModel` or runs the pilot's own training loop. The first boot of the WP-20 arm in
the real trainer ended "agent never came up" (the boot check hid it, 11g), and no test could have said why: none constructs
`DevelopmentalAI` with `world_model.core: pc`, acts with it, trains it through the loop's own path, or publishes it to the acting
copy. This does, on Crafter (fast, no game engine), with the async trainer's inference copy on as in `configs/universal.yaml`.

CONTRACTS
  A. THE LOOP BUILDS WITH THE PC CORE: construction succeeds, the world model's core is the PC class, its local networks are buffers.
  B. IT ACTS: `_collect_segment` runs for three segments (the settle runs inside the acting path under no_grad) and the rollout and replay hold rows.
  C. IT TRAINS THROUGH THE LOOP'S OWN PATH: `_train_world_model()` returns finite metrics, no local update was refused, and the local
     weights MOVED (the Hebbian update ran, not only the optimiser).
  D. THE ACTING COPY RECEIVES THE LOCAL WEIGHTS: with the inference copy on, `_publish_wm()` makes the acting model's local networks equal the
     trainer copy's (they are buffers: a parameters-only copy would leave the acting model on its initial random networks and look healthy).
  E. A CHECKPOINT CAN BE WRITTEN AND READ BACK: `_save_checkpoint_locked()` writes world_model.pt and the architecture file, and
     the saved architecture names the core (`pc`) and its settings, so a resume or a grow knows what it is looking at.
  F. FALSIFIER OF THE HARNESS: the same run with `core: rssm` builds the plain RSSM and reports no Hebbian state (the test sees the difference).

Run: PYTHONPATH=. python tests/_pc_loop_smoke.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

NAME = "pc-loop"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def make(core, tmp):
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8, "deterministic_size": 64,
                               "encoder_hidden": 64, "decoder_hidden": 64, "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 4000, "core": core})
    if core == "pc":
        cfg["world_model"].update({"pc_iters": 2, "pc_step": 0.5, "pc_weight": 1.0, "pc_learning": "hebbian",
                                   "pc_hebbian_lr": 0.01, "pc_hebbian_iters": 6})
    cfg["environment"]["max_episode_steps"] = 200
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["lifelong"] = {"enabled": True, "forever": False, "segment_len": 40, "wm_train_every": 10 ** 9, "goal_horizon": 10 ** 9,
                       "state_decay": 1.0, "reset_on_death_only": True, "stop_file": os.path.join(tmp, "STOP")}
    cfg["policy"].update({"n_steps": 10 ** 9, "minibatch_size": 8, "min_rows_per_update": 0, "max_env_steps_per_update": 0,
                          "target_kl": 0.05})
    cfg["skill_bank"]["storage_dir"] = os.path.join(tmp, "sb")
    cfg["loop"]["verbose"] = 0
    cfg["loop"]["log_dir"] = os.path.join(tmp, "logs")
    cfg["async_wm"] = {"enabled": True, "inference_copy": True, "publish_every": 2}     # as configs/universal.yaml
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    return DevelopmentalAI(config=cfg)


def locals_of(wm):
    r = wm.rssm
    return {n: [getattr(net, k).detach().clone() for k in ("W1", "b1", "W2", "b2")]
            for n, net in (("prior", r.prior_net), ("posterior", r.posterior_net), ("generative", r.obs_pred_net))}


def main():
    import torch
    tmp = tempfile.mkdtemp(prefix="pc_loop_smoke_")
    ai = None
    try:
        ai = make("pc", tmp)
        r = ai.world_model.rssm
        check(getattr(r, "is_predictive_coding", False) and r.hebbian and hasattr(r.prior_net, "W1")
              and not any(True for _ in r.prior_net.parameters()),
              "A. the loop built the predictive-coding core with Hebbian (buffer) local networks")

        for _ in range(3):
            ai._collect_segment()
        rows = len(ai.policy.rollout_obs)
        check(rows > 0, "B. three segments ran through the settle: %d rollout rows" % rows)

        ai._ensure_wm_trainer()                                    # creates the trainer's copy (async_wm.inference_copy) as the loop does
        trainer = ai._wm_train
        check(trainer is not None, "C. the inference copy exists (async_wm.inference_copy on)")
        w0 = locals_of(trainer)
        m = ai._train_world_model()
        check(m and all(np.isfinite(float(v)) for v in m.values() if isinstance(v, (int, float, np.floating))),
              "C. the loop's own training path returned finite metrics (%d keys)" % len(m or {}))
        w1 = locals_of(trainer)
        moved = [n for n in w0 if any(not torch.equal(a, b) for a, b in zip(w0[n], w1[n]))]
        check(trainer.rssm.hebbian_refused == 0, "C. no local update was refused")
        check(sorted(moved) == ["generative", "posterior", "prior"],
              "C. the local weights of the model the loop trained all moved (moved: %s)" % moved)

        with torch.no_grad():                                      # make the trainer's local weights differ from the acting copy's
            trainer.rssm.prior_net.W2.add_(0.5)
        ai._publish_wm()
        a, t = locals_of(ai.world_model), locals_of(trainer)
        check(all(torch.equal(x, y) for n in a for x, y in zip(a[n], t[n])),
              "D. after a publish the acting model's local networks equal the trainer copy's (buffers are published)")

        ai._save_checkpoint_locked()
        cdir = os.path.join(tmp, "logs", "checkpoints")
        from developmental_ai.world_model.growth import ARCH_FILE, load_arch
        arch = load_arch(cdir)
        check(os.path.exists(os.path.join(cdir, "world_model.pt")) and arch.get("core") == "pc" and arch.get("pc_learning") == "hebbian",
              "E. the checkpoint and the architecture file were written, and the architecture names core=pc, hebbian (%s)" % ARCH_FILE)
        sd = torch.load(os.path.join(cdir, "world_model.pt"), map_location="cpu")
        check(any("prior_net.W1" in k for k in sd), "E. the saved state holds the local networks (they are buffers)")
    finally:
        try:
            if ai is not None:
                ai.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    tmp = tempfile.mkdtemp(prefix="pc_loop_smoke_ref_")
    ai = None
    try:
        ai = make("rssm", tmp)
        check(not getattr(ai.world_model.rssm, "is_predictive_coding", False) and not hasattr(ai.world_model.rssm, "hebbian_update"),
              "F. falsifier: core=rssm builds the plain RSSM, with no Hebbian state, in the same harness")
    finally:
        try:
            if ai is not None:
                ai.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
