#!/usr/bin/env python3
"""Does world-model SIZE matter? A controlled RSSM size sweep on fixed Doom data.

docs/research/res-004-rssm-size-experiment.md (stage A). The question comes before any
"autoscaled / growing RSSM" work: if a 16x smaller world model learns Doom's
dynamics as well per step — or better per second — growing it buys nothing.

Controlled by construction:
  * ONE fixed dataset (sticky-random play of the real ViZDoom adapter, the
    exact game config SkyBot runs), split by EPISODE into train / held-out;
  * the REAL `WorldModel` with the Doom run's composed world_model settings
    (flow head, inverse dynamics, latent horizons, FiLM, coord channels...);
    only the size knobs change: deterministic_size, stochastic_size x classes,
    encoder_hidden;
  * every size sees the SAME batches in the SAME order (one sampler seed).

What it measures, on held-out sequences only (never trained on):
  recon        the model's own reconstruction loss (what training minimises)
  kl           posterior/prior KL (how surprised the dynamics are)
  open-loop    observe CONTEXT steps, then imagine HORIZON steps from the
               true actions and decode — pixel MSE vs the real frames. This is
               the imagination quality curiosity and dreaming depend on.
  baseline     the same MSE for "the last seen frame never changes", so a
               model that learned nothing scores ~1.0 relative to it.
and per size: parameters, seconds per gradient step.

Not measured here (needs the live loop on `main`, stage B of the plan):
behaviour — does a different size change exits / kills / maps reached.
Proprio and image sensors are left out (pixels + actions only): the size
question is about the latent, and it keeps every size on identical inputs.

    PYTHONPATH=. python tools/rssm_size_experiment.py collect --frames 24000 --out DATA.npz
    PYTHONPATH=. python tools/rssm_size_experiment.py train --data DATA.npz \\
        --sizes tiny,small,medium,live --steps 3000 --device mps --out RESULTS.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (deterministic_size, stochastic_size, stochastic_classes, encoder_hidden)
SIZES = {
    "tiny":   (256, 16, 16, 128),
    "small":  (512, 24, 24, 256),
    "medium": (1024, 32, 32, 512),
    "live":   (2048, 48, 48, 768),     # what SkyBot runs on main today
}
ACTIONS = 16                            # doom.yaml controller slots
ACTION_DIM = 48                         # universal.action_slots (one-hot width)
OVERRIDES: Dict = {}                    # settings changed from the live config (recorded)


# ------------------------------------------------------------------ collect
def collect(frames: int, out: str, seed: int, image_size: int = 128) -> None:
    from developmental_ai.environments.vizdoom_env import ViZDoomEnvAdapter
    game = yaml.safe_load(open(os.path.join(ROOT, "configs/games/doom.yaml")))["game"]
    game = dict(game, realtime=False, action_repeat=4)   # synchronous: fast + exact
    game["checkpoint"] = {"enabled": False}
    game["stuck"] = {"enabled": False}
    env = ViZDoomEnvAdapter(game, image_size=image_size, seed=seed)
    rng = np.random.default_rng(seed)
    obs_l, act_l, ep_l = [], [], []
    o, _ = env.reset()
    ep, a, hold = 0, 0, 0
    t0 = time.time()
    while len(obs_l) < frames:
        if hold <= 0:                   # sticky random: hold each choice 1-8 steps
            a, hold = int(rng.integers(ACTIONS)), int(rng.integers(1, 9))
        hold -= 1
        obs_l.append((np.asarray(o).reshape(3, image_size, image_size) * 255).astype(np.uint8))
        act_l.append(a)
        ep_l.append(ep)
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            o, _ = env.reset()
            ep += 1
        if len(obs_l) % 2000 == 0:
            print(f"  collected {len(obs_l)}/{frames} frames, {ep} episodes, "
                  f"{len(obs_l) / (time.time() - t0):.0f} fps", flush=True)
    env.close()
    np.savez_compressed(out, obs=np.stack(obs_l), act=np.asarray(act_l, np.int64),
                        ep=np.asarray(ep_l, np.int64))
    print(f"saved {out}: {len(obs_l)} frames, {ep + 1} episodes")


# ---------------------------------------------------------------- the model
def wm_cfg() -> Dict:
    """The world_model block SkyBot's Doom run composes (tools/compose_run_config)."""
    from tools.compose_run_config import compose
    cfg = compose(["doom"], os.path.join(ROOT, "configs/minecraft_skybot.yaml"),
                  os.path.join(ROOT, "configs/universal.yaml"),
                  os.path.join(ROOT, "configs/games"))
    return cfg["world_model"]


def build(size: str, image_size: int, device, core: str = "rssm", pc: dict = None):
    """The same constructor call the loop makes (developmental_loop.py ~612)."""
    import torch
    from developmental_ai.world_model.rssm import WorldModel
    det, st, cl, enc = SIZES[size]
    w = wm_cfg()
    if device.type == "mps" and w.get("flow_head"):
        # Apple's MPS backend cannot run the flow head: non-divisible area
        # pooling 128 -> 48 (pytorch#96056) and grid_sample's "border"
        # padding are both unimplemented. So on a Mac the flow head is OFF
        # for EVERY size alike (recorded in the results): the comparison is
        # still controlled, but it is the core RSSM + its other heads, not
        # the exact live model. The faithful run is on main (CUDA): --device cuda.
        w = dict(w, flow_head=False)
        OVERRIDES["flow_head"] = False
    pcx = dict(pc or {})
    if core == "pc_bp":                       # WP-20 ablation: settling kept, ordinary (backprop) networks
        core, pcx["pc_learning"] = "pc", "backprop"
    elif core == "pc":
        pcx.setdefault("pc_learning", "hebbian")
    torch.manual_seed(0)
    m = WorldModel(
        obs_dim=3 * image_size * image_size, action_dim=ACTION_DIM,
        stochastic_size=st, stochastic_classes=cl, deterministic_size=det,
        hidden_dim=enc, learning_rate=float(w.get("learning_rate", 3e-4)),
        pixel_obs=True, image_channels=3, image_size=image_size,
        film_conditioning=bool(w.get("film_conditioning", False)),
        inverse_dynamics=bool(w.get("inverse_dynamics", False)),
        inverse_dynamics_scale=float(w.get("inverse_dynamics_scale", 1.0)),
        discrete_actions=True,
        min_grid=int(w.get("spatial_grid", 4)),
        coord_channels=bool(w.get("coord_channels", False)),
        readout_channels=int(w.get("readout_channels", 0)),
        flow_head=bool(w.get("flow_head", False)),
        flow_size=int(w.get("flow_size", 32)),
        flow_photo_size=int(w.get("flow_photo_size", 0)),
        flow_weight=float(w.get("flow_weight", 1.0)),
        flow_smooth_weight=float(w.get("flow_smooth_weight", 0.05)),
        recon_residual_lambda=float(w.get("recon_residual_lambda", 0.0)),
        latent_horizons=w.get("latent_horizons") or [],
        latent_horizon_weight=float(w.get("latent_horizon_weight", 1.0)),
        flow_mode=str(w.get("flow_mode", "raw")),
        flow_automask=bool(w.get("flow_automask", False)),
        flow_scales=w.get("flow_scales") or [1],
        flow_strides=w.get("flow_strides") or [1],
        camera_fov_deg=float(w.get("camera_fov_deg", 70.0)),
        proprio_dim=0,
        # VRAM settings exactly as the live run passes them; inert off CUDA
        amp_dtype=w.get("amp_dtype"),
        grad_checkpoint=bool(w.get("grad_checkpoint", False)),
        # the posterior-collapse fix, as configs/universal.yaml sets it
        recon_reduction=str(w.get("recon_reduction", "mean")),
        inverse_input=str(w.get("inverse_input", "latent")),
        fused_adam=bool(w.get("fused_adam", False)),
        core=core, **pcx,                       # WP-20 (GOV-016): "rssm" (default) or "pc" (hebbian / backprop)
    ).to(device)
    m.action_shift = bool(w.get("causal_align", True))
    return m, int(w.get("batch_size", 8)), int(w.get("sequence_length", 32))


# ----------------------------------------------------------------- batches
def windows(ep: np.ndarray, seq: int, keep) -> np.ndarray:
    """Start indices of length-`seq` windows inside one episode, episodes in `keep`."""
    starts = []
    for e in np.unique(ep):
        if e not in keep:
            continue
        idx = np.nonzero(ep == e)[0]
        for s in range(idx[0], idx[-1] - seq + 2):
            starts.append(s)
    return np.asarray(starts, np.int64)


def make_batch(obs, act, starts, seq, device):
    import torch
    o = np.stack([obs[s:s + seq] for s in starts]).astype(np.float32) / 255.0
    a = np.zeros((len(starts), seq, ACTION_DIM), np.float32)
    for i, s in enumerate(starts):
        a[i, np.arange(seq), act[s:s + seq]] = 1.0
    B = len(starts)
    o = torch.from_numpy(o.reshape(B, seq, -1)).to(device)
    a = torch.from_numpy(a).to(device)
    r = torch.zeros(B, seq, device=device)
    c = torch.ones(B, seq, device=device)
    return o, a, r, c


def evaluate(m, obs, act, starts, seq, device, context=16, horizon=12) -> Dict[str, float]:
    import torch
    m.eval()
    rec, kl, ol, base = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 8):
            o, a, r, c = make_batch(obs, act, starts[i:i + 8], seq, device)
            L = m.compute_loss(o, a, r, c)
            rec.append(float(L["reconstruction"]))
            kl.append(float(L["kl"]))
            # open loop: observe `context` real steps, imagine `horizon` more
            states, _ = m.observe_sequence(o[:, :context], a[:, :context])
            st = {"h": states["h"][:, -1], "z": states["z"][:, -1]}
            preds = []
            for k in range(horizon):
                # CAUSAL ALIGNMENT (ASS-020 F-02, fixed 2026-10-25): a[:, t] is the action taken AT o_t, so it
                # carries o_t to o_(t+1); the state after observing frame context-1 needs a[:, context-1] to
                # predict frame context. (This read a[:, context + k] before: one frame late.)
                st = m.rssm.imagine_step(st, a[:, context - 1 + k])
                lat = torch.cat([st["h"], st["z"]], dim=-1)
                preds.append(m.decoder(lat).reshape(o.shape[0], -1))
            pred = torch.stack(preds, 1).clamp(0, 1)
            true = o[:, context:context + horizon]
            ol.append(float(((pred - true) ** 2).mean()))
            last = o[:, context - 1:context].expand_as(true)
            base.append(float(((last - true) ** 2).mean()))
    m.train()
    return {"recon": float(np.mean(rec)), "kl": float(np.mean(kl)),
            "open_loop_mse": float(np.mean(ol)), "copy_last_mse": float(np.mean(base)),
            "open_loop_rel": float(np.mean(ol) / max(1e-12, np.mean(base)))}


def act_ms(m, obs, act, device, streams: int = 2, reps: int = 100) -> float:
    """Milliseconds for ONE perception step of `streams` streams (no grad):
    the per-decision cost while acting, which is what limits steps/s."""
    import torch
    o, a, _, _ = make_batch(obs, act, np.arange(streams, dtype=np.int64), 1, device)
    m.eval()
    with torch.no_grad():
        for _ in range(10):
            m.observe_sequence(o, a)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            m.observe_sequence(o, a)
        if device.type == "cuda":
            torch.cuda.synchronize()
    m.train()
    return (time.time() - t0) / reps * 1000.0


def train(data: str, sizes: List[str], steps: int, eval_every: int, device_name: str,
          out: str, held_frac: float = 0.15, n_eval: int = 64, cores=("rssm",), pc: dict = None) -> None:
    import torch
    device = torch.device(device_name)
    if device.type == "mps":
        print("WARNING: Apple MPS diverged to NaN by step ~107 in the 2026-10-17 run "
              "while CPU stayed finite on the same batches; MPS numbers are smoke "
              "only. The real sweep runs on main: --device cuda.", flush=True)
    d = np.load(data)
    obs, act, ep = d["obs"], d["act"], d["ep"]
    eps = np.unique(ep)
    n_held = max(1, int(round(len(eps) * held_frac)))
    held, tr = set(eps[-n_held:].tolist()), set(eps[:-n_held].tolist())
    image_size = obs.shape[-1]
    results = {"data": data, "frames": int(len(obs)), "episodes": int(len(eps)),
               "held_episodes": n_held, "steps": steps, "device": device_name, "runs": {}}
    runs = [(size, core) for size in sizes for core in cores]
    for size_, core in runs:
        # a plain RSSM-only sweep keeps its historical key (the size); a comparison keys "size:core"
        size = size_ if tuple(cores) == ("rssm",) else f"{size_}:{core}"
        m, B, T = build(size_, image_size, device, core=core, pc=pc)
        tr_starts, ev_starts = windows(ep, T, tr), windows(ep, T, held)
        rng = np.random.default_rng(1234)                  # SAME batches for every size
        ev = ev_starts[np.random.default_rng(99).choice(len(ev_starts),
                                                        min(n_eval, len(ev_starts)),
                                                        replace=False)]
        # the PC core's locally-learned weights are buffers, not parameters: count them or G0 would flatter it
        params = sum(p.numel() for p in m.parameters()) + (m.rssm.local_values() if hasattr(m.rssm, "local_values") else 0)
        print(f"\n== {size}: {params / 1e6:.1f}M params, batch {B}x{T}, "
              f"{len(tr_starts)} train windows, {len(ev)} held-out windows", flush=True)
        curve, t_train = [], 0.0
        curve.append({"step": 0, **evaluate(m, obs, act, ev, T, device)})
        for s in range(1, steps + 1):
            o, a, r, c = make_batch(obs, act, tr_starts[rng.integers(len(tr_starts), size=B)],
                                    T, device)
            t0 = time.time()
            met = m.train_step(o, a, r, c)
            if device.type == "mps":
                torch.mps.synchronize()
            t_train += time.time() - t0
            # STOP ON DIVERGENCE (2026-10-17). The first Mac run logged NaN
            # for an hour: tiny and small diverged by step 200 on the Apple
            # GPU (every loss at once, step 107 in a repeat), while the SAME
            # batches stayed finite on CPU. A diverged size is recorded with
            # its step and the sweep moves on.
            bad = [k for k, v in met.items() if not np.isfinite(v)]
            if bad:
                print(f"  {size:6s} DIVERGED at step {s} ({', '.join(bad)} non-finite) "
                      f"on {device_name}", flush=True)
                results["runs"][size] = {"params": params, "dims": SIZES[size_], "core": core,
                                         "diverged_at": s, "curve": curve}
                break
            if s % eval_every == 0 or s == steps:
                e = evaluate(m, obs, act, ev, T, device)
                curve.append({"step": s, "train_seconds": round(t_train, 1), **e})
                print(f"  {size:6s} step {s:5d}  recon {e['recon']:.4f}  kl {e['kl']:.3f}  "
                      f"open-loop {e['open_loop_mse']:.5f} ({e['open_loop_rel']:.3f}x copy-last)  "
                      f"{t_train / s:.3f}s/step", flush=True)
        results["overrides"] = dict(OVERRIDES)
        results["world_model"] = {"recon_reduction": m.recon_reduction,
                                  "inverse_input": m.inverse_input}
        if "diverged_at" not in results["runs"].get(size, {}):
            results["runs"][size] = {"params": params, "dims": SIZES[size_], "core": core,
                                     "s_per_step": t_train / steps, "curve": curve,
                                     "act_ms_2_streams": act_ms(m, obs, act, device)}
            if getattr(m.rssm, "is_predictive_coding", False):
                results["runs"][size]["pc_refused"] = m.rssm.refused + m.rssm.hebbian_refused
                results["runs"][size]["pc_obs_var"] = float(m.rssm.obs_var)
            print(f"  {size:6s} acting: {results['runs'][size]['act_ms_2_streams']:.1f} ms "
                  f"per perception step for 2 streams", flush=True)
        if device.type == "cuda":
            results["runs"][size]["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
        del m
        if device.type == "mps":
            torch.mps.empty_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nwrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--frames", type=int, default=24000)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--sizes", default="tiny,small,medium,live")
    t.add_argument("--cores", default="rssm", help="comma list of rssm,pc,pc_bp (WP-20: pc = Hebbian, pc_bp = backprop ablation); more than one keys results size:core")
    t.add_argument("--pc-iters", type=int, default=4)
    t.add_argument("--pc-step", type=float, default=0.5)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--device", default="cpu")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.frames, a.out, a.seed)
    else:
        train(a.data, [s.strip() for s in a.sizes.split(",")], a.steps, a.eval_every,
              a.device, a.out, cores=tuple(c.strip() for c in a.cores.split(",")),
              pc={"pc_iters": a.pc_iters, "pc_step": a.pc_step})
    return 0


if __name__ == "__main__":
    sys.exit(main())
