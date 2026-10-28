#!/usr/bin/env python3
"""Resize SkyBot's world model between runs — grow exactly, shrink by relearning.

Stage D of docs/research/res-004-rssm-size-experiment.md; decisions come from
infra/autoscaler.py. Runs on the host, between runs (the supervisor calls it
through scripts/launch_skybot.sh when the loop asked for a scale restart).

  GROW (exact):   python tools/grow_brain.py --checkpoints DIR --to medium
      world_model.pt is grown function-preservingly (world_model/growth.py),
      VERIFIED against the old model on real frames from the saved replay
      buffer (random frames if there is none), and written only if every
      output matches within --tol. Nothing is thrown away.

  SHRINK (approximate):   python tools/grow_brain.py --checkpoints DIR --to small
      A smaller model cannot hold everything a bigger one does. It is RELEARNED
      from the saved replay buffer (the brain's own experience) and accepted
      only if its held-out reconstruction is within --shrink-tol of the bigger
      model's on the same held-out windows; otherwise the bigger model stays.

Either way:
  * the old files are kept (world_model.pt.pre_resize_<stamp>, arch too);
  * <scale-file> (default brain_universal/scale.json) records the new size
    and stream count; tools/compose_run_config.py applies it at every launch,
    so the next run BUILDS the size the checkpoint now has;
  * any OTHER resumed component whose tensors are sized to the old latent
    (symbolic_decoder.pt, glue_layer.pt) is moved aside to start fresh — the
    resume pre-flight checks shapes only for the world model, and a reshaped
    load anywhere else would leave a partial hybrid.

  --apply-pending reads runlogs/SCALE_RESTART (written by the loop), applies
  it, and removes it. A refused resize still removes it and says why, so a
  bad request cannot make every launch retry it (CLAUDE.md §4.1).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.infra.autoscaler import SIZE_LADDER, SIZE_NAMES       # noqa: E402
from developmental_ai.world_model.growth import (                          # noqa: E402
    ARCH_FILE, Dims, GrowthError, build_from_arch, dims_of, grow_state_dict,
    load_arch, save_arch, verify, grow_policy_state, grow_dream_state,
    verify_policy, verify_dream)

PENDING = os.path.join("runlogs", "SCALE_RESTART")
OTHER_RESUMED = ("symbolic_decoder.pt", "glue_layer.pt")


def rung(name: str) -> Dims:
    det, st, cl, _enc = dict(SIZE_LADDER)[name]
    return Dims(det, st, cl)


def nearest_rung(d: Dims) -> str:
    return min(SIZE_NAMES, key=lambda n: abs(rung(n).det - d.det))


# ------------------------------------------------------------- replay reader
def replay_windows(replay_dir: str, seq: int, n: int, seed: int,
                   newest: bool = False, oldest: bool = False
                   ) -> Optional[Dict[str, torch.Tensor]]:
    """n windows of length `seq` read from a saved replay buffer (memory-
    mapped, read-only). newest=True takes each stream's most recent windows
    (held-out for a relearned model, which is trained on the older part)."""
    streams = sorted(d for d in os.listdir(replay_dir)
                     if d.startswith("stream_")) if os.path.isdir(replay_dir) else []
    cols: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]] = []
    for s in streams:
        p = os.path.join(replay_dir, s)
        try:
            meta = json.load(open(os.path.join(p, "manifest.json")))
            ob = np.load(os.path.join(p, "observations.npy"), mmap_mode="r")
            ac = np.load(os.path.join(p, "actions.npy"), mmap_mode="r")
            pp = (np.load(os.path.join(p, "proprio.npy"), mmap_mode="r")
                  if os.path.isfile(os.path.join(p, "proprio.npy")) else None)
        except (OSError, ValueError):
            continue
        start = int(meta.get("start", 0))
        if len(ob) - start >= seq + 1:
            cols.append((ob, ac, pp, start))
    if not cols:
        return None
    rng = np.random.default_rng(seed)
    O, A, P = [], [], []
    for i in range(n):
        ob, ac, pp, start = cols[i % len(cols)]
        hi = len(ob) - seq
        if newest:
            s0 = hi - (i // len(cols)) * seq
            s0 = max(start, s0)
        elif oldest:                      # WP-10 retention probe: the oldest data
            s0 = min(hi, start + (i // len(cols)) * seq)
        else:
            s0 = int(rng.integers(start, max(start + 1, hi - 4 * seq)))
        O.append(np.asarray(ob[s0:s0 + seq]))
        A.append(np.asarray(ac[s0:s0 + seq]))
        if pp is not None:
            P.append(np.asarray(pp[s0:s0 + seq]))
    o = np.stack(O)
    o = o.astype(np.float32) / 255.0 if o.dtype == np.uint8 else o.astype(np.float32)
    out = {"observations": torch.from_numpy(o),
           "actions": torch.from_numpy(np.stack(A).astype(np.float32))}
    if P:
        out["proprio"] = torch.from_numpy(np.stack(P).astype(np.float32))
    return out


def random_windows(arch: Dict, n: int, seq: int, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, int(arch["action_dim"]), (n, seq), generator=g)
    out = {"observations": torch.rand(n, seq, int(arch["obs_dim"]), generator=g),
           "actions": torch.nn.functional.one_hot(a, int(arch["action_dim"])).float()}
    if int(arch.get("proprio_dim") or 0):
        out["proprio"] = torch.zeros(n, seq, int(arch["proprio_dim"]))
    return out


# --------------------------------------------------------------- the resize
def _backup(path: str, stamp: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    dst = f"{path}.pre_resize_{stamp}"
    shutil.copy2(path, dst)
    return dst


def _aside_latent_sized(ckpt: str, old: Dims, stamp: str) -> List[str]:
    sizes = {old.latent, old.det, old.z, 2 * old.latent}
    moved = []
    for name in OTHER_RESUMED:
        p = os.path.join(ckpt, name)
        if not os.path.isfile(p):
            continue
        try:
            sd = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        flat = sd if isinstance(sd, dict) else {}
        hit = False
        stack = [flat]
        while stack and not hit:
            d = stack.pop()
            for v in d.values():
                if isinstance(v, dict):
                    stack.append(v)
                elif torch.is_tensor(v) and any(int(x) in sizes for x in v.shape):
                    hit = True
                    break
        if hit:
            os.replace(p, f"{p}.pre_resize_{stamp}")
            moved.append(name)
    return moved


@torch.no_grad()
def _grow_readers(ckpt: str, old_wm, grown, old: Dims, new: Dims, b: Dict,
                  tol: float, seed: int, report: Dict) -> List[Tuple[str, object]]:
    """Grow policy.pt / dream_actor.pt to read the grown latent (2026-10-18).

    Verified on the SAME frames the world model was: the old policy on the
    old model's latents vs the grown policy on the grown model's latents.
    Returns [(filename, grown state)] for the readers that passed; one that
    fails or does not read a latent is reported and left untouched (it then
    starts fresh at the next launch, as before this change)."""
    old_wm.eval()
    grown.eval()
    so, _ = old_wm.observe_sequence(b["observations"], b["actions"], b.get("proprio"))
    sn, _ = grown.observe_sequence(b["observations"], b["actions"], b.get("proprio"))
    lo = torch.cat([so["h"], so["z"]], -1).reshape(-1, old.latent)
    ln = torch.cat([sn["h"], sn["z"]], -1).reshape(-1, new.latent)
    writes, readers = [], {}
    for name, grow_fn, check_fn in (
            ("policy.pt", grow_policy_state,
             lambda o, n: verify_policy(o, n, lo, ln, seed)),
            ("dream_actor.pt", grow_dream_state,
             lambda o, n: verify_dream(o, n, lo, ln))):
        p = os.path.join(ckpt, name)
        if not os.path.isfile(p):
            readers[name] = "absent"
            continue
        try:
            sd = torch.load(p, map_location="cpu", weights_only=False)
            g = grow_fn(sd, old, new, seed=seed)
            diff = check_fn(sd, g)
        except Exception as e:                      # GrowthError, shape, load
            readers[name] = f"left as is ({type(e).__name__}: {e})"
            continue
        if max(diff.values()) > tol:
            readers[name] = (f"left as is (grown copy differs: max |diff| "
                             f"{max(diff.values()):.2e} > {tol})")
            continue
        readers[name] = {"grown (exact)": diff}
        writes.append((name, g))
    report["readers"] = readers
    return writes


def resize(ckpt: str, target: str, scale_file: str, streams: Optional[int] = None,
           tol: float = 1e-4, shrink_tol: float = 1.10, shrink_steps: int = 2000,
           device: str = "cpu", dry_run: bool = False, seed: int = 0,
           expand_optim: bool = False, full_tol: bool = False,
           mech_tol: float = 0.25) -> Dict:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    policy_writes: List[Tuple[str, object]] = []
    if not os.path.isfile(os.path.join(ckpt, ARCH_FILE)):
        raise GrowthError(f"{ckpt}/{ARCH_FILE} missing — the run that wrote this "
                          f"checkpoint predates stage D; run once with the new code first")
    arch = load_arch(ckpt)
    old_wm = build_from_arch(arch)
    old_wm.load_state_dict(torch.load(os.path.join(ckpt, "world_model.pt"),
                                      map_location="cpu", weights_only=False))
    old, new = dims_of(old_wm), rung(target)
    replay = os.path.join(ckpt, "replay_buffer")
    report = {"from": nearest_rung(old), "to": target, "old": old.__dict__,
              "new": new.__dict__, "stamp": stamp}
    if new == old:
        report["result"] = "unchanged (already this size)"
    elif new.det >= old.det and new.stoch >= old.stoch and new.classes >= old.classes:
        grown = build_from_arch(arch, new)
        sd = grow_state_dict(old_wm.state_dict(), grown.state_dict(), old, new, seed=seed)
        grown.load_state_dict(sd, strict=True)
        b = (replay_windows(replay, 16, 8, seed) or random_windows(arch, 8, 16, seed))
        diff = verify(old_wm, grown, b["observations"], b["actions"], b.get("proprio"))
        report["verify_max_abs"] = diff
        if max(diff.values()) > tol:
            raise GrowthError(f"grown model does not match the old one "
                              f"(max |diff| {max(diff.values()):.2e} > {tol}): {diff}")
        report["result"] = "grown (exact)"
        new_wm = grown
        # the latent's READERS grow with it (2026-10-18): policy.pt and
        # dream_actor.pt, verified on the same frames before anything is
        # written; a reader that fails verification is left untouched
        policy_writes = _grow_readers(ckpt, old_wm, grown, old, new, b, tol,
                                      seed, report)
    else:
        new_wm, rep = _relearn_smaller(arch, old_wm, new, replay, shrink_steps,
                                       shrink_tol, device, seed, full_tol, mech_tol)
        report.update(rep)
        if new_wm is None:
            report["result"] = "shrink REFUSED — the bigger model stays"
    if dry_run or report["result"].startswith(("unchanged", "shrink REFUSED")):
        report["written"] = False
        if (report["result"].startswith("shrink REFUSED") and not dry_run
                and "shrink_held_out_recon" in report):
            _note_refusal(scale_file, report)      # WP-10: saturation evidence
        if not report["result"].startswith("shrink REFUSED") and streams is not None:
            _write_scale(scale_file, nearest_rung(old), streams, report, dry_run,
                         encoder=arch.get("hidden_dim"))
        return report
    report["backups"] = [x for x in (_backup(os.path.join(ckpt, "world_model.pt"), stamp),
                                     _backup(os.path.join(ckpt, ARCH_FILE), stamp)) if x]
    torch.save(new_wm.state_dict(), os.path.join(ckpt, "world_model.pt"))
    save_arch(new_wm, ckpt)
    for name, obj in policy_writes:
        p = os.path.join(ckpt, name)
        b_ = _backup(p, stamp)
        if b_:
            report["backups"].append(b_)
        torch.save(obj, p + ".tmp")
        os.replace(p + ".tmp", p)
    if policy_writes:
        # Adam's moments are shaped for the OLD input layer: a full resume
        # would load them and fail at the first step. Moved aside; Adam
        # restarts, the weights do not.
        _po = os.path.join(ckpt, "policy_optim.pt")
        if os.path.isfile(_po) and expand_optim:
            # WP-10: keep the moments; the loop expands them at resume
            from developmental_ai.world_model.growth_optim import write_sidecar
            _bo = _backup(_po, stamp)
            if _bo:
                report["backups"].append(_bo)
            write_sidecar(ckpt, old, new, stamp)
            report["optim"] = "kept; expanded at resume (sidecar written)"
        elif os.path.isfile(_po):
            os.replace(_po, f"{_po}.pre_resize_{stamp}")
            report.setdefault("moved_aside_optim", []).append("policy_optim.pt")
    report["moved_aside"] = _aside_latent_sized(ckpt, old, stamp)
    report["written"] = True
    _write_scale(scale_file, target, streams, report, False,
                 encoder=arch.get("hidden_dim"))
    return report


def _note_refusal(scale_file, report) -> None:
    """Record a shrink refused by the tolerance in scale.json (read by
    infra/scale_order.py as 'the next smaller rung failed': saturation
    evidence, ADR-005 Amendment 1). Expires there; never blocks a grow."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(scale_file)), exist_ok=True)
        cur = {}
        if os.path.isfile(scale_file):
            try:
                cur = json.load(open(scale_file))
            except (OSError, ValueError):
                cur = {}
        ref = cur.setdefault("refused_shrinks", {})
        ref[report["to"]] = {"from": report["from"], "to": report["to"],
                             "t": time.time(), "stamp": report.get("stamp"),
                             "evidence": {k: report[k] for k in
                                          ("shrink_held_out_recon", "shrink_mechanism",
                                           "shrink_retention") if k in report}}
        tmp = scale_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=1, default=str)
        os.replace(tmp, scale_file)
    except OSError:
        pass


def _write_scale(scale_file, size, streams, report, dry, encoder=None):
    if dry:
        return
    os.makedirs(os.path.dirname(os.path.abspath(scale_file)), exist_ok=True)
    cur = {}
    if os.path.isfile(scale_file):
        try:
            cur = json.load(open(scale_file))
        except (OSError, ValueError):
            cur = {}
    dims = dict(rung(size).__dict__)
    # growth/shrink keep the encoder width (growth.py widens the latent only),
    # so record the width the brain HAS, or the next launch builds another
    enc = encoder or (cur.get("dims") or {}).get("encoder")
    if enc:
        dims["encoder"] = int(enc)
    cur.update({"size": size, "dims": dims})
    if streams is not None:
        cur["streams"] = int(streams)
    cur["last_resize"] = {k: report[k] for k in ("from", "to", "result", "stamp")
                          if k in report}
    tmp = scale_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=1)
    os.replace(tmp, scale_file)


def _relearn_smaller(arch, big, new: Dims, replay: str, steps: int, tol: float,
                     device: str, seed: int, full: bool = False,
                     mech_tol: float = 0.25):
    """Train a smaller model on the brain's own saved experience; accept it
    only if its held-out reconstruction is within `tol` x the big one's."""
    seq = 32
    held = replay_windows(replay, seq, 16, seed, newest=True)
    if held is None:
        return None, {"shrink": "no saved replay buffer to relearn from"}
    dev = torch.device(device)
    small = build_from_arch(arch, new).to(dev)
    big = big.to(dev)

    def held_recon(m):
        m.eval()
        with torch.no_grad():
            o, a = held["observations"].to(dev), held["actions"].to(dev)
            p = held.get("proprio")
            L = m.compute_loss(o, a, torch.zeros(o.shape[:2], device=dev),
                               torch.ones(o.shape[:2], device=dev),
                               proprio=None if p is None else p.to(dev))
        m.train()
        return float(L["reconstruction"])

    ref = held_recon(big)
    for i in range(steps):
        b = replay_windows(replay, seq, 8, seed + 1 + i)
        o, a = b["observations"].to(dev), b["actions"].to(dev)
        p = b.get("proprio")
        small.train_step(o, a, torch.zeros(o.shape[:2], device=dev),
                         torch.ones(o.shape[:2], device=dev),
                         proprio=None if p is None else p.to(dev))
    got = held_recon(small)
    rep = {"shrink_held_out_recon": {"big": ref, "small": got, "tol": tol,
                                     "steps": steps}}
    if got > ref * tol:
        return None, rep
    if full:
        # ADR-005 Amendment 2: the loss is not the only measure. MECHANISM =
        # action attribution (rolled-action KL minus true-action KL) on the
        # same held-out windows; RETENTION = reconstruction on the OLDEST
        # windows. Either regressing beyond its tolerance refuses the shrink.
        # (mech_tol and `tol` stand in for the A/A noise floor until measured.)
        def attribution(m):
            m.eval()
            S, C = m.rssm.stochastic_size, m.rssm.stochastic_classes
            o, a = held["observations"].to(dev), held["actions"].to(dev)
            p = held.get("proprio")
            p = None if p is None else p.to(dev)

            def kl(act):
                with torch.no_grad():
                    _, inf = m.observe_sequence(o, act, p)
                    lsm = torch.nn.functional.log_softmax
                    pri = lsm(inf["prior_logits"].reshape(-1, S, C).float(), -1)
                    post = lsm(inf["posterior_logits"].reshape(-1, S, C).float(), -1)
                    return float((post.exp() * (post - pri)).sum(-1).sum(-1).mean())
            g = kl(torch.roll(a, 1, dims=1)) - kl(a)
            m.train()
            return g
        gb, gs = attribution(big), attribution(small)
        rep["shrink_mechanism"] = {"attribution_big": gb, "attribution_small": gs,
                                   "tol": mech_tol}
        old_w = replay_windows(replay, seq, 16, seed, oldest=True)
        if old_w is not None:
            held_new, held = held, old_w
            rb, rs = held_recon(big), held_recon(small)
            held = held_new
            rep["shrink_retention"] = {"big": rb, "small": rs, "tol": tol}
            if rs > rb * tol:
                return None, rep
        if gb > 0 and gs < gb * (1.0 - mech_tol):
            return None, rep
    rep["result"] = f"shrunk (relearned in {steps} steps, held-out {got:.4f} vs {ref:.4f})"
    return small.cpu(), rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints", default="brain_universal/logs/checkpoints")
    ap.add_argument("--scale-file", default="brain_universal/scale.json")
    ap.add_argument("--to", choices=SIZE_NAMES)
    ap.add_argument("--streams", type=int)
    ap.add_argument("--apply-pending", action="store_true",
                    help=f"apply and remove {PENDING} (written by the loop)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--shrink-tol", type=float, default=1.10)
    ap.add_argument("--shrink-steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--expand-optim", action="store_true",
                    default=os.environ.get("SKYBOT_GROW_EXPAND_OPTIM", "") == "1",
                    help="WP-10: keep the policy's Adam moments across a grow "
                         "(expanded at resume) instead of moving them aside")
    ap.add_argument("--full-tolerance", action="store_true",
                    default=os.environ.get("SKYBOT_SHRINK_FULL_TOL", "") == "1",
                    help="WP-10: a shrink must also keep action attribution and "
                         "retention on the oldest windows (ADR-005 Amendment 2)")
    ap.add_argument("--mech-tol", type=float, default=0.25)
    a = ap.parse_args(argv)
    target, streams = a.to, a.streams
    if a.apply_pending:
        if not os.path.isfile(PENDING):
            print("grow_brain: nothing pending")
            return 0
        try:
            req = json.load(open(PENDING))
        except (OSError, ValueError) as e:
            os.remove(PENDING)
            print(f"grow_brain: unreadable {PENDING} ({e}) — removed, not applied")
            return 0
        target = req.get("target_size") or target
        streams = req.get("target_streams", streams)
    if not target:
        ap.error("--to (or --apply-pending with a request) is required")
    try:
        rep = resize(a.checkpoints, target, a.scale_file, streams, a.tol, a.shrink_tol,
                     a.shrink_steps, a.device, a.dry_run,
                     expand_optim=a.expand_optim, full_tol=a.full_tolerance,
                     mech_tol=a.mech_tol)
        rc = 0
    except (GrowthError, OSError) as e:
        rep, rc = {"result": f"REFUSED: {e}"}, 1
    finally:
        if a.apply_pending and os.path.isfile(PENDING):
            os.remove(PENDING)            # applied or refused: never retried forever
    print("GROW-BRAIN " + json.dumps(rep, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
