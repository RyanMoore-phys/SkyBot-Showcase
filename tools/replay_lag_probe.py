#!/usr/bin/env python3
"""Which action does a replay row's interval actually reflect? (2026-10-18)

An inverse-dynamics probe on a SAVED replay buffer, independent of the world
model: for every usable row t, features = f(o_{t+1}) - f(o_t) plus
position-independent motion features of the pair (f = a small
grayscale downsample of the frame; the CHANGE, because a single frame can
encode the state the previous action left behind, which would confound the
shifts — `--with-frame` adds f(o_t) back), and a softmax classifier is
fitted to predict the action label stored at row t+s for s in -1, 0, +1. The
shift with the best held-out accuracy is the one the frames agree with:

  * synchronous game (Minecraft), or real-time with the credit fix: peak at 0
  * real-time buffer saved BEFORE the fix (core/realtime_credit.py): the
    interval t -> t+1 was driven by a_{t-1}, so the peak is at -1

It gates `realtime_credit.legacy_replay: relabel` (only relabel a buffer the
probe shows lagged) and checks a fixed run's new buffer (peak at 0).

    python tools/replay_lag_probe.py <replay_dir> [--streams 0,1] [--rows 6000]

`replay_dir` holds stream_<k>/ (MultiStreamReplayBuffer.save) or is a single
stream directory. Prints one line per stream and a final JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load_stream(d: str, max_rows: int):
    meta = json.load(open(os.path.join(d, "manifest.json")))
    obs = np.load(os.path.join(d, "observations.npy"), mmap_mode="r")
    act = np.load(os.path.join(d, "actions.npy"), mmap_mode="r")
    done = np.load(os.path.join(d, "dones.npy"), mmap_mode="r")
    rp = os.path.join(d, "restarts.npy")
    rst = np.load(rp, mmap_mode="r") if os.path.isfile(rp) else None
    start = int(meta.get("start", 0) or 0)
    n = int(obs.shape[0])
    lo = max(start, n - int(max_rows))
    return meta, obs[lo:n], act[lo:n], done[lo:n], (None if rst is None else rst[lo:n])


def _feat(obs_rows: np.ndarray, side: int = 16) -> np.ndarray:
    """(N, 3*S*S) uint8/float frames -> (N, side*side) grayscale in [0,1]."""
    N = obs_rows.shape[0]
    S = int(round((obs_rows.shape[1] / 3) ** 0.5))
    x = np.asarray(obs_rows, dtype=np.float32).reshape(N, 3, S, S)
    if x.max() > 1.5:
        x = x / 255.0
    g = x.mean(axis=1)
    k = max(1, S // side)
    g = g[:, :k * side, :k * side].reshape(N, side, k, side, k).mean(axis=(2, 4))
    return g.reshape(N, -1)


def _motion(Fa: np.ndarray, Fb: np.ndarray, side: int = 16) -> np.ndarray:
    """Position-independent motion features of each frame pair: correlation
    of frame b with frame a shifted by dx in [-4, 4] (turning / strafing)
    and dy in [-2, 2], plus mean and mean-absolute change (firing flashes,
    walking). A linear classifier cannot read 'the image moved right' off a
    raw difference; it can off these."""
    A = Fa.reshape(-1, side, side)
    B = Fb.reshape(-1, side, side)
    A = A - A.mean(axis=(1, 2), keepdims=True)
    B = B - B.mean(axis=(1, 2), keepdims=True)
    out = []
    for dx in range(-4, 5):
        a = np.roll(A, dx, axis=2)[:, :, 4:-4]
        b = B[:, :, 4:-4]
        out.append((a * b).mean(axis=(1, 2))
                   / (a.std(axis=(1, 2)) * b.std(axis=(1, 2)) + 1e-6))
    for dy in range(-2, 3):
        a = np.roll(A, dy, axis=1)[:, 2:-2, :]
        b = B[:, 2:-2, :]
        out.append((a * b).mean(axis=(1, 2))
                   / (a.std(axis=(1, 2)) * b.std(axis=(1, 2)) + 1e-6))
    d = Fb - Fa
    out += [d.mean(1), np.abs(d).mean(1)]
    return np.stack(out, 1).astype(np.float32)


def _softmax_fit(X, y, K, Xv, yv, steps=300, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xv = (X - mu) / sd, (Xv - mu) / sd
    W = np.zeros((X.shape[1], K), np.float32)
    b = np.zeros(K, np.float32)
    Y = np.eye(K, dtype=np.float32)[y]
    for _ in range(steps):
        z = X @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(X)
        W -= lr * (X.T @ g + l2 * W)
        b -= lr * g.sum(0)
    return float(((Xv @ W + b).argmax(1) == yv).mean())


def probe_stream(d: str, max_rows: int = 6000, seed: int = 0,
                 with_frame: bool = False):
    meta, obs, act, done, rst = _load_stream(d, max_rows)
    lab = np.asarray(act).argmax(1)
    N = len(lab)
    bad = np.asarray(done) > 0.5
    if rst is not None:
        bad |= np.asarray(rst).astype(bool)
    F = _feat(np.asarray(obs))
    # usable t: t-1, t, t+1 inside the window and no boundary at t-1..t+1
    ok = np.zeros(N, bool)
    ok[1:N - 2] = True
    for s in (-1, 0, 1):
        idx = np.arange(N) + s
        ok &= ~bad[np.clip(idx, 0, N - 1)]
    t = np.nonzero(ok)[0]
    if len(t) < 200:
        return {"dir": d, "rows": int(len(t)), "error": "too few usable rows"}
    X = np.concatenate([F[t + 1] - F[t], _motion(F[t], F[t + 1])], axis=1)
    if with_frame:
        X = np.concatenate([F[t], X], axis=1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(t))
    cut = int(0.7 * len(t))
    tr, va = perm[:cut], perm[cut:]
    K = int(act.shape[1])
    out = {"dir": d, "rows": int(len(t)),
           "semantics": meta.get("action_semantics", "unstamped")}
    for s in (-1, 0, 1):
        y = lab[t + s]
        maj = float(np.bincount(y[va], minlength=K).max() / max(1, len(va)))
        acc = _softmax_fit(X[tr], y[tr], K, X[va], y[va])
        out[f"acc_{s:+d}"] = round(acc, 4)
        out[f"chance_{s:+d}"] = round(maj, 4)
    out["peak_shift"] = max((-1, 0, 1), key=lambda s: out[f"acc_{s:+d}"]
                            - out[f"chance_{s:+d}"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("replay_dir")
    ap.add_argument("--streams", default="")
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--with-frame", action="store_true")
    a = ap.parse_args(argv)
    d = a.replay_dir
    if os.path.isfile(os.path.join(d, "manifest.json")):
        dirs = [d]
    else:
        ks = sorted(int(x.split("_")[1]) for x in os.listdir(d)
                    if x.startswith("stream_")
                    and os.path.isfile(os.path.join(d, x, "manifest.json")))
        if a.streams:
            want = {int(x) for x in a.streams.split(",")}
            ks = [k for k in ks if k in want]
        dirs = [os.path.join(d, f"stream_{k}") for k in ks]
    res = []
    for sd in dirs:
        r = probe_stream(sd, a.rows, with_frame=a.with_frame)
        res.append(r)
        print("LAG-PROBE", json.dumps(r))
    print("LAG-PROBE-SUMMARY", json.dumps(
        {"peaks": [r.get("peak_shift") for r in res]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
