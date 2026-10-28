"""Contracts for Consolidator.strength reaching the update (GOV-014 item B2; ASS-014 F-04).

    PYTHONPATH=. python tests/_consolidation_strength_smoke.py

THE INCIDENT. `Consolidator.run` multiplied the distillation and EWC losses by `strength` and then
stepped Adam. Adam's update is invariant to a constant loss scale, so with AWR off the update norms
at strength 0.25, 1 and 4 were equal to 0.001% (categorical cat02 CT-01). The RW-1 -> consolidation
chain (and RW-3's four factors) could not change what the phase does. The fix also scales the
optimizer's learning rate by `strength`, with the base lr captured ONCE at optimizer creation (not
lazily from the live param group, which is the F-19 pattern that compounds across a restore).

CONTRACTS
  A  the update norm of one phase rises monotonically with strength (AWR off), 4x vs 0.25x by more than 4
  B  strength 1.0 is the previous behaviour: lr is exactly policy lr x lr_scale
  C  strength is not a latch: after 4.0 then 1.0 the lr is back at the base, not compounded
  D  the base lr is captured once: changing the policy optimizer's lr afterwards does not move it
  E  a non-finite or non-positive strength falls back to 1.0 (no NaN lr)
  F  falsification: with the scaling disabled the spread collapses (the measured defect)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.policy.actor_critic import StandaloneActorCritic  # noqa: E402
from developmental_ai.policy.consolidation import Consolidator  # noqa: E402
from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer  # noqa: E402

FAILS = []
DEV = torch.device("cpu")
NA, OBS, NS = 4, 10, 3
CFG = dict(enabled=True, every_segments=1, steps=2, batch=4, seq_len=4, lr_scale=0.1,
           distill=dict(enabled=True, snapshot_every_segments=100, max_snapshots=2),
           regularizer={"kind": "ewc", "lambda": 100.0, "anchor_every_segments": 100, "fisher_batches": 1})


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def fixture(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = MultiStreamReplayBuffer(NS, 3000, OBS, NA)
    rs = np.random.RandomState(7)
    for s in range(NS):
        for t in range(160):
            o = rs.rand(OBS).astype(np.float32)
            o[0] = (s * 10000 + t) / 100000.0
            buf.add(o, int(rs.randint(NA)), 0.0, False, stream=s)
    pol = StandaloneActorCritic(OBS, NA, hidden_dim=16, knowledge_dim=0, knowledge_gate_mode="open", device=DEV)
    return pol, buf


def update_norm(strength):
    pol, buf = fixture()
    c = Consolidator(CFG)
    c.strength = 1.0
    c.run(pol, None, buf, DEV)                      # takes the first snapshot / anchor
    with torch.no_grad():
        for q in pol.actor.parameters():            # move the policy so the retention terms pull back
            q.add_(0.05 * torch.ones_like(q))
    base = [q.detach().clone() for q in pol.actor.parameters()]
    torch.manual_seed(2)
    np.random.seed(2)
    c.strength = strength
    c.run(pol, None, buf, DEV)
    return sum(float((q - b).norm()) for q, b in zip(pol.actor.parameters(), base))


def main() -> int:
    print("A the update follows strength")
    mv = {s: update_norm(s) for s in (0.25, 1.0, 4.0)}
    check("monotonic in strength", mv[0.25] < mv[1.0] < mv[4.0], mv)
    check("4x moves the policy more than 4x as far as 0.25x", mv[4.0] / max(mv[0.25], 1e-12) > 4.0, mv)

    print("B strength 1.0 is unchanged")
    pol, buf = fixture()
    c = Consolidator(CFG)
    c.run(pol, None, buf, DEV)
    want = float(pol.optimizer.param_groups[0]["lr"]) * CFG["lr_scale"]
    check("lr == policy lr x lr_scale exactly", c._opt.param_groups[0]["lr"] == want, (c._opt.param_groups[0]["lr"], want))

    print("C not a latch")
    c.strength = 4.0
    c.run(pol, None, buf, DEV)
    scaled = c._opt.param_groups[0]["lr"]
    c.strength = 1.0
    c.run(pol, None, buf, DEV)
    check("scaled while strength is 4", abs(scaled - 4.0 * want) < 1e-12, scaled)
    check("back at the base after strength returns to 1 (no compounding)", c._opt.param_groups[0]["lr"] == want)

    print("D base captured once")
    for g in pol.optimizer.param_groups:
        g["lr"] = 123.0                              # the policy's own lr changes afterwards
    c.strength = 2.0
    c.run(pol, None, buf, DEV)
    check("the consolidator's base did not follow it", abs(c._opt.param_groups[0]["lr"] - 2.0 * want) < 1e-12,
          c._opt.param_groups[0]["lr"])

    print("E a bad strength is neutral")
    for bad in (float("nan"), float("inf"), 0.0, -3.0):
        c.strength = bad
        c.run(pol, None, buf, DEV)
        lr = c._opt.param_groups[0]["lr"]
        check(f"strength {bad} -> finite base lr", lr == want, lr)

    print("F falsification: scaling disabled reproduces the defect")
    real = Consolidator._scale_lr
    Consolidator._scale_lr = lambda self: None
    try:
        off = {s: update_norm(s) for s in (0.25, 4.0)}
    finally:
        Consolidator._scale_lr = real
    spread = abs(off[4.0] - off[0.25]) / max(off[0.25], 1e-12)
    check("without the scaling the 4x and 0.25x updates are equal to within 1% (the measured defect)", spread < 0.01, off)

    print(f"[consolidation-strength] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
