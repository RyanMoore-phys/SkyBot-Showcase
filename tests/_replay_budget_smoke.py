"""Replay MEMORY BUDGET smoke (2026-10-18): the budget counts every byte.

THE LIVE INCIDENT THIS ENCODES

    Mock deploy on `main` (2026-10-18), universal brain at tiny x 8 Doom
    streams, resumed from the 2-stream brain. The log said

        replay memory budget: 2.0 GB
        replay buffer AT ITS MEMORY CEILING at 25000 transitions (2.0 GB budget)

    and the memory census measured the replay at 5.60 GB, agent RSS 9.5 GB
    steady (11.2 GB peak) on a 16 GB box, swap in use. Two bytes the budget
    never saw:
      * each stream's FIRST block is allocated without asking:
        8 x 5000 rows x 69.9 KB = 2.8 GB, already past 2.0 GB before step 1;
      * the grow request left out the proprio column (5170 float32 =
        20.7 KB of a 69.9 KB row), so every grown block reserved ~70 % of
        what it allocated.
    CLAUDE.md §6 had recorded the first one since 2026-09-22 ("a 2 GB
    ceiling measured 3.70 GB"); going from 2 streams to 8 quadrupled it.

Contracts:
    A. EXACT ACCOUNTING. A growable buffer with a proprio column: right after
       construction, and after every grow, MEMORY_BUDGET.reserved equals the
       bytes its column blocks actually hold (summed here from .nbytes).
    B. N FIRST BLOCKS FIT. 8 streams whose configured first blocks would
       exceed the budget get a smaller block, so the held bytes are <= the
       limit (the old sizing is computed alongside as the witness: > limit).
       2 streams under the same budget keep the configured block unchanged.
    C. THE CEILING HOLDS UNDER GROWTH AND RESTORE. Growing every stream until
       refused, and loading a larger saved buffer into it, never takes the
       held bytes past the limit, and reserved == held throughout. The
       restore keeps the NEWEST rows.
    D. The real `main` row: bytes_per_row(49152 uint8, 12, 5170) x 5000 x 8
       = 2.79 GB, the number in the incident.

Run: PYTHONPATH=. python tests/_replay_budget_smoke.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")

import logging

import numpy as np

logging.disable(logging.WARNING)

from developmental_ai.world_model import replay_buffer as rb  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def reset_budget():
    rb.MEMORY_BUDGET.limit = 0
    rb.MEMORY_BUDGET.reserved = 0
    rb.MEMORY_BUDGET.source = "unset"


# scaled shapes with the same proportions as main's row (obs:proprio ~ 2.4:1)
OD, AD, PD = 4915, 12, 517


def growth(gb, block=5000):
    return {"enabled": True, "block_transitions": block,
            "grow_at_frac": 0.85, "max_ram_frac": 1.0, "hard_max_gb": gb}


def held(buf):
    streams = buf.streams if hasattr(buf, "streams") else [buf]
    n = 0
    for s in streams:
        for c in ("observations", "actions", "rewards", "dones",
                  "priorities", "restarts", "proprio"):
            col = getattr(s, c, None)
            if col is not None:
                n += sum(b.nbytes for b in col._blocks)
    return n


def fill(s, n, base=0):
    for i in range(n):
        s.add(np.full(OD, (base + i) % 251, np.float32) / 255.0,
              np.zeros(AD, np.float32), 0.0, False,
              proprio=np.full(PD, float(base + i), np.float32))


def test_exact():
    print("A. exact accounting")
    reset_budget()
    b = rb.ReplayBuffer(capacity=2000, obs_dim=OD, action_dim=AD,
                        obs_uint8=True, growth=growth(1.0, block=1000),
                        proprio_dim=PD)
    check(rb.MEMORY_BUDGET.reserved == held(b) > 0,
          f"after construction reserved {rb.MEMORY_BUDGET.reserved:,} == "
          f"held {held(b):,} (first block counted, proprio included)")
    grew = 0
    for _ in range(3):
        fill(b, 900, base=grew * 1000)
        b._grow_pending = True
        grew += bool(b.maybe_grow())
        check(rb.MEMORY_BUDGET.reserved == held(b),
              f"after grow {grew}: reserved == held ({held(b):,} B)")
    check(grew == 3, f"grew 3 times ({grew})")


def test_first_blocks_fit():
    print("B. N first blocks fit the budget")
    row = rb.bytes_per_row(OD, AD, PD, True)
    limit_gb = 0.20
    old = 8 * 5000 * row
    check(old > limit_gb * 1e9,
          f"witness: old sizing 8 x 5000 rows = {old / 1e9:.3f} GB > "
          f"{limit_gb} GB limit")
    reset_budget()
    m = rb.MultiStreamReplayBuffer(8, 8 * 5000, OD, AD, obs_uint8=True,
                                   growth=growth(limit_gb), proprio_dim=PD)
    lim = rb.MEMORY_BUDGET.limit
    blk = m.streams[0]._block
    check(blk < 5000 and held(m) <= lim,
          f"8 streams: block {blk} rows, held {held(m) / 1e9:.3f} GB <= "
          f"{lim / 1e9:.2f} GB")
    check(rb.MEMORY_BUDGET.reserved == held(m),
          "reserved == held after construction")
    reset_budget()
    m2 = rb.MultiStreamReplayBuffer(2, 2 * 5000, OD, AD, obs_uint8=True,
                                    growth=growth(limit_gb), proprio_dim=PD)
    check(m2.streams[0]._block == 5000,
          f"2 streams: block unchanged at {m2.streams[0]._block}")
    return m


def test_ceiling(tmp):
    print("C. the ceiling holds under growth and restore")
    reset_budget()
    m = rb.MultiStreamReplayBuffer(8, 8 * 5000, OD, AD, obs_uint8=True,
                                   growth=growth(0.20), proprio_dim=PD)
    lim = rb.MEMORY_BUDGET.limit
    for _ in range(4):
        for s in m.streams:
            fill(s, s.capacity)
            s._grow_pending = True
        m.maybe_grow()
    check(held(m) <= lim and rb.MEMORY_BUDGET.reserved == held(m),
          f"after growth attempts: held {held(m) / 1e9:.3f} GB <= "
          f"{lim / 1e9:.2f} GB, reserved == held")
    # a larger 2-stream buffer, saved, restored into a budgeted 8-stream one
    reset_budget()
    big = rb.MultiStreamReplayBuffer(2, 2 * 3000, OD, AD, obs_uint8=True,
                                     growth=growth(1.0, block=3000),
                                     proprio_dim=PD)
    for i, s in enumerate(big.streams):
        fill(s, 3000, base=10000 * i)
    p = os.path.join(tmp, "big")
    big.save(p)
    del big
    reset_budget()
    m = rb.MultiStreamReplayBuffer(8, 8 * 5000, OD, AD, obs_uint8=True,
                                   growth=growth(0.20), proprio_dim=PD)
    n = m.load(p)
    s0 = m.streams[0]
    newest = float(s0.proprio[s0.size - 1][0])
    check(held(m) <= rb.MEMORY_BUDGET.limit
          and rb.MEMORY_BUDGET.reserved == held(m),
          f"restore: {n} rows loaded, held {held(m) / 1e9:.3f} GB <= limit, "
          f"reserved == held")
    check(newest == 2999.0,
          f"restore keeps the newest rows (last row id {newest:.0f})")


def test_main_row():
    print("D. the real main row")
    row = rb.bytes_per_row(49152, 12, 5170, True)
    tot = 8 * 5000 * row
    check(abs(tot / 1e9 - 2.79) < 0.02,
          f"bytes_per_row = {row:,} B; 8 x 5000 first blocks = "
          f"{tot / 1e9:.2f} GB")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="replay_budget_smoke_")
    try:
        test_exact()
        test_first_blocks_fit()
        test_ceiling(tmp)
        test_main_row()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILS:
        print(f"[replay_budget_smoke] FAILED ({len(FAILS)})")
        sys.exit(1)
    print("[replay_budget_smoke] ALL PASS")
