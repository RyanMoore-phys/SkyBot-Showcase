"""Contracts for the life-boundary guard clock and its consumers (GOV-014 item B1; ASS-014 F-13).

    PYTHONPATH=. python tests/_guard_polarity_smoke.py

THE INCIDENT. `curiosity/action_surprise.py` documents `StepSurprise.guard` as "True = boundary:
ignore row", and `GuardClock.update` returns True for rows WITHIN `k_guard` steps of a life boundary.
`core/relations_attention.py` reads it that way. Two other consumers read True as ELIGIBLE:
`InfoGainTerm.pay` (WP-5) and `SurpriseAgencyTrigger.step` (WP-1). Measured by the categorical
harness (cat04 CT-05): information gain was paid only just after a boundary and the skill trigger
fired on no mid-life spike. Only the consumers are wrong; the producer, its docstring and the third
reader agree.

CONTRACTS
  A  the producer: True for rows within k steps of a boundary, False after, a clock per stream
  B  InfoGainTerm pays mid-life rows and pays NOTHING for a boundary row
  C  SurpriseAgencyTrigger fires on a mid-life spike and never on a boundary row, and counts it as guarded
  D  end to end (clock -> term): no payment for k+1 steps after a boundary, payments afterwards
  E  default parity: weight 0 pays nothing whatever the guard says (the packages are OFF by default)
  F  falsification: the OLD polarity (consumer reading True as eligible) fails contracts B to D
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.curiosity.action_surprise import GuardClock  # noqa: E402
from developmental_ai.curiosity.intrinsic_balance import InfoGainTerm  # noqa: E402
from developmental_ai.skill_bank.effect_discovery import SurpriseAgencyTrigger  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


class Fake:
    """A StepSurprise stand-in: per-row surprise `s`, the guard list, and per-row agency."""

    def __init__(self, s, guard, agency):
        self.s = torch.tensor(s, dtype=torch.float32)
        self.guard = list(guard)
        self._ag = dict(agency)

    def agency_for(self, rows):
        return {r: self._ag[r] for r in rows}


def old_pay(term, surprise, rest):
    """The pre-fix reading, kept as a witness: rows where guard is True are treated as eligible."""
    n = int(surprise.s.shape[0])
    out = [0.0] * n
    rows = [i for i in range(n) if surprise.guard[i]]
    ag = surprise.agency_for(rows)
    for i in rows:
        out[i] = term.weight * min(term.clip, max(0.0, ag[i] - term.floor))
    return out


def main() -> int:
    print("A the producer")
    gc = GuardClock(k_guard=3)
    first = gc.update(2, [False, False])
    check("a stream that has never ended is not guarded", first == [False, False], first)
    seq = [gc.update(2, [True, False]) for _ in range(1)]
    after = [gc.update(2, [False, False])[0] for _ in range(5)]
    check("a boundary row is True", seq[0][0] is True, seq)
    check("it stays True for k steps, then re-opens by itself", [seq[0][0]] + after == [True, True, True, True, False, False], [seq[0][0]] + after)
    check("another stream's clock is independent", seq[0][1] is False)

    print("B InfoGainTerm")
    t = InfoGainTerm(weight=1.0, floor=0.5, clip=5.0)
    s = Fake([1.0, 1.0], guard=[True, False], agency={0: 3.0, 1: 3.0})
    out = t.pay(s, [0.0, 0.0])
    check("a boundary row (guard True) is not paid", out[0] == 0.0, out)
    check("a mid-life row (guard False) is paid", out[1] > 0.0, out)

    print("C SurpriseAgencyTrigger")
    trig = SurpriseAgencyTrigger(z_thr=3.0, s_floor=1.0, agency_frac=0.25, horizon=1000, warmup=5)
    quiet = Fake([1.0, 1.0], guard=[False, False], agency={0: 0.0, 1: 0.0})
    for _ in range(30):
        trig.step(quiet)
    spike = Fake([10.0, 10.0], guard=[True, False], agency={0: 5.0, 1: 5.0})
    fires = trig.step(spike)
    check("a mid-life spike fires", any(f["stream"] == 1 for f in fires), fires)
    check("a boundary row never fires", all(f["stream"] != 0 for f in fires), fires)
    check("the boundary row is counted as guarded", trig.stats["guarded"] >= 1, dict(trig.stats))

    print("D end to end: clock -> term")
    clock, term = GuardClock(k_guard=3), InfoGainTerm(weight=1.0, floor=0.5, clip=5.0)
    pays = []
    for step in range(10):
        g = clock.update(1, [step == 0])                 # a boundary at step 0
        pays.append(term.pay(Fake([1.0], guard=g, agency={0: 3.0}), [0.0])[0])
    check("no payment for the k+1 steps at and after a boundary", all(p == 0.0 for p in pays[:4]), pays)
    check("payment from then on", all(p > 0.0 for p in pays[4:]), pays)

    print("E default parity")
    off = InfoGainTerm(weight=0.0)
    check("weight 0 pays nothing for any guard", off.pay(Fake([1.0, 1.0], [True, False], {0: 9.0, 1: 9.0}), [0.0, 0.0]) == [0.0, 0.0])
    check("and counts no rows", off.n_rows == 0)

    print("F falsification: the old reading is caught")
    clock, term = GuardClock(k_guard=3), InfoGainTerm(weight=1.0, floor=0.5, clip=5.0)
    bad = []
    for step in range(10):
        g = clock.update(1, [step == 0])
        bad.append(old_pay(term, Fake([1.0], guard=g, agency={0: 3.0}), [0.0])[0])
    check("the old polarity pays ONLY right after a boundary (the measured defect)",
          all(p > 0 for p in bad[:4]) and all(p == 0.0 for p in bad[4:]), bad)

    print(f"[guard-polarity] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
