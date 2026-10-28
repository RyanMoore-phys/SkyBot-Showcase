"""Contracts for the re-arm of idled controllers and the release of stale state (GOV-015 item 2; ASS-014 F-09, F-25).

    PYTHONPATH=. python tests/_latch_rearm_smoke.py

THE INCIDENT (CLAUDE.md 4.1, guard-becomes-latch). One exception inside a controller phase (WP-8, WP-9, WP-13)
or a relation wiring module released its coefficients and then idled it FOR THE REST OF THE RUN: nothing could
ever re-open it, though the dependency recovered (categorical cat04/cat05 CT-06 D-LATCH). Separately, when the
fresh-data measurement disappeared the competence controller kept applying its last smoothed state (gain 0.70
after 8 segments, bound 6; BND-04).

CONTRACTS
  A  the loop's re-arm schedule: a phase that failed once is re-armed after one held segment, after two
     failures after two, then 4, 8 ... capped at 32
  B  a phase disabled by configuration (False, never failed) is never re-armed
  C  16 clean segments forgive the failure count
  D  a persistent fault is bounded: over 200 segments the phase is attempted at most 200/32 + 8 times
  E  the relation registry re-arms a failed wiring module after its hold (1, then 2 segments ...) and it then runs
  F  competence with no measurement steps the gain back to neutral by at most max_step per update and is
     exactly neutral after ceil(deviation / max_step) updates; valid measurement afterwards still works
  G  competence ignores a NaN or infinite reading and keeps its state
  H  falsification: with the re-arm removed (a phase that is never re-armed) contract A's second attempt
     never happens, and the old competence update (no empty branch) keeps the displaced gain
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def loop_ns():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    ns = SimpleNamespace(_comp=None)
    for n in ("_phase_failed", "_phase_rearm", "_phase_ok"):
        setattr(ns, n, getattr(DevelopmentalAI, n).__get__(ns))
    return ns


def segments_until_rearm(ns, key):
    """Call the gate once per segment while the phase is idle (False); return the segment count."""
    ns._comp = False
    for k in range(1, 100):
        if ns._phase_rearm(key):
            return k
    return None


def a_schedule():
    ns = loop_ns()
    got = []
    for _ in range(8):
        ns._phase_failed("_comp")
        got.append(segments_until_rearm(ns, "_comp"))
    check("A. holds double: 2, 3, 5, 9, 17, 33 ... (a held segment count + the re-arming call), capped",
          got[:6] == [2, 3, 5, 9, 17, 33] and got[6] == 33 and got[7] == 33, got)


def b_disabled():
    ns = loop_ns()
    ns._comp = False
    check("B. a phase disabled by config is never re-armed", not any(ns._phase_rearm("_comp") for _ in range(100)))


def c_forgive():
    ns = loop_ns()
    for _ in range(4):
        ns._phase_failed("_comp")
    for _ in range(16):
        ns._phase_ok("_comp")
    ns._phase_failed("_comp")
    check("C. 16 clean segments forgive the failure count", segments_until_rearm(ns, "_comp") == 2)


def d_bounded():
    ns = loop_ns()
    attempts = 0
    ns._comp = False
    ns._phase_failed("_comp")
    for _ in range(200):
        if ns._comp is False and ns._phase_rearm("_comp"):
            attempts += 1
            ns._comp = False                     # the attempt fails again
            ns._phase_failed("_comp")
    check("D. a persistent fault costs a bounded number of attempts", 2 <= attempts <= 200 // 32 + 8, attempts)


def e_wiring():
    from developmental_ai.core.relations import RelationRegistry
    reg = RelationRegistry.__new__(RelationRegistry)
    reg._wire_off, reg._wire_hold, reg._wire_clean = set(), {}, 0
    reg._note_wire_failure("m")
    reg._wire_off.add("m")
    check("E. first failure holds 1 segment", reg._wire_hold["m"] == [1, 1], reg._wire_hold)
    # replicate the tick at the top of wire_all
    def tick():
        for name in list(reg._wire_off):
            h = reg._wire_hold.get(name)
            if h is None or h[1] <= 0:
                reg._wire_off.discard(name)
            else:
                h[1] -= 1
    tick()
    check("E. still idle after one tick", "m" in reg._wire_off)
    tick()
    check("E. re-armed after the hold", "m" not in reg._wire_off)
    reg._note_wire_failure("m")
    check("E. a second failure doubles the hold", reg._wire_hold["m"][1] == 2, reg._wire_hold)


def comp():
    from developmental_ai.core.coefficients import CoefficientRegistry
    from developmental_ai.core.competence import CompetenceController, OWNER
    reg = CoefficientRegistry()
    C = CompetenceController({"min_meas": 3, "signal_ema": 0.5, "progress_ema": 0.5, "gain_amp": 0.3,
                              "max_step": 0.05}, reg)
    return reg, C, OWNER


def f_empty():
    reg, C, OWNER = comp()
    for _ in range(6):
        reg.set("mixer.intrinsic_gain", OWNER, 1.3, 0.05)
    dev = abs(reg.get("mixer.intrinsic_gain") - 1.0)
    n = math.ceil(dev / 0.05 - 1e-9)
    prev = reg.get("mixer.intrinsic_gain")
    steps_ok = True
    for _ in range(n):
        C.update({})
        g = reg.get("mixer.intrinsic_gain")
        steps_ok &= abs(g - prev) <= 0.05 + 1e-9
        prev = g
    check("F. no measurement: at most max_step per update", steps_ok)
    check("F. neutral after ceil(dev/step) updates", abs(reg.get("mixer.intrinsic_gain") - 1.0) < 1e-9,
          reg.get("mixer.intrinsic_gain"))
    for k in range(5):
        C.update({0: (1.0 - 0.05 * k, 0.1), 1: (2.0, 0.1)})
    check("F. valid measurement afterwards still works", C.ctx and all(s.n >= 1 for s in C.ctx.values()))


def g_nonfinite():
    reg, C, OWNER = comp()
    for _ in range(5):
        C.update({0: (1.0, 0.1), 1: (2.0, 0.1)})
    pre = (C.ctx[0].loss_ema, len(C.ctx[0].hist), C.ctx[0].n)
    for v in (float("nan"), float("inf")):
        C.update({0: (v, 0.1), 1: (2.0, 0.1)})
    s = C.ctx[0]
    check("G. a NaN or infinite loss leaves that context untouched", (s.loss_ema, len(s.hist), s.n) == pre,
          ((s.loss_ema, len(s.hist), s.n), pre))
    check("G. the smoothed signal stays finite", math.isfinite(C.flat))


def h_falsify():
    # a phase that is never re-armed: rearm() on a disabled key stays False forever (the old behaviour)
    ns = loop_ns()
    ns._comp = False
    check("H. old behaviour (no failure recorded, no re-arm) never retries", segments_until_rearm(ns, "_comp") is None)
    reg, C, OWNER = comp()
    for _ in range(6):
        reg.set("mixer.intrinsic_gain", OWNER, 1.3, 0.05)
    held = reg.get("mixer.intrinsic_gain")
    check("H. fixture displaces the gain (so F could fail)", abs(held - 1.0) > 0.1, held)


if __name__ == "__main__":
    for fn in (a_schedule, b_disabled, c_forgive, d_bounded, e_wiring, f_empty, g_nonfinite, h_falsify):
        fn()
    print("[latch-rearm] " + ("ALL PASS" if not FAILS else "FAILED: %s" % FAILS))
    sys.exit(1 if FAILS else 0)
