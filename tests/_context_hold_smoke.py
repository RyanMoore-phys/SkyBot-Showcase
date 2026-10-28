"""WP-8 does not score a curriculum context change as lost learning progress (GOV-015 section 9.6; found by BND-04, ASS-012).

    PYTHONPATH=. python tests/_context_hold_smoke.py

THE FINDING. The re-run of the categorical boundaries (stage gov015_rerun, 2026-10-26) showed the competence tracker booking the
loss jump that follows a WP-3 context switch as lost progress (lp -0.213 against 0.029 for the same run without the jump): the
tracker had no input saying the context had changed. `CompetenceController.note_context_change(i)` now makes stream i's next
`context_hold` updates re-baseline the loss instead of scoring it, and the loop calls it where it applies a curriculum context.

CONTRACTS (the controller on its own, no loop):
    A. A jump right after a change is NOT scored: the progress estimate equals the no-jump control's.
    B. FALSIFIER: the same jump with no change IS scored (progress falls clearly below the control): the hold hides nothing else.
    C. FALSIFIER: with `context_hold: 0` (the old behaviour) the post-change jump is scored: the test can fail.
    D. THE HOLD EXPIRES BY ITSELF (CLAUDE.md 4.1): a genuine regression two updates after a change is registered again.
    E. Only the named stream is held: another stream's regression in the same update is scored.
    F. A stream not yet seen is a no-op (no state is created, nothing raises).
    G. WIRING: the loop calls note_context_change where it applies a curriculum context, once, guarded for a missing controller.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.core.coefficients import CoefficientRegistry   # noqa: E402
from developmental_ai.core.competence import CompetenceController   # noqa: E402

NAME = "context-hold"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def ctl(hold=None):
    cfg = {"min_meas": 3, "signal_ema": 0.5, "progress_ema": 0.5}
    if hold is not None:
        cfg["context_hold"] = hold
    return CompetenceController(cfg, CoefficientRegistry())


def run(jump_at=None, change_at=None, hold=None, steps=10, jump=2.0, stream=0):
    """Two streams, loss falling 3 % per update; stream `stream` jumps by `jump` at update `jump_at`; a context change is
    noted for it after update `change_at`. Returns the controller."""
    C = ctl(hold)
    loss = [1.0, 1.0]
    for k in range(steps):
        loss = [x * 0.97 for x in loss]
        if jump_at is not None and k == jump_at:
            loss[stream] *= jump
        C.update({0: (loss[0], 0.1), 1: (loss[1], 0.1)})
        if change_at is not None and k == change_at:
            C.note_context_change(stream)
    return C


def main():
    control = run()
    switched = run(jump_at=8, change_at=7)
    check(abs(switched.ctx[0].lp - control.ctx[0].lp) < 0.1,
          "A. a jump right after a context change is not scored (lp %.3f vs %.3f without it)"
          % (switched.ctx[0].lp, control.ctx[0].lp))

    regress = run(jump_at=8)
    check(regress.ctx[0].lp < control.ctx[0].lp - 0.1,
          "B. falsifier: the same jump with no context change IS scored (lp %.3f vs %.3f)" % (regress.ctx[0].lp, control.ctx[0].lp))

    old = run(jump_at=8, change_at=7, hold=0)
    check(old.ctx[0].lp < control.ctx[0].lp - 0.1,
          "C. falsifier: with context_hold 0 the post-change jump is scored again (lp %.3f): the test can fail" % old.ctx[0].lp)

    late = run(jump_at=8, change_at=3)                    # the hold (updates 4 and 5) has expired by update 8
    check(late.ctx[0].lp < control.ctx[0].lp - 0.1,
          "D. the hold expires by itself: a regression five updates after the change is registered (lp %.3f vs %.3f)"
          % (late.ctx[0].lp, control.ctx[0].lp))
    check(late.ctx[0].hold == 0, "D. no hold remains after it has been used (%d)" % late.ctx[0].hold)

    other = run(jump_at=8, change_at=7, stream=1)          # stream 1 jumps, but the change was noted for stream 1 as well
    two = ctl()
    ls = [1.0, 1.0]
    for k in range(10):
        ls = [x * 0.97 for x in ls]
        if k == 8:
            ls[1] *= 2.0                                    # stream 1 jumps with NO change noted for it
        two.update({0: (ls[0], 0.1), 1: (ls[1], 0.1)})
        if k == 7:
            two.note_context_change(0)                      # only stream 0 was held
    check(two.ctx[1].lp < control.ctx[1].lp - 0.1 and abs(two.ctx[0].lp - control.ctx[0].lp) < 0.1,
          "E. only the named stream is held (stream 1 lp %.3f scored, stream 0 lp %.3f untouched)" % (two.ctx[1].lp, two.ctx[0].lp))
    check(other.ctx[1].hold == 0, "E. a held stream's counter is spent")

    fresh = ctl()
    fresh.note_context_change(7)
    check(7 not in fresh.ctx, "F. noting a change for an unseen stream creates no state")
    check(all(math.isfinite(s.lp) for s in switched.ctx.values()) and all(math.isfinite(s.loss_ema) for s in switched.ctx.values()),
          "F. state stays finite through a held update")

    src = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    i = src.index("for e, ctx in Cu.step(meas).items():")
    block = src[i:i + 700]
    check(src.count("note_context_change(") == 1 and "note_context_change(e)" in block and 'getattr(self, "_comp", None)' in block,
          "G. the loop notes the change once, where it applies the curriculum context, guarded for a missing controller")

    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
