#!/usr/bin/env python3
"""GOV-015 9.9 diagnostic: why BND-04 CT-05's second half (a genuine regression still registers) is red after the context-hold fix.

Replays that scenario's three arms (control, switched + jump, regression with no switch noted by the scenario) and prints, per
segment, the stream's loss, the curriculum context count, the competence hold counter and the progress estimate, so the reading
"the brain's own curriculum switches inside the regression arm and the hold masks the jump" is checked instead of assumed.
Reports only; changes nothing; exit 0 when it ran, 2 when it could not.

    PYTHONPATH=. python tools/gov015_bnd04_diag.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


OUT = sys.__stdout__          # the scenario module may redirect sys.stdout while it runs (the first run of this tool printed nothing)


def say(*a):
    print(*a, file=OUT, flush=True)


def main() -> int:
    say("gov015_bnd04_diag: start")
    try:
        import tempfile
        import types
        from categorical_lib import seed_all
        # the scenario module ends with `sys.exit(H.finish())`, which would run the whole suite on import: load it without that
        src = open(os.path.join(ROOT, "tests", "_categorical_boundaries_smoke.py")).read()
        assert src.rstrip().endswith("sys.exit(H.finish())")
        src = src.rstrip()[: -len("sys.exit(H.finish())")] + "pass\n"
        B = types.ModuleType("bnd_diag")
        B.__file__ = os.path.join(ROOT, "tests", "_categorical_boundaries_smoke.py")
        exec(compile(src, B.__file__, "exec"), B.__dict__)
    except Exception as e:                                   # pragma: no cover
        import traceback
        traceback.print_exc()
        say(f"CANNOT LOAD THE SCENARIO MODULE: {e!r}")
        return 2
    for label, jump, switch in (("control (no jump)", False, True), ("switched + jump", True, True),
                                ("regression: jump, scenario notes no switch", True, False)):
        seed_all(0)
        say(f"\n== {label}")
        say("  k   loss0   n_ctx  hold  lp      n   (the scenario reads lp at 'at' and at at+2)")
        with B.measuring(), tempfile.TemporaryDirectory() as d:
            ns = B.b04_brain(tmp=d)
            seen, at = 0, None
            loss = [1.0, 1.0]
            for k in range(14):
                loss = [x * 0.97 for x in loss]
                if jump and at is not None and k == at + 1:
                    loss[0] *= 2.0
                B.b04_seg(ns, list(loss))
                c0 = ns._comp.ctx[0]
                n_ctx = len(ns._parallel_envs[0].ctx)
                if k >= 3 and c0.n >= 3 and at is None:
                    if (switch and n_ctx > seen and seen > 0) or (not switch and k == 6):
                        at = k
                say(f"  {k:2d}  {loss[0]:.4f}  {n_ctx:4d}  {getattr(c0, 'hold', '-'):4}  {c0.lp:+.4f}  {c0.n:2d}"
                      + ("   <- at" if at == k else "") + ("   <- jump applied next" if at is not None and k == at else ""))
                seen = n_ctx
    # CT-03 (INCONCLUSIVE: 7 of the 8 required checks): its snapshot records the curriculum pulls but nothing asserts them.
    say("\n== CT-03: do the curriculum pulls move on one NaN stream plus one valid stream?")
    for label, bad in (("NaN", float("nan")), ("inf", float("inf"))):
        seed_all(0)
        with B.measuring(), tempfile.TemporaryDirectory() as d:
            ns = B.b04_brain(tmp=d)
            B.b04_run(ns, 10, "changing")
            Cu = ns._curr
            before = list(Cu.bandits["p"].rewards_n)
            for _ in range(2):
                B.b04_seg(ns, [bad, 1.0])
            after = list(Cu.bandits["p"].rewards_n)
            say(f"  {label}: pulls before {before} after {after}  (valid stream 1 is measured both times)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
