"""Evaluation machinery contracts (GOV-009 WP-14 test items a, b, c; ASS-007 5, 12, 14).

Born 2026-10-21. Pure numpy/Python: no engine, no torch. A comparison tool that
cannot recover a known effect, or that "finds" effects in noise, would corrupt
every later decision, so each contract is paired with its falsification.

Contracts:
    A. SPLIT. 60 minutes -> 40 source + 20 target by default; fresh-control arms
       have the target phase only; plan order is seed-major (arms interleaved).
    B. KNOWN EFFECT RECOVERED. Synthetic series with a true paired effect of +0.20
       on five seeds: the five-seed rule says positive. With NO effect (A/A-like
       noise only), repeated 200 times, the share of "positive" verdicts is at most
       ~3x the nominal and far below the effect case.
    C. BLOCK BOOTSTRAP IS WIDER THAN IID on an autocorrelated series (the reason
       it is used), and block_length is > 1 there, == 1 on white noise.
    D. CENSORING. No checkpoint reached -> area 0.0 and every checkpoint counted
       censored; all reached at t=0 -> area 1.0; unreached beats nothing.
    E. A/A POOL. With ten pairs the floor is positive and the exact subset
       enumeration covers C(10,5)=252; WITHOUT a pool the verdict is "no claim".
    F. CRASH RULE. A crash re-runs the whole seed once with the same seed; a second
       crash is failed_final; resume after a "driver died" phase counts as a crash;
       a changed frozen file refuses to resume.
    G. MINECRAFT LADDER IS LOUD. get_detector("minecraft") raises DetectorNotDefined
       naming "ladder not defined".
    H. RECORD. The generated record contains the verdict, floor, power statement,
       the evidence class and the 'development evidence' disclaimer.

Run:  PYTHONPATH=. python tests/_eval_machinery_smoke.py
NOT RUN at build time (GOV-011 entry WP-14).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

NAME = "eval-machinery"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _fail(msg):
    print("[%s] FAIL: %s" % (NAME, msg))
    sys.exit(1)


def check(c, msg):
    if not c:
        _fail(msg)


def main():
    from developmental_ai.evaluator import analysis, protocol, record, runner, stats
    from developmental_ai.evaluator.registry import DetectorNotDefined, get_detector

    # A
    runs = protocol.plan_runs(["base", "chg", "fresh"], [1, 2], fresh_control_arms=["fresh"])
    check([r.run_id for r in runs] == ["base_s1", "chg_s1", "fresh_s1", "base_s2", "chg_s2", "fresh_s2"],
          "A: order %r" % [r.run_id for r in runs])
    check(protocol.split_minutes() == (40.0, 20.0), "A: split")
    check([p.name for p in runs[2].phases] == ["target"] and
          [p.name for p in runs[0].phases] == ["source", "target"], "A: phases")
    print("[%s] A. 40/20 split, seed-major order, fresh control has target only" % NAME)

    # B
    rng = np.random.default_rng(0)

    def make_pairs(effect, n_seeds=5, noise=0.05, n=12):
        out = []
        for s in range(n_seeds):
            base = rng.normal(0.4, noise, n)
            chg = base + effect + rng.normal(0, noise, n)
            out.append(stats.paired_seed(s + 1, base, chg, rng, n_boot=400))
        return out

    aa = [stats.paired_seed(100 + i, rng.normal(0.4, .05, 12),
                            rng.normal(0.4, .05, 12), rng, n_boot=400) for i in range(10)]
    floor = stats.noise_floor(aa)
    eff = stats.five_seed_rule(make_pairs(0.20), floor)
    check(eff["verdict"] == "positive", "B: known effect not recovered: %r" % eff)
    fp = sum(stats.five_seed_rule(make_pairs(0.0), floor)["verdict"] == "positive"
             for _ in range(200)) / 200.0
    check(fp <= 0.03, "B: false-positive rate under no effect is %.3f" % fp)
    print("[%s] B. +0.20 recovered on five seeds; no-effect all-five-positive rate %.3f" % (NAME, fp))

    # C
    w = rng.normal(size=60)
    ar = np.zeros(60)
    for i in range(1, 60):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    check(stats.block_length(ar) > stats.block_length(w) >= 1, "C: block length")
    lo_b, hi_b, _ = stats.interval(ar, np.random.default_rng(1))
    iid = stats.moving_block_bootstrap_means(ar, 1, 2000, np.random.default_rng(1))
    check((hi_b - lo_b) > (np.quantile(iid, .975) - np.quantile(iid, .025)),
          "C: block bootstrap not wider than iid on an autocorrelated series")
    print("[%s] C. block length adapts; block interval wider than iid on AR(1)" % NAME)

    # D
    ids = ["a", "b", "c"]
    check(stats.curve_area({}, ids, 1200.0) == 0.0, "D: nothing reached")
    check(stats.curve_area({"a": 0.0, "b": 0.0, "c": 0.0}, ids, 1200.0) == 1.0, "D: all at 0")
    check(stats.curve_area({"a": 600.0}, ids, 1200.0) < stats.curve_area({"a": 100.0}, ids, 1200.0),
          "D: earlier must score higher")
    print("[%s] D. censoring: unreached = 0, immediate = 1, earlier is better" % NAME)

    # E
    fpr = stats.aa_false_positive(aa, floor, 5)
    check(fpr["subsets"] == 252 and fpr["exact"], "E: enumeration %r" % fpr)
    check(floor > 0.0, "E: floor")
    one = {1: {"checkpoints": ["a"], "attained": {"a": 10.0}, "at_entry": [], "budget_s": 1200.0}}
    two = {1: {"checkpoints": ["a"], "attained": {"a": 5.0}, "at_entry": [], "budget_s": 1200.0}}
    out = analysis.analyse(one, two, [], test_type="T1", evidence_class="development evidence",
                           dev_baseline_id="b0", arms=["x", "y"])
    check(out["rule"]["verdict"].startswith("no claim"), "E: no pool must be no claim")
    print("[%s] E. 252 exact subsets; no A/A pool -> no claim" % NAME)

    # F
    with tempfile.TemporaryDirectory() as d:
        plan = protocol.plan_runs(["base"], [1, 2])
        ov = os.path.join(d, "ov.yaml")
        open(ov, "w").write("a: 1\n")
        fz = runner.file_hash([ov])
        st = runner.EvalState.load_or_create(os.path.join(d, "s.json"), plan, fz)
        rid, ph, _ = st.next_task()
        st.mark_started(rid, ph)
        st.mark_phase_done(rid, ph)
        st.mark_started(rid, "target")
        check(st.mark_crash(rid, "target", "boom") == "pending" and st.run(rid)["attempts"] == 2
              and st.run(rid)["phase_done"] == [], "F: first crash must re-run the whole seed")
        check(st.mark_crash(rid, "source", "boom2") == "failed_final", "F: second crash final")
        check(st.next_task()[0] == "base_s2", "F: next run")
        st.mark_started("base_s2", "source")
        st2 = runner.EvalState.load_or_create(os.path.join(d, "s.json"), plan, fz)
        check(st2.run("base_s2")["attempts"] == 2, "F: died mid-phase must count as a crash")
        open(ov, "w").write("a: 2\n")
        try:
            runner.EvalState.load_or_create(os.path.join(d, "s.json"), plan, runner.file_hash([ov]))
            _fail("F: changed frozen file resumed")
        except runner.FrozenConfigChanged:
            pass
    print("[%s] F. crash rule, mid-phase death, frozen-config refusal" % NAME)

    # G
    try:
        get_detector("minecraft")
        _fail("G: minecraft detector did not raise")
    except DetectorNotDefined as e:
        check("ladder not defined" in str(e), "G: message %r" % str(e))
    print("[%s] G. Minecraft detector fails loudly: ladder not defined" % NAME)

    # H
    out = analysis.analyse(one, two, aa, test_type="T1", evidence_class="development evidence",
                           dev_baseline_id="b0", arms=["x", "y"])
    txt = record.render(out, "smoke", "2026-10-21")
    for needle in ("Verdict", "noise floor", "Power.", "development evidence", "Deviations"):
        check(needle in txt, "H: record lacks %r" % needle)
    print("[%s] H. record carries verdict, floor, power, evidence class, deviations" % NAME)
    print("[%s] ALL PASS" % NAME)


if __name__ == "__main__":
    main()
