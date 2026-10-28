#!/usr/bin/env python3
"""Score runs and compare arms under the five-seed evidence standard (ASS-007 5, 7, 14).

    # 1. score one finished run's target phase from its log (needs the game's detector;
    #    a game with none, e.g. Minecraft today, FAILS LOUDLY):
    python tools/eval_analyze.py score --run-dir ~/eval_exp/runs/baseline_s1 \
        --game minecraft --budget-min 20 --out result.json

    # 2. compare two arms (dirs of result.json named <arm>_s<seed>/result.json or flat
    #    files <run_id>.json), with the A/A pool, and write the result record:
    python tools/eval_analyze.py compare --base DIR_BASE --changed DIR_CHANGED \
        --aa-base DIR_AA_A --aa-changed DIR_AA_B --test-type T1 \
        --evidence-class "development evidence" --dev-baseline baseline-0 \
        --title "WP-5 episodic weight" --out record.md

Pure analysis on files already on disk; no GPU, no run is started.
NOT RUN at build time (GOV-011 entry WP-14).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _hms_to_s(h: str) -> float:
    a, b, c = (int(x) for x in h.split(":"))
    return a * 3600 + b * 60 + c


def cmd_score(a) -> int:
    from developmental_ai.evaluator import scoring
    phase = json.load(open(os.path.join(a.run_dir, "phase_target.json")))
    if phase.get("crash"):
        print("target phase crashed: this run is not scorable (crash rule, ASS-007 14.6)")
        return 3
    logs = sorted(glob.glob(os.path.join(a.run_dir, "results", "target", "minecraft_skybot_run.log*")))
    if not logs:
        print("no target run log found")
        return 2
    events = []
    for lg in logs:
        events += scoring.parse_evaluator_log(lg, a.game_filter)
    res = scoring.score_events(a.game, events, _hms_to_s(phase["start_hms"]), a.budget_min * 60.0)
    m = re.search(r"_s(\d+)$", os.path.basename(a.run_dir.rstrip("/")))
    res["seed"] = int(m.group(1)) if m else phase.get("seed")
    res["run_dir"] = a.run_dir
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"scored {a.run_dir}: attained={sorted(res['attained'])} at_entry={res['at_entry']}")
    return 0


def _load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*", "result.json")) + glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(p))
        out[int(r["seed"])] = r
    return out


def cmd_compare(a) -> int:
    import numpy as np
    from developmental_ai.evaluator import analysis, record, stats
    base, changed = _load(a.base), _load(a.changed)
    aa_pairs, uninf = [], []
    if a.aa_base and a.aa_changed:
        ab, ac = _load(a.aa_base), _load(a.aa_changed)
        ckpts = next(iter(list(ab.values()) + list(ac.values())))["checkpoints"]
        uninf = stats.uninformative_checkpoints(ckpts, list(ab.values()) + list(ac.values()))
        aa_pairs = analysis.paired_from_results(
            ab, ac, uninf, np.random.default_rng(a.rng_seed))["pairs"]
    res = analysis.analyse(base, changed, aa_pairs, test_type=a.test_type,
                           evidence_class=a.evidence_class, dev_baseline_id=a.dev_baseline,
                           arms=[os.path.basename(a.base.rstrip("/")),
                                 os.path.basename(a.changed.rstrip("/"))],
                           uninformative=uninf, rng_seed=a.rng_seed,
                           deviations=a.deviation)
    text = record.render(res, a.title, a.date)
    with open(a.out, "w") as f:
        f.write(text)
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--game", required=True, help="detector name (evaluator/detectors/<game>.py)")
    s.add_argument("--game-filter", default=None, help="only log lines mentioning this text")
    s.add_argument("--budget-min", type=float, default=20.0)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--changed", required=True)
    c.add_argument("--aa-base")
    c.add_argument("--aa-changed")
    c.add_argument("--test-type", default="T1", help="T1 | T2 | unit | A/A")
    c.add_argument("--evidence-class", default="development evidence")
    c.add_argument("--dev-baseline", required=True)
    c.add_argument("--title", required=True)
    c.add_argument("--date", default=datetime.date.today().isoformat())
    c.add_argument("--rng-seed", type=int, default=0)
    c.add_argument("--deviation", action="append", default=[])
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_score(a) if a.cmd == "score" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
