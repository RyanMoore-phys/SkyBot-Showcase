#!/usr/bin/env python3
"""Seeded, resumable driver for the adaptability protocol (ASS-007 3, 12.2, 12.5, 14.6).

Plans the runs (developmental_ai.evaluator.protocol), keeps ONE state file, executes
each phase through tools/eval_run.sh and applies the crash rule (a crashed seed is
re-run once with the same seed; a second crash is final and means no claim).
Re-running the same command RESUMES. It refuses to resume if a frozen file changed.

    # a plan only (safe anywhere; starts nothing):
    python tools/eval_runner.py --root ~/eval_exp --plan-only \
        --arm baseline=- --arm changed=overlays/changed.yaml --seeds 1,2,3,4,5 \
        --source-games doom --target-game doom

    # the A/A set (ten seeds, two identical arms):
    python tools/eval_runner.py --root ~/eval_exp --aa --arm baseline_a=- --arm baseline_b=- \
        --source-games doom --target-game doom

    # execute (on the training host; one run at a time, 16 GB host):
    python tools/eval_runner.py ... --execute

Default is --plan-only. Nothing here changes a learning module; the only thing that differs
between arms is the overlay file named for the arm.

Release for a stuck comparison (CLAUDE.md 4.1): delete <root>/eval_state.json on purpose to
start a new comparison; a failed_final run is never retried silently.
NOT RUN at build time (GOV-011 entry WP-14).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> int:
    from developmental_ai.evaluator import protocol, runner
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="experiment root (state, runs, results)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=OVERLAY|-",
                    help="arm name = overlay yaml merged into configs/universal.yaml ('-' none)")
    ap.add_argument("--fresh-control", action="append", default=[], metavar="ARM",
                    help="arm with no source phase (reported, not gated)")
    ap.add_argument("--seeds", default=None, help="comma list (default 1..5, or 11..20 with --aa)")
    ap.add_argument("--aa", action="store_true", help="the A/A set: seeds 11..20 (ten, ASS-007 14.3)")
    ap.add_argument("--source-games", default="doom")
    ap.add_argument("--target-game", default="doom",
                    help="development proxy defaults to held-out Doom; pass minecraft only for the later program evaluation")
    ap.add_argument("--total-minutes", type=float, default=protocol.TOTAL_MINUTES)
    ap.add_argument("--source-frac", type=float, default=protocol.SOURCE_FRAC)
    ap.add_argument("--phase-cmd", default=f"bash {os.path.join(ROOT, 'tools/eval_run.sh')}")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="default: print the plan and stop")
    mode.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)

    arms, overlays = [], {}
    for spec in a.arm:
        name, _, ov = spec.partition("=")
        if not name or not ov:
            ap.error(f"--arm needs NAME=OVERLAY or NAME=-, got {spec!r}")
        arms.append(name)
        overlays[name] = ov
    if not arms:
        ap.error("at least one --arm")
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds
             else list(protocol.AA_SEEDS if a.aa else protocol.MAIN_SEEDS))
    plan = protocol.plan_runs(arms, seeds, a.total_minutes, a.source_frac, a.fresh_control)
    summ = protocol.plan_summary(plan)
    print("plan:", json.dumps(summ))
    for r in plan:
        print(f"  {r.run_id}: " + " -> ".join(f"{p.name} {p.minutes:g} min" for p in r.phases))
    if not a.execute:
        print("(plan only; pass --execute on the training host to run)")
        return 0

    code_root = os.path.abspath(os.environ.get(
        "EVAL_CODE", os.path.join(a.root, "code")))
    code_files = sorted(
        str(p) for p in Path(code_root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and p.suffix != ".pyc")
    frozen_paths = [o for o in overlays.values() if o != "-"] + code_files
    frozen = runner.file_hash(frozen_paths)
    state = runner.EvalState.load_or_create(os.path.join(a.root, "eval_state.json"), plan, frozen)
    while True:
        task = state.next_task()
        if task is None:
            break
        rid, phase, minutes = task
        run = state.run(rid)
        games = a.source_games if phase == "source" else a.target_game
        cmd = (f"EVAL_EXP={a.root} {a.phase_cmd} {phase} {rid} {run['seed']} "
               f"{overlays[run['arm']]} {games} {minutes}")
        print(f"[{rid}] attempt {run['attempts']} phase {phase}: {cmd}", flush=True)
        state.mark_started(rid, phase)
        rc = subprocess.call(cmd, shell=True)
        if rc == 0:
            state.mark_phase_done(rid, phase, {"rc": rc})
        else:
            status = state.mark_crash(rid, phase, f"exit code {rc}")
            print(f"[{rid}] CRASH (rc={rc}) -> {status}", flush=True)
    print("crashes by arm:", json.dumps(state.crash_counts_by_arm()))
    for arm in arms:
        miss = state.missing_seeds(arm)
        if miss:
            print(f"arm {arm}: seeds not completed {miss} -> NO CLAIM from this set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
