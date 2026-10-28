#!/usr/bin/env python3
"""Unattended, resumable test campaign for the training host (ASS-013).

ONE driver that runs the whole queue of testing on `main`, stage by stage, serially, with
no operator: the A/A noise floor, the test suites, the split-sensitivity pilot, the shared
baseline, every T1 block A/B, every T2 relationship A/B, and the categorical operational
pilots (GOV-013 / ASS-012). It owns ORDER, GATES, RESUME and the report. It owns no
learning code and no verdict: scoring and comparison reuse tools/eval_runner.py and
tools/eval_analyze.py; the arms come from tools/campaign_arms.py.

    python tools/campaign.py --root R plan        # print the queue; starts nothing (default)
    python tools/campaign.py --root R run         # execute (what the service/cron runs)
    python tools/campaign.py --root R status      # table + CAMPAIGN_STATUS.md
    python tools/campaign.py --root R resume [STAGE|all]   # clear a pause / failed stage
    python tools/campaign.py --root R gate NAME   # open a human gate (see below)
    touch R/CAMPAIGN_STOP                         # stop AFTER the current stage (never mid-run)

Design rules (each one answers a recorded failure; CLAUDE.md 4.1 guard-becomes-latch):
  * SERIAL and EXCLUSIVE. A stage starts only when no other trainer is alive on the host
    (16 GB box; ASS-012 5.1). If something else holds the host the driver WAITS; it never
    kills it and never runs beside it.
  * RESUMABLE. State is one JSON file. A reboot, a service restart or a re-run continues:
    a stage found 'running' re-attaches to its live process or, if gone, is re-entered (the
    eval runner resumes its own state; a single-phase run restarts clean).
  * A stage's process runs in its OWN session, so restarting this driver never kills a run.
  * ESCAPES. Every way a stage can stop the queue states what re-opens it:
      pause-on-fail  -> `resume STAGE` (a failed timed A/B is a human decision, not a retry loop)
      human gate     -> `gate NAME` (ASS-012 threshold freeze: thresholds must not be chosen by
                        a script after seeing results)
      blocked        -> re-evaluated every pass; a stage blocked on a missing test harness or a
                        blocked prerequisite starts by itself once the files exist
      on_fail=continue stages (tests, analysis) never stop the queue; they are reported.
  * NOTHING is hidden: blocked and waiting stages print their reason in `plan` and `status`.
  * The code under test is a FROZEN staged copy (--code); its file hashes are recorded at the
    first run and the driver refuses to continue if they change (use --accept-code-change after
    a fix; a fix starts a new snapshot, ASS-012 6).

Evidence class: everything here is development evidence (ASS-007 14.1). Results are filed
under <root>/stages/<id>/ and <root>/analysis/; verdicts are made by the analysis records,
not by this driver.
NOT RUN on the host at build time; tests/_campaign_smoke.py exercises it with fake stages.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import campaign_arms as arms_mod  # noqa: E402

TERMINAL = ("done", "done_with_failures", "failed", "skipped")
OK = ("done", "done_with_failures")
TRAINER_PAT = "(run_minecraft|scripts/run)[.]py --config runlogs/run_config.yaml"
POLL_S = 20
IDLE_S = 300


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------- stages
def build_stages(c: "Cfg") -> List[Dict[str, Any]]:
    """The whole queue, in order. A stage: id, title, kind (eval|cmd|analyze|gate), requires,
    on_fail (pause|continue), est_hours, plus kind-specific fields. Pure: no side effects."""
    py, code = c.py, c.code
    S: List[Dict[str, Any]] = []

    def add(**k):
        k.setdefault("requires", [])
        k.setdefault("on_fail", "pause")
        k.setdefault("est_hours", 0.0)
        k.setdefault("blocked", None)
        S.append(k)

    def ev(sid, title, arms, seeds, requires, hours, extra=None, blocked=None, on_fail="pause", code_root=None):
        root = os.path.join(c.stages, sid)
        code = code_root or c.code                      # a staged snapshot (WP-20) or the frozen copy
        cmd = [py, os.path.join(code, "tools/eval_runner.py"), "--root", root,
               "--source-games", "doom", "--target-game", "doom", "--seeds", ",".join(map(str, seeds))]
        for a, ov in arms:
            cmd += ["--arm", f"{a}={ov}"]
        cmd += (extra or []) + ["--execute"]
        add(id=sid, title=title, kind="eval", cmd=cmd, root=root, requires=requires,
            est_hours=hours, blocked=blocked, on_fail=on_fail, cwd=code,
            env={"EVAL_CODE": code, "EVAL_VENV": c.venv})

    # 1. the A/A noise floor, already launched by hand: attach to it, resume it if it is dead
    add(id="ab00", title="AB-00 Doom A/A noise floor (10 paired seeds)", kind="eval",
        root=c.ab00_root, cwd=os.path.join(c.ab00_root, "code"), est_hours=20.0,
        cmd=[py, "tools/eval_runner.py", "--root", c.ab00_root, "--aa", "--arm", "baseline_a=-",
             "--arm", "baseline_b=-", "--source-games", "doom", "--target-game", "doom", "--execute"],
        env={}, external_pat=f"eval_runner.py --root {c.ab00_root}")
    add(id="an_ab00", title="Score the A/A pool", kind="analyze", requires=["ab00"], on_fail="continue",
        base="ab00", changed="ab00", test_type="A/A", arm_base="baseline_a", arm_changed="baseline_b",
        title_text="AB-00 A/A")
    # 2. the suites nobody has run (build-only phase); failures are reported, not fatal
    # The suites read docs/, experiments/, docker/ and the Doom ROM, which the frozen code copy deliberately
    # does not carry (it must stay the code under test). They run from a SEPARATE test tree built from the
    # copy plus <root>/test_extras (rsynced from the repository) plus the host's ROMs, so a missing directory is
    # not reported as a failing suite (2026-10-23: 5 of 13 reds on `main` were exactly that, ASS-017).
    test_tree = os.path.join(c.root, "test_code")
    extras = os.path.join(c.root, "test_extras")

    def test_cmd(tier):
        sh = (f"set -e; rsync -a --delete --exclude __pycache__ {shlex.quote(code)}/ {shlex.quote(test_tree)}/; "
              f"[ -d {shlex.quote(extras)} ] && rsync -a {shlex.quote(extras)}/ {shlex.quote(test_tree)}/ || true; "
              f"ln -sfn /workspace/devai/roms {shlex.quote(test_tree)}/roms; cd {shlex.quote(test_tree)}; "
              f"PYTHONPATH={shlex.quote(test_tree)} {shlex.quote(py)} tests/run_all.py {tier}")
        return ["bash", "-c", sh]
    for tier in ("unit", "contract"):
        add(id=f"tests_{tier}", title=f"tests/run_all.py {tier} on a complete test tree (code + docs + experiments + docker)",
            kind="cmd", cmd=test_cmd(tier), cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.2)
    # 2a. GOV-015 (2026-10-23): the corrective changes are staged in <root>/gov015_code, a COMPLETE copy of the
    # repository sent from the development machine. They are tested there, never in the frozen code copy, so the
    # copy stays the code under test and the running experiments are untouched. Two stages: the new code, and the
    # OLD code (frozen copy) run with the NEW tests, which must fail (a test that cannot fail proves nothing).
    gov = os.path.join(c.root, "gov015_code")
    gnew, gold = os.path.join(c.root, "gov015_test_new"), os.path.join(c.root, "gov015_test_old")
    new_suites = "tests/_nonfinite_guards_smoke.py tests/_latch_rearm_smoke.py"

    def gov_cmd(tree, body):
        sh = (f"[ -d {shlex.quote(gov)} ] || {{ echo 'gov015_code is not staged'; exit 3; }}; "
              f"mkdir -p {shlex.quote(tree)}; {body} "
              f"ln -sfn /workspace/devai/roms {shlex.quote(tree)}/roms; cd {shlex.quote(tree)}; rc=0; "
              f"export PYTHONPATH={shlex.quote(tree)}; ")
        return sh
    new_body = (f"rsync -a --delete --exclude __pycache__ --exclude venv --exclude runlogs --exclude .git "
                f"{shlex.quote(gov)}/ {shlex.quote(gnew)}/; ")
    new_sh = gov_cmd(gnew, new_body) + (f"for t in unit contract categorical; do {shlex.quote(py)} tests/run_all.py $t || rc=1; done; "
                                        f"for t in {new_suites}; do {shlex.quote(py)} $t || rc=1; done; exit $rc")
    old_body = (f"rsync -a --delete --exclude __pycache__ {shlex.quote(code)}/ {shlex.quote(gold)}/; "
                f"[ -d {shlex.quote(extras)} ] && rsync -a {shlex.quote(extras)}/ {shlex.quote(gold)}/ || true; "
                f"rsync -a --exclude __pycache__ {shlex.quote(gov)}/tests/ {shlex.quote(gold)}/tests/; ")
    old_sh = gov_cmd(gold, old_body) + (f"{shlex.quote(py)} tests/run_all.py categorical || rc=1; "
                                        f"for t in {new_suites}; do {shlex.quote(py)} $t || rc=1; done; exit $rc")
    add(id="gov015_tests", title="GOV-015: unit, contract and categorical tiers plus the new suites on the CORRECTED code (staged copy)",
        kind="cmd", cmd=["bash", "-c", new_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.4)
    add(id="gov015_tests_before", title="GOV-015: the new tests and categorical tier on the OLD code (expected RED: the failure-before evidence)",
        kind="cmd", cmd=["bash", "-c", old_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.3)
    # 2a2. GOV-015 re-run (2026-10-25): the first pass printed only the tail of a failing suite, so the failing
    # scenarios of the corrected categorical suites were not visible. Each suite runs here on its own with its
    # whole output; _vizdoom_smoke runs 3x on both trees (a timing contract: flaky or deterministic?).
    cats = "boundaries cat01 cat02 cat03 cat04 cat05 cat06 cat07"
    rr_sh = gov_cmd(gnew, new_body) + (
        f"for s in {cats}; do echo \"=== NEW $s\"; {shlex.quote(py)} tests/_categorical_${{s}}_smoke.py 2>&1 | grep -vE 'Warning|init\\.|PyKEEN|Geometric|Stable-Base' || true; done; "
        f"for i in 1 2 3; do echo \"=== NEW vizdoom run $i\"; {shlex.quote(py)} tests/_vizdoom_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS' || rc=1; done; "
        f"for t in _clean_exit _incident_bundle; do echo \"=== NEW $t (run.py rename)\"; {shlex.quote(py)} tests/${{t}}_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS|Error' || rc=1; done; "
        f"echo '=== NEW foundation runtime unit'; {shlex.quote(py)} tests/unit/test_foundation_runtime_unit.py 2>&1 | tail -3 || rc=1; "
        f"echo '=== NEW universal_loop'; {shlex.quote(py)} tests/_universal_loop_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS' || rc=1; "
        f"{old_body} ln -sfn /workspace/devai/roms {shlex.quote(gold)}/roms; cd {shlex.quote(gold)}; export PYTHONPATH={shlex.quote(gold)}; "
        f"for i in 1 2 3; do echo \"=== OLD vizdoom run $i\"; {shlex.quote(py)} tests/_vizdoom_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS' || rc=1; done; exit $rc")
    add(id="gov015_rerun", title="GOV-015 re-run: full output of the corrected categorical suites, vizdoom J on both trees (3x), universal-loop G",
        kind="cmd", cmd=["bash", "-c", rr_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.3)
    # GOV-015 9.6: the two contracts the re-run showed to be test defects (vizdoom J has no think time; universal-loop G tests the
    # removed `legacy` action) are corrected; this re-runs only them, on the staged tree, to confirm.
    rr2 = (f"cd {shlex.quote(gov)}; rc=0; export PYTHONPATH={shlex.quote(gov)}; echo '=== universal_loop'; "
           f"{shlex.quote(py)} tests/_universal_loop_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS' || rc=1; "
           f"for i in 1 2 3; do echo \"=== vizdoom run $i\"; {shlex.quote(py)} tests/_vizdoom_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS' || rc=1; done; exit $rc")
    add(id="gov015_rerun2", title="GOV-015 re-run 2: vizdoom J (3x) and universal-loop G after the two test corrections",
        kind="cmd", cmd=["bash", "-c", rr2], cwd=c.root, env={}, requires=["gov015_rerun"], on_fail="continue", est_hours=0.2)
    # GOV-015 9.6 verification (separate from the re-run above): the fix for the competence/context-switch defect, the
    # corrected categorical scenarios, and the two tiers they must not break. The new tree is rebuilt from the staged copy;
    # the OLD code runs the new suite and must FAIL it (failure-before evidence: a test that cannot fail proves nothing).
    ver_sh = gov_cmd(gnew, new_body) + (
        "echo '=== NEW _context_hold'; " + f"{shlex.quote(py)} tests/_context_hold_smoke.py 2>&1 | grep -E 'FAIL|ALL PASS|Error'; [ ${{PIPESTATUS[0]}} -eq 0 ] || rc=1; "
        f"for s in {cats}; do echo \"=== NEW $s\"; {shlex.quote(py)} tests/_categorical_${{s}}_smoke.py 2>&1 | grep -E '^\\[categorical|FAIL|BLOCKED'; [ ${{PIPESTATUS[0]}} -eq 0 ] || rc=1; done; "
        f"for t in unit contract; do echo \"=== NEW tier $t\"; {shlex.quote(py)} tests/run_all.py $t 2>&1 | tail -4; [ ${{PIPESTATUS[0]}} -eq 0 ] || rc=1; done; "
        f"{old_body} ln -sfn /workspace/devai/roms {shlex.quote(gold)}/roms; cd {shlex.quote(gold)}; export PYTHONPATH={shlex.quote(gold)}; "
        f"echo '=== OLD _context_hold (must FAIL)'; "
        f"if {shlex.quote(py)} tests/_context_hold_smoke.py >/dev/null 2>&1; then echo 'UNEXPECTED PASS on the old code'; rc=1; else echo 'failed on the old code, as required'; fi; exit $rc")
    add(id="gov015_verify", title="GOV-015 9.6 verification: context-hold suite (new green, old red), the corrected categorical suites, unit and contract tiers",
        kind="cmd", cmd=["bash", "-c", ver_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.5)
    # GOV-015 9.9: five categorical scenarios and one contract-tier traceback stayed red after the verification. This stage
    # reports what the harness cut off (full tracebacks via CAT_TRACE) and the BND-04 arm trajectories. It decides nothing, is
    # required by nothing and cannot pause the campaign.
    diag_sh = gov_cmd(gnew, new_body) + (
        "export CAT_TRACE=1; for s in cat05 cat06 boundaries; do echo \"=== $s (full tracebacks)\"; "
        f"{shlex.quote(py)} tests/_categorical_${{s}}_smoke.py 2>&1 | grep -E -A16 'TRACEBACK|^  FAIL' | grep -vE 'Warning|PyKEEN' | head -150; done; "
        f"echo '=== BND-04 arms'; {shlex.quote(py)} tools/gov015_bnd04_diag.py 2>&1 | tail -60; "
        f"echo '=== contract tier (failing suites with context)'; {shlex.quote(py)} tests/run_all.py contract 2>&1 | grep -E -B4 -A14 'Traceback|FAIL' | head -120; exit 0")
    add(id="gov015_diag2", title="GOV-015 9.9 diagnostic: full tracebacks of the red categorical scenarios, the BND-04 arm trajectories and the contract-tier failure",
        kind="cmd", cmd=["bash", "-c", diag_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.3)
    # GOV-015 9.11: the diagnostic above could not print the BND-04 arm trajectories (the scenario module runs its suite on import);
    # the tool now loads it without running it. Reports only. The re-run (gov015_verify2) is the verification of the test fixes of
    # 9.10 and has its own id so the first verification's result stays on record.
    diag3_sh = gov_cmd(gnew, new_body) + (f"{shlex.quote(py)} tools/gov015_bnd04_diag.py 2>&1 | grep -vE 'Warning|PyKEEN'; exit 0")
    add(id="gov015_diag3", title="GOV-015 9.11 diagnostic: BND-04 arm trajectories (hold, context count, progress) and the CT-03 curriculum pulls",
        kind="cmd", cmd=["bash", "-c", diag3_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.1)
    add(id="gov015_verify2", title="GOV-015 9.10 verification (second): the same checks as gov015_verify on the corrected tests and staging",
        kind="cmd", cmd=["bash", "-c", ver_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.5)
    # GOV-015 9.13: the BND-04 probe printed nothing on its first run (the scenario module takes over stdout); the tool now writes to the real
    # stdout. The contract tier is listed in full so every red suite is named. Reports only.
    diag4_sh = gov_cmd(gnew, new_body) + (f"{shlex.quote(py)} tools/gov015_bnd04_diag.py 2>&1 | grep -vE 'Warning|PyKEEN'; "
                                          f"echo '=== contract tier, every suite'; {shlex.quote(py)} tests/run_all.py contract 2>&1 | grep -E '^  (PASS|FAIL)|suite' ; exit 0")
    add(id="gov015_diag4", title="GOV-015 9.13 diagnostic: BND-04 arm trajectories and CT-03 pulls (stdout fixed), and every contract-tier suite's verdict",
        kind="cmd", cmd=["bash", "-c", diag4_sh], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.2)
    # 2a3. WP-20 (GOV-016): predictive-coding core vs the RSSM. Everything runs on <root>/pc_code, a SNAPSHOT of
    # gov015_code taken when pc_snapshot starts (so it carries the GOV-015 fixes AND the PC core); the frozen `code` copy
    # is untouched. The snapshot has its own baseline arm (the RSSM, on the same snapshot), so the A/B compares one
    # difference. The pilot is a GATE: t1_wp20_pc_core requires pc_pilot, and pc_pilot exits non-zero when the
    # pre-registered gate (tools/pc_pilot.py, GOV-016 section 5) fails, which blocks the five-seed test.
    pcc = os.path.join(c.root, "pc_code")
    snap = (f"set -e; [ -d {shlex.quote(gov)} ] || {{ echo 'gov015_code is not staged'; exit 3; }}; "
            f"rsync -a --delete --exclude __pycache__ --exclude venv --exclude runlogs --exclude .git "
            f"{shlex.quote(gov)}/ {shlex.quote(pcc)}/; "
            f"[ -e {shlex.quote(pcc)}/roms ] || ln -sfn /workspace/devai/roms {shlex.quote(pcc)}/roms; cd {shlex.quote(pcc)}; "
            f"find developmental_ai scripts tools configs tests -type f \\( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \\) "
            f"| sort | xargs sha256sum | sha256sum | cut -c1-12 | tee SNAPSHOT_HASH")
    add(id="pc_snapshot", title="WP-20: snapshot the staged code (GOV-015 fixes + predictive-coding core) for the PC tests",
        kind="cmd", cmd=["bash", "-c", snap], cwd=c.root, env={}, requires=["ab00"], on_fail="continue", est_hours=0.05)
    # ASS-020 F-01: the suites that decide whether the PC build is sane gate the pilot and, through it, the A/B. The wider
    # list (the unit tier and suites that exercise the campaign, the entry point and growth) is recorded but required by
    # NOTHING: a suite that is red for an unrelated reason must not latch the experiment shut (CLAUDE.md 4.1).
    pc_req = "tests/_predictive_coding_smoke.py tests/_pc_pilot_gate_smoke.py tests/_config_keys_smoke.py"
    pc_info = "tests/_campaign_smoke.py tests/_clean_exit_smoke.py tests/_incident_bundle_smoke.py tests/_growth_smoke.py"

    def pc_sh(suites, unit):
        return (f"cd {shlex.quote(pcc)}; rc=0; export PYTHONPATH={shlex.quote(pcc)}; "
                + (f"{shlex.quote(py)} tests/run_all.py unit || rc=1; " if unit else "")
                + f"for t in {suites}; do echo \"=== $t\"; {shlex.quote(py)} $t 2>&1 | grep -vE 'Warning|PyKEEN|Geometric|Stable-Base' | tail -40 || true; "
                  f"{shlex.quote(py)} $t >/dev/null 2>&1 || {{ echo \"FAILED $t\"; rc=1; }}; done; exit $rc")
    add(id="pc_tests", title="WP-20 REQUIRED suites: predictive-coding, pilot-gate and config-key suites on the snapshot (the pilot and the A/B wait for these)",
        kind="cmd", cmd=["bash", "-c", pc_sh(pc_req, False)], cwd=c.root, env={}, requires=["pc_snapshot"],
        on_fail="continue", est_hours=0.1)
    add(id="pc_tests_info", title="WP-20 informational suites: unit tier, campaign, entry-point, incident-bundle and growth suites on the snapshot (required by nothing)",
        kind="cmd", cmd=["bash", "-c", pc_sh(pc_info, True)], cwd=c.root, env={}, requires=["pc_snapshot"],
        on_fail="continue", est_hours=0.1)
    # GOV-016 section 11c: pc_tests contract G failed on the first run (generative error 0.0379 -> 1.5309). The diagnostic
    # separates "the target moved" from "the local rule does not converge". It runs on the staged tree (no snapshot), reports,
    # decides nothing and is required by nothing.
    add(id="pc_hebbian_diag", title="WP-20 diagnostic: why contract G's generative error rose (isolated rule vs in-model, Hebbian vs backprop ablation)",
        kind="cmd", cmd=[py, os.path.join(gov, "tools/pc_hebbian_diag.py"), "--out", os.path.join(c.stages, "pc_hebbian_diag")],
        cwd=gov, env={"PYTHONPATH": gov}, requires=["ab00"], on_fail="continue", est_hours=0.1)
    add(id="pc_boot_check", title="WP-20: boot the predictive-coding arm on the snapshot (4 min; a traceback blocks the A/B)",
        kind="cmd", on_fail="continue", requires=["pc_snapshot", "pc_tests"], est_hours=0.1,
        cmd=[py, os.path.join(pcc, "tools/arm_boot_check.py"), "--exp", os.path.join(c.stages, "pc_boot_check"),
             "--code", pcc, "--venv", c.venv, "--overlays", c.overlays, "--names", "wp20_pc_core",
             "--minutes", "4", "--out", os.path.join(c.analysis, "arm_boot_checks.json")], cwd=pcc, env={})
    add(id="pc_pilot", title="WP-20 PILOT GATE: RSSM vs Hebbian predictive coding (plus a backprop ablation) on fixed Doom data (tiny, 3000 steps; exit 1 = gate failed)",
        kind="cmd", on_fail="continue", requires=["pc_snapshot", "pc_tests", "pc_boot_check"], est_hours=1.6,
        cmd=[py, os.path.join(pcc, "tools/pc_pilot.py"), "--out", os.path.join(c.stages, "pc_pilot"),
             "--device", "cuda", "--frames", "12000", "--steps", "3000"], cwd=pcc, env={"PYTHONPATH": pcc})
    # GOV-016 11e: the SECOND pilot attempt and the learning-rate diagnostic, in one run on the first pilot's data (copied, so the
    # data is identical). The registered arms (p2a, p2b) get the UNCHANGED gate; the diagnostic arms are reported only. It waits for
    # its own few-step smoke so a defect costs a minute, not the run; it gates nothing and no stage requires it (a pass is a licence
    # for an A/B whose overlay is decided after the result).
    p2_dir = os.path.join(c.stages, "pc_pilot2")
    add(id="pc_pilot2_tests", title="WP-20 second pilot: the tool's few-step smoke (registration, unchanged gate, a real CPU run of every arm kind)",
        kind="cmd", on_fail="continue", requires=["pc_snapshot"], est_hours=0.1, cwd=pcc, env={"PYTHONPATH": pcc, "CUDA_VISIBLE_DEVICES": ""},
        cmd=["bash", "-c", f"cd {shlex.quote(pcc)}; {shlex.quote(py)} tests/_pc_pilot2_smoke.py 2>&1 | grep -vE 'Warning|PyKEEN'| tail -40; exit ${{PIPESTATUS[0]}}"])
    add(id="pc_pilot2", title="WP-20 PILOT, SECOND ATTEMPT (registered arms p2a/p2b gated; lr/clip diagnostic arms reported) on the first pilot's data, tiny, 3000 steps",
        kind="cmd", on_fail="continue", requires=["pc_snapshot", "pc_tests", "pc_pilot2_tests"], est_hours=1.6, cwd=pcc, env={"PYTHONPATH": pcc},
        cmd=["bash", "-c", f"set -e; mkdir -p {shlex.quote(p2_dir)}; src={shlex.quote(os.path.join(c.stages, 'pc_pilot', 'data.npz'))}; "
             f"[ -f \"$src\" ] || {{ echo 'the first pilot data is missing'; exit 3; }}; cp -n \"$src\" {shlex.quote(p2_dir)}/data.npz; cd {shlex.quote(pcc)}; "
             f"{shlex.quote(py)} tools/pc_pilot2.py --data {shlex.quote(p2_dir)}/data.npz --out {shlex.quote(p2_dir)} --device cuda --steps 3000"])
    # GOV-016 11f: the optional target normalisation (pc_target_norm, default off). Its suite and the required PC suites re-run on the
    # snapshot (the core's code changed after pc_tests passed), then a DIAGNOSTIC-ONLY run of two normalised arms against the RSSM on the
    # same data. Nothing is gated, no attempt is counted, nothing requires it.
    n_dir = os.path.join(c.stages, "pc_pilot3_diag")
    add(id="pc_norm_tests", title="WP-20 target-normalisation suite plus the required PC suites on the refreshed snapshot",
        kind="cmd", cmd=["bash", "-c", pc_sh("tests/_pc_target_norm_smoke.py tests/_predictive_coding_smoke.py tests/_pc_pilot_gate_smoke.py "
                                             "tests/_pc_pilot2_smoke.py tests/_config_keys_smoke.py", False)],
        cwd=c.root, env={}, requires=["pc_snapshot"], on_fail="continue", est_hours=0.1)
    add(id="pc_pilot3_diag", title="WP-20 DIAGNOSTIC ONLY: RSSM vs the Hebbian core with pc_target_norm (lr 0.05 and 0.01) on the first pilot's data (nothing gated)",
        kind="cmd", on_fail="continue", requires=["pc_snapshot", "pc_norm_tests"], est_hours=0.7, cwd=pcc, env={"PYTHONPATH": pcc},
        cmd=["bash", "-c", f"set -e; mkdir -p {shlex.quote(n_dir)}; src={shlex.quote(os.path.join(c.stages, 'pc_pilot', 'data.npz'))}; "
             f"[ -f \"$src\" ] || {{ echo 'the first pilot data is missing'; exit 3; }}; cp -n \"$src\" {shlex.quote(n_dir)}/data.npz; cd {shlex.quote(pcc)}; "
             f"{shlex.quote(py)} tools/pc_pilot2.py --data {shlex.quote(n_dir)}/data.npz --out {shlex.quote(n_dir)} --device cuda --steps 3000 --arms rssm,d_norm,d_norm01"])
    # GOV-016 11g: the FIRST pc_boot_check ended "done" (exit 0) while its record said `wp20_pc_core: FAIL exit code 3` ("agent never
    # came up"), and the booted root was deleted, so the cause was unreadable. The boot tool now keeps the logs and, with --strict, fails
    # visibly. This re-boots the arm on the refreshed snapshot (it rewrites the arm's entry in arm_boot_checks.json, which is what blocks or
    # releases the A/B). Its own suite first; nothing requires it, it cannot pause the campaign.
    add(id="pc_boot_tests", title="WP-20 boot-check tool suite (keeps logs, strict exit) and the report-tools suite on the refreshed snapshot",
        kind="cmd", cmd=["bash", "-c", pc_sh("tests/_arm_boot_check_smoke.py tests/_report_tools_smoke.py", False)],
        cwd=c.root, env={}, requires=["pc_snapshot"], on_fail="continue", est_hours=0.05)
    # GOV-016 11g: the first test that runs the PC core inside the real loop (construction, acting, training through the loop's own path,
    # publishing the buffers to the acting copy, a checkpoint). It runs BEFORE the 4-minute boot so a loop-level defect shows with a traceback.
    add(id="pc_loop_tests", title="WP-20 live-loop suite: the PC core built, acted, trained, published and checkpointed by the real loop (Crafter, CPU)",
        kind="cmd", cmd=["bash", "-c", pc_sh("tests/_pc_loop_smoke.py", False).replace("grep -vE 'Warning|PyKEEN|Geometric|Stable-Base' | tail -40", "grep -vE 'Warning|PyKEEN|Geometric|Stable-Base' | tail -80")],
        cwd=c.root, env={}, requires=["pc_snapshot"], on_fail="continue", est_hours=0.1)
    add(id="pc_boot_check2", title="WP-20: boot the predictive-coding arm again on the refreshed snapshot, keeping the logs, failing loudly (4 min)",
        kind="cmd", on_fail="continue", requires=["pc_snapshot", "pc_boot_tests"], est_hours=0.1, cwd=pcc, env={},
        cmd=[py, os.path.join(pcc, "tools/arm_boot_check.py"), "--exp", os.path.join(c.stages, "pc_boot_check2"),
             "--code", pcc, "--venv", c.venv, "--overlays", c.overlays, "--names", "wp20_pc_core",
             "--minutes", "4", "--out", os.path.join(c.analysis, "arm_boot_checks.json"), "--strict"])
    # GOV-016 11i: diagnostic only (no registered arm, nothing gated, no attempt counted): the normalised head still has its step pinned at the
    # clip (11f result), so two smaller clips against the RSSM reference on the same data.
    n4 = os.path.join(c.stages, "pc_pilot4_diag")
    add(id="pc_pilot4_diag", title="WP-20 DIAGNOSTIC ONLY: normalised Hebbian core with a smaller step clip (0.01, 0.002) against the RSSM (nothing gated)",
        kind="cmd", on_fail="continue", requires=["pc_snapshot", "pc_norm_tests"], est_hours=0.7, cwd=pcc, env={"PYTHONPATH": pcc},
        cmd=["bash", "-c", f"set -e; mkdir -p {shlex.quote(n4)}; src={shlex.quote(os.path.join(c.stages, 'pc_pilot', 'data.npz'))}; "
             f"[ -f \"$src\" ] || {{ echo 'the first pilot data is missing'; exit 3; }}; cp -n \"$src\" {shlex.quote(n4)}/data.npz; cd {shlex.quote(pcc)}; "
             f"{shlex.quote(py)} tools/pc_pilot2.py --data {shlex.quote(n4)}/data.npz --out {shlex.quote(n4)} --device cuda --steps 3000 --arms rssm,d_norm_c01,d_norm_c002"])
    ev("t1_wp20_pc_core", "T1 WP-20 wp20_pc_core: RSSM vs predictive coding, both on the snapshot (5 seeds each)",
       [("baseline", "-"), ("changed", os.path.join(c.overlays, "wp20_pc_core.yaml"))], [1, 2, 3, 4, 5],
       ["pc_pilot", "pc_boot_check"], 10.0, code_root=pcc)
    add(id="an_t1_wp20_pc_core", title="Score and compare T1 wp20_pc_core (same-snapshot baseline vs predictive coding)",
        kind="analyze", requires=["t1_wp20_pc_core"], on_fail="continue", base="t1_wp20_pc_core",
        changed="t1_wp20_pc_core", test_type="T1", arm_base="baseline", arm_changed="changed",
        title_text="WP-20 wp20_pc_core")
    # 2b. boot every arm for a few minutes (a never-booted overlay must not cost a 5-hour slot)
    boot_names = [n for n in arms_mod.T1 if arms_mod.blocked_reason("t1", n) is None and n not in arms_mod.OWN_CODE]
    for n in arms_mod.T2:
        if arms_mod.blocked_reason("t2", n) is None:
            boot_names += [f"{n}_connected", f"{n}_cut"]
    boot_names += list(arms_mod.CATEGORIES) + ["combined"]
    add(id="arm_boot_checks", title=f"Boot-check {len(boot_names)} arm overlays (4 min each; traceback = arm blocked)",
        # NOT gated on the test tiers: they are informational and known reds exist (ASS-017). Gating an A/B queue on
        # a suite that cannot go green is a guard-becomes-latch (CLAUDE.md 4.1): it blocked every A/B on 2026-10-23.
        kind="cmd", on_fail="continue", requires=["ab00"], est_hours=len(boot_names) * 0.1,
        cmd=[py, os.path.join(code, "tools/arm_boot_check.py"), "--exp", os.path.join(c.stages, "arm_boot_checks"),
             "--code", code, "--venv", c.venv, "--overlays", c.overlays, "--names", ",".join(boot_names),
             "--minutes", "4", "--out", os.path.join(c.analysis, "arm_boot_checks.json")], cwd=code, env={})
    # 2c. GOV-014 item C: does the diverse source phase really change maps in the real adapter? (6 min)
    add(id="wp6_coverage_check", title="WP-6 live coverage check: >= 12 distinct source maps in 6 minutes, none held out",
        kind="cmd", on_fail="continue", requires=["arm_boot_checks"], est_hours=0.2,
        cmd=[py, os.path.join(code, "tools/map_coverage_check.py"), "--exp", os.path.join(c.stages, "wp6_coverage_check"),
             "--code", code, "--venv", c.venv, "--overlay", os.path.join(c.overlays, "wp6_coverage_probe.yaml"),
             "--minutes", "6", "--min-maps", "12", "--out", os.path.join(c.analysis, "arm_boot_checks.json"),
             "--key", "wp6_coverage"], cwd=code, env={})
    # 2d. GOV-014 item A: the fresh control (ASS-007 section 4, Amendment 8): target phase only, no source training
    ev("fresh_control", "Fresh control: no source phase, target only (seeds 11-20)", [("fresh_control", "-")],
       list(range(11, 21)), ["ab00"], 3.4, ["--fresh-control", "fresh_control"])
    add(id="an_fresh_control", title="Score the fresh control against the source-trained AB-00 target phases (descriptive)",
        kind="analyze", requires=["fresh_control"], on_fail="continue", base="ab00", changed="fresh_control",
        test_type="T1", arm_base="baseline_a", arm_changed="fresh_control", title_text="Fresh control vs AB-00 arm A")
    # 3. split sensitivity pilot (ASS-007 14.9 b): the same baseline at two other splits
    for frac, tag in ((0.5, "split50"), (0.8, "split80")):
        ev(f"sens_{tag}", f"Split-sensitivity pilot, source fraction {frac} (baseline, seeds 11-13)",
           [("baseline", "-")], [11, 12, 13], ["ab00"], 3.0, ["--source-frac", str(frac)])
    # 4. the categorical campaign front half (deterministic harness + sizing + human gate)
    add(id="cat_harness", title="Categorical deterministic contracts CT-01..08 and boundaries BND-01..07",
        kind="cmd", cmd=[py, "tests/run_all.py", "categorical"], cwd=code,
        # A real defect found here must BLOCK the categorical pilots (ASS-012 5.2: a failed required
        # check blocks live execution of that composition) but must not stop the T1/T2 queue.
        # The per-contract records (ASS-012 7) are archived under the stage directory.
        env={"PYTHONPATH": code, "CATEGORICAL_OUT": os.path.join(c.stages, "cat_harness", "records")},
        on_fail="continue", requires=["tests_unit"], est_hours=0.5,
        needs_glob=("tests/_categorical_*_smoke.py",
                    "CS-1/CS-2 harness not built (no tests/_categorical_*_smoke.py; GOV-013 4)"))
    add(id="cat_sizing", title="Independent sizing pilot (baseline, 20 min, seed 99, resources sampled)",
        kind="cmd", cmd=["bash", os.path.join(code, "tools/eval_run.sh"), "source", "sizing_s99", "99", "-",
                         "doom", "20"], cwd=code, requires=["cat_harness"], est_hours=0.4, sample=True,
        env={"EVAL_EXP": os.path.join(c.stages, "cat_sizing"), "EVAL_CODE": code, "EVAL_VENV": c.venv})
    add(id="cat_thresholds_frozen", title="HUMAN GATE: freeze ASS-012 resource/latency thresholds from the sizing pilot",
        kind="gate", gate="cat_thresholds_frozen", requires=["cat_sizing"])
    for cat in arms_mod.CATEGORIES:
        for variant in ("baseline", "enabled"):
            sid = f"pilot_{cat}_{variant}"
            ov = "-" if variant == "baseline" else os.path.join(c.overlays, f"{cat}.yaml")
            add(id=sid, title=f"Categorical operational pilot {cat} {variant} (20 min, seed 100)",
                kind="cmd", sample=True, requires=["cat_thresholds_frozen"], est_hours=0.35,
                cmd=["bash", os.path.join(code, "tools/eval_run.sh"), "source", f"{cat}_{variant}_s100",
                     "100", ov, "doom", "20"], cwd=code,
                env={"EVAL_EXP": os.path.join(c.stages, sid), "EVAL_CODE": code, "EVAL_VENV": c.venv})
    comb = os.path.join(c.overlays, "combined.yaml")
    ev("cat_combined", "Categorical combined operational run (seeds 101-103, 40/20)",
       [("combined", comb)], [101, 102, 103],
       [x["id"] for x in S if x["id"].startswith("pilot_")], 3.0)
    # 5. shared baseline + every T1 (block) A/B
    ev("ab_baseline", "Shared baseline for T1/T2 (5 seeds)", [("baseline", "-")], [1, 2, 3, 4, 5],
       ["ab00"], 5.0)
    for name, (wp, _, _) in arms_mod.T1.items():
        if name in arms_mod.OWN_CODE:
            continue                                    # queued earlier with its own snapshot and baseline (2a3)
        why = arms_mod.blocked_reason("t1", name)
        ev(f"t1_{name}", f"T1 {wp} {name}: baseline vs +block (5 seeds)",
           [("changed", os.path.join(c.overlays, f"{name}.yaml"))], [1, 2, 3, 4, 5],
           ["ab_baseline"], 5.0, blocked=why)
        add(id=f"an_t1_{name}", title=f"Score and compare T1 {name}", kind="analyze",
            requires=[f"t1_{name}"], on_fail="continue", base="ab_baseline", changed=f"t1_{name}",
            test_type="T1", arm_base="baseline", arm_changed="changed", title_text=f"{wp} {name}")
    # 6. every T2 (relationship) A/B: connected vs cut, same blocks in both
    for name in arms_mod.T2:
        why = arms_mod.blocked_reason("t2", name)
        ev(f"t2_{name}", f"T2 {name}: line connected vs cut (5 seeds)",
           [("connected", os.path.join(c.overlays, f"{name}_connected.yaml")),
            ("cut", os.path.join(c.overlays, f"{name}_cut.yaml"))], [1, 2, 3, 4, 5], ["ab00"], 10.0,
           blocked=why)
        add(id=f"an_t2_{name}", title=f"Score and compare T2 {name}", kind="analyze",
            requires=[f"t2_{name}"], on_fail="continue", base=f"t2_{name}", changed=f"t2_{name}",
            test_type="T2", arm_base="cut", arm_changed="connected", title_text=name)
    # every arm-carrying stage waits for the boot checks and is blocked if ITS arm failed them
    for st in S:
        i = st["id"]
        arms_of = ([i[3:]] if i.startswith("t1_") else
                   [i[3:] + "_connected", i[3:] + "_cut"] if i.startswith("t2_") else
                   [i[len("pilot_"):].rsplit("_enabled", 1)[0]] if i.startswith("pilot_") and i.endswith("_enabled") else
                   ["combined"] if i == "cat_combined" else None)
        if arms_of is not None or (i.startswith("pilot_") and st["kind"] == "cmd"):
            st["requires"] = st["requires"] + ["arm_boot_checks"]
            st["boot_arms"] = arms_of or []
        if i == "t1_wp6_diversity":                       # also gated by the live coverage check (GOV-014 item C)
            st["requires"] = st["requires"] + ["wp6_coverage_check"]
            st["boot_arms"] = st["boot_arms"] + ["wp6_coverage"]
    return S


class Cfg:
    def __init__(self, a):
        self.root = os.path.abspath(a.root)
        self.code = os.path.abspath(a.code or os.path.join(self.root, "code"))
        self.venv = a.venv
        self.py = os.path.join(a.venv, "bin", "python") if a.venv else sys.executable
        self.ab00_root = a.ab00_root
        self.stages = os.path.join(self.root, "stages")
        self.overlays = os.path.join(self.root, "overlays")
        self.gates = os.path.join(self.root, "gates")
        self.analysis = os.path.join(self.root, "analysis")
        self.state_path = os.path.join(self.root, "campaign_state.json")
        self.poll = a.poll
        self.idle = a.idle
        self.min_free_gb = getattr(a, "min_free_gb", 30.0)
        self.sweep_every = getattr(a, "sweep_every", 600.0)


# ------------------------------------------------------------------------------ the driver
class Campaign:
    def __init__(self, cfg: Cfg, stages: Optional[List[Dict[str, Any]]] = None, accept_code: bool = False):
        self.c = cfg
        for d in (cfg.root, cfg.stages, cfg.overlays, cfg.gates, cfg.analysis):
            os.makedirs(d, exist_ok=True)
        self.write_overlays()
        self.stages = stages if stages is not None else build_stages(cfg)
        self.by_id = {s["id"]: s for s in self.stages}
        self.procs: Dict[str, subprocess.Popen] = {}
        self.state = self._load()
        self._code_check(accept_code)

    # ---- state
    @staticmethod
    def valid_json(path: str) -> bool:
        try:
            json.load(open(path))
            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def atomic_write(path: str, text: str) -> None:
        """Write, fsync the file, rename over, fsync the directory: after a power cut the file is
        the old version or the new one, never a truncated one."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _load(self) -> Dict[str, Any]:
        p = self.c.state_path
        d = None
        if os.path.exists(p) or os.path.exists(p + ".bak"):
            for cand, label in ((p, "state"), (p + ".bak", "backup")):
                if os.path.exists(cand) and self.valid_json(cand):
                    d = json.load(open(cand))
                    if cand != p:
                        print(f"{now()} RESTORED campaign state from its backup (the main file was unreadable, "
                              "e.g. a power cut mid-write)", flush=True)
                    break
            if d is None:
                # Never start from scratch over a damaged record: that would re-run finished work and
                # forget the frozen code hash. A human decides (delete the file on purpose to restart).
                raise SystemExit(f"{p} and its .bak are both unreadable; refusing to start over silently. "
                                 "Inspect them, restore one, or delete both deliberately to begin a new campaign.")
        else:
            d = {"created": now(), "code_hash": None, "paused": None, "stages": {}}
        for s in self.stages:
            d["stages"].setdefault(s["id"], {"status": "pending", "attempts": 0})
        return d

    def save(self) -> None:
        p = self.c.state_path
        if os.path.exists(p) and self.valid_json(p):       # keep the last good copy; never copy a bad one
            import shutil
            shutil.copyfile(p, p + ".bak")
        self.atomic_write(p, json.dumps(self.state, indent=1, sort_keys=True))

    def log(self, msg: str) -> None:
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(os.path.join(self.c.root, "campaign.log"), "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def st(self, sid: str) -> Dict[str, Any]:
        return self.state["stages"][sid]

    def set(self, sid: str, **kw) -> None:
        self.st(sid).update(kw)
        self.save()
        self.write_status()

    # ---- overlays and code freeze
    def write_overlays(self) -> None:
        import yaml
        o = self.c.overlays
        out: Dict[str, dict] = {}
        for n in arms_mod.T1:
            if arms_mod.blocked_reason("t1", n) is None:
                out[n] = arms_mod.t1_overlay(n)
        for n in arms_mod.T2:
            if arms_mod.blocked_reason("t2", n) is None:
                out[f"{n}_connected"], out[f"{n}_cut"] = arms_mod.t2_overlays(n)
        comb: dict = {}
        for n in arms_mod.CATEGORIES:
            out[n] = arms_mod.category_overlay(n)
            comb = arms_mod.merge(comb, out[n])
        out["combined"] = comb
        out["wp6_coverage_probe"] = arms_mod.wp6_probe_overlay()
        for n, ov in out.items():
            Path(o, f"{n}.yaml").write_text(yaml.safe_dump(ov, sort_keys=True))

    def code_hash(self) -> str:
        h = hashlib.sha256()
        base = Path(self.c.code)
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    h.update(str(p.relative_to(base)).encode())
                    h.update(hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()

    def _code_check(self, accept: bool) -> None:
        cur = self.code_hash()
        old = self.state.get("code_hash")
        if old is None or accept:
            if old and accept and old != cur:
                self.log(f"CODE CHANGE ACCEPTED: {old[:12]} -> {cur[:12]} (a new snapshot, ASS-012 6)")
            self.state["code_hash"] = cur
            self.save()
        elif old != cur:
            raise SystemExit(f"staged code changed since the campaign froze it ({old[:12]} != {cur[:12]}); "
                             "re-run with --accept-code-change after a fix (a fix starts a new snapshot)")

    # ---- status of a stage
    def gate_open(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.c.gates, name))

    def block_reason(self, s: Dict[str, Any]) -> Optional[str]:
        if s.get("blocked"):
            return s["blocked"]
        if s.get("needs_glob") and not sorted(Path(self.c.code).glob(s["needs_glob"][0])):
            return s["needs_glob"][1]
        if s.get("boot_arms"):
            bp = os.path.join(self.c.analysis, "arm_boot_checks.json")
            res = json.load(open(bp)) if os.path.exists(bp) else {}
            for arm in s["boot_arms"]:
                if arm in res and not res[arm]["ok"]:
                    return f"arm {arm} failed its boot check ({res[arm]['reason']}); fix, then `resume arm_boot_checks`"
        for r in s["requires"]:
            rs = self.st(r)["status"]
            if rs in ("failed", "skipped"):
                return f"requires {r} which {rs}"
            if rs not in OK and self.block_reason(self.by_id[r]):
                return f"requires {r}, which is blocked"
        return None

    def ready(self, s: Dict[str, Any]) -> bool:
        return all(self.st(r)["status"] in OK for r in s["requires"])

    def classify(self, s: Dict[str, Any]) -> str:
        status = self.st(s["id"])["status"]
        if status in TERMINAL:
            return status
        if self.block_reason(s):
            return "blocked"
        if not self.ready(s):
            return "queued"
        if s["kind"] == "gate" and not self.gate_open(s["gate"]):
            return "waiting-human"
        return "runnable" if status in ("pending", "running") else status

    # ---- host exclusivity
    def trainer_alive(self) -> bool:
        r = subprocess.run(["pgrep", "-f", TRAINER_PAT], capture_output=True, text=True)
        pids = [p for p in r.stdout.split() if p.isdigit() and int(p) != os.getpid()]
        return bool(pids)

    def pat_alive(self, pat: str) -> bool:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        return any(p.isdigit() and int(p) != os.getpid() for p in r.stdout.split())

    def pid_alive(self, pid: Optional[int], sid: Optional[str] = None) -> bool:
        """Our own child is judged by poll() (a finished child stays a zombie that os.kill still
        'finds'); a process from before a driver restart is not our child, so kill -0 is right."""
        if sid is not None and sid in self.procs:
            return self.procs[sid].poll() is None
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    # ---- disk: finished runs keep their results, not their 10 GB of checkpoints and replay
    def prune_runs(self, runs_dir: str, only: Optional[set] = None) -> int:
        """Delete the heavy, no-longer-needed parts of FINISHED runs: `devai/` (the run's code copy,
        checkpoints and replay buffers; ~5 GB) and `snap_source_end/logs` (a second copy of them; ~5 GB).
        Kept: results/, phase_*.json, the small snapshot parts. Measured 2026-10-22: one run is ~11 GB,
        results ~1 MB; 185 queued runs would otherwise need ~2 TB on a 232 GB disk. Only runs the
        caller says are finished (`only`) are touched; symlinks (venv_mc) are unlinked, never followed."""
        import shutil
        freed = 0
        if not os.path.isdir(runs_dir):
            return 0
        for name in sorted(os.listdir(runs_dir)):
            rd = os.path.join(runs_dir, name)
            if not os.path.isdir(rd) or os.path.exists(os.path.join(rd, ".pruned")):
                continue
            if only is not None and name not in only:
                continue
            for sub in ("devai", os.path.join("snap_source_end", "logs")):
                p = os.path.join(rd, sub)
                if os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    freed += sum(f.stat(follow_symlinks=False).st_size for f in Path(p).rglob("*") if f.is_file())
                    shutil.rmtree(p, ignore_errors=True)
            self.compress_logs(os.path.join(rd, "results"))
            Path(rd, ".pruned").write_text(f"{now()} {freed}\n")
        return freed

    @staticmethod
    def compress_logs(results_dir: str, min_bytes: int = 4096) -> int:
        """gzip every log-like file under a finished run's results/ (the scorer reads .gz). The original
        is removed only after the .gz reads back to the same size; phase_*.json stay plain. Returns
        bytes saved."""
        import gzip
        import shutil
        saved = 0
        for f in sorted(Path(results_dir).rglob("*")):
            if not f.is_file() or f.suffix == ".gz" or f.stat().st_size < min_bytes:
                continue
            gz = Path(str(f) + ".gz")
            try:
                with open(f, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
                with gzip.open(gz, "rb") as chk:
                    n = sum(len(b) for b in iter(lambda: chk.read(1 << 20), b""))
                if n != f.stat().st_size:
                    gz.unlink()
                    continue
                saved += f.stat().st_size - gz.stat().st_size
                f.unlink()
            except OSError:
                if gz.exists():
                    gz.unlink()
        return saved

    # ---- power loss and reboot
    @staticmethod
    def host_boot_time() -> Optional[float]:
        try:
            for ln in open("/proc/stat"):
                if ln.startswith("btime"):
                    return float(ln.split()[1])
        except OSError:
            pass
        return None

    def repair_after_reboot(self, boot: Optional[float] = None) -> List[str]:
        """Make the runners' state files safe to resume after a power cut or reboot. Two repairs, both
        done HERE because a running experiment's own code is frozen and must not change:
          1. integrity: a truncated/corrupt eval_state.json is restored from its last good backup
             (written by every sweep); with neither readable the queue PAUSES (never starts over).
          2. a run that was in progress when the HOST went down (its start time is before this boot) is
             an interruption, not a crash. The eval runner would count it as a crash and spend the seed's
             one retry (a second power cut would then end the seed with no claim, ASS-007 14.6); instead
             the run is reset to its beginning with its attempt count untouched and the interruption is
             recorded. A runner that is alive, or a driver death without a reboot, is left to the runner.
        Returns the run ids reset."""
        import shutil
        boot = boot if boot is not None else self.host_boot_time()
        hits: List[str] = []
        for s in self.stages:
            root = s.get("root")
            if s["kind"] != "eval" or not root:
                continue
            sp = os.path.join(root, "eval_state.json")
            if not (os.path.exists(sp) or os.path.exists(sp + ".bak")):
                continue
            if self.pat_alive(f"eval_runner.py --root {root}"):
                continue
            if not self.valid_json(sp):
                if self.valid_json(sp + ".bak"):
                    shutil.copyfile(sp + ".bak", sp)
                    self.log(f"RESTORED {sp} from its backup (unreadable after the interruption)")
                else:
                    self.state["paused"] = {"stage": s["id"], "at": now(),
                                            "reason": f"{sp} and its backup are unreadable; restore one by hand, then resume"}
                    self.save()
                    self.log(f"PAUSED: {self.state['paused']['reason']}")
                    continue
            d = json.load(open(sp))
            changed = False
            for rid, r in d.get("runs", {}).items():
                ip = r.get("in_progress")
                if ip and boot and float(ip.get("t", 0)) < boot:
                    r["in_progress"] = None
                    r["phase_done"] = []
                    r.setdefault("interruptions", []).append(
                        {"phase": ip.get("phase"), "started": ip.get("t"), "host_boot": boot, "handled": now()})
                    changed = True
                    hits.append(f"{s['id']}:{rid}")
                    self.log(f"INTERRUPTED {s['id']}:{rid} (phase {ip.get('phase')}) by a host restart: reset to the start of "
                             f"the seed, attempt count unchanged ({r.get('attempts')}), not counted as a crash")
            if changed:
                self.atomic_write(sp, json.dumps(d, indent=1, sort_keys=True))
        return hits

    def backup_eval_states(self) -> None:
        import shutil
        for s in self.stages:
            sp = os.path.join(s["root"], "eval_state.json") if s["kind"] == "eval" and s.get("root") else None
            if sp and os.path.exists(sp) and self.valid_json(sp):
                shutil.copyfile(sp, sp + ".bak")

    def sweep(self) -> int:
        """Prune every finished run in every stage root (AB-00 included)."""
        freed = 0
        self.backup_eval_states()
        for s in self.stages:
            root = s.get("root")
            if s["kind"] == "eval" and root and os.path.exists(os.path.join(root, "eval_state.json")):
                runs = json.load(open(os.path.join(root, "eval_state.json"))).get("runs", {})
                freed += self.prune_runs(os.path.join(root, "runs"), {k for k, v in runs.items() if v.get("status") == "done"})
            elif s["kind"] == "cmd" and self.st(s["id"])["status"] in OK:
                freed += self.prune_runs(os.path.join(self.c.stages, s["id"], "runs"))
        if freed:
            self.log(f"PRUNE freed {freed / 1e9:.1f} GB from finished runs (results and phase files kept)")
        return freed

    def free_gb(self) -> float:
        import shutil
        return shutil.disk_usage(self.c.root).free / 1e9

    # ---- executing one stage
    def stage_dir(self, sid: str) -> str:
        d = os.path.join(self.c.stages, sid)
        os.makedirs(d, exist_ok=True)
        return d

    def eval_done(self, s: Dict[str, Any]) -> Optional[str]:
        """'done' | 'done_with_failures' | None (not finished) from the runner's own state file."""
        p = os.path.join(s["root"], "eval_state.json")
        if not os.path.exists(p):
            return None
        runs = json.load(open(p)).get("runs", {})
        if not runs or any(r.get("status") not in ("done", "failed_final") for r in runs.values()):
            return None
        return "done_with_failures" if any(r["status"] == "failed_final" for r in runs.values()) else "done"

    def launch(self, s: Dict[str, Any]) -> None:
        sd = self.stage_dir(s["id"])
        exitf = os.path.join(sd, "exit_code")
        if os.path.exists(exitf):
            os.remove(exitf)
        script = f"{shlex.join(s['cmd'])}; echo $? > {shlex.quote(exitf)}"
        env = dict(os.environ, **{k: str(v) for k, v in (s.get("env") or {}).items()})
        logf = open(os.path.join(sd, "stage.log"), "ab")
        p = subprocess.Popen(["bash", "-c", script], cwd=s.get("cwd") or self.c.code, env=env,
                             stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        self.procs[s["id"]] = p
        self.set(s["id"], status="running", pid=p.pid, started=now(), attempts=self.st(s["id"])["attempts"] + 1)
        self.log(f"START {s['id']} pid {p.pid}: {s['title']}")

    def finish(self, s: Dict[str, Any], rc: Optional[int]) -> None:
        sid = s["id"]
        if s["kind"] == "eval":
            res = self.eval_done(s)
            ok = res is not None and (rc in (0, None))
            status = res if ok else "failed"
        else:
            ok = rc == 0
            status = "done" if ok else "failed"
        self.set(sid, status=status, ended=now(), rc=rc, pid=None)
        self.log(f"END   {sid}: {status} (rc={rc})")
        if status == "failed":
            if self.st(sid)["attempts"] < 2 and s["kind"] == "eval":
                self.log(f"RETRY {sid} once (same state, same seeds: the runner resumes)")
                self.set(sid, status="pending")
            elif s["on_fail"] == "pause":
                self.state["paused"] = {"stage": sid, "reason": f"exit {rc}", "at": now()}
                self.save()
                self.log(f"PAUSED on {sid}. Look at {self.stage_dir(sid)}/stage.log, then: campaign.py resume {sid}")

    def sampler(self, sid: str, stop: threading.Event) -> None:
        path = os.path.join(self.stage_dir(sid), "resources.csv")
        with open(path, "a") as f:
            f.write("ts,mem_available_mb,gpu_used_mb\n")
            while not stop.is_set():
                mem = gpu = ""
                try:
                    for ln in open("/proc/meminfo"):
                        if ln.startswith("MemAvailable"):
                            mem = str(int(ln.split()[1]) // 1024)
                    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5)
                    gpu = g.stdout.strip().split("\n")[0] if g.returncode == 0 else ""
                except Exception:
                    pass
                f.write(f"{time.time():.0f},{mem},{gpu}\n")
                f.flush()
                stop.wait(1.0)

    def run_stage(self, s: Dict[str, Any]) -> None:
        sid = s["id"]
        if s["kind"] == "gate":
            self.set(sid, status="done", ended=now())
            self.log(f"GATE  {sid} open")
            return
        if s["kind"] == "analyze":
            self.analyze(s)
            return
        stt = self.st(sid)
        ext = s.get("external_pat")
        attached = (self.pid_alive(stt.get("pid"), sid) and stt["status"] == "running") or (ext and self.pat_alive(ext))
        if not attached:
            while self.trainer_alive():
                self.log(f"WAIT  {sid}: another trainer holds the host (never run beside it; never killed)")
                if self.sleep(self.c.idle):
                    return
            # Disk guard (CLAUDE.md 4.1: it must be able to re-open). Prune finished runs first; if the
            # disk is STILL short, pause rather than fill it. Re-opens by freeing space, then `resume`.
            self.sweep()
            if self.free_gb() < self.c.min_free_gb:
                self.state["paused"] = {"stage": sid, "at": now(),
                                        "reason": f"only {self.free_gb():.0f} GB free (< {self.c.min_free_gb:g}); free space, then resume"}
                self.save()
                self.log(f"PAUSED before {sid}: {self.state['paused']['reason']}")
                return
            self.launch(s)
        else:
            self.log(f"ATTACH {sid}: already running; waiting for it")
            self.set(sid, status="running")
        stop = threading.Event()
        if s.get("sample"):
            threading.Thread(target=self.sampler, args=(sid, stop), daemon=True).start()
        exitf = os.path.join(self.stage_dir(sid), "exit_code")
        last_sweep = 0.0
        try:
            while True:
                pid = self.st(sid).get("pid")
                alive = self.pid_alive(pid, sid) or bool(ext and self.pat_alive(ext))
                if not alive:
                    break
                if time.time() - last_sweep > self.c.sweep_every:      # AB-00 finishes a run an hour
                    last_sweep = time.time()
                    self.sweep()
                time.sleep(self.c.poll)
        finally:
            stop.set()
        rc = int(open(exitf).read().strip()) if os.path.exists(exitf) else None
        if s["kind"] == "eval" and rc is None and self.eval_done(s):
            rc = 0     # attached to a run started outside this driver (AB-00): its own state decides
        self.finish(s, rc)

    def analyze(self, s: Dict[str, Any]) -> None:
        """Score each finished run's target phase and write the comparison record. Mechanical
        only (tools/eval_analyze.py); a verdict is made by reading the record (ASS-007)."""
        sid = s["id"]
        out = os.path.join(self.c.analysis, sid)
        os.makedirs(out, exist_ok=True)
        rc_all = 0
        dirs: Dict[str, str] = {}
        for arm in {s["arm_base"], s["arm_changed"]}:
            src = self.by_id[s["base"] if arm == s["arm_base"] and s["test_type"] == "T1" else s["changed"]]["root"]
            d = os.path.join(out, arm)
            os.makedirs(d, exist_ok=True)
            dirs[arm] = d
            runs = sorted(Path(src, "runs").glob(f"{arm}_s*"))
            for r in runs:
                seed = r.name.rsplit("_s", 1)[-1]
                cmd = [self.c.py, os.path.join(self.c.code, "tools/eval_analyze.py"), "score", "--run-dir", str(r),
                       "--game", "doom", "--budget-min", "20", "--out", os.path.join(d, f"{r.name}.json")]
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=self.c.code)
                with open(os.path.join(out, "analyze.log"), "a") as f:
                    f.write(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}\n")
                if p.returncode != 0:
                    self.log(f"ANALYZE {sid}: seed {seed} of {arm} not scorable (rc={p.returncode}); see analyze.log")
                    rc_all = 1
        cmp_cmd = [self.c.py, os.path.join(self.c.code, "tools/eval_analyze.py"), "compare",
                   "--base", dirs[s["arm_base"]], "--changed", dirs[s["arm_changed"]],
                   "--test-type", s["test_type"], "--dev-baseline", "campaign-" + self.state["created"][:10],
                   "--title", s["title_text"], "--out", os.path.join(out, "record.md")]
        if s["id"] != "an_ab00" and "ab00" in self.by_id and os.path.isdir(os.path.join(self.c.analysis, "an_ab00")):
            a, b = os.path.join(self.c.analysis, "an_ab00", "baseline_a"), os.path.join(self.c.analysis, "an_ab00", "baseline_b")
            if os.path.isdir(a) and os.path.isdir(b):
                cmp_cmd += ["--aa-base", a, "--aa-changed", b]
        p = subprocess.run(cmp_cmd, capture_output=True, text=True, cwd=self.c.code)
        with open(os.path.join(out, "analyze.log"), "a") as f:
            f.write(f"$ {' '.join(cmp_cmd)}\n{p.stdout}{p.stderr}\n")
        ok = p.returncode == 0 and rc_all == 0
        self.set(sid, status="done" if ok else "failed", ended=now(), rc=p.returncode)
        self.log(f"ANALYZE {sid}: {'record written' if ok else 'incomplete (see analyze.log)'}")

    # ---- loop
    def sleep(self, seconds: float) -> bool:
        """Sleep in slices; True if a stop was requested."""
        t_end = time.time() + seconds
        while time.time() < t_end:
            if os.path.exists(os.path.join(self.c.root, "CAMPAIGN_STOP")):
                return True
            time.sleep(min(self.c.poll, max(0.0, t_end - time.time())))
        return os.path.exists(os.path.join(self.c.root, "CAMPAIGN_STOP"))

    def next_stage(self) -> Optional[Dict[str, Any]]:
        """A stage already running is re-attached BEFORE any other is considered. Otherwise, after a driver
        restart, an earlier stage in the list would start polling for a free host while the running stage's own
        runner is between phases (a gap of a minute or two with no trainer alive) and could start a second trainer
        beside it. Found 2026-10-23 within a minute of the restart that exposed it (ASS-017)."""
        runnable = [s for s in self.stages if self.classify(s) == "runnable"]
        for s in runnable:
            if self.st(s["id"])["status"] == "running":
                return s
        return runnable[0] if runnable else None

    def run(self) -> int:
        lockf = open(os.path.join(self.c.root, "campaign.lock"), "w")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.log("another campaign driver holds the lock; exiting")
            return 0
        self.log(f"RUN   campaign root {self.c.root} (code {self.state['code_hash'][:12]})")
        for sid, v in self.state["stages"].items():     # a stage left 'running' by a reboot re-enters
            if v["status"] == "running" and not self.pid_alive(v.get("pid"), sid):
                self.log(f"RECOVER {sid}: was running, process gone; it re-enters (the runner resumes its state)")
                v["status"] = "pending"
        self.save()
        self.repair_after_reboot()
        self.sweep()
        while True:
            if os.path.exists(os.path.join(self.c.root, "CAMPAIGN_STOP")):
                self.log("CAMPAIGN_STOP present: stopped between stages")
                self.write_status()
                return 0
            if self.state.get("paused"):
                self.log(f"PAUSED ({self.state['paused']['stage']}); `resume` clears it")
                self.write_status()
                return 0
            s = self.next_stage()
            if s is not None:
                self.run_stage(s)
                continue
            kinds = {self.classify(x) for x in self.stages}
            if "waiting-human" in kinds:
                self.write_status()
                self.log("IDLE  waiting for a human gate (see `status`); checking again")
                if self.sleep(self.c.idle):
                    continue
                continue
            self.log("FINISHED  nothing runnable: " + ", ".join(sorted(k for k in kinds)))
            self.write_status()
            return 0

    # ---- report
    def write_status(self) -> None:
        rows = []
        tot = 0.0
        for s in self.stages:
            cl = self.classify(s)
            tot += s["est_hours"] if cl not in OK + ("failed", "skipped") else 0.0
            why = self.block_reason(s) if cl == "blocked" else (
                f"gate `{s['gate']}`" if cl == "waiting-human" else "")
            rows.append(f"| {s['id']} | {cl} | {s['est_hours']:g} | {s['title']}{' - ' + why if why else ''} |")
        txt = (f"# Campaign status\n\nUpdated {now()}. Remaining estimate {tot:.1f} host-hours "
               f"(includes blocked stages: that is the work they carry once unblocked).\n\n"
               "| stage | state | est. h | what |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")
        if self.state.get("paused"):
            txt += f"\n**PAUSED** on `{self.state['paused']['stage']}` ({self.state['paused']['reason']}). " \
                   f"Clear with `campaign.py resume {self.state['paused']['stage']}`.\n"
        Path(self.c.root, "CAMPAIGN_STATUS.md").write_text(txt)

    def plan_text(self) -> str:
        self.write_status()
        return Path(self.c.root, "CAMPAIGN_STATUS.md").read_text()

    def resume(self, which: str) -> None:
        self.state["paused"] = None
        for sid, v in self.state["stages"].items():
            if which in ("all", sid) and v["status"] == "failed":
                v["status"], v["attempts"] = "pending", 0
                self.log(f"RESUME {sid}")
        self.save()
        self.write_status()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--code", default=None, help="frozen staged code (default <root>/code)")
    ap.add_argument("--venv", default="/workspace/devai/venv_mc")
    ap.add_argument("--ab00-root", default="/workspace/experiments/doom_ab00_20261022")
    ap.add_argument("--poll", type=float, default=POLL_S)
    ap.add_argument("--idle", type=float, default=IDLE_S)
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="pause before a stage if less is free (after pruning)")
    ap.add_argument("--sweep-every", type=float, default=600.0, help="seconds between prune/compress sweeps while waiting")
    ap.add_argument("--accept-code-change", action="store_true")
    ap.add_argument("cmd", nargs="?", default="plan", choices=["plan", "run", "status", "resume", "gate"])
    ap.add_argument("arg", nargs="?", default="all")
    a = ap.parse_args(argv)
    cfg = Cfg(a)
    camp = Campaign(cfg, accept_code=a.accept_code_change)
    if a.cmd in ("plan", "status"):
        print(camp.plan_text())
        return 0
    if a.cmd == "gate":
        Path(cfg.gates, a.arg).write_text(now() + "\n")
        print(f"gate {a.arg} opened")
        return 0
    if a.cmd == "resume":
        camp.resume(a.arg)
        return 0
    return camp.run()


if __name__ == "__main__":
    sys.exit(main())
