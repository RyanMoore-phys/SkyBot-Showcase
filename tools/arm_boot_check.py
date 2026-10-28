#!/usr/bin/env python3
"""Boot every campaign arm for a few minutes and report which ones crash (ASS-013).

    python tools/arm_boot_check.py --exp DIR --code DIR --venv VENV --overlays DIR \
        --names wp3_curriculum,rw2_attention_connected --minutes 4 --out arm_boot_checks.json

Why: the T1/T2 overlays were written in the testing phase and none has ever been booted. An arm that
raises on startup would otherwise burn a 5-hour A/B slot (the crash rule re-runs it once per seed). Each
arm gets one short source run through tools/eval_run.sh (same code path as the real runs), then its logs
are scanned for a traceback. Writes {arm: {ok, reason}}; the campaign blocks an arm whose check failed.
This proves only that the arm BOOTS and runs a few minutes without a traceback; it says nothing about
whether the switched-on module does anything (that is the dead-wire telemetry and the A/B).
Runs serially on the host; never run it beside another trainer.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys


def scan(results_dir: str) -> str:
    for p in sorted(glob.glob(os.path.join(results_dir, "minecraft_skybot_run.log*"))):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", errors="replace") as f:
            for line in f:
                if "Traceback (most recent call last)" in line:
                    return f"traceback in {os.path.basename(p)}"
    return ""


def harvest(runlogs: str, out: str, tail: int = 80) -> None:
    """Keep what explains a failed boot before the 10 GB root is deleted (GOV-016 11g: the first PC boot check failed with
    'agent never came up' and the root was gone, so the cause was unreadable). Writes the supervisor log's tail, the run log's
    tail, the first traceback found and the crash history to `out`; silent when there is nothing to keep."""
    parts = []
    names = ["supervisor.log"] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(runlogs, "minecraft_skybot_run.log*")))]
    for n in names + ["crash_history", "CRASHLOOP"]:
        p = os.path.join(runlogs, n)
        if not os.path.isfile(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        try:
            with op(p, "rt", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        parts.append(f"===== {n} (last {min(tail, len(lines))} of {len(lines)} lines)")
        parts += lines[-tail:]
        if n != "supervisor.log" and "crash" not in n.lower():
            for i, line in enumerate(lines):
                if "Traceback (most recent call last)" in line:
                    parts.append(f"===== first traceback in {n} (line {i + 1})")
                    parts += lines[i:i + 40]
                    break
    if parts:
        with open(out, "w") as f:
            f.write("\n".join(parts) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--overlays", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any named arm failed (default exit 0: the result file is what blocks an arm). Used by the "
                         "single-arm WP-20 check so a failure is visible in the stage's own status (GOV-016 11g).")
    a = ap.parse_args(argv)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    env = dict(os.environ, EVAL_EXP=a.exp, EVAL_CODE=a.code, EVAL_VENV=a.venv)
    for name in [n for n in a.names.split(",") if n]:
        ov = os.path.join(a.overlays, f"{name}.yaml")
        if not os.path.exists(ov):
            res[name] = {"ok": False, "reason": "overlay file missing"}
            continue
        rid = f"bootchk_{name}"
        rc = subprocess.call(["bash", os.path.join(a.code, "tools/eval_run.sh"), "source", rid, "7", ov, "doom",
                              f"{a.minutes:g}"], env=env, cwd=a.code,
                             stdout=open(os.path.join(a.exp, f"{rid}.out"), "ab"), stderr=subprocess.STDOUT)
        why = scan(os.path.join(a.exp, "runs", rid, "results", "source"))
        res[name] = {"ok": rc == 0 and not why, "reason": why or (f"exit code {rc}" if rc else "")}
        print(f"{name}: {'ok' if res[name]['ok'] else 'FAIL ' + res[name]['reason']}", flush=True)
        harvest(os.path.join(a.exp, "runs", rid, "devai", "runlogs"), os.path.join(a.exp, f"{rid}.diag.txt"))
        for sub in ("devai", os.path.join("snap_source_end", "logs")):     # keep results, drop the 10 GB
            p = os.path.join(a.exp, "runs", rid, sub)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
        json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    if a.strict and any(not res.get(n, {}).get("ok") for n in a.names.split(",") if n):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
