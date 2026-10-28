#!/usr/bin/env python3
"""Live check that a diverse source phase really visits many maps (GOV-014 item C; ASS-013).

    python tools/map_coverage_check.py --exp DIR --code DIR --venv VENV --overlay wp6_coverage_probe.yaml \
        --minutes 6 --min-maps 12 --out arm_boot_checks.json --key wp6_coverage

Why: every AB-00 source phase stayed on MAP01 although MAP01..MAP28 were listed (ASS-015). The unit tests
prove the overlay merges and composes; only the real adapter proves the maps actually CHANGE at episode
starts. This runs the probe overlay (the same rotation on a 20 s clock) for a few minutes through
tools/eval_run.sh, reads the `map` field of the evaluator lines afterwards (the learner never sees it,
ADR-002), and records pass when at least `--min-maps` distinct maps were seen and none is held out.
Writes {key: {ok, reason, maps}} into the same JSON the arm boot checks use, so the campaign blocks
`t1_wp6_diversity` when it fails. Runs serially on the host; never beside another trainer.
"""
from __future__ import annotations

import argparse
import ast
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_arms as arms  # noqa: E402

EV = re.compile(r"doom stream (\d+): (\{.*\})\s*$")


def maps_seen(results_dir: str) -> dict:
    seen: dict = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "minecraft_skybot_run.log*"))):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", errors="replace") as f:
            for line in f:
                m = EV.search(line)
                if not m:
                    continue
                try:
                    d = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    continue
                seen.setdefault(d.get("map"), set()).add(int(m.group(1)))
    return {k: sorted(v) for k, v in seen.items() if k}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--min-maps", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="wp6_coverage")
    a = ap.parse_args(argv)
    os.makedirs(a.exp, exist_ok=True)
    env = dict(os.environ, EVAL_EXP=a.exp, EVAL_CODE=a.code, EVAL_VENV=a.venv)
    rid = "wp6cov"
    rc = subprocess.call(["bash", os.path.join(a.code, "tools/eval_run.sh"), "source", rid, "7", a.overlay, "doom",
                          f"{a.minutes:g}"], env=env, cwd=a.code,
                         stdout=open(os.path.join(a.exp, f"{rid}.out"), "ab"), stderr=subprocess.STDOUT)
    seen = maps_seen(os.path.join(a.exp, "runs", rid, "results", "source"))
    leaked = sorted(set(seen) & set(arms.HELD_OUT))
    if rc != 0:
        res = {"ok": False, "reason": f"run exit code {rc}", "maps": sorted(seen)}
    elif leaked:
        res = {"ok": False, "reason": f"held-out map(s) {leaked} seen in a source phase", "maps": sorted(seen)}
    elif len(seen) < a.min_maps:
        res = {"ok": False, "reason": f"only {len(seen)} distinct maps seen (< {a.min_maps})", "maps": sorted(seen)}
    else:
        res = {"ok": True, "reason": "", "maps": sorted(seen)}
    data = json.load(open(a.out)) if os.path.exists(a.out) else {}
    data[a.key] = res
    json.dump(data, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"{a.key}: {'ok' if res['ok'] else 'FAIL ' + res['reason']} ({len(seen)} maps: {sorted(seen)})", flush=True)
    for sub in ("devai", os.path.join("snap_source_end", "logs")):         # keep results, drop the 10 GB
        p = os.path.join(a.exp, "runs", rid, sub)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
