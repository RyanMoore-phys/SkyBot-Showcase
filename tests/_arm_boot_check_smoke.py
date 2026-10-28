"""The arm boot check keeps what explains a failure and can say so (GOV-016 section 11g).

Born 2026-10-28. The first boot check of the predictive-coding arm printed `wp20_pc_core: FAIL exit code 3`, the stage still
ended with exit 0 (it reads as done), and the 10 GB root it had booted was deleted, so the cause ("agent never came up") was
unreadable. This was found only because the A/B stayed blocked. No model, no game: the tool is run against a fake launcher.

Contracts:
    A. A FAILED BOOT LEAVES ITS EVIDENCE: a launcher that exits 3 after writing a supervisor log and a run log with a traceback
       leaves `<rid>.diag.txt` holding the supervisor tail, the run-log tail and the first traceback, and the root is still removed.
    B. THE RECORD IS RIGHT: the result file says ok False with the reason, and a passing launcher says ok True.
    C. STRICT: with --strict a failing arm gives exit 1 and a passing arm 0; without it a failing arm still exits 0 (the result file is
       what blocks an arm; the multi-arm check must not become a stage failure that pauses the campaign).
    D. NOTHING TO KEEP IS NOT AN ERROR: a launcher that leaves no logs gives no diag file and no exception.
    E. A TRACEBACK IN AN OTHERWISE CLEAN RUN STILL FAILS THE ARM (the old rule is kept).

Run: PYTHONPATH=. python tests/_arm_boot_check_smoke.py
"""
import json
import os
import stat
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import arm_boot_check as ABC    # noqa: E402

NAME = "arm-boot-check"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


LAUNCHER = """#!/bin/bash
# fake tools/eval_run.sh: <phase> <run_id> <seed> <overlay> <games> <minutes>
RID=$2
M="$EVAL_EXP/runs/$RID/devai"; mkdir -p "$M/runlogs" "$EVAL_EXP/runs/$RID/results/source"
case "$FAKE" in
  never)  printf 'start\\nsupervisor: launching run.py\\nsupervisor: run exited 1 (crash 1)\\n' > "$M/runlogs/supervisor.log"
          printf 'boot\\nTraceback (most recent call last):\\n  File "x.py", line 1, in <module>\\nKeyError: core_mismatch\\n' > "$M/runlogs/minecraft_skybot_run.log"
          exit 3 ;;
  ok)     printf 'fine\\n' > "$M/runlogs/supervisor.log"; printf 'segment 1\\n' > "$M/runlogs/minecraft_skybot_run.log"; exit 0 ;;
  silent) exit 3 ;;
  trace)  printf 'fine\\n' > "$M/runlogs/supervisor.log"
          printf 'segment 1\\nTraceback (most recent call last):\\nValueError: late\\n' > "$EVAL_EXP/runs/$RID/results/source/minecraft_skybot_run.log"
          exit 0 ;;
esac
"""


def run(mode, strict=False):
    with tempfile.TemporaryDirectory() as d:
        code = os.path.join(d, "code")
        os.makedirs(os.path.join(code, "tools"))
        lp = os.path.join(code, "tools", "eval_run.sh")
        open(lp, "w").write(LAUNCHER)
        os.chmod(lp, os.stat(lp).st_mode | stat.S_IEXEC)
        ov = os.path.join(d, "ov")
        os.makedirs(ov)
        open(os.path.join(ov, "arm.yaml"), "w").write("a: 1\n")
        exp = os.path.join(d, "exp")
        os.makedirs(exp)
        out = os.path.join(d, "res.json")
        os.environ["FAKE"] = mode
        argv = ["--exp", exp, "--code", code, "--venv", "/v", "--overlays", ov, "--names", "arm", "--minutes", "0.01", "--out", out]
        rc = ABC.main(argv + (["--strict"] if strict else []))
        diag = os.path.join(exp, "bootchk_arm.diag.txt")
        return {"rc": rc, "res": json.load(open(out)), "diag": open(diag).read() if os.path.exists(diag) else None,
                "root_gone": not os.path.exists(os.path.join(exp, "runs", "bootchk_arm", "devai"))}


def main():
    r = run("never")
    check(r["diag"] is not None and "supervisor: run exited 1" in r["diag"] and "KeyError: core_mismatch" in r["diag"]
          and "first traceback" in r["diag"], "A. a failed boot leaves its supervisor tail, run-log tail and first traceback")
    check(r["root_gone"], "A. the root is still removed after the evidence is kept")
    check(r["res"]["arm"]["ok"] is False and "exit code 3" in r["res"]["arm"]["reason"],
          "B. the result file says the arm failed, with a reason (%s)" % r["res"]["arm"]["reason"])
    ok = run("ok")
    check(ok["res"]["arm"]["ok"] is True and ok["rc"] == 0, "B. a passing launcher is recorded ok")
    check(run("never", strict=True)["rc"] == 1 and run("ok", strict=True)["rc"] == 0,
          "C. --strict: a failing arm exits 1 and a passing arm 0")
    check(run("never", strict=False)["rc"] == 0, "C. without --strict a failing arm still exits 0 (the multi-arm stage must not pause the campaign)")
    s = run("silent")
    check(s["diag"] is None and s["res"]["arm"]["ok"] is False, "D. no logs to keep: no diag file, no exception, the arm still fails")
    t = run("trace")
    check(t["res"]["arm"]["ok"] is False and "traceback" in t["res"]["arm"]["reason"],
          "E. a traceback in an otherwise clean run still fails the arm")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
