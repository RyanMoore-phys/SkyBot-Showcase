"""The two report tools stay two tools (GOV-015 section 9.12).

Born 2026-10-28. `tools/learning_report.py` was two different programs: the offline telemetry report written on 2026-10-08 (OPS-004;
`--learning --positions --out --png`; `_telemetry_ingest_smoke` check 8) and, from 2026-10-20, a per-segment run-log analysis used
for ASS-003 (`RUNLOG [--json out]`) that silently replaced it. The telemetry suite then failed on every run since, "red on a pristine
HEAD", and nothing said why. The per-segment tool is now `tools/segment_report.py`; the telemetry report is back under its name.

Contracts:
    A. THE NAMES SAY WHAT THEY DO: learning_report.py documents `--learning`, `--positions`, `--out` and `--png` and writes report.md;
       segment_report.py documents `RUNLOG` and `--json`; neither docstring describes the other's interface.
    B. THE TELEMETRY CLI ACCEPTS WHAT THE TELEMETRY SUITE PASSES: `--help` exits 0 and lists every option the suite uses.
    C. THE PER-SEGMENT TOOL WORKS ON A LOG AND ITS `--json` OUTPUT PATH IS NOT READ AS A LOG (it was: `--json out.json` raised
       FileNotFoundError on out.json): a synthetic six-segment log gives the metrics and a JSON file; falsifier: without segments there
       is nothing to report (the tool does not invent numbers).
    D. NO OTHER PROGRAM CLAIMS THE OLD NAME: no file under tools/ or scripts/ calls learning_report.py with a bare RUNLOG argument.

Run: PYTHONPATH=. python tests/_report_tools_smoke.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "report-tools"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def run(*argv, **kw):
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, timeout=120, cwd=ROOT, **kw)


def main():
    tel = open(os.path.join(ROOT, "tools/learning_report.py")).read()
    seg = open(os.path.join(ROOT, "tools/segment_report.py")).read()
    d_tel, d_seg = tel.split('"""')[1], seg.split('"""')[1]
    check(all(o in d_tel for o in ("--learning", "--positions", "--out", "--png", "report.md")) and "RUNLOG" not in d_tel,
          "A. learning_report.py documents the telemetry interface and not the per-segment one")
    check("RUNLOG" in d_seg and "--json" in d_seg and "--positions" not in d_seg and "segment_report.py" in d_seg,
          "A. segment_report.py documents RUNLOG and --json under its own name")

    r = run("tools/learning_report.py", "--help")
    check(r.returncode == 0 and all(o in r.stdout for o in ("--learning", "--positions", "--out", "--png")),
          "B. learning_report.py --help exits 0 and lists --learning --positions --out --png")
    suite = open(os.path.join(ROOT, "tests/_telemetry_ingest_smoke.py")).read()
    used = set(re.findall(r'"(--[a-z-]+)"', suite[suite.index("def check_report"):]))
    check({"--learning", "--positions", "--out", "--png"} <= used and all(u in r.stdout for u in used),
          "B. every option the telemetry suite passes (%s) is accepted by the tool" % sorted(used))

    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "run.log")
        lines = []
        for k in range(6):
            # the tool reads the timestamp from a line that has one and the autoscale record from a line that STARTS with it
            lines.append(f"12:00:{k * 10:02d} Policy: entropy {2.7 - 0.05 * k:.2f}/2.77")
            lines.append(f"  infra/autoscale: fresh={0.02 - 0.001 * k:.4f} train=0.010 gap=1.0 z=20 rr=1 act_ms=200 vram=0.4")
        open(log, "w").write("\n".join(lines) + "\n")
        out = os.path.join(d, "out.json")
        r = run("tools/segment_report.py", log, "--json", out)
        ok = r.returncode == 0 and os.path.exists(out)
        data = json.load(open(out)) if ok else []
        check(ok and data and data[0]["segments"] == 6 and "entropy" in data[0]["metrics"],
              "C. the per-segment tool reports 6 segments and writes the --json file (rc %d)" % r.returncode)
        check(data[0]["metrics"]["entropy"]["first"] > data[0]["metrics"]["entropy"]["last"],
              "C. its first-third and last-third means follow the data (entropy falls)")
        empty = os.path.join(d, "empty.log")
        open(empty, "w").write("12:00:00 nothing to see\n")
        r = run("tools/segment_report.py", empty)
        check(r.returncode == 0 and "0 segments" in r.stdout and "entropy" not in r.stdout,
              "C. falsifier: a log with no segments reports none and invents no metric")

    bad = []
    for sub in ("tools", "scripts"):
        for fn in os.listdir(os.path.join(ROOT, sub)):
            p = os.path.join(ROOT, sub, fn)
            if os.path.isfile(p) and fn not in ("learning_report.py", "segment_report.py"):
                try:
                    t = open(p).read()
                except UnicodeDecodeError:
                    continue
                if re.search(r"learning_report\.py\s+[^-\s][^\s]*\.log", t) or re.search(r"learning_report\.py\s+RUNLOG", t):
                    bad.append(fn)
    check(not bad, "D. no tool or script calls learning_report.py with a bare run log (%s)" % bad)
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
