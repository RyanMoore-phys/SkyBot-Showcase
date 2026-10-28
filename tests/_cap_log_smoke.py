"""cap_log.sh smoke (2026-10-18): growing logs are capped, MineRL's included.

THE INCIDENT THIS ENCODES

    `/workspace/devai/logs/mc_928.log` + `mc_929.log` on `main`: 6.1 GB of
    MineRL client stdout from one run (2026-10-08), read by nothing. The
    stat-name filter (scripts/patch_minerl_log_filter.py) removed that spam,
    but nothing capped these files, and only runlogs/ollama.log had a capper.
    MineRL names one log per client at launch (`mc_<port>.log`), so the
    capper takes a GLOB.

Contracts:
    A. Glob mode: every matching file over the cap is truncated and its
       newest KEEP bytes land in <file>.1; a file under the cap is untouched;
       a non-matching file is untouched.
    B. A plain path still works (the ollama.log call is unchanged).
    C. Wired: launch_skybot.sh starts the glob capper for logs/mc_*.log
       inside the Minecraft-only block (no Java client, no log, no capper).

Run: PYTHONPATH=. python tests/_cap_log_smoke.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(REPO, "scripts", "cap_log.sh")
FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def run(cwd, target, max_kb, keep):
    env = dict(os.environ, CAP_LOG_ONCE="1")
    return subprocess.run(["bash", CAP, target, str(max_kb), str(keep)],
                          cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=60)


def main():
    tmp = tempfile.mkdtemp(prefix="cap_log_smoke_")
    try:
        os.makedirs(os.path.join(tmp, "logs"))
        big = {n: os.urandom(3_000_000) for n in ("mc_928.log", "mc_929.log")}
        for n, b in big.items():
            open(os.path.join(tmp, "logs", n), "wb").write(b)
        small = os.urandom(100_000)
        open(os.path.join(tmp, "logs", "mc_277.log"), "wb").write(small)
        other = os.urandom(3_000_000)
        open(os.path.join(tmp, "logs", "other.txt"), "wb").write(other)
        print("A. glob mode")
        p = run(tmp, "logs/mc_*.log", 1000, 1000)
        check(p.returncode == 0, f"exits 0 in once mode (rc={p.returncode})")
        for n, b in big.items():
            f = os.path.join(tmp, "logs", n)
            check(os.path.getsize(f) == 0
                  and open(f + ".1", "rb").read() == b[-1000:],
                  f"{n}: truncated, newest 1000 bytes in {n}.1")
        check(open(os.path.join(tmp, "logs", "mc_277.log"), "rb").read()
              == small, "mc_277.log under the cap: untouched")
        check(open(os.path.join(tmp, "logs", "other.txt"), "rb").read()
              == other, "non-matching file: untouched")
        print("B. plain path")
        f = os.path.join(tmp, "ollama.log")
        open(f, "wb").write(os.urandom(2_000_000))
        run(tmp, "ollama.log", 1000, 500)
        check(os.path.getsize(f) == 0 and os.path.getsize(f + ".1") == 500,
              "a plain filename is capped as before")
        print("C. wired in the launcher")
        src = open(os.path.join(REPO, "scripts", "launch_skybot.sh")).read()
        i_mc = src.find('if [ "$MC_ON" = 1 ]; then\npgrep -x tailscaled')
        i_cap = src.find("cap_log.sh 'logs/mc_*.log'")
        i_end = src.find("\nfi\n", i_mc)
        check(i_mc >= 0 and i_mc < i_cap < i_end,
              "launch_skybot.sh starts the mc_*.log capper in the Minecraft block")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILS:
        print(f"[cap_log_smoke] FAILED ({len(FAILS)})")
        sys.exit(1)
    print("[cap_log_smoke] ALL PASS")


if __name__ == "__main__":
    main()
