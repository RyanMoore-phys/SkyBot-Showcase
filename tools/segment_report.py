#!/usr/bin/env python3
"""Per-segment learning metrics from a SkyBot run log + first/last-third
comparison with bootstrap CIs. Measurement only.

NAME (2026-10-28): this was `tools/learning_report.py` from 2026-10-20 (ASS-003 analysed with it under that name), which
silently replaced the offline telemetry report of the same name written on 2026-10-08 (OPS-004, `_telemetry_ingest_smoke`). The
two are different tools. This one keeps its logic and takes the name that says what it does; the telemetry report is back at
`tools/learning_report.py` (GOV-015 9.12).

    python3 segment_report.py RUNLOG [RUNLOG_OFF] [--json out.json]
"""
import ast
import json
import math
import random
import re
import sys

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_()/;:.\-]*?)=([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
TS = re.compile(r"^(\d\d):(\d\d):(\d\d) ")


def _secs(line, day_off):
    m = TS.match(line)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + day_off


def parse(path):
    segs, cur = [], {}
    doom = {}            # stream -> list of (t, dict)
    travel = []          # (t, blocks travelled, cumulative per stream 0?)
    last_t, day_off = None, 0
    for line in open(path, errors="replace"):
        t = _secs(line, day_off)
        if t is not None:
            if last_t is not None and t < last_t - 3600:
                day_off += 86400
                t += 86400
            last_t = t
        s = line.strip()
        m = re.search(r"Policy: entropy ([\d.]+)/([\d.]+)", s)
        if m:
            cur["entropy"], cur["entropy_max_printed"] = float(m.group(1)), float(m.group(2))
            m2 = re.search(r"valid[^\d]*([\d.]+)", s)
            if m2:
                cur["entropy_max_valid"] = float(m2.group(1))
        if s.startswith("PPO trust region:") or s.startswith("PPO forensics:"):
            for k, v in KV.findall(s):
                cur["ppo." + k] = float(v)
        m = re.search(r"Action attribution: mean ([\d.eE+-]+)", s)
        if m:
            cur["attribution"] = float(m.group(1))
        m = re.search(r"Reward weights:\s+intrinsic=([\d.]+)", s)
        if m:
            cur["w_int"] = float(m.group(1))
        if s.startswith("infra/ledger:") and "ALARM" not in s:
            m = re.search(r"total=([-+\d.]+)", s)
            if m:
                cur["ledger_total"] = float(m.group(1))
            for name, pct in re.findall(r"(\w+)=(\d+)%", s):
                cur["share." + name] = int(pct) / 100.0
        if s.startswith("infra/autoscale:"):
            for k in ("fresh", "train", "gap", "z", "rr", "act_ms", "vram"):
                m = re.search(r"\b%s=([-+\d.e]+)" % k, s)
                if m:
                    cur["wm." + k] = float(m.group(1))
            cur["t"] = last_t
            segs.append(cur)
            cur = {}
        m = re.search(r"doom stream (\d+): (\{.*\})", s)
        if m and last_t is not None:
            try:
                doom.setdefault(int(m.group(1)), []).append((last_t, ast.literal_eval(m.group(2))))
            except Exception:
                pass
        m = re.search(r"over ([\d.]+) blocks travelled", s)
        if m and last_t is not None:
            travel.append((last_t, float(m.group(1))))
    return segs, doom, travel


def boot_ci(a, b, stat=lambda x: sum(x) / len(x), n=2000, seed=0):
    """CI of stat(b) - stat(a) by resampling segments."""
    rng = random.Random(seed)
    if not a or not b:
        return None
    d = []
    for _ in range(n):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        d.append(stat(rb) - stat(ra))
    d.sort()
    return stat(b) - stat(a), d[int(0.025 * n)], d[int(0.975 * n)]


def rates(doom, t_lo, t_hi):
    """Per stream-hour rates of counter increases within [t_lo, t_hi]."""
    out = {"kills": 0.0, "items": 0.0, "deaths": 0.0, "exits": 0.0, "secrets": 0.0}
    hours = 0.0
    for st, rows in doom.items():
        rows = [r for r in rows if t_lo <= r[0] <= t_hi]
        if len(rows) < 2:
            continue
        (t0, a), (t1, b) = rows[0], rows[-1]
        hours += (t1 - t0) / 3600.0
        for k in out:
            out[k] += max(0.0, float(b.get(k, 0)) - float(a.get(k, 0)))
    return {k: (v / hours if hours else None) for k, v in out.items()}, hours


def thirds(segs, key):
    v = [s[key] for s in segs if key in s and s[key] is not None and math.isfinite(s[key])]
    k = max(1, len(v) // 3)
    return v[:k], v[-k:], v


def report(path):
    segs, doom, travel = parse(path)
    keys = sorted({k for s in segs for k in s if k != "t"})
    res = {"log": path, "segments": len(segs), "metrics": {}}
    for k in keys:
        a, b, allv = thirds(segs, k)
        ci = boot_ci(a, b)
        res["metrics"][k] = {"first": sum(a) / len(a), "last": sum(b) / len(b),
                             "n": len(allv), "diff_ci95": ci}
    if segs and segs[0].get("t") is not None:
        t0, t1 = segs[0]["t"], segs[-1]["t"]
        span = (t1 - t0) / 3
        r1, h1 = rates(doom, t0, t0 + span)
        r3, h3 = rates(doom, t1 - span, t1)
        res["behaviour"] = {"first_third": r1, "last_third": r3,
                            "stream_hours": [h1, h3], "minutes": (t1 - t0) / 60}
    return res


def show(res):
    print(f"== {res['log']}: {res['segments']} segments")
    for k, m in res["metrics"].items():
        ci = m["diff_ci95"]
        cis = "" if ci is None else f"  d={ci[0]:+.4g} [{ci[1]:+.4g}, {ci[2]:+.4g}]"
        print(f"  {k:28s} first={m['first']:.4g} last={m['last']:.4g} n={m['n']}{cis}")
    b = res.get("behaviour")
    if b:
        print(f"  behaviour over {b['minutes']:.0f} min (stream-hours {b['stream_hours'][0]:.2f} / {b['stream_hours'][1]:.2f}):")
        for k in b["first_third"]:
            print(f"    {k:8s} first={b['first_third'][k]} last={b['last_third'][k]}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    skip = {argv.index("--json") + 1} if "--json" in argv and argv.index("--json") + 1 < len(argv) else set()
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i not in skip)]   # `--json out.json`: out.json is the OUTPUT, not a log
    out = [report(p) for p in args]
    for r in out:
        show(r)
    if "--json" in argv:
        json.dump(out, open(argv[argv.index("--json") + 1], "w"), indent=1, default=str)
