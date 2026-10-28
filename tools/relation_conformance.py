#!/usr/bin/env python3
"""Conformance REPORT (R1, GOV-010 section 3): which lines of the owner's map have
a software channel. Built, untested (GOV-012 entry R1).

    ./venv/bin/python tools/relation_conformance.py                 # print
    ./venv/bin/python tools/relation_conformance.py --out FILE      # write
    ./venv/bin/python tools/relation_conformance.py --strict        # exit 1 on drift

WHAT IT REPORTS: IMPLEMENTATION COVERAGE ONLY - whether a line has a channel in
the registry, line by line, next to its RES-008 status. It is NOT evidence that a
line's function equals the human one, nor that a channel is alive (that is the
runtime dead-wire alarm) or beneficial (that is T1/T2). It checks BOOKKEEPING:

  1. map JSON is current against assets/ai_learning.mmd (regenerate with
     tools/map_to_registry.py);
  2. the ids the generator derives agree with the RES-008 tables (same ids, same
     from/to nodes) - disagreement is "ID DRIFT" and the report says so;
  3. every registered channel names a real map line with the SAME producer and
     consumer nodes;
  4. every line RES-008 calls Connected has a channel (RW-15).
Descriptive lines (central-node links and dotted lines) are RECORDED, never wired.
N/A-excluded lines are listed as excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
RES008 = os.path.join(ROOT, "docs", "research",
                      "res-008-relationship-gap-analysis.md")
MAPJSON = os.path.join(ROOT, "assets", "ai_learning_map.json")

_ROW = re.compile(
    r"^\|\s*([A-Z]{2}-\d{2})\s*\|\s*(.*?)\[([A-Z0-9_]+)\]\s*\|\s*(.*?)"
    r"\[([A-Z0-9_]+)\]\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|")


def read_res008(path: str = RES008) -> dict:
    rows = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            m = _ROW.match(ln)
            if m:
                rows[m.group(1)] = {"from": m.group(3), "to": m.group(5),
                                    "kind": m.group(6), "status": m.group(7)}
    return rows


def build_report(strict_out: list) -> str:
    import map_to_registry
    from developmental_ai.core.relations import (
        build_default_registry, is_descriptive_line)

    out = []
    fresh = map_to_registry.parse()
    try:
        with open(MAPJSON, "r", encoding="utf-8") as fh:
            have = json.load(fh)
        current = have.get("source_sha256") == fresh["source_sha256"] and \
            have.get("lines") == fresh["lines"]
    except Exception:
        current = False
    if not current:
        strict_out.append("map JSON missing or stale")
    res = read_res008()
    reg = build_default_registry(None)
    lines = fresh["lines"]
    compared = [r for r in lines if not r["excluded"]]

    out.append("RELATION CONFORMANCE - IMPLEMENTATION COVERAGE ONLY")
    out.append("(a channel existing is bookkeeping; it is not evidence that the "
               "line works as in a human, is alive, or helps - see T1/T2)")
    out.append("")
    out.append("map JSON current: %s | map lines %d, compared %d, excluded %d | "
               "RES-008 rows read %d | channels registered %d"
               % (current, len(lines), len(compared),
                  len(lines) - len(compared), len(res), len(reg.channels())))
    out.append("")

    drift = []
    for r in compared:
        row = res.get(r["id"])
        if row is None:
            drift.append("%s: in map, not in RES-008" % r["id"])
        elif (row["from"], row["to"]) != (r["from"], r["to"]):
            drift.append("%s: map %s->%s but RES-008 %s->%s"
                         % (r["id"], r["from"], r["to"], row["from"], row["to"]))
    mapids = {r["id"] for r in compared}
    for rid in res:
        if rid not in mapids:
            drift.append("%s: in RES-008, not derived from the map" % rid)
    out.append("ID DRIFT (map-derived id vs RES-008): %s"
               % ("none" if not drift else "%d" % len(drift)))
    out.extend("  " + d for d in drift[:40])
    if drift:
        strict_out.append("id drift: %d" % len(drift))
    out.append("")

    problems = []
    by_pair = {(r["from"], r["to"]): r for r in compared}
    for ch in reg.channels():
        r = next((x for x in compared if x["id"] == ch.line_id), None)
        if r is None:
            problems.append("%s (%s): no such compared line in the map"
                            % (ch.name, ch.line_id))
        elif (r["from"], r["to"]) != (ch.frm, ch.to):
            problems.append("%s (%s): map has %s->%s"
                            % (ch.name, ch.line_id, r["from"], r["to"]))
    out.append("CHANNEL vs MAP mismatches: %s"
               % ("none" if not problems else "%d" % len(problems)))
    out.extend("  " + p for p in problems)
    if problems:
        strict_out.append("channel/map mismatch: %d" % len(problems))
    out.append("")

    counts = {"channel": 0, "descriptive": 0, "no channel": 0}
    by_status = {}
    out.append("%-6s %-8s %-8s %-10s %-34s %s"
               % ("ID", "FROM", "TO", "TYPE", "RES-008 STATUS", "COVERAGE"))
    missing_connected = []
    for r in compared:
        ch = reg.by_line(r["id"])
        st = (res.get(r["id"]) or {}).get("status", "?")
        if ch is not None:
            cov, typ = "CHANNEL %s" % ch.name, ch.type
            counts["channel"] += 1
        elif is_descriptive_line(r):
            cov, typ = "descriptive (recorded, not wired)", "descriptive"
            counts["descriptive"] += 1
        else:
            cov, typ = "no channel", "-"
            counts["no channel"] += 1
        if st.startswith("Connected") and ch is None:
            missing_connected.append(r["id"])
        key = (st, cov.split(" ")[0] if ch else cov)
        by_status[key] = by_status.get(key, 0) + 1
        out.append("%-6s %-8s %-8s %-10s %-34s %s"
                   % (r["id"], r["from"], r["to"], typ, st[:34], cov))
    out.append("")
    out.append("EXCLUDED (touch an N/A node): %d lines" % (len(lines) - len(compared)))
    out.append("")
    out.append("SUMMARY (coverage only): channels %d | descriptive %d | no channel %d "
               "| of %d compared lines"
               % (counts["channel"], counts["descriptive"], counts["no channel"],
                  len(compared)))
    out.append("RES-008 Connected lines without a channel: %s"
               % (", ".join(missing_connected) or "none"))
    if missing_connected:
        strict_out.append("connected lines without channel: %s"
                          % ",".join(missing_connected))
    out.append("")
    out.append("By RES-008 status x coverage:")
    for (st, cov), n in sorted(by_status.items()):
        out.append("  %-34s %-34s %d" % (st[:34], cov[:34], n))
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="write the report to this file")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on stale JSON, id drift, mismatch, or a "
                         "Connected line without a channel")
    a = ap.parse_args(argv)
    bad = []
    rep = build_report(bad)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(rep)
        print("wrote %s" % a.out)
    else:
        sys.stdout.write(rep)
    if bad:
        print("NONCONFORMANT: " + "; ".join(bad), file=sys.stderr)
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
