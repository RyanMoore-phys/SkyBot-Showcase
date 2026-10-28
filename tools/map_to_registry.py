#!/usr/bin/env python3
"""Generate the machine-readable copy of the owner's map (R1, GOV-010 section 3).

Parses `assets/ai_learning.mmd` (the owner's file, never edited here) and writes
`assets/ai_learning_map.json`: every node (id, label, group, N/A / DNE flags) and
every line with the RES-008 id it carries.

RES-008 numbering rule (reproduced, then CHECKED by tools/relation_conformance.py
against the RES-008 tables): lines are numbered per map SECTION (the `%%` comment
blocks of the edge list) in file order, with the 20 lines that touch an N/A node
SKIPPED and not counted. Lines touching DNE nodes are included.

The JSON is GENERATED: regenerate on demand, never hand-edit. It carries the
sha256 of the map it came from, so a stale copy is detectable.

    ./venv/bin/python tools/map_to_registry.py            # write the JSON
    ./venv/bin/python tools/map_to_registry.py --check    # exit 1 if stale
Built, untested (GOV-012 entry R1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMD = os.path.join(ROOT, "assets", "ai_learning.mmd")
OUT = os.path.join(ROOT, "assets", "ai_learning_map.json")

SECTION_PREFIX = {
    "Neural substrate internal": "NS",
    "Neuromodulators": "NM",
    "Systems": "SY",
    "Physiology": "PH",
    "Cognitive pipeline": "CP",
    "Memory systems": "MS",
    "Mechanisms": "ME",
    "Motivation and emotion": "MO",
    "Strategies to processes": "ST",
    "Social and environment": "SO",
    "Individual differences": "ID",
    "Outcomes": "OU",
    "Link to center": "LC",
}
_NODE = re.compile(r'^\s*([A-Z0-9_]+)\s*(?:\(\(|\[)\s*"(.*?)"\s*(?:\)\)|\])\s*$')
_SUB = re.compile(r'^\s*subgraph\s+([A-Z0-9_]+)\s*\["(.*?)"\]')
_EDGE = re.compile(
    r'^\s*([A-Z0-9_]+)\s+(?:(-->)|-\.\s*"([^"]*)"\s*\.->)\s+([A-Z0-9_]+)\s*$')
_SECT = re.compile(r'^\s*%%\s*(.+?)\s*$')


def parse(path: str = MMD) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8")
    nodes, group, in_edges = {}, "", False
    section, sect_seen, edges = None, [], []
    for ln in text.splitlines():
        m = _SUB.match(ln)
        if m:
            group = m.group(2)
            continue
        m = _SECT.match(ln)
        if m and m.group(1) in SECTION_PREFIX:
            section = m.group(1)
            sect_seen.append(section)
            in_edges = True
            continue
        m = _EDGE.match(ln)
        if m and in_edges:
            kind = "solid" if m.group(2) else "dotted"
            edges.append((section, m.group(1), m.group(4), kind,
                          m.group(3) or ""))
            continue
        m = _NODE.match(ln)
        if m:
            nid, label = m.group(1), m.group(2)
            nodes.setdefault(nid, {"id": nid, "label": label, "group": group,
                                   "na": "(N/A)" in label,
                                   "dne": "(DNE)" in label})
    missing = [s for s in SECTION_PREFIX if s not in sect_seen]
    if missing:
        raise SystemExit("map sections not found: %s" % missing)
    for _, a, b, _, _ in edges:
        for n in (a, b):
            if n not in nodes:
                raise SystemExit("edge node %s has no declaration" % n)
    counters, lines = {}, []
    for section, a, b, kind, label in edges:
        pre = SECTION_PREFIX[section]
        excl = nodes[a]["na"] or nodes[b]["na"]
        rec = {"from": a, "to": b, "kind": kind, "section": section,
               "group": pre, "excluded": bool(excl)}
        if label:
            rec["label"] = label
        if excl:
            rec["id"] = None
            rec["excluded_reason"] = "touches an N/A node"
        else:
            counters[pre] = counters.get(pre, 0) + 1
            rec["id"] = "%s-%02d" % (pre, counters[pre])
        lines.append(rec)
    return {
        "generated_by": "tools/map_to_registry.py (GENERATED - do not edit)",
        "source": "assets/ai_learning.mmd",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "counts": {"lines": len(lines),
                   "compared": sum(1 for r in lines if not r["excluded"]),
                   "excluded": sum(1 for r in lines if r["excluded"]),
                   "nodes": len(nodes)},
        "nodes": nodes,
        "lines": lines,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the JSON is missing or stale")
    ap.add_argument("--map", default=MMD)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    data = parse(a.map)
    if a.check:
        try:
            with open(a.out, "r", encoding="utf-8") as fh:
                have = json.load(fh)
        except Exception:
            print("map JSON missing or unreadable: %s" % a.out)
            return 1
        ok = have.get("source_sha256") == data["source_sha256"] and \
            have.get("lines") == data["lines"]
        print("map JSON is %s" % ("current" if ok else "STALE"))
        return 0 if ok else 1
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    c = data["counts"]
    print("wrote %s: %d lines (%d compared, %d excluded), %d nodes"
          % (os.path.relpath(a.out, ROOT), c["lines"], c["compared"],
             c["excluded"], c["nodes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
