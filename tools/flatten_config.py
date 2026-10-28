"""Flatten the universal configuration into ONE self-contained file, with a parity report.

WP-15a (GOV-009 / ADR-004 Amendment 2), deviation D-06: the universal config is
composed ON TOP of configs/minecraft_skybot.yaml (the legacy base). Removing the
base means writing the merged result as one file. That step is NOT done at
build time: it cannot be parity-checked without running, and a hand-flatten would
lose the base file's incident comments. This tool does it LATER, mechanically:

    python tools/flatten_config.py --out configs/universal_flat.yaml \
        --report runlogs/flatten_parity.txt

What it does (nothing here runs a game, a model or a network call):
  1. flat = deep_merge(base, universal), exactly the first step of
     tools/compose_run_config.compose(). Games are NOT applied: per-game overlays
     stay in configs/games/*.yaml and the composer keeps applying them.
  2. Writes `flat` to --out (comments are NOT preserved by a YAML round trip; the
     header says the base file remains the incident record until it is removed).
  3. PARITY: for each game set (every configs/games/<g>.yaml alone, plus all
     together; or --games a,b), composes once with the original (base, universal)
     and once with (empty base, flat), and diffs the two composed configs leaf by
     leaf. The report lists every differing key path. Parity = zero differences.
     Exit code 0 only when every game set has zero differences.

The tool changes no file other than --out and --report. After the owner accepts
the report, the switch is: point the composer's default `--base` at an empty
mapping / drop the base and `--universal` at the flat file, then delete the legacy
base (a separate, reviewed step, ADR-004). Not wired to anything.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _compose_module():
    import compose_run_config as c  # lazy: only when the tool runs
    return c


def leaves(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings to {dotted.path: value}; lists are leaves."""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out[prefix] = {}
        for k, v in obj.items():
            out.update(leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def diff_configs(a: Dict, b: Dict) -> List[Tuple[str, Any, Any]]:
    la, lb = leaves(a), leaves(b)
    missing = object()
    rows = []
    for k in sorted(set(la) | set(lb)):
        va, vb = la.get(k, missing), lb.get(k, missing)
        if va is missing or vb is missing or va != vb:
            rows.append((k, "<absent>" if va is missing else va,
                         "<absent>" if vb is missing else vb))
    return rows


def build_flat(base_path: str, universal_path: str) -> Dict[str, Any]:
    c = _compose_module()
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    with open(universal_path) as f:
        uni = yaml.safe_load(f) or {}
    return c.deep_merge(base, uni)


def parity(flat: Dict, base_path: str, universal_path: str, games_dir: str,
           game_sets: List[List[str]]) -> List[Tuple[List[str], List[Tuple[str, Any, Any]]]]:
    c = _compose_module()
    results = []
    with tempfile.TemporaryDirectory(prefix="flatten_parity_") as d:
        empty = os.path.join(d, "empty_base.yaml")
        flat_p = os.path.join(d, "flat.yaml")
        with open(empty, "w") as f:
            f.write("{}\n")
        with open(flat_p, "w") as f:
            yaml.safe_dump(flat, f, sort_keys=False, default_flow_style=False)
        for games in game_sets:
            orig = c.compose(games, base_path, universal_path, games_dir)
            new = c.compose(games, empty, flat_p, games_dir)
            for cfg in (orig, new):  # a scale-seed marker is launch bookkeeping
                (cfg.get("universal") or {}).pop("_scale_seeded", None)
            results.append((games, diff_configs(orig, new)))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=os.path.join(ROOT, "configs/minecraft_skybot.yaml"))
    ap.add_argument("--universal", default=os.path.join(ROOT, "configs/universal.yaml"))
    ap.add_argument("--games-dir", default=os.path.join(ROOT, "configs/games"))
    ap.add_argument("--games", help="comma-separated games for ONE joint parity set "
                                    "(default: each game alone, then all together)")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs/universal_flat.yaml"))
    ap.add_argument("--report", default=None, help="write the parity report here too")
    a = ap.parse_args(argv)

    flat = build_flat(a.base, a.universal)
    if a.games:
        sets = [[g.strip().lower() for g in a.games.split(",") if g.strip()]]
    else:
        names = sorted(f[:-5] for f in os.listdir(a.games_dir) if f.endswith(".yaml"))
        sets = [[n] for n in names] + ([names] if len(names) > 1 else [])
    results = parity(flat, a.base, a.universal, a.games_dir, sets)

    lines = ["FLATTEN PARITY REPORT", f"base={a.base}", f"universal={a.universal}",
             f"flat={a.out}", ""]
    total = 0
    for games, rows in results:
        total += len(rows)
        lines.append(f"games={','.join(games)}: {len(rows)} differing key(s)")
        for k, va, vb in rows[:200]:
            lines.append(f"    {k}: original={va!r} flat={vb!r}")
    lines.append("")
    lines.append("PARITY: " + ("EXACT (0 differences in every game set)" if total == 0
                              else f"NOT EXACT ({total} differences) -- do not drop the base"))
    report = "\n".join(lines)
    print(report)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            f.write(report + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# GENERATED by tools/flatten_config.py: base + universal merged, games NOT applied.\n"
                "# Comments are not preserved; configs/minecraft_skybot.yaml stays the incident\n"
                "# record until the owner removes the base (ADR-004). Compose per-game overlays on top.\n")
        yaml.safe_dump(flat, f, sort_keys=False, default_flow_style=False)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
