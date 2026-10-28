"""Contracts for the diverse source phase (GOV-014 item C; ASS-015 section 1 point 4).

    PYTHONPATH=. python tests/_wp6_diversity_smoke.py

THE INCIDENT. The Doom source config lists MAP01..MAP28, but the adapter plays `maps[map_i]` and
advances only on a level exit, so all 16 AB-00 source phases stayed on MAP01 and the world model's
fresh/train gap grew 0.95 -> 1.75. WP-6's `game.map_assignment` can fix that, but the runner could only
merge an overlay into universal.yaml, so no arm could set it. The overlay now carries a reserved
`_games` key merged into the game YAML by tools/eval_run.sh.

This test executes the REAL python blocks of tools/eval_run.sh (extracted from the script) in a temp
copy of the configs, then the composer's `configure_stream` for every stream, so a mistake in the
merge, the generated rotation or the held-out guard shows up here and not in a 40-minute run.

CONTRACTS
  A  the merge: `_games` lands in configs/games/doom.yaml and is NOT left in universal.yaml
  B  an overlay without `_games` behaves exactly as before (universal only, game YAML untouched)
  C  coverage: across the 8 composed streams and the whole schedule every source map (28) is started on
  D  no held-out map is reachable in the source phase (composer accepts), and a planted leak is refused
  E  the schedule is applied by the adapter's own rule (schedule_maps at 0, 300, 2100 s)
  F  the held-out list in campaign_arms mirrors eval_run.sh
  G  default parity: the unmodified game YAML composes with no map schedule
  H  falsification: the pre-change merge leaves `_games` in universal.yaml
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import campaign_arms as A  # noqa: E402
from developmental_ai.environments.data_diversity import (  # noqa: E402
    LeakError, configure_stream, schedule_maps)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def blocks():
    """The two embedded python heredocs of build_root(), verbatim."""
    txt = open(os.path.join(ROOT, "tools/eval_run.sh")).read()
    found = re.findall(r"<<'PY'\n(.*?)\nPY\n", txt, re.S)
    held = next(b for b in found if "held_out_maps" in b)
    merge = next(b for b in found if "def merge" in b)
    return held, merge


def sandbox():
    d = tempfile.mkdtemp(prefix="wp6_")
    os.makedirs(os.path.join(d, "configs/games"))
    shutil.copy(os.path.join(ROOT, "configs/games/doom.yaml"), os.path.join(d, "configs/games/doom.yaml"))
    shutil.copy(os.path.join(ROOT, "configs/universal.yaml"), os.path.join(d, "configs/universal.yaml"))
    held, _ = blocks()
    subprocess.run([sys.executable, "-"], input=held, text=True, cwd=d, check=True)
    return d


def apply_overlay(d, overlay, merge_src=None):
    p = os.path.join(d, "ov.yaml")
    yaml.safe_dump(overlay, open(p, "w"))
    src = merge_src if merge_src is not None else blocks()[1]
    r = subprocess.run([sys.executable, "-", p], input=src, text=True, cwd=d, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])


def game(d):
    return yaml.safe_load(open(os.path.join(d, "configs/games/doom.yaml")))["game"]


def compose_streams(g, n=8, phase="source"):
    return [configure_stream(copy.deepcopy(g) | {"streams": n}, k, n, phase) for k in range(n)]


def main() -> int:
    ov = A.T1["wp6_diversity"][1]
    print("A the merge")
    d = sandbox()
    base_u = yaml.safe_load(open(os.path.join(d, "configs/universal.yaml")))
    apply_overlay(d, ov)
    g = game(d)
    check("map_assignment is in the game YAML", "map_assignment" in g and len(g["map_assignment"]["schedule"]) == 7)
    check("held-out maps are still set (merged after them)", g["held_out_maps"] == A.HELD_OUT)
    uni = yaml.safe_load(open(os.path.join(d, "configs/universal.yaml")))
    check("`_games` is NOT left in universal.yaml", "_games" not in uni)
    check("universal.yaml is otherwise unchanged", uni == base_u)

    print("B an overlay without _games is as before")
    d2 = sandbox()
    g_before = game(d2)
    apply_overlay(d2, {"curriculum": {"enabled": True}})
    check("game YAML untouched", game(d2) == g_before)
    u2 = yaml.safe_load(open(os.path.join(d2, "configs/universal.yaml")))
    check("universal merged as before", u2["curriculum"]["enabled"] is True)

    print("C coverage")
    streams = compose_streams(g)
    starts = set()
    for s in streams:
        starts |= set(s["stream_maps"])
        for ph in s.get("map_schedule", []):
            starts |= set(ph["maps"])
    src = set(A._doom_source_maps())
    check("every one of the 28 source maps is started on by some stream", starts == src, sorted(src - starts))
    check("each stream starts on a different map at time 0", len({s["stream_maps"][0] for s in streams}) == 8)

    print("D leaks")
    check("no held-out map anywhere in the composed source streams", not (starts & set(A.HELD_OUT)), sorted(starts & set(A.HELD_OUT)))
    bad = copy.deepcopy(g)
    bad["map_assignment"]["schedule"][0]["per_stream"][0] = ["MAP30"]
    try:
        compose_streams(bad)
        refused = False
    except LeakError:
        refused = True
    check("a planted held-out map in the schedule is refused (LeakError)", refused)
    tgt = compose_streams(g, 8, "target")
    check("the target phase still plays only the held-out maps", {m for s in tgt for m in s["stream_maps"]} == set(A.HELD_OUT))

    print("E the adapter's own schedule rule")
    s0 = streams[0]
    check("before the first phase the schedule gives nothing (initial list stands)", schedule_maps(s0["map_schedule"], 0.0) is None)
    check("at 300 s stream 0 switches to its second map", schedule_maps(s0["map_schedule"], 310.0) == s0["map_schedule"][0]["maps"])
    check("late in the run it is on the last phase", schedule_maps(s0["map_schedule"], 2400.0) == s0["map_schedule"][-1]["maps"])

    print("F the held-out mirror")
    held, _ = blocks()
    m = re.search(r'held_out_maps"\] = (\[[^\]]*\])', held)
    check("campaign_arms.HELD_OUT equals the list in eval_run.sh", m and yaml.safe_load(m.group(1).replace("'", '"')) == A.HELD_OUT)

    print("G default parity")
    plain = compose_streams(game(sandbox()))
    check("no schedule and a held-out-filtered map list when no WP-6 key is set",
          all("map_schedule" not in s and "stream_maps" not in s for s in plain)
          and all(not (set(s["maps"]) & set(A.HELD_OUT)) for s in plain))

    print("H falsification: the pre-change merge")
    old = ("import sys, yaml\n"
           "def merge(a, b):\n"
           "    for k, v in (b or {}).items():\n"
           "        if isinstance(v, dict) and isinstance(a.get(k), dict):\n"
           "            merge(a[k], v)\n"
           "        else:\n"
           "            a[k] = v\n"
           "c = yaml.safe_load(open('configs/universal.yaml'))\n"
           "merge(c, yaml.safe_load(open(sys.argv[1])))\n"
           "yaml.safe_dump(c, open('configs/universal.yaml', 'w'), sort_keys=False)\n")
    d3 = sandbox()
    apply_overlay(d3, ov, merge_src=old)
    check("the old merge puts `_games` into universal.yaml and never reaches the game YAML",
          "_games" in yaml.safe_load(open(os.path.join(d3, "configs/universal.yaml"))) and "map_assignment" not in game(d3))

    print(f"[wp6-diversity] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
