"""CS-0: the category manifests agree with the code (GOV-013 4; ASS-012 2).

    PYTHONPATH=. python tests/_categorical_inventory_smoke.py

Not a behaviour test: it checks BOOKKEEPING, so that every later categorical test starts from an
honest inventory.  Contracts:
  I-1  every named module file exists
  I-2  every named switch path exists in configs/universal.yaml
  I-3  every relation-cluster module imports and (if it has WIRED channels) each is a registered channel
  I-4  every WIRED channel in the tree is claimed by exactly one category (no orphan control path)
  I-5  every disposition is one of the allowed words, and a non-implemented member carries a note
  I-6  the boundary table names only categories that exist
  I-7  unassigned registry channels are LISTED (CS-0 must find paths the table did not name)
  I-8  deliberate break: a manifest naming a missing file / switch / channel is reported
"""
from __future__ import annotations

import importlib
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorical_lib import Harness, Obs  # noqa: E402
from categorical_manifest import BOUNDARIES, CATEGORIES, DISPOSITIONS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = Harness("inventory", "CS-0 category inventory against the code")


def switch_exists(cfg: dict, path: str) -> bool:
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    return True


def check_manifest(cats: dict, cfg: dict, wired_in_tree: dict) -> list:
    """Returns the list of problems (empty = consistent)."""
    problems = []
    claimed = {}
    for cid, cat in cats.items():
        for name, disp, files, switches, note in cat["blocks"]:
            if disp not in DISPOSITIONS:
                problems.append(f"{cid}/{name}: disposition {disp!r}")
            if disp not in ("implemented", "existing") and not note:
                problems.append(f"{cid}/{name}: {disp} without a note")
            for f in files:
                if not os.path.exists(os.path.join(ROOT, f)):
                    problems.append(f"{cid}/{name}: missing file {f}")
            for s in switches:
                if not switch_exists(cfg, s):
                    problems.append(f"{cid}/{name}: switch {s} not in configs/universal.yaml")
        for rid, disp, mod, note in cat["relations"]:
            if disp not in DISPOSITIONS:
                problems.append(f"{cid}/{rid}: disposition {disp!r}")
            if disp not in ("implemented", "existing") and not note:
                problems.append(f"{cid}/{rid}: {disp} without a note")
            if mod is None:
                continue
            if not os.path.exists(os.path.join(ROOT, "developmental_ai/core", mod + ".py")):
                problems.append(f"{cid}/{rid}: missing module {mod}")
                continue
            for ch in wired_in_tree.get(mod, ()):
                claimed.setdefault(ch, []).append(cid)
    for mod, chans in wired_in_tree.items():
        for ch in chans:
            owners = {c for c in claimed.get(ch, [])}
            if len(owners) != 1:
                problems.append(f"channel {ch} (module {mod}) claimed by {sorted(owners) or 'no category'}")
    return problems


def wired_by_module() -> dict:
    out = {}
    for fn in sorted(os.listdir(os.path.join(ROOT, "developmental_ai/core"))):
        if fn.startswith("relations_") and fn.endswith(".py"):
            mod = importlib.import_module("developmental_ai.core." + fn[:-3])
            w = getattr(mod, "WIRED", None)
            out[fn[:-3]] = tuple(w) if isinstance(w, tuple) else ()
    return out


@H.scenario("CT-01", "manifest vs code: files, switches, wired channels, ownership", min_exercised=20,
            brk="manifest_names_missing_file_switch_and_orphan_channel",
            oracle="no problem reported for the real manifest; every planted defect reported")
def inventory(brk):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    wired = wired_by_module()
    cats = {k: dict(v) for k, v in CATEGORIES.items()}
    if brk:
        cats["CAT-01"] = dict(cats["CAT-01"], blocks=cats["CAT-01"]["blocks"] + [
            ("PLANTED", "implemented", ["developmental_ai/core/does_not_exist.py"], ["no.such.switch"], "")])
        wired = dict(wired, relations_modulation=wired["relations_modulation"] + ("rel.ORPHAN.CHAN",))
    problems = check_manifest(cats, cfg, wired)
    n = sum(len(c["blocks"]) + len(c["relations"]) for c in cats.values())
    return Obs(ok=not problems, exercised=n, note="; ".join(problems[:4]) or f"{n} members consistent")


@H.scenario("CT-03", "every wired channel is a registered, switchable channel", min_exercised=10,
            brk="channel_name_not_registered",
            oracle="each WIRED name resolves in the default registry and its switch is OFF by default")
def registered(brk):
    from developmental_ai.core.relations import build_default_registry
    reg = build_default_registry({"enabled": False})
    names = [c for chans in wired_by_module().values() for c in chans]
    if brk:
        names = names + ["rel.NOT.REAL"]
    bad = [n for n in names if reg.get(n) is None]
    on = [n for n in names if reg.get(n) is not None and reg.is_on(n)]
    return Obs(ok=not bad and not on, exercised=len(names), note=f"unregistered {bad[:3]} on-by-default {on[:3]}")


@H.scenario("CT-08", "dispositions and the boundary table are well formed", min_exercised=7,
            brk="boundary_names_unknown_category", oracle="every boundary joins existing categories")
def boundaries(brk):
    b = dict(BOUNDARIES)
    if brk:
        b["BND-99"] = ("CAT-42", "CAT-01", "planted")
    bad = [k for k, (a, c, _) in b.items() if (a not in CATEGORIES and a != "ALL") or (c not in CATEGORIES and c != "ALL")]
    return Obs(ok=not bad, exercised=len(b), note=f"bad {bad}")


def report_unassigned():
    """CS-0 duty: list implemented relation modules whose channels no category names (information)."""
    from developmental_ai.core.relations import build_default_registry
    reg = build_default_registry({"enabled": False})
    wired = {c for chans in wired_by_module().values() for c in chans}
    other = [c.name for c in reg.channels() if c.name not in wired and getattr(c, "ctype", "") != "descriptive"]
    print(f"  info: {len(other)} observation/other channels carry no wired role "
          f"(RW-15 and the Stage-A clusters); {len(wired)} wired channels are assigned")


if __name__ == "__main__":
    code = H.finish()
    report_unassigned()
    sys.exit(code)
