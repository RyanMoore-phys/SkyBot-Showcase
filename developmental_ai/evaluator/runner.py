"""Resumable, seeded run state machine (ASS-007 12.5, 14.6). Evaluator side. Pure Python.

The state lives in ONE json file, written atomically after every change, so a
driver that dies (or a host that reboots) resumes exactly where it was:

    pending -> (phase done ...) -> done
    a crash re-runs the WHOLE seed once with the same seed; a second crash makes
    the run `failed_final` (a missing seed means no claim, section 5 / 14.6).

The CRASH RULE is data, not code paths: `attempts` counts how many times the run
started from its first phase. A phase that was marked started but never finished
when the file is loaded is a crash (the driver was killed or the host rebooted).

FROZEN CONFIGURATION: `freeze` records a hash of every file the comparison
depends on (overlay files, the games files); a resume that finds a different
hash REFUSES (the configuration is frozen for the whole comparison, section 6).
That refusal is released by deleting the state file on purpose (a new comparison).

Nothing here starts a process; tools/eval_runner.py does, through tools/eval_run.sh.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from developmental_ai.evaluator.protocol import RunSpec

MAX_ATTEMPTS = 2        # the first run plus ONE re-run with the same seed
TERMINAL = ("done", "failed_final")


class FrozenConfigChanged(RuntimeError):
    pass


def file_hash(paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        out[p] = h.hexdigest()
    return out


class EvalState:
    def __init__(self, path: str, data: Dict[str, Any]):
        self.path = path
        self.data = data

    # ------------------------------------------------------------ creation
    @classmethod
    def load_or_create(cls, path: str, plan: Sequence[RunSpec],
                       frozen: Optional[Dict[str, str]] = None) -> "EvalState":
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            if frozen is not None and data.get("frozen") != frozen:
                raise FrozenConfigChanged(
                    "frozen files differ from the recorded comparison; delete "
                    f"{path} deliberately to start a new comparison")
            st = cls(path, data)
            st.recover()
            return st
        runs = {r.run_id: {"arm": r.arm, "seed": r.seed,
                           "phases": [[p.name, p.minutes] for p in r.phases],
                           "status": "pending", "attempts": 1,
                           "phase_done": [], "in_progress": None,
                           "crashes": [], "info": {}} for r in plan}
        st = cls(path, {"frozen": frozen or {}, "order": [r.run_id for r in plan],
                        "runs": runs})
        st.save()
        return st

    def save(self) -> None:
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    # --------------------------------------------------------------- queries
    def run(self, run_id: str) -> Dict[str, Any]:
        return self.data["runs"][run_id]

    def next_task(self) -> Optional[Tuple[str, str, float]]:
        """(run_id, phase, minutes) of the first unfinished phase, in plan order."""
        for rid in self.data["order"]:
            r = self.run(rid)
            if r["status"] in TERMINAL:
                continue
            for name, minutes in r["phases"]:
                if name not in r["phase_done"]:
                    return rid, name, float(minutes)
        return None

    def crash_counts_by_arm(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.data["runs"].values():
            out[r["arm"]] = out.get(r["arm"], 0) + len(r["crashes"])
        return out

    def missing_seeds(self, arm: str) -> List[int]:
        return sorted(r["seed"] for r in self.data["runs"].values()
                      if r["arm"] == arm and r["status"] != "done")

    # ------------------------------------------------------------ transitions
    def mark_started(self, run_id: str, phase: str) -> None:
        r = self.run(run_id)
        r["in_progress"] = {"phase": phase, "t": time.time()}
        self.save()

    def mark_phase_done(self, run_id: str, phase: str, info: Optional[Dict] = None) -> None:
        r = self.run(run_id)
        r["in_progress"] = None
        if phase not in r["phase_done"]:
            r["phase_done"].append(phase)
        if info:
            r["info"][phase] = info
        if all(n in r["phase_done"] for n, _ in r["phases"]):
            r["status"] = "done"
        self.save()

    def mark_crash(self, run_id: str, phase: str, reason: str) -> str:
        """Apply the crash rule; returns the run's new status."""
        r = self.run(run_id)
        r["in_progress"] = None
        r["crashes"].append({"phase": phase, "reason": str(reason)[:300], "t": time.time()})
        if r["attempts"] >= MAX_ATTEMPTS:
            r["status"] = "failed_final"
        else:
            r["attempts"] += 1
            r["phase_done"] = []          # the whole seed again, same seed
            r["status"] = "pending"
        self.save()
        return r["status"]

    def recover(self) -> List[str]:
        """A phase left in progress by a dead driver is a crash."""
        hit = []
        for rid, r in self.data["runs"].items():
            if r.get("in_progress"):
                self.mark_crash(rid, r["in_progress"]["phase"], "driver died mid-phase")
                hit.append(rid)
        return hit
