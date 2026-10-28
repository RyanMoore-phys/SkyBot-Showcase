"""Shared harness for the categorical system tests (CS-1 / CS-2; GOV-013, ASS-012).

Not a test itself (its name does not match `_categorical_*_smoke.py`, so the campaign's
`cat_harness` stage never runs it directly). Every `tests/_categorical_*_smoke.py` uses it.

WHAT THE HARNESS ENFORCES (ASS-012 3, 6, 7), so a test file cannot quietly weaken them:

  1. EVERY scenario runs TWICE. Once as built (`brk is None`): it must report `ok` and have
     exercised at least `min_exercised` real consumer updates. Then once with a DELIBERATE BREAK
     (`brk` = the defect's name, injected into the harness or a disposable copy, never into
     production code): it must report NOT ok, or raise. A break that goes undetected FAILS the
     scenario: a test that cannot fail proves nothing ("a matching log counter alone is
     insufficient", ASS-012 CT-01).
  2. A scenario that exercised fewer paths than required is INCONCLUSIVE and fails the file; it is
     never counted as a pass. A legitimate zero is distinguished from absence by the scenario.
  3. A scenario whose consumer or interface does not exist yet is declared with `blocked(...)`
     and a reason. It prints as BLOCKED, is listed in the verdict, and the file's verdict becomes
     "pass for declared scope" (never "pass"). A blocked scenario is not a pass and not a failure.
  4. No scenario may supply scripted skills or task meanings to a production learner, read the
     evaluator package, or touch a running trainer. Fixtures inject numbers, delays and faults
     into the REAL module objects (ASS-012 3).
  5. The result is printed as numbered contract lines and, when CATEGORICAL_OUT is set, written as
     `<CATEGORICAL_OUT>/<file id>.json` (the ASS-012 7 record: contract id, required, exercised,
     oracle, observation, verdict) for the campaign to archive.

USAGE
    from categorical_lib import Harness, Obs
    H = Harness("cat03", "Planning, uncertainty and risk")

    @H.scenario("CT-01", "valid ensemble evidence reaches the arbiter", min_exercised=3,
                brk="consumer_omitted", oracle="arbiter output flips with the disagreement signal")
    def ct01(brk):
        ...                              # build REAL modules, drive them, inspect the consumer
        if brk == "consumer_omitted":    # inject the named defect (and only then)
            ...
        return Obs(ok=cond, exercised=n, note="what was observed")

    H.blocked("CT-05", "no consumer for X exists yet (GOV-012 D-R11)")
    sys.exit(H.finish())
"""
from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

CT_IDS = ("CT-01", "CT-02", "CT-03", "CT-04", "CT-05", "CT-06", "CT-07", "CT-08")


@dataclass
class Obs:
    ok: bool
    exercised: int = 0
    note: str = ""


@dataclass
class Row:
    cid: str
    title: str
    required: bool
    oracle: str
    verdict: str = "PENDING"
    exercised: int = 0
    min_exercised: int = 1
    brk: Optional[str] = None
    observation: str = ""
    break_detected: Optional[bool] = None
    seconds: float = 0.0


def seed_all(n: int = 0) -> None:
    """Fixed seeds for the deterministic fixtures (ASS-012 CT-02: restore RNG before comparisons)."""
    random.seed(n)
    try:
        import numpy as np
        np.random.seed(n)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(n)
        torch.set_num_threads(1)
    except Exception:
        pass



# ---------------------------------------------------------------------------
# Deliberate breaks for the GOV-015 finiteness guards
# ---------------------------------------------------------------------------
class _MathAcceptsAll:
    """`math` with isfinite() always True: the guard-less behaviour GOV-015 removed."""

    def __getattr__(self, name):
        return (lambda x: True) if name == "isfinite" else getattr(math, name)


@contextlib.contextmanager
def accept_nonfinite(*modules):
    """Reintroduce the defect inside `modules` (each guards with `math.isfinite`)."""
    from unittest import mock
    with contextlib.ExitStack() as st:
        for m in modules:
            st.enter_context(mock.patch.object(m, "math", _MathAcceptsAll()))
        yield


def reintroduce_nonfinite(*module_names):
    """Decorator for a scenario function: when the harness runs it with its
    deliberate break, the named modules accept non-finite numbers again, so the
    scenario must fail (otherwise it cannot fail and proves nothing)."""
    def deco(fn):
        @functools.wraps(fn)
        def run(brk):
            if not brk:
                return fn(brk)
            import importlib
            mods = [importlib.import_module(n) for n in module_names]
            with accept_nonfinite(*mods):
                return fn(brk)
        return run
    return deco


class Harness:
    def __init__(self, file_id: str, title: str, boundary: bool = False):
        self.file_id, self.title, self.boundary = file_id, title, boundary
        self.rows: List[Row] = []
        self._fns: List[Callable] = []
        self.t0 = time.time()
        seed_all(0)

    # ---- declaring scenarios
    def scenario(self, cid: str, title: str, *, brk: str, oracle: str, min_exercised: int = 1,
                 required: bool = True):
        """Decorator. `brk` names the deliberate break the function must honour."""
        if not brk:
            raise ValueError(f"{cid}: a scenario must name a deliberate break (ASS-012 3)")

        def deco(fn):
            self.rows.append(Row(cid, title, required, oracle, min_exercised=min_exercised, brk=brk))
            self._fns.append(fn)
            return fn
        return deco

    def blocked(self, cid: str, reason: str, title: str = "", required: bool = True) -> None:
        r = Row(cid, title or reason[:60], required, "", verdict="BLOCKED", observation=reason)
        self.rows.append(r)
        self._fns.append(None)

    # ---- running
    def _run(self, fn, brk):
        torch_seed = 0
        seed_all(torch_seed)
        return fn(brk)

    def finish(self) -> int:
        failed, blocked, passed = [], [], []
        for row, fn in zip(self.rows, self._fns):
            if fn is None:
                blocked.append(row)
                print(f"  BLOCKED {row.cid} {row.title}: {row.observation}")
                continue
            t = time.time()
            try:
                good = self._run(fn, None)
            except Exception:
                row.verdict, row.observation = "FAIL", "raised: " + traceback.format_exc(limit=3).strip().splitlines()[-1]
                failed.append(row)
                print(f"  FAIL    {row.cid} {row.title}: {row.observation}")
                if os.environ.get("CAT_TRACE"):          # GOV-015 9.9: the one-line reading hid WHICH class/line raised
                    print("  TRACEBACK for the line above:\n" + traceback.format_exc())
                continue
            row.exercised, row.observation = good.exercised, good.note
            if not good.ok:
                row.verdict = "FAIL"
                failed.append(row)
                print(f"  FAIL    {row.cid} {row.title}: oracle not met - {good.note}")
                continue
            if good.exercised < row.min_exercised:
                row.verdict = "INCONCLUSIVE"
                row.observation += f" (exercised {good.exercised} < required {row.min_exercised})"
                failed.append(row)
                print(f"  INCONC  {row.cid} {row.title}: {row.observation}")
                continue
            try:
                bad = self._run(fn, row.brk)
                row.break_detected = not bad.ok
            except Exception:
                row.break_detected = True         # the defect made the real module refuse or raise: detected
            row.seconds = time.time() - t
            if not row.break_detected:
                row.verdict = "FAIL"
                row.observation += f" | deliberate break '{row.brk}' NOT detected"
                failed.append(row)
                print(f"  FAIL    {row.cid} {row.title}: deliberate break '{row.brk}' went undetected (the test cannot fail)")
                continue
            row.verdict = "PASS"
            passed.append(row)
            print(f"  ok      {row.cid} {row.title} [exercised {row.exercised}, break '{row.brk}' detected]")
        if failed:
            verdict = "FAIL"
        elif blocked:
            verdict = f"pass for declared scope ({len(blocked)} blocked)"
        else:
            verdict = "PASS"
        print(f"[categorical {self.file_id}] {len(passed)} passed, {len(blocked)} blocked, {len(failed)} failed: {verdict}")
        if not failed:
            print(f"[categorical {self.file_id}] ALL PASS" if not blocked else
                  f"[categorical {self.file_id}] ALL PASS (declared scope; blocked: " + ", ".join(b.cid for b in blocked) + ")")
        out = os.environ.get("CATEGORICAL_OUT")
        if out:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"{self.file_id}.json"), "w") as f:
                json.dump({"file": self.file_id, "title": self.title, "boundary": self.boundary, "verdict": verdict,
                           "seconds": round(time.time() - self.t0, 1),
                           "contracts": [r.__dict__ for r in self.rows]}, f, indent=1, sort_keys=True)
        return 1 if failed else 0


def fresh_loop(**attrs):
    """A bare namespace standing in for the developmental loop in unit fixtures (the modules read
    attributes off `loop`). Use the REAL module objects around it; this only carries state."""
    from types import SimpleNamespace
    return SimpleNamespace(**attrs)
