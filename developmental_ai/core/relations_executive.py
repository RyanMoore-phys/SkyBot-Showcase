"""RW-5: executive control (GOV-010 section 7; GOV-012 entry RW-5).
BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier behaviour.

Blocks used (built, GOV-011): WP-4 `PlanningPhase` and its `Arbiter` (the choice
between habit and plan, in the LEARNING phase only, D-30), WP-7 `AttentionMemory`
(the short window of recent latents), WP-4 calibration.

WIRED (control): SY-10 PFC -> WM, the WORKING-MEMORY GATE (O'Reilly and Frank:
the prefrontal-basal-ganglia model gates what enters working memory). Whether
stream 0's newest latent enters the attention window is decided by a gate:

    change  = 1 - cosine(newest latent, latest window row)
    z       = running z-score of `change` against its own baseline
    admit   = (z >= thr_z)  OR  (steps since the last admission >= max_skip)

so the window keeps rows where the state changed relative to its own history, and
long uneventful stretches are thinned instead of filling the window with
near-copies. The `max_skip` rule is the ESCAPE PATH (CLAUDE.md 4.1): the gate
cannot starve the window; a failure inside the gate opens it for good (the layer
then reads as before). With the line off nothing is installed.
The channel value is 1.0 when a latent is admitted and 0.0 when it is held out
(neutral 1.0 = always admit = the previous behaviour; the dead-wire alarm reads
"no effect" when the gate never holds anything out).

OBSERVATION (read-only): SY-11 PFC -> EF (share of the arbiter's decisions that
chose the plan this segment), SY-28 FPN -> EF (planning-phase time as a share of
its time budget, the orchestration load), ME-08 MB -> PFC (model-based advantage,
the planner's gain over habit, against the plan share), CP-41 PROC -> AUTO (the
distillation KL of plans into the policy, when `planning.distill.weight > 0`).

NOT BUILT (reasons in GOV-012 RW-5): SY-13 (inhibition as action masks driven by
the controller touches the acting path, which is untouched, D-30), SY-12, SY-14,
CP-29, CP-30, CP-40, CP-44, ID-19 (no controller output reaches these blocks and
a coupling would be declared), SY-30, SY-31, ME-15, ST-31 (reflection and
insight blocks are absent or not instrumented), ID-18 (RES-009 D-18), ID-14
(D-14).

Switches: relations.enabled; relations.PFC.WM.enabled (the gate);
relations.PFC.EF.enabled, relations.FPN.EF.enabled, relations.MB.PFC.enabled,
relations.PROC.AUTO.enabled; relations.executive.{thr_z -0.5, max_skip 4,
ema 0.9, min_n 20}.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import delta
from developmental_ai.core.relations_modulation import ZTrack, _pair

WIRED = ("rel.PFC.WM",)
_S = "per segment"


def _plan(loop):
    P = getattr(loop, "_plan", None)
    return P if P else None


def _plan_n(loop) -> Optional[float]:
    P = _plan(loop)
    return None if P is None else float(P.arbiter.why.get("plan", 0))


def _seen_n(loop) -> Optional[float]:
    P = _plan(loop)
    return None if P is None else float(P.arbiter.seen)


_plan_d = delta(_plan_n, "rw5_plan")
_seen_d = delta(_seen_n, "rw5_seen")


def plan_share(loop) -> Optional[float]:
    a, b = _plan_d(loop), _seen_d(loop)
    if a is None or b is None or b <= 0:
        return None
    return a / b


def phase_load(loop) -> Optional[float]:
    P = _plan(loop)
    if P is None:
        return None
    s = (P.last or {}).get("secs")
    return None if s is None else float(s) / max(1e-6, float(P.max_phase_s))


def plan_gain(loop) -> Optional[float]:
    P = _plan(loop)
    g = (P.last or {}).get("plan_gain") if P is not None else None
    return None if g is None else float(g)


def distill_kl(loop) -> Optional[float]:
    P = _plan(loop)
    g = (P.last or {}).get("distill_kl") if P is not None else None
    return None if g is None else float(g)


def register(reg: RelationRegistry) -> None:
    g = "executive"
    R = reg.register
    R("SY-10", "PFC", "WM", "executive gate (change of the newest latent)",
      "WP-7 attention window", "latent admitted to the window (1) or held out (0)",
      "0/1", "per step", "control", neutral=1.0, group=g, deviation=True)
    R("SY-11", "PFC", "EF", "WP-4 arbiter", "planning phase",
      "share of arbitration decisions that chose the plan", "fraction", _S,
      "data", probe=plan_share, group=g)
    R("SY-28", "FPN", "EF", "planning-phase time", "latency budget",
      "phase seconds as a share of the budget", "fraction", _S, "data",
      probe=phase_load, group=g)
    R("ME-08", "MB", "PFC", "planner gain over habit", "arbiter",
      "plan gain vs plan share (correlation over segments)", "r", _S, "data",
      probe=_pair(plan_gain, plan_share), pair=True, group=g)
    R("CP-41", "PROC", "AUTO", "policy", "plan distillation",
      "distillation KL of plans into the policy", "nats", _S, "data",
      probe=distill_kl, group=g)


class WindowGate:
    """Callable installed as `AttentionMemory.push_gate` (z0, window) -> bool."""

    def __init__(self, reg: RelationRegistry):
        c = dict(reg.cfg.get("executive", {}) or {})
        self.reg = reg
        self.thr = float(c.get("thr_z", -0.5))
        self.max_skip = max(1, int(c.get("max_skip", 4)))
        self.z = ZTrack(float(c.get("ema", 0.9)), int(c.get("min_n", 20)))
        self.since = 0

    def __call__(self, z0: torch.Tensor, win) -> bool:
        n = win.n[0]
        if not n:
            self.since = 0
            return True
        last = win.buf[0, -1]
        change = 1.0 - float(F.cosine_similarity(
            z0.float().reshape(1, -1), last.float().reshape(1, -1)))
        z = self.z.update(change)
        self.since += 1
        admit = (z >= self.thr) or self.since >= self.max_skip
        if admit:
            self.since = 0
        self.reg.send("rel.PFC.WM", 1.0 if admit else 0.0)
        return bool(admit)


def apply(loop, reg: RelationRegistry) -> None:
    am = getattr(loop, "_am", None)
    st = reg.state.setdefault("exe", {})
    if not am or getattr(am, "gate_error", None):
        return                              # failed once: open for the run
    if st.get("gate") is None:
        st["gate"] = WindowGate(reg)
    am.push_gate = st["gate"]


def release(loop, reg: RelationRegistry) -> None:
    am = getattr(loop, "_am", None)
    if am:
        am.push_gate = None
        am.pushed0 = True
    reg.state.pop("exe", None)
