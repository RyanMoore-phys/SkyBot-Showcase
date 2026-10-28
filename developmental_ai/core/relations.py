"""Relation registry (R1, GOV-010 section 3 and 6; ARC-006). BUILT, UNTESTED.

The owner's map (`assets/ai_learning.mmd`) is made of blocks AND lines. GOV-009
built blocks; GOV-010 connects them. This module makes each line an auditable
object: a named CHANNEL `rel.<FROM>.<TO>` (node ids of the map) with

    producer, consumer      what sends and what uses (learner-side only)
    signal, units, time     the quantity that flows, and its time scale
    type                    data | control | descriptive
    switch                  relations.<FROM>.<TO>.enabled   (default OFF)
    statistics              per SEGMENT: mean, spread, last, n (never per step)
    status                  the RES-008 line id (e.g. "NM-01")
    source tag              "learner" (an "evaluator" registration RAISES)

WHAT THIS MODULE IS NOT
    * Not evidence that a line works as it does in a human. It measures
      IMPLEMENTATION COVERAGE only: has a software channel, carries signal, is
      alive. Benefit claims come only from T1/T2 A/Bs (GOV-010 4A).
    * Not a reward. A channel never pays, and never sends an engine or
      evaluator value (ADR-002). `register()` raises on an evaluator-tagged
      source, on an unknown tag, and on a probe defined under the evaluator
      package or a game's detector package.
    * Not wiring. Descriptive lines are RECORDED, never wired. A data/control
      channel that a later cluster wires uses `send()`: with the switch OFF it
      returns the NEUTRAL constant (the previous behaviour, bit for bit); ON it
      records the value and returns it.

SWITCHES (all default OFF; OFF = this module is never imported by the loop)
    relations.enabled                       master (false)
    relations.connected_core.enabled        default for the RW-15 observation
                                            channels (false)
    relations.<FROM>.<TO>.enabled           per channel, overrides the above
    relations.window (20)                   segments in the dead-wire window
    relations.eps (0.01)                    dead below eps x decayed peak
    relations.peak_decay (0.99)             per-segment decay of the peak
    relations.corr_window (20)              segments for pair correlations
    relations.min_corr_n (5)                pairs needed before a correlation

DEAD-WIRE ALARM (guard-becomes-latch review, CLAUDE.md 4.1)
    A channel is flagged when the mean of its consumer-side effect |v| over the
    last `window` segments is <= eps x (decayed peak of its own per-segment
    mean |v|); a channel that has never been nonzero is flagged after a full
    window. The alarm is TELEMETRY ONLY: it gates nothing, so it cannot trap
    the agent. It is a pure function of the window, so it RELEASES by itself the
    first time the flow comes back above the threshold (reachable by the
    system, not by a human); the peak decays so a transient spike cannot make
    normal flow look dead for ever. Nothing is persisted: a relaunch starts
    with an empty window.
"""
from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

LEARNER = "learner"
EVALUATOR = "evaluator"
TYPES = ("data", "control", "descriptive")
_NAME_RE = re.compile(r"^rel\.[A-Z0-9_]+\.[A-Z0-9_]+$")
_LINE_RE = re.compile(r"^[A-Z]{2}-\d{2}$")
# A probe defined in any of these modules reads evaluator state by definition.
_FORBIDDEN_PROBE_PREFIXES = (
    "developmental_ai.evaluator",
    "developmental_ai.game_template.detectors",
)
# Builders: modules exposing `register(registry)`. Later clusters append here.
_BUILDERS = ("developmental_ai.core.relations_connected",
             "developmental_ai.core.relations_modulation",
             "developmental_ai.core.relations_consolidation",
             "developmental_ai.core.relations_semantic",
             "developmental_ai.core.relations_attention",
             "developmental_ai.core.relations_executive",
             "developmental_ai.core.relations_strategy",
             "developmental_ai.core.relations_motivation",
             "developmental_ai.core.relations_transfer",
             "developmental_ai.core.relations_skills",
             "developmental_ai.core.relations_peers",
             "developmental_ai.core.relations_resources")
# Wiring modules (WIRED clusters): each exposes `apply(loop, reg)` (once per
# segment, before the probes are read), optional `finalize(loop, reg)` (after all
# applies; combines factors) and `release(loop, reg)` (coefficients back to
# base). Order matters only for `factors`: modulation first, then the later
# clusters that add factors, with modulation's finalize combining them.
_WIRES = ("developmental_ai.core.relations_modulation",
          "developmental_ai.core.relations_consolidation",
          "developmental_ai.core.relations_semantic",
          "developmental_ai.core.relations_attention",
          "developmental_ai.core.relations_executive",
          "developmental_ai.core.relations_strategy",
          "developmental_ai.core.relations_motivation")


class RelationError(ValueError):
    """A malformed or duplicate registration."""


class EvaluatorLeakError(RuntimeError):
    """A channel tried to carry evaluator or engine state (ADR-002)."""


def channel_name(frm: str, to: str) -> str:
    return "rel.%s.%s" % (frm, to)


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) else None


class Channel:
    """One line of the map. Statistics are accumulated in-segment with O(1)
    state per sample and flushed once per segment."""

    def __init__(self, name: str, line_id: str, frm: str, to: str,
                 producer: str, consumer: str, signal: str, units: str,
                 time_scale: str, ctype: str, source_tag: str,
                 probe: Optional[Callable[[Any], Any]] = None,
                 pair: bool = False, neutral: float = 0.0,
                 note: str = "", deviation: bool = False):
        self.name, self.line_id, self.frm, self.to = name, line_id, frm, to
        self.producer, self.consumer = producer, consumer
        self.signal, self.units, self.time_scale = signal, units, time_scale
        self.type, self.source_tag = ctype, source_tag
        self.probe, self.pair = probe, bool(pair)
        self.neutral = float(neutral)
        self.note = note
        # A wired CONTROL channel whose neutral constant is not 0 (a gain of 1.0)
        # records value - neutral, so the dead-wire alarm means "no effect on the
        # consumer" and not "the value is near zero". Observation channels: False.
        self.deviation = bool(deviation)
        self.enabled = False
        self._reset_segment()
        self.hist: Deque[float] = deque()      # per-segment mean |v|
        self.vals: Deque[float] = deque()      # per-segment mean v (y)
        self.xs: Deque[float] = deque()        # per-segment mean x (pairs)
        self.peak = 0.0
        self.segments = 0
        self.skipped = 0                       # samples the probe could not give
        self.alarm = False

    # ------------------------------------------------------------------
    def _reset_segment(self) -> None:
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.sa = 0.0
        self.sx = 0.0
        self.last: Optional[float] = None

    def observe(self, value: Any, x: Any = None) -> bool:
        """Record one sample (the consumer-side value `value`; for a pair
        channel also the producer-side `x`). Non-finite values are skipped."""
        v = _finite(value)
        if v is None:
            self.skipped += 1
            return False
        if self.pair:
            xv = _finite(x)
            if xv is None:
                self.skipped += 1
                return False
            self.sx += xv
        self.n += 1
        self.s += v
        self.ss += v * v
        self.sa += abs(v)
        self.last = v
        return True

    def flush(self, window: int, eps: float, peak_decay: float,
              corr_window: int) -> Dict[str, Any]:
        """End of segment: summarise, update the window, evaluate the alarm."""
        out: Dict[str, Any] = {"n": self.n, "skipped": self.skipped}
        if self.n > 0:
            mean = self.s / self.n
            var = max(0.0, self.ss / self.n - mean * mean)
            amean = self.sa / self.n
            out.update(mean=mean, spread=math.sqrt(var), last=self.last)
            self.hist.append(amean)
            self.vals.append(mean)
            if self.pair:
                self.xs.append(self.sx / self.n)
                out["x_mean"] = self.xs[-1]
            self.peak = max(self.peak * peak_decay, amean)
            self.segments += 1
            while len(self.hist) > max(window, corr_window):
                self.hist.popleft()
                self.vals.popleft()
                if self.pair:
                    self.xs.popleft()
        else:
            self.peak *= peak_decay
        out["corr"] = (self.corr(corr_window, getattr(self, "min_corr_n", 5))
                       if self.pair else None)
        self.alarm = self._dead(window, eps)
        out["alarm"] = self.alarm
        self._reset_segment()
        return out

    def _dead(self, window: int, eps: float) -> bool:
        if len(self.hist) < window:
            return False
        recent = list(self.hist)[-window:]
        m = sum(recent) / len(recent)
        if self.peak <= 0.0:
            return True                      # never nonzero over a full window
        return m <= eps * self.peak

    def corr(self, corr_window: int, min_n: int = 5) -> Optional[float]:
        """Pearson r of (producer x, consumer y) over the last segments."""
        if not self.pair or len(self.xs) < min_n:
            return None
        xs = list(self.xs)[-corr_window:]
        ys = list(self.vals)[-corr_window:]
        n = min(len(xs), len(ys))
        if n < min_n:
            return None
        xs, ys = xs[-n:], ys[-n:]
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        if sxx <= 1e-18 or syy <= 1e-18:
            return None
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        return sxy / math.sqrt(sxx * syy)


class RelationRegistry:
    def __init__(self, cfg: Optional[dict] = None):
        c = dict(cfg or {})
        self.cfg = c
        self.master = bool(c.get("enabled", False))
        self.window = max(2, int(c.get("window", 20)))
        self.eps = float(c.get("eps", 0.01))
        self.peak_decay = float(c.get("peak_decay", 0.99))
        self.corr_window = max(3, int(c.get("corr_window", 20)))
        self.min_corr_n = max(3, int(c.get("min_corr_n", 5)))
        self.core_default = bool((c.get("connected_core", {}) or {}).get(
            "enabled", False))
        self._ch: Dict[str, Channel] = {}
        self._by_line: Dict[str, str] = {}
        self.last_segment: Dict[str, Any] = {}
        self.segments = 0
        self.factors: Dict[str, Dict[str, float]] = {}   # group -> {channel: factor}
        self.state: Dict[str, Any] = {}                   # per-module private state
        self._wire_off: set = set()
        self._wire_hold: Dict[str, list] = {}     # name -> [failures, segments to wait]
        self._wire_clean = 0

    # ---------------- registration ----------------
    def register(self, line_id: str, frm: str, to: str, producer: str,
                 consumer: str, signal: str, units: str, time_scale: str,
                 ctype: str = "data", source_tag: str = LEARNER,
                 probe: Optional[Callable[[Any], Any]] = None,
                 pair: bool = False, neutral: float = 0.0,
                 group: str = "", note: str = "",
                 deviation: bool = False) -> Channel:
        """Declare a channel. Raises EvaluatorLeakError for an evaluator-side
        source (ADR-002), RelationError for anything malformed or duplicate."""
        if source_tag == EVALUATOR:
            raise EvaluatorLeakError(
                "%s: a channel source may not be evaluator-side (ADR-002)"
                % channel_name(frm, to))
        if source_tag != LEARNER:
            raise RelationError("%s: source tag must be %r, got %r"
                                % (channel_name(frm, to), LEARNER, source_tag))
        mod = getattr(probe, "__module__", "") or ""
        if any(mod.startswith(p) for p in _FORBIDDEN_PROBE_PREFIXES):
            raise EvaluatorLeakError(
                "%s: probe defined in %s reads evaluator state (ADR-002)"
                % (channel_name(frm, to), mod))
        if ctype not in TYPES:
            raise RelationError("type must be one of %s, got %r" % (TYPES, ctype))
        name = channel_name(frm, to)
        if not _NAME_RE.match(name):
            raise RelationError("bad channel name %r" % name)
        if not _LINE_RE.match(line_id or ""):
            raise RelationError("%s: line id %r is not a RES-008 id"
                                % (name, line_id))
        if name in self._ch or line_id in self._by_line:
            raise RelationError("duplicate channel %s / line %s"
                                % (name, line_id))
        if ctype == "descriptive" and probe is not None:
            raise RelationError("%s: a descriptive line is recorded, never "
                                "wired (no probe)" % name)
        ch = Channel(name, line_id, frm, to, producer, consumer, signal,
                     units, time_scale, ctype, source_tag, probe, pair,
                     neutral, note, deviation)
        ch.group = group
        ch.enabled = self._resolve_enabled(ch, group)
        self._ch[name] = ch
        self._by_line[line_id] = name
        return ch

    def _resolve_enabled(self, ch: Channel, group: str) -> bool:
        if ch.type == "descriptive" or not self.master:
            return False
        per = (self.cfg.get(ch.frm, {}) or {})
        per = per.get(ch.to) if isinstance(per, dict) else None
        if isinstance(per, dict) and "enabled" in per:
            return bool(per["enabled"])
        if group == "connected_core":
            return self.core_default
        return False

    # ---------------- access ----------------
    def channels(self) -> List[Channel]:
        return list(self._ch.values())

    def get(self, name: str) -> Optional[Channel]:
        return self._ch.get(name)

    def by_line(self, line_id: str) -> Optional[Channel]:
        n = self._by_line.get(line_id)
        return self._ch.get(n) if n else None

    # ---------------- flow ----------------
    def send(self, name: str, value: float, x: Any = None) -> float:
        """For a wired relation: OFF returns the NEUTRAL constant (previous
        behaviour); ON records the sample and returns `value` unchanged."""
        ch = self._ch[name]
        if not ch.enabled:
            return ch.neutral
        ch.observe(value - ch.neutral if ch.deviation else value, x)
        return value

    def is_on(self, name: str) -> bool:
        ch = self._ch.get(name)
        return bool(ch is not None and ch.enabled)

    def any_on(self, names) -> bool:
        return any(self.is_on(n) for n in names)

    def add_factor(self, group: str, name: str, value: float) -> None:
        self.factors.setdefault(group, {})[name] = float(value)

    def wire_all(self, loop: Any) -> None:
        """Run every wiring module that has an ENABLED wired channel, once per
        segment. A module that raises is released (its coefficients return to
        base, the guard-becomes-latch rule) and idled for the run; the rest
        continue. With every wired channel off nothing is imported or run."""
        import importlib
        import logging
        self.factors = {}
        # GOV-015 F-09: a module that raised was idled for the rest of the run.
        # It is re-armed after a hold of 1, 2, 4, ... segments (cap 32); the
        # failure count is forgiven after 16 segments without a failure.
        for name in list(self._wire_off):
            h = self._wire_hold.get(name)
            if h is None or h[1] <= 0:
                self._wire_off.discard(name)
            else:
                h[1] -= 1
        self._wire_clean += 1
        if self._wire_clean >= 16:
            self._wire_hold.clear()
            self._wire_clean = 0
        mods = []
        for name in _WIRES:
            if name in self._wire_off:
                continue
            m = importlib.import_module(name)
            if self.any_on(getattr(m, "WIRED", ())):
                mods.append((name, m))
        for name, m in mods:
            try:
                m.apply(loop, self)
            except Exception as exc:
                self._wire_off.add(name)
                self._note_wire_failure(name)
                self.release_one(m, loop)
                logging.getLogger(__name__).warning(
                    "relation wiring %s failed (idle, released): %r", name, exc,
                    exc_info=True)
        if self.factors:                      # a factor mixer must always finalize
            have = {n for n, _ in mods}
            for name in _WIRES:
                if name in self._wire_off or name in have:
                    continue
                m = importlib.import_module(name)
                if getattr(m, "FINALIZES_FACTORS", False):
                    mods.append((name, m))
        for name, m in mods:
            if name in self._wire_off or not hasattr(m, "finalize"):
                continue
            try:
                m.finalize(loop, self)
            except Exception as exc:
                self._wire_off.add(name)
                self._note_wire_failure(name)
                self.release_one(m, loop)
                logging.getLogger(__name__).warning(
                    "relation finalize %s failed (idle, released): %r", name,
                    exc, exc_info=True)

    def _note_wire_failure(self, name: str) -> None:
        n = (self._wire_hold[name][0] if name in self._wire_hold else 0) + 1
        self._wire_hold[name] = [n, min(32, 2 ** (n - 1))]
        self._wire_clean = 0

    def release_one(self, m: Any, loop: Any) -> None:
        try:
            m.release(loop, self)
        except Exception:
            pass

    def release_all(self, loop: Any) -> None:
        """Every wiring module back to neutral (used when the registry idles)."""
        import importlib
        for name in _WIRES:
            try:
                self.release_one(importlib.import_module(name), loop)
            except Exception:
                pass

    def sample_all(self, loop: Any) -> int:
        """Read every enabled observation channel's probe once. A probe that
        raises or gives nothing is counted as skipped, never propagated."""
        got = 0
        for ch in self._ch.values():
            if not ch.enabled or ch.probe is None:
                continue
            try:
                r = ch.probe(loop)
            except Exception:
                ch.skipped += 1
                continue
            if r is None:
                ch.skipped += 1
                continue
            if ch.pair:
                try:
                    x, y = r
                except Exception:
                    ch.skipped += 1
                    continue
                got += int(ch.observe(y, x))
            elif isinstance(r, (list, tuple)):
                for item in r:
                    got += int(ch.observe(item))
            else:
                got += int(ch.observe(r))
        return got

    def end_segment(self) -> Dict[str, Any]:
        """Flush every enabled channel; return and stash the segment summary."""
        self.segments += 1
        chans: Dict[str, Any] = {}
        alarms: List[str] = []
        for ch in self._ch.values():
            if not ch.enabled or ch.type == "descriptive":
                continue
            ch.min_corr_n = self.min_corr_n
            s = ch.flush(self.window, self.eps, self.peak_decay,
                         self.corr_window)
            s["line"] = ch.line_id
            chans[ch.name] = s
            if s["alarm"]:
                alarms.append(ch.name)
        self.last_segment = {"segment": self.segments, "channels": chans,
                             "alarms": alarms,
                             "live": sum(1 for c in chans.values()
                                         if c["n"] > 0)}
        return self.last_segment

    def line(self) -> str:
        s = self.last_segment or {}
        ch = s.get("channels", {})
        al = s.get("alarms", [])
        tail = ""
        if al:
            tail = " DEAD-WIRE? " + ", ".join(
                "%s(%s)" % (a, ch[a]["line"]) for a in al[:6])
            if len(al) > 6:
                tail += " +%d" % (len(al) - 6)
        return "%d channels on, %d sampled this segment, %d dead-wire alarms%s" % (
            len(ch), s.get("live", 0), len(al), tail)


def build_default_registry(cfg: Optional[dict] = None) -> RelationRegistry:
    """The registry with every built cluster's channels declared (switches per
    `cfg`; with cfg None or enabled false every channel is OFF)."""
    import importlib
    reg = RelationRegistry(cfg)
    for mod in _BUILDERS:
        importlib.import_module(mod).register(reg)
    return reg


def is_descriptive_line(line: Dict[str, Any]) -> bool:
    """Rule for map lines that are hypotheses with no literal software channel
    (GOV-010 section 3): every line to or from the central node (group LC) and
    every dotted (modulating or negated) line. Recorded, never wired."""
    return line.get("group") == "LC" or line.get("kind") == "dotted"
