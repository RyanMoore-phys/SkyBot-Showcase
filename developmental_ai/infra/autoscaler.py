"""SkyBot's autoscaler: two dials — world-model SIZE and stream COUNT.

docs/research/res-004-rssm-size-experiment.md ("the pyramid"). User idea (2026-10-17): start
with many small SkyBots, grow the brain and cut the streams as it masters its
worlds — and shrink it back when a task needs less. One brain throughout:
"more SkyBots" are more streams feeding it.

This module DECIDES; it never acts. The loop calls `observe()` once per
segment with what it measured, logs the decision, and writes it to
`runlogs/autoscale.json`. Growing/shrinking the RSSM itself is stage D of the
plan and does not exist yet, so every decision is a RECOMMENDATION.

THE SIGNALS (each optional; a missing one is never a reason to act):

  wm_fresh      world-model reconstruction on the NEWEST experience, measured
                before training reuses it — generalisation to new frames
  wm_train      the same loss on the replay batches it trained on
  z_info        nats of information the posterior carries about a frame
                (unclamped KL(post || prior), summed over latents)
  replay_ratio  trained frames / new frames this segment
  act_s, act_budget_s   seconds per decision, and the game's budget (only for
                games whose clock demands one; None = no latency limit)
  vram_frac     reserved / total GPU memory
  novelty       intrinsic income still arriving (any monotone measure)

THE RULES, in priority order:

  1. LATENT COLLAPSED (z_info < z_min) -> HOLD. Found 2026-10-17: the live
     world model's posterior carried 0.00-0.01 nats per frame, so every size
     scored the same and "size matters?" had no meaning. Scaling a model that
     is not using its latent measures nothing.
  2. Over budget -> shrink: acting slower than the game allows, or VRAM over
     `vram_cap`. These win over everything below.
  3. Cooldown after any change, and `window` segments of history first.
  4. PLATEAU (wm_fresh improved < `plateau_rel` over the window):
       fresh/train <= gap_up and novelty still arriving  -> GROW the model
           (and, the pyramid: one step fewer streams, if above the minimum)
       fresh/train >= gap_down or replay_ratio >= rr_high -> MORE DATA
           (more streams; at the maximum, SHRINK the model instead)
       in between                                        -> HOLD
     `gap_up` < `gap_down` is the hysteresis band: no decision flips back and
     forth on noise.

WP-10 (GOV-011): with `adr005.enabled` rule 2 (over budget) follows ADR-005's
order (model, then streams, then stepping rate) through
infra/scale_order.py; off, it is the `_shrink_or` fallback below, unchanged.

NO LATCH (CLAUDE.md §4.1): every HOLD names what releases it, the cooldown is
counted in segments and always expires, sizes and streams are bounded at both
ends, and a recommendation that cannot be applied (already at a bound) falls
through to the other dial or to HOLD — it never repeats itself forever.
"""

from __future__ import annotations

from collections import deque
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Deque, Dict, List, Optional

# The size ladder: same names as tools/rssm_size_experiment.py.
# (deterministic_size, stochastic_size, stochastic_classes, encoder_hidden)
SIZE_LADDER = [
    ("tiny", (256, 16, 16, 128)),
    ("small", (512, 24, 24, 256)),
    ("medium", (1024, 32, 32, 512)),
    ("live", (2048, 48, 48, 768)),
]
SIZE_NAMES = [n for n, _ in SIZE_LADDER]

HOLD, GROW, SHRINK, MORE_STREAMS, FEWER_STREAMS = (
    "hold", "grow_model", "shrink_model", "more_streams", "fewer_streams")


@dataclass
class Signals:
    step: int = 0
    wm_fresh: Optional[float] = None
    wm_train: Optional[float] = None
    z_info: Optional[float] = None
    replay_ratio: Optional[float] = None
    act_s: Optional[float] = None
    act_budget_s: Optional[float] = None
    vram_frac: Optional[float] = None
    novelty: Optional[float] = None
    # WP-10 (GOV-011): game time per decision, only for a real-time game
    # (None = synchronous game or unknown); used to log what slowing implies.
    game_s_per_decision: Optional[float] = None


@dataclass
class Decision:
    action: str
    reason: str
    releases_when: str = ""
    size: str = ""
    streams: int = 0
    target_size: str = ""
    target_streams: int = 0
    step: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        tgt = ""
        if self.action != HOLD:
            tgt = f" -> {self.target_size} x {self.target_streams}"
        rel = f" | releases when {self.releases_when}" if self.releases_when else ""
        ev = " ".join(f"{k}={v}" for k, v in self.evidence.items() if v is not None)
        return (f"{self.action.upper()} ({self.size} x {self.streams}{tgt}): "
                f"{self.reason}{rel}" + (f" | {ev}" if ev else ""))


class Autoscaler:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None,
                 size: str = "live", streams: int = 2):
        c = dict(cfg or {})
        self.window = int(c.get("window", 8))
        self.cooldown = int(c.get("cooldown_segments", 16))
        self.plateau_rel = float(c.get("plateau_rel", 0.02))
        self.gap_up = float(c.get("gap_up", 1.15))
        self.gap_down = float(c.get("gap_down", 1.5))
        self.rr_high = float(c.get("replay_ratio_high", 64.0))
        self.z_min = float(c.get("z_min_nats", 1.0))
        self.vram_cap = float(c.get("vram_cap", 0.85))
        self.latency_headroom = float(c.get("latency_headroom", 0.8))
        self.min_streams = int(c.get("min_streams", 2))
        self.max_streams = int(c.get("max_streams", 8))
        self.min_size = SIZE_NAMES.index(str(c.get("min_size", "tiny")))
        self.max_size = SIZE_NAMES.index(str(c.get("max_size", "live")))
        # actuate=False (the live default): a recommendation is ADVICE, so the
        # state it describes stays the brain that is actually running
        self.actuate = bool(c.get("actuate", False))
        if not self.gap_up < self.gap_down:
            raise ValueError("autoscaler: gap_up must be < gap_down (the hysteresis band)")
        if not self.min_streams <= self.max_streams:
            raise ValueError("autoscaler: min_streams must be <= max_streams")
        self.size = SIZE_NAMES.index(size) if size in SIZE_NAMES else self.max_size
        self.streams = int(streams)
        self.hist: Deque[Signals] = deque(maxlen=max(2, self.window))
        self.segments_since_change = 10 ** 9
        self.history: List[Decision] = []
        # WP-10 (GOV-011): ADR-005 order of response to compute pressure.
        # None (the default) = the old `_shrink_or` fallback, unchanged.
        self.order = None
        _oc = c.get("adr005") or {}
        if _oc.get("enabled", False):
            from developmental_ai.infra.scale_order import PressureOrder
            self.order = PressureOrder(_oc, self)

    # ------------------------------------------------------------ helpers
    def _dec(self, action, reason, sig, releases="", size=None, streams=None) -> Decision:
        d = Decision(action=action, reason=reason, releases_when=releases,
                     size=SIZE_NAMES[self.size], streams=self.streams,
                     target_size=SIZE_NAMES[self.size if size is None else size],
                     target_streams=self.streams if streams is None else streams,
                     step=sig.step, evidence=self._evidence(sig))
        self.history.append(d)
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return d

    def _evidence(self, s: Signals) -> Dict[str, Any]:
        gap = (None if not s.wm_fresh or not s.wm_train
               else round(s.wm_fresh / max(1e-12, s.wm_train), 3))
        return {"fresh": _r(s.wm_fresh), "train": _r(s.wm_train), "gap": gap,
                "z": _r(s.z_info, 2), "rr": _r(s.replay_ratio, 1),
                "vram": _r(s.vram_frac, 2), "act_ms": _r(None if s.act_s is None
                                                          else 1000 * s.act_s, 1)}

    def _window_gap(self) -> Optional[float]:
        """fresh/train over the WINDOW, not one segment (2026-10-18). On main
        the per-segment gap swung 0.62 -> 1.72 between neighbouring segments
        of one tiny x 8 run, and a single 1.72 was enough to act on."""
        f = [h.wm_fresh for h in self.hist if h.wm_fresh and h.wm_train]
        t = [h.wm_train for h in self.hist if h.wm_fresh and h.wm_train]
        if not f:
            return None
        return (sum(f) / len(f)) / max(1e-12, sum(t) / len(t))

    def restore(self, state: Optional[Dict[str, Any]]) -> None:
        """Carry the COOLDOWN across a relaunch (2026-10-18). An actuated
        change is applied BY a restart, and every launch started with
        segments_since_change = 1e9 -- so the change's own restart erased its
        cooldown and the next verdict could come after `window` segments, not
        `cooldown`. Measured on main: tiny x 8 -> x 7 at 40 min, then a fresh
        window. Size and streams are NOT taken from the file: the brain that
        was actually built (configure_autoscaler) is the truth."""
        try:
            n = int((state or {}).get("segments_since_change"))
        except (TypeError, ValueError):
            return
        self.segments_since_change = max(0, n)
        # GOV-015 F-23: the ADR-005 stepping scale was saved but never read
        # back, so the restore-first rule and the comparison fingerprint were
        # lost at the very restart that applied it. Restored within its bounds.
        od = (state or {}).get("order")
        if self.order is not None and isinstance(od, dict):
            try:
                v = float(od.get("step_scale"))
                if math.isfinite(v):
                    self.order.step_scale = min(self.order.max_step_scale,
                                                max(1.0, v))
            except (TypeError, ValueError):
                pass

    def _plateaued(self) -> Optional[bool]:
        f = [h.wm_fresh for h in self.hist if h.wm_fresh is not None]
        if len(f) < self.window:
            return None
        half = max(1, len(f) // 2)
        old, new = sum(f[:half]) / half, sum(f[-half:]) / half
        return (old - new) / max(1e-12, abs(old)) < self.plateau_rel

    def _shrink_or(self, sig, reason, releases):
        """Shrink the model; at the minimum size, fall back to fewer streams."""
        if self.order is not None:
            return self.order.respond(sig, reason, releases)
        if self.size > self.min_size:
            return self._dec(SHRINK, reason, sig, releases, size=self.size - 1)
        if self.streams > self.min_streams:
            return self._dec(FEWER_STREAMS, reason + " (already the smallest model)",
                             sig, releases, streams=self.streams - 1)
        return self._dec(HOLD, reason + " — but already the smallest model at the "
                         "fewest streams", sig, "a bound is raised in the config")

    # --------------------------------------------------------------- decide
    def observe(self, sig: Signals) -> Decision:
        # GOV-015 F-28: infinity (like NaN) is invalid evidence, not pressure.
        # `act_s = +inf` read as a blown latency budget and `vram_frac = +inf`
        # as a full GPU, both giving an actuated SHRINK. Unknown instead.
        bad = {k: None for k, v in asdict(sig).items()
               if isinstance(v, float) and not math.isfinite(v)}
        if bad:
            sig = replace(sig, **bad)
        self.hist.append(sig)
        self.segments_since_change += 1
        d = self._decide(sig)
        if d.action != HOLD:
            # counted as a change either way, so it is not re-issued every
            # segment. Only an ACTUATED change moves the state: in advice mode
            # the old code advanced it anyway, and a live x 2 run logged
            # "live x 3 -> live x 4" (Stage C, 2026-10-18) — advice about a
            # brain that did not exist. Unapplied, the same advice repeats
            # after the cooldown, which is the honest answer.
            self.segments_since_change = 0
            if self.actuate:
                self.size = SIZE_NAMES.index(d.target_size)
                self.streams = d.target_streams
                if self.order is not None:
                    self.order.apply(d)
        return d

    def _decide(self, s: Signals) -> Decision:
        # 1. a collapsed latent makes every other number meaningless
        if s.z_info is not None and s.z_info < self.z_min:
            return self._dec(HOLD, f"world-model latent collapsed ({s.z_info:.2f} nats < "
                             f"{self.z_min}): size decisions would measure nothing", s,
                             f"the posterior carries >= {self.z_min} nats per frame")
        # 2. hard budgets win
        if (s.act_s is not None and s.act_budget_s
                and s.act_s > self.latency_headroom * s.act_budget_s):
            return self._shrink_or(s, f"acting {1000 * s.act_s:.1f} ms > "
                                   f"{self.latency_headroom:.0%} of the game's "
                                   f"{1000 * s.act_budget_s:.1f} ms budget",
                                   "acting fits the budget")
        if s.vram_frac is not None and s.vram_frac > self.vram_cap:
            return self._shrink_or(s, f"VRAM {s.vram_frac:.0%} > cap {self.vram_cap:.0%}",
                                   "VRAM is under the cap")
        # 3. patience
        if (self.order is not None and self.order.step_scale > 1.0
                and self.segments_since_change >= self.cooldown):
            return self.order.restore(s)      # no pressure: undo slowing first
        if self.segments_since_change < self.cooldown:
            return self._dec(HOLD, "cooling down after the last change", s,
                             f"{self.cooldown - self.segments_since_change} more segment(s)")
        plat = self._plateaued()
        if plat is None:
            return self._dec(HOLD, "not enough history", s,
                             f"{self.window} segments with a fresh-data loss")
        if not plat:
            return self._dec(HOLD, "still improving on fresh data", s,
                             f"fresh-data loss improves < {self.plateau_rel:.0%} "
                             f"over {self.window} segments")
        # 4. plateaued: is the brain out of room, or out of data? Judged on
        # the window's gap (see _window_gap), the same window as the plateau.
        gap = self._window_gap()
        memorising = ((gap is not None and gap >= self.gap_down)
                      or (s.replay_ratio is not None and s.replay_ratio >= self.rr_high))
        if memorising:
            why = (f"memorising (fresh/train {gap:.2f} >= {self.gap_down})" if gap
                   and gap >= self.gap_down else
                   f"each frame reused {s.replay_ratio:.0f}x (>= {self.rr_high:.0f})")
            if self.streams < self.max_streams:
                return self._dec(MORE_STREAMS, why + ": more data, not more capacity", s,
                                 streams=self.streams + 1)
            # Less capacity memorises less; FEWER STREAMS would be less data,
            # i.e. MORE memorising. This used _shrink_or, whose fallback is
            # for budgets (VRAM, latency), and on main it turned "memorising
            # at 8 streams" into tiny x 8 -> tiny x 7 (2026-10-18).
            if self.size > self.min_size:
                return self._dec(SHRINK, why + " at the maximum stream count", s,
                                 "the gap closes", size=self.size - 1)
            return self._dec(HOLD, why + " at the maximum stream count, already the "
                             "smallest model", s, "the gap closes, or max_streams is raised")
        underfit = gap is not None and gap <= self.gap_up
        arriving = s.novelty is None or s.novelty > 0.0
        vram_ok = s.vram_frac is None or s.vram_frac < self.vram_cap * 0.8
        if underfit and arriving and vram_ok:
            if self.size < self.max_size:
                return self._dec(GROW, f"out of room: plateaued with fresh/train "
                                 f"{gap:.2f} <= {self.gap_up} and novelty still arriving",
                                 s, size=self.size + 1,
                                 streams=max(self.min_streams, self.streams - 1))
            return self._dec(HOLD, "out of room but already the largest model", s,
                             "max_size is raised in the config")
        return self._dec(HOLD, "plateaued inside the hysteresis band"
                         + ("" if arriving else " (no novelty arriving)"), s,
                         f"fresh/train leaves [{self.gap_up}, {self.gap_down}]")

    def state(self) -> Dict[str, Any]:
        st = {"size": SIZE_NAMES[self.size], "streams": self.streams,
              "segments_since_change": min(self.segments_since_change, 10 ** 6),
              "last": asdict(self.history[-1]) if self.history else None}
        if self.order is not None:
            st["order"] = self.order.state()
        return st


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)
