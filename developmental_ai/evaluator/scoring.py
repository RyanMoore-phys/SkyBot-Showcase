"""Score one run's checkpoints from its evaluator log lines (ASS-007 3, 12.4, 14.5). Evaluator side.

The RED channel reaches the run log as `<game> stream <i>: {evaluator dict}` lines
(e.g. the Doom adapter's `log_every` line). This module reads them AFTER the run,
feeds a per-game detector (registry.get_detector) and returns a plain dict. It
never touches a live run (ASS-007 2: detecting does not change the run).

One detector per stream; the brain is one learner (section 6), so a checkpoint's
time is the EARLIEST across streams. Times are seconds from the target phase's
start (`t_target_start`, seconds-of-day on the log's clock; a day rollover is
handled like tools/learning_report.py).
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from developmental_ai.evaluator.registry import get_detector

TS = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
EVLINE = re.compile(r"stream (\d+): (\{.*\})\s*$")


def parse_evaluator_log(path: str, game: Optional[str] = None
                        ) -> List[Tuple[float, int, Dict[str, Any]]]:
    """[(t_seconds, stream, evaluator dict)] in log order. `game` filters lines that
    mention it (e.g. 'doom stream 0: {...}'); None keeps every stream line."""
    out: List[Tuple[float, int, Dict[str, Any]]] = []
    last_t, day = None, 0.0
    # A finished run's logs are gzipped by tools/campaign.py (ASS-013); read either form.
    if path.endswith(".gz"):
        import gzip
        fh = gzip.open(path, "rt", errors="replace")
    else:
        fh = open(path, errors="replace")
    for line in fh:
        m = TS.match(line)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + day
        if last_t is not None and t < last_t - 3600:
            day += 86400.0
            t += 86400.0
        last_t = t
        if game and game.lower() not in line.lower():
            continue
        e = EVLINE.search(line)
        if not e:
            continue
        try:
            ev = ast.literal_eval(e.group(2))
        except (ValueError, SyntaxError):
            continue
        if isinstance(ev, dict):
            out.append((t, int(e.group(1)), ev))
    return out


def score_events(game: str, events: List[Tuple[float, int, Dict[str, Any]]],
                 t_target_start: float, budget_s: float) -> Dict[str, Any]:
    """Run the game's detector over the events at/after `t_target_start`.

    Raises DetectorNotDefined (loud) if the game has no detector."""
    # The state at target entry is each stream's LAST line before the phase began
    # (log lines are sparse; a first line inside the phase would hide progress made
    # before it). No earlier line (a fresh-control run) means nothing held at entry.
    entry: Dict[int, Dict[str, Any]] = {}
    for t, stream, ev in events:
        if t < t_target_start:
            entry[stream] = ev
    dets: Dict[int, Any] = {}
    for t, stream, ev in events:
        if t < t_target_start or t > t_target_start + budget_s:
            continue
        det = dets.get(stream)
        if det is None:
            det = dets[stream] = get_detector(game)
            det.begin_phase(entry.get(stream), step=0, t_s=t_target_start)
        det.observe(ev, step=0, t_s=t)
    probe = get_detector(game)                                   # also raises when undefined
    ids = [s.id for s in probe.specs()]
    attained: Dict[str, float] = {}
    at_entry: set = set()
    for det in dets.values():
        r = det.result()
        at_entry |= set(r["at_entry"])
        for c, v in r["attained"].items():
            attained[c] = min(attained.get(c, v["seconds"]), v["seconds"])
    for c in at_entry:
        attained.pop(c, None)
    return {"game": game, "checkpoints": ids, "attained": attained,
            "at_entry": sorted(at_entry), "streams_seen": sorted(dets),
            "budget_s": float(budget_s), "events_used": sum(
                1 for t, _, _ in events if t_target_start <= t <= t_target_start + budget_s)}
