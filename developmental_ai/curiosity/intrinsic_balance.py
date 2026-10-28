"""WP-5 (GOV-009): rebalancing the intrinsic signals. BUILT, UNTESTED.

Two small, switchable parts, both imported lazily by the loop:

* `ShareCap`: a per-term cap on the share of the stream's income one term may
  supply. It does not zero a term; it limits what the term can add on top of
  the rest of the stream's income (an exponential average of |rest|), so a
  term that would be 90 % of income is held to `share`. A small `floor`
  keeps a term alive when the rest is ~0 (guard-becomes-latch, CLAUDE.md 4.1:
  without a floor a quiet rest would starve the term and the term could
  never recover the rest's income, because it is a cause of it).
* `InfoGainTerm`: pays for the part of the world model's surprise that the
  agent's OWN action explains (`agency`, see `action_surprise.py`). Universal
  and learner-side: computed from the latent's posterior and priors only.

Reward-channel questions (CLAUDE.md 4.4), answered up front:
  channel: the stream's `intrinsic` (zeroed with the rest of intrinsic when a
  GUI is open, like every curiosity term); gated: by the life-boundary guard
  and the weight (0 = off); survives occlusion: it is a function of the
  latent, so a black screen explained by the action pays what the latent
  says (the share cap bounds it); pays while the agent does nothing: a no-op
  action is one of the actions averaged into pbar, so a no-op that changes
  nothing has agency ~0 and pays ~0 (a contract to test).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


class ShareCap:
    def __init__(self, share: float = 0.5, decay: float = 0.99,
                 floor: float = 0.002):
        # share in (0, 1): the largest fraction of (rest + term) the term may be
        self.share = min(0.99, max(0.01, float(share)))
        self.decay = float(decay)
        self.floor = max(0.0, float(floor))
        self._rest: Dict[int, float] = {}

    def limit(self, stream: int, amount: float, rest: float) -> float:
        """The amount the term may pay this step (<= `amount`, >= 0).
        Updates the stream's running |rest| first, so a burst of rest income
        raises the allowance and a quiet period lowers it (it re-opens by
        itself; nothing latches)."""
        s = int(stream)
        r = abs(float(rest))
        prev = self._rest.get(s)
        if not math.isfinite(r):
            # GOV-015 F-10: a NaN or infinite income is not evidence. It used
            # to enter the running mean and make every later allowance NaN.
            r = prev if prev is not None else 0.0
        cur = r if prev is None else self.decay * prev + (1.0 - self.decay) * r
        self._rest[s] = cur
        allowed = cur * self.share / (1.0 - self.share) + self.floor
        a = float(amount)
        if not math.isfinite(a):
            return 0.0
        return a if a <= allowed else allowed


class InfoGainTerm:
    """pay_t = weight * min(clip, max(0, agency_t - floor)) per unguarded row."""

    def __init__(self, weight: float, floor: float = 0.5, clip: float = 5.0,
                 agency_frac: float = 0.0, cap_share: float = 0.0,
                 cap_floor: float = 0.002):
        self.weight = float(weight)
        self.floor = float(floor)
        self.clip = float(clip)
        self.agency_frac = float(agency_frac)
        self.cap = ShareCap(cap_share, floor=cap_floor) if cap_share > 0 else None
        self.last_mean_agency = 0.0
        self.n_rows = 0

    def pay(self, surprise, rest: Sequence[float]) -> List[float]:
        """Payments per stream for this step. `surprise` is a StepSurprise."""
        n = int(surprise.s.shape[0])
        out = [0.0] * n
        # guard[i] True = the row is within k steps of a life boundary: IGNORE it
        # (StepSurprise.guard, GuardClock). Before 2026-10-22 this read True as
        # eligible, so information gain was paid only just after a boundary
        # (ASS-014 F-13, GOV-014 item B1).
        rows = [i for i in range(n) if not surprise.guard[i]]
        if not rows or self.weight <= 0.0:
            return out
        ag = surprise.agency_for(rows)
        if ag is None:
            return out
        s = surprise.s.tolist()
        tot = 0.0
        for i in rows:
            a = ag[i]
            tot += a
            if self.agency_frac > 0.0 and a < self.agency_frac * s[i]:
                continue
            v = min(self.clip, max(0.0, a - self.floor))
            p = self.weight * v
            if p > 0.0 and self.cap is not None:
                p = self.cap.limit(i, p, rest[i] if i < len(rest) else 0.0)
            out[i] = p
        self.last_mean_agency = tot / max(1, len(rows))
        self.n_rows += len(rows)
        return out
