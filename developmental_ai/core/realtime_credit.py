"""Real-time credit: label every transition with the action that DROVE it.

THE BUG THIS FIXES (found 2026-10-18, verified in code)
    A real-time game never waits for SkyBot. ViZDoom's clock thread
    (`vizdoom_env._clock_loop`) steps the engine every 1/35 s with whatever
    buttons are HELD; `_step_realtime(a)` only swaps the held buttons to `a`
    and returns the newest frame AT ONCE. The loop thinks ~165 ms (~5.7
    tics) between decisions, so every tic in the returned interval ran with
    the PREVIOUS decision's buttons:

        step(a_t) returns o_{t+1} = the outcome of a_{t-1}

    The Player adapter (`player/adapter.py`: `set_slot` then `_collect`) is
    built the same way. Every consumer then paired the interval with the
    wrong action: the world model and its inverse head, the curiosity
    forward model and its action-attribution gate (measured 0.0076, "very
    low"), empowerment, and PPO. A policy trained on that is near uniform
    (97-98 % of ln 16 after 40 minutes on main).

THE FIX (three parts; synchronous games are untouched)
    1. EXECUTED LABELS. A real-time env reports `info["executed_action"]`,
       the slot actually held for (the majority of) the returned interval.
       Every transition consumer (o_t -> o_{t+1}) uses it.
    2. POLICY LOOKAHEAD. At decision k the belief is the posterior of o_k,
       and a_{k-1} is still held until step(a_k): a_k will START from the
       state after a_{k-1}'s interval. The policy therefore reads
       `get_latent(imagine_step(posterior_k, onehot(held)))` — one prior
       step through the action that is already in flight.
    3. DELAYED PPO CREDIT (`CreditDelay`). A decision row emitted at loop
       step k is PENDING until step k+1: only then is the interval it drove
       observed. Its reward is recomputed exactly from the per-stream
       history of mixed rewards, R' = sum_i gamma^i m[k0+1+i] over the
       intervals its own decisions drove, and it takes the done flag of the
       step its last action's interval ended at. Rows whose actions never
       ran (a death or restart cut them off, or a standby gap took over)
       are dropped or truncated — never paid for intervals they did not
       drive.

Everything here is pure Python except `lookahead_latent`, so it can be
unit-tested without a world model (tests/_realtime_credit_smoke.py G).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from developmental_ai.environments.held_tags import majority_slot  # noqa: F401  (re-export)



def lookahead_latent(rssm, state: Dict[str, Any], rows: Sequence[int],
                     held: Sequence[int], action_dim: int):
    """Latent for every row of `state`; rows listed in `rows` advanced one
    prior step through their HELD action (`held[i]` for `rows[i]`).

    Returns (latent, hit) where `hit` says whether any row was advanced.
    Caller holds no_grad. Rows not listed keep the posterior latent, so a
    synchronous stream's input is byte-identical."""
    import torch
    lat = rssm.get_latent(state)
    if not rows:
        return lat, False
    idx = torch.as_tensor(list(rows), dtype=torch.long, device=lat.device)
    sub = {"h": state["h"].index_select(0, idx),
           "z": state["z"].index_select(0, idx)}
    a = torch.zeros(len(rows), int(action_dim), device=lat.device,
                    dtype=sub["h"].dtype)
    for i, s in enumerate(held):
        s = int(s)
        a[i, s if 0 <= s < int(action_dim) else 0] = 1.0
    nxt = rssm.imagine_step(sub, a)
    lat = lat.clone()
    lat[idx] = rssm.get_latent(nxt).to(lat.dtype)
    return lat, True


class CreditDelay:
    """Per-stream delayed PPO credit for real-time streams.

    Protocol, once per loop step, for every real-time stream `e`:
        submit(e, row)                (0..1 times: a decision row emitted
                                       this step by the existing store code)
        out = step_end(e, m, done, restart)
                                      (exactly once: this step's mixed
                                       reward, terminal flag, and whether
                                       the step was a restart/rewind whose
                                       interval was never observed)
    and at a segment end / standby gap:
        out = flush(e); gap(e)

    `row` is the dict of `policy.store_transition` keyword arguments,
    including `reward`, `done`, `tau`. Returned rows are the same dicts with
    `reward`, `done` and `tau` rewritten — emit them in order.
    """

    def __init__(self, gamma: float, hist_len: int = 512):
        self.gamma = float(gamma)
        self.hist_len = int(hist_len)
        self.k: Dict[int, int] = {}
        self.hist: Dict[int, deque] = {}
        self.pending: Dict[int, Optional[Dict[str, Any]]] = {}
        self.new: Dict[int, List[Dict[str, Any]]] = {}
        self.emitted = 0
        self.dropped = 0
        self.truncated = 0

    def _init(self, e: int) -> None:
        if e not in self.k:
            self.k[e] = 0
            self.hist[e] = deque(maxlen=self.hist_len)
            self.pending[e] = None
            self.new[e] = []

    def submit(self, e: int, row: Dict[str, Any]) -> None:
        self._init(e)
        r = dict(row)
        r["_k"] = self.k[e]                      # the step it was emitted at
        r["_tau"] = max(1, int(r.get("tau", 1) or 1))
        self.new[e].append(r)

    def _m(self, e: int, j: int) -> Optional[float]:
        for kk, m in reversed(self.hist[e]):
            if kk == j:
                return m
            if kk < j:
                break
        return None

    def _final(self, e: int, row: Dict[str, Any], last_valid: int,
               done: bool) -> Optional[Dict[str, Any]]:
        """Pay `row` for the intervals its decisions drove, up to
        `last_valid` (inclusive). None = nothing it did was ever observed."""
        tau = row["_tau"]
        k0 = row["_k"] - tau + 1                 # step its first decision ran
        first, last = k0 + 1, min(k0 + tau, last_valid)
        if last < first:
            self.dropped += 1
            return None
        R, gp, n = 0.0, 1.0, 0
        for j in range(first, last + 1):
            m = self._m(e, j)
            if m is None:                        # history gap: stop paying
                break
            R += gp * m
            gp *= self.gamma
            n += 1
        if n == 0:
            self.dropped += 1
            return None
        if n < tau:
            self.truncated += 1
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        out["reward"] = float(R)
        out["tau"] = int(n)
        out["done"] = bool(done) and (first + n - 1) == last_valid
        self.emitted += 1
        return out

    def step_end(self, e: int, m: float, done: bool,
                 restart: bool) -> List[Dict[str, Any]]:
        self._init(e)
        k = self.k[e]
        if not restart:
            self.hist[e].append((k, float(m)))
        last_valid = k if not restart else k - 1
        # `done` is the flag the loop stores today (terminal or crash
        # restart); `restart` only decides which intervals were OBSERVED. A
        # rewind is not made a terminal here: that would change what the
        # agent learns about being rewound, which is not this fix's business.
        out: List[Dict[str, Any]] = []
        P, self.pending[e] = self.pending[e], None
        if P is not None:
            r = self._final(e, P, last_valid, done)
            if r is not None:
                out.append(r)
        rows, self.new[e] = self.new[e], []
        for row in rows:
            if done or restart:
                r = self._final(e, row, last_valid, done)
                if r is not None:
                    out.append(r)
            else:
                if self.pending[e] is not None:          # defensive: emit
                    r = self._final(e, self.pending[e], k, False)
                    if r is not None:
                        out.append(r)
                self.pending[e] = row
        if done or restart:
            # a new life / timeline: its first interval is driven by the
            # reset's no-op, and no history may be read across the boundary
            self.hist[e].clear()
        self.k[e] = k + 1
        return out

    def flush(self, e: int) -> List[Dict[str, Any]]:
        """Segment end: emit what was observed of the pending row (its last
        interval runs while the standby actor plays and is never seen)."""
        self._init(e)
        out: List[Dict[str, Any]] = []
        P, self.pending[e] = self.pending[e], None
        if P is not None:
            r = self._final(e, P, self.k[e] - 1, False)
            if r is not None:
                out.append(r)
        for row in self.new[e]:                         # never left over
            r = self._final(e, row, self.k[e] - 1, False)
            if r is not None:
                out.append(r)
        self.new[e] = []
        return out

    def gap(self, e: int) -> None:
        """Time passed that the loop did not record: nothing pending may be
        paid across it, and no history may be read across it."""
        self._init(e)
        if self.pending[e] is not None:
            self.dropped += 1
        self.pending[e] = None
        self.new[e] = []
        self.hist[e].clear()

    def stats(self) -> Dict[str, int]:
        return {"emitted": self.emitted, "dropped": self.dropped,
                "truncated": self.truncated}
