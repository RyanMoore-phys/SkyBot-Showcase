"""Paired analysis for the five-seed evidence standard (ASS-007 5, 14.2-14.7). Evaluator side.

numpy only. Every function is deterministic given the numpy Generator passed in.

Unit of independence: the SEED. Within a seed, observations are repeated events of
one evolving run, so the within-seed interval uses a MOVING-BLOCK bootstrap whose
block length comes from the series' own autocorrelation (14.2). That interval is
descriptive; the gate is the five-seed rule.

Positive on a seed (12.5): the paired difference favours the changed arm AND the
lower bound of its interval exceeds zero AND exceeds the A/A noise floor.

A/A calibration (14.3): the pool of baseline-versus-baseline paired differences
(ten at first, growing as the program creates more) gives
  * the noise floor,
  * the empirical share of five-seed subsets that are "all five positive" (exact
    enumeration when feasible; NOT the per-seed rate to the fifth power, because
    independence is assumed, not verified),
  * the smallest effect the five-seed design could have shown (14.4).

Right-censoring: a checkpoint not reached in the budget contributes 0 to the
attainment-curve area (it was attained at "never"); see `curve_area`.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

LEVEL = 0.95


# ------------------------------------------------------------ autocorrelation
def acf(x: Sequence[float], lag: int) -> float:
    a = np.asarray(x, dtype=float)
    n = len(a)
    if lag >= n or n < 3:
        return 0.0
    a = a - a.mean()
    den = float(np.dot(a, a))
    if den <= 0.0:
        return 0.0
    return float(np.dot(a[:-lag], a[lag:]) / den)


def block_length(x: Sequence[float]) -> int:
    """Smallest lag whose autocorrelation is inside the +-2/sqrt(n) noise band,
    at least 1, at most half the series."""
    n = len(x)
    if n < 4:
        return 1
    band = 2.0 / math.sqrt(n)
    for k in range(1, n // 2 + 1):
        if abs(acf(x, k)) < band:
            return max(1, k)
    return max(1, n // 2)


# ------------------------------------------------------- moving-block bootstrap
def moving_block_bootstrap_means(x: Sequence[float], block_len: int, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    n = len(a)
    if n == 0:
        raise ValueError("empty series")
    L = max(1, min(int(block_len), n))
    starts_n = n - L + 1
    need = int(math.ceil(n / L))
    starts = rng.integers(0, starts_n, size=(n_boot, need))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return a[idx].mean(axis=1)


def interval(x: Sequence[float], rng: np.random.Generator, n_boot: int = 2000,
             level: float = LEVEL, block_len: Optional[int] = None):
    L = block_length(x) if block_len is None else block_len
    m = moving_block_bootstrap_means(x, L, n_boot, rng)
    a = (1.0 - level) / 2.0
    return float(np.quantile(m, a)), float(np.quantile(m, 1.0 - a)), L


# --------------------------------------------------------- censored checkpoints
def curve_area(attained: Dict[str, float], scored: Sequence[str], budget_s: float) -> float:
    """Normalised area under the checkpoint-attainment curve over [0, budget]
    (ASS-007 3.5): mean over scored checkpoints of (budget - t) / budget, with an
    unreached (right-censored) checkpoint contributing 0. In [0, 1]; higher is better."""
    if not scored:
        raise ValueError("no scored checkpoints")
    tot = 0.0
    for c in scored:
        t = attained.get(c)
        if t is not None and t <= budget_s:
            tot += (budget_s - float(t)) / budget_s
    return tot / len(scored)


def window_series(attained: Dict[str, float], scored: Sequence[str], budget_s: float,
                  window_s: float = 300.0) -> List[float]:
    """Fraction of scored checkpoints attained by the END of each fixed window."""
    if not scored:
        raise ValueError("no scored checkpoints")
    n = int(math.ceil(budget_s / window_s))
    out = []
    for w in range(1, n + 1):
        end = min(budget_s, w * window_s)
        out.append(sum(1 for c in scored
                       if attained.get(c) is not None and attained[c] <= end) / len(scored))
    return out


def common_scored(checkpoints: Sequence[str], results: Sequence[Dict],
                  uninformative: Iterable[str] = ()) -> List[str]:
    """Checkpoints scored for one seed across arms (14.5): not true at entry in ANY
    of the arms' runs, and not uninformative in the A/A pool."""
    drop = set(uninformative)
    for r in results:
        drop |= set(r.get("at_entry") or [])
    return [c for c in checkpoints if c not in drop]


def uninformative_checkpoints(checkpoints: Sequence[str], aa_results: Sequence[Dict]) -> List[str]:
    """Never reached by any run in the A/A pool (14.5): excluded and listed."""
    reached = set()
    for r in aa_results:
        reached |= set((r.get("attained") or {}).keys())
    return [c for c in checkpoints if c not in reached]


# --------------------------------------------------------------- paired seeds
@dataclass
class PairedSeed:
    seed: int
    d: float            # mean paired difference (changed - baseline), higher = better
    lo: float
    hi: float
    block_len: int
    n: int

    def halfwidth(self) -> float:
        return (self.hi - self.lo) / 2.0


def paired_seed(seed: int, base: Sequence[float], changed: Sequence[float],
                rng: np.random.Generator, n_boot: int = 2000,
                higher_is_better: bool = True) -> PairedSeed:
    if len(base) != len(changed):
        raise ValueError("paired series must have equal length")
    diff = np.asarray(changed, float) - np.asarray(base, float)
    if not higher_is_better:
        diff = -diff
    lo, hi, L = interval(diff, rng, n_boot)
    return PairedSeed(int(seed), float(diff.mean()), lo, hi, L, len(diff))


# ----------------------------------------------------------- A/A calibration
def noise_floor(aa: Sequence[PairedSeed], alpha: float = 0.05) -> float:
    """(1 - alpha) quantile of |paired difference| between IDENTICAL arms."""
    if len(aa) < 2:
        raise ValueError("need at least two A/A pairs for a noise floor")
    return float(np.quantile(np.abs([p.d for p in aa]), 1.0 - alpha))


def is_positive(p: PairedSeed, floor: float) -> bool:
    return p.d > 0.0 and p.lo > max(0.0, floor)


def is_negative(p: PairedSeed, floor: float) -> bool:
    return p.d < 0.0 and p.hi < -max(0.0, floor)


def five_seed_rule(pairs: Sequence[PairedSeed], floor: float, required: int = 5) -> Dict:
    """The owner's gate (section 5): positive only if ALL `required` seeds are
    positive. A missing seed means no claim."""
    n = len(pairs)
    pos = [p.seed for p in pairs if is_positive(p, floor)]
    neg = [p.seed for p in pairs if is_negative(p, floor)]
    if n < required:
        verdict = "no claim (missing seeds)"
    elif len(pos) == n:
        verdict = "positive"
    elif len(neg) == n:
        verdict = "negative"
    else:
        verdict = "mixed or null"
    return {"verdict": verdict, "seeds": n, "positive": pos, "negative": neg, "floor": floor}


def aa_false_positive(aa: Sequence[PairedSeed], floor: float, k: int = 5,
                      max_subsets: int = 50000,
                      rng: Optional[np.random.Generator] = None) -> Dict:
    """Empirical rate at which k A/A seeds are ALL positive, over k-subsets of the
    pool (exact when C(n, k) <= max_subsets, else a random sample)."""
    n = len(aa)
    if n < k:
        raise ValueError(f"A/A pool of {n} is smaller than k={k}")
    pos = [is_positive(p, floor) for p in aa]
    total = math.comb(n, k)
    if total <= max_subsets:
        hits = sum(all(pos[i] for i in c) for c in itertools.combinations(range(n), k))
        used = total
    else:
        g = rng or np.random.default_rng(0)
        hits = 0
        used = max_subsets
        for _ in range(used):
            hits += all(pos[i] for i in g.choice(n, size=k, replace=False))
    return {"pool": n, "k": k, "per_seed_rate": sum(pos) / n,
            "all_k_rate": hits / used, "subsets": used, "exact": total <= max_subsets}


def expected_false_promotions(rate: float, m_tests: int) -> float:
    return rate * m_tests


def minimum_detectable_effect(aa: Sequence[PairedSeed], floor: float, k: int = 5,
                              power: float = 0.8, draws: int = 4000,
                              rng: Optional[np.random.Generator] = None,
                              steps: int = 200) -> Optional[float]:
    """Smallest true paired effect at which k seeds would ALL be positive with
    probability >= power, simulating seed noise by resampling the centred A/A
    differences and using the pool's mean within-seed half-width as the interval.
    None when no effect on the grid reaches the power."""
    g = rng or np.random.default_rng(0)
    d = np.asarray([p.d for p in aa], float)
    noise = d - d.mean()
    half = float(np.mean([p.halfwidth() for p in aa]))
    thr = max(0.0, floor) + half
    top = max(4.0 * float(np.abs(noise).max()), 2.0 * thr, 1e-9)
    for delta in np.linspace(0.0, top, steps):
        sim = delta + g.choice(noise, size=(draws, k), replace=True)
        if float(np.mean(np.all(sim > thr, axis=1))) >= power:
            return float(delta)
    return None


def serial_correlation(aa: Sequence[PairedSeed]) -> float:
    """Lag-1 correlation of the A/A differences in seed order: a crude check of the
    independence assumption (section 5). Near 0 is consistent with independence."""
    d = [p.d for p in sorted(aa, key=lambda p: p.seed)]
    return acf(d, 1)
