"""Paired comparison of two arms from scored runs (ASS-007 5, 7, 14). Evaluator side.

Input: scored runs, one dict per (arm, seed), as produced by scoring.score_events:
    {"seed": int, "checkpoints": [...], "attained": {id: seconds}, "at_entry": [...],
     "budget_s": float}
Output: one plain dict (JSON-serialisable) that record.py renders.

What it enforces:
  * pairing by seed; a seed present in only one arm is reported missing (no claim);
  * checkpoints true at entry in either arm are excluded for that seed and listed;
    checkpoints never reached anywhere in the A/A pool are excluded and listed;
  * censoring: unreached checkpoints score 0 in the curve area and are counted;
  * the judgement uses the A/A noise floor; WITHOUT an A/A pool no claim is possible
    (the verdict says so, whatever the differences look like).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from developmental_ai.evaluator import stats
from developmental_ai.evaluator.protocol import WINDOW_S


def paired_from_results(base: Dict[int, Dict], changed: Dict[int, Dict],
                        uninformative: Sequence[str], rng: np.random.Generator,
                        window_s: float = WINDOW_S, n_boot: int = 2000,
                        higher_is_better: bool = True) -> Dict[str, Any]:
    pairs: List[stats.PairedSeed] = []
    detail: Dict[int, Dict[str, Any]] = {}
    excluded: Dict[int, str] = {}
    for seed in sorted(set(base) & set(changed)):
        b, c = base[seed], changed[seed]
        scored = stats.common_scored(b["checkpoints"], [b, c], uninformative)
        if not scored:
            excluded[seed] = "no scorable checkpoint (entry-true or uninformative)"
            continue
        budget = float(b["budget_s"])
        sb = stats.window_series(b["attained"], scored, budget, window_s)
        sc = stats.window_series(c["attained"], scored, budget, window_s)
        pairs.append(stats.paired_seed(seed, sb, sc, rng, n_boot, higher_is_better))
        detail[seed] = {
            "scored": scored,
            "area_base": stats.curve_area(b["attained"], scored, budget),
            "area_changed": stats.curve_area(c["attained"], scored, budget),
            "times_base": {k: b["attained"].get(k) for k in scored},
            "times_changed": {k: c["attained"].get(k) for k in scored},
            "censored_base": sum(1 for k in scored if b["attained"].get(k) is None),
            "censored_changed": sum(1 for k in scored if c["attained"].get(k) is None),
            "at_entry": sorted(set(b.get("at_entry") or []) | set(c.get("at_entry") or [])),
            "n_windows": len(sb),
        }
    return {"pairs": pairs, "detail": detail, "excluded_seeds": excluded,
            "missing_seeds": sorted(set(base) ^ set(changed))}


def analyse(base: Dict[int, Dict], changed: Dict[int, Dict],
            aa_pairs: Sequence[stats.PairedSeed], *, test_type: str,
            evidence_class: str, dev_baseline_id: str, arms: Sequence[str],
            uninformative: Sequence[str] = (), rng_seed: int = 0,
            required_seeds: int = 5, higher_is_better: bool = True,
            compute: Optional[Dict[str, Any]] = None,
            deviations: Sequence[str] = ()) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    pc = paired_from_results(base, changed, uninformative, rng,
                             higher_is_better=higher_is_better)
    pairs = pc["pairs"]
    out: Dict[str, Any] = {
        "test_type": test_type, "evidence_class": evidence_class,
        "dev_baseline_id": dev_baseline_id, "arms": list(arms),
        "seeds": [p.seed for p in pairs], "missing_seeds": pc["missing_seeds"],
        "excluded_seeds": pc["excluded_seeds"], "uninformative": list(uninformative),
        "per_seed": [{"seed": p.seed, "d": p.d, "lo": p.lo, "hi": p.hi,
                      "block_len": p.block_len, "n_windows": p.n,
                      **pc["detail"][p.seed]} for p in pairs],
        "compute": compute or {}, "deviations": list(deviations),
        "rng_seed": rng_seed,
    }
    if len(aa_pairs) >= 2:
        floor = stats.noise_floor(aa_pairs)
        out["aa"] = {"pool": len(aa_pairs), "calibrated": len(aa_pairs) >= 10,
                     "floor": floor,
                     "serial_correlation": stats.serial_correlation(aa_pairs)}
        if len(aa_pairs) >= required_seeds:
            out["aa"]["false_positive"] = stats.aa_false_positive(
                aa_pairs, floor, required_seeds, rng=rng)
        mde = stats.minimum_detectable_effect(aa_pairs, floor, required_seeds, rng=rng)
        out["power"] = {"min_detectable_effect": mde, "unit": "attainment-curve fraction",
                        "statement": (
                            f"with this design, an effect smaller than {mde:.3f} on the "
                            "windowed attainment fraction could not have been shown on all five seeds"
                            if mde is not None else
                            "no effect on the searched grid reaches 80% power: a null result "
                            "says nothing about effects of any size here")}
        rule = stats.five_seed_rule(pairs, floor, required_seeds)
        if pc["missing_seeds"] or pc["excluded_seeds"]:
            rule["verdict"] = "no claim (missing or excluded seeds)"
        out["rule"] = rule
    else:
        out["aa"] = {"pool": len(aa_pairs), "calibrated": False}
        out["power"] = {"min_detectable_effect": None, "statement":
                        "no A/A pool: power and noise floor unknown"}
        out["rule"] = {"verdict": "no claim (no A/A calibration)", "seeds": len(pairs),
                       "positive": [], "negative": [], "floor": None}
    return out
