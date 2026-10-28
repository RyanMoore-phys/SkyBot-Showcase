"""Result-record generator (ASS-007 section 7). Evaluator side. Pure string building.

Renders the dict from analysis.analyse as markdown that is APPENDED to the result
section of the protocol that applies (never edited into a protocol, ASS-007 7).
The same function every time, so records are comparable: arms, seeds, per-seed
paired differences with intervals and the A/A floor, per-checkpoint times, censoring
counts, compute and memory, power statement, deviations, evidence class.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(a: Dict[str, Any], title: str, date: str) -> str:
    rule = a["rule"]
    aa = a.get("aa", {})
    L: List[str] = [f"### Result record: {title}", "",
                    f"**Date:** {date}. **Type:** {a['test_type']}. "
                    f"**Evidence class:** {a['evidence_class']}. "
                    f"**Development baseline:** {a['dev_baseline_id']}.",
                    f"**Arms:** {' vs '.join(a['arms'])}. **Seeds scored:** {a['seeds']}. "
                    f"**Missing:** {a['missing_seeds'] or 'none'}. "
                    f"**Excluded seeds:** {a['excluded_seeds'] or 'none'}.", "",
                    f"**Verdict (five-seed rule): {rule['verdict']}.** "
                    "Positive on a seed means the paired difference is favourable and its "
                    "lower bound is above zero and above the A/A noise floor; all five must pass.", ""]
    if a["evidence_class"].lower().startswith("development"):
        L += ["This is development evidence (ASS-007 14.1): it is not evidence that SkyBot "
              "learns well and is not presented as program evidence.", ""]
    L += ["**A/A calibration.** " + (
        f"pool {aa.get('pool')} pairs ({'calibrated' if aa.get('calibrated') else 'NOT calibrated: fewer than ten'}); "
        f"noise floor {_f(aa.get('floor'))}; lag-1 serial correlation {_f(aa.get('serial_correlation'))}. "
        if aa.get("pool") else "none: no claim is possible. "), ""]
    fp = aa.get("false_positive")
    if fp:
        L += [f"Empirical share of five-seed A/A subsets that are all positive: {_f(fp['all_k_rate'], 4)} "
              f"({fp['subsets']} subsets, {'exact' if fp['exact'] else 'sampled'}); per-seed rate "
              f"{_f(fp['per_seed_rate'])}. Expected false promotions over m tests = m x {_f(fp['all_k_rate'], 4)}.", ""]
    L += ["**Power.** " + a["power"]["statement"] + ".", ""]
    L += ["**Per seed** (paired difference of the windowed attainment fraction, "
          "changed minus baseline; moving-block bootstrap 95% interval, block length from the series' "
          "own autocorrelation, descriptive only):", "",
          "| seed | d | 95% interval | block | windows | area base | area changed | censored base/changed | at entry |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in a["per_seed"]:
        L.append(f"| {r['seed']} | {_f(r['d'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['block_len']} | "
                 f"{r['n_windows']} | {_f(r['area_base'])} | {_f(r['area_changed'])} | "
                 f"{r['censored_base']}/{r['censored_changed']} | {r['at_entry'] or '-'} |")
    L += ["", "**Per-checkpoint attainment time (seconds from target entry; `-` = censored at the budget):**", ""]
    for r in a["per_seed"]:
        tb = ", ".join(f"{k}={'-' if v is None else int(v)}" for k, v in r["times_base"].items())
        tc = ", ".join(f"{k}={'-' if v is None else int(v)}" for k, v in r["times_changed"].items())
        L.append(f"* seed {r['seed']}: baseline {tb}; changed {tc}")
    if a["uninformative"]:
        L += ["", f"**Excluded as uninformative (never reached by any A/A run):** {a['uninformative']}."]
    cmp = a.get("compute") or {}
    L += ["", "**Compute and memory:** " + (json.dumps(cmp, sort_keys=True) if cmp else "not supplied")
          + ".", "", "**Deviations from the protocol:** "
          + ("; ".join(a["deviations"]) if a["deviations"] else "none disclosed") + ".", "",
          "Null and negative results are reported with the same prominence as positive ones "
          "(ASS-007 7)."]
    return "\n".join(L) + "\n"
