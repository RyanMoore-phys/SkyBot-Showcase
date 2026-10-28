"""Arm registry for the test campaign (tools/campaign.py; ASS-013).

ONE place that says, for every package and relation cluster, which config keys the
A/B arm switches ON. Overlays are generated from it (never hand-copied), so an arm
cannot drift from its record, and `validate()` fails on a key that does not exist in
configs/universal.yaml (a typo would otherwise be a silent no-op arm: an A/A in disguise).

NOT RUN at build time of the packages (GOV-011); written in the testing phase.

Values are the FIRST ARM of each package as its GOV-011 entry names it (the simplest
combination), NOT tuned values. Change a value here and the next generated overlay
carries it; a run already started keeps the overlay it was frozen with (eval_state.json
hashes the overlay file).

BLOCKED entries are scheduled but never started; the reason is printed in the plan and
the campaign report. A blocked arm is not a pass and not a failure.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mirrors tools/eval_run.sh (which sets game.held_out_maps before merging an overlay). A test asserts the
# two agree; the generated rotation must never name one of these (data_diversity.LeakError would refuse it).
HELD_OUT = ["MAP29", "MAP30", "MAP31", "MAP32"]
STREAMS = 8


def _doom_source_maps() -> List[str]:
    import yaml
    g = yaml.safe_load(open(os.path.join(ROOT, "configs/games/doom.yaml")))["game"]
    return [m for m in g["maps"] if m not in HELD_OUT]


def wp6_overlay(period_s: float = 300.0, phases: int = 8, streams: int = STREAMS) -> dict:
    """WP-6 source diversity (GOV-014 item C). The Doom adapter plays maps[map_i] and advances only on a level
    exit, so without this every source phase stayed on MAP01 (ASS-015). `game.map_assignment` starts stream k on
    a different map and, every `period_s`, re-assigns each stream ONE new starting map (applied at the next
    episode start): phase p gives stream k source map (p * streams + k) mod M, so `phases` x `streams` slots
    walk the whole source list. Environment structure only: no map has a name or goal for the learner.
    The overlay uses the reserved `_games` key that tools/eval_run.sh merges into the game YAML."""
    maps = _doom_source_maps()
    M = len(maps)
    sched = [{"at_s": float(period_s * p),
              "per_stream": {k: [maps[(p * streams + k) % M]] for k in range(streams)}}
             for p in range(1, int(phases))]
    return {"_games": {"doom": {"game": {"map_assignment": {"mode": "spread", "schedule": sched}}}}}


def wp6_probe_overlay() -> dict:
    """Same mechanism on a 20 s clock for the 6-minute live coverage check (tools/map_coverage_check.py)."""
    return wp6_overlay(period_s=20.0, phases=18)


# ---- T1: baseline vs baseline + ONE package (GOV-009; ASS-007 section 8) -------------
# name -> (work package, overlay dict | None, blocked reason | None)
T1: Dict[str, Tuple[str, Optional[dict], Optional[str]]] = {
    "wp5_threshold": ("WP-5", {"curiosity": {"episodic": {"cluster_distance": 0.008}}}, None),
    "wp5_info_gain": ("WP-5", {"curiosity": {"info_gain": {"weight": 0.05}, "report_split": True}}, None),
    "wp1_skills": ("WP-1", None,
                   "RES-005 Step 0 (measure the trigger on recorded play) is not done (GOV-011 D-14)"),
    "wp2_consolidation": ("WP-2", {"consolidation": {"enabled": True, "distill": {"enabled": True},
                                                     "regularizer": {"kind": "ewc"}}}, None),
    "wp3_curriculum": ("WP-3", {"curriculum": {"enabled": True}}, None),
    "wp4_planning": ("WP-4", {"planning": {"enabled": True, "ensemble": {"enabled": True}}}, None),
    "wp6_diversity": ("WP-6", wp6_overlay(), None),
    "wp7_attention": ("WP-7", {"attention_memory": {"enabled": True}}, None),
    "wp8_competence": ("WP-8", {"competence": {"enabled": True}}, None),
    "wp9_adaptive_optim": ("WP-9", {"adaptive_optim": {"enabled": True, "entropy": {"enabled": True}}}, None),
    "wp10_scaling": ("WP-10", None,
                     "long-run protocol deferred (RES-007 H-17): the first growth decision comes after "
                     "about forty minutes, so a one-hour seed barely reaches it"),
    "wp11_risk": ("WP-11", {"risk": {"enabled": True}}, None),
    "wp12_peers": ("WP-12", {"peer_observation": {"enabled": True}}, None),
    "wp13_meta": ("WP-13", {"meta_adapt": {"enabled": True}}, None),
    "wp18_memory": ("WP-18", {"outcome_memory": {"enabled": True}}, None),
    "wp19_representation": ("WP-19", {"representation": {"enabled": True}}, None),
    # WP-20 (GOV-016): the predictive-coding latent core in place of the RSSM posterior. It exists only in the staged
    # snapshot (OWN_CODE below), so it is compared against a baseline run on THAT snapshot, not against ab_baseline.
    "wp20_pc_core": ("WP-20", {"world_model": {"core": "pc", "pc_iters": 4, "pc_step": 0.5, "pc_weight": 1.0,
                                   "pc_learning": "hebbian", "pc_hebbian_lr": 0.01}}, None),
}

# T1 arms that run on their own staged code snapshot (<campaign root>/<dir>) with their OWN baseline arm in the same
# stage: the frozen `code` copy does not contain them, so a boot check or an A/B there would silently run the old model
# (an A/A in disguise). name -> snapshot directory name.
OWN_CODE: Dict[str, str] = {"wp20_pc_core": "pc_code"}

# ---- T2: relationship connected vs cut, blocks present in both (GOV-010 4A) ----------
# cluster -> (prerequisite T1 arms, wired channels as (FROM, TO), blocked reason | None)
T2: Dict[str, Tuple[List[str], List[Tuple[str, str]], Optional[str]]] = {
    "rw1_modulation": (["wp4_planning", "wp9_adaptive_optim", "wp7_attention", "wp2_consolidation"],
                       [("LC", "NE"), ("NE", "AROUS"), ("NE", "CONS"), ("NE", "ATT")], None),
    "rw2_attention": (["wp7_attention"],
                      [("SAL", "ATT"), ("PRED", "ATT"), ("CUR", "ATT"), ("BORE", "ATT")], None),
    "rw3_consolidation": (["wp2_consolidation", "wp11_risk", "wp3_curriculum"],
                          [("EMO", "CONS"), ("AMY", "CONS"), ("RP", "CONS"), ("BREAK", "CONS")], None),
    "rw4_semantic": (["wp18_memory", "wp19_representation"], [("EPI", "SEM")], None),
    "rw5_executive": (["wp4_planning", "wp7_attention"], [("PFC", "WM")], None),
    "rw6_strategy": (["wp2_consolidation", "wp3_curriculum"],
                     [("SRL", "STRAT_ENTRY"), ("STRAT_ENTRY", "SP"), ("STRAT_ENTRY", "INTL"),
                      ("CALIB", "SRL")],
                     "ASS-014 F-24: Consolidator._sample raises when spaced replay is on (RW-6 arms 1 and 3), "
                     "so the connected arm would silently run without consolidation; re-opens when F-24 is fixed"),
    "rw7_motivation": (["wp3_curriculum", "wp11_risk", "wp5_info_gain"],
                       [("ZPD", "MOT"), ("SAFE", "CUR")], None),
}

# ---- Categorical operational pilots (GOV-013 section 2; ASS-012) ---------------------
# category -> member T1 arms whose overlays are merged (observation channels on too)
CATEGORIES: Dict[str, List[str]] = {
    "cat01_perception": ["wp19_representation", "wp4_planning"],
    "cat02_memory": ["wp18_memory", "wp7_attention", "wp2_consolidation"],
    "cat03_planning": ["wp4_planning", "wp11_risk"],
    "cat04_motivation": ["wp5_info_gain", "wp8_competence"],
    "cat05_curriculum": ["wp3_curriculum", "wp9_adaptive_optim", "wp13_meta"],
    "cat06_peers": ["wp12_peers", "wp19_representation"],
    "cat07_runtime": [],
}
CATEGORY_NOTES = {
    "cat04_motivation": "WP-1 skill discovery is blocked (T1 wp1_skills): the category pilots its implemented subset",
    "cat06_peers": "WP-6 data diversity is blocked (T1 wp6_diversity): the category pilots its implemented subset",
    "cat07_runtime": "WP-10 scaling is blocked (long-run protocol): runtime pilot = baseline with resource sampling",
}
OBSERVATION = {"relations": {"enabled": True, "connected_core": {"enabled": True}}}


def merge(a: dict, b: Optional[dict]) -> dict:
    """Deep-merge b into a copy of a (same rule as tools/eval_run.sh)."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def channels_overlay(pairs: List[Tuple[str, str]]) -> dict:
    ov: dict = {"relations": {"enabled": True}}
    for f, t in pairs:
        ov["relations"].setdefault(f, {})[t] = {"enabled": True}
    return ov


def prereq_overlay(names: List[str]) -> dict:
    ov: dict = {}
    for n in names:
        o = T1[n][1]
        if o is None:
            raise KeyError(f"prerequisite {n} is blocked")
        ov = merge(ov, o)
    return ov


def t1_overlay(name: str) -> dict:
    return copy.deepcopy(T1[name][1])


def t2_overlays(name: str) -> Tuple[dict, dict]:
    """(connected, cut). Same blocks in both; ONLY the cluster's wired channels differ
    (cut = relations master on for telemetry, the channels left at their default OFF)."""
    pre, pairs, _ = T2[name]
    base = merge(prereq_overlay(pre), {"relations": {"enabled": True}})
    return merge(base, channels_overlay(pairs)), base


def category_overlay(name: str) -> dict:
    ov: dict = {}
    for n in CATEGORIES[name]:
        ov = merge(ov, T1[n][1])
    return merge(ov, OBSERVATION)


def blocked_reason(kind: str, name: str) -> Optional[str]:
    if kind == "t1":
        return T1[name][2] if T1[name][1] is None else None
    if kind == "t2":
        r = T2[name][2]
        if r:
            return r
        for p in T2[name][0]:
            if T1[p][1] is None:
                return f"prerequisite {p} is blocked: {T1[p][2]}"
    return None


def validate(universal: dict) -> List[str]:
    """Every overlay key must exist in configs/universal.yaml (relations.<FROM>.<TO> excepted:
    those are free-form node ids, checked against the channel registry by the caller)."""
    problems: List[str] = []

    def walk(ov: dict, ref: Any, path: str, free: bool = False) -> None:
        for k, v in ov.items():
            here = f"{path}.{k}" if path else k
            if not isinstance(ref, dict) or (k not in ref and not free):
                problems.append(f"{here}: not in configs/universal.yaml")
                continue
            if isinstance(v, dict):
                walk(v, ref.get(k, {}), here, free=(here == "relations" and False))

    def check_games(games: dict, label: str) -> None:
        import yaml
        for gname, patch in games.items():
            gp = os.path.join(ROOT, "configs/games", f"{gname}.yaml")
            if not os.path.exists(gp):
                problems.append(f"[{label}] _games.{gname}: no configs/games/{gname}.yaml")
                continue
            gdoc = yaml.safe_load(open(gp))
            gsec = patch.get("game", {}) if isinstance(patch, dict) else {}
            # WP-6 keys are ABSENT from the shipped YAML by design (commented out); every other key must exist
            allowed_new = {"map_assignment", "held_out_maps", "visual", "visual_per_stream"}
            for k in gsec:
                if k not in gdoc.get("game", {}) and k not in allowed_new:
                    problems.append(f"[{label}] _games.{gname}.game.{k}: not in configs/games/{gname}.yaml")

    def check(ov: dict, label: str) -> None:
        ov = copy.deepcopy(ov)
        games = ov.pop("_games", None)
        if games:
            check_games(games, label)
        rel = ov.pop("relations", None)
        before = len(problems)
        walk(ov, universal, "")
        if rel is not None:
            ref = universal.get("relations")
            if not isinstance(ref, dict):
                problems.append("relations: not in configs/universal.yaml")
            else:
                for k, v in rel.items():
                    if k in ("enabled", "connected_core"):
                        if k not in ref:
                            problems.append(f"relations.{k}: not in configs/universal.yaml")
                    elif not (isinstance(v, dict) and all(isinstance(x, dict) for x in v.values())):
                        problems.append(f"relations.{k}: expected {{TO: {{enabled: bool}}}}")
        for i in range(before, len(problems)):
            problems[i] = f"[{label}] {problems[i]}"

    for n in T1:
        if T1[n][1] is not None:
            check(T1[n][1], n)
    for n in T2:
        if blocked_reason("t2", n) is None:
            c, k = t2_overlays(n)
            check(c, n + "/connected")
            check(k, n + "/cut")
    for n in CATEGORIES:
        check(category_overlay(n), n)
    return problems


def channel_problems() -> List[str]:
    """Every (FROM, TO) the T2 table names must be a channel the registry declares."""
    from developmental_ai.core import relations as R
    names = {c.name for c in R.build_default_registry({"enabled": False}).channels()}
    out = []
    for n, (_, pairs, _) in T2.items():
        for f, t in pairs:
            if f"rel.{f}.{t}" not in names:
                out.append(f"{n}: rel.{f}.{t} is not a registered channel")
    return out
