"""A third game is added by FILES ONLY (WP-16 tests 5a, 5c, 5d; ARC-005).

Born 2026-10-21 with GOV-009 WP-16. The synthetic grid game
(environments/synthetic_game.py) is onboarded through exactly the path a real
game takes: a YAML, an adapter subclass of the template, a detector. If any
step needs an edit to a learning module, the standard has failed.

Contracts:
    A. CONFIG. Both templates/ YAMLs pass `validate_game_config`; broken
       variants (slot 0 not {}, unknown backend, plugin without adapter,
       too many slots) are each reported.
    B. COMPOSE. `compose_run_config.compose` accepts the synthetic game (with
       --games-dir configs/games/templates) into a universal run config with
       identical curiosity block to the same compose for the real games' base.
    C. make_env builds it through the generic plugin hook, wrapped to the
       brain's width, and refuses a plugin game on a non-universal run.
    D. EXECUTED-ACTION CONTRACT, BOTH TIMING STYLES. `check_episode` finds no
       problem for synchronous and realtime; and FALSIFICATION: a synchronous
       subclass that reports executed_action, and a realtime subclass that omits
       it, are each flagged.
    E. LAG SEMANTICS. Realtime with lag_tics=1 of 3: executed_action == the
       decision just taken (purity 2/3); with lag_tics=2: executed_action == the
       PREVIOUSLY held slot (the Doom defect, reproduced and labelled right).
    F. INVALID SLOTS run the no-op and are counted, never crash.
    G. RED CHANNEL. info["evaluator"] exists; none of its keys appears in
       info["world"]; reward is exactly 0.0.
    H. DETECTOR. A scripted trajectory reaches moved -> touched -> collected in
       order; one that never touches reaches only `moved` (right-censored);
       prerequisites are enforced (a late-true checkpoint is not scored before
       its prerequisite); a checkpoint already true at entry is flagged; a
       game with no detector raises DetectorNotDefined (loud).
    I. ONBOARDING COST. The files added are listed, none lives in a learning
       directory, and file/line counts are printed (the WP-16 5d record).

Run:  PYTHONPATH=. python tests/_synthetic_game_onboarding_smoke.py
NOT RUN at build time (GOV-011 entry WP-16).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

NAME = "synthetic-onboarding"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

ONBOARD_FILES = [
    "configs/games/templates/synthetic.yaml",
    "developmental_ai/environments/synthetic_game.py",
    "developmental_ai/evaluator/detectors/synthetic.py",
]
LEARNING_DIRS = ("core", "policy", "world_model", "curiosity", "memory", "slots",
                 "skill_bank", "knowledge_graph", "spatial", "infra", "foundation",
                 "sensors", "llm")
S = 32


def _fail(msg):
    print("[%s] FAIL: %s" % (NAME, msg))
    sys.exit(1)


def check(cond, msg):
    if not cond:
        _fail(msg)


def game_cfg(**over):
    with open(os.path.join(REPO, "configs/games/templates/synthetic.yaml")) as f:
        g = yaml.safe_load(f)["game"]
    g.update(over)
    return g


def make(timing="synchronous", cls=None, **over):
    from developmental_ai.environments.synthetic_game import SyntheticGameAdapter
    return (cls or SyntheticGameAdapter)(game_cfg(timing=timing, **over),
                                         image_size=S, seed=3, action_slots=48)


def main():
    from developmental_ai.game_template import contract
    from developmental_ai.game_template.config_schema import validate_game_config

    # A. config schema (with falsification)
    for fn in ("synthetic.yaml", "template_game.yaml"):
        doc = yaml.safe_load(open(os.path.join(REPO, "configs/games/templates", fn)))
        stem = fn[:-5]
        if fn == "template_game.yaml":
            doc["game"]["name"] = "template_game"
        problems = validate_game_config(doc, 48, stem)
        check(not problems, "A: %s: %s" % (fn, problems))
    base = {"game": game_cfg()}
    broken = {
        "slot0": {"controller": [{"UP": 1}, {}]},
        "backend": {"backend": "nonsense"},
        "adapter": {"adapter": ""},
        "slots": {"controller": [{}] + [{"UP": 1}] * 60},
    }
    for label, over in broken.items():
        d = {"game": dict(base["game"], **over)}
        check(validate_game_config(d, 48, "synthetic"), "A: broken variant %r not reported" % label)
    print("[%s] A. config templates validate; 4 broken variants are each reported" % NAME)

    # B. compose
    import compose_run_config as C
    cfg = C.compose(["synthetic"], os.path.join(REPO, "configs/minecraft_skybot.yaml"),
                    os.path.join(REPO, "configs/universal.yaml"),
                    os.path.join(REPO, "configs/games/templates"))
    streams = cfg["parallel_envs"]["streams"]
    check(len(streams) == 2 and streams[0]["backend"] == "plugin", "B: streams %r" % streams)
    cfg_doom = C.compose(["doom"], os.path.join(REPO, "configs/minecraft_skybot.yaml"),
                         os.path.join(REPO, "configs/universal.yaml"),
                         os.path.join(REPO, "configs/games"))
    check(cfg["curiosity"] == cfg_doom["curiosity"],
          "B: curiosity block differs between games (one learning process)")
    print("[%s] B. composes; 2 plugin streams; curiosity block identical to Doom's" % NAME)

    # C. make_env through the plugin hook
    from developmental_ai.environments.wrappers import make_env
    uni = dict(cfg["universal"], enabled=True)
    env, _ = make_env(env_name=streams[0]["env_name"], curriculum=False, image_size=S,
                      universal=uni, game_cfg=streams[0]["game_cfg"], stream_index=0, seed=1)
    obs, info = env.reset()
    check(env.action_space.n == 48 and np.asarray(obs).shape == (3 * S * S,),
          "C: wrapped width wrong: %s %s" % (env.action_space, np.asarray(obs).shape))
    raised = False
    try:
        make_env(env_name=streams[0]["env_name"], curriculum=False, image_size=S,
                 universal=None, game_cfg=streams[0]["game_cfg"])
    except ValueError:
        raised = True
    check(raised, "C: plugin game accepted on a non-universal run")
    print("[%s] C. make_env builds it via the plugin hook at the brain's 48-slot width" % NAME)

    # D. executed-action contract, both timings, with falsification
    for timing in ("synchronous", "realtime"):
        p = contract.check_episode(make(timing), timing, S, 13, n_steps=120)
        check(not p, "D: %s: %s" % (timing, p))
    from developmental_ai.environments.synthetic_game import SyntheticGameAdapter

    class LiarSync(SyntheticGameAdapter):
        def step(self, a):
            o, r, t, tr, i = super().step(a)
            i["executed_action"] = int(a)
            return o, r, t, tr, i

    class SilentRealtime(SyntheticGameAdapter):
        def step(self, a):
            o, r, t, tr, i = super().step(a)
            i.pop("executed_action", None)
            return o, r, t, tr, i

    check(contract.check_episode(make("synchronous", LiarSync), "synchronous", S, 13, 10),
          "D: a synchronous game reporting executed_action was NOT flagged")
    check(contract.check_episode(make("realtime", SilentRealtime), "realtime", S, 13, 10),
          "D: a realtime game omitting executed_action was NOT flagged")
    print("[%s] D. contract holds for both timings; both falsifying subclasses flagged" % NAME)

    # E. lag semantics
    e = make("realtime", lag_tics=1, tics=3)
    e.reset()
    e.step(1)
    _, _, _, _, i = e.step(4)
    check(i["executed_action"] == 4 and abs(i["executed_purity"] - 2 / 3) < 1e-6
          and i["held_action"] == 4, "E: lag 1/3: %r" % {k: i[k] for k in ("executed_action", "executed_purity", "held_action")})
    e = make("realtime", lag_tics=2, tics=3)
    e.reset()
    e.step(1)
    _, _, _, _, i = e.step(4)
    check(i["executed_action"] == 1 and i["held_action"] == 4,
          "E: lag 2/3 must label the PREVIOUS slot: %r" % i)
    print("[%s] E. executed_action is the slot that drove the interval (lag 1/3 and 2/3)" % NAME)

    # F. invalid slots
    e = make()
    e.reset()
    for a in (-1, 99, 10 ** 6):
        e.step(a)
    check(e.invalid_steps == 3, "F: invalid steps %d" % e.invalid_steps)
    print("[%s] F. invalid slots run the no-op and are counted (3)" % NAME)

    # G. RED channel and reward
    e = make()
    _, info = e.reset()
    _, r, _, _, info = e.step(5)
    check(r == 0.0 and "evaluator" in info, "G: reward/evaluator")
    check(not (set(info["evaluator"]) & set(info["world"])), "G: evaluator keys leak into world")
    print("[%s] G. reward 0.0; evaluator keys absent from the learner-visible world" % NAME)

    # H. detector
    from developmental_ai.evaluator.registry import DetectorNotDefined, get_detector

    def run(script, entry_first=False):
        env = make()
        _, info = env.reset()
        det = get_detector("synthetic")
        det.begin_phase(info["evaluator"] if entry_first else None)
        got = []
        for k, a in enumerate(script):
            _, _, term, trunc, info = env.step(a)
            got += det.observe(info["evaluator"], k + 1, float(k + 1))
            if term or trunc:
                break
        return det, got

    walk = [1, 4, 2, 3, 1, 1, 4, 4] * 3
    sweep = [a for _ in range(8) for a in ([5] + [4])] + [5]
    # a long tour that visits every cell and touches each: collects the items
    tour = []
    for row in range(8):
        tour += [5] + [4 if row % 2 == 0 else 3, 5] * 7 + [2]
    tour = [3] * 8 + [1] * 8 + tour
    det, got = run(walk + tour)
    check(got[:3] == ["moved", "touched", "collected"], "H: ladder order: %r" % got)
    det, got = run(walk)
    check(got == ["moved"] and det.result()["not_attained"] == ["touched", "collected"],
          "H: no-touch run: %r %r" % (got, det.result()))
    det, got = run([5] * 4)       # touches but never moves: `touched` needs `moved`
    check(got == [], "H: prerequisite not enforced: %r" % got)
    det, _ = run([], entry_first=True)
    check(det.at_entry == [], "H: fresh life should hold nothing at entry")
    det = get_detector("synthetic")
    det.begin_phase({"cells_visited": 9, "touches": 0, "collected": 0})
    check(det.at_entry == ["moved"], "H: at-entry flag: %r" % det.at_entry)
    raised = False
    try:
        get_detector("no_such_game_xyz")
    except DetectorNotDefined:
        raised = True
    check(raised, "H: missing detector did not raise")
    print("[%s] H. detector: ladder order, censoring, prerequisites, at-entry flag, loud missing" % NAME)

    # I. onboarding cost
    total = 0
    for f in ONBOARD_FILES:
        path = os.path.join(REPO, f)
        check(os.path.isfile(path), "I: missing %s" % f)
        check(not any(f.startswith("developmental_ai/%s/" % d) for d in LEARNING_DIRS),
              "I: %s is inside a learning directory" % f)
        total += sum(1 for _ in open(path))
    print("[%s] I. onboarding = %d files, %d lines, zero learning modules edited"
          % (NAME, len(ONBOARD_FILES), total))
    print("[%s] ALL PASS" % NAME)


if __name__ == "__main__":
    main()
