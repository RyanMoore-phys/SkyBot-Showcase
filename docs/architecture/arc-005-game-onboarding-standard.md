# Game onboarding standard

> **Document control:** [ARC-005](../README.md#architecture) · Architecture / integration standard · **State:** In progress (built, untested) · **Custodian:** Repository maintainer · Revision history: Git.

**Date opened:** 2026-10-21. **Implements:** GOV-009 WP-16 (GOV-008 "adding a game must be easy").
**Status:** the template, the evaluator package and the synthetic proof game are **built and
untested** ([GOV-011](../governance/gov-011-execution-record-gov-009.md), entry WP-16). Nothing
here has run. Extends [ARC-004](arc-004-universal-games.md); bound by [ADR-001](../decisions/adr-001-shared-universal-brain.md)
and [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md).

## 1. The standard in one sentence

A new game is added by **a game interface and a checkpoint detector, both new files**, with no
edit to any learning module. The only generic hook is one branch in `make_env`
(`backend: plugin`), added once.

## 2. What a game supplies

| Piece | Where | Notes |
|---|---|---|
| Game config | `configs/games/<game>.yaml` (template: `configs/games/templates/template_game.yaml`) | Schema and validator: `developmental_ai/game_template/config_schema.py`. Slot 0 is `{}`. |
| Adapter | subclass of `GameAdapterTemplate` (`developmental_ai/game_template/adapter_template.py`), named in the YAML as `adapter: "pkg.module:Class"` | Implement only the `_engine_*` hooks. The base class owns the contract. |
| Launcher | none for in-process engines; `backend: player` + `launcher:` for unmodified games; `GameLauncher` skeleton for the rest | `launcher_template.py`. |
| Install step | the YAML `install:` block, run by `tools/game_install.py` | One install path (ARC-004 section 12). |
| Checkpoint detector | `developmental_ai/evaluator/detectors/<game>.py` exposing `build()` | Evaluator side only (section 4). Template: `detectors/template.py`. |

Choose the path: an **unmodified game seen through a screen and keys** needs no Python (YAML with
`backend: player`). An **engine you can drive in process** needs the adapter subclass.

## 3. The shared environment contract

Stated once, checked by `developmental_ai/game_template/contract.py` (`check_episode` drives any
environment and returns problems).

| Item | Rule |
|---|---|
| Observation | flat float32 `3*S*S` in [0, 1], CHW; the same shape in every stream. |
| Sound | optional `info["audio"]`, float32 `(2, mels, frames)` in [0, 1] (`stereo_mel`). |
| Body senses | `info["world"]` (dict) and `info["proprio"]` (fixed width, default 13). Only what a player's body feels; no positions an engine would know. |
| Actions | fixed slot table, slot 0 the no-op, at most `universal.action_slots`; `valid_mask` with slot 0 true; an invalid slot executes slot 0 and is counted. No action that opens a state without one that closes it (CLAUDE.md 4.5). |
| Reward | the adapter pays exactly 0.0. |
| Life boundaries | life ends: `terminated=True`. Forced discontinuity (rewind, engine rebuilt): `env_restarted=True` and `truncated=True`. A plain truncation is not a boundary for the lifelong body (CLAUDE.md 4.1). |
| Health, stuck | health is a body sense; stuck watching and snapshot rings are optional per-game YAML blocks (`stuck`, `checkpoint`) as in Doom. |
| Evaluator channel | everything only an evaluator may see goes in `info["evaluator"]` (RED). |

### 3.1 Timing: which contract a game follows

| Style | Frame returned by `step(a)` | Adapter must report |
|---|---|---|
| `synchronous` | the outcome of `a` | **no** `executed_action` or `held_action` (the loop's lag handling stays off) |
| `realtime` (the game runs on its own clock) | the outcome of the **previous** held action | `executed_action` (slot that drove the interval), `executed_purity` in [0, 1], `held_action`, and a `held_action` attribute |

The action-lag defect (ASS-003, `core/realtime_credit.py`) came from treating a real-time game as
synchronous: every transition was paired with the wrong action. Declare `timing:` in the YAML; the
template base class implements both styles and the contract check fails a game that mixes them.

## 4. Evaluator isolation (ADR-002, mechanical)

Detectors live in `developmental_ai/evaluator/`. **No module outside that package imports it**, and
the package imports nothing from the learner. The import-graph test
`tests/_evaluator_isolation_smoke.py` scans every module for all import forms and has its own
falsification cases. The checkpoint ladder per game is the **owner's** to define (ASS-007 section
15); `detectors/template.py` and `detectors/synthetic.py` carry placeholder ladders only.

`Detector` (`evaluator/detector.py`) implements the ASS-007 14.5 rules once for every game:
first attainment, time from target entry, prerequisites, "already true at entry" flagged and
excluded, unreached checkpoints left absent for right-censoring.

## 5. Onboarding checklist

1. Copy `configs/games/templates/template_game.yaml` to `configs/games/<game>.yaml`; set `name`,
   `env_name`, `backend`, `timing`, `controller`, `install`.
2. In-process engine: subclass `GameAdapterTemplate`, implement the `_engine_*` hooks, point
   `adapter:` at it. Player-adapter game: nothing to write.
3. Run `validate_game_config` on the YAML and `check_episode` on the adapter (both timings if the
   game can run either).
4. Add `developmental_ai/evaluator/detectors/<game>.py` from the template; list ordered checkpoints
   with prerequisites.
5. `game_ctl.sh <game> install`, then `enable`. No edit to `core/`, `policy/`, `world_model/`,
   `curiosity/`, `infra/` or any other learning directory.
6. Record files and lines added (the synthetic game's count is printed by its onboarding test).

## 6. The synthetic proof game

`developmental_ai/environments/synthetic_game.py` (an 8x8 grid game, optionally real-time with a
configurable lag), `configs/games/templates/synthetic.yaml`, `evaluator/detectors/synthetic.py`.
It exists only to prove sections 1 and 3; it is never enabled for a real run and lives in
`templates/` so `game_ctl.sh` and the default composer directory do not offer it.

## 7. Limits and open points

* Untested. The tests exist as files: `tests/_evaluator_isolation_smoke.py`,
  `tests/_synthetic_game_onboarding_smoke.py`.
* The template's `_proprio` default is zeros. Existing adapters reuse Minecraft's `_proprio`
  (`vizdoom_env` imports the MineRL adapter class); translating that to a game-neutral body-sense
  helper is part of WP-15b (deferred), not done here.
* The sensor-bus wiring in the template mirrors the ViZDoom adapter and has never run.
* Real-time engines still need engine-specific care (a clock thread that owns every engine call);
  the template states the contract, not the threading.
* Passing the contract is not evidence that a game is learnable or that transfer works.

## 8. Related records

[ARC-004](arc-004-universal-games.md), [ADR-001](../decisions/adr-001-shared-universal-brain.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md),
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md),
[GOV-009](../governance/gov-009-implementation-plan.md), [GOV-011](../governance/gov-011-execution-record-gov-009.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-009](arc-009-aiops-observation-and-budget.md): AIOps observation, reporting and API budget
<!-- backlinks:end -->
