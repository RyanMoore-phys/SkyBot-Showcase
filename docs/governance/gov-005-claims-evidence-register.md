# Claims and evidence register

> **Document control:** GOV-005 · Governance / research traceability · **State:** Register · **Custodian:** Repository maintainer · Revision history: Git.

## 1. Purpose

This register connects consequential claims about Developmental-AI to their
implementation, protocol, evidence, limits, and next decision gate. It is an
index and status summary; the linked protocol and result record remain the
authoritative source for methods and measurements.

The project goal is to study measurable functional analogs of development,
including cumulative learning, self-directed goals, skill composition,
reasoning, and transfer. Those are research aims, not a single demonstrated
capability. The [research roadmap](gov-001-research-roadmap.md) defines the
broader ladder. This register focuses on claims that materially shape current
decisions or are easy to overstate.

## 2. Evidence and status vocabulary

Evidence class describes where the evidence was produced; it does not rank the
quality of every result. Status describes the current strength and scope of the
claim.

| Status | Meaning |
|---|---|
| **Supported—bounded** | Evidence supports the stated result in its named setup; do not generalize beyond the recorded scope. |
| **Inconclusive** | The result does not resolve the claim under the stated protocol. |
| **Negative** | The protocol did not support the claimed effect for the tested route or setup. |
| **Observed failure** | A behavioral or mechanism failure was measured; a proposed or implemented fix is not yet evidence of resolution. |
| **Implemented—not behaviorally validated** | Relevant code or integration exists, but the target behavior has not been demonstrated. |
| **Supported—offline only** | The result holds in an offline or replay experiment; live behavior remains open. |

Evidence classes used below include synthetic task, offline/replay, live-game
measurement, and code/integration inspection. A test pass establishes only its
software contract; it does not by itself establish task competence, transfer,
or developmental behavior.

## 3. Active and decision-relevant claims

| ID | Claim | Current status and evidence class | Primary record | Implementation / protocol | Limits and next gate |
|---|---|---|---|---|---|
| CLM-001 | Learning-progress curiosity can avoid persistent attraction to an unpredictable noise source while autonomously expanding a coverable frontier. | **Supported—bounded**; synthetic task, 4 seeds. | [GOV-001, Rung 3](gov-001-research-roadmap.md#rung-3--autotelic-self-directed-goals-) | `developmental_ai/curiosity/learning_progress.py`; `rung3_autotelic.py` as identified in the roadmap. | This demonstrates a function in a controlled gridworld. Reconfirm the gate on current code and do not infer live-game behavior from it. |
| CLM-002 | Chaining two learned navigation skills can solve the tested sequential-goal task without additional training and faster than the tested learning baselines. | **Supported—bounded**; synthetic task, 3 seeds. | [GOV-001, Rung 2](gov-001-research-roadmap.md#rung-2--skill-composition-) | `CompositeSkillExecutor`; `rung2_composition.py` as identified in the roadmap. | Scope is the specified SeqGoal task and skill set. This does not establish general skill composition across games. |
| CLM-003 | Policy-weight warm-start from a banked skill can accelerate transfer to the harder SeqGoal 9x9 target. | **Supported—bounded**; synthetic task, 3 seeds. | [GOV-001, Rung 1 revisit](gov-001-research-roadmap.md#rung-1--cumulative-transfer-) | `rung1_revisit.py` as identified in the roadmap. | This supports the warm-start route on this task; it does not establish that skill retrieval or arbitrary cross-game transfer works. The original DoorKey skill-bank retrieval route remains null. |
| CLM-004 | Posterior-conditioned branching can improve action-consequence matching over prior-only prediction in the reported DoorKey counterfactual experiment. | **Supported—bounded, low replication**; local synthetic task, one seed. | [GOV-001, Rung 8](gov-001-research-roadmap.md#rung-8--counterfactual-reasoning-pearls-rung-3-) | `developmental_ai/world_model/rssm.py`; counterfactual harness and protocol described in the roadmap. | One seed and a limited fixture do not establish robust counterfactual reasoning. Preserve the ablation and expand replication before broad claims. |
| CLM-005 | The observed Doom policy was near-uniform and did not leave MAP01; the documented action-credit and policy fixes resolve that behavior. | **Observed failure; resolution pending**; live-game observation plus preregistered follow-up. | [ASS-003, finding and protocol](../assurance/ass-003-policy-learning.md) | `developmental_ai/core/realtime_credit.py`; `tools/replay_lag_probe.py`; [ASS-003 protocol](../assurance/ass-003-policy-learning.md). | The baseline failure is documented. The fix remains unconfirmed until the protocol’s mechanism, learning, and behavioral gates are reported. |
| CLM-006 | A shared universal brain can accept game-specific observation/action adapters while keeping evaluator-only engine state out of learning inputs. | **Implemented—not behaviorally validated across games**; design and code/integration inspection. | [ARC-004](../architecture/arc-004-universal-games.md) | `developmental_ai/environments/universal.py`; `developmental_ai/environments/wrappers.py`; per-game YAML under `configs/games/`. | Doom integration is documented but host proof and durable cross-game behavior are separate gates. Adapter conformance is not evidence of transfer. |
| CLM-007 | At the tested offline stage, increasing RSSM size did not yield a meaningful predictive advantage sufficient to justify starting large. | **Supported—offline only**; replay/offline data, one seed. | [RES-004, Stage A](../research/res-004-rssm-size-experiment.md#stage-a-results--valid-main-cuda-posterior-collapse-fix-2026-10-17) | `tools/rssm_size_experiment.py`; `developmental_ai/world_model/growth.py`. | The evidence is limited to the stated dataset, steps, and seed. Live behavior A/B remains unstarted; growth actuation is documented as off. |

## 4. Claim record requirements

Add or revise an entry when a result changes a project-level belief, advances
or retracts a roadmap claim, changes the interpretation of a behavioral
failure, or is used to justify a consequential architecture or training
decision. Routine implementation details do not need claim IDs.

Each entry must provide:

1. A concise, falsifiable statement with an explicit scope.
2. A stable claim ID that is never reused.
3. A status and evidence class, including seed count and environment where
   those affect interpretation.
4. A primary protocol or result record, plus relevant implementation paths.
5. The most important limitation and the next evidence gate, or an explicit
   statement that no follow-up is planned.

When a claim is superseded or retracted, keep its ID and history, change its
status, explain why, and link the new claim or evidence. Do not erase negative,
null, inconclusive, or invalidated results. Frozen protocols remain unchanged;
record results and deviations beside them.

## 5. Maintenance

The register is reviewed when a roadmap status changes, a preregistered result
is added, a live run changes the interpretation of a behavioral claim, or a
document path is renamed. The [documentation index](../README.md) defines
document IDs and change control.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ADR-001](../decisions/adr-001-shared-universal-brain.md): One shared brain for universal games
- [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md): Evaluator state is not a learning input
- [GOV-006](gov-006-decision-record-process.md): Decision record process
- [GOV-007](gov-007-contributor-and-change-guide.md): Contributor and change guide
- [ADR-003](../legacy/decisions/adr-003-preserve-legacy-brain-isolation.md): Preserve legacy brain isolation
<!-- backlinks:end -->
