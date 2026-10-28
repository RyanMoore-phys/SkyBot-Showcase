# Predictive-coding core versus the RSSM: comparison protocol

> **Document control:** [ASS-019](../README.md#assurance) · Assurance / pre-registered protocol · **State:** Protocol; pre-registered, frozen when the first WP-20 stage runs; **no result yet** · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-25. **Plan:** [GOV-016](../governance/gov-016-predictive-coding-core-plan.md). **Design and hypotheses:** [RES-010](../research/res-010-predictive-coding-core.md).
**Standard:** [ASS-007](ass-007-adaptability-evaluation-protocol.md) (five seeds of one hour, positive on all five, each above its region of uncertainty; development evidence is not program evidence, section 14.1).
Results are appended below the line at the end, never edited into this protocol.

## 1. Question

At the same size, data and code snapshot, does the predictive-coding latent core (`world_model.core: pc`) learn at least as well as the RSSM, and does it adapt better to unfamiliar maps?

## 2. Two stages, in order

**Stage 1, the pilot gate (`pc_pilot`).** Offline, fixed data, same batches; three runs: `rssm`, `pc` (Hebbian, the gated candidate) and `pc_bp` (the same settling with backprop-trained networks: reported, never gated). Tests H-1 to H-3 (stability, model quality, cost) with the five gates G0 to G4 of [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 5.
It is a gate: a failure blocks stage 2 and is reported as a negative pilot result, not hidden.

**Stage 2, the A/B (`t1_wp20_pc_core`).** Only if stage 1 passes. Two arms in one stage on the one staged snapshot (`pc_code`):

| Arm | Configuration |
|---|---|
| `baseline` | `world_model.core: rssm` (the unchanged model; no overlay) |
| `changed` | overlay `world_model: {core: pc, pc_iters: 4, pc_step: 0.5, pc_weight: 1.0, pc_learning: hebbian, pc_hebbian_lr: 0.01}` |

Paired seeds 1 to 5. Each seed is one hour (source phase on MAP01 to MAP28 with the WP-6 diverse source schedule not applied, as in the other T1s; target exposure on the held-out maps MAP29 to MAP32;
evaluation-only checkpoints), exactly the shape of every other T1 stage. Eight streams, tiny size as the other T1s. Both arms run the same configuration ([ADR-005](../decisions/adr-005-compute-pressure-order.md) Amendment 2).

## 3. Measures

- **Primary (H-4):** the target-phase adaptability measure the existing analysis computes for every T1 (`tools/eval_analyze.py`), per seed, `changed` minus `baseline`.
- **Secondary, reported always:** held-out world-model open-loop error and reconstruction at the end of source and of target phases; z-information; steps per second per stream; acting latency; VRAM; settle refusals and mean energy before and after settling; deaths, kills, items and exits per stream-hour (reported, not gated).
- **Integrity:** no traceback; the refusal counter; no non-finite loss.

## 4. Decision rules (fixed now)

| Outcome | Reading | Next |
|---|---|---|
| Pilot gate fails | the PC core is not worth a five-seed test as built | record which gate; the A/B is not run; follow-ups: tune `pc_iters`/`pc_step`/`pc_weight` as a NEW pre-registered pilot, or retire |
| Gate passes, A/B positive on all five seeds, each above the noise floor | evidence (development class) that PC adapts better | the `pc_iters = 0` ablation, then the owner decides promotion ([ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md) supersession) |
| Gate passes, A/B positive on some seeds only, or inside the noise floor | **no claim**; recorded as inconclusive | the owner decides whether more seeds or a tuned arm is worth a slot |
| Gate passes, A/B null or negative | the RSSM stays the default; `pc` stays selectable or is retired by the owner | recorded honestly |

## 5. Known limits

- The A/A noise floor (AB-00) was measured on the **frozen** code, which lacks the [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md) fixes; the A/B here runs on a snapshot that has them. The floor is an approximation for this comparison. The same-snapshot paired baseline is the primary control; a snapshot A/A would tighten it and is an owner choice.
- One configuration of the PC core is tested (the simplest first arm, not tuned). A negative result is about that arm, not about predictive coding in general.
- The pilot's data are sticky-random play, as in [RES-004](../research/res-004-rssm-size-experiment.md); it says nothing about the policy's behaviour.
- Growth and autoscaler actuation do not support `pc` ([ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md)); the A/B runs with actuation off, as the other T1s.
- The Hebbian rule covers the latent core's three networks only; the encoder, GRU, decoder and heads still use backpropagation. A positive result is about this mixed arrangement, not about a fully local world model.
- The `pc_bp` ablation has the Hebbian core's three networks (same topology and initial values) but trains them with Adam; it separates the learning rule from iterative inference, not the optimiser from the rule.
- The A/B changes iterative inference and local learning together. The pilot's `pc_bp` run is the only separation of the two; a positive A/B would be followed by a `pc_bp` A/B if the owner wants the attribution.

## 5a. Revisions before the first run

Changes after results exist must be disclosed (docs/README.md section 3); none exist. Before any WP-20 stage ran: the learning rule became local ([GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11a), and the audit corrections of
[ASS-020](ass-020-recent-changes-static-audit.md) were applied (section 11b there): the open-loop evaluation is now causally aligned (F-02), the ablation has the same network topology (F-05), the required suites gate the pilot (F-01), and a measurement no longer moves the precision (F-04).
The gate's numbers did not change.

## 5b. First run (2026-10-26): no result yet

The chain started on 2026-10-26. The required suite `pc_tests` failed (contract G of `_predictive_coding_smoke`: the head's error rose, 0.0379 to 1.5309, over 41 steps on a fixed batch), so by the rule in section 2 the boot check and the pilot are blocked. **No pilot number exists and no section 4 decision rule has been applied.**
Nothing in sections 2-4 has changed. If the cause is a defect in the contract rather than in the rule, the change to the contract will be listed here with the failing numbers and the date, as a revision made after a failure ([GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11c).

### 5b.1 Diagnostic and revision (2026-10-26)

The diagnostic `pc_hebbian_diag` found the first failure was in the contract, not the rule: the embedding the head predicts grew from 0.0016 to 75.8 in mean square, the error relative to it fell 23.7 to 0.020, and the backprop ablation's raw error rose as well. Contract G now judges the relative error. Sections 2-4 and the gate G0-G4 are unchanged; details and the caution are in [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11c.1. The pilot has still not run.

### 5c. Pilot result (2026-10-27): the gate failed

`pc_pilot` ran once at the registered setting and **failed G2, G3 and G4** (open-loop relative error 1.213 against the RSSM's 0.822; acting time 4.78 x). G0 and G1 passed; the backprop ablation reached 0.873. Sections 2-4 and the gate were not changed. The five-seed A/B (`t1_wp20_pc_core`) is blocked and has produced no data. Numbers, the reading and the rule for a second attempt are in [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11d.

### 5d. Second pilot attempt registered (2026-10-27)

Arms, hypotheses, predictions and the attempt count are written in [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11e before the run. The gate of section 5 is unchanged. The first result (5c) stays on record; the registered arms count as attempts 2 and 3.

### 5e. Second pilot result (2026-10-27): no registered arm passed

`pc_pilot2`: p2a failed G3 (reconstruction 2.2 x the RSSM's) and G4 (acting 4.85 x); p2b failed G3 (2.1 x) and G4 (acting 3.17 x, limit 3). Attempt count 3. The gate was not changed. Diagnostic findings and the reading against the predictions are in [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11e.1; a diagnostic-only run of an optional target normalisation (section 11f) is queued and gates nothing.

## 6. Where results go

`stages/pc_pilot/verdict.json` and `results.json`, `stages/t1_wp20_pc_core/`, and the analysis record written by `an_t1_wp20_pc_core`; a results section is appended to this document and to [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 10 onward.

## 7. Related records

[GOV-016](../governance/gov-016-predictive-coding-core-plan.md), [RES-010](../research/res-010-predictive-coding-core.md), [ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md),
[ASS-007](ass-007-adaptability-evaluation-protocol.md), [ASS-013](ass-013-unattended-test-campaign.md), [ASS-016](ass-016-ab00-final-and-fresh-control.md).
