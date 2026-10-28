# Categorical system testing: coherence within and between learning groups

> **Document control:** [GOV-013](../README.md#governance) · Governance / testing plan · **State:** Proposed execution plan; category scope confirmed by owner · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Parents:** [GOV-009](gov-009-implementation-plan.md)
and [GOV-010](gov-010-relational-implementation-plan.md).
**Procedure:** [ASS-012](../assurance/ass-012-categorical-system-testing-protocol.md).
**Authority:** the owner requested categorical system testing and confirmed testing
both within functional groups and between groups. This document plans that work;
it does not claim a suite exists, has passed, or is scheduled on the host.

## 1. Purpose and evidence boundary

Test whether related learning mechanisms work coherently as a system: the right
information reaches the right consumer, state stays consistent across updates and
restarts, controllers obey ownership and bounds, and failures release safely.
Then test whether those properties survive interactions between groups.

| Existing or added evidence | Question it answers |
|---|---|
| A/A, ASS-007 | How variable is the frozen baseline and measurement procedure? |
| T1 block A/B, GOV-009 | Does a block improve learning? |
| T2 relationship A/B, GOV-010 | Does a connection improve learning given its blocks? |
| **Categorical system tests, this plan** | Do the assembled groups behave consistently under normal operation, conflicts, missing data and recovery? |
| Combined A/B and leave-one-out, ASS-007 | Do measured benefits survive the final combination? |

Categorical testing is an additional integration layer in P5/P6 and before final
combined evaluation. It does not replace T1/T2, relax the five-seed rule, or
establish human-like function from a biological name. A group is an engineering
boundary, not an extra learner: ADR-001's single shared brain remains in force.

## 2. Categories and traceability

These functional groups overlay the owner's map; they do not rename its nodes or
change its edges. Map sections below refer to
[RES-008](../research/res-008-relationship-gap-analysis.md). Shared mechanisms can
belong to several groups; one result is referenced wherever relevant, not counted
as independent evidence more than once. WP/RW membership defines intended coverage,
not implemented completeness.

| ID | Functional group | Blocks and relationships in scope | Map sections | Concrete coherence question |
|---|---|---|---|---|
| CAT-01 | Perception, representation and prediction | Existing encoder/RSSM/replay; WP-19; WP-4 ensemble input; RW-8, relevant RW-15 probes | 1, 3, 5, 7, 10 | Does each prediction use the correct observation, executed action, stream and model version? |
| CAT-02 | Memory, attention and consolidation | WP-18, WP-7, WP-2; RW-2, RW-3, RW-4 | 4, 5, 6, 9 | Can storage, retrieval, attention and consolidation coexist without mixing streams, stale shapes or unintended policy writes? |
| CAT-03 | Planning, uncertainty and risk | WP-4, WP-11; RW-5, RW-10 | 3, 5, 11 | Does valid model evidence govern planning, and does absent evidence restore the documented fallback? |
| CAT-04 | Motivation, exploration and skills | WP-5, WP-1, WP-8; RW-1, RW-7, RW-11 | 2, 7, 8, 12 | Are intrinsic income, competence, skill discovery and control modulation consistent without double payment or false confidence? |
| CAT-05 | Curriculum and learning adaptation | WP-3, WP-9, WP-13; RW-6, RW-12 | 1, 5, 7, 9, 11 | Do practice selection and optimizer controllers change only the values they own and preserve attribution of updates? |
| CAT-06 | Experience diversity, peers and transfer | WP-6, WP-12, WP-16, WP-19; RW-8, RW-9 | 5, 9, 10, 12 | Does sharing experience preserve provenance, evaluation partitions and the shared learner's input contract? |
| CAT-07 | Runtime, resources and evaluation isolation | WP-10, WP-14, WP-15, WP-17; RW-13, RW-14, RW-15 | 1, 4, 10, 11 | Do budgets, persistence and telemetry preserve the learner/evaluator boundary through faults and restarts? |

The map's central-node/descriptive links have no artificial executable contract.
Record them separately, with the exclusions in RES-008/RES-009 preserved. Runtime
traces must use original RES-008 line IDs and `rel.<FROM>.<TO>` channel names from
[ARC-006](../architecture/arc-006-relation-registry.md), not just category labels.

## 3. Readiness and incomplete interfaces

Before a category is instantiated, list every member as **implemented**, **partial**,
**observation-only**, **deferred**, or **excluded** using GOV-011/GOV-012 and targeted
code inspection. Record the exact switches and prerequisites. An unavailable
required consumer blocks that scenario; it must not silently become a passing
zero-valued probe. A partial category may pass its named implemented subset, but
cannot be reported as fully integrated.

Known constraints to preserve in that inventory:

- WP-4 currently plans during learning; acting-time search is deferred. Its
  action-predictability calibration is not competence or outcome calibration.
- RW-4 semantic storage has no policy retrieval consumer; storage checks do not
  establish semantic knowledge improving actions.
- Several RW-8/RW-9/RW-11/RW-14 lines are diagnostics only. Check observation
  integrity without claiming a downstream behavioral effect.
- Peer auxiliary training does not currently train the shared encoder; full
  causal discovery, richer transfer and learning-stage mechanisms remain deferred.
- New behaviors remain default OFF. Test overlays enable an explicit subset only.

## 4. Work packages and execution sequence

| Package | Deliverable | Completion condition |
|---|---|---|
| CS-0 Inventory (BUILT 2026-10-22: `tests/categorical_manifest.py` + inventory test) | Category manifests mapping WP/RW, line IDs, switches, producer/consumer sites, state, cadence and limitations | Every required member has an explicit implementation/readiness disposition |
| CS-1 Within-group harness (BUILT 2026-10-22: `tests/_categorical_cat01..07_smoke.py`, ASS-012 section 8) | Standalone contract scenarios for ASS-012 CT-01 through CT-08, using real module composition where available | Applicable contracts pass and deliberate broken connections are detected |
| CS-2 Boundary harness (BUILT 2026-10-22: `tests/_categorical_boundaries_smoke.py`) | ASS-012 boundary matrix using the same component instances and realistic phase ordering | Each available required boundary passes; absent interfaces are blocked/deferred explicitly |
| CS-3 Doom operational pilots | Serialized fixed-duration runs on `main`, isolated roots, frozen overlays and resource measurements | Required flows were actually exercised and limits met; inactivity is inconclusive |
| CS-4 Combined operational run | One declared composition with state/recovery and resource checks | No unresolved required contract failure; report partial coverage accurately |
| CS-5 Behavioral handoff | Results linked to existing T1/T2 and final combined A/B protocols | Integration readiness is established without treating it as promotion evidence |

Start with CAT-03 and its CAT-04 boundary: the confidence-integrity checks in
GOV-011 section 7 provide a narrow existing example. Then CAT-01, CAT-02,
CAT-04, CAT-05, CAT-06 and CAT-07, respecting actual prerequisites. CAT-07's
evaluation-isolation and resource prerequisites apply before **every** live pilot,
even though its broader recovery campaign comes later.

For each category: inventory -> deterministic composition -> boundary scenarios
-> operational pilot. Build and inspect each related block before enabling it.
Use fresh isolated state; no experimental switches or edits enter an active
A/A/A/B snapshot. Re-run affected contracts after a fix; retain the failed record.

## 5. Host scheduling and budget

**Update 2026-10-22.** An executable host queue now exists: [ASS-013](../assurance/ass-013-unattended-test-campaign.md) (`tools/campaign.py`). It schedules the categorical stages after the A/A, keeps them blocked until the CS-1/CS-2 harness exists, and holds the ASS-012 threshold freeze as a human gate. The statements below that no reservation exists were true when written.

All tests, including fixtures and documentation checks, execute on `main`.
Real-game development runs use Doom; later Minecraft program evaluation remains
governed by ASS-007. Synthetic fixtures are isolated mechanism tests, not alternate
game benchmarks or learning data for a behavioral run.

Finish the active frozen A/A queue before running this campaign. Do not run
category tests concurrently with timed A/A/A/B arms. This request adds documents
and a testing plan, not an executable host reservation or a second training job.

Proposed initial operational allocation: one 20-minute baseline pilot and one
20-minute category-enabled pilot per eligible category, serialized. Across seven
categories this is **4 hours 40 minutes**, excluding startup, analysis and retries.
These are engineering observations, not an A/B success claim. Comparable pilot
results may be reused only with identical code/configuration/state provenance,
resource envelope and observation procedure; log every reuse.

After the category pilots, propose three one-hour combined operational seeds
(101, 102, 103; separate from ASS-007's evaluation seeds), with the normal 40/20
Doom partition, for **3 more host-hours**. This supplies operational evidence only.
It cannot establish long-term retention, growth stability or learning benefit.
Long-run questions require a separate frozen duration and budget.

Thus initial live operational allocation is **7 hours 40 minutes** if all seven
categories are ready. Harness construction, fixture execution, unresolved
interfaces, fault-recovery scenarios and behavioral A/Bs are additional work, not
hidden inside that estimate. Freeze run-specific resource thresholds after an
independent sizing pilot and before collecting acceptance evidence (ASS-012).

## 6. Decision and maintenance

Record category results as **pass for declared scope**, **fail**, **inconclusive**,
or **blocked/deferred**. No single averaged coherence score can conceal a failing
required contract. A test cannot pass merely because nothing crashed.

A failed integration blocks promotion of that combined configuration while the
cause is diagnosed. Existing individual evidence remains historical evidence,
conditional on its original configuration. A fix changes the snapshot and invokes
the appropriate P5/P6 and, if behavior changes, T1/T2 requirements. Never tune on
held-out evaluator checkpoints or feed them into the learner.

## 7. Related records

[ASS-012](../assurance/ass-012-categorical-system-testing-protocol.md),
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md),
[GOV-011](gov-011-execution-record-gov-009.md),
[GOV-012](gov-012-execution-record-gov-010.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-014](../assurance/ass-014-categorical-harness-first-run-findings.md): Categorical harness: first-run findings
- [GOV-014](gov-014-corrective-plan-after-ab00.md): Corrective plan after the AB-00 interim analysis: source diversity, intrinsic-signal repairs and a fresh control
- [GOV-015](gov-015-corrective-plan-categorical-findings.md): Corrective plan for the categorical-harness findings and the host contract-tier failures
<!-- backlinks:end -->
