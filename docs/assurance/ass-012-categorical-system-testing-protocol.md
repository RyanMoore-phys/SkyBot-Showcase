# Categorical system testing protocol

> **Document control:** [ASS-012](../README.md#assurance) · Assurance / integration testing protocol · **State:** Draft template; harness built, first run recorded in ASS-014; no category run frozen under it · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21 (section 8 added 2026-10-22). **Plan:**
[GOV-013](../governance/gov-013-categorical-system-testing-plan.md).
**Scope confirmed by owner:** coherence both within functional learning groups and
between groups. This is a testing procedure, not a result or a promotion decision.

## 1. Hypothesis and operational meaning

For a named, versioned composition, every required consumer receives the intended
learner-side signal with the right identity, timing, units and validity. Shared
state and coefficient ownership remain consistent. Invalid evidence and failures
produce the specified fallback and recovery within a declared bound. Resource use
stays inside the frozen limits while the required paths are exercised.

This hypothesis is falsified by a violated required contract, even if aggregate
reward rises. A connection can be implemented correctly and still be unhelpful;
learning benefit is decided by [ASS-007](ass-007-adaptability-evaluation-protocol.md).

## 2. Preregistration for each category or boundary

Create an immutable manifest and a separate result record in an isolated
`/workspace/experiments/categorical_<id>_<date>/` root on `main`. At minimum record:

| Field | Required content |
|---|---|
| Identity | Run ID, category/boundary IDs, date, operator, hypothesis and evidence class |
| Provenance | Commit plus dirty-source manifest and hashes, exact composed configuration, environment/dependency versions, fixture hashes |
| Scope | WP/RW members, original map/line IDs, implementation dispositions, required versus optional scenarios, omissions with reasons |
| Interface | Producer, consumer, tensor shape/units, stream ID, action alignment, model version, sample validity/age, actual phase/call order |
| Controls | Enabled modules/relations, minimal prerequisites, cut configuration, neutral values, expected state differences |
| State | Fresh/resumed status, checkpoint/replay hashes, restoration/reset rules, permitted parameter and optimizer writers |
| Inputs | Fixture cases and seeds; recorded trace provenance; Doom maps/split for live runs; no evaluator input to learner |
| Oracles | Expected causal effect or invariant, tolerances, required exercise counts, response/recovery deadlines and fault detector |
| Resources | RAM/VRAM peaks, latency p50/p95/p99, missed-decision rate, throughput floor, disk budget, measurement cadence and stop rules |
| Analysis | Exact pass/fail rules, exclusions, missing-data handling, repeat/retry rules and result artifacts |

No blank required threshold is a runnable protocol. Estimate thresholds in a
separate disclosed sizing pilot; freeze them before acceptance runs. Never choose
them after viewing acceptance results. Use the same resource configuration for
paired controls (ADR-005), and record the cost of any capacity reduction separately.

## 3. Within-category contract matrix

All cases use actual module wiring where it exists. A fixture may inject numeric
signals, delayed messages or faults into a test double; it must never supply
scripted skills or task meanings to a production learner. A mock-only pass is
labelled as such until the corresponding real-module path is exercised.

| ID | Scenario and intervention | Required observation / failure condition |
|---|---|---|
| CT-01 | Normal causal flow: perturb one producer, hold other inputs/state fixed | Inspect the consumer's actual input/update; correct sign, shape and declared delay. A matching log counter alone is insufficient. |
| CT-02 | Cut/neutral parity: disable each required channel, then the group | Neutral behavior and unrelated state preserved. For pure deterministic fixtures require exact parity; freeze numerical tolerances for floating/stochastic paths. Restore RNG/state before comparisons. |
| CT-03 | Missing/invalid/stale source: no sample, NaN, infinity, constant evidence, old model version or wrong stream | Explicit refusal/fallback and no invalid state update. Distinguish a legitimate zero from absence; do not require all channels to be nonzero. |
| CT-04 | Competing controls: exercise all enabled coefficient writers in real phase order | One declared owner or explicit factor combiner per coefficient; bounded writes; no silent last-writer overwrite, duplicate reward booking or unexplained update. |
| CT-05 | Temporal/state coherence: episode end, delayed executed action, stream interleave, latent-width mismatch | No unintended history/credit leakage across streams or episode boundaries; correct action-to-observation alignment; shape mismatch refused or migrated by a declared supported path. |
| CT-06 | Failure and recovery: force a dependency failure, then restore valid input | Safe fallback and recovery by the manifest's deadline; no permanent closed gate or stranded coefficient. Verify the error is observable. |
| CT-07 | Persistence: save, recreate modules, restore, continue the same fixture | Required state restored with equivalent behavior within tolerance; intentionally reset state follows its documented warmup. No claim of persistence for unsaved optimizers/calibration windows. |
| CT-08 | Isolation and non-interference: attempt evaluator-field injection; change logging frequency | Evaluator input rejected; enabling read-only probes does not change learner state, reward or RNG. Trace real entry points; registration tags alone are insufficient. |

For every required path, specify a deliberate break that the test must detect
(for example, wrong stream ID, omitted consumer call or duplicated coefficient
write). Apply breaks only to the isolated harness or disposable copy. Report
unexercised paths as inconclusive. Fault injection must never target an active
training/evaluation process or exhaust the host's physical RAM/VRAM.

## 4. Between-category scenarios

The following is the initial required boundary inventory. CS-0 must add other
implemented cross-group data/control paths found in the named composition; it
must not assume this table exhausts the code. Every boundary gets an interface
contract, CT-01/02/03/06/08, and state/persistence checks where relevant.

| ID | Groups | End-to-end scenario |
|---|---|---|
| BND-01 | CAT-01 -> CAT-03 | Prediction/ensemble evidence reaches planning; absent, constant or invalid evidence cannot become trusted planning guidance. |
| BND-02 | CAT-03 -> CAT-04 | Current confidence validity governs modulation; evidence loss releases the actual exploration/consolidation/attention coefficients. |
| BND-03 | CAT-02 -> CAT-03 | Retrieval and attention features have the expected identity and width at policy/planner entry; a path that uses zero knowledge context must be declared, not reported as memory-aware planning. |
| BND-04 | CAT-04 <-> CAT-05 | Competence, practice selection and adaptive optimization preserve single ownership and avoid self-amplifying or stranded coefficients under fixed inputs. |
| BND-05 | CAT-06 -> CAT-01/CAT-02 | Peer/diversity experience retains source identity through replay and memory; held-out maps and evaluator fields cannot leak into source training. |
| BND-06 | CAT-02 <-> CAT-05 | Consolidation, replay selection and optimizer updates occur in declared order; updates are attributed and unrelated parameters remain unchanged. |
| BND-07 | CAT-07 <-> every enabled group | Checkpoint/restart, instrumentation and simulated resource pressure preserve each group's state contract, fallback and frozen configuration. |

Apply supported controller combinations together, not just one edge at a time.
At minimum include confidence modulation + attention + consolidation, and
curriculum + adaptive optimization + strategy selection. Inventory absent
prerequisites first; mark unavailable combinations blocked rather than inventing
a production connection. Measure bounded coefficients and release under constant
and changing inputs. For suspected interaction defects, isolate components and
then reassemble the failing combination; retain the original failure trace.

## 5. Operational procedure on main

**Scheduling (2026-10-22).** The host-side queue that runs this campaign, with the other tests, unattended is [ASS-013](ass-013-unattended-test-campaign.md) (`tools/campaign.py`). Its categorical stages are blocked until the CS-1/CS-2 harness (`tests/_categorical_*_smoke.py`) exists, and `cat_thresholds_frozen` is a human gate.


1. Confirm no A/A/A/B or another timed behavioral job is active. Serialize tests;
   preserve all frozen snapshots. Stage a separate source/configuration copy and
   freeze the manifest. Verify host identity and use the approved SSH identity.
2. Run the deterministic contracts and boundary harness. Require the deliberate
   breaks to be detected. A failed required check blocks live execution of that
   composition; a deferred interface permits only an explicitly narrower scope.
3. Run the independent sizing pilot if limits are unknown, then freeze the live
   protocol. Do not count sizing data as acceptance evidence.
4. Run the 20-minute baseline and category-enabled Doom operational pilots in
   separate fresh roots. Keep maps, model size, streams, seeds, startup policy and
   measurement cadence fixed. Use source MAP01-MAP28 only; held-out MAP29-MAP32
   stay reserved for protocols that require the target phase. Fix pilot seed 100
   for both arms. These short paired observations are not statistical benefit tests.
5. Inspect consumer effects, exercise counts, failures, resources and state. A
   successful launch or unused feature is not an exercised integration. Add fixture
   coverage where rare pathways did not occur; do not label it live evidence.
6. Once the declared groups and boundaries pass, run the combined operational
   composition for seeds 101-103, one hour each, serialized, 40 source minutes and
   20 held-out Doom minutes per ASS-007. Record all checkpoints evaluator-side.
   Score no learning-benefit claim from these operational seeds.
7. Run controlled recovery scenarios in separate roots. A requested graceful stop
   and resume is not a crash; simulate process-loss cases separately, never by
   killing the active A/A/A/B driver. Preserve both pre-fault and post-fault state.
8. Archive manifests, commands, logs, traces, resource series, state hashes and
   verdicts. Hand the eligible composition to the existing behavioral protocols.

The normal source/target split and meaning of a one-hour **behavioral** seed remain
unchanged. Recovery, sizing and short contract runs are separate engineering
evidence with their own wall-clock budgets. No Minecraft development tests are
introduced by this protocol.

## 6. Acceptance, resource limits and failures

**Mechanism acceptance:** every required applicable contract passes, the exercised
path count meets the frozen minimum, the deliberate break is detected, and no
unexplained learner/evaluator leak or coefficient/state corruption occurs. There
is no averaged score that can cancel a failed required invariant.

**Resources:** report peaks and distributions, not a single `nvidia-smi` sample.
Freeze minimum available RAM, maximum VRAM/RSS, p95/p99 action latency, missed
decision rate, throughput floor and checkpoint/disk allowance. Express latency
against the configured decision interval using ADR-005's budget, with the exact
sampling denominator declared. If the baseline already fails, record that and
resize/rebaseline before acceptance testing; do not normalize an existing failure
into a pass. Sample at least once per second for system resources and retain
per-decision timing where available; report instrumentation overhead separately.

**Release bound:** for a slew-limited coefficient with step `s > 0`, use
`ceil(abs(value - neutral) / s)` consumer updates plus documented scheduling delay.
Freeze bounds for other state machines explicitly. Intentionally paused updates
do not count as successful recovery. Missing telemetry yields inconclusive
resource/flow evidence, not a zero or a pass.

**Failure handling:** stop the affected isolated run gracefully on a required
invariant failure, non-finite learner update, or resource-limit breach. Preserve
all artifacts. An unplanned crash gets at most one same-manifest retry, retained
alongside the first attempt; repeat failure is fail. Unrelated host interruption
is inconclusive with cause recorded. A repair starts a new manifest/revision and
re-runs affected scenarios. Never discard a bad seed to make the group pass.

## 7. Result record template and claim limits

Record results separately from this protocol, using the repository document-control
format and a newly registered ID for an authored result. Generated experiment
artifacts keep their run IDs. Minimum report:

```text
Run ID / category / boundaries / manifest hash:
Evidence class: fixture | replay trace | Doom operational
Implemented subset / absent interfaces / deliberate exclusions:
Hypothesis, controls, commands and actual start/end times:
Contract ID | required? | exercised count | oracle | observation | verdict
Deliberate-break sensitivity and neutral-control result:
Consumer traces, state hashes, controller ownership and release timing:
Peak RAM/VRAM, latency distributions, missed decisions, throughput:
Failures, retries, deviations, host interference and missing measurements:
Verdict: pass for declared scope | fail | inconclusive | blocked/deferred
Limits of claim / eligible next gate / artifact paths:
```

Do not convert correlated streams or repeated segments into independent seeds.
Category and boundary results can overlap; do not count them as independent
replication. Retention, cross-environment adaptability and beneficial interaction
claims still require the appropriate ASS-007/T1/T2 or separate long-run protocol.
Leave-one-out behavioral tests remain required for the final combined system.

## 8. The harness (CS-0 to CS-2, built 2026-10-22)

The deterministic layer of this protocol exists as code. Results of its first run are in
[ASS-014](ass-014-categorical-harness-first-run-findings.md).

| File | What it is |
|---|---|
| `tests/categorical_lib.py` | The shared `Harness`. Not a test; it enforces the rules below |
| `tests/categorical_manifest.py` | CS-0: each category's blocks, relation clusters, module files, switches and **disposition** (implemented, partial, observation-only, excluded), with the boundary table |
| `tests/_categorical_inventory_smoke.py` | Checks the manifest against the code: files exist, switches exist in `configs/universal.yaml`, every wired channel is claimed by exactly one category |
| `tests/_categorical_cat01..cat07_smoke.py` | CT-01 to CT-08 for each group, on the real module objects |
| `tests/_categorical_boundaries_smoke.py` | BND-01 to BND-07 plus the two controller combinations of section 4, composed in the loop's real phase order, and extra cross-group paths found in the code |

**Run:** `PYTHONPATH=. ./venv/bin/python tests/run_all.py categorical` (about 70 s, CPU, no game, no GPU, no
trainer), or any one file. `CATEGORICAL_OUT=<dir>` also writes one JSON record per file (contract id, required,
exercised count, oracle, observation, verdict: the section 7 template). `tests/run_all.py categorical` is not
part of `all` and reports "no categorical tests exist" with exit 2 if the directory is empty. The campaign
(ASS-013) runs it as `cat_harness`.

**Rules the harness enforces, so a test file cannot weaken them:**
1. Every scenario runs **twice**: as built (must pass and meet its minimum exercised count) and with a named
   **deliberate break** injected into the harness or a disposable copy, which must be **detected**. A break that
   goes undetected fails the scenario.
2. A scenario that exercised fewer real paths than required is inconclusive and fails; it is never a pass.
3. A scenario whose consumer does not exist is declared `blocked` with a reason verified in code. The file's
   verdict is then "pass for declared scope", never "pass". A blocked scenario is neither pass nor failure.
4. Fixtures inject numbers, delays and faults into the real module objects; they supply no scripted skill or
   task meaning, read nothing from the evaluator package, and never touch a trainer or the host's RAM/VRAM.
5. A failing scenario is left failing and reported with file and line. Production code is not changed by a
   test author, and a test is not loosened to hide a defect.

**What it does not cover (stay open):** the operational Doom pilots, the combined run, the sizing pilot and the
section 5.7 process-loss scenarios are campaign stages, not fixtures; the threshold freeze is a human gate. A
passing fixture is not evidence that the real loop composes the modules the same way (the loop was exercised
through its own phase helpers where possible). Contract ids repeat where one id carries several members; the
JSON keeps them separate.

**Adding a test or a group:** add the module to `categorical_manifest.py` first (the inventory test then
requires its switches and files to exist), write the scenario with a named break, and run the tier. The
campaign picks the new file up on its next `cat_harness` run.

## 9. Related records

[GOV-009](../governance/gov-009-implementation-plan.md),
[GOV-010](../governance/gov-010-relational-implementation-plan.md),
[GOV-013](../governance/gov-013-categorical-system-testing-plan.md),
[ARC-006](../architecture/arc-006-relation-registry.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-005](ass-005-testing-plan.md): Testing plan — proving the new machinery works, and keeps working
- [ASS-017](ass-017-host-test-stages-and-queue-incidents.md): Host test stages and two queue incidents (2026-10-23)
- [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md): Corrective plan after the AB-00 interim analysis: source diversity, intrinsic-signal repairs and a fresh control
- [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md): Corrective plan for the categorical-harness findings and the host contract-tier failures
<!-- backlinks:end -->
