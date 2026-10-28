# Corrective plan after the AB-00 interim analysis: source diversity, intrinsic-signal repairs and a fresh control

> **Document control:** [GOV-014](../README.md#governance) · Governance / corrective implementation plan · **State:** Approved by owner 2026-10-22; documents revised before any change; execution recorded in section 10 · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-22. **Parents:** [GOV-009](gov-009-implementation-plan.md) (blocks and gates P0 to P10),
[GOV-013](gov-013-categorical-system-testing-plan.md).
**Evidence it answers:** [ASS-015](../assurance/ass-015-ab00-interim-analysis.md) (16 finished AB-00 runs) and
[ASS-014](../assurance/ass-014-categorical-harness-first-run-findings.md) (28 findings).
**Authority:** the owner, 2026-10-22: pursue ASS-015 recommendations 3 (source diversity), 4 (intrinsic-signal
repairs) and 2 (fresh control); ignore 1 (metric change) and 5 (random-policy reference); revise the plan and
documents before any change; revise or add tests the changes depend on; make the documents reference each
other. This record carries that out. It makes no behavioural claim.

## 1. Scope and dispositions

| ASS-015 recommendation | Disposition | Section |
|---|---|---|
| 1. Rate-based Doom metric | **Not pursued** (owner). Consequence in section 9 | 9 |
| 2. Fresh-control arm | **Pursued** (item A) | 3 |
| 3. Diverse source phase (WP-6) | **Pursued** (item C) | 5 |
| 4. Fix the inverted guard (F-13) and consolidation strength (F-04) | **Pursued** (item B) | 4 |
| 5. Random-policy reference | **Not pursued** (owner) | 9 |

Everything else in ASS-014 stays open and unchanged (section 9), including F-24, which the queue now guards
against (section 6).

## 2. Constraints that bind every item (CLAUDE.md section 0; GOV-009 P4)

- **Learned, not declared.** Nothing here names a goal, a skill, or a meaning to the learner. Item B repairs
  signals built only from the world model's own surprise. Item C varies *which experience* the learner gets.
  Item A is a measurement.
- **One universal brain, one learning process.** No per-game branch. The map rotation of item C is configuration
  of one game's environment through the existing WP-6 keys; the same keys exist for any game that has levels.
- **Default-OFF parity.** Each change must leave a default configuration bit-identical in behaviour, so the
  running AB-00 (older code) and the baseline of every later A/B stay comparable. Section 6 gives the test.
- **Evaluator isolation (ADR-002).** No evaluator value reaches the learner; the coverage check of item C reads
  the log after the fact.

## 3. Item A: fresh control (recommendation 2)

**Question.** Does source training change what the agent does on held-out maps at all? ASS-007 section 4 defines a
*fresh control* (a brain with no source phase, run in the target game) "to show how much of the result is
transfer rather than learning from scratch", reported and not gated. None was ever queued.

**Design.** One arm, `fresh_control`, seeds 11 to 20 (the AB-00 seeds), target phase only (20 minutes, held-out
MAP29 to MAP32, the same configuration, the same code snapshot family). `--fresh-control fresh_control` makes the
runner plan a target-only run and `eval_run.sh` build a fresh root. Ten runs of 20 minutes: **3.3 host-hours**.
Compared with the 20 source-trained target phases of AB-00 (arms A and B pooled).

**Reading.** Development evidence only (ASS-007 14.1). Because the checkpoint scorer is empty for source-trained
runs (ASS-015 section 4: first kill and first pickup are held at entry), the comparison is *descriptive*: the
scorer's output for both groups plus the evaluator counts per run (kills, items, deaths, exits), read by hand in a
result record. A fresh run has no entry state, so the scorer will assign it attainment times where source-trained
runs have none; that asymmetry is reported, not hidden. **No promotion or benefit claim is drawn from item A.**

**Kill criterion.** None (it is a measurement). If a fresh run crashes twice the crash rule applies unchanged.

**Files.** `tools/campaign.py` (stage `fresh_control`, analysis `an_fresh_control`). No learning code.

## 4. Item B: intrinsic-signal repairs (recommendation 4)

### B1. F-13: the guard clock is read backwards by two consumers

**Evidence.** `curiosity/action_surprise.py`: `GuardClock.update` returns True for a row *within `k_guard` steps of a
life boundary*, and `StepSurprise.guard` is documented "True = boundary: ignore row".
`core/relations_attention.py:136` reads it that way. `curiosity/intrinsic_balance.py:73` (`InfoGainTerm.pay`) and
`skill_bank/effect_discovery.py:98` (`SurpriseAgencyTrigger.step`) treat True as *eligible*. Effect: information
gain is paid only just after a boundary and the WP-1 trigger fires on no mid-life spike (ASS-014 F-13; cat04 CT-05).

**Change.** Invert the two consumer tests (skip a row when `guard` is True), with a comment naming the
documented polarity. The producer and its docstring are correct and are not changed. **Why the consumers, not the
producer:** the producer, its docstring, the field comment and the third reader already agree.

**Parity.** `curiosity.info_gain.weight` is 0 and `goals.discovery.enabled` is false by default, so neither
consumer runs in a default configuration. The change is invisible until a package switch is turned on.

### B2. F-04: consolidation strength is cancelled by Adam

**Evidence.** `policy/consolidation.py` multiplies the distillation and EWC terms by `Consolidator.strength` and
then steps an Adam optimizer, whose update is invariant to a constant loss scale. With AWR off, update norms for
strength 0.25, 1 and 4 were equal to 0.001% (ASS-014 F-04). The RW-1 to consolidation chain (and RW-3's four
factors) therefore cannot change the update.

**Change.** Keep the loss weighting (it still sets the balance of retention terms against AWR) and *also* scale the
optimizer's learning rate by `strength` before each step: `lr = base_lr x strength`, where `base_lr` is captured
**once when the optimizer is created** (not lazily from the live param group, which is the F-19 pattern). The
bounded range of the strength coefficient (0.25 to 4) bounds the step.

**Parity.** `strength` is 1.0 unless RW-1/RW-3 are on, and `lr x 1.0` is the same float: a default or
strength-1 run is unchanged. The test (section 6) compares a strength-1 run against the previous formula.

### B3. Verification and what B enables (pre-registered)

- **Prediction P-B1.** After B1, cat04's `CT-05 DEFECT D-GUARD` scenario flips from FAIL to PASS and no other
  categorical scenario changes verdict because of B1.
- **Prediction P-B2.** After B2, cat02's `CT-01 strength changes the consolidation UPDATE` scenario flips to PASS and
  update norms rise monotonically with strength, the 4x to 0.25x ratio exceeding 4.
- **If a prediction fails** the change is wrong or incomplete; it is not "adjusted" to pass (CLAUDE.md section 5).
- **Outcome (2026-10-22).** P-B2 held as written. P-B1 held for the scenario that flips but **was incomplete about
  fixtures**: after the fix, two more cat04 scenarios and the default of the cat04 helper failed, because they
  had been written against the old polarity (`[True] * B  # polarity: the consumer's`). That is the defect baked
  into the test, not a second defect in the change; three fixture lines were corrected and the result is exactly
  the predicted one (below). The categorical tier now fails 39 scenarios (was 41): the two targets flipped, no
  other verdict changed.
- **Enables:** `t1_wp5_info_gain` (and WP-1 and RW-7, which read the same clock) now test the term as designed;
  `t2_rw1_modulation` and `t2_rw3_consolidation` can show an effect through the strength chain. All run on the
  *new* snapshot, which is why `ab_baseline` also runs after the change.

## 5. Item C: a source phase that is actually diverse (recommendation 3)

**Evidence.** The source config lists MAP01 to MAP28, but the Doom adapter plays `maps[map_i]` and advances only
on a level exit. Nobody exits, so all 16 AB-00 source phases stayed on MAP01 (ASS-015 section 1, point 4), and the
world model's fresh-to-train loss gap grew from 0.95 to 1.75.

**Mechanism that already exists (WP-6, GOV-011).** `game.map_assignment` with a `schedule`: at an episode start the
adapter takes the latest schedule phase whose `at_s` has passed, switches the stream's list, and starts it at its
first map. `configure_stream` raises `LeakError` if any listed map is held out. What was missing: the runner could
only merge an overlay into `universal.yaml`, not into the game YAML, so no arm could set the key (the reason
`t1_wp6_diversity` was blocked).

**Change.**
1. `tools/eval_run.sh`: an overlay may carry a reserved top-level key `_games: {<game>: {game: {...}}}`, deep-merged
   into `configs/games/<game>.yaml` after the held-out maps are set, and removed before the rest is merged into
   `universal.yaml`. An overlay without the key behaves as before.
2. `tools/campaign_arms.py`: the `wp6_diversity` arm is generated from `configs/games/doom.yaml`: the source maps
   (held-out removed) are rotated across 8 streams in 300 s phases so that, over the 40-minute source phase,
   64 stream-slots cover all 28 maps two to three times each. Each phase assigns one starting map per stream
   (`per_stream: {k: [map]}`); a stream never gets more than it can reach. A second, *probe* overlay uses 20 s phases
   for a 6-minute coverage check.
3. `tools/map_coverage_check.py` and a campaign stage `wp6_coverage_check`: run the probe for 6 minutes through
   `eval_run.sh`, read the distinct `map` values in the evaluator lines, and record pass if at least 12 are seen
   (and none held out). *Revised 2026-10-22, before the live check had run:* the plan first said 20; a stream changes
   map only at an episode start (a death every 20 to 85 s), so 20 distinct maps in 6 minutes was an arithmetic
   guess, not a requirement of the mechanism. 12 shows the maps change; the full 40-minute arm reports how many of
   the 28 it visits. The probe runs the rotation on a 20 s clock. A failure blocks `t1_wp6_diversity` with the reason.

**Why a rotation, not random choice.** The adapter has no random-map mode, and adding one would be a new code path;
the rotation uses only the built, reviewed WP-6 keys. It is environment structure, not meaning: no map has a name
or a goal for the learner.

**Parity.** No change to learning or adapter code. A game YAML with none of the keys, and every existing overlay,
composes as before.

**Verification (pre-registered).** (i) Unit: the merge, the generated schedule (coverage and no held-out map), and the
composer raising on a planted leak. (ii) Live, 6 minutes: at least 12 distinct maps, none held out. (iii) In the A/B: the
mechanism measures are the fresh-to-train gap at the end of source (AB-00 baseline: 1.75, range 0.91 to 3.25 over
16 runs) and the number of distinct maps; the behavioural verdict follows section 9.

**Kill criteria (GOV-009 P1).** Drop the arm if (a) the coverage check fails after one fix attempt, (b) any run
reports a held-out map in its source phase, or (c) it raises a traceback in the boot check. A result in which the
arm is not different from the baseline on the *descriptive* counts is recorded as a null, not as a failure of the
mechanism, because the primary metric cannot currently discriminate (section 9).

## 6. Testing list and queue changes

### New and revised tests

| Test | Kind | Covers |
|---|---|---|
| `tests/_guard_polarity_smoke.py` (new, `contract` tier) | contract | B1: producer meaning; both consumers skip boundary rows and pay/fire mid-life; a planted inverted consumer is caught; default parity |
| `tests/_consolidation_strength_smoke.py` (new, `contract` tier) | contract | B2: update norm rises with strength with AWR off; strength 1 equals the previous formula; base lr captured once; release returns to base |
| `tests/_wp6_diversity_smoke.py` (new, `contract` tier) | contract | C: `_games` merge, rotation coverage, no held-out map, composer `LeakError`, absent-key parity |
| `tests/_campaign_smoke.py` (revised) | contract | fresh-control stage and analysis exist and are ordered; `wp6_coverage_check` gates `t1_wp6_diversity`; `t2_rw6_strategy` blocked by F-24 until fixed; arm overlays validate |
| `tests/_categorical_cat04_smoke.py`, `_categorical_cat02_smoke.py` (unchanged files) | categorical | P-B1 and P-B2: exactly two scenarios flip; the rest of the failing list is unchanged |
| `tests/run_all.py` (revised) | registry | the three new suites in the `contract` tier |

Each new suite is shown to fail against the code before the change (GOV-009 P5) and to pass after.

### Queue (ASS-013) delta

| Stage | Change |
|---|---|
| `fresh_control` | **new**, after `tests_contract`; 10 runs, 3.3 h |
| `an_fresh_control` | **new** analysis, descriptive |
| `wp6_coverage_check` | **new**, 6 minutes; gates `t1_wp6_diversity` |
| `t1_wp6_diversity` | **unblocked** (needs the coverage check and its own boot check) |
| `arm_boot_checks` | now includes `wp6_diversity` |
| `t2_rw6_strategy` | **blocked with reason** (ASS-014 F-24: spaced replay raises in the consolidation phase, so the connected arm would silently run without consolidation); re-opens when F-24 is fixed |
| `ab_baseline`, every T1/T2 | unchanged, and run on the new snapshot |
| `cat_harness` | unchanged; expected to fail on the remaining findings (section 4 predictions) |

Net: +3.5 host-hours, +5 for `t1_wp6_diversity`, -10 held back for `t2_rw6_strategy`.

### What did not change, and why

AB-00 keeps running on the older code. Its configuration enables none of the changed paths, and the parity tests
make that a tested statement. Its results stay a valid A/A reference for the new snapshot.

## 7. Order of work and gates (GOV-009 P0 to P10, applied per item)

1. **P0/P2/P3** (this record, the amendments to GOV-009 and ASS-007, the register): done before any change.
2. **P4** build: B1, B2, then C (overlay merge, arm generator, check script), then A (stage). Each behind its
   existing switch or as pure tooling.
3. **P5** tests: write each, show it fails on the old code, then passes; run the `unit` and `contract` tiers, the
   categorical tier, and the documentation check.
4. **P6** mock run: the live 6-minute coverage check on `main` (C); the boot check of `wp5_info_gain` and
   `wp6_diversity` (queue).
5. **P7** A/B: `t1_wp5_info_gain`, `t1_wp6_diversity`, and the T2 arms that read the strength chain, run by the
   campaign on the new snapshot.
6. **P8** decision for each A/B follows section 9. **P9/P10** unchanged: every change is switchable or tooling-only.

## 8. Risks and rollback

- **B1 or B2 exposes a second problem behind the first** (for example the information-gain income becomes large
  once it pays mid-life). That is what the A/B and the ledger share are for; the cap knob exists
  (`curiosity.info_gain.cap_share`) and is not changed here.
- **C changes the target-phase difficulty indirectly** by changing the trained policy: intended and measured.
- **C makes source training harder** (more deaths on harder maps). The mechanism measures are reported
  alongside the fresh-to-train gap, so a worse source phase is visible and is not read as failure by itself.
- **Rollback.** B1: restore the two tests. B2: remove the lr scaling line. C: the arm is a switch; the `_games`
  key is inert when absent. A: remove the stage. None needs a data migration.

## 9. Not pursued, and its consequence

- **Recommendation 1 (metric)** is not pursued. Consequence recorded here so it is not forgotten: the checkpoint
  metric is empty for source-trained Doom runs (ASS-015 section 4), so no T1 or T2 A/B can show a *benefit* on it.
  Until the owner adopts a discriminating measure, A/B results are reported as mechanism measures and descriptive
  target counts, and P8 decisions are **deferred**, not made. The raw logs keep everything a later measure needs,
  so adopting one later requires no rerun.
- **Recommendation 5 (random-policy reference)** is not pursued; absolute levels remain unanchored.
- **Still open from ASS-014:** F-01 to F-03, F-05 to F-12, F-14 to F-28. F-24 is the one the queue now
  guards (section 6). The categorical pilots stay blocked while `cat_harness` fails.

## 10. Execution record

| Step | Status | Evidence |
|---|---|---|
| Documents revised before changes | done 2026-10-22 | this record, [GOV-009](gov-009-implementation-plan.md) Amendment 8, [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md) Amendment 8 |
| B1 F-13 consumers inverted | **done** 2026-10-22 | `curiosity/intrinsic_balance.py` and `skill_bank/effect_discovery.py`, one test each, comments naming the polarity; `tests/_guard_polarity_smoke.py` failed on the old code (5 checks) and passes |
| B2 F-04 lr scaling | **done** 2026-10-22 | `policy/consolidation.py` `_scale_lr`; `tests/_consolidation_strength_smoke.py` failed on the old code (update norm 0.0019845 at all three strengths) and passes; the strength-1 lr is exactly the previous lr |
| C `_games` overlay, arm generator, coverage check | **done offline**; live check queued | `tools/eval_run.sh`, `tools/campaign_arms.py`, `tools/map_coverage_check.py`; `tests/_wp6_diversity_smoke.py` passes (28 of 28 source maps started on, no held-out map, planted leak refused, old merge shown to fail). The live 6-minute check is the campaign stage `wp6_coverage_check` |
| A fresh-control stage | **done** (queued) | `fresh_control` and `an_fresh_control` in `tools/campaign.py`; planned at 3.4 h |
| Tests written, fail before, pass after | done | the three new suites; `_campaign_smoke` revised and passing |
| P-B1 and P-B2 predictions checked | done | section 4 outcome: categorical failing scenarios 41 to 39 |
| Deployed to the campaign copy on `main` | see the execution entry in [GOV-011](gov-011-execution-record-gov-009.md) D-63 to D-66 | |

### Test tiers after the changes (this Mac, 2026-10-22)

- **unit:** all pass. **categorical:** 39 failing scenarios, all recorded findings ([ASS-014](../assurance/ass-014-categorical-harness-first-run-findings.md)).
- **contract:** 6 suites red, **none caused by these changes**. `_incident_bundle_smoke` (the known macOS `flock`
  gap, CLAUDE.md section 2), `_vizdoom_smoke` and `_telemetry_ingest_smoke` also fail on a pristine export of `HEAD`.
  Three pass on `HEAD` and fail now because of the earlier build-only edits, not these: `_config_keys_smoke` (the
  new packages' config keys are not in its base list), `_universal_loop_smoke` contract F (absent games file) and
  `_no_engine_reward_smoke` contract F (it matches the source text of the episodic helper, which WP-5 edited). They
  are test-maintenance items, recorded here, not fixed here.

### First results of the corrective items (2026-10-23)

- **Item A (fresh control) ran.** Ten target-only runs against the twenty AB-00 target phases: no transfer visible on the
  held-out maps (items +2%, deaths 0%, kills -16%); see [ASS-016](../assurance/ass-016-ab00-final-and-fresh-control.md).
  This is the answer the program had not yet been able to ask, and it raises the weight of item C.
- **Items B and C are queued behind the boot checks.** Those were blocked by a queue defect (a failing informational test
  tier gated them) found and fixed on 2026-10-23; see [ASS-017](../assurance/ass-017-host-test-stages-and-queue-incidents.md). The WP-6
  live coverage check and its A/B have not yet run.

### Repository hygiene done in the same session (owner request)

Retired files moved to `legacy/` and `docs/legacy/` (list and reasons: [legacy README](../../legacy/README.md));
one exact duplicate removed (the unregistered copy of ASS-008); host-access guidance consolidated in
[OPS-008](../operations/ops-008-host-access.md) with the stale `root@`/`skybot@`/pod statements corrected in
CLAUDE.md and three operations guides; reciprocal document references made mandatory and enforced
(`tools/doc_backlinks.py`, `tools/check_docs.py`).

## 11. Related records

[ASS-015](../assurance/ass-015-ab00-interim-analysis.md), [ASS-014](../assurance/ass-014-categorical-harness-first-run-findings.md),
[ASS-013](../assurance/ass-013-unattended-test-campaign.md), [ASS-012](../assurance/ass-012-categorical-system-testing-protocol.md),
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md), [GOV-009](gov-009-implementation-plan.md),
[GOV-011](gov-011-execution-record-gov-009.md), [GOV-013](gov-013-categorical-system-testing-plan.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[RES-005](../research/res-005-skill-discovery.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-018](../assurance/ass-018-campaign-progress-2026-10-23.md): Campaign progress check, 23 October 2026 (evening)
- [GOV-002](gov-002-next-objectives.md): Next Objectives — post-audit roadmap (July 2026)
- [GOV-015](gov-015-corrective-plan-categorical-findings.md): Corrective plan for the categorical-harness findings and the host contract-tier failures
<!-- backlinks:end -->
