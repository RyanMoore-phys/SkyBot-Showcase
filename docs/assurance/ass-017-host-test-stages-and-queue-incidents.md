# Host test stages and two queue incidents (2026-10-23)

> **Document control:** [ASS-017](../README.md#assurance) · Assurance / result record · **State:** Record; contract tier re-queued, five suites unexplained · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-23. **Source:** the first unattended run of the test stages `tests_unit` and `tests_contract` on
`main` by the campaign ([ASS-013](ass-013-unattended-test-campaign.md)), the state of the queue behind them, and two
defects in the queue itself found while checking on it. **Evidence class:** fixture and tooling evidence; nothing
here is a behavioural result. Nothing in the learning code was changed by this record.

## 1. Summary

1. **Unit tier: green on the host.** All suites passed (about 20 s).
2. **Contract tier: 13 suites failed on the host (7 minutes), against 6 on the Mac.** Five of the 13 are staging
   omissions (the frozen code copy lacks `docs/`, `experiments/`, `docker/` and the ROM, which those suites read);
   three are the build-phase regressions already recorded ([GOV-014](../governance/gov-014-corrective-plan-after-ab00.md)
   section 10); **five are not yet explained** (section 3). None involves code changed by GOV-014.
3. **The queue had a latch.** A failing contract tier blocked the boot checks, the WP-6 live check, every T1 and T2 A/B
   and the categorical pilots (section 4). It cost nothing because other stages were runnable; it would have ended
   the queue with the A/Bs unrun. Fixed.
4. **A restart exposed a second hazard** that could have started two trainers at once. Caught within a minute and fixed
   (section 5).

## 2. What ran and in what state the queue is (03:25 UTC)

| Stage | State | Note |
|---|---|---|
| `ab00`, `an_ab00` | done | [ASS-016](ass-016-ab00-final-and-fresh-control.md) |
| `tests_unit` | done, all pass | section 3 |
| `tests_contract` | failed, 13 suites, re-queued | section 3 |
| `fresh_control`, `an_fresh_control` | done | ASS-016 |
| `sens_split50` | running (1 of 3 runs done, the second in its target phase) | |
| `arm_boot_checks`, `wp6_coverage_check`, T1 and T2 A/Bs, categorical pilots | were blocked by the latch; unblocked | section 4 |

Disk 171 GB free; no crash, no pause, no interruption in the campaign log.

## 3. The contract tier on the host

Counts are suites, not scenarios. "Mac" is the same tier on the development machine on 2026-10-22.

| Suite | Host | Mac | Reading |
|---|---|---|---|
| `_foundation_baseline_smoke` | fail: `docs/foundation/fnd-002-assumptions.md` not found | pass | staging: no `docs/` in the code copy |
| `_foundation_ab_verify_smoke`, `_foundation_ab_verify9_11_smoke` | fail: `No module named 'experiments'` | pass | staging: no `experiments/` |
| `_telemetry_ingest_smoke` | fail: `docker/monitor/ingest.py` not found | fail | staging on the host; a temp-file error on the Mac (also red on a pristine export of `HEAD`) |
| `_vizdoom_smoke` | fail: `roms/doom/freedoom2.wad` missing | fail | staging: the ROM lives in `/workspace/devai/roms`, not in the copy |
| `_config_keys_smoke` | fail | fail | build-phase regression: the new packages' keys are not in its base list |
| `_universal_loop_smoke` contract F | fail | fail | build-phase regression (absent games file) |
| `_no_engine_reward_smoke` contract F | fail | fail | build-phase regression: it matches the source text of the episodic helper, which WP-5 edited |
| `_foundation_mechanisms_smoke` | fail: `AssertionError 0.0703` | pass | **not explained** |
| `_foundation_shadow_smoke` | fail: `AssertionError ['torch_rng']` | pass | **not explained** (a shadow-module non-interference check) |
| `_brain_mirror_smoke` | fail | pass | **not explained** |
| `_memory_census_smoke` | fail: `AssertionError` | pass | **not explained** |
| `_launchers_smoke` contract G | fail: "exec: nothing missing when all tools exist" | pass | **not explained**; plausibly a tool absent on the host |
| `_incident_bundle_smoke` | pass | fail | the known macOS `flock` gap; passes on Linux as documented |

**Interpretation.** The five staging failures are an artefact of the test tree, not of the code under test: the frozen
copy must stay exactly the code under test, so the suites need a separate tree. That is now built (section 4). The
five unexplained suites differ between the machines and are the ones worth reading first: the host runs a newer
torch and a different OS, so a platform sensitivity is as likely as a defect, and `_foundation_shadow` failing on
`torch_rng` (a learner-state non-interference property, the same property ASS-012 CT-08 asserts) should be read
before the A/Bs are trusted. They are **not** explained by this record and no cause is claimed. The re-run on the
complete test tree (queued) will show which of the 13 remain.

## 4. Incident 1: a failing informational stage blocked the A/B queue

**What happened.** `arm_boot_checks` was defined as requiring `tests_contract` to have *passed*. The contract tier has
known red suites, so on the first failure every stage that requires the boot checks (`wp6_coverage_check`, each T1
and T2 A/B, the categorical pilots, the combined run) was blocked with the reason "requires tests_contract which
failed". The driver kept running the stages that did not depend on it (the split pilots, the fresh control, the
baseline).

**Cause.** A guard with no reachable release (CLAUDE.md 4.1, guard-becomes-latch): the condition could not become true
without someone fixing suites unrelated to the A/B arms.

**Fix.** `arm_boot_checks` now requires only `ab00`; the test tiers are informational and gate nothing. A test
(`tests/_campaign_smoke.py`) sets `tests_contract` to failed and asserts the boot checks, the WP-6 check and a T1 A/B
are not blocked. The test tiers now run from a separate complete tree (`test_code`, built from the frozen copy plus
`test_extras` and the host ROMs), so the frozen copy stays the code under test, and `tests_contract` was re-queued.

## 5. Incident 2: a restart could have started a second trainer

**What happened.** The restart that deployed the fix above chose `arm_boot_checks` (earlier in the stage list) before
re-attaching to the running `sens_split50`, and began polling for a free host every five minutes. A phase boundary
of a split-pilot run leaves a gap of roughly a minute or two with no trainer alive (stop, snapshot, build), so a poll
could have started the boot checks beside the next phase on a 16 GB host.

**Containment.** The driver was stopped within a minute (03:29 UTC); only the pilot's trainer was running and
nothing else had started. The experiment was not interrupted.

**Fix.** `next_stage` now re-attaches to a stage already running **before** considering any other. A test asserts a
running later stage is chosen over an earlier runnable one, and that with nothing running the first runnable is chosen as
before. The service was restarted and logged "ATTACH sens_split50: already running" with no other stage started.

**Residual risk, stated.** Both deploys changed files in the campaign's frozen code copy while `sens_split50` was
running. Its runner checks its frozen hash only at start, so the running stage is unaffected; if its runner were
restarted (a crash or power loss) it would refuse to resume on the hash mismatch and the stage would fail and pause
the queue until the owner decides. This applies for the remaining 1 to 2 hours of that stage only.

## 6. Limits

- Staging explanations (section 3) were read from the error text and the missing paths; the re-run is the check.
- The Mac comparison is the Mac's own earlier run, not a paired rerun against the same commit.
- Nothing here says any suite's *claim* is false; a red suite is a prompt to read it.

## 7. Related records

[ASS-013](ass-013-unattended-test-campaign.md), [ASS-014](ass-014-categorical-harness-first-run-findings.md),
[ASS-016](ass-016-ab00-final-and-fresh-control.md), [ASS-012](ass-012-categorical-system-testing-protocol.md),
[GOV-014](../governance/gov-014-corrective-plan-after-ab00.md),
[GOV-011](../governance/gov-011-execution-record-gov-009.md).

## 8. Addendum (2026-10-28): the older failed stages, reviewed

Appended; sections 1-7 are as written. Read from the stage logs and the campaign status on `main` on 2026-10-28. "Frozen" means the `code/` copy every A/B ran on; "staged" means `gov015_code/` with the [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md) fixes.

| Stage | Result recorded | Cause | State and action |
|---|---|---|---|
| `tests_contract` (frozen) | failed: 13 suites (`_config_keys`, `_foundation_baseline`, `_foundation_mechanisms`, `_foundation_shadow`, `_foundation_ab_verify`, `_foundation_ab_verify9_11`, `_brain_mirror`, `_memory_census`, `_universal_loop`, `_vizdoom`, `_launchers`, `_no_engine_reward`, `_telemetry_ingest`) | platform-sensitive or stale assertions, the rename, a test/tool name collision | The "before" state. On the staged tree the first run left 4 contract suites red (`_clean_exit`, `_universal_loop`, `_vizdoom`, `_telemetry_ingest`); the first three were corrected in GOV-015 9.6 (`gov015_rerun2` rc 0); the fourth is the name collision of GOV-015 9.12, **fixed and queued** (`gov015_verify2`). |
| `gov015_tests` (staged) | failed: 4 contract and 4 categorical suites (boundaries, cat04, cat05, cat06) | as above, plus the test defects of GOV-015 9.5-9.10 | cat04, cat05, cat06 and the boundaries suite are now 63/6/2 (GOV-015 9.11); the two remaining BND-04 items are queued for diagnosis (`gov015_diag3`). |
| `gov015_tests_before` (old code) | failed, as required: 8 categorical suites (43 boundary scenarios raised `AttributeError: ... no attribute '_phase_rearm'`) | the new tests against code that lacks the fixes | The failure-before evidence. Nothing to fix. |
| `gov015_verify` | failed: see GOV-015 9.9 | | Superseded by the corrected tests; its result stays on record; `gov015_verify2` is the re-run. |
| `cat_harness` (frozen) | failed: 8 of 9 categorical suites (boundaries 49/6/16, cat01 6/0/2, cat02 15/0/5, ...) | **real defects in the frozen code**, the 28 findings of [ASS-014](ass-014-categorical-harness-first-run-findings.md) | **Blocks by design**: `cat_sizing`, the human gate `cat_thresholds_frozen`, 14 operational pilots and `cat_combined` (about 9 host-hours). The fixes exist only in the staged tree. Landing them in `code/` breaks "the frozen code is the code under test" and needs the shared baselines re-run. **Not applied; owner decision** (options below). |
| `t2_rw6_strategy` (blocked) | blocked on ASS-014 F-24 | the frozen code still has the defect; the staged code has the fix | Same decision as `cat_harness`. |
| `t1_wp1_skills`, `t1_wp10_scaling` | blocked | RES-005 step 0 not done; long-run protocol deferred | Unchanged; not a fault. |
| `pc_pilot`, `pc_pilot2` | failed (gate) | the Hebbian core misses G3 and G4 | Recorded in GOV-016 11d-11e; not a stage defect. |
| `t1_wp20_pc_core` | blocked | **the WP-20 arm's boot check failed** (`exit code 3`, agent never came up), which my earlier reports missed | GOV-016 11g; `pc_boot_check2` queued with the logs kept. |

**Options for the frozen-code decision** (none applied): (A) land the GOV-015 and competence fixes in a new frozen copy and re-run the shared baseline (`ab_baseline`, 5 h) and every A/B that must be read against it; (B) run the categorical chain on a corrected snapshot, as WP-20 does for its own A/B (the operational pilots then test the corrected code, and are not compared with the older A/Bs); (C) keep them blocked. B is the cheapest and keeps every existing comparison intact; A is the cleanest; neither changes a result already recorded.

**Process findings from this review.** (1) A stage can end `done` while its record says FAIL (the boot check): read the record, not the status. (2) A tool that deletes its evidence on exit (the boot check removed the booted root) cannot explain its failure. (3) Staging with `tar` on macOS leaves `._*` files in the target (GOV-015 9.10). (4) Copying the driver into the frozen `code/` changes the evaluator's recorded hash (GOV-015 9.10).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-015](ass-015-ab00-interim-analysis.md): AB-00 Doom A/A: interim analysis (16 of 20 runs)
- [ASS-018](ass-018-campaign-progress-2026-10-23.md): Campaign progress check, 23 October 2026 (evening)
<!-- backlinks:end -->
