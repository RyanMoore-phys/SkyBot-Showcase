# Campaign progress check, 23 October 2026 (evening)

> **Document control:** [ASS-018](../README.md#assurance) · Assurance / progress record · **State:** Record; queue running, no behavioural result yet · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-23, read at 22:08 UTC. **Source:** read-only inspection of the campaign on `main`
([ASS-013](ass-013-unattended-test-campaign.md)): `campaign_state.json`, `campaign.log`, `CAMPAIGN_STATUS.md`
and the stage logs. **Evidence class:** fixture and tooling evidence; nothing here is a behavioural result.
Nothing was changed on `main`, in the code, or pushed.

## 1. Summary

1. **The queue is healthy.** Service active, 177 GB free, 8 GB of 14 GB RAM in use, no crash, interruption or pause
   in the log since the two fixes in [ASS-017](ass-017-host-test-stages-and-queue-incidents.md).
2. **Done since the last record:** `sens_split50`, `arm_boot_checks`, `wp6_coverage_check`, `sens_split80`,
   `cat_harness` (failed, as predicted), `ab_baseline`. Running: `t1_wp5_threshold` (seed 5 of 5, source phase).
3. **WP-6 live coverage passed:** 22 distinct source maps in 6 minutes (threshold 12), none held out. The
   diverse-source change does what it was built to do; whether that helps is the A/B, not yet run.
4. **Contract tier on the complete tree: 9 red suites, down from 13.** The four staging failures are gone.
5. **The categorical harness found real defects** (section 4); its failure correctly blocks the categorical pilots.
6. **Time left:** about 125 host-hours of runnable work (about 5 days), plus about 28 hours that is blocked.

## 2. Queue state

| Stage | State | Note |
|---|---|---|
| `arm_boot_checks` | done (4.1 h) | 34 arm overlays booted |
| `wp6_coverage_check` | done | 22 maps, threshold 12, none held out |
| `sens_split50`, `sens_split80` | done | not yet analysed |
| `ab_baseline` | done | shared baseline for T1 and T2; not yet scored |
| `cat_harness` | failed (rc 8) | expected; section 4 |
| `t1_wp5_threshold` | running since 17:32 UTC | last seed |
| 18 further T1 and T2 A/Bs with their scorers | runnable or queued | about 5 h each |
| `t1_wp1_skills`, `t1_wp10_scaling`, `t2_rw6_strategy` | blocked | [RES-005](../research/res-005-skill-discovery.md) step 0 not done; first growth decision after a one-hour seed ends; F-24 |
| `cat_sizing`, `cat_thresholds_frozen`, 14 categorical pilots, `cat_combined` | blocked | behind `cat_harness` |

## 3. The contract tier, second run

Nine red suites. Four are still unexplained: `_foundation_mechanisms` (`AssertionError 0.0703`),
`_foundation_shadow` (`['torch_rng']`), `_memory_census`, and `_brain_mirror` (its `.env` gitignore check; likely
a staging gap, not confirmed). Three are the known build-phase regressions (`_config_keys`, `_universal_loop` F,
`_no_engine_reward` F). Two are staging again: `_vizdoom` (the ROM, which differs between runs) and
`_telemetry_ingest` (`docker/monitor/ingest.py` absent). `_launchers` G no longer fails. `_foundation_shadow`
is the one to read first: it is a learner-state non-interference property. No cause is claimed here.

## 4. Categorical harness, first complete host run

Eight suites red (boundaries 16 failed, 6 blocked by missing modules; per-category failures 2, 5, 2, 4, 7, 3, 1).
The messages repeat a few defects, all already class-listed in
[ASS-014](ass-014-categorical-harness-first-run-findings.md):

- **Non-finite input poisons state.** One NaN batch leaves the uncertainty ensemble, the world model, the WP-7
  attention layer and the RW-3 tracker NaN for good.
- **Latches after one transient fault.** WP-8, WP-9, WP-13 and RW-1 wiring stay off for the run.
- **Restore errors.** The WP-9 learning rate after restore is the scaled value again (4.6e-5 against 8.4e-5), and
  the [ADR-005](../decisions/adr-005-compute-pressure-order.md) stepping scale is not restored (2.0 saved, 1.0 rebuilt).
- **Boundaries.** WP-7 windows span episode ends (7 of 11 pairs); WP-12 mixes view widths in one ring.

These are the harness doing its job. They are fixture-level findings on the code under test, not evidence about
learning. The decision on which to fix is the owner's (open since [ASS-014](ass-014-categorical-harness-first-run-findings.md)).

## 5. Reading

- **Promising:** operations. 20/20 AB-00, three pilots and the baseline finished with no crash, the queue now
  survives its own restarts, and the WP-6 change produces the intended source diversity.
- **Not promising yet, as measured:** [ASS-016](ass-016-ab00-final-and-fresh-control.md), no transfer visible. The
  first behavioural test of a remedy is the WP-6 A/B, which is still queued.
- **Needs a decision, not a repair:** NaN poisoning and the one-fault latches (section 4) sit under several
  queued A/Bs; an arm hitting one would silently run without its block. They are the highest-value fixes
  before the A/B results are trusted.

## 6. Limits

Read from logs only; no result files were opened and the sensitivity pilots and the baseline have not been scored.
Hours are the campaign's own estimates.

## 7. Follow-up (2026-10-23, later)

The owner asked for the recorded errors to be corrected. The corrections, the new tests and the host-test triage are in
[GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md); the findings' resolution is in
[ASS-014](ass-014-categorical-harness-first-run-findings.md) section 4B. The five suites section 3 called unexplained
now have a recorded reading: `_memory_census` assumed modules live on the CPU, `_foundation_shadow` required a count of moved
fingerprints that depends on the device, `_foundation_mechanisms` used a tolerance the host's numerics miss, and
`_brain_mirror` needs a repository the test tree lacks (a staging gap). `_launchers` G passed on the second run.
These readings are from the code, **not from a run**; the queued stage `gov015_tests` is the check.

**Follow-up, 2026-10-25.** The host was off about 41 h (04:20 UTC on 24 Oct to 21:25 UTC on 25 Oct) and the driver recovered by
itself: the interrupted seed of `t1_wp2_consolidation` restarted from its beginning (under an hour lost). The first `gov015_tests`
results are in [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md) section 9.5: the two new suites pass on the
corrected code and fail on the old; categorical boundaries, cat04, cat05 and one cat06 scenario are still red and are being
read by the queued `gov015_rerun`. Remaining estimate on 25 Oct: 120.3 runnable host-hours (about 5 days) plus 28.3 blocked.

## 8. Related records

[ASS-013](ass-013-unattended-test-campaign.md), [ASS-014](ass-014-categorical-harness-first-run-findings.md),
[ASS-016](ass-016-ab00-final-and-fresh-control.md), [ASS-017](ass-017-host-test-stages-and-queue-incidents.md),
[GOV-014](../governance/gov-014-corrective-plan-after-ab00.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-020](ass-020-recent-changes-static-audit.md): Recent changes: static implementation and documentation audit
<!-- backlinks:end -->
