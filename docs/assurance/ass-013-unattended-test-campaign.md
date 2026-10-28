# Unattended test campaign

> **Document control:** [ASS-013](../README.md#assurance) · Assurance / test-queue procedure · **State:** Record; campaign installed on `main` and running · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-22. **Authority:** owner instruction, 2026-10-22 ("schedule everything to run, including the
new testing, so that I do not need to start the runs by hand; analyse the logs one by one afterwards").
**Plan and protocols it carries out:** [ASS-007](ass-007-adaptability-evaluation-protocol.md) (A/A, T1, T2),
[GOV-013](../governance/gov-013-categorical-system-testing-plan.md) and
[ASS-012](ass-012-categorical-system-testing-protocol.md) (categorical tests),
[ASS-011](ass-011-main-evaluation-readiness.md) (AB-00).

## 1. What it is

One driver, `tools/campaign.py`, runs the whole testing queue on `main`, serially, with no operator. It owns
order, gates, resume and a status report. It owns no learning code and makes no verdict: arms come from
`tools/campaign_arms.py`, runs from `tools/eval_runner.py` and `tools/eval_run.sh`, scoring from
`tools/eval_analyze.py`. Its own contracts are `tests/_campaign_smoke.py` (in the `contract` tier of
`tests/run_all.py`). Everything it produces is development evidence (ASS-007 14.1).

## 2. The queue (in order; `campaign.py plan` prints the live version)

| Stage | What | Est. host-hours |
|---|---|---|
| `ab00` | Doom A/A noise floor, ten paired seeds. Attaches to the job started by hand; resumes it if it dies | 20 |
| `an_ab00` | Score the A/A pool | 0 |
| `tests_unit`, `tests_contract` | The suites never run during the build-only phase, on the staged code. Failures are reported, they do not stop the queue | 0.4 |
| `arm_boot_checks` | Boot each of the 35 generated arm overlays for 4 minutes (`tools/arm_boot_check.py`); an arm that raises a traceback is blocked, with the reason, and the others go on. Proves it boots, not that the module does anything | 3.5 |
| `sens_split50`, `sens_split80` | 40/20 split sensitivity pilot (ASS-007 14.9 b): the baseline at 30/30 and 48/12, seeds 11-13 | 6 |
| `cat_harness`, `cat_sizing`, `cat_thresholds_frozen`, `pilot_cat01..07_{baseline,enabled}`, `cat_combined` | GOV-013 / ASS-012 categorical campaign. **Blocked** until `tests/_categorical_*_smoke.py` exist (CS-1/CS-2 are not built); then it starts by itself. `cat_thresholds_frozen` is a human gate | 7.7 |
| `ab_baseline` | Shared baseline for T1 and T2, five seeds | 5 |
| `t1_*` (+ `an_t1_*`) | One block against the baseline, five seeds each | 5 each |
| `t2_*` (+ `an_t2_*`) | Relationship connected against cut, same blocks in both arms, five seeds each | 10 each |

About 190 host-hours with every blocked stage counted (the run is serial on one 16 GB host).

**Queue revision after the AB-00 interim analysis (2026-10-22; [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md) section 6).**
New stages: `fresh_control` (10 target-only runs, 3.3 h) with `an_fresh_control`; `wp6_coverage_check` (6 minutes, gates
`t1_wp6_diversity`). `t1_wp6_diversity` is no longer blocked and `arm_boot_checks` now includes `wp6_diversity`.
`t2_rw6_strategy` is blocked by ASS-014 F-24 until it is fixed. Every stage after `ab00` runs on the corrected
snapshot (deployed 2026-10-22, snapshot `eaa8dd4905bf`); status of each change is in GOV-014 section 10. Evidence for the changes:
[ASS-015](ass-015-ab00-interim-analysis.md).

**Corrections after the first host results (2026-10-23; [ASS-017](ass-017-host-test-stages-and-queue-incidents.md)).**
`arm_boot_checks` no longer requires the test tiers (a failing informational stage had blocked every A/B: a latch).
A stage already running is re-attached before any other is considered (a restart had started polling for a free
host ahead of a running stage). The test tiers run from a separate complete tree so that missing `docs/`,
`experiments/`, `docker/` or ROM files are not reported as failing suites. First results: [ASS-016](ass-016-ab00-final-and-fresh-control.md).

**Categorical harness (2026-10-22).** The CS-0 to CS-2 files now exist ([ASS-012](ass-012-categorical-system-testing-protocol.md) section 8), so `cat_harness` is no longer blocked: it runs `tests/run_all.py categorical` after AB-00 and archives one JSON record per file. Its first run found real defects ([ASS-014](ass-014-categorical-harness-first-run-findings.md)), so it is expected to **fail**; a failed `cat_harness` blocks `cat_sizing`, the pilots and the combined run (ASS-012 5.2) but, being `on_fail=continue`, does not stop the T1/T2 queue. It re-opens with `campaign.py resume cat_harness` after fixes.

**Blocked, scheduled, never started** (reason shown in `plan`): `t1_wp1_skills` (RES-005 Step 0 not done,
GOV-011 D-14), `t1_wp6_diversity` (needs a game-YAML overlay; the runner merges overlays into
`universal.yaml` only), `t1_wp10_scaling` (long-run protocol deferred, RES-007 H-17), and the categorical
stages above. A blocked stage is neither a pass nor a failure, and its dependents are blocked with it.

## 3. Rules the driver keeps (each answers a recorded failure; CLAUDE.md 4.1)

- **Serial and exclusive.** A stage starts only when no other trainer is alive; otherwise it waits. It never
  kills a run and never runs beside one.
- **Resumable.** State is `campaign_state.json`. After a reboot a stage found running re-attaches to its live
  process or re-enters; the eval runner resumes its own state. A stage's process has its own session, so
  restarting the driver never kills a run.
- **Frozen code.** The staged copy's hashes are recorded at first run; a changed tree is refused until
  `--accept-code-change` (a fix starts a new snapshot, ASS-012 6).
- **Escapes (what re-opens each stop).** A failed timed stage **pauses** the queue: `campaign.py resume
  STAGE`. A **human gate** (`cat_thresholds_frozen`: thresholds must not be chosen by a script after seeing
  results): `campaign.py gate NAME`. **Blocked** stages are re-evaluated each pass. Tests and analysis stages
  never stop the queue. `touch <root>/CAMPAIGN_STOP` stops *between* stages.
- **Disk (measured 2026-10-22).** One run is about 11 GB, of which results are about 1 MB: the rest is two
  copies of the brain's checkpoints and replay. Left alone, AB-00 alone (20 runs) would fill the 232 GB disk
  and the full queue (about 185 runs) would need about 2 TB. So a sweep (every 10 minutes while waiting, and
  before each stage) **prunes** each *finished* run's `devai/` and `snap_source_end/logs` and **gzips** its
  logs under `results/` (the original is removed only after the `.gz` reads back at the same size; the scorer
  reads `.gz`). Kept: results, phase files, small snapshot parts, and any `failed_final` run whole, for
  diagnosis. Running and pending runs are never touched; symlinks (the shared venv) are unlinked, never
  followed. If less than 30 GB is free after pruning, the queue **pauses before** the next stage rather than
  fill the disk; free space, then `resume`. The pruned raw checkpoints are gone by design: a rerun needs a new
  run.
- **Nothing hidden.** `CAMPAIGN_STATUS.md` lists every stage with its state and reason.
- **Arms cannot be no-ops.** `campaign_arms.validate()` checks every overlay key against
  `configs/universal.yaml`; a mistyped key would otherwise make an A/B an A/A (contract A, falsified by test).

## 4. Installation on `main` (2026-10-22)

Staged copy and state: `/workspace/experiments/campaign_20261022/` (`code/`, `overlays/`, `stages/`,
`analysis/`, `gates/`, `campaign.log`, `CAMPAIGN_STATUS.md`). Started at boot by the systemd unit
`skybot-campaign` (template `scripts/campaign.service`; wrapper `scripts/campaign_start.sh`). Without sudo, a
`@reboot bash <root>/code/scripts/campaign_start.sh <root>` cron line does the same. `/workspace/devai` is
not touched. Host login and address are not recorded here (CLAUDE.local.md).

Commands (run from `<root>/code`, venv `/workspace/devai/venv_mc`): `tools/campaign.py --root <root> plan |
status | resume [STAGE] | gate NAME`.

## 5. Limits of this record

- First-arm overlay **values** (for example `info_gain.weight 0.05`, `cluster_distance 0.008`) are the
  packages' documented starting points, not tuned values; the T2 prerequisite sets are read from each
  cluster's module header and have not been run. The first run of an arm may expose a wrong one, which the
  crash rule reports.
- The split pilot uses seeds 11-13 and two other splits: a design chosen here, open to the owner's change.
- Nothing here establishes a result. It schedules and reports; verdicts come from the analysis records
  (`analysis/<stage>/record.md`), read one by one.

## 6. Cleanup of earlier test data (2026-10-22, owner instruction)

On `main`, after the logs and results (2,176 text files, 8.7 MB) were archived to
`~/old_test_evidence_text_20261022.tar.gz` and the archive was read back, these were deleted: the
Stage C, policy-learning and size-experiment folders (about 56 GB, mostly replay buffers and
checkpoints), the AB-00 dry run, the confidence-integrity scratch root, the discarded first AB-00 root and
scratch files in `/tmp`. Disk use fell from 116 GB to 49 GB. Raw checkpoints and replay behind ASS-003,
RES-004 and GOV-005 no longer exist; their logs do, in the archive. The live AB-00, the campaign root and
`/workspace/devai` were not touched.

## 7. Power loss and reboot (2026-10-22, owner instruction)

**What happens.** `skybot-campaign` is enabled, so it starts at boot (after a 20 s settle delay). It then, before
any stage:

1. **Repairs the runners' state.** A run that was in progress when the host went down (its start time is before
   this boot) is recorded as an **interruption, not a crash**: it restarts from the beginning of its seed with
   its attempt count unchanged. Reason: the eval runner's own rule counts a run found mid-phase as a crash and
   spends the seed's one retry, so a second power cut would end the seed with no claim (ASS-007 14.6); ASS-012 6
   calls an unrelated host interruption inconclusive, not a failure. The repair is done by the campaign because a
   running experiment's own code is frozen. A runner that is alive, or a driver death with no reboot, is left to
   the runner's own rule.
2. **Restores damaged state.** `eval_state.json` is backed up at every sweep and restored if unreadable; the
   campaign's own state is fsynced, renamed into place and kept with a `.bak` of the last good copy. If neither
   copy is readable it **refuses to start over** and the queue pauses (restore one by hand, then `resume`).
3. **Re-enters what was running.** An eval stage relaunches its runner (which resumes its own state); a
   one-off command stage (tests, pilots, boot checks) restarts clean.

**Logs.** Every `campaign.log` line is fsynced; completed runs keep their results, phase files and gzipped logs
(a log is gzipped to a new file and the original removed only after the `.gz` reads back whole). Interrupted
runs are replayed, so their partial logs (copied to `results/` only at the end of a phase) are not kept.
`interruptions` entries are added to the run's record with the boot time.

**Limits, stated plainly.**
- A **real power cut or reboot has not been rehearsed** on `main`: it would cost the running seed up to an hour.
  The repair is tested against the real runner's `EvalState` with an injected boot time
  (`tests/_campaign_smoke.py` N, O), and a driver restart while a run is live was done on the host and reattached.
- A seed cut mid-way is **replayed from its start**, so each power cut costs up to one run (about an hour, ten
  at the very worst for an unlucky T2 block); a brain checkpoint half-written at the cut is not used because the
  run is rebuilt.
- The state backup is one save behind, so a restore can forget the last transition; the driver re-derives it
  (eval stages from the runner's state, command stages by re-running).
- Coming back **after** a power cut needs the machine to power on by itself (a BIOS "restore on AC power loss"
  setting) and to boot without a prompt; neither is in this repository's control. Check the BIOS setting once.

## 8. Related records

[ASS-007](ass-007-adaptability-evaluation-protocol.md), [ASS-011](ass-011-main-evaluation-readiness.md),
[ASS-012](ass-012-categorical-system-testing-protocol.md),
[GOV-013](../governance/gov-013-categorical-system-testing-plan.md),
[GOV-011](../governance/gov-011-execution-record-gov-009.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-008](../architecture/arc-008-training-run-lifecycle.md): Training run contract, queue and environment lifecycle
- [ASS-018](ass-018-campaign-progress-2026-10-23.md): Campaign progress check, 23 October 2026 (evening)
- [ASS-019](ass-019-predictive-coding-core-comparison-protocol.md): Predictive-coding core versus the RSSM: comparison protocol
- [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md): Corrective plan for the categorical-harness findings and the host contract-tier failures
- [GOV-016](../governance/gov-016-predictive-coding-core-plan.md): Implementation plan: predictive-coding core versus the RSSM (WP-20)
- [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md): Containerization, Kubernetes and AIOps proposal
- [OPS-008](../operations/ops-008-host-access.md): Host access: how to reach `main`, and what is never in this repository
- [OPS-009](../operations/ops-009-cluster-delivery-and-operations.md): Cluster delivery, automatic training and routine operations
<!-- backlinks:end -->
