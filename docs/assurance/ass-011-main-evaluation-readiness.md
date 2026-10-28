# Main evaluation readiness and run record

> **Document control:** ASS-011 · Assurance / evaluation run record · **State:** Record; restarted run in progress on `main` · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21  
**Purpose:** Record the readiness check for the A/A and A/B procedures in GOV-009,
GOV-010, and ASS-007. This record distinguishes work actually performed from
the runs that remain blocked.

## 1. Outcome

The ten-pair Doom development A/A schedule is now running on `main`, using
held-out Doom maps as development evidence. Minecraft is reserved for the later
program-level performance comparison after the implementation-plan tests. An
earlier draft of this record incorrectly treated Minecraft as the immediate
test target; that interpretation is superseded by ASS-007 Amendment 6.

Two bounded, non-behavioral checks from the implementation plans were run in
isolated temporary directories on `main`. The deployed `/workspace/devai`
checkout was not changed. No experiment root or long-running job was created.

## 2. Readiness procedure and findings

The procedure was to verify host identity, live trainer state, installed
checkout revision, and the evaluation Python environment before transferring
or running anything. This follows the host verification and non-interruption
rules in [OPS-005](../operations/ops-005-operations.md) and the
pre-run connection check in [OPS-007](../operations/ops-007-runner-mini.md).

| Check | Method | Finding |
|---|---|---|
| Tailscale CLI | `tailscale ssh ... <login>@$MAIN_HOST hostname` | Local CLI could not load preferences. Direct SSH worked instead. |
| SSH as `root` / `skybot` | Batch-mode SSH to `$MAIN_HOST:22` | Both rejected authentication. No remote command ran. |
| SSH as the host login (see the untracked local notes) | Read-only host, trainer, checkout, and venv check using `~/.ssh/id_ed25519` | Connected; hostname `main`; no trainer process found; Python 3.10.21 available. `/workspace/devai` exists but is not a Git working tree. |
| Test-file availability in deployed checkout | Checked for the focused WP-14 test, evaluator runner, and phase script under `/workspace/devai` | All four files are absent, so tests were staged in isolated `/tmp` directories from this local working tree. |

The local branch is `Adjustments`, at working-tree `HEAD` `f4164b4e94b3c2dc20d7f6d060490608935a2445`, with substantial uncommitted and untracked implementation changes. This is not a deployable immutable revision. No deploy or source transfer to `/workspace/devai` was attempted. The deployed copy's source revision is unknown.

## 3. Applicable test procedures

### 3.1 WP-14 synthetic machinery procedure

GOV-009 WP-14 specifies synthetic known-effect and null-effect analysis,
censoring cases, scripted detector trajectories, and an A/A calibration.
The repository's focused synthetic procedure is
[`tests/_eval_machinery_smoke.py`](../../tests/_eval_machinery_smoke.py):

```bash
PYTHONPATH=. python tests/_eval_machinery_smoke.py
```

Its documented methodology checks the 40/20-minute phase split and fresh
control planning; recovery of a known paired effect; false-positive behavior
under no effect; moving-block bootstrap behavior; right censoring; A/A pool
enumeration; crash/resume and frozen-file rules; the intentionally undefined
Minecraft detector; and the generated record fields. The file says it was not
run at build time. It was staged with only `developmental_ai/evaluator/` in
`/tmp/ass011_wp14_20261021/` and run with the host's
`/workspace/devai/venv_mc/bin/python`, using
`PYTHONPATH=/tmp/ass011_wp14_20261021`. The first invocation used an incorrect
staging path and did not execute the test; after correcting the package layout,
this host command completed successfully:

```text
cd /tmp/ass011_wp14_20261021 && \
  PYTHONPATH=/tmp/ass011_wp14_20261021 \
  /workspace/devai/venv_mc/bin/python \
  /tmp/ass011_wp14_20261021/_eval_machinery_smoke.py

A. 40/20 split, seed-major order, fresh control has target only
B. +0.20 recovered on five seeds; no-effect all-five-positive rate 0.000
C. block length adapts; block interval wider than iid on AR(1)
D. censoring: unreached = 0, immediate = 1, earlier is better
E. 252 exact subsets; no A/A pool -> no claim
F. crash rule, mid-phase death, frozen-config refusal
G. Minecraft detector fails loudly: ladder not defined
H. record carries verdict, floor, power, evidence class, deviations
ALL PASS (exit 0)
```

This is a synthetic machinery result, not a real A/A, A/B, or behavioral
result.

### 3.2 GOV-010 R1 relation conformance

GOV-010 section 3 / GOV-012 R1 specify a strict implementation-coverage
conformance report. The following branch files and their input map/RES-008
documents were staged under `/tmp/ass011_r1_20261021/`; the deployed checkout
was not modified:

```bash
PYTHONPATH=/tmp/ass011_r1_20261021 \
  /workspace/devai/venv_mc/bin/python \
  /tmp/ass011_r1_20261021/tools/relation_conformance.py --strict
```

**Result:** exit 0; map JSON current; 346 map lines, 326 compared and 20
excluded; 326 RES-008 rows; 106 registered channels; zero ID drift; zero
channel/map mismatches; all 36 RES-008 `Connected` lines have channels.
Coverage summary: 106 channels, 12 descriptive links, 208 lines with no
channel. This procedure checks bookkeeping and implementation coverage only;
it does not test runtime signal delivery, behavior, or benefit, and it is not
an A/B result.

### 3.3 AB-00 Doom A/A development calibration

ASS-007 Amendment 6 specifies ten paired A/A seeds (11–20), two identical
baseline arms, source Doom MAP01–MAP28 and held-out Doom MAP29–MAP32, 40/20
minutes per one-hour run, seed-major interleaving, and the existing crash rule.
This estimates development noise on the held-out Doom proxy only. It does not
establish final Minecraft performance. The Minecraft checkpoint list is
reserved for the later program comparison.

The implementation runner would be invoked in this form after its code and
host readiness are established:

```bash
python tools/eval_runner.py --root <experiment-root> --aa \
  --arm baseline_a=- --arm baseline_b=- \
  --source-games doom --target-game doom --execute
```

**First start (superseded, 2026-10-21).** The command executed on `main` under
`/workspace/experiments/doom_ab00_20261021` (driver PID `50482`) with the
uncorrected evaluator code. See section 9: it was stopped in the target phase of
`baseline_a_s11` and discarded.

**Current job (restart, 2026-10-22 00:29 UTC).** The command is executing on `main`
under `/workspace/experiments/doom_ab00_20261022`, driver PID `56636`, with the
corrected code (C-1, C-2). The run state contains 20 ordered arm/seed runs and hashes
for 353 staged source files (the same set and count as the first start; three files
differ, see section 9). The first run, `baseline_a_s11`, began in its 40-minute source
phase. The driver is detached with `nohup`; it advances seed-major and interleaves the
identical baseline arms.

## 4. Schedule and duration reconciliation

The Doom development A/A schedule is installed on `main`. Based on the runner's plan,
The runner's plan-only schedule was generated locally from this branch using:

```bash
python tools/eval_runner.py --root /tmp/ass011_ab00_plan_only \
  --aa --arm baseline_a=- --arm baseline_b=- \
  --source-games doom --target-game doom --plan-only
```

It reports 20 one-hour runs, seeds 11–20, in seed-major order: for each seed,
run `baseline_a` source then target, followed by `baseline_b` source then
target, then advance to the next seed. This is the requested back-to-back
order being executed on the training host. The total planned runtime is **20
serial host-hours**, before startup, checkpointing, analysis, crashes or retries.

The older estimates in ASS-007 section 14.9(c) and GOV-009 section 4 do not
match the implemented ten-pair, two-arm, serial schedule. They should be
reconciled in a later plan edit; the current execution is budgeted at 20 hours.

## 5. Result status

- **A/A runs completed:** 0 / 10 seed pairs.
- **A/A runs in progress:** `baseline_a_s11`, source phase (restarted job, section 9). The first job's partial `baseline_a_s11` is not counted.
- **Split sensitivity pilot runs started:** 0.
- **A/B runs completed:** 0.
- **WP-14 synthetic machinery smoke:** pass, 8 contracts, isolated staging.
- **GOV-010 R1 strict conformance:** pass, zero reported drift/mismatches and
  zero Connected lines missing a channel.
- **Host trainer:** no trainer process found during the readiness check.
- **Deployed source revision:** unknown; `/workspace/devai` is not a Git tree.
- **A/A / A/B training runs:** 0.
- **Claims:** no behavioral or promotion claims. The two passing checks are
  development-tooling and coverage evidence only.

## 6. Owner correction and Doom development test status

On 2026-10-21 the owner clarified that implementation-plan A/A and A/B tests
run on Doom; the Minecraft checkpoint list and persistent Paper world apply to
the later final-performance comparison. A two-minute source/target dry run
completed on `main`: source exited cleanly; the target config contained only
MAP29–MAP32 and exited cleanly. The short target log had no Doom evaluator event
lines (the adapter logs every 5,000 steps), so its score had zero events and is
not behavioral evidence. The AB-00 job uses the 40/20 protocol; it was restarted
with corrected code on 2026-10-22 (section 9).

## 7. References

- [ASS-007 — Adaptability evaluation protocol](ass-007-adaptability-evaluation-protocol.md), especially sections 5, 9, 12, 14.2–14.9, and 15.
- [GOV-009 — Implementation plan](../governance/gov-009-implementation-plan.md), WP-14 and phases P0–P9.
- [GOV-010 — Relational implementation plan](../governance/gov-010-relational-implementation-plan.md), sections 4A–4B (T1/T2, prerequisites, result records).
- [GOV-011 — Execution record for GOV-009](../governance/gov-011-execution-record-gov-009.md), WP-14 status and pending tests.
- [GOV-012 — Execution record for GOV-010](../governance/gov-012-execution-record-gov-010.md), built-but-untested relation work and pending A/Bs.
- [OPS-005 — Operations](../operations/ops-005-operations.md), safe host deployment and run handling.
- [OPS-007 — Runner mini](../operations/ops-007-runner-mini.md), documented host connection check.
- [`tests/_eval_machinery_smoke.py`](../../tests/_eval_machinery_smoke.py), synthetic WP-14 procedure.
- [`tools/eval_runner.py`](../../tools/eval_runner.py) and [`tools/eval_run.sh`](../../tools/eval_run.sh), runner and isolated phase execution implementation.

## 8. Review corrections (2026-10-21, applied after Luna's run began)

Source: a review of this record, ASS-007 Amendments 5 and 6 and the evaluator tools against
GOV-009. Corrections made to the records and to the code (the running job uses its own
frozen copy and is **not** changed by these edits).

| # | Finding | Correction |
|---|---|---|
| C-1 | **The target phase cannot be scored at useful resolution.** `tools/eval_run.sh` did not set the Doom evaluator's logging interval; the default is one line per 5,000 decisions per stream. A 20-minute target phase has only about 6,000 decisions per stream, so it produces about one evaluator line, near minute 17. The earlier policy-learning runs set 512 for this reason (ASS-003 Amendment 1); the generalised script dropped it. The dry run in section 6 already showed zero events. | `eval_run.sh` now sets `game.log_every` to 256 (about every 50 s) for every evaluation root |
| C-2 | **`first_kill` and `first_pickup` read per-life counters.** The game's `KILLCOUNT` and `ITEMCOUNT` reset on death or a new level (`deaths` and `exits` are cumulative). A sparse line can read 0 after earlier kills. | The Doom adapter's evaluator record gains cumulative `kills_total` and `items_total` (evaluator-only); the Doom detector uses them, falling back to the old fields for old logs |
| C-3 | **The live AB-00 job predates C-1 and C-2.** Its staged code lacks both fixes, so its target-phase scores would be coarse and biased. | Recommended: stop the job, re-stage the corrected code, and restart AB-00 under a new freeze date. Any A/A pool built from the current job should not be used for scoring. The owner decides when to restart. **Done 2026-10-22: see section 9** |
| C-4 | **The 40/20 split sensitivity pilot was not run before the Doom AB-00 procedure was frozen.** The owner accepted the split on the condition that a pilot precede freezing the protocol (ASS-007 15). Section 5 shows 0 pilot runs. Amendment 6 froze the Doom development procedure without it, and this record did not list that as a deviation. | Recorded here as a deviation. AB-00 results are provisional with respect to the split until the pilot is run; the pilot stays a required step |
| C-5 | **Private host details were in a tracked record.** A real LAN address and the host login appeared in section 2. | Replaced with `$MAIN_HOST` and a placeholder. The standing rule is that real addresses and the login live only in the untracked local notes |
| C-6 | **State label.** The control line used a state outside the register's vocabulary. | Now "Record; run in progress" |
| C-7 | **Duration figures.** GOV-009 section 4 counted the A/A noise floor as 5 hours; ten paired seeds with two arms are 20 hours. | Corrected in GOV-009 Amendment 5 and ASS-007 Amendment 7 |

## 9. Stop and restart of AB-00 with corrected code (2026-10-22)

The owner instructed: stop AB-00 and restart it with the corrected code (C-3).

| Item | Record |
|---|---|
| First job | `/workspace/experiments/doom_ab00_20261021`, driver PID `50482`; ran about 49 minutes; `baseline_a_s11` had finished its source phase and was about 4 minutes into its target phase |
| Stop | Driver ended first (so no later phase could start); the trainer then stopped through its `runlogs/STOP` file, not by killing it; the supervisor and trainer exited cleanly (about 4 minutes); a wrapper script that only waited out the phase timer was ended last. No trainer, supervisor, Doom or driver process remained |
| Disposition | Root renamed `doom_ab00_20261021_DISCARDED_uncorrected`, then **deleted on 2026-10-22 on the owner's instruction** (see ASS-013 section 6). **Its results are not usable:** the target-phase scores come from the uncorrected detector and logging interval. Nothing from it enters any A/A pool |
| What differs in the new stage | Three files only: `developmental_ai/environments/vizdoom_env.py` (C-2, cumulative evaluator totals), `developmental_ai/evaluator/detectors/doom.py` (C-2), `tools/eval_run.sh` (C-1, log every 256 decisions). Verified by a recursive diff of the two staged trees |
| New job | `/workspace/experiments/doom_ab00_20261022`, driver PID `56636`, started 2026-10-22 00:29 UTC; 353 files hashed into `eval_state.json` (the **new freeze manifest**); same command and seeds (11 to 20, two identical arms, source MAP01 to MAP28, held-out MAP29 to MAP32, 40/20 minutes) |
| Early checks | The composed run config shows `log_every: 256` and `held_out_maps` on every stream; the trainer log shows `kills_total` and `items_total` beside the per-life `kills` and `items` (for example a stream with `kills_total 8.0` and per-life `kills 0.0`, the case C-2 describes). These are wiring checks only, not results |
| Unchanged | The 40/20 sensitivity pilot is still owed (C-4); the 20 serial host-hours; the deployed `/workspace/devai` was not touched (its pre-existing `runlogs/STOP` and `games.enabled` are as found) |

Results remain provisional development evidence on the Doom proxy (ASS-007 Amendment 6).

**Follow-on queue.** AB-00 is the first stage of the unattended campaign [ASS-013](ass-013-unattended-test-campaign.md) (installed 2026-10-22). It attaches to this job and resumes it if the driver dies; the job itself was not restarted.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-015](ass-015-ab00-interim-analysis.md): AB-00 Doom A/A: interim analysis (16 of 20 runs)
- [ASS-016](ass-016-ab00-final-and-fresh-control.md): AB-00 final result and the fresh-control comparison
<!-- backlinks:end -->
