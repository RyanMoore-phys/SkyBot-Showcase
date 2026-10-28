# AB-00 Doom A/A: interim analysis (16 of 20 runs)

> **Document control:** [ASS-015](../README.md#assurance) · Assurance / result record · **State:** Record; interim, superseded by ASS-016 (AB-00 finished) · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-22. **Run:** [ASS-011](ass-011-main-evaluation-readiness.md) section 9 (restarted job,
`doom_ab00_20261022`), queued by [ASS-013](ass-013-unattended-test-campaign.md). **Evidence class:** development
evidence on the Doom proxy (ASS-007 14.1); nothing here is a program claim or a promotion decision.
**Method:** read-only. Nothing on `main` was changed. The small result files of the 16 finished runs (logs,
configs, phase records; about 3 MB compressed) were copied to the Mac, analysed with the project's own scorer
(`tools/eval_analyze.py score`) and a throwaway script, and the copies were deleted afterwards.

## 1. Summary

1. **Operationally smooth.** 16 of 20 runs finished (seeds 11 to 18, both arms), the 17th was in its source phase.
   0 crashes, 0 interruptions, 0 tracebacks, identical step budgets in every run (90,112 source steps, 49,152
   target steps), about 65 minutes per run, disk steady (178 GB free, finished runs pruned and gzipped by the
   campaign), VRAM 0.41 of the card, 220 ms per decision.
2. **The learning mechanisms look healthier than in any earlier run; behaviour has not moved.** Action
   attribution is 0.19 (it was 0.0076 before the real-time credit fix), state dependence I(S;A) 0.20 nats,
   explained variance 0.92, z-info 25 nats. But the policy is still 92% of uniform entropy after 40 minutes,
   there were **0 level exits in 128 stream-hours**, and kills and items per minute did not rise during training.
3. **The primary AB-00 metric carries no information, as built.** The project scorer reports
   `attained = []` and `at_entry = [first_kill, first_pickup]` for **all 16 runs**. The only checkpoint left to
   discriminate, `first_exit`, was never reached. The noise floor of the checkpoint metric is therefore undefined
   (every run scores the same), and any A/B scored on checkpoints would return "no difference" whatever the change.
4. **"Source = MAP01 to MAP28" is nominal.** Every source phase of every run stayed on MAP01 (a level list advances
   only by exiting a level, and nobody exits). Source training is one map; the world model's fresh-data loss
   against train loss widens from 0.95 to 1.75 over the phase, which is what memorising one map looks like.
5. **What can be salvaged now:** rate measures on the held-out maps have a measurable noise floor (section 5),
   so AB-00 can still deliver a usable calibration if the analysis, not the runs, is changed.

## 2. What was measured

Per run: source phase 40 minutes on MAP01 (8 streams), then target phase 20 minutes on held-out maps (each stream
pinned to one of MAP29 to MAP32). Counts are summed over the 8 streams from the evaluator record
(`kills_total` and `items_total` are cumulative adapter counters, `deaths` and `exits` the game's).

| | Source (40 min, MAP01) | Target (20 min, MAP29 to MAP32) |
|---|---|---|
| kills per run | 278 to 404 (mean 326) | 165 to 423 (mean 289) |
| items per run | 295 to 524 (mean 408) | 77 to 105 (mean 94) |
| deaths per run | 194 to 282 (mean 236) | 430 to 576 (mean 530) |
| exits | 0 in all 16 runs | 0 in all 16 runs |
| deaths per stream-hour | 44 (range 36 to 53) | 199 |

The target numbers are a different map, not a different agent: items drop to a quarter and deaths per
stream-hour rise 4.5 times at the same moment. **Source and target counts are not comparable and must never be
differenced as "progress".**

## 3. Is it learning? (signals over the 40-minute source phase, mean of 16 runs)

| Signal | start | middle | end | Reading |
|---|---|---|---|---|
| policy entropy, % of uniform over 16 actions | 99 | 95 | 92 (90 to 93) | moving, slowly; still near random |
| explained variance (critic) | 0.46 | 0.81 | 0.92 | the critic fits well |
| I(S;A), nats | 0.03 | 0.12 | 0.20 | the policy has become state dependent |
| action attribution | 0.01 | 0.17 | 0.19 | credit assignment now works (fix confirmed again) |
| episodic share of income, % | 89 | 82 | 83 | one term still dominates |
| fresh / train loss gap | 0.95 | 1.54 | 1.75 | the world model memorises the one map |
| z-info, nats | 10 | 21 | 25 | the latent is carrying information (no collapse) |

34% of PPO updates (59 of 176 logged) ended with the actor frozen by the KL stop. Per stream-minute during the
source phase (four equal time bins, mean of 16 runs, ± standard error): deaths 0.85, 0.70, 0.68, 0.72; kills
1.14, 0.97, 0.93, 0.97; items 1.34, 1.31, 1.21, 1.20. Deaths fell about 15% after the first quarter, **but kills
and items fell too**. That is consistent with a more cautious or less active agent, not a more competent one; the
data cannot separate the two. In the target phase all three rise (kills 1.54 to 1.84, items 0.45 to 0.60, deaths
2.88 to 3.21 per stream-minute): more activity on the new maps. That could be adaptation or could be the streams
settling into their pinned maps; no run isolates it.

**Verdict on learning:** the machinery that was broken (credit, attribution, critic, latent) now works. The
policy has not turned that into better play at this horizon, and nothing here shows an upward behavioural trend.
This agrees with ASS-003 (behaviour did not improve consistently) and extends it to 16 independent runs.

## 4. Why the primary metric is empty

`score_events` treats a checkpoint already held at the target phase's start as "at entry" and removes it. Here
the source phase is also Doom, and kills and pickups occur within the first minutes of MAP01, so both are held at
entry in every run; `first_exit` is never reached. All 16 runs score identically (no attained checkpoint, two at
entry). This is correct behaviour for the cross-game design (a skill already held should not count as learned in
the target), and the wrong instrument for a same-game proxy whose checkpoints are all either trivial or out of
reach. It is a property of the proxy, not a bug in the scorer.

## 5. The A/A noise floor that does exist (8 pairs, target phase, A minus B per seed)

| Measure | mean (A) | mean (B) | paired difference: mean, sd | sd as % of mean | rough minimum detectable effect, 5 paired seeds |
|---|---|---|---|---|---|
| kills | 303 | 276 | +27, 106 | 37% | about 130 (45%) |
| items | 92 | 96 | -5, 12 | 12% | about 15 (16%) |
| deaths | 516 | 545 | -30, 63 | 12% | about 80 (15%) |
| exits | 0 | 0 | 0, 0 | undefined | none |

(The minimum detectable effect is 2.8 times the standard error of a 5-pair mean, a coarse planning figure, not
the moving-block bootstrap of ASS-007.) Source-phase differences are smaller in relative terms (kills sd 39 on 326,
12%). Items and deaths on the held-out maps are the stable measures; kills are noisy. No A-versus-B difference is
distinguishable from zero for any measure, as it should be for identical arms: the A/A set is behaving as an A/A.

## 6. Do systems need to change? (recommendations; nothing was changed)

1. **Change the Doom proxy's metric, not the runs.** Score rate measures on the held-out maps (items per
   stream-minute and deaths per stream-hour are the quietest; survival time per life from the record's `tic` and
   `deaths` is a candidate) with `first_exit` kept as a ceiling. The raw logs already hold everything needed
   (evaluator line every 256 decisions), so this is an amendment to ASS-007 and the analysis code, with **no rerun**.
   The queued `an_*` stages read the same checkpoint scores and would otherwise report empty comparisons.
2. **Add a fresh-control arm.** The runner supports one (target phase only, 20 minutes, no source training) and
   none is queued. Without it nothing can say whether source training helps at all, which is the question the
   program exists to ask. About 3.5 host-hours for ten seeds.
3. **Make the source phase actually diverse (WP-6).** The 28-map list is not a pool. Spreading streams across
   maps is the structural change most likely to matter for adaptability, and it is currently the one A/B the
   campaign cannot run (its overlay needs a game-YAML merge in the runner). This is a runner change, not a
   learning-code change.
4. **Fix the two defects that sit on the intrinsic signal before judging WP-5 and WP-2/RW-1** (ASS-014 F-13: the
   guard clock is inverted so information gain pays only just after a life boundary; F-04: consolidation strength is
   cancelled by Adam). With episodic novelty at about 83% of income and the one action-dependent term effectively
   off, the policy is being paid mostly for seeing new frames.
5. **Add a reference floor.** No random-policy run exists, so "92% of uniform" and "0.7 deaths a minute" have no
   anchor. One cheap untrained-policy arm would give it.
6. **Longer horizon is a separate question.** The replay buffer stops growing at 10,000 transitions per stream
   (about the whole 40-minute phase), and 34% of updates are KL-stopped; both matter more for consolidation and
   long runs than for these one-hour seeds. Do not infer a learning-rate or KL change from this data: there is
   no ablation here.

## 7. Limits

- One configuration (tiny by 8, seeds 11 to 18, 8 of 10 pairs); the last 4 runs are not in this analysis.
- Counts mix streams and lives; streams within a run are not independent, so run-level sd is the honest unit.
- Within-run trends are descriptive. Activity changes could come from map phase, episode timing or stream
  settling; no control isolates them.
- No random or scripted reference exists, so none of these levels is "good" or "bad" in absolute terms.
- Minor log findings: the replay-ceiling warning (240 lines, informational) and a foundation-shadow module that
  switches itself off for the run on cost (14 lines). Neither affected results.

## 8. Resolution (owner decisions of 2026-10-22; plan [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md))

| Section 6 recommendation | Decision | What changes | Status |
|---|---|---|---|
| 1. Rate-based Doom metric | **Not pursued** | nothing. The limitation stands: no A/B can show a benefit on checkpoint times in this proxy, so A/B results are mechanism measures plus descriptive counts and promotion decisions are deferred (GOV-014 section 9) | open by decision |
| 2. Fresh-control arm | **Pursued** | a `fresh_control` stage (seeds 11 to 20, target only, 3.3 h) and a descriptive analysis join the campaign ([ASS-007](ass-007-adaptability-evaluation-protocol.md) Amendment 8, [ASS-013](ass-013-unattended-test-campaign.md)) | **queued 2026-10-22** (3.4 h, after AB-00) |
| 3. Diverse source phase (WP-6) | **Pursued** | the runner accepts a game-YAML overlay, the `wp6_diversity` arm rotates the source maps, a live 6-minute coverage check gates its A/B (item C) | **built and tested offline 2026-10-22; live check queued** |
| 4. Fix F-13 and F-04 | **Pursued** | two consumer tests corrected; consolidation strength also scales the learning rate (item B). Pre-registered: exactly two categorical scenarios flip to pass ([ASS-014](ass-014-categorical-harness-first-run-findings.md) section 4A) | **done 2026-10-22; prediction held** |
| 5. Reference floor | **Not pursued** | nothing; absolute levels stay unanchored | open by decision |

**Consequences recorded with the decision.**
- AB-00 keeps running on the older code; its configuration enables none of the changed paths, and parity tests make
  that a tested statement. Its results are an A/A of the *as-run* procedure (MAP01 source) and are labelled so.
- The 16-run numbers in this record are the baseline against which the diverse-source arm and the fresh control are
  read: mean 289 kills, 94 items and 530 deaths per target phase, run-level sd about 12% for items and deaths and 37%
  for kills.
- `t2_rw6_strategy` is blocked in the queue by [ASS-014](ass-014-categorical-harness-first-run-findings.md) F-24 (not
  part of these decisions) so that 10 host-hours are not spent on an A/B whose connected arm cannot run its own
  consolidation.
- **Update 2026-10-23.** AB-00 finished (20 of 20) and the fresh control ran: the final figures and the transfer comparison are in
  [ASS-016](ass-016-ab00-final-and-fresh-control.md) (no transfer visible on the held-out maps); queue incidents and the first host test
  results are in [ASS-017](ass-017-host-test-stages-and-queue-incidents.md).
- **Resolved items are updated here, not rewritten:** when GOV-014 section 10 records an item done, this table's
  status column is changed and the date added.

## 9. Related records

[ASS-011](ass-011-main-evaluation-readiness.md), [ASS-013](ass-013-unattended-test-campaign.md),
[ASS-014](ass-014-categorical-harness-first-run-findings.md),
[GOV-014](../governance/gov-014-corrective-plan-after-ab00.md), [GOV-011](../governance/gov-011-execution-record-gov-009.md),
[ASS-007](ass-007-adaptability-evaluation-protocol.md), [ASS-003](ass-003-policy-learning.md),
[GOV-009](../governance/gov-009-implementation-plan.md).

## 10. Addendum (2026-10-26): what the campaign results say, and why the drive is narrow rather than sparse

Appended; sections 1 to 9 are unchanged. Basis: the analysis records on `main` (`analysis/an_ab00`, `an_fresh_control`, `an_t1_wp2` to `wp6`), the reward lines in the baseline runs' logs (`stages/ab_baseline/runs/baseline_s1`, source and target), `configs/universal.yaml`, and [RES-006](../research/res-006-human-like-learning-gap-analysis.md). Nothing was run for this section and no code was changed.

### 10.1 What the results test

1. **The instrument is empty, so every comparison returns "no claim".** Every analysed comparison (the A/A pool, the fresh control, WP-2 to WP-6) scored no seeds. In the A/A pool `first_kill` and `first_pickup` hold at target entry in all ten seeds and `first_exit` is reached by no run, so every paired difference is exactly 0.000 with a zero-width interval and the five-seed rule cannot pass. This is the section 4 finding, now seen in every later record.
2. **It does not show lack of capability, and it does not show the game is too hard.** A weak agent and a good one both score zero on this instrument. The separate evidence for weak play is section 3 and [ASS-003](ass-003-policy-learning.md): 0 exits in 128 stream-hours, policy at 92% of uniform entropy after 40 minutes, kills and items not rising. That is consistent with weak play and also with a proxy whose only informative checkpoint is out of reach in a 40 plus 20 minute seed; no run separates the two.
3. **The T1 and T2 queue cannot answer a behaviour question in this proxy.** [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md) section 9 already defers promotion decisions for this reason; the stages still produce mechanism measures and descriptive counts.

### 10.2 The drive is narrow, not sparse

The reward is not sparse. In the baseline runs it is a steady +0.017 to +0.020 per step. The problem is what it pays for.

| Measured in `baseline_s1` | Source phase (MAP01) | Target phase |
|---|---|---|
| Census: base share of the drive | 96% | 82% |
| of which episodic novelty | +0.0175 per step | +0.0167 per step |
| of which learning progress | +0.0010 per step | -0.0000 per step |
| Ledger shares | episodic 86%, imagination 7%, empowerment 6%, novelty 1% | episodic 78%, novelty 10%, imagination 7%, empowerment 5% |
| Ledger alarm | `DOMINANT source=episodic`, share 0.86 for 9 segments | none in the lines read |
| Episodic event record | `sighting never`, `break never` | same |

Episodic novelty pays for views unlike this life's earlier views. It does not depend on what the action did and it is not tied to progress in the level. Learning progress, the term closest to a human's sense of getting better at something, is about zero. So the signal is dense, one term carries 78 to 96% of it, and it pays for seeing, not for succeeding.

### 10.3 Are the other human-learning inputs present?

Mostly present in code, mostly switched off in the baseline arm. [RES-006](../research/res-006-human-like-learning-gap-analysis.md) section 7 ranks the gaps (no durable-skill route in Doom, no consolidation, no curriculum, no planning or metacognition, a single dominant intrinsic signal). The universal config currently has these switched **off** by default: goals, outcome memory, planning, slots, attention memory, risk, meta-adaptation, representation, competence, adaptive optimisation, peer observation, curriculum and consolidation. On: real-time credit, the infrastructure ledger and the asynchronous world model. This is by design (build-only mode, one difference per A/B, [GOV-009](../governance/gov-009-implementation-plan.md) and [GOV-010](../governance/gov-010-relational-implementation-plan.md)): the T1 and T2 stages each switch one on against that baseline. The consequence is that the baseline and the A/A pool are, on purpose, the least human-like configuration, so "the drive has few inputs" describes the control, not the finished design. The chart's inputs that have no counterpart at all (relatedness, need-satisfaction, a wellbeing node, reflection, skills in Doom) are listed in RES-006 section 8.

### 10.4 Why the instrument is empty (likely causes, not tested)

- Kills and pickups happen in the first minutes of MAP01, so they hold at entry; exits are never reached.
- 40 plus 20 minutes may be too short for a near-uniform policy to find an exit; the ASS-003 fixes improved attribution and state dependence but not exits.
- 34% of PPO updates end with the actor frozen by the KL stop and updates are few (about 17 an hour), so the policy moves slowly.
- Episodic novelty pays for any new view, so exploring the first room again and again is nearly as good as leaving it.

### 10.5 Further ideas (none adopted; each is an owner decision)

1. **An intermediate checkpoint between "held at entry" and "never reached"**, learned or measured from the player's own view, not read from the engine: for example health lost or regained, a door or switch used (any change in the view that follows an own action), distance covered by the view-based place estimate, or a map change. It must stay evaluation-only (ASS-007).
2. **Rate and survival measures on held-out maps** (section 6 item 1, declined on 2026-10-22; the section 5 noise floor makes items and deaths usable). Revisit if the checkpoint idea fails.
3. **A longer target budget or a cheaper checkpoint map** (a small map or one whose exit is near the start) used only for calibration, so that some runs reach it and some do not.
4. **A reference floor**: one untrained-policy arm gives the anchor for every level quoted here (section 6 item 5, declined).
5. **Calibrate the A/A on a measure that varies** before spending more T1 and T2 hours; the queue's only way to return a claim is an instrument with variance.
6. **Re-weight or bound the dominant term**: cap the episodic share, or pay it only for views that the action changed, so learning progress and empowerment carry weight. This changes the learning, so it is a T1 change of its own, one difference at a time.
7. **Make the competing drives pay in Doom**: the skill and learning-progress paths need the surprise-times-agency trigger of [RES-005](../research/res-005-skill-discovery.md) (not built) before skills can form without an extrinsic spike.
8. **Reuse the data already collected**: the evaluator logs every 256 decisions, so any new measure above can be scored on the existing runs without rerunning them.

### 10.6 Limits

One configuration, one game, one baseline run read in detail (`baseline_s1`; the shares are consistent with the section 3 means over 16 runs). The causes in 10.4 are untested. The switch state in 10.3 is read from the default config, and each arm's overlay turns on its own block; no overlay was read for this section.
