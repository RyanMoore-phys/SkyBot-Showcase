# AB-00 final result and the fresh-control comparison

> **Document control:** [ASS-016](../README.md#assurance) · Assurance / result record · **State:** Record; AB-00 complete, fresh control complete · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-23. **Runs:** AB-00, `doom_ab00_20261022` (20 runs, seeds 11 to 20, two identical arms), and the
campaign stage `fresh_control` (10 runs, seeds 11 to 20), both queued by
[ASS-013](ass-013-unattended-test-campaign.md). **Supersedes** the interim figures in
[ASS-015](ass-015-ab00-interim-analysis.md) (16 runs) where they differ. **Plan:**
[GOV-014](../governance/gov-014-corrective-plan-after-ab00.md) item A. **Evidence class:** development evidence on
the Doom proxy ([ASS-007](ass-007-adaptability-evaluation-protocol.md) 14.1); no program claim, no promotion decision.
**Method:** read-only. The small result files were copied to the Mac, analysed with the project's scorer and a
throwaway script, and the copies were deleted. Nothing on `main` was changed by the analysis.

## 1. Summary

1. **AB-00 completed cleanly:** 20 of 20 runs, 0 crashes, 0 interruptions, about 21 wall-clock hours, identical step
   budgets in every run.
2. **Source training did not help on the held-out maps, as measured here.** The 20 source-trained target phases and the
   10 fresh-control target phases (no source training at all) produced the **same** items (95.5 against 94.0 per run)
   and deaths (528 against 527), and **fewer** kills (284 against 339; paired by seed, 2 of 10 seeds favour source
   training). This is the first time the program's central question has been asked of data, and the answer is
   "no transfer visible", not "transfer".
3. **The trained policy is measurably different** (less random, more state-dependent, a better critic) **and that did not
   turn into different outcomes.** Learning happened; benefit on the held-out maps did not appear.
4. **The campaign's own comparison is empty, as predicted.** The scorer reports "no claim" for every seed because every
   source-trained run holds first kill and first pickup at entry and nobody reaches first exit
   ([ASS-015](ass-015-ab00-interim-analysis.md) section 4). The comparison above is therefore descriptive, from the
   evaluator counts.
5. **The A/A noise floor over ten pairs** is in section 4: items and deaths vary 11 to 17% between identical arms,
   kills 36%.

## 2. AB-00 final: the source phase (40 minutes, 20 runs)

All 20 source phases ran on **MAP01 only** (the listed MAP01 to MAP28 are a progression the agent never advances;
[GOV-014](../governance/gov-014-corrective-plan-after-ab00.md) item C addresses this).

| Measure | Mean | Range over runs |
|---|---|---|
| kills | 333 | 278 to 404 |
| items | 423 | 295 to 540 |
| deaths | 240 | 194 to 282 |
| exits | 0 | 0 in all 20 runs |

| Learning signal | First reading | Last reading | Range of last |
|---|---|---|---|
| policy entropy, % of uniform over 16 actions | 99 | 91.8 | 89 to 93 |
| explained variance (critic) | 0.49 | 0.93 | 0.84 to 0.96 |
| I(S;A), nats | 0.03 | 0.20 | 0.17 to 0.24 |
| action attribution | 0.01 | 0.19 | 0.15 to 0.25 |

These agree with the interim 16-run figures (ASS-015 section 3) to within rounding.

## 3. The fresh control against the source-trained target phases

Same target configuration for both groups: 20 minutes, 8 streams, held-out MAP29 to MAP32 (all four seen in both).
The only difference is the source phase (40 minutes on MAP01, or none).

| Measure (per target phase, summed over 8 streams) | Source-trained (n = 20) | Fresh control (n = 10) | Difference |
|---|---|---|---|
| kills | 284 (sd 69) | 339 (sd 61) | -55 (-16%) |
| items | 95.5 (sd 10.6) | 94.0 (sd 4.6) | +1.5 (+2%) |
| deaths | 528 (sd 38.7) | 527 (sd 20.4) | +1.5 (0%) |
| exits | 0 | 0 | none |

**Paired by seed** (mean of the two source-trained arms minus the fresh run of the same seed, ten seeds):

| Measure | Mean difference | sd | Seeds favouring source training | Rough standard error | Ratio |
|---|---|---|---|---|---|
| kills | -55.4 | 85.6 | 2 of 10 | 27 | about -2.0 |
| items | +1.6 | 5.8 | 7 of 10 | 1.8 | about +0.9 |
| deaths | +1.5 | 29.9 | 5 of 10 | 9.5 | about +0.2 |

The kills difference is about two standard errors in the *unfavourable* direction, before any correction for looking
at three measures; it is suggestive, not established. Items and deaths are indistinguishable.

**How the two groups differ in the learner**, last readings of the target phase (mean of runs):

| Signal | Source-trained | Fresh |
|---|---|---|
| policy entropy, % of uniform | 90.1 | 95.5 |
| explained variance | 0.86 | 0.77 |
| I(S;A), nats | 0.23 | 0.11 |
| action attribution | 0.13 | 0.14 |

**Within the 20-minute phase** (per stream-minute, four equal time bins, mean of runs):

| Measure | Source-trained | Fresh |
|---|---|---|
| kills | 1.53, 1.43, 1.70, 1.82 | 1.64, 1.74, 1.90, 2.12 |
| items | 0.46, 0.49, 0.56, 0.62 | 0.45, 0.47, 0.55, 0.53 |
| deaths | 2.87, 3.08, 2.96, 3.21 | 2.66, 2.95, 3.04, 2.82 |

Both groups become more active over the phase; the fresh control starts with more kills per minute and keeps its lead.

## 4. The A/A noise floor over ten pairs (target phase, A minus B)

| Measure | Mean A | Mean B | Difference: mean | sd | sd as % of mean |
|---|---|---|---|---|---|
| kills | 301.5 | 266.6 | +34.9 | 103.0 | 36% |
| items | 96.0 | 95.1 | +0.9 | 16.4 | 17% |
| deaths | 515.9 | 540.7 | -24.8 | 58.6 | 11% |

The items noise rose from 12% (eight pairs, ASS-015) to 17% with all ten; deaths stayed near 11%. No A-versus-B
difference is distinguishable from zero, as it should be for identical arms.

## 5. What the results mean

- **The transfer gap on this proxy is about zero.** Training for 40 minutes on MAP01 gave no advantage on
  MAP29 to MAP32 in items or deaths, and no advantage in kills. If the proxy is a fair test, SkyBot is not yet adapting
  better because of what it learned; it learned something (section 2) that these measures do not reward.
- **Why that is plausible, not proven.** A single-map source phase; a world model that memorises that map (the
  fresh-to-train gap grew from 0.95 to 1.75, ASS-015); a policy only 8% below uniform. The diverse source phase
  (GOV-014 item C) is the direct test of the first explanation and is queued.
- **It does not show the agent cannot learn.** Items and deaths on these maps may be dominated by map hazards and
  engine mechanics that a policy at 90% of uniform cannot influence; both groups die about once every 18 seconds.
- **It changes what a later A/B can claim.** An effect on these measures would have to exceed the noise in section 4
  (items and deaths about 11 to 17% per run; five paired seeds detect about 15 to 16% by the rough rule in ASS-015
  section 5). A modest real improvement would be invisible.

## 6. Limits

- **Code snapshot.** AB-00 ran on the code before the GOV-014 changes; the fresh control ran after them. The changed
  paths (information gain, consolidation strength, the WP-6 overlay) are off in both configurations, and the parity
  tests ([GOV-014](../governance/gov-014-corrective-plan-after-ab00.md) section 6) are the evidence that behaviour is
  the same; they are not a proof.
- Ten fresh runs against twenty trained ones; one configuration (tiny by 8); counts mix streams and lives, so the
  run is the honest unit.
- Three measures were examined; no multiplicity correction is applied.
- The checkpoint scorer carries no information here; no moving-block bootstrap record exists for these data.
- No random-policy reference was run (owner decision), so "90% of uniform" and absolute rates remain unanchored.

## 7. Related records

[ASS-015](ass-015-ab00-interim-analysis.md), [ASS-013](ass-013-unattended-test-campaign.md),
[ASS-011](ass-011-main-evaluation-readiness.md), [ASS-007](ass-007-adaptability-evaluation-protocol.md),
[GOV-014](../governance/gov-014-corrective-plan-after-ab00.md),
[ASS-017](ass-017-host-test-stages-and-queue-incidents.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-018](ass-018-campaign-progress-2026-10-23.md): Campaign progress check, 23 October 2026 (evening)
- [ASS-019](ass-019-predictive-coding-core-comparison-protocol.md): Predictive-coding core versus the RSSM: comparison protocol
<!-- backlinks:end -->
