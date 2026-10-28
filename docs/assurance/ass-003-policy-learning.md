# Why the policy stayed random, the fixes, and how learning is verified (2026-10-18)

> **Document control:** [ASS-003](../README.md#assurance-and-evidence) · Assurance / evaluation protocol · **State:** Protocol; in progress · **Custodian:** Repository maintainer · Revision history: Git.

Status: **fixes being built; this protocol is PRE-REGISTERED before any run of
the fixed code.** Results are appended below the protocol, never edited into
it.

## The finding

In every Doom run (Stage C, the mock deploy, the scale test), policy entropy
stayed at 2.68–2.73 nats. The log printed that as "70 % of max", but it
divided by ln 48. Doom has 16 valid actions (ln 16 = 2.77), so the policy was
**97–98 % uniform** after 40 minutes. It never left MAP01 and never exited.

Causes, verified in code (details in CLAUDE.md §10):

1. **Real-time action lag.**
   - `vizdoom_env._step_realtime` (and `player/adapter.step`) set the held
     buttons to a_t and return the newest frame at once.
   - The game clock played that whole interval with a_{t-1}.
   - So every transition was labelled one decision late. This affected:
     - the world model, inverse dynamics and empowerment;
     - curiosity's action gate (measured attribution 0.0076);
     - PPO.
2. **The dominant reward (episodic novelty, 72–89 %) barely depends on the
   action.** The learning-progress term was ≈0, because its gate depends
   on (1).
3. **The intrinsic weight annealed toward an extrinsic reward Doom never
   pays.** It reaches 0 in about 8 hours, and resets at every launch.
4. **PPO barely moved.**
   - A per-32-row KL stop froze the actor at ~0.02 nats per segment.
   - The bootstrap value was computed without proprio.
   - The critic and action head were reset on every relaunch.
5. **Dream distillation pulled toward a near-uniform teacher.** Adam cancels
   its small weight, and it ignored the action mask.
6. **Smaller issues.**
   - Masked logits counted toward the logit cap.
   - The 7 scout streams were paid by a different reward function from
     stream 0.
   - The farm damp read the engine's position.

**The fixes correct how credit is assigned and how the optimizer behaves.
They add no reward term and do not steer behaviour.** The game's kills,
items, deaths and exits stay evaluation-only.

## Pre-registered protocol

**Arms.** Two runs on `main`, run one after the other (RAM allows only one at
a time):

| Arm | Code | Switches |
|---|---|---|
| **ON** | fixed | every new switch on |
| **OFF** | same build | every new switch off = the old behaviour |

**Shared conditions.**
- The same starting brain: a copy of `main`'s brain. Its live-size world
  model is refused at resume, so both arms start a fresh tiny world model
  and a fresh policy.
- Same restored replay, tiny × 8 Doom streams, mock root on `main`, training
  paused.
- **Length:** 120 minutes each. If an arm is inconclusive, ON may be
  extended; OFF is never re-run selectively.
- **Snapshots:** the per-segment telemetry (`learning.jsonl`, run log) is the
  data.

**Mechanism criteria** (ON arm, absolute; these test the fixes themselves):

| # | Measure | Pass |
|---|---|---|
| M1 | lag probe (`tools/replay_lag_probe.py`) on replay written by the arm | best shift = 0 (OFF: expected −1) |
| M2 | action attribution, mean of the last 60 min | ≥ 0.05 (was 0.0076) |
| M3 | inverse-dynamics accuracy (lag probe at shift 0) | ≥ 2 × chance (chance = 1/16) |
| M4 | critic explained variance, mean of the last 60 min | > 0.2 |
| M5 | intrinsic mixer weight | constant over the run |

**Learning criteria** (does the policy learn anything, without reference to
its reward?):

| # | Measure | Pass |
|---|---|---|
| L1 | I(S;A) on the fixed probe states (nats; how much the action depends on the state) | rises: mean of the last 6 updates > mean of updates 2–7, with a bootstrap 95 % CI excluding 0; AND that rise in ON > the rise in OFF |
| L2 | conditional policy entropy / ln(valid actions), end of run | < 0.90 (was 0.97–0.98) |
| L3 | KL(π_first ‖ π_now) on the probe | grows over the run (the policy keeps changing, not frozen) |

**Behaviour, judged on the game's own counters (evaluation only, never fed
back):**
- per stream-hour: kills, items, deaths, distance travelled (oracle
  dead-reckoning line), maps reached, exits;
- for each, compare the last third against the first third within each arm.

| # | Measure | Pass |
|---|---|---|
| B1 | direction | ON's within-arm change is better than OFF's for at least 3 of the 4 rates (kills ↑, items ↑, deaths ↓, distance ↑) |
| B2 | magnitude | reported with per-segment bootstrap CIs; not gated (two hours is short) |

**Decision rules.**
- **M1–M5 all pass and L1–L3 pass:** learning is progressing. The fixes are
  confirmed; B1 is reported.
- **Mechanism passes but learning fails:** the credit is now right, but the
  objective does not reward anything action-dependent enough. Report, then
  examine the reward (episodic weight and sharpness are the owner's call)
  before claiming anything.
- **Mechanism fails:** a fix is not working. Bisect it with its switch, fix
  it, and re-run ON only.
- Persistence is checked inside the ON arm:
  - one relaunch at about 60 min: the policy is restored, I(S;A) and entropy
    are continuous, no boost;
  - one forced tiny→small grow in a separate short run: policy logits are
    identical across the grow, and the policy is not reset.

## Amendment 1 (2026-10-18, before any run of the fixed code; no results existed)

Reason: [RES-001](../research/res-001-central-behavioral-problem-and-plan.md)
asks that the real-time credit fix be judged on its own, with one change at a
time, and that behavioural claims rest on more than one seed. The original
two-arm design (ON vs OFF) could not separate the credit fix from the other
fixes. Changes:

- **Three arms instead of two:**

  | Arm | Switches |
  |---|---|
  | **OFF** | every new switch off |
  | **CREDIT** | only `realtime_credit` on; measurement-only telemetry on in every arm |
  | **ALL** | every switch on |

  "ON" in the criteria above means ALL. M1–M3 and L1–L3 are also reported
  for CREDIT versus OFF; that is the primary test of the timing fix.
- **Seeds:** B1 (behaviour) is gated only after a second seed of OFF and ALL
  (`--seed 1`). The first seed's behaviour is reported, not gated.
- **Two added measurement items:**
  - the distribution of decision durations (tics per interval) in each arm,
    because RES-001 asks whether variable durations are discounted
    consistently;
  - the actor gradient norm over the updates actually applied, plus the
    KL-freeze trip point (RES-001: the final-epoch value can hide an early
    freeze).
- **Evaluator logging:** `game.log_every: 512` in every arm's Doom config.
  This is evaluation-only, so it gives a counter line per stream per
  segment.

Nothing else changed: same criteria, thresholds, durations and decision rules.

## Amendment 2 (2026-10-19, still no results)

- **Fresh brain in both arms.** The protocol said "a copy of `main`'s
  brain". Its replay was written with the lagged labels, and restoring it
  would hand the fixed arm the very data under test. Both arms now start
  from nothing (tiny x 8, empty replay, fresh policy).
- **The runner is `tools/policy_learning_arms.sh`, the analysis is
  `tools/learning_report.py`.** The ON arm stops and relaunches once, at 60
  minutes, to test policy persistence.
- **Status: not run.** `main` was offline when this was written (no route
  for ~11 h).

## Results (2026-10-20, appended; the protocol above is unchanged)

Runs on `main`, tiny x 8 Doom, 120 minutes each, fresh brain per arm. Five
runs: OFF and ON at seed 0, ON re-run at seed 0 after fixing a bug in ON
(below), OFF and ON at seed 1. The ON seed-0 re-run and ON seed 1 stopped and
relaunched once at 60 minutes. Method: `tools/policy_learning_arms.sh`,
`tools/learning_report.py`; values are the mean of the first third and the
last third of each run's segments.

**Bug found by the first ON run.** The scouts' imagination policy lacked the
knowledge input, so the imagination-curiosity term raised on every call
(9,515 swallowed tracebacks, bonus silently 0). Fixed
(`_imagination_policy_fn`, `_imagination_curiosity_smoke` E, red without the
fix). The first ON run therefore ran without that term; the later runs have 0
tracebacks. Its mechanism and learning numbers matched the re-run.

| # | Criterion | Seed 0 (ON re-run) | Seed 1 (ON) | OFF (seeds 0 / 1) | Result |
|---|---|---|---|---|---|
| M1 | lag probe best shift | 0 on 8/8 streams | 0 on 8/8 streams | -1 on 8/8 / -1 on 8/8 | **Pass** |
| M2 | action attribution, last third | 0.154 (0.05 -> 0.15) | 0.200 | ~0.02 / 0.025 | **Pass** |
| M3 | inverse accuracy at shift 0 (probe) | 0.19-0.24 vs 0.12 chance | same range | peak at -1 | **Pass** (probe-based; the world model's own inverse accuracy was not logged) |
| M4 | critic explained variance | 0.95-0.98 | 0.98 | 0.88 / 0.93 | **Pass**, but OFF also exceeds 0.2, so it does not discriminate |
| M5 | intrinsic weight constant | 0.70 | 0.70 | fell 0.66 -> 0.53 / 0.67 -> 0.53 | **Pass** |
| L1 | I(S;A) rises, ON rise > OFF rise | 0.05 -> 0.23 (first hour) | 0.13 -> 0.32 | 0.06 -> 0.14 / 0.05 -> 0.19 | **Pass** (bootstrap CIs exclude 0; ON higher at the end in both seeds) |
| L2 | conditional entropy / ln(valid) < 0.90 at end | 0.876 | 0.866 | 0.939 / 0.918 | **Pass** (OFF fails in both) |
| L3 | KL from first policy grows | 0.08 -> 0.41 | 0.17 -> 0.38 | 0.19 -> 0.33 / 0.12 -> 0.39 | Grows in both arms; it does not separate them |
| B1 | behaviour direction (kills, items, deaths; distance not measured) | mixed | ON better on all 3 | worse on kills and items in both seeds | **Not met** |

**Persistence.** After the 60-minute relaunch the whole policy was restored
(reward fingerprint unchanged), and state dependence and entropy continued
without a reset. The forced tiny -> small grow check was not repeated here.

**Behaviour, stated plainly.** Neither arm reached an exit or left MAP01. In
seed 1 ON was better than OFF on kills (up 0.16 vs down 0.58 per
stream-hour), items (flat vs down 1.5) and deaths (down 32 vs down 14). In
seed 0 the re-run did not beat OFF: kills and items fell and deaths did not
improve over its first hour. Counters restart at a relaunch, so the second
half of an ON run is not comparable. The behaviour criterion therefore fails
the "consistent across seeds" bar and is **not** claimed. Distance travelled
was not extracted.

**Decision-rule outcome.** Mechanism criteria pass and the learning criteria
L1 and L2 pass in both seeds: the credit is now correct and the policy
depends on state more, with lower entropy, than before. This is the
"mechanism passes, behaviour not yet" branch: do not claim improved play. The
open question, per [RES-001](../research/res-001-central-behavioral-problem-and-plan.md),
is whether the objective rewards anything action-dependent enough, with
episodic novelty at 80-90 % of income unchanged.

*Note (2026-10-28, appended; nothing above is changed).* The analysis tool this record calls `tools/learning_report.py` is now `tools/segment_report.py`: the name belonged to the offline telemetry report of [OPS-004](../operations/ops-004-monitoring.md), which the per-segment tool had silently replaced on 2026-10-20 ([GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md) section 9.12). Its logic is unchanged.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-005](ass-005-testing-plan.md): Testing plan — proving the new machinery works, and keeps working
- [ASS-006](ass-006-realtime-action-lag-incident.md): Real-time action-lag incident (2026-10-18)
- [ASS-007](ass-007-adaptability-evaluation-protocol.md): Adaptability evaluation protocol (the program's main metric)
- [ASS-010](ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [ASS-015](ass-015-ab00-interim-analysis.md): AB-00 Doom A/A: interim analysis (16 of 20 runs)
- [GOV-005](../governance/gov-005-claims-evidence-register.md): Claims and evidence register
- [GOV-008](../governance/gov-008-human-like-learning-motivation.md): Human-like learning program: motivation, scope and change record
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [RES-006](../research/res-006-human-like-learning-gap-analysis.md): Human-like learning gap analysis: SkyBot against the AI-counterpart learning map
<!-- backlinks:end -->
