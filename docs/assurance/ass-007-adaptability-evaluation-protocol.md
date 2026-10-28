# Adaptability evaluation protocol (the program's main metric)

> **Document control:** [ASS-007](../README.md#assurance-and-evidence) · Assurance / evaluation protocol · **State:** Doom development protocol frozen by Amendment 6; later Minecraft program evaluation remains draft · **Custodian:** Repository maintainer · Revision history: Git.

**Date opened:** 2026-10-20. **Amended:** section 12 (owner decisions, before any run). **Status:** a draft of the procedure by which the
human-like-learning program ([GOV-008](../governance/gov-008-human-like-learning-motivation.md))
will judge whether SkyBot learns well. Amendment 6 freezes only the Doom
development proxy procedure; the final Minecraft program-level metric remains
draft until its later gates are met.

## 1. The question

The owner's definition (2026-10-20): SkyBot has learned well if it **adapts
quickly to unfamiliar environments**, measured by whether and how fast it hits
**specific checkpoints** the owner sets. Worked example from the owner: train
SkyBot only on Doom, then release it into Minecraft; if it cannot manage to
break wood, it has not learned well.

This measures adaptability and speed of pick-up in an environment the brain was
not trained on. It does not measure performance in the training game, and it
does not reward the checkpoints.

## 2. Evaluation-only rule

Checkpoints are detected from the game's own counters or events by the
evaluator. They are **never** a reward, a learning input or a target
([ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md)).
Detecting a checkpoint does not change the run.

## 3. Design

1. **Source phase.** A brain is trained on the source game(s) (today Doom only)
   under the configuration being evaluated.
2. **Target phase.** The trained brain is released into a target game it has not
   seen. The target phase has a fixed budget (section 6) and the same learning
   process as every game (one brain, no per-game forks).
3. **Checkpoints.** A fixed, ordered list set by the owner before any run. Each
   has an identifier, a plain description and an evaluator-side detector. They
   are listed in section 9 as an owner input.
4. **Per-checkpoint outcome.** The time (and decision count) to first attainment
   within the budget, or **not attained** (right-censored at the budget).
5. **Primary measure (proposed, to be frozen).** The *checkpoint-attainment
   curve*: the fraction of checkpoints attained by time *t*, summarised by its
   normalised area over the budget (0 to 1). Higher is better and earlier is
   better. Per-checkpoint times and the count attained are reported with it.

## 4. Arms

Arms differ only in the switch under test (GOV-008 section 4, principle 5).

| Arm | Meaning |
|---|---|
| **Baseline** | The code and configuration before the change. |
| **Changed** | The change on, everything else identical. |
| **Fresh control** (reported, not gated) | A brain with no source phase, run in the target game, to show how much of the result is transfer rather than learning from scratch. |

Arms share seeds and budgets. Paired by seed.

## 5. Evidence standard for a behaviour claim

Fixed by the owner (2026-10-20):

- **Five seeds**, each a run of **one hour**, per arm.
- A claim is **positive only if the change is positive on all five seeds.**
- On each seed, "positive" means better than baseline and **above that seed's
  region of uncertainty**; being only slightly above it is acceptable. One seed
  that is not above its region means the claim is not made.
- A mixed or null result is **recorded as such** and is not rounded up. A
  negative result is recorded as negative. Neither is hidden.

**Region of uncertainty (to be frozen).** The size of a difference that noise
alone could produce on one seed must be estimated before judging any change.
Two ingredients, both required:

1. *Within-seed uncertainty:* a bootstrap interval of the paired difference over
   that seed's own observations (checkpoint times; curve segments).
2. *A/A noise floor:* before the first claim, run Baseline against Baseline on
   five seeds. The spread of "differences" between identical arms gives the
   run-to-run noise, and the share of A/A seeds that falsely look positive
   gives the per-seed false-positive rate.

A seed counts as positive when its paired difference is on the favourable side
and its lower confidence bound is above zero **and** above the A/A noise floor.
The chance that five seeds all pass by luck is the per-seed false-positive rate
raised to the fifth power **if the seeds are independent**; independence is
assumed, not verified, and the A/A runs are what check it.

## 6. Budget and fixed conditions

- Target-phase budget per seed: **one hour** of run time (owner). Whether the
  hour includes the source phase or follows it is open (section 9).
- The configuration is frozen for the whole comparison, including stream count,
  model size and stepping rate. The development compute rule
  ([ADR-005](../decisions/adr-005-compute-pressure-order.md)) does not act during
  a frozen run.
- The streams count as one brain per run; peer observation, when it exists
  ([ADR-006](../decisions/adr-006-streams-may-observe-one-another.md)), is part of
  the arm under test and is not switched on or off mid-run.

## 7. Reporting

Each result record (appended to the result section of the protocol that applies,
never edited into this one) gives: the arms; the seeds; per-seed paired
differences with intervals and the A/A noise floor; per-checkpoint times;
censoring counts; compute and memory used; and any deviation from this
protocol, disclosed. Null and negative results are reported with the same
prominence as positive ones.

## 8. Relation to earlier evidence

[ASS-003](ass-003-policy-learning.md) used two seeds of 120 minutes in one game
and judged mechanism and learning signals plus a behaviour direction. It does
**not** meet this standard and is not re-labelled to. Its behaviour result stays
"not met". It remains the reference for the real-time-credit mechanism.

## 9. Open items to settle before this protocol is frozen

1. **The checkpoint list** and each detector (owner input). The ordered
   Paper Minecraft progress list was supplied in Amendment 5; its external
   recorder and event rules must still be implemented and validated before
   scored runs.
2. **The target game.** The sole target is the owner's persistent Paper
   Minecraft world. No generated or proxy world is selected.
3. **What the one hour covers:** source training, target exposure, or each.
4. **The source phase:** its length and content, held fixed across arms.
5. **The summary statistic**, if the attainment-curve area is not what the owner
   wants.
6. **The uncertainty method:** confirm the two-part region in section 5, and the
   confidence level.
7. **The seed list**, fixed in advance (including the A/A seeds).
8. **The budget for failure:** what to do when a run crashes or a seed cannot
   complete (a missing seed means no claim).

## 10. Results

(None. Appended, as a separate dated record, after runs made under a frozen
version of this protocol.)

## 11. Related records

[GOV-008](../governance/gov-008-human-like-learning-motivation.md),
[ASS-003](ass-003-policy-learning.md),
[FND-001](../foundation/fnd-001-ab-protocol.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md),
[ADR-006](../decisions/adr-006-streams-may-observe-one-another.md),
[RES-006](../research/res-006-human-like-learning-gap-analysis.md).

## 12. Amendment 1 (2026-10-20, before any run; no results exist)

Source: the owner's second round of answers, and determinations marked
**(assistant)**, made under the owner's delegation to settle unanswered points
consistently with the owner's directive. Each is revisable by the owner.

**12.1 Learned, not declared (always).** Nothing in an arm may declare meaning,
skills, goals or checkpoints to the agent. Checkpoints are detected by the
evaluator only and are never visible to the learner. This holds for every arm,
including peer observation (ADR-006) and any future mechanism.

**12.2 What the one hour covers.** The hour is a **complete seed run**: source
training, target exposure and everything else, so each seed yields enough
quality data to judge a change on **short time segments** of learning. Section
6's "one hour of target-phase budget" is replaced by this. The aim is to promote
changes that make SkyBot learn **quicker**.

- **Promotion** means adopting a change into the baseline. A change is promoted
  when it shows quicker learning within the hour on **all five seeds**, each
  above its region of uncertainty (section 5). Owner to confirm this reading of
  "promote".
- **Limit recorded:** a change whose benefit appears only over horizons longer
  than an hour would not be promoted by this rule. That is a limitation of the
  standard, not a finding about any change.
- **(assistant) Default split:** two thirds source training and one third target
  exposure (40 minutes and 20 minutes), identical in every arm and never tuned
  per change.

**12.3 Checkpoints (assistant proposal, owner to confirm or replace).** A
checkpoint is a plain, ordered, evaluator-side event in the target game. For the
owner's Minecraft example, an initial ladder of increasing difficulty:
(1) any movement away from the spawn point; (2) the first block broken;
(3) the first log collected (**break wood**, the owner's example); (4) several
logs collected; (5) a block placed; (6) a crafting step. For any other target
game, the same shape applies: a ladder from first effect on the world to a
first reproducible useful effect. Source-game checkpoints (for Doom: first
kill, first pickup, first exit) are reported but are not the main metric.

**12.4 Checkpoint detectors are per game and pluggable.** A new game needs a
detector file of the same shape as every other game's, and nothing in the
learning code changes to add it.

**12.5 Uncertainty and seeds (assistant).** The two-part region in section 5 is
adopted: a 95 % bootstrap interval of the paired difference on the seed's own
observations, and an A/A noise floor from five baseline-versus-baseline seeds.
A seed is positive when its lower bound is above zero and above the A/A noise
floor. Main seeds: 1 to 5. A/A seeds: 11 to 15. A crashed seed is re-run once
with the same seed; if it fails again, no claim is made.

**12.6 Interim target game.** The owner has not decided whether an interim game
will be used before Minecraft returns. If none is chosen, the main metric waits
for Minecraft and development tests use their own mechanism measures, labelled
as such. If one is chosen, it is labelled an interim proxy. Either way, adding a
game must be easy (GOV-008 WP-16).

**12.7 Status of section 9.** Items 3 (what the hour covers), 4 (source phase,
defaulted in 12.2), 5 (summary statistic, unchanged), 6 (uncertainty method,
adopted), 7 (seed list, set) and 8 (failure rule, set) are settled by this
amendment. Items 1 (the final checkpoint list) and 2 (the interim game) stay
with the owner.

## 13. Amendment 2 (2026-10-21, before any run; no results exist)

Source: the owner's instruction of 2026-10-21 on test types; details marked
**(assistant)** are judgement calls, revisable.

The protocol applies to **two kinds of A/B**, with the same evidence standard
(sections 5 and 12):

- **T1, block test:** baseline versus baseline plus one block.
- **T2, relation test:** both blocks present, the relationship cut versus connected.
- **Unit test (a T1 variant):** baseline versus a block plus its positive
  relationships connected, used for conditional promotion.

**(assistant)** Arms of a T2 share runs with the changed arm of the block's T1 when
the configuration, code and seeds are identical. Each result record states its type
and the identifier of its **development baseline** (the baseline plus blocks not yet
promoted). The A/A noise floor is re-estimated after every three promotions and
whenever the development baseline changes materially. Rules in full: GOV-010
section 4A.

## 14. Amendment 3 (2026-10-21, response to the branch audit; no results exist)

Source: [ASS-009](ass-009-audit-response-adjustments-branch.md), findings 3.1 to 3.3. The
rules marked **(assistant)** are judgement calls and revisable; those that need an owner
decision are listed in section 14.9.

**14.1 Evidence classes.** Every result is labelled as one of two classes.

- **Development evidence:** mechanism and learning measures, held-out-map probes and
  T1/T2 results that are not the adaptability metric. It supports an engineering decision
  (provisional promotion) and is **never** presented as evidence for the program's main goal.
- **Program evidence:** adaptability to an unfamiliar game at the owner's checkpoints
  (this protocol, once frozen and runnable). Only this supports a claim that SkyBot
  learns well.

**What may proceed before the target game and checkpoints are fixed (assistant):**
Phase 0 enablers, building blocks and relationships, mock runs, and development-evidence
T1/T2 tests. Nothing in that work may be reported as program evidence.

**14.2 Statistical unit (assistant).** The independent unit is the **seed**: one
paired difference per seed per test. Within a seed, observations are repeated events from
one evolving run and are not independent, so:

- the within-seed uncertainty uses a **moving-block bootstrap** over time (block length set
  from the measured autocorrelation of the summary statistic), never an ordinary bootstrap
  over checkpoint times or curve points;
- the within-seed interval is descriptive; the **gate** is the five-seed rule with each seed
  above its region of uncertainty, as the owner set.

**14.3 A/A calibration (assistant).** Five A/A seeds are too few to estimate a noise floor
or a false-positive rate reliably. The calibration therefore pools: the first A/A set is
**ten** paired A/A seeds (not five), and every later baseline-versus-baseline pair created
by the program (the shared runs in GOV-010 4B) is added to the pool. The rate of "all five
positive" under A/A is estimated empirically from repeated five-seed subsets of the pool,
not from the per-seed rate raised to the fifth power, because seed independence is not
verified. The expected number of false promotions across *m* tests is reported as *m*
times that empirical rate.

**14.4 Power (assistant).** Before freezing, the pooled A/A variance sets the smallest
effect the five-seed design could detect. Every result record states it. A null result is
reported as "no effect larger than this was detectable".

**14.5 Checkpoint edge cases (assistant).**

- *Censoring:* a checkpoint not reached within the hour is right-censored at the budget.
- *Already achieved at target entry:* excluded from that seed's score and flagged (it
  indicates source-phase leakage or a trivial checkpoint).
- *Impossible or never reached by any arm in the A/A pool:* excluded as uninformative and
  listed in the result record.
- *Dependent checkpoints:* the checkpoint list declares its order and prerequisites; a later
  checkpoint is scored only if its prerequisites are reached, and its time is measured from
  target entry.
- *Time segments:* evaluation windows are fixed **five-minute** windows (twelve per hour).

**14.6 Crashes and restarts (assistant).** A crashed seed is re-run once with the same
seed. Crash counts per arm are reported. If crashes cluster in one arm, stability is
reported as an outcome and no claim is made from the re-run seeds.

**14.7 Multiple comparisons (assistant).** Each test has one pre-registered primary
measure. Across the program, false-promotion risk is controlled by the five-seed rule and
estimated as in 14.3. The final combined test (AB-15) is the independent confirmation.

**14.8 Promotion levels and reversal (assistant).**

- **Provisional promotion** (an engineering decision): T1 or T2 positive on all five seeds,
  plus a **no-regression guard** (held-out-map performance and a retention probe not worse
  beyond the noise floor). It lets the program build on the change.
- **Confirmed promotion:** supported by program evidence once the main metric is runnable.
- **Reversal:** in the combined test, each provisionally promoted change gets a leave-one-out
  arm. A change is reverted if removing it improves the program evidence on all five seeds
  above the uncertainty region (the same standard as promotion, applied symmetrically), or by
  owner decision. A change that cannot be tested on program evidence stays provisional and
  is listed as such.

**14.9 Needs the owner.** (a) The target game and the checkpoint ladder; (b) whether 40
minutes of source training and 20 of target exposure is an acceptable budget for the
transfer question (the split is an assistant default; before freezing, a pilot with other
splits should show how sensitive the score is); (c) accepting ten A/A seeds instead of five
(10 extra hours).

## 15. Amendment 4 (2026-10-21, owner decisions; superseded in part by Amendment 5)

Source: the owner, 2026-10-21, answering the open items of ASS-009.

- **Target game: Minecraft.** The program's main metric is run with Minecraft as the
  unfamiliar target after source training on Doom. Minecraft's return to the universal
  brain is still deferred until after Doom (GOV-008 2.2), so the metric stays non-runnable
  until then. No interim proxy game is chosen; none is needed yet. Anything run before
  then is development evidence (section 14.1).
- **Checkpoint ladder:** Amendment 5 supplies the owner's ordered progress list and
  initial survivability definition for the persistent Paper Minecraft world. Recorder
  implementation, validation, and the thrivability summary remain open as specified there.
- **Accepted:** ten A/A seeds (14.3), and the 40-minute source / 20-minute target split with
  a sensitivity pilot before the protocol is frozen (14.9 b, c).
- **Approved:** provisional versus confirmed promotion, the no-regression guard and the
  reversal rule (14.8).
- **Status of section 9 and 14.9:** Amendment 5 supplies the later Minecraft
  checkpoint list. Amendment 6 selects held-out Doom maps for current development
  A/A and A/B tests; the Minecraft target is reserved for the later program test.

## 16. Amendment 5 (2026-10-21, owner checkpoint list; no results exist)

Source: the owner's Minecraft checkpoint list and definitions of survivability
and thrivability, supplied 2026-10-21. This amendment records the owner's
outcomes; implementation details below marked **(assistant procedure)** are
proposals for the recorder and require validation before a run is scored.

**16.1 Target and progress checkpoints.** The target is the owner's persistent
Paper Minecraft world after 40 minutes of Doom source training. This protocol
uses that Paper world only; it does not use a generated or local game world.
The owner's ordered progress ladder is:

| Order | ID | Owner checkpoint | Evaluator event definition to validate |
|---:|---|---|---|
| 1 | `move` | Move | First horizontal displacement of at least 1 block from target-phase entry. |
| 2 | `break_any` | Break any block | First positive increment in the game `mine_block` statistic for a solid block. |
| 3 | `break_stone` | Break stone | First positive increment for the `stone` block. |
| 4 | `break_wood` | Break wood | First positive increment for a log or wood block (`*_log` or `*_wood`). |
| 5 | `craft_sticks` | Craft sticks | First positive increment in `craft_item[stick]`. |
| 6 | `craft_table` | Craft a crafting table | First positive increment in `craft_item[crafting_table]`. |
| 7 | `wooden_pickaxe` | Make a wooden pickaxe | First positive increment in `craft_item[wooden_pickaxe]`. |
| 8 | `collect_cobblestone` | Collect cobblestone | First positive inventory count of `cobblestone`. |
| 9 | `stone_pickaxe` | Craft a stone pickaxe | First positive increment in `craft_item[stone_pickaxe]`. |
| 10 | `craft_furnace` | Craft a furnace | First positive increment in `craft_item[furnace]`. |
| 11 | `smelt_iron` | Find and smelt iron | Record iron ore mined; attain when a later `smelt_item[iron_ingot]` increment is observed. |

Later checkpoints require the earlier checkpoint in this order, as section
14.5 requires for dependent milestones. A checkpoint true at target entry is
flagged `at_entry` and excluded. These event definitions use game counters for
measurement only; the checkpoint IDs and detector output must never enter the
policy observation, reward, replay labels, or controller.

**16.2 External recorder (assistant procedure).** Use a read-only evaluator
recorder attached to the persistent Paper Minecraft world. It must capture
timestamped video or screenshots and, where available from the Paper server,
authoritative player statistics, inventory, death, and smelting events. Keep
the recorder outside the SkyBot process: its output must not enter policy
observations, rewards, replay labels, or controller inputs. Score the ordered
IDs above from predeclared visual/server evidence rules, with two independent
reviewers resolving ambiguous frames while blinded to arm labels. Preserve raw
recordings and the decision log. Before freezing, validate each checkpoint,
non-attainment, ordering, and timestamp against scripted Paper-world
trajectories. If authoritative server evidence or visual evidence cannot
reliably distinguish a checkpoint, mark that detector unavailable and do not
start scored runs.

Because this is a persistent shared world, each scheduled run must begin from
the same documented world snapshot and player reset state. Record snapshot
identifier/hash, player inventory/position, world time, weather, difficulty,
and connected players before and after each run. Restore the snapshot between
runs and verify the reset before resuming. If a verifiable reset is unavailable,
stop the sequence: sequential runs would be contaminated by changed blocks,
inventory, or world conditions. Do not treat world persistence as evidence of
independent trials.

**16.3 Secondary outcomes.** Survivability is the elapsed target-phase time to
the first death, right-censored at the target budget if no death occurs.
Thrivability means survival while progressing through the ordered checkpoints;
its summary statistic and gate are deferred until the initial recorder and
checkpoint ladder work. For the first AB-00 set, report survival time as a
secondary descriptive outcome, and do not combine it with the primary
checkpoint-attainment score. No survivability or thrivability result is yet
defined as a promotion gate.

**16.4 Status and remaining pre-run gates.** The owner has supplied the
progress order and outcome concepts and confirmed the persistent Paper world as
the sole target. The protocol remains **draft, not frozen** until the external
recorder and checkpoint rules are validated on Paper, snapshot restoration is
verified between trials, and the 40/20 split sensitivity pilot required by
section 14.9 is resolved. The ten-pair AB-00 schedule and seed list remain
seeds 11–20; no run or result is implied by this amendment.

## 17. Amendment 6 (2026-10-21, owner instruction; Doom development testing)

The owner clarified that implementation-plan A/A and A/B experiments on this
branch use **Doom**, with held-out Doom maps as development evidence. These
experiments are not the final cross-game evaluation and must not be described
as Minecraft results. For the development evaluation, source-phase Doom maps
are MAP01–MAP28 and held-out target-phase maps are MAP29–MAP32. The split is
40 minutes source and 20 minutes held-out Doom per one-hour seed, with the same
map partition, baseline and seed schedule in every arm. Score the Doom
evaluator events `kills`, `items`, and `exits` as first-kill, first-pickup,
and first-exit milestones, alongside each package's pre-registered mechanism
measures. Keep these outcomes evaluator-only.

AB-00 is the first Doom A/A calibration: ten paired seeds (11–20), two
identical baseline arms, seed-major interleaving, 40/20 minutes per arm and
the existing crash/retry rule. It estimates development noise on this Doom
held-out-map proxy only. Then run the plan's five-seed Doom A/B procedures,
one pre-registered change at a time, after the A/A pool is available. A
separate A/A calibration on the persistent Paper Minecraft world is required
when the branch's implementation tests are complete and the program-level
Minecraft comparison is ready; Doom calibration cannot substitute for it.

This instruction supersedes Amendment 4's interpretation that no development
target is selected. Amendment 5's Paper Minecraft checkpoint list remains
reserved for the later program-level evaluation; it does not define the Doom
development checkpoints.

**Frozen AB-00 procedure.** Freeze date: 2026-10-21. Baseline: the staged
`Adjustments` working tree, identified by the file-hash manifest in the run
state on `main`; the two A/A arms use byte-identical code and configuration.
Use seeds 11–20, seed-major interleaving, 40 source minutes followed by 20
held-out target minutes per arm. Source maps are MAP01–MAP28; target maps are
MAP29–MAP32. The evaluator records `first_kill` (kills ≥ 1), `first_pickup`
(items ≥ 1), and `first_exit` (exits ≥ 1) from Doom's evaluator-only event log.
The primary development score is the ASS-007 checkpoint-attainment curve area
over the 20-minute target window; report per-checkpoint times, censoring,
within-seed moving-block intervals, the pooled A/A noise floor, empirical
all-five rate and minimum detectable effect. Also record crashes, throughput,
map coverage, deaths, resource usage and deviations. An absent or malformed
target evaluator log makes that seed unscorable; do not fill missing events
with learner reward or telemetry. A second crash of a seed is final and means
no A/A claim from this set. These results calibrate Doom development tests only.

The staged code hash, composed phase configs and run-state order are retained
under the experiment root on `main`. The later Minecraft program evaluation
requires its own frozen protocol and target-specific A/A calibration.

## 18. Amendment 7 (2026-10-21, review corrections; no results exist)

Source: ASS-011 section 8 (review of the AB-00 setup).

- **Scope of the freeze.** Amendment 6's "frozen AB-00 procedure" freezes only the **Doom
  development A/A**. This protocol as a whole stays draft, not frozen (control line and
  16.4). Its freeze date is superseded by the restart carried out on 2026-10-22 (ASS-011
  section 9): the corrected evaluator code and logging interval are part of the frozen
  baseline, and the freeze date and file-hash manifest are those of the restarted job
  (`doom_ab00_20261022`, started 2026-10-22 00:29 UTC). The first job's partial results are
  discarded.
- **Evaluator events (17).** `first_kill` and `first_pickup` are scored from the adapter's
  cumulative `kills_total` and `items_total`, not the game's per-life counters; the Doom
  evaluator logs a line every 256 decisions per stream in evaluation roots. A target phase
  with no usable evaluator lines is unscorable (as Amendment 6 states).
- **Deviation recorded.** The 40/20 sensitivity pilot (14.9, accepted by the owner as a
  condition of freezing) was not run before the Doom AB-00 procedure was frozen. AB-00 is
  therefore provisional with respect to the split. The pilot remains required.
- **Duration.** AB-00 is ten paired seeds with two identical arms at one hour each: **20
  serial hours**. The earlier "10 extra hours" in 14.9 (c) meant the increase over five
  paired seeds.

## 19. Amendment 8 (2026-10-22, owner instruction; interim analysis of AB-00)

Source: [ASS-015](ass-015-ab00-interim-analysis.md); plan [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md).

- **Fresh control added to the Doom development procedure.** One arm, `fresh_control`, seeds 11 to 20, target phase
  only (20 minutes, held-out MAP29 to MAP32), runs after the test suites and before the T1/T2 A/Bs (about 3.3
  serial hours). It is the fresh control defined in section 4, reported and not gated. It is compared with the 20
  source-trained target phases of AB-00 and read **descriptively** (scorer output plus the evaluator counts), because
  the checkpoint scorer holds first kill and first pickup "at entry" for every source-trained Doom run
  (ASS-015 section 4) while a fresh run has no entry state. No benefit claim is drawn from this comparison.
- **Open, by owner decision.** The primary Doom metric is not changed in this amendment. Its limitation is
  recorded: no A/B can show a benefit on checkpoint times in this proxy, so A/B results are reported as mechanism
  measures and descriptive counts and their promotion decisions are deferred (GOV-014 section 9).
- **Source phase.** The source maps listed as MAP01 to MAP28 were not all used (every source phase stayed on MAP01);
  the WP-6 rotation arm is the planned remedy (GOV-014 item C). The AB-00 results remain valid as an A/A of the
  *as-run* procedure (MAP01 source) and are labelled so wherever they are used.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-005](../architecture/arc-005-game-onboarding-standard.md): Game onboarding standard
- [ASS-005](ass-005-testing-plan.md): Testing plan — proving the new machinery works, and keeps working
- [ASS-010](ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [ASS-011](ass-011-main-evaluation-readiness.md): Main evaluation readiness and run record
- [ASS-012](ass-012-categorical-system-testing-protocol.md): Categorical system testing protocol
- [ASS-013](ass-013-unattended-test-campaign.md): Unattended test campaign
- [ASS-016](ass-016-ab00-final-and-fresh-control.md): AB-00 final result and the fresh-control comparison
- [ASS-019](ass-019-predictive-coding-core-comparison-protocol.md): Predictive-coding core versus the RSSM: comparison protocol
- [ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md): Two selectable world-model cores until a comparison decides
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [GOV-010](../governance/gov-010-relational-implementation-plan.md): Relational implementation plan: connecting the learning blocks as the map does
- [GOV-011](../governance/gov-011-execution-record-gov-009.md): Execution record: block implementation plan (GOV-009)
- [GOV-012](../governance/gov-012-execution-record-gov-010.md): Execution record: relational implementation plan (GOV-010)
- [GOV-013](../governance/gov-013-categorical-system-testing-plan.md): Categorical system testing: coherence within and between learning groups
- [GOV-016](../governance/gov-016-predictive-coding-core-plan.md): Implementation plan: predictive-coding core versus the RSSM (WP-20)
- [RES-007](../research/res-007-hard-and-deferred-differences.md): Hard and deferred differences: what cannot yet be built reliably
- [RES-010](../research/res-010-predictive-coding-core.md): Predictive coding as the world model's latent core
<!-- backlinks:end -->
