# Human-like learning program: motivation, scope and change record

> **Document control:** [GOV-008](../README.md#governance) · Governance / program motivation · **State:** Proposed (a living change record; sections 9 and 10 are appended as work proceeds) · **Custodian:** Repository maintainer · Revision history: Git.

**Date opened:** 2026-10-20. **Revised:** 2026-10-20, twice, to record the owner's
answers to the first draft's open questions (sections 2.2, 2.3, 9 and 10). **Branch:**
the working branch (`Improv` when first drafted; `Adjustments` as of 2026-10-21, which is where these records now live). **Status of every
item below:** intent or plan. Nothing in sections 5 to 8 is built unless section
9 says so, and nothing here is a claim that a change improves learning. **Plan status (2026-10-21, updated):** the implementation plans now exist, [GOV-009](gov-009-implementation-plan.md) (blocks) and [GOV-010](gov-010-relational-implementation-plan.md) (relationships); both are Proposed; their **scope was approved by the owner on 2026-10-21**, and code changes still wait for the owner's first instruction. Statements in this record dated 2026-10-20 that say the plan is "next" are a historical snapshot.

## 1. Purpose

This record states why SkyBot's learning is about to be changed, what will be
changed and in what order, which documents and A/B tests the changes require,
and what is not yet decided. It is the parent of the work: later records link
back here, and section 9 logs each change as the owner issues instructions.

## 2. The owner's goal and decisions

### 2.1 The goal and the success metric

The goal is to make SkyBot's learning as **human-like** as achievable, to give it
the best chance of learning as well as it can. The owner produced a map of the
dependencies of human learning translated into their AI counterparts
([`ai_learning.mmd`](../../assets/ai_learning.mmd), rendered in
[`ai_learning_full.png`](../../assets/ai_learning_full.png)).
[RES-006](../research/res-006-human-like-learning-gap-analysis.md) compares
SkyBot with that map; that comparison is the factual basis for this record.

**Success is adaptability** (owner, 2026-10-20). SkyBot has learned well if it
adapts quickly to unfamiliar environments, judged by whether and how fast it hits
specific checkpoints the owner sets. Example: train SkyBot only on Doom, then
release it into Minecraft; if it cannot break wood, it has not learned well. The
checkpoints are measured by the evaluator and are never rewards. The procedure is
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md) (a draft that
is not yet frozen).

### 2.2 Decisions recorded on 2026-10-20

| Topic | Decision | Recorded in |
|---|---|---|
| Main metric | Adaptability and speed to pick up unfamiliar environments, by owner-set checkpoints | ASS-007 |
| Order of work | Take each difference from RES-006 **one by one**; do not start changes before an implementation plan exists | Sections 7, 8 |
| Episodic novelty weight | To be revisited soon, as part of the one-by-one order, not first | WP-5 |
| Minecraft | Its return to the universal brain is **deferred**; Doom is the easier first target | ADR-004, ASS-007 section 9 |
| Peer observation | Streams **may observe one another** for cooperation and collective learning | ADR-006 |
| When a change needs more compute | Reduce the model first; at saturation, fewer streams; if those are low, slower stepping | ADR-005 |
| Evidence for a behaviour claim | **Five seeds of one hour**; positive on **all five**, each above its region of uncertainty (slightly above is enough) | ASS-007 section 5 |
| Universality | The brain must be **completely universal to every game** | ADR-001, section 4 |
| Legacy brain | **None on this branch**; the owner holds copies | ADR-004 |
| Openness | Everything is open to change and optimisation | Section 4, principle 9 |
| Merge | When every difference has been tested and addressed; null and negative results recorded honestly | Section 8 |

### 2.3 Further decisions (owner, 2026-10-20, second round)

| Topic | Decision | Recorded in |
|---|---|---|
| **Learned, not declared** | **ALWAYS.** Every mechanism, signal and checkpoint must be learned from experience; nothing is scripted, declared or told to the agent. Not revisable except by an explicit owner change. | Section 4 principle 1; root `CLAUDE.md` section 0; ASS-007 12.1 |
| The one hour | Covers source training, target exposure and everything: a complete seed run, so each seed gives enough quality data to judge a change on short time segments and to **promote** changes that make SkyBot learn quicker (reading of "promote" to be confirmed) | ASS-007 12.2 |
| Minecraft-only code | Removed, unless it is literally the method by which SkyBot interacts with Minecraft (game-dependent scripts and configuration every game needs). Anything else that cannot be made universal is removed on the assistant's judgement; Minecraft-only reward files are the owner's example. | ADR-004 Amendment 1 |
| Peer observation | A stream may observe a peer's **chosen actions** as well as its frames and sound | ADR-006 Amendment 1 |
| Interim game before Minecraft | Not decided. **Adding any new game must be easy**, for the loop, the evaluator's checkpoint detector and the game interface | WP-16; ASS-007 12.4, 12.6 |
| Unanswered points | The assistant determines them, consistently with the owner's directive and goals; each is marked "(assistant)" and is revisable | Section 10 |

### 2.4 Testing decisions (owner, 2026-10-21)

| Topic | Decision | Recorded in |
|---|---|---|
| Two test types | **T1** tests individual blocks; **T2** tests with and without the relationship; both are A/B under the five-seed standard | GOV-010 section 4A; ASS-007 section 13 |
| Build order for relationships | A relationship is built once its two blocks **exist**, before the blocks are promoted | GOV-010 sections 1, 4A, 5 |
| Everything else | Left to the assistant's judgement, which may be situation-dependent (shared runs, minimal wiring set, cluster versus line tests, conditional promotion as a unit, development baseline, noise-floor re-estimation) | GOV-010 section 4A |

### 2.5 Response to the branch audit (2026-10-21)

Source: [ASS-009](../assurance/ass-009-audit-response-adjustments-branch.md). Judgement calls
are the assistant's and revisable.

**What "one difference at a time" means.** Three different things: *one-at-a-time
implementation* (one package per change); *one-at-a-time causal testing* (each T1/T2
changes exactly one thing between its arms, apart from declared minimal wiring and unit
tests); and *prerequisite wiring* (declared exceptions with defined arms). See GOV-010 4B.

**What "addressed" means** (the merge gate). A difference is addressed when it is one of:
promoted (provisionally or confirmed); rejected with the null or negative result recorded;
deferred to RES-007 or RES-009 **with the owner's acknowledgement**; or declared deliberate.
The merge list shows each difference's state, and which promotions are only provisional
(ASS-007 14.8).

**Evidence classes.** Development evidence supports engineering decisions; only program
evidence (adaptability at the owner's checkpoints) supports the claim that SkyBot learns
well (ASS-007 14.1).

**Scope approval status.** The owner directed: WP-5 (episodic novelty weight, later), WP-12
(peer observation permitted; mechanism undecided), WP-15 (legacy and Minecraft-only
removal) and WP-16 (easy game onboarding). WP-14 is implied by the success metric. **All
other packages (WP-1 to WP-4, WP-6 to WP-11, WP-13, WP-17 to WP-19) are the assistant's
proposals awaiting the owner's approval**, as are the relation clusters of GOV-010. **(Superseded by section 2.6: all were approved on 2026-10-21.)**

**Cross-references.** RES-007 and RES-009 cite each other; when a deferral changes, update
both in one change.

### 2.6 Owner decisions, third round (2026-10-21)

| Topic | Decision | Recorded in |
|---|---|---|
| Target game | **Minecraft**, as the unfamiliar game after Doom source training | ASS-007 15 |
| Checkpoint ladder | A combination of **progress through the game**, **survivability** and **thrivability**; the owner will set it later, "when we get there" | ASS-007 15 |
| Interim proxy | None chosen for now | ASS-007 15 |
| Ten A/A seeds; 40/20 split with a pilot before freezing | **Accepted** | ASS-007 15 |
| Promotion levels and reversal rule | **Approved** | ASS-007 14.8 |
| Work packages and relation clusters | **Approved as scope** (all of WP-1 to WP-19 and RW-1 to RW-15, R0, R1). The order stays provisional and is set by the owner's instructions | GOV-009, GOV-010 |
| Meaning of "addressed" | **Approved** (promoted provisionally or confirmed; rejected with the null or negative result recorded; deferred with the owner's acknowledgement; or deliberate) | Section 2.5 |

**Approving scope is not the first instruction to change code.** The plans' *scope* is
approved; **no code is authorised until the owner issues the first instruction.**

## 3. Why this is needed (evidence)

- The incident and fixes in [ASS-006](../assurance/ass-006-realtime-action-lag-incident.md)
  and the comparison in [ASS-003](../assurance/ass-003-policy-learning.md) show
  that the learning machinery now runs and is measurable: transitions are
  labelled with the right action, action attribution rose from about 0.02 to
  0.15-0.20, and the policy's dependence on the state grew.
- The same evidence shows that **play did not consistently improve**: no exit,
  one map, behaviour counters that disagreed between seeds.
- RES-006 sections 7 and 8 show why: most of the structure that turns experience
  into durable, transferable, self-directed knowledge is absent or partial.
- [RES-001](../research/res-001-central-behavioral-problem-and-plan.md) set the
  evidence gates (transition integrity, action dependence, useful intrinsic
  signal, skill validity, behavioural progress, persistence and transfer). Only
  the first two have been met. The owner's metric (2.1) sits at the last gate.
- ASS-003 used two seeds of 120 minutes in one game. It does **not** meet the
  standard the owner has now set (five seeds, one hour) and is not re-labelled
  to.

## 4. Principles that bound every change

1. **LEARNED, NOT DECLARED, ALWAYS.** Meaning and skill are earned from experience.
   Human-likeness is pursued through mechanisms, never scripted behaviour.
2. **One universal brain.** Every learning component runs identically in every
   game; there are no per-game forks ([ADR-001](../decisions/adr-001-shared-universal-brain.md)).
   A change that only works in one game is re-derived, not switched off per game.
3. **No engine reads as learning inputs or rewards.** Game counters, including
   checkpoints, are evaluation-only ([ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md)).
4. **Do not steer.** A change improves how SkyBot learns; it does not tell it
   what to do. Task outcomes are measured, not rewarded.
5. **Switches for A/B.** A change under test sits behind a switch whose off
   position reproduces the previous universal-brain behaviour, so the baseline
   arm can be run. Compatibility with the legacy brain is **not** required
   ([ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md)).
6. **Falsify first.** Each change ships with a test shown to fail without it, and
   a pre-registered A/B per [FND-001](../foundation/fnd-001-ab-protocol.md) before
   any claim of benefit.
7. **Human-likeness is a hypothesis, not a proof.** A mechanism is kept only if
   it helps the measured outcome; the map is a guide to what to try.
8. **Document as you go.** New records follow the naming and registration rules
   in [`docs/README.md`](../README.md); results are appended, never edited into a
   protocol; null and negative results are recorded as such.
9. **Everything is open to change and optimisation**: code, configuration,
   architecture and defaults. Principles 1-4 above come from the owner's earlier
   standing rules and stay in force unless the owner changes them explicitly
   (confirmed or revised in section 10).
10. **Compute yields in a fixed order** ([ADR-005](../decisions/adr-005-compute-pressure-order.md)).
11. **Adding a game must be easy.** A new game needs only its game interface (adapter, launcher, configuration) and a checkpoint detector, both of the same shape as every other game's; no learning code changes.
12. **Game-specific code is translated or removed.** Code that only works in one game is re-derived from what any player has so it runs in every game, or it is removed; it is kept only if it is the game interface itself (ADR-004 Amendment 1).

## 5. Scope: what the program may change

**Two layers (2026-10-21).** Layer A is the set of *blocks* below (work packages WP-1 to WP-19, planned in [GOV-009](gov-009-implementation-plan.md)). Layer B is the set of *relationships* between the blocks, compared with the owner's map in [RES-008](../research/res-008-relationship-gap-analysis.md) and planned in [GOV-010](gov-010-relational-implementation-plan.md) as relation work packages (R0, R1 and RW-1 to RW-15). A relationship is built once both of its blocks exist, before either is promoted; blocks and relationships are tested separately (section 2.4).

From RES-006, in the order the gaps leave the most of the map unrealised. Each
work package (WP) is a *candidate* difference; the owner's instructions decide
the order, and each is taken **one at a time** (section 8). "Gap" cites RES-006
sections 5 and 8.

| WP | Candidate change | Gap addressed | Why it matters for learning |
|---|---|---|---|
| WP-1 | Game-agnostic skill discovery and validation (surprise caused by the agent's own action, effect keys, probation) as designed in [RES-005](../research/res-005-skill-discovery.md) | No durable-skill route in Doom; empty knowledge store; empty episodic record | Without it the map's knowledge, capability and transfer outcomes have no source. |
| WP-2 | Consolidation and forgetting protection: an offline phase, replay into the policy, weight consolidation, generative replay | No consolidation loop; policy unprotected; replay never trains the policy | Humans consolidate during sleep; SkyBot only measures forgetting. |
| WP-3 | Curriculum and learnability frontier | No curriculum; one map | Keeps the agent near what it can learn next. |
| WP-4 | Decision-time planning and metacognition: search over imagined futures, uncertainty, calibration, strategy selection | Feed-forward acting; no calibration | Lets the agent deliberate and know what it does not know. |
| WP-5 | Intrinsic-signal rebalance: information gain, competence, and the **episodic novelty weight and sharpness** (owner: revisit soon) | One term carries 80-90 % of income | Reduces dependence on a single, possibly proxy-able signal. |
| WP-6 | Data diversity: more Doom maps first; **Minecraft and other games later** (deferred, ADR-004); domain randomisation | One game, one map | The widest gap in the map's environment group, and the precondition for the main metric. |
| WP-7 | Attention and working memory in the policy; longer-range context | No attention live; no long context | Selective focus is central to human learning. |
| WP-8 | Self-evaluation wired into effort and strategy; a self-model | Self-evaluation not connected | Persistence and strategy depend on it in the map. |
| WP-9 | Adaptive optimisation: learning-rate scheduling, adaptive exploration temperature and discount | Fixed schedules | Plasticity should change with experience. |
| WP-10 | Autonomous scaling (actuated autoscaler; grown policy verified live), **reconciled with ADR-005's compute order** | Advice only | Developmental growth needs to run unattended. |
| WP-11 | Risk and safe-exploration module | No learned danger estimate | Cautious exploration is part of human learning. |
| WP-12 | **Peer observation between streams** for cooperation and collective learning (permitted by ADR-006; mechanism undecided) | Streams never interact | Observation of peers is a major human channel; the oxytocin/belonging links in the map. |
| WP-13 | Meta-learning and automated hyperparameter adaptation | Hand-set hyperparameters | Learning to learn. |
| WP-14 | The evaluation machinery: checkpoint detectors, held-out environments, 5-seed 1-hour runs, the A/A noise floor | No held-out evaluation | Needed to say whether anything above helped; implements ASS-007. |
| WP-15 | **Remove the legacy brain and all Minecraft-only code that is not the game interface** (translate what can be made universal, remove the rest, per ADR-004 Amendment 1), and make the universal brain the only path, including the base configuration the universal configuration is composed on | Legacy paths and compatibility still in the tree | ADR-004. Owner: nothing legacy on this branch. |
| WP-16 | **Make adding a game easy**: one template for a game interface (adapter, launcher, install step, configuration) and one for a checkpoint detector, so a new game, including any interim game, needs no change to the learning code | Per-game work is spread across files that differ in shape | Needed for the transfer metric and for "completely universal to all games". |
| WP-17 | **Run continuity and tooling** (start at boot, time-based checkpoints, software version match) | Power loss and version mismatch (RES-006 4) | Added by [GOV-009](gov-009-implementation-plan.md). |
| WP-18 | **Queryable episodic memory** (retrieval of past outcomes into the policy) | No event-level memory the agent can query | Added by GOV-009. |
| WP-19 | **Representation, abstraction and robustness** (contrastive loss, augmentation, more horizons) | Hierarchical abstraction, contrastive learning, augmentation | Added by GOV-009. |

The map's DNE items (RES-006 section 5.13) are in scope as differences; they are
carried by WP-8 (autonomy, competence), WP-12 (belonging, relatedness) and are
otherwise unassigned until the implementation plan.

Deliberately out of scope (RES-006 section 3): pretraining, demonstrations,
imitation, human feedback and language. Reopening any is for the owner and
would need its own decision record. (Peer observation, WP-12, is not imitation
of demonstrations; whether it becomes imitation of a peer is for its research
design.)

## 6. A/B tests this program requires

Every test follows [FND-001](../foundation/fnd-001-ab-protocol.md) and the
standard in [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md):
**five seeds, one hour each, positive on all five seeds and above each seed's
region of uncertainty**, with an A/A noise-floor run first. The hypothesis,
metric, arms, seeds and failure conditions are frozen in a protocol record
**before** results exist. Evaluation counters stay evaluation-only. Where the
adaptability metric cannot yet be run (the target game is deferred), a test uses
the mechanism and learning measures of its own protocol, labelled as such, and
makes no adaptability claim.

| ID | Question | Arms | Primary evidence (measurement only) | Needs first |
|---|---|---|---|---|
| AB-00 | What is the run-to-run noise floor for current development tests? | Baseline vs baseline (A/A), ten paired seeds on held-out Doom maps | Development noise floor and false-positive calibration per ASS-007 Amendment 6 | Frozen Doom protocol; WP-14 |
| AB-01 | Does correct action credit improve learning? | Done: [ASS-003](../assurance/ass-003-policy-learning.md) (two seeds, 120 min; below the new standard) | Reported there | None |
| AB-02 | Does skill discovery (WP-1) produce skills that reproduce their effect above the base policy's rate? | Off vs on | Probation pass rate on held-out invocations; learned-option gate on a non-empty bank | WP-1 built; trigger measured on recorded play (RES-005 step 0) |
| AB-03 | Does policy consolidation (WP-2) reduce forgetting when the task changes? | Off vs on, with a map or game change mid-run | Forgetting probes; performance on the earlier task after the change | WP-2; WP-6 for a task change |
| AB-04 | Does a learnability-based curriculum (WP-3) beat a fixed sequence? | Fixed vs curriculum | Progress rate on held-out maps | WP-3, WP-6 |
| AB-05 | Does decision-time planning (WP-4) change outcomes? | Reactive vs planning | Checkpoint measure; act-latency cost | WP-4 |
| AB-06 | Does calibrated uncertainty (WP-4) improve exploration or safety? | Off vs on | Calibration error; death rate; learning progress | WP-4 |
| AB-07 | What does rebalancing intrinsic reward (WP-5) change, including the episodic weight? | Episodic weight and information-gain variants | Income shares; farming flags; checkpoint measure | The order reaches WP-5 |
| AB-08 | Does more data diversity (WP-6) improve generalisation? | One map vs several | Held-out map performance | WP-6, WP-14 |
| AB-09 | Does attention or longer context (WP-7) help? | Off vs on | State dependence; held-out performance; compute cost | WP-7 |
| AB-10 | Does wiring self-evaluation into effort (WP-8) help? | Off vs on | Learning progress; time to a milestone | WP-8 |
| AB-11 | Does adaptive optimisation (WP-9) help? | Fixed vs adaptive | Update stability; progress rate | WP-9 |
| AB-12 | Does unattended scaling (WP-10) improve progress per hour? | Advice vs actuated | Progress per hour; resize cost | WP-10, ADR-005 thresholds |
| AB-13 | Does peer observation (WP-12) help? | Isolated vs observing streams | Progress rate; **diversity of visited states** | WP-12 mechanism chosen |
| AB-14 | Does meta-learning (WP-13) beat fixed hyperparameters? | Fixed vs adapted | Progress rate across tasks | WP-13 |
| AB-15 | Is the combined program better than the baseline at adaptability? | Baseline vs full program | **The adaptability metric** (ASS-007) | All earlier tests; the target game available; checkpoints set |

A result that is null or negative is recorded as such, the change is left off or
removed, and RES-006 is updated honestly.

## 7. Documents this program will produce

Identifiers are assigned only when a document is created: the next unused one in
its series, never recycled ([`docs/README.md`](../README.md) section 2).

**Created so far (2026-10-20):**

| Record | Class | Purpose |
|---|---|---|
| [ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md) | Decision | No legacy brain on this branch (supersedes ADR-003 here). |
| [ADR-005](../decisions/adr-005-compute-pressure-order.md) | Decision | Reduce model, then streams, then stepping rate. |
| [ADR-006](../decisions/adr-006-streams-may-observe-one-another.md) | Decision | Streams may observe one another, within ADR-001 and ADR-002. |
| [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md) | Protocol (draft) | The main metric and the five-seed standard. |
| [RES-006](../research/res-006-human-like-learning-gap-analysis.md) | Record | The difference report. |

**Created on 2026-10-21:** [GOV-009](gov-009-implementation-plan.md) (the implementation plan) and [RES-007](../research/res-007-hard-and-deferred-differences.md) (hard and deferred differences).

**Created on 2026-10-21 (second layer):** [RES-008](../research/res-008-relationship-gap-analysis.md) (every line of the map against SkyBot), [GOV-010](gov-010-relational-implementation-plan.md) (the relational implementation plan, built on GOV-009) and [RES-009](../research/res-009-hard-and-deferred-relationships.md) (lines that cannot be built reliably yet).

**Still to be written (tentative):**

| Record | Class | Purpose | Timing |
|---|---|---|---|
| ~~Implementation plan~~ | Created as GOV-009 (2026-10-21) | Awaiting the owner's approval before any code | Done |
| **Current system limitations** (assurance record, next free `ASS-` ID) | Record, dated | The dated statement of what the current system cannot do, with evidence for each; the baseline every later result is compared against. **Required by the owner.** | After the plan, before changes |
| ASS-007, frozen | Protocol | Resolve its section 9 items and freeze. | Before AB-00 |
| One research design per work package (next free `RES-` IDs) | Proposed, then In progress | Mechanism, switches, tests, risks. WP-1 already has RES-005. | Before each build |
| One protocol per A/B test (next free `ASS-` IDs) | Protocol | Frozen hypothesis, metrics, seeds, duration, failure conditions; results appended separately. | Before each run |
| Incident records (next free `ASS-` IDs) | Record | For any failure the program causes or finds, as with ASS-006. | As needed |
| **Game onboarding standard** (architecture record, next free `ARC-` ID) | Proposed | The template and checklist for adding a game, including its checkpoint detector (WP-16). | With WP-16 |
| Updates to RES-006 | Record revision | Rows move from absent to present with evidence; never rewritten silently. | After each result |
| Updates to this record | Change log | Section 9 per instruction; section 10 per decision. | Continuous |

Each new controlled document is registered in [`docs/README.md`](../README.md)
in the same change, and `tools/check_docs.py` must pass.

## 8. How work proceeds

1. The owner issues an instruction. It is logged in section 9 before work starts.
2. The implementation plans exist (GOV-009, GOV-010) and had their scope approved on 2026-10-21; no code changes until the owner's first instruction. After that, the differences are taken
   **one by one**, each universal (it must hold in every game, ADR-001), none
   bundled with another.
3. For each difference: its research design, then its protocol, then the build
   behind a switch with a test shown to fail without it.
4. The A/B in section 6 is run under its protocol and ASS-007's evidence standard;
   results are appended.
5. RES-006 and the root `CLAUDE.md` are updated with the evidence. A change that
   did not help is recorded as null or negative and left off.
6. **The owner merges the branch when every difference has been tested and
   addressed**, some of which may have proved null or negative. Nothing is
   committed or deployed unless the owner asks.

## 9. Change log (appended as instructions arrive)

| Date | Instruction or change | Records touched | State |
|---|---|---|---|
| 2026-10-20 | Program opened; map comparison completed; this record drafted | RES-006, GOV-008 | Proposed |
| 2026-10-20 | Owner answered the first draft's questions: adaptability metric with checkpoints; one difference at a time, plan first; Minecraft deferred; streams may observe each other; compute order; five seeds of one hour, all positive; universal brain; no legacy on this branch; everything open to change; merge when all addressed | GOV-008, ADR-004, ADR-005, ADR-006, ASS-007, ADR-003 (status), root `CLAUDE.md` (branch override) | Proposed |
| 2026-10-20 | Second round of owner answers: **learned not declared, always**; the one hour covers source training, target exposure and everything; remove Minecraft-only code except the game interface (translate or remove on judgement); streams may observe peers' chosen actions; new games must be easy to add; interim game undecided; assistant may determine unanswered points. Assistant determinations recorded in section 10. | GOV-008, ADR-004 (Amendment 1), ADR-005 (Amendment 1), ADR-006 (Amendment 1), ASS-007 (Amendment 1), root `CLAUDE.md` | Proposed |
| 2026-10-21 | Implementation plan written (GOV-009) with per-difference theory, intended change, explicit changes, testing and inclusion procedure; hard and deferred differences recorded (RES-007); WP-17 to WP-19 added. No code changed. | GOV-008, GOV-009, RES-007 | Proposed |
| 2026-10-21 | Both implementation plans built in build-only mode (no tests, no runs, no promotion): GOV-009 (19 block packages; WP-15b deferred) and GOV-010 (R1, RW-15 and clusters RW-1 to RW-14, narrowed to observation or deferral where a block or a measurement is missing). Every switch default OFF. Record: GOV-011, GOV-012. Tests, mock runs, T1/T2/unit A/Bs, R0 audit and promotion are all still to do. | GOV-011, GOV-012, GOV-009, GOV-010, RES-009, ARC-005, ARC-006, ASS-010 | Proposed |
| 2026-10-21 | Owner decisions (third round): target game Minecraft; checkpoint ladder (progress, survivability, thrivability) set later; ten A/A seeds and 40/20 accepted; promotion levels, scope of all work packages and relation clusters, and the meaning of "addressed" approved. Docs only; code still awaits the first instruction. | GOV-008, GOV-009, GOV-010, ASS-007, ASS-009, root `CLAUDE.md` | Proposed |
| 2026-10-21 | Branch audit applied (ASS-009): evidence classes, statistics, promotion levels, corrected run sharing, learned-versus-declared review, removal test, compute-configuration rule, peer-observation geometry, scope approval, stale statements fixed. Docs only. | GOV-008, GOV-009, GOV-010, ASS-007, ADR-004, ADR-005, ADR-006, RES-007/008/009, ASS-009, root `CLAUDE.md` | Proposed |
| 2026-10-21 | Owner: two test types (T1 block, T2 relationship); a relationship is built once both blocks exist, before promotion; rest to assistant judgement. Recorded in GOV-010 4A, GOV-009 Amendment 1, ASS-007 Amendment 2. No code changed. | GOV-008, GOV-009, GOV-010, ASS-007 | Proposed |
| 2026-10-21 | The owner asked for the relationships (the lines) of the ideal learning map to be compared and planned, built on top of GOV-009: block first, then relations. Created RES-008, GOV-010 and RES-009. No code changed. | GOV-008, RES-008, GOV-010, RES-009 | Proposed |
| 2026-10-21 | Owner clarified that current implementation-plan A/A and A/B tests run on Doom and held-out Doom maps; the Minecraft checkpoint evaluation is a later final-performance comparison after branch testing. | ASS-007 Amendment 6, GOV-009 WP-14, GOV-008 AB-00 | In progress |

## 10. Open questions and determinations

**Resolved by the owner on 2026-10-20** (see 2.2 and 2.3): order of work (one by
one, plan first), definition of success (adaptability by checkpoints), the
episodic novelty weight (later), Minecraft (deferred), social learning (allowed,
including peers' chosen actions), the compute order, the run policy (five seeds,
one hour, all positive), the one hour (covers everything), the legacy brain
(removed; Minecraft-only code removed except the game interface), the principles
(**learned, not declared, always**; the others stay in force), branch and merge
(merge when every difference is tested and addressed).

**Determined by the assistant under the owner's delegation** (revisable; each
follows the owner's stated directive):

1. **Order of the one-by-one differences (provisional).** Enablers first, since
   every later test needs them: WP-15 (legacy and Minecraft-only removal), WP-16
   (easy game onboarding), WP-14 (evaluation machinery, including AB-00), and the
   current-limitations record. Then WP-5 (episodic novelty weight, which the owner
   asked to revisit soon), WP-1, WP-2, WP-6, WP-3, WP-4, WP-12, WP-7, WP-8, WP-9,
   WP-11, WP-13 and WP-10. The implementation plan sets the final order.
2. **Saturation and low-stream thresholds** for ADR-005: ADR-005 Amendment 1
   (provisional).
3. **Merge gate.** The branch merges when every difference has been tested and
   addressed, null and negative results included. AB-15 (the combined adaptability
   test) is run and recorded whatever its outcome; it does not have to pass for
   the merge, because the owner's gate is that all differences are addressed.
4. **Source-to-target split, seeds, uncertainty method, failure rule,
   checkpoint proposals:** ASS-007 section 12.
5. **Legacy removal scope:** the rule in ADR-004 Amendment 1.

**Still open (owner only):**

1. **The final checkpoint ladder** and its detectors: the owner will set it later; its composition is progress, survivability and thrivability (ASS-007 15).
2. ~~The interim game~~ Resolved 2026-10-21: the target is Minecraft; no interim game for now.
3. ~~The reading of "promote"~~ Resolved 2026-10-21: provisional and confirmed promotion, as in ASS-007 14.8.

## 11. Related records

- [RES-006](../research/res-006-human-like-learning-gap-analysis.md): the
  difference report this program responds to.
- [`ai_learning.mmd`](../../assets/ai_learning.mmd): the target map.
- [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md),
  [ASS-003](../assurance/ass-003-policy-learning.md),
  [ASS-006](../assurance/ass-006-realtime-action-lag-incident.md),
  [RES-001](../research/res-001-central-behavioral-problem-and-plan.md),
  [RES-005](../research/res-005-skill-discovery.md),
  [FND-001](../foundation/fnd-001-ab-protocol.md),
  [GOV-006](gov-006-decision-record-process.md),
  [GOV-007](gov-007-contributor-and-change-guide.md).
- [ADR-001](../decisions/adr-001-shared-universal-brain.md),
  [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
  [ADR-003](../legacy/decisions/adr-003-preserve-legacy-brain-isolation.md) (superseded
  here),
  [ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md),
  [ADR-005](../decisions/adr-005-compute-pressure-order.md),
  [ADR-006](../decisions/adr-006-streams-may-observe-one-another.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-010](../assurance/ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [GOV-011](gov-011-execution-record-gov-009.md): Execution record: block implementation plan (GOV-009)
- [GOV-012](gov-012-execution-record-gov-010.md): Execution record: relational implementation plan (GOV-010)
- [RES-010](../research/res-010-predictive-coding-core.md): Predictive coding as the world model's latent core
<!-- backlinks:end -->
