# Implementation plan: human-like learning program

> **Document control:** [GOV-009](../README.md#governance) · Governance / implementation plan · **State:** In progress (build-only; untested); scope approved by the owner 2026-10-21 · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Parent:** [GOV-008](gov-008-human-like-learning-motivation.md).
**Status:** implementation is underway under build-only instructions; see
[GOV-011](gov-011-execution-record-gov-009.md). Tests, A/B runs, and promotion remain
pending. Nothing in this record claims that a change will improve learning.

**How this record was written.** Without scanning the repository, from what was
learned and measured in the working sessions of 2026-10-17 to 2026-10-21, the
owner's instructions, [RES-006](../research/res-006-human-like-learning-gap-analysis.md)
and the map [`ai_learning.mmd`](../../assets/ai_learning.mmd). File and function
names in the "explicit changes" steps are from that knowledge and **must be
re-read before any edit** (procedure step P3). Theory is stated at the level of
well-known results, named so they can be checked; they are not a literature
review, and any formal citation must be verified first. Where a difference
cannot be built with what is known, it is recorded in
[RES-007](../research/res-007-hard-and-deferred-differences.md) so it can be built
later.

## 1. Inputs: what the owner has instructed

**Goal.** Make SkyBot's learning as human-like as achievable, to give it the best
chance of learning as well as it can, using the map above as the guide.

**Success metric.** Adaptability and speed to pick up unfamiliar environments,
judged at owner-set checkpoints (for example: trained only on Doom, released into
Minecraft, can it break wood?). Checkpoints are evaluation-only
([ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md)).

**Standing rules.**

- **Learned, not declared: ALWAYS.** Nothing is scripted, declared or told to the
  agent: not meaning, skills, goals or checkpoints.
- One completely universal brain for every game; no per-game learning forks
  ([ADR-001](../decisions/adr-001-shared-universal-brain.md)).
- No engine reads as learning inputs or rewards
  ([ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md)).
- Everything is open to change and optimisation.
- No legacy brain on this branch
  ([ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md)).
  Minecraft-only code is removed unless it is the game interface itself; what can
  be made universal is translated; the rest is removed on the assistant's
  judgement, recorded for the owner to overrule.
- Minecraft's return is deferred; Doom first. Any interim game is undecided.
  Adding a new game must be easy.
- Streams may observe a peer's frames, sound and chosen actions, never its engine
  state ([ADR-006](../decisions/adr-006-streams-may-observe-one-another.md)).
- When a change needs more compute: reduce model size, then streams, then stepping
  rate ([ADR-005](../decisions/adr-005-compute-pressure-order.md)).

**Evidence standard.** A behaviour claim needs **five seeds of one hour**, positive
on **all five**, each above its region of uncertainty (slightly above is enough).
The hour is a complete seed run (source training, target exposure, everything), so
each seed yields enough data to judge a change on short segments and to **promote**
changes that make SkyBot learn quicker. Null and negative results are recorded
honestly.

**Order and merge.** Take each difference one by one, plan first. The owner merges
the branch when every difference has been tested and addressed.

## 2. Method for every difference

Each difference below has the same structure, in the order the owner asked for.

1. **Theoretical perspective:** why the difference should be included; what it
   adds to learning from a general psychological, neurological and behavioural
   view. An *evidence strength* label says how well the idea is established for
   artificial learners: **A** established in machine learning at comparable scale;
   **B** established in neuroscience or psychology, with partial or small-scale
   machine-learning support; **C** speculative or without a reliable method for
   this setting (these also appear in RES-007).
2. **What the change should add:** the capability, stated generally.
3. **What will be added or changed explicitly:** the new components and the
   existing systems altered, by name (to be re-read before editing).
4. **Thoughts, alternatives and risks.**
5. **Testing procedure** specific to the difference.
6. **Inclusion procedure:** the common gates in section 3 with the
   difference-specific prerequisites and the A/B it feeds.

## 3. Common procedure for including a difference

Every work package passes the same gates. A gate that fails stops the package;
the reason is recorded.

| Gate | Action | Record |
|---|---|---|
| **P0 Preconditions** | The baseline is frozen; the A/A noise floor exists (AB-00); the current-limitations record exists; the work package's prerequisites are done. | GOV-008 log |
| **P1 Research design** | A research record states theory, mechanism, switch names, expected effects, risks and **kill criteria** (what result would make us drop it). | New `RES-` record |
| **P2 Protocol** | A frozen protocol states hypothesis, arms, metrics, seeds, thresholds and failure rule *before* any run. | New `ASS-` record |
| **P3 Verify before edit** | Re-read the code and configuration to be changed; list the files; record every difference from this plan. | Research record |
| **P4 Build behind a switch** | A config key whose off position reproduces the previous behaviour; one helper called from every stepping body (never duplicated); no game-specific branch (universal); `learned, not declared` checked. | Code, change log |
| **P5 Tests** | Unit and contract tests, each shown to **fail without the change**; the whole unit and contract tiers; the documentation check. | Test files |
| **P6 Mock run** | An isolated-root run on the training host: mechanism measurements, memory, GPU and act-latency against the ADR-005 limits. | Result record |
| **P7 A/B** | Baseline versus changed, five seeds of one hour, paired; analysis by the evaluation tool; adaptability metric when a target game exists, otherwise the package's mechanism measures, labelled as such. | Result record |
| **P8 Decision** | **Promote** (adopt into the baseline, default on), **reject** (leave off; record null or negative), or **defer** (needs a longer horizon or a method that does not yet exist; goes to RES-007). | GOV-008, RES-006 |
| **P9 Rebaseline** | The promoted state is the new baseline for the next package. After every three promotions, re-run the A/A noise floor. | GOV-008 log |
| **P10 Rollback** | Any package can be turned off by its switch; a promoted change that later hurts is reverted through its switch and recorded. | Result record |

**Leave-one-out check.** Sequential promotion keeps tests clean but can hide
interactions. After the last package, the combined test (AB-15) is run and, for
each of the three largest contributors, a leave-one-out arm. Results are
recorded whatever they show.

## 4. Resources and schedule (honest estimate)

A frozen comparison is two arms times five seeds times one hour: **10 hours**
of run time, serial, because the 16 GB host fits one run at a time (peak process
memory was about 11 GB). The A/A noise floor is another 5 hours. With about 19
packages, the A/B time is **roughly 190 hours (about 8 days of continuous
running)** before development, analysis, crashes and re-runs. Realistic calendar
time is several weeks. Consequences:

- Packages that cannot show an effect inside one hour (growth, long-horizon
  consolidation) are poorly served by this standard; that limit is recorded for
  each and in RES-007.
- A package whose mechanism measures clearly fail in a short mock run (P6) is
  stopped before it spends 10 hours.
- Running several arms at once is not available on this host.

## 5. Order of work (provisional)

Phase 0, enablers: P0-1 current-limitations record; P0-2 verification of the
RES-006 "not established" rows; WP-15; WP-16; WP-14; AB-00.
Phase 1, learning packages, in this order unless the owner changes it: WP-5,
WP-1, WP-2, WP-6, WP-3, WP-18, WP-4, WP-12, WP-7, WP-19, WP-8, WP-9, WP-11,
WP-13, WP-17, WP-10. Rationale: signal quality first (WP-5), then durable
knowledge and its protection (WP-1, WP-2, WP-18), then the diversity that makes
the transfer metric meaningful (WP-6) and the curriculum that uses it (WP-3),
then deliberation and social learning, then adaptation of the learner itself,
then continuity and scaling. **Gate G1** (before WP-6 completes): decide the
interim target game, if any.

## 6. Phase 0: enablers

### P0-1. Current-limitations record

A dated assurance record of what the current system cannot do, with the evidence
for each limitation, derived from RES-006 and the measured runs; the baseline all
results are compared with. No code. Written first.

### P0-2. Verify the "not established" rows

RES-006 lists rows coded U (no basis). Read the relevant code once, for the rows
that matter to a package, and revise RES-006 with evidence (a revision, not a
rewrite). No code changes. Done as each package reaches P3, and in one pass for
the remaining rows before the combined test.

### WP-15. Remove the legacy brain and Minecraft-only code

1. **Theory.** No learning-psychology analogue; this is an engineering enabler.
   Its justification is experimental: an A/B is only interpretable if the
   baseline has one learner and one code path, and the owner wants one universal
   brain. Evidence strength: not applicable.
2. **What it should add.** One universal path; nothing left that only works in
   one game; the only per-game material is the game interface.
3. **Explicit changes (to be re-read).**
   - *Keep as game interface:* per-game adapters (the ViZDoom adapter, the
     Player adapter, the Minecraft adapter), per-game launchers and install
     steps, per-game configuration files and workflows that every game needs.
   - *Translate (re-derive from pixels, sound, body senses, buttons) rather than
     delete,* where the idea is universal: for example, view-based farm damping
     already replaced an engine-position version; reach, persistence and
     gaze-level shaping read Minecraft sensors and are candidates to translate or
     remove; break-memory and habituation to breaks are Minecraft-event based and
     are candidates to remove or generalise to "repeated effect".
   - *Remove:* the legacy brain's launch path and state handling (the
     "games file absent means legacy" logic in the launcher and supervisor; the
     `Minecraft → legacy` workflow action); the legacy base configuration file
     once the universal configuration is made self-contained; Minecraft-only
     reward and diagnostic code with no universal form (for example chop-budget
     and break diagnostics, and the symbol-grounding reward paths that need the
     language-model symbolizer, which is off for Doom).
   - *Decision per item:* each item is classed keep / translate / remove with a
     one-line reason in the research record, for the owner to overrule.
   - Flatten the universal configuration (today composed on top of the legacy
     file) into one file; keep the composer for per-game overlays only.
4. **Thoughts and risks.** The universal configuration inherits hundreds of keys
   from the legacy file; flattening risks dropping a default that changes
   behaviour silently. Mitigation: dump the composed configuration before and
   after and diff it; every non-Minecraft key must be unchanged. Removing
   Minecraft-only code also removes tests that cover it; each deleted test is
   listed, with the reason. The owner's copies of the legacy brain make removal
   recoverable.
5. **Testing.** (a) Before removal: capture the composed Doom configuration. (b)
   After each small removal step: the full unit and contract tiers and the
   documentation check. (c) Composed-configuration diff for Doom is empty except
   removed legacy keys. (d) A 20-minute mock run: mechanism numbers (lag probe,
   attribution, entropy, explained variance) within the A/A noise floor of the
   pre-removal baseline (no behaviour claim). (e) Grep-level test that no module
   imports a removed Minecraft-only module.
6. **Inclusion.** Gates P1, P3 to P6. No A/B (it is a no-behaviour-change
   refactor), but the parity check in (d) must pass. Prerequisite: none.

### WP-16. Make adding a game easy

1. **Theory.** Humans transfer between games largely because the interface is
   standardised (screen, speakers, a controller), so only the content differs.
   Evidence strength: not applicable to learning; an enabler for the transfer
   metric.
2. **What it should add.** A new game (including any interim game) needs only its
   game interface and a checkpoint detector, with no change to learning code.
3. **Explicit changes (to be re-read).**
   - A *game template*: an adapter skeleton implementing the shared environment
     contract (pixels, sound, body senses, a fixed action-slot table with a valid
     mask, executed-action reporting, life boundaries, health and stuck signals),
     a launcher skeleton, an install step, and a game configuration schema.
   - A *checkpoint detector template*: an evaluator-side module, kept in a
     separate package that learning code does not import, mapping engine events
     to the owner's checkpoints.
   - A synthetic third game used only to prove the template.
   - An architecture record (the "game onboarding standard") with the checklist.
4. **Thoughts and risks.** Real-time (clock-driven) and synchronous games differ in
   timing; the template must state which contract each follows, since the
   action-lag defect came from exactly that difference. Keeping detectors out of
   the learner's import graph is how ADR-002 is enforced mechanically.
5. **Testing.** (a) Onboard the synthetic game by adding files only; assert zero
   edits to learning modules. (b) An import-graph test: no learner module imports a
   detector. (c) The executed-action contract test runs against the template for
   both timing styles. (d) Record the number of files and lines needed to onboard.
6. **Inclusion.** Gates P1, P3 to P6; no A/B. Prerequisite: WP-15.

### WP-14. Evaluation machinery

1. **Theory.** Developmental science judges progress with standardised, repeatable
   milestone assessments, and good practice separates the measure from the thing
   measured. Evidence strength: A (experimental method).
2. **What it should add.** The ability to run and analyse ASS-007 exactly:
   frozen configurations, paired seeds, A/A noise floor, censored time-to-checkpoint
   statistics, and result records generated the same way every time.
3. **Explicit changes (to be re-read).**
   - Generalise the existing arm runner and learning-report tools into a seeded,
     resumable runner (source phase, target phase, snapshotting, crash rule).
   - Checkpoint detectors per ASS-007 12.4; a results store; the paired-bootstrap
     analysis with right-censoring; the A/A estimator; a generator for result
     records.
   - A held-out-environment registry (WP-6 feeds it) and a long-run protocol for
     effects that need more than an hour. For current development experiments,
     use Doom MAP01–MAP28 for source and MAP29–MAP32 as the held-out Doom proxy;
     the Minecraft checkpoint metric is a later program-level evaluation.
4. **Thoughts and risks.** Five seeds all positive is a strict gate and has low
   power for small real effects; the A/A runs estimate the false-positive rate but
   not the power. Add a power note to each result (the smallest effect the five-seed
   design could have shown). Seeds are not bit-reproducible on this host, so
   run-to-run noise must be measured, not assumed.
5. **Testing.** (a) Synthetic data with a known effect: the analysis recovers it;
   with no effect: the false-positive rate matches the nominal one. (b) Censoring
   cases (no checkpoint reached). (c) Detectors versus scripted Doom trajectories
   that do and do not reach each checkpoint. (d) One real Doom A/A set of ten
   paired seeds (AB-00), followed by the five-seed development A/B procedures.
6. **Inclusion.** Gates P1 to P7. Prerequisite: a frozen Doom development
   protocol and an available held-out Doom evaluator; the Minecraft program
   metric remains deferred.

## 7. Phase 1: learning packages

### WP-5. Rebalance the intrinsic signals (including the episodic novelty weight)

1. **Theory.** In humans curiosity is not novelty alone: it follows *learning
   progress* and information gain (Kaplan and Oudeyer's learning-progress
   account; Berlyne's inverted-U between novelty and interest), it habituates, and
   it is sustained by competence. A drive that pays for any unfamiliar view
   without regard to learnability invites the "couch-potato" and "noisy-TV"
   failures known from machine curiosity research (random-network-distillation
   and never-give-up style methods address them with pseudo-counts and
   thresholds). Evidence strength: **A** for the failure modes and for
   learning-progress and episodic-memory curiosity at game scale; **B** for the
   human account.
2. **What it should add.** A mix in which no single term supplies 80-90 % of
   income; novelty that decays when a view is no longer informative; and an
   information-gain term that responds to *consequences of the agent's own
   actions*.
3. **Explicit changes (to be re-read).**
   - The episodic term (episodic novelty module and the loop helper that pays it
     on every stream): its weight, and a cluster-distance threshold so near-identical
     views stop paying (as in never-give-up style methods); optionally a per-term
     cap on income share.
   - The learning-progress term and its action-attribution gate: now that credit
     labels are correct, re-measure what it pays; revisit its significance gate
     before changing weights (RES-001).
   - A new information-gain term using the world model's own quantities (the
     posterior-versus-prior divergence on steps where the action explains the
     change), universal and computed from the latent, never from engine state.
   - Ledger and census: report each term's income separately (the learning-progress
     base was previously reported together with episodic novelty).
   - Every new coefficient lives in the universal configuration under a named key.
4. **Thoughts and risks.** The owner has said the weight is to be revisited soon.
   Treat the weight sweep as arms of one protocol, not separate packages. The
   risk is moving the farm from one term to another; each variant is tested
   against synthetic farm stimuli (spinning, camera-only motion, repeated loops)
   before any run. A reward change is a steering risk only if it rewards an
   outcome; these terms reward learning-related properties of the agent's own
   experience, which stays within learned-not-declared.
5. **Testing.** (a) Unit: synthetic view sequences (repeat, new, noisy,
   spinning) give the intended payments, falsified against the current module. (b)
   Farm falsifiers for each term. (c) Mock run: income shares per term, farm-flag
   rate. (d) A/B AB-07: baseline against variants (weight and threshold arms), five
   seeds. Mechanism measures: income shares, state dependence, attribution; the
   adaptability metric when available.
6. **Inclusion.** Gates P0 to P10. Prerequisite: Phase 0, AB-00.

### WP-1. Game-agnostic skill discovery and validation

1. **Theory.** Infants discover what they can cause through *contingency
   detection*: they notice outcomes that depend on their own action (Watson's
   contingency work; infant operant "mobile" experiments), forming a sense of
   agency. Repeated, reproducible effects become chunked into reusable units
   (chunking in memory research; options in hierarchical reinforcement learning
   after Sutton, Precup and Singh). Evidence strength: **B** for the psychology;
   **A** for options given a good skill set; **C** for *discovering* skills with
   no reward signal in a new game (active research).
2. **What it should add.** A way for SkyBot to create reusable skills in any game
   from effects it caused, keep only those that reproduce their effect, and let
   those skills populate its knowledge and episodic stores.
3. **Explicit changes.** As designed in [RES-005](../research/res-005-skill-discovery.md):
   - a trigger from world-model surprise caused by the agent's own action (a
     batched prior over all actions on spike steps only), replacing the
     extrinsic-reward spike;
   - effect keys from clustering the before-and-after change of the frozen view
     embedding, replacing the engine-derived Minecraft keys;
   - probation (a binomial test against the base-policy rate; retire with
     doubling back-off); the fossil guard admits the new key family;
   - one helper called from every stepping body; telemetry per segment.
   Also decide, under WP-15, what becomes of the knowledge graph and symbolizer,
   which only the language-model symbolizer fed: either feed the graph from
   learned effect keys or remove it.
4. **Thoughts and risks.** RES-005 step 0 (measuring the trigger on recorded play
   against evaluator counters) is the first deliverable; its pre-registered rules
   say to stop if the trigger does no better than random. A young world model is
   surprised by everything, so early discovery will be quiet. Probation needs a
   non-uniform base policy; skills are policy copies, so a skill with no
   conditioning effect must fail probation. Discovery pays no reward, to avoid
   a new farm.
5. **Testing.** RES-005 contracts A to G. Step 0 on recorded play with training
   paused. Mock run: trigger rate, key purity against evaluator events, probation
   pass and fail counts, learned-option gate on a non-empty bank. A/B AB-02:
   off versus on, five seeds; mechanism: held-out invocation success above the
   base rate.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-5 (stable intrinsic signals),
   WP-15 (so keys do not depend on engine counters).

### WP-2. Consolidation and forgetting protection

1. **Theory.** Complementary learning systems (McClelland, McNaughton and
   O'Reilly): a fast episodic store and a slow generalising store, with offline
   *replay* of experience interleaved into the slow store, prevents catastrophic
   interference (McCloskey and Cohen). Hippocampal replay during slow-wave and REM
   sleep is observed in animals (Wilson and McNaughton; reviews by Diekelmann and
   Born); synaptic downscaling in sleep has been proposed to restore capacity
   (Tononi and Cirelli). Evidence strength: **B** for the neuroscience; **A** for
   replay in value-based reinforcement learning, elastic weight consolidation
   (Kirkpatrick et al.), policy distillation and generative replay (Shin et al.);
   **C** for how well these work together with on-policy PPO at this scale.
2. **What it should add.** Protection of what the policy and world model have
   learned when the task or map changes, and the ability to learn from replayed
   experience, not only from the latest rollout.
3. **Explicit changes (to be re-read).**
   - A *consolidation phase* in the learning step of each segment (or every few
     segments): train the policy from replay.
   - The replay buffer gains **policy log-probabilities and value estimates** as
     stored columns so off-policy corrections are possible (today it stores
     observations, actions, rewards, flags, priorities and body senses; the policy
     keeps no replay).
   - An off-policy-safe learner for that phase (advantage-weighted regression or
     a truncated importance-weighted actor-critic), kept separate from the PPO
     update so PPO's on-policy assumptions hold.
   - A regulariser on the policy's important weights (elastic weight consolidation
     or synaptic intelligence), optional by switch.
   - Snapshot distillation: periodically distil older policy snapshots into the
     current policy.
   - Replay sampling: prioritise by error or learning progress (partly present),
     and add spaced revisiting of old material; both by switch.
   - Reuse the existing paired-progress forgetting probes as the measure.
4. **Thoughts and risks.** Mixing off-policy updates with PPO can destabilise
   learning; the safest first arm is distillation plus regularisation without any
   off-policy gradient. Replay is capped at 5.5 GB, so old material is overwritten;
   protection must work from the weights, not only the buffer. A one-hour seed has
   little forgetting to protect; the A/B needs a mid-run task change (WP-6).
5. **Testing.** (a) Unit: regulariser penalises movement of important weights on a
   toy task; off-policy corrections are unbiased on a known toy; stored columns
   round-trip through save and load. (b) Forgetting test: train on map A, switch to
   map B, measure retention on A against the baseline. (c) Mock run: stability
   (value loss, policy KL), act latency, memory. (d) A/B AB-03.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-6 for a meaningful task
   change; WP-14. Hard parts (true sleep-stage structure, exact hippocampal-cortical
   mapping) go to RES-007.

### WP-6. Data diversity

1. **Theory.** Variability of practice improves transfer in human motor and
   cognitive learning (Schmidt's schema theory; Bjork's "desirable difficulties");
   training in many contexts is how invariant knowledge is extracted. In machine
   learning, domain randomisation and procedurally generated levels improve
   generalisation (for example the procedurally generated game benchmarks).
   Evidence strength: **A**.
2. **What it should add.** Many environments inside the source phase, and
   *held-out* environments the brain never trains on, so adaptability can be
   measured.
3. **Explicit changes (to be re-read).**
   - Per-stream map assignment in the game configuration and the composer, so the
     eight streams play different maps and the assignment can change over a run.
   - A held-out-map list that the source phase can never use (enforced by a test).
   - Visual variation options the engine already offers (texture, lighting,
     monsters, skill level) as configuration, never engine reads.
   - The onboarding template (WP-16) is how further games join; Minecraft stays
     deferred.
4. **Thoughts and risks.** Doom's Freedoom files hold many maps, but their number
   and difficulty spread should be checked before relying on them. More diversity
   lowers per-map experience in a one-hour run. Interim target (Gate G1): held-out
   Doom maps are a proxy only and are labelled so.
5. **Testing.** (a) No-leakage test: held-out maps never appear in the source
   phase. (b) Coverage report: states visited per map. (c) A/B AB-08: one map
   versus several, measured on held-out maps (mechanism measures until a transfer
   target exists).
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-16, WP-14. Feeds WP-2 and WP-3.

### WP-3. Curriculum and learnability frontier

1. **Theory.** Vygotsky's zone of proximal development and Csikszentmihalyi's flow
   describe learning fastest where tasks are neither trivial nor impossible.
   Learning-progress curricula (Oudeyer; Graves et al.'s automated curriculum;
   teacher-student curricula after Matiisen et al.; prioritised level replay) select
   tasks by recent improvement. Evidence strength: **A** in games and simple
   environments; **C** when no task reward exists (progress must be measured from
   intrinsic quantities).
2. **What it should add.** A learned allocation of practice across maps and
   difficulty levels, favouring where learning is currently fastest.
3. **Explicit changes.**
   - A *scheduler* module that assigns map and difficulty to each stream and
     updates it from a progress signal (a bandit over contexts).
   - The progress signal: world-model learning progress and action attribution per
     context, plus held-out fresh-data loss; no extrinsic reward.
   - Difficulty levers exposed by the game interface (skill level, map).
   - "Targeted fine-tuning" is the same mechanism: revisit contexts with falling
     competence.
4. **Thoughts and risks.** Without an extrinsic reward, progress proxies can be
   gamed or noisy; the bandit needs forced exploration. Curricula can reduce
   diversity if the bandit collapses on one map; track entropy of the assignment.
5. **Testing.** (a) Bandit unit tests on synthetic contexts with known progress
   rates. (b) Collapse test: assignment entropy stays above a floor. (c) A/B AB-04:
   fixed sequence versus curriculum, five seeds.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-6, WP-14.

### WP-18. Queryable episodic memory

1. **Theory.** The hippocampus supports retrieval of specific past episodes to guide
   present choices (episodic memory; one-shot learning from single experiences).
   Machine counterparts include episodic control and neural episodic control
   (Blundell et al.; Pritzel et al.), retrieval-augmented agents, and the
   episodic memory already used for novelty. Evidence strength: **A** for
   retrieval in reinforcement learning; **B** for the neuroscience.
2. **What it should add.** Letting the policy use *what happened after similar
   situations before*, so a single rewarding or harmful episode can change
   behaviour immediately.
3. **Explicit changes (to be re-read).**
   - Extend the existing per-stream view memory and the episodic record store into
     a keyed store of (view embedding, action taken, short-horizon outcome
     summary).
   - A retrieval module returns a small vector of neighbours' outcomes to the
     policy input (alongside the latent), in the same position the knowledge
     vector occupies today.
   - Decide, under WP-15 and WP-1, whether the knowledge graph is replaced by this
     store.
4. **Thoughts and risks.** Outcome summaries must be learned quantities (changes in
   the latent, learning progress), not engine events. Retrieval adds latency; use a
   small, cached nearest-neighbour index.
5. **Testing.** (a) Unit: retrieval returns correct neighbours on a toy; policy
   input width is consistent. (b) Mechanism: one-shot repeat of a harmful or useful
   event changes the policy's next choice more than in the baseline. (c) A/B.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-1 (outcome summaries).

### WP-4. Decision-time planning, uncertainty and metacognition

1. **Theory.** Humans combine habit (model-free) and deliberate planning
   (model-based) and arbitrate between them by confidence (Daw, Niv and Dayan's
   two-system account; prefrontal planning and hippocampal "preplay" of routes).
   Metacognition (Flavell) lets learners know what they do not know and allocate
   effort accordingly; calibration of confidence is measurable. Evidence strength:
   **A** for learned-model planning in games (PlaNet and Dreamer with trajectory
   optimisation; MuZero search) and for uncertainty via ensembles and calibration
   methods; **C** for doing so in real time with eight streams on this host.
2. **What it should add.** The ability to look ahead through imagined futures
   before acting when it is worth it, and calibrated confidence that governs when to
   plan, explore or act on habit.
3. **Explicit changes (to be re-read).**
   - A planner using the world model's imagined rollouts and the critic's value
     (random-shooting or cross-entropy search over short action sequences).
   - Uncertainty from disagreement among cheap ensemble heads on the forward
     prediction (a few small heads, not whole models) and from value variance.
   - A calibration monitor for the critic and policy (reliability of predicted
     confidence against outcomes) and a temperature-scaling correction.
   - An arbitration rule: plan when uncertainty is high and a time budget allows.
   - **Real-time handling:** the Doom loop already uses about 165 ms of its decision
     interval. Decision-time search for eight streams is unlikely to fit. First
     version: plan *during the learning phase and while the standby actor runs*
     (planning to train and to correct the policy, as in dream distillation), not
     inside the acting step. Acting-time planning is only enabled when measured
     latency permits (ADR-005 may force fewer streams).
4. **Thoughts and risks.** Models that are wrong in compounding ways make planning
   worse than habit; the arbitration must fall back to the policy when calibration is
   poor. The existing dream actor is an amortised planner already; compare against
   it.
5. **Testing.** (a) Planner on a known toy model finds the known best action
   sequence. (b) Calibration metric on synthetic predictions. (c) Latency test: act
   time stays inside the ADR-005 budget. (d) A/B AB-05 (planning) and AB-06
   (calibrated uncertainty).
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-14; WP-5 for stable signals. Real-time
   acting-time planning at scale goes to RES-007.

### WP-12. Peer observation between streams

1. **Theory.** Observational learning is central in humans (Bandura's social
   learning theory; early imitation in infants, Meltzoff and Moore; social
   referencing). Watching another's attempt gives outcome information at no risk to
   the observer. Evidence strength: **B** for psychology; **A** for imitation and
   multi-agent learning from demonstrations; **C** for *unlabelled, undeclared*
   peer observation in separate game instances.
2. **What it should add.** Each stream learns from what the other streams do and
   what follows, so collective experience speeds learning, within the rule that
   nothing is declared good.
3. **Explicit changes (to be re-read).**
   - A *peer observation buffer* per stream holding a few recent frames and the
     chosen actions of the other streams, passed to the encoder or policy as extra
     context (no engine state; ADR-006).
   - An auxiliary prediction task: predict a peer's next frame and next action. It is
     learned from experience and rewards nothing.
   - Use of peers' transitions for vicarious model learning (already partly true
     through shared replay): attribute effects to the peer that caused them.
   - A diversity measure: visited-state spread across streams.
4. **Thoughts and risks.** Today's streams play separate game instances, not one
   shared world, so "cooperation" is observation of another run, closer to watching
   another player than to teamwork. True cooperation needs several agents in the
   same world (a multiplayer engine mode); that is deferred (RES-007). Risks: all
   streams converge on one habit (diversity collapse); extra compute per stream
   (ADR-005). The oxytocin-like links in the map have no established machine
   analogue; they are recorded in RES-007.
5. **Testing.** (a) Isolation test: no engine fields appear in a stream's
   observation of a peer. (b) Auxiliary task unit test on scripted peers. (c)
   Diversity metric on synthetic collapsing and spreading streams. (d) A/B AB-13:
   isolated versus observing streams, with diversity reported alongside progress.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-14, WP-16.

### WP-7. Attention and working memory

1. **Theory.** Selective attention (Posner; Treisman) concentrates processing on
   relevant parts of the input; working memory holds a handful of chunks (Miller;
   Baddeley), with gating (prefrontal and basal-ganglia gating accounts). Evidence
   strength: **A** for attention in vision and sequence models (Transformers,
   slot attention, Perceiver); **B** for the neuroscience; **C** as a clear win in
   this recurrent-world-model setting, where recurrent latents often suffice.
2. **What it should add.** Focus on parts of the scene and past latents that matter,
   and a longer, selectively retrieved context.
3. **Explicit changes (to be re-read).**
   - Turn on and test the world model's existing slot-attention option (present but
     off in all measured runs).
   - Give the policy a short window of recent latents and an attention layer over it
     (a small gated memory), by switch.
   - Cost report per option (parameters, milliseconds, memory).
4. **Thoughts and risks.** Easy to add and hard to show benefit in one hour; the
   likely outcome is null at this scale, which will be recorded as null. Attention
   raises compute; ADR-005 applies.
5. **Testing.** Unit: attention weights sum to one and gradient flows; shape tests
   through save, load and growth (the growth helpers must handle any new
   latent-reading layers). A/B AB-09.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-14.

### WP-19. Representation, abstraction and robustness

1. **Theory.** Humans build schemas that generalise across instances (Piaget's
   assimilation and accommodation; Bartlett) and map structure by analogy
   (Gentner). Self-supervised contrastive learning (contrastive predictive coding,
   van den Oord et al.; SimCLR) and data augmentation for pixel agents (random
   shifts in DrQ-style methods) are established ways to get invariant, reusable
   features. Evidence strength: **A** for contrastive objectives and augmentation in
   pixel reinforcement learning; **C** for few-shot analogy.
2. **What it should add.** Features that survive changes of map, lighting or game,
   and temporal abstraction beyond the two horizons now used.
3. **Explicit changes (to be re-read).**
   - A contrastive or predictive auxiliary loss on the encoder embeddings.
   - Random-shift and colour-jitter augmentation applied only to the encoder's
     training batches (never to stored data or to reward inputs).
   - Additional latent-prediction horizons (extend the 4 and 16 steps).
   - A transfer probe: freeze the encoder and train a small head on a held-out map;
     report accuracy (feature reuse), as a measurement.
4. **Thoughts and risks.** Augmentation interacts with the novelty term: if a view
   key is computed from augmented frames, novelty changes. Keep the view key on
   un-augmented frames. Few-shot analogy has no reliable method here (RES-007).
5. **Testing.** Unit: augmentation leaves labels and rewards unchanged; the
   contrastive loss separates positives from negatives on a toy. Probe on held-out
   maps. A/B (a new AB id in its protocol) on the adaptability proxy.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-6, WP-14.

### WP-8. Self-evaluation, competence and autonomy

1. **Theory.** Self-efficacy (Bandura) and the competence and autonomy needs in
   self-determination theory (Deci and Ryan) drive persistence and choice of
   challenge; learners monitor their own progress and adjust effort (metacognitive
   monitoring and control). Evidence strength: **B**; **A** for progress-based
   intrinsic motivation; **C** for a unified need-satisfaction model.
2. **What it should add.** A running estimate of "how well am I doing here, and am I
   improving", wired to how much effort, exploration and persistence the learner
   spends; autonomy as measured agency.
3. **Explicit changes (to be re-read).**
   - Generalise the existing competence predictor (now skill-only) into a
     per-context competence tracker built from learning progress and action
     attribution (both already measured).
   - Wire it to the exploration coefficient and the intrinsic mixture (more
     exploration when progress is flat, more exploitation when it is rising).
   - A self-model head predicting the agent's own near-future progress; its error is
     logged.
   - Keep autonomy (attribution, empowerment) and competence as separate measured
     terms and report them together.
4. **Thoughts and risks.** Overlaps WP-9 and WP-13 (adaptive hyperparameters);
   keep one controller, not three. A competence signal that is used as reward would
   drift towards the steering rule; here it modulates learning rates and exploration
   only.
5. **Testing.** Unit: the tracker reflects synthetic rising, flat and falling
   progress. Stability: the controlled coefficient stays bounded. A/B (protocol id
   to be assigned).
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-5, WP-14.

### WP-9. Adaptive optimisation and plasticity

1. **Theory.** Plasticity declines with experience and is regulated (critical and
   sensitive periods; Hensch); learning rates anneal. Deep networks also lose
   plasticity in long non-stationary training (Dohare et al.). Learning-rate
   schedules, automatic entropy tuning (soft actor-critic) and return normalisation
   are standard. Evidence strength: **A** for schedules, entropy tuning and
   normalisation; **B/C** for deliberately engineered critical periods.
2. **What it should add.** Optimisation that adapts to the stage of learning: a
   schedule for the policy, entropy that is tuned to a target, return normalisation,
   and a guard against loss of plasticity.
3. **Explicit changes (to be re-read).**
   - A learning-rate schedule for the policy and the world model.
   - Entropy coefficient set by a controller towards a target entropy measured over
     valid actions (replacing the one-time boost).
   - A return or reward normaliser for the policy (none exists today).
   - Discount: evaluate, not replace, adaptive discounting.
   - Plasticity maintenance: monitor dormant units; test a continual-backprop-style
     refresh.
   - Wire the shadow drift detectors to an adaptation response if they prove useful.
4. **Thoughts and risks.** Within a one-hour seed little decay occurs; effects
   appear late and may not be visible in one hour (recorded). Several controllers
   (this, WP-8, WP-13) must not fight; single owner of each coefficient.
5. **Testing.** Unit: schedule shape; controller reaches a target entropy on a toy;
   normaliser leaves advantage ordering unchanged. A/B.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-14.

### WP-11. Risk estimation and safe exploration

1. **Theory.** Threat learning in the amygdala and loss aversion (Kahneman and
   Tversky) bias humans toward caution after harm; exploration is bounded by learned
   danger. Evidence strength: **B**; **A** for safe-exploration formulations
   (constrained decision processes) in simulation.
2. **What it should add.** A learned estimate of danger (probability of death or
   large state change soon) available to the policy, so it can explore without
   needless repeated death.
3. **Explicit changes (to be re-read).**
   - Use the world model's continue head (a learned survival prediction, which the
     model already trains) as a risk feature, and feed it as a *state input* to the
     policy, not as a reward.
   - Optionally train a short-horizon risk head.
   - Report risk calibration.
4. **Thoughts and risks.** In Doom, death is cheap (an instant restart), so heavy risk
   aversion may reduce useful exploration. This is an input, not a penalty, to stay
   within learned-not-declared and not steer. Death comes from the game's own
   terminal event (what a player sees), not from reading engine counters.
5. **Testing.** Unit: risk estimate rises before scripted deaths. A/B on death rate
   and progress.
6. **Inclusion.** Gates P0 to P10. Prerequisite: WP-14.

### WP-13. Meta-learning and hyperparameter adaptation

1. **Theory.** Learning sets (Harlow): experience on many problems makes new
   problems easier to learn. Machine versions include model-agnostic meta-learning,
   recurrent meta-reinforcement learning and population-based training (Jaderberg
   et al.). Evidence strength: **A** that they work in benchmarks; **C** at the
   scale and constraints here (one brain, one hour, one GPU).
2. **What it should add.** Faster adaptation to a new game through experience across
   games, and hyperparameters that adapt online.
3. **Explicit changes.**
   - First version: *online hyperparameter adaptation* inside the one brain (a
     bandit or hill-climber over a few coefficients, using learning progress).
   - Population-based training is **not** compatible with one shared brain; it is not
     planned.
   - Meta-learning proper (gradient-based or recurrent) is deferred to RES-007.
4. **Thoughts and risks.** Meta-learning needs many tasks; one hour of Doom cannot
   supply them. Without cross-game experience (Minecraft deferred), there is nothing
   to meta-learn from. Honest expectation: null.
5. **Testing.** Unit: the adapter finds the best of a few synthetic coefficients. A/B
   when WP-6 provides tasks.
6. **Inclusion.** Gates P0 to P10; likely rejected or deferred.

### WP-17. Run continuity and tooling

1. **Theory.** Learning must survive interruptions, as sleep and memory consolidation
   protect continuity in humans. Engineering enabler. Evidence strength: not
   applicable.
2. **What it should add.** Automatic resumption after power or network loss, a
   checkpoint cadence by time as well as steps, and a tested software version match
   between continuous integration and the host.
3. **Explicit changes (to be re-read).**
   - Start the supervisor as a service at boot on the host.
   - Checkpoint on a wall-clock interval in addition to the step interval.
   - Pin the continuous-integration torch version to the host's, or document why not.
   - Gate Minecraft-only progress lines (handled by WP-15 where they remain).
4. **Thoughts and risks.** The host was unreachable for about eleven hours on
   2026-10-19 (cause not established); this package does not diagnose it.
5. **Testing.** Simulated host restart; resume equality of policy and replay.
6. **Inclusion.** Gates P1, P3 to P6; no A/B.

### WP-10. Autonomous scaling

1. **Theory.** Capacity grows with development and experience (Elman's "starting
   small"; synaptogenesis and later pruning). Evidence strength: **B**; **A** that
   progressive networks and growth work in machine learning when growth is
   function-preserving.
2. **What it should add.** A system that grows or shrinks its own world model and
   policy according to evidence, in ADR-005's order, and carries its policy across
   the change.
3. **Explicit changes (to be re-read).**
   - Reconcile the autoscaler's rules with ADR-005 and its Amendment 1 thresholds.
   - Verify policy growth live (it is verified only in unit tests); expand the
     optimiser state across growth.
   - Actuate the autoscaler in a test arm only.
4. **Thoughts and risks.** In the measured runs the first growth decision came after
   about forty minutes, so a one-hour seed barely reaches it; this package needs a
   separate long-run protocol (recorded in RES-007). A shrink-by-relearning path
   exists but is approximate.
5. **Testing.** Unit: growth is function-preserving for policy and world model;
   live check on the host with logits compared before and after. Long-run study.
6. **Inclusion.** Gates P0 to P10, with a long-run protocol instead of the one-hour
   standard. Prerequisite: ADR-005 thresholds measured.

## 8. Coverage: where every recorded difference goes

Every difference recorded in RES-006 (sections 5 and 6, and the DNE terms) is
assigned. "RES-007" means it cannot be built reliably with what is known and is
recorded there for later. "Deliberate" means a standing owner rule.

| RES-006 difference | Assigned to |
|---|---|
| Pretraining, behaviour cloning, imitation of demonstrations, human feedback, language, chain-of-thought, rationales, scratchpad notes, summarisation, prompt priming | Deliberate (learned, not declared) |
| Extrinsic reward in Doom | Deliberate |
| Critical learning periods | WP-9 (plasticity); engineered critical periods in RES-007 |
| Learning-rate scheduling; exploration temperature; discount; gain scaling; return normalisation; concept-drift adaptation | WP-9 |
| Pruning and sparsity; normalisation and activation audit | P0-2 (verify) then WP-10; pruning in RES-007 |
| Dynamic expansion; autoscaler advice only | WP-10 |
| Frozen layers | WP-2 and WP-9 |
| Excitatory/inhibitory (positive and negative weights) split | RES-007 |
| Attention (live), long-context attention, working-memory context | WP-7 |
| Reasoning controller; planning; model-based action choice; uncertainty estimator; confidence and calibration; strategy selection; reward model use | WP-4 |
| Safety and risk module; risk penalties; safe exploration | WP-11 |
| Importance weighting (prioritised replay audit); spaced replay; hard-negative mining; offline phase; generative replay; replay for the policy; offline reinforcement learning; consolidation (EWC); continual-learning pipeline; catastrophic forgetting; distillation | WP-2 |
| Power resilience; training instability; continuous-integration versus host software versions | WP-17 |
| Sensor resolution; adaptive resolution | RES-007 |
| Embodied robotics | RES-007 |
| Tokenisation and chunking; skills; goal-conditioned behaviour; causal structure (agency part) | WP-1 |
| Full causal structure discovery | RES-007 |
| Retrieval; episodic memory; knowledge base and graph (empty in Doom); interaction history as learning input | WP-18 (and WP-1) |
| Model editing | RES-007 |
| Hierarchical abstraction; contrastive learning; augmentation; multimodal embedding quality; feature probe | WP-19 |
| Few-shot analogy | RES-007 |
| Meta-learning; AutoML; hyperparameter configuration; population approaches | WP-13 (PBT excluded; proper meta-learning in RES-007) |
| Intrinsic reward dominance; information gain; relevance scoring; habituation; data saturation; bias-induced farming; reward shaping audit | WP-5 |
| Self-evaluation; competence; autonomy; self-reflection (non-linguistic) | WP-8 |
| Curriculum; learnability frontier; targeted fine-tuning; mastery thresholds | WP-3 |
| Real-world data diversity; training environment; domain randomisation; coverage gaps; data distribution | WP-6 |
| Peer agents; multi-agent learning; observation of peers | WP-12 |
| Cooperation in a shared world | RES-007 |
| Evaluation and validation; held-out tasks; zero-shot evaluation; benchmarks; long training runs; reproducibility; random seeds; negative transfer; generalisation and transfer measures | WP-14 (and P0-1) |
| Federated learning | RES-007 (does not fit one shared brain) |
| Graph neural networks | WP-15 judgement (remove unless used) |
| Legacy and Minecraft-only code | WP-15 |
| New-game ease; checkpoint detectors | WP-16 |
| **DNE terms:** Oxytocin (bonding) | RES-007 |
| Endocannabinoids (retrograde plasticity tuning) | RES-007 |
| Autonomy and competence | WP-8 |
| Relatedness; Belonging | RES-007 (WP-12 gives only observation) |
| Learning styles (myth) | None; nothing to build |
| Wellbeing | RES-007 |

## 9. Cross-cutting testing procedure

The same test layers apply to every package; each package's section lists only
what is specific.

1. **Unit tests** for pure functions and shapes (the existing unit tier).
2. **Contract tests**, each shown to fail without the change, with a docstring
   naming the incident or hypothesis it protects (the repository convention).
3. **Wiring tests** that assert a helper is called once in each stepping body.
4. **Universality tests** that the same configuration block is composed for every
   game, with no game name used in learning code.
5. **Isolation tests** that no learning code imports evaluator or detector modules
   (ADR-002), and that checkpoint events never enter any observation.
6. **Declared-knowledge test.** A review item for every package: nothing in the
   change tells the agent what is good (learned, not declared).
7. **Mock run** in an isolated root on the training host; **mechanism measures**
   (lag probe, attribution, state dependence, entropy over valid actions, explained
   variance, KL from the first policy); **resource measures** (memory, GPU,
   act latency).
8. **A/B** per ASS-007 with the five-seed standard, the A/A noise floor, and the
   adaptability metric when a transfer target exists.
9. **Documentation check** and register entries in the same change.

## 10. Risks that apply to the whole program

- **Power.** Five-seed, one-hour tests may miss real but small or slow effects. The
  design controls false positives, not misses; each result states what size of
  effect it could have shown.
- **Metric availability.** The main metric needs a target game; Minecraft is
  deferred. Until Gate G1, packages are judged on mechanism measures and labelled
  so. An interim target (if chosen) is labelled a proxy.
- **Interactions.** Sequential promotion can hide interactions; the leave-one-out
  check at the end is the guard.
- **Compute.** About 190 hours of A/B running; ADR-005's rule will act on several
  packages (planning, attention, peers).
- **Reward and steering.** Several packages touch reward or inputs. Each is reviewed
  against learned-not-declared; a package that would reward an outcome (rather than a
  property of learning) is rejected.
- **Overfitting to Doom.** Source training is Doom only; a gain that is Doom-specific
  is not adaptability. The held-out environments and the final transfer test are the
  guard.
- **Plan accuracy.** Names and thresholds here come from memory; P3 (verify before
  edit) exists because of this.

## 11. Thoughts on the plan

- The five-seed rule is conservative and slow. If most packages come out null, that
  may reflect low power in a one-hour seed rather than no effect; the plan keeps
  null results as findings but flags the horizon limit for slow-acting packages
  (growth, consolidation, curriculum).
- The strongest early evidence will come from mechanism measures (what the learner
  does with its experience); the adaptability metric is the real goal and cannot be
  run until a transfer target exists. Choosing the interim game (Gate G1) early
  would unlock it.
- Several map items are speculative for machines. Recording them (RES-007) beats
  building something that only looks like the idea.
- The enablers (WP-15, WP-16, WP-14) are not learning improvements but decide
  whether the rest can be judged at all; doing them first is deliberate.

## 12. Documents this plan will produce

Identifiers are assigned at creation, next unused in each series.

| Record | Class | Timing |
|---|---|---|
| Current-limitations record (assurance, dated) | Record | First |
| Game onboarding standard (architecture) | Proposed | With WP-16 |
| ASS-007 frozen, with the owner's checkpoints | Protocol | Before AB-00 |
| AB-00 noise-floor protocol and result | Protocol, Record | After WP-14 |
| One research design and one protocol per package (P1, P2), and a result record (P7/P8) | Proposed, Protocol, Record | Per package |
| Legacy-removal decisions list (keep / translate / remove, with reasons) | Record | During WP-15 |
| Updates to RES-006, RES-007, GOV-008, the root `CLAUDE.md` | Revisions | After each decision |
| Long-run protocol (growth, consolidation horizon) | Protocol | With WP-10 / WP-2 |
| Incident records for failures found | Record | As needed |

## 13. Open items

- The final checkpoint list and detectors (owner).
- Gate G1: whether to use an interim target game, and which.
- Confirmation of the reading of "promote" (adopting a change into the baseline on
  quicker learning within the hour across all five seeds).
- Approval of this plan and the order in section 5, or a different order.
- Whether WP-17, WP-18 and WP-19 (added by this plan to the list in GOV-008) are
  accepted.

## 14. Change log

| Date | Change |
|---|---|
| 2026-10-21 | Plan drafted with the owner's decisions of 2026-10-20; no code changed |
| 2026-10-21 | Amendment 1 (section 16): block tests and relationship tests are separate A/B types; a relationship is built once both blocks exist |

## 15. Related records

**Layer B (2026-10-21):** the relationships between these blocks are planned in [GOV-010](gov-010-relational-implementation-plan.md) and compared in [RES-008](../research/res-008-relationship-gap-analysis.md); a relationship is built once both of its blocks exist, before either is promoted (Amendment 1, section 16).

[GOV-008](gov-008-human-like-learning-motivation.md),
[RES-006](../research/res-006-human-like-learning-gap-analysis.md),
[RES-007](../research/res-007-hard-and-deferred-differences.md),
[RES-005](../research/res-005-skill-discovery.md),
[RES-001](../research/res-001-central-behavioral-problem-and-plan.md),
[ASS-003](../assurance/ass-003-policy-learning.md),
[ASS-006](../assurance/ass-006-realtime-action-lag-incident.md),
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md),
[FND-001](../foundation/fnd-001-ab-protocol.md),
[ADR-001](../decisions/adr-001-shared-universal-brain.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md),
[ADR-006](../decisions/adr-006-streams-may-observe-one-another.md),
[`ai_learning.mmd`](../../assets/ai_learning.mmd).

## 16. Amendment 1 (2026-10-21): two test types

Source: the owner, 2026-10-21; judgement rules are the assistant's and revisable.
The full rules are in [GOV-010](gov-010-relational-implementation-plan.md) section 4A.

- **Gate P7 is a T1 block test** (baseline versus baseline plus the block). The
  block's own A/B is run with the block's **minimal wiring set** connected, if the
  block cannot act without lines. A **T2 relation test** (with and without a
  relationship, both blocks present) is run for each relationship in GOV-010.
- **Gate P1 gains a field:** the research design for each block lists its minimal
  wiring set, by RES-008 line identifier.
- **Gate P8 gains conditional promotion:** a block whose T1 is not positive can be
  promoted together with its positive relationships as a unit, only if a unit test
  (block plus those relationships against the baseline) is positive on all five seeds.
- **Order of work (section 5):** a relationship may be built once both of its blocks
  exist, before either is promoted; the block's T1 and the relationship's T2 are
  then run separately, sharing runs where the configuration is identical.
- **Resources (section 4):** the shared-run rule lowers the combined A/B estimate to
  roughly 220 to 290 hours across both layers.

## 17. Amendment 2 (2026-10-21, response to the branch audit)

Source: [ASS-009](../assurance/ass-009-audit-response-adjustments-branch.md), findings
3.5, 3.8, 3.11, 3.12. Judgement calls are the assistant's and revisable.

**17.1 Which gates are mandatory.** The gates of section 3 are not all required of every
package.

| Gate | Required |
|---|---|
| P1 research design, P3 verify before edit, P4 build behind a switch, P8 decision, P9 rebaseline | Every package |
| P2 frozen protocol, P7 A/B | Packages that change learning behaviour (not WP-15a, WP-16, WP-17) |
| P5 tests (unit, contract with falsification, documentation check) | Every package; the import-graph isolation test whenever the evaluator boundary is touched; wiring tests whenever a stepping body is touched |
| P6 mock run | Behaviour-changing packages and infrastructure that touches the loop |
| Second-reader check | Audit and registry infrastructure only |
| Long-run protocol | Packages whose effect needs more than an hour (growth, consolidation horizon) |
| P10 rollback | Every package (the switch) |

**17.2 Costs the schedule left out (planning assumptions, revised from actuals).**
The 190-hour figure in section 4 counts A/B run time only. Add, per behaviour-changing
package: implementation and review (about 1 to 3 working days for a block), protocol
writing and freezing (about half a day), analysis and records (about half a day), and
re-runs after crashes (allow about 15 % extra run time). Also maintained: a promoted
baseline and a development baseline (GOV-010 4B). **Storage:** keep metrics, logs and
the final checkpoint of designated baselines; delete replay buffers and intermediate
checkpoints after analysis (a run's replay is up to about 5 GB). The calendar estimate
is "several months of part-time work"; it is recorded as an assumption and replaced by
measured effort per package after the first three.

**17.3 Learned-versus-declared review (gate P4/P5, every package).** "Computed from
experience" is not the same as "learned". Each research design contains this record:

| Question | Answer to record |
|---|---|
| What is **learned from data** by this mechanism? | the parameters, features or estimates |
| What is **fixed by design**? | architecture, loss form, hyperparameters, the definition of the measured quantity |
| Does anything fixed encode a **game or task meaning** (a named event, a named goal, a hand-set threshold tied to a task)? | must be "no" |
| Which **observation** would show the intended distinction was learned, not a proxy exploited? | an evaluation-only measurement (ADR-002), for example correlation of the signal with the intended property on held-out data |
| What would a **proxy-exploiting policy** do, and does a test detect it? | a synthetic proxy-maximiser and a test that flags it |

A mechanism that cannot answer these is not built. Fixed meanings are not allowed; fixed
mechanics are.

**17.4 WP-15 split.** WP-15a (legacy path removal) and WP-15b (removal or translation of
Minecraft-only learning terms) as in ADR-004 Amendment 2. WP-15a needs no A/B; a term in
WP-15b that pays in Doom is a behaviour change and gets a T1.

**17.5 Status of the coverage table (section 8).** The table assigns every recorded
difference to a *proposed* package, a deferral or a deliberate rule. It is a mapping,
not an approval of scope; approval status is in GOV-008 section 2.5.

## 18. Amendment 3 (2026-10-21, owner decisions)

The owner approved the scope of this plan: work packages WP-1 to WP-19, including the
three added here (WP-17, WP-18, WP-19), and the meaning of "addressed". The order in section
5 stays provisional. The target game is Minecraft; the checkpoint ladder (progress,
survivability, thrivability) is set later; the gate G1 question of an interim game is
answered "none for now" (ASS-007 15). **No code is authorised until the owner issues the
first instruction.** Section 13's open items are resolved except the checkpoint ladder.

## 19. Amendment 4 (2026-10-21): execution mode

Source: the owner's instruction to follow through on this plan without running tests or
scanning code. Written **before** any implementation, as the owner requires for departures.
The rules are in [GOV-011](gov-011-execution-record-gov-009.md) (E1 to E11): build-only mode;
gates P5 to P7 deferred; P8 not reached; P1 and P2 deferred in favour of brief entries; P3
narrowed to the regions edited; every behaviour default OFF; WP-15b and the R0 audit not
done now. Each package ends in the state "Built, untested" and is **not** promoted. The
execution record is GOV-011.

## 20. Amendment 5 (2026-10-21, correction)

Section 4 counted the A/A noise floor as 5 hours. Ten paired A/A seeds with two identical
arms are **20 serial hours** (ASS-007 Amendment 7). Add 15 hours to the totals in section 4
and in GOV-010 4B. No other figure in section 4 changes.

## 21. Amendment 6 (2026-10-21): WP-4 confidence integrity

**Authority and scope.** The owner requested suggested plan amendments followed by
implementation. This amendment covers WP-4's uncertainty and calibration path and
its RW-1 consumer in GOV-010. It does not promote a feature or alter a frozen
experiment. All checks run on `main`; development A/A and A/B use Doom. Preserve
the running experiment's code snapshot and use a separate validation directory.

**Observed implementation errors.** `uncertainty.spearman` assigns different ranks
to equal readings: two constant vectors can appear perfectly correlated. WP-4
passes `model_ok=True` when its ensemble is absent. Temperature fitting and its
reported calibration error use the same window. RW-1 reads replay-state confidence
without requiring a passing calibration result. These can turn missing or
uninformative evidence into apparent confidence.

**Exact implementation.**

1. Use average ranks for ties. Correlation is undefined for a constant vector,
   fewer than three observations, or non-finite input; undefined evidence never
   passes `CalibrationMonitor.model_ok`. Reject mismatched pair lengths.
2. Keep a missing ensemble uncalibrated. Under `require_calibrated: true`, absent
   or failed model evidence blocks planning-derived distillation. Reject invalid
   per-row confidence/disagreement and exclude it from the trust history.
3. Measure calibration with the temperature present at entry, then fit the
   temperature for the next call. Use the scored temperature for this call's
   planning confidence too. Return this call's recent-window mean confidence
   and explicit policy/model validity flags. Missing windows invalidate the current
   result even if historical observations passed. This is an online diagnostic;
   overlapping recent windows are not independent evidence or a final holdout.
4. RW-1 accepts only finite, bounded recent-window confidence whose current policy
   calibration flag passed; unavailable evidence uses its existing neutral-release
   path. GOV-010 Amendment 10 defines the consumer contract.

**Meaning of the signal.** Agreement with an action sampled by the behavior policy
measures action-distribution predictability, not whether that action was useful.
Call `1 - confidence` an action-concentration heuristic, not calibrated competence
or knowledge of success. Critic/outcome calibration remains unbuilt (D-29).
Ensemble disagreement is a proxy, not proof that RSSM multi-step rollouts are
accurate. Future promotion requires the existing prediction and behavior gates.

**Verification and inclusion.** Add one standalone contract script covering tied
and constant ranks, missing/non-finite evidence, recovery with informative data,
calibration-before-fitting, and RW-1 rejection/release. Run on `main` with CPU
threads bounded and no GPU use, recording any overlap with behavioral runs.
Keep planning and relation defaults OFF. These checks establish mechanism behavior
only; AB-05/AB-06 and the RW-1 T2 comparison remain required before promotion.

**Architecture recommendation.** Retain the RSSM foundation while testing prediction,
memory retention, skill reuse and transfer separately. Prioritize trustworthy
learning signals before adding more controllers. World-model control has strong
benchmark evidence ([Hafner et al., 2025](https://www.nature.com/articles/s41586-025-08744-2));
that does not validate this branch's intrinsic objectives or its particular wiring.
Human-like learning also involves compositional reuse and learning-to-learn
([Lake et al., 2017](https://arxiv.org/abs/1604.00289)). Treat these as testable
capabilities, not claims established by resemblance to a brain diagram. Neither
"most human-like" nor general superhuman learning is established here.

## 22. Amendment 7 (2026-10-21): categorical integration testing

The owner requested coherence testing both within related learning groups and
between groups. [GOV-013](gov-013-categorical-system-testing-plan.md) defines the
category inventory and execution plan;
[ASS-012](../assurance/ass-012-categorical-system-testing-protocol.md) defines the
contracts, boundary scenarios, failure/resource rules and evidence template.
Add this integration layer to P5/P6 and before the final combined evaluation.
It complements A/A, T1 and T2; it does not replace their promotion requirements.
Only implemented, explicitly declared subsets can be tested. All execution is on
`main`, live development uses Doom, and the active frozen A/A queue is preserved.
This amendment authorizes the documentation scope; no category run is scheduled
or claimed by writing the plan.

## 23. Amendment 8 (2026-10-22): corrective items after the AB-00 interim analysis

Source: the owner, 2026-10-22, on [ASS-015](../assurance/ass-015-ab00-interim-analysis.md) and
[ASS-014](../assurance/ass-014-categorical-harness-first-run-findings.md). Full plan, tests and queue changes:
[GOV-014](gov-014-corrective-plan-after-ab00.md). This amendment records only what changes in this plan.

- **WP-5 (section on the intrinsic signals).** Two consumers of the life-boundary guard clock read it with the
  wrong polarity (ASS-014 F-13), so the information-gain term and the WP-1 trigger did not work as designed.
  They are corrected (GOV-014 item B1). The `t1_wp5_info_gain` A/B (AB-07's information-gain arm) is valid only on
  the corrected snapshot.
- **WP-2 and RW-1/RW-3.** `Consolidator.strength` was cancelled by the optimizer (F-04); it now also scales the
  consolidation learning rate (GOV-014 item B2), which makes the strength chain of [GOV-010](gov-010-relational-implementation-plan.md)
  RW-1 and RW-3 able to change the update.
- **WP-6.** The source phase must actually be diverse: the runner accepts a game-YAML overlay (`_games`), the
  `wp6_diversity` arm rotates the source maps, and a live coverage check gates its A/B (GOV-014 item C). This
  removes WP-6's recorded blocker (GOV-011).
- **Testing (P7 and AB-00 follow-ups).** A fresh-control arm (ASS-007 section 4) is added to the queue (GOV-014
  item A). Three contract suites are added (`_guard_polarity`, `_consolidation_strength`, `_wp6_diversity`).
- **Not pursued by the owner:** the metric change and the random-policy reference. Until a discriminating
  measure exists, P7 A/B results are mechanism measures plus descriptive counts, and P8 decisions are deferred
  (GOV-014 section 9).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-005](../architecture/arc-005-game-onboarding-standard.md): Game onboarding standard
- [ASS-010](../assurance/ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [ASS-011](../assurance/ass-011-main-evaluation-readiness.md): Main evaluation readiness and run record
- [GOV-012](gov-012-execution-record-gov-010.md): Execution record: relational implementation plan (GOV-010)
- [GOV-016](gov-016-predictive-coding-core-plan.md): Implementation plan: predictive-coding core versus the RSSM (WP-20)
- [RES-009](../research/res-009-hard-and-deferred-relationships.md): Hard and deferred relationships: lines that cannot yet be built reliably
<!-- backlinks:end -->
