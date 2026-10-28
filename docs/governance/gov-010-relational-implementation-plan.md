# Relational implementation plan: connecting the learning blocks as the map does

> **Document control:** [GOV-010](../README.md#governance) · Governance / relational implementation plan · **State:** In progress (build-only; untested); scope approved by the owner 2026-10-21 · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21; **revised 2026-10-21** (two test types and the build-order rule, section 4A). **Parent:** [GOV-008](gov-008-human-like-learning-motivation.md).
**Builds on:** [GOV-009](gov-009-implementation-plan.md) (the *blocks*).
**Based on:** [RES-008](../research/res-008-relationship-gap-analysis.md) (the *lines*).
**Status:** implementation is underway under build-only instructions; see
[GOV-012](gov-012-execution-record-gov-010.md). Tests, A/B runs, and promotion remain
pending. Nothing here claims that a connection will improve learning.

## 1. Purpose: two layers

GOV-009 plans the *blocks* of the human-like learning map: attention,
consolidation, executive control, curriculum, and so on. A block is not enough.
In the map the blocks are joined by lines that say what feeds what, and the
learning behaviour comes from those lines. RES-008 compared all 326 lines with
SkyBot: 36 are connected, 44 partial, 106 missing because a block is absent, 26
deliberately excluded, and 114 not established.

This plan is the second layer, built on the first:

1. **Layer A (GOV-009): update the learning blocks.**
2. **Layer B (this plan): correct and create the relationships** so information
   flows between the new and existing blocks as in the map, letting each addition
   be used the way it is in human learning.

**Rule (revised 2026-10-21):** a relationship is built once both blocks it joins **exist** (are built and switchable), **before either block is promoted**. Blocks and relationships are tested separately (section 4A). A relationship to a block that is finally rejected is removed with it; one to a deferred block is itself deferred ([RES-009](../research/res-009-hard-and-deferred-relationships.md)).

## 2. Inputs (unchanged from GOV-009)

The owner's goal, success metric (adaptability at owner-set, evaluation-only
checkpoints), the standing rules (**learned, not declared, always**; one universal
brain; no engine reads; no legacy; peers may be observed including their chosen
actions; compute yields in the order model, streams, stepping), the five-seed,
one-hour evidence standard, and "one difference at a time, plan first, merge when
every difference has been tested and addressed with null and negative results
recorded honestly" apply here without change. Everything is open to change and
optimisation.

## 3. The design idea: make the map executable

The map is only useful if the code can be compared with it. The plan therefore
starts with infrastructure that turns each line into an auditable object.

**The relation registry.** Every relationship becomes a named *channel*,
`rel.<FROM>.<TO>`, declared in one place, with:

| Field | Meaning |
|---|---|
| producer | the learner-side quantity that is sent (never an evaluator or engine value) |
| consumer | what uses it, and how |
| signal | its name, units and time scale |
| switch | `relations.<FROM>.<TO>.enabled`; off sends a neutral constant, reproducing the earlier behaviour |
| statistics | a per-segment summary (mean, spread, last value) written to the existing telemetry and ledger |
| status | which RES-008 line it implements |

**Why this helps.** (1) *Conformance:* a test compares the registry with a
machine-readable copy of the owner's map and lists, line by line, which lines
exist. (2) *Cut tests and ablations:* any line can be switched off for an A/B arm.
(3) *Dead-wire alarms:* a channel whose consumer-side effect stays near zero over a
window is flagged, catching the "guard becomes latch" class the project has met
repeatedly (CLAUDE.md section 4.1). (4) *Leakage guard:* every channel source is
tagged learner-side or evaluator-side, and a test fails if a channel carries an
evaluator value (ADR-002). (5) *Learned-not-declared review:* each channel's
parameters are derived from measured quantities, never fixed by hand to a declared
meaning.

**Scope of the registry (audit response).** The registry measures **implementation
coverage only**: which lines of the map have a software channel, are alive and are
tested. A line being present is not evidence that its function or benefit equals the
human one; progress claims come only from T1/T2 results. Lines are typed:

- *data channel* (a value flows from producer to consumer);
- *control channel* (a signal gates or scales another);
- *descriptive* (a hypothesis in the map with no literal software channel). Descriptive
  lines are recorded and **not wired**, to avoid forcing artificial wiring.

Only data and control channels get a switch and statistics. The conformance test checks
bookkeeping consistency between registry and map; it does not prove that the map-to-code
interpretation is right. RES-008 statuses are derived from RES-006 block status and do not
validate a line's causal direction or the completeness of the map.

**Integration (to be re-read).** The registry hooks into the existing infrastructure
stack, ledger and telemetry sink; the channel calls are made from one helper per
stepping body, never duplicated (the repository's duplicated-body rule).

## 4. Method for every relation cluster

Each cluster follows the owner's structure, with testing and inclusion added:

1. **Theoretical perspective:** why the connection matters in human learning
   (psychology, neuroscience, behaviour), with the evidence-strength label from
   GOV-009 section 2 (A, B, C).
2. **What the connection should add.**
3. **What will be added or changed explicitly**, by existing system name (to be
   re-read before editing).
4. **Thoughts, alternatives and risks.**
5. **Testing procedure.**
6. **Inclusion procedure.**

**Gates.** The gates P0 to P10 of GOV-009 section 3 apply unchanged, with these
additions for relations:

- **Cut test (P5):** the channel can be switched off and the earlier behaviour is
  reproduced exactly.
- **Perturbation test (P5):** a known change at the producer produces the expected
  change at the consumer, with the right sign.
- **Dead-wire check (P6):** the channel's consumer-side effect is not near zero in
  the mock run.
- **A/B arms (P7):** *line connected* versus *line cut*, with every block present in
  both arms, five seeds of one hour, positive on all five.
- **Evaluator isolation (P5):** the channel source is learner-side.

## 4A. Test types and promotion (owner decision, 2026-10-21; the rules marked (assistant) are judgement calls, revisable)

**Owner's instruction.** There are two types of testing, both A/B: **T1**, which
tests individual **blocks**, and **T2**, which tests **with and without the
relationship**. **A relationship is built once the two blocks it joins exist,
before the blocks are promoted.** The rest is left to the assistant's judgement,
which may depend on the situation.

**Definitions.**

| Type | Arms | Question |
|---|---|---|
| **T1 Block test** | Baseline versus baseline plus the block (its own switch on; its relationships connected only as in the block's minimal wiring set, rule 2) | Does the block help by itself? |
| **T2 Relation test** | Baseline plus both blocks with the relationship **cut**, versus the same with it **connected** | Does the connection add anything beyond the blocks being present? |

Both follow [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md): five
seeds of one hour, paired, positive on all five seeds above each seed's region of
uncertainty, an A/A noise floor, mechanism checks before the long run, and null and
negative results recorded honestly.

**Judgement rules (assistant).**

1. **Share runs (corrected in 4B).** The cut arm of T2 can be the changed arm of a T1 *only when that T1's changed arm is exactly the T2's cut configuration* (see 4B). Where it is not, both T2 arms are run.
2. **Minimal wiring set.** A block that cannot act without lines (for example
   attention) declares in its research design the smallest set W of RES-008 lines
   it needs. T1 runs with W connected. Cutting W is reported as an ablation, not
   gated separately.
3. **Cluster-level by default, line-level by exception.** T2 connects or cuts a
   whole cluster when its lines share a function. It escalates to a single line when
   (a) a cluster result is mixed or null despite a positive mechanism measure; (b)
   the line is a guard or gate with latch risk (risk, inhibition); (c) the line has
   evaluator-leak risk (RW-13); (d) the line is a hub whose effect would dominate;
   or (e) the owner asks.
4. **Order inside a cluster.** Build and mechanism-check the blocks (P6); run T1 for
   each block; build the relationship when both blocks exist; run T2.
5. **Promotion.** A block is promoted when its T1 is positive. A relationship is
   promoted when its T2 is positive. **If a block's T1 is not positive but a
   relationship of that block has a positive T2,** that alone is not enough (T2 only
   shows the connection helps given the block). Run the **unit test** (a T1 whose
   changed arm has the block plus its positive relationships connected, against the
   baseline). Promote them together as a unit only if it is positive on all five
   seeds.
6. **Rejection.** A rejected block takes its relationships with it (their results
   are archived as conditional on the block). A null or negative relationship is cut
   and recorded; the block may stay. A promoted relationship whose block is later
   removed is removed.
7. **Development baseline.** A T2 runs on a baseline that includes blocks not yet
   promoted. This development baseline is named in the protocol. Promotion updates
   the promoted baseline; earlier T2 results are re-checked in the final combined
   test.
8. **Noise floor.** Re-run the A/A noise floor after every three promotions, and
   whenever a development baseline changes materially.
9. **Stop early.** If a mechanism or dead-wire check fails in the mock run (P6),
   stop before the 10-hour test.
10. **Final check.** The combined test (AB-15) and the leave-one-out arms include
    leaving out the top relationship clusters as well as the top blocks.
11. **Records.** Every result record states its type (T1, T2 or unit), arms and
    development-baseline identifier.

**Cost.** T1 for about 15 blocks at 10 hours each is about 150 hours; T2 with shared
runs adds about 5 hours per cluster, about 70 hours for 14 clusters; line-level
escalations add 5 to 10 hours each; noise-floor re-estimates add about 35 hours.
The total across both layers is roughly **220 to 290 hours** (earlier, unshared
estimate: about 330).

## 4B. Arm matrix, development baseline and corrected run sharing (audit response, 2026-10-21)

Source: [ASS-009](../assurance/ass-009-audit-response-adjustments-branch.md), findings 3.4
and 3.7. The assistant's judgement; revisable.

**"Exists".** A block **exists** when it is built, has passed its mechanism and wiring
checks (P6), and has joined the **development baseline**. Joining is not promotion.

**Development baseline.** D0 is the promoted baseline. Blocks join one at a time, in
build order: D1 = D0 + X, D2 = D1 + Y, and so on. A block leaves the development baseline
if its T1 is clearly negative (negative on all five seeds above the uncertainty region)
or the owner removes it.

**Arm matrix for two blocks X (joined first) and Y and their relationship r.**

| Test | Baseline arm | Changed arm |
|---|---|---|
| T1(X) | D0 | D1 (X, with the minimal-wiring lines whose other end is in D0) |
| T1(Y) | D1 | D2 (Y, with the minimal-wiring lines whose other end is in D1; **r cut**) |
| T2(r) | D2 with r **cut** | D2 with r **connected** |
| Unit(X, r) | D0 | D1 with the positive relationships of X connected |

**What can be shared.** T1(Y)'s changed arm (D2 with r cut) is identical to T2(r)'s cut arm
when the code, configuration and seeds are the same, so one run set serves both. T2 then
costs one extra arm (about 5 hours). When X and Y are built together and neither is in the
baseline of the other's T1, there is **no** sharing and T2 costs two arms (about 10 hours).

**A block that is inert without a partner.** If every line in X's minimal wiring set points
at a block not yet in the development baseline, T1(X) is deferred until that partner
exists, and X and its partner are tested as a pair (a T1 whose changed arm contains both,
with the connecting lines connected).

**Retention of conditional results.** If a block is rejected, every result computed on a
development baseline that contained it is archived with that baseline's identifier and
marked "conditional on a block later rejected". It is not deleted and not used for
promotion.

**Meaning of "one difference at a time".** Three different things, not to be confused:
(1) *one-at-a-time implementation*: one package is built per change; (2) *one-at-a-time
causal testing*: each T1 or T2 changes exactly one thing between its two arms, apart from
the declared minimal wiring set and unit tests; (3) *prerequisite wiring*: the minimal
wiring set, clusters and unit tests are declared exceptions with defined arms, so what a
result supports is always stated.

**Corrected cost.** Using the sharing rule above, assume T2 shares one arm for about 60 %
of clusters: about 7 hours per cluster, about 100 hours for 14 clusters. With about 150
hours of T1, line-level escalations, noise-floor pooling (ASS-007 14.3) and non-run costs
(GOV-009 17.2), the A/B time across both layers is roughly **250 to 330 hours** plus the
non-run costs. This supersedes the 220 to 290 hours of section 4A.

## 5. Sequence

| Phase | Work | Notes |
|---|---|---|
| **R0** | Audit the 114 "not established" lines | Read the code once for the lines that matter; revise RES-008 with evidence (a revision, not a rewrite). No code changes. |
| **R1** | Build the relation registry and conformance test | Infrastructure; no behaviour change; prerequisite for all clusters. |
| **RW-15** | Register and instrument the 36 connected lines | Protects the existing core. |
| **Clusters** | RW-1, RW-3, RW-4, RW-2, RW-5, RW-6, RW-7, RW-8, RW-11, RW-9, RW-10, RW-13, RW-12, RW-14 | Provisional order, each after the blocks it needs. Final order set with the owner. |

Blocks are built first and relations second (a relation needs its blocks to *exist*, not to be promoted): for example RW-2 (attention network) needs WP-7
first; RW-3 (consolidation chain) needs WP-2; RW-8 (abstraction and transfer) needs
WP-19, WP-6 and WP-1 and is the cluster closest to the owner's main metric.

## 6. Phases R0 and R1 in detail

### R0. Audit the lines not established

- **Why:** a third of the lines have no basis either way; building or deleting on a
  guess would break the "report only what is certain" rule.
- **What:** for the lines inside a cluster about to be built, read the producing and
  consuming code and classify the line connected, partial or missing, with evidence.
- **Explicit:** documentation only; update RES-008 rows and the cluster's research
  record.
- **Test:** a second reader reproduces the classification from the evidence.

### R1. Relation registry and conformance

1. **Theory:** an explicit wiring diagram is how engineers keep a complex system
   consistent with a specification (design-by-contract; model-based verification).
   Evidence strength: A (engineering practice).
2. **Adds:** the ability to say, at any time, which lines of the owner's map exist in
   the code, carry signal, and are alive.
3. **Explicit changes (to be re-read):** a registry module and a machine-readable
   copy of the map (generated from the owner's file, never edited by hand); channel
   statistics in the telemetry sink and ledger; a conformance test; dead-wire alarms;
   the evaluator-tag leakage test; a configuration namespace for the switches.
4. **Thoughts and risks:** overhead per step must be small (statistics aggregated per
   segment, not per step); the registry must not become a second source of truth
   that drifts from the map, hence generation from the owner's file.
5. **Testing:** registry round-trip; conformance reports exactly the implemented
   lines; a synthetic dead channel triggers the alarm; an evaluator-tagged source
   fails the leakage test; overhead measured on the training host against the act
   latency budget (ADR-005).
6. **Inclusion:** Gates P1, P3 to P6; no A/B.

## 7. Relation clusters

### RW-15. Protect and instrument the connected core

**Lines in scope (36):** NS-09, NS-10, NS-15, NS-13, NM-01, NM-02, SY-01, SY-18, SY-19, SY-20, SY-21, SY-09, SY-25, CP-22, PH-14, PH-16, MS-01, MS-05, MS-11, MS-13, ME-04, ME-05, ME-06, ME-09, ME-11, ME-19, ME-20, ME-22, ME-23, MO-05, MO-23, MO-24, MO-31, ID-08, OU-05, CP-24. **Status of those lines now:** Connected 36. **Blocks needed first:** none (existing flows).

1. **Theoretical perspective.** Homeostasis and monitoring: a system that learns by one dominant loop (reward and prediction error driving weight updates) must keep that loop observable, because a silent break in it cancels every addition built on top. Evidence strength: A (engineering practice in reinforcement learning).
2. **What the connection should add.** Each existing connected line becomes a named, logged channel whose flow is measured every segment, so a later change cannot silently cut it.
3. **What will be added or changed explicitly (to be re-read).** Register the 36 connected lines as channels in the relation registry (R1) with producer, consumer, units and a per-segment flow statistic (for example the correlation between the reward signal and the weight update magnitude; the share of encoder gradient coming from replay). Add dead-wire alarms that reuse the existing ledger and telemetry. No behaviour change.
4. **Thoughts, alternatives and risks.** This is the cheapest relational work and protects everything else. Several of these lines were broken before and found only by accident (the action-credit lag cut the line between an action and its consequence for the whole of the earlier runs).
5. **Testing procedure.** Perturbation tests: change a producer by a known amount and confirm the consumer responds with the expected sign. A regression test that every connected line has a channel entry. A cut test per channel. No A/B (no behaviour change); a parity mock run must stay inside the noise floor.
6. **Inclusion procedure.** Gates P1 to P6; no A/B. Prerequisite: R1.

### RW-1. Learning-signal modulation: gain, discount and uncertainty control learning dynamics

**Lines in scope (14):** NM-17, NM-05, NM-06, NM-07, NM-08, NM-09, MO-17, NM-10, NM-11, NM-12, MO-26, ID-21, NM-03, NM-04. **Status of those lines now:** Not established 7, Missing 7. **Blocks needed first:** WP-9, WP-4, WP-7.

1. **Theoretical perspective.** Neuromodulators set the gain, learning rate and exploration of the brain according to expected and unexpected uncertainty (Yu and Dayan on acetylcholine and norepinephrine; Aston-Jones and Cohen on adaptive gain; Doya on neuromodulators as meta-parameters). Evidence strength: B for the neuroscience; A for uncertainty-modulated learning rates, temperature control and adaptive entropy in reinforcement learning.
2. **What the connection should add.** Learning rate, exploration temperature, discount and consolidation strength respond to measured uncertainty instead of staying fixed.
3. **What will be added or changed explicitly (to be re-read).** After WP-9 (controllers) and WP-4 (uncertainty): a modulation layer maps measured quantities (calibrated uncertainty, learning progress, surprise) to a small set of coefficients (policy learning rate, entropy target, consolidation weight, return normaliser). One owner per coefficient. Channels: uncertainty to entropy; uncertainty to consolidation; discount to reward normalisation.
4. **Thoughts, alternatives and risks.** Three controllers (WP-8, WP-9, WP-13) can fight; this plan makes the modulation layer the single owner of each coefficient. Excitatory and inhibitory weight lines (GLU, GABA) and the receptor line (NMDA) have no established machine analogue and go to RES-009.
5. **Testing procedure.** Unit: each coefficient stays inside bounds under synthetic uncertainty traces; no oscillation under a step change. Cut tests per channel. A/B with the modulation layer on and off.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-9, WP-4. Feeds RW-2, RW-3.

### RW-2. The attention network

**Lines in scope (23):** CP-01, CP-02, CP-03, CP-04, SY-24, SY-27, SY-29, CP-23, MO-04, MO-06, MO-21, MO-15, MO-20, ME-01, PH-08, PH-10, ID-13, ST-22, ST-35, SO-22, SO-23, SO-25, CP-31. **Status of those lines now:** Missing 22, Partial 1. **Blocks needed first:** WP-7.

1. **Theoretical perspective.** Attention is a control signal that decides what is encoded and remembered. Bottom-up salience and top-down relevance (Corbetta and Shulman) combine; Bayesian surprise attracts attention (Itti and Baldi); curiosity and interest bias it. Evidence strength: A for attention mechanisms in machine learning; B for the control-signal account.
2. **What the connection should add.** Attention weights computed from prediction error, learning progress, relevance and habituation, and applied to encoding and memory formation, so processing follows what is informative.
3. **What will be added or changed explicitly (to be re-read).** After WP-7 builds the attention blocks: route prediction error, curiosity and learning-progress signals as attention logits; route habituation and boredom as attenuators; apply the result to the encoder input and to which transitions are written to or sampled from replay. About 24 of the missing lines in RES-008 converge here.
4. **Thoughts, alternatives and risks.** Highest leverage: one block unlocks about a quarter of the missing lines. Risk: attention that keeps itself high on noisy inputs (the noisy-TV problem); the surprise signal must be the action-caused part, as in WP-1.
5. **Testing procedure.** Unit: attention weights normalise and respond to scripted prediction-error spikes. Perturbation: inject surprise into one image region and confirm attention shifts there. Cut tests. A/B on mechanism measures and the adaptability proxy.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-7, WP-5.

### RW-3. The consolidation chain

**Lines in scope (26):** SY-03, PH-03, PH-04, PH-05, PH-07, PH-01, PH-02, PH-09, CP-14, CP-15, CP-13, NS-14, ST-03, ST-02, ST-34, MO-16, SY-16, NM-06, CP-18, CP-19, CP-20, CP-21, ST-04, SY-32, SY-07, PH-06. **Status of those lines now:** Partial 6, Missing 14, Not established 6. **Blocks needed first:** WP-2.

1. **Theoretical perspective.** Complementary learning systems (McClelland, McNaughton and O'Reilly): offline replay of recent experience trains the slow store while a measure of what matters (emotional salience, repetition, reward) decides what is consolidated; forgetting is regulated. Evidence strength: B for the neuroscience; A for replay and regularisation in machine learning.
2. **What the connection should add.** A defined flow from replay, through a consolidation phase with importance weighting, into the durable policy and world model, with forgetting measured and regulated.
3. **What will be added or changed explicitly (to be re-read).** After WP-2: channels from replay and the forgetting probes into the consolidation phase; importance weights from learning progress and salience; checkpointing tied to consolidation (the checkpoint becomes the end of a consolidation, not only a timer); evaluation results (RP) and early stopping (BREAK) feed consolidation strength. The link from sleep to creativity (REM to CREAT) is a hypothesis and goes to RES-009.
4. **Thoughts, alternatives and risks.** Evaluation feeding consolidation (RP to CONS) must not leak the evaluator's held-out checkpoints (ADR-002); only learner-side measures such as fresh-data loss are used.
5. **Testing procedure.** Perturbation: add a synthetic important episode and confirm it survives a task change more than an unimportant one. Cut tests. A/B AB-03.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-2, WP-6, RW-1.

### RW-4. From episodes to semantic knowledge

**Lines in scope (21):** SY-02, SY-04, CP-39, MS-06, MS-07, MS-08, MS-09, MS-10, MS-04, CP-16, CP-17, SY-08, ST-07, CP-11, CP-12, OU-01, OU-02, ST-01, ST-14, SO-24, PH-12. **Status of those lines now:** Partial 5, Not established 14, Missing 2. **Blocks needed first:** WP-18, WP-1, WP-19.

1. **Theoretical perspective.** Episodic memories are separated (pattern separation, Marr; Yassa and Stark), and repeated, similar episodes are gradually abstracted into semantic knowledge (Tulving; complementary learning systems). Retrieval brings the relevant past into present decisions. Evidence strength: B; A for episodic control and retrieval in reinforcement learning.
2. **What the connection should add.** A pipeline from raw episodes, through separated and indexed memory, to abstracted knowledge, and from retrieval back into decisions.
3. **What will be added or changed explicitly (to be re-read).** After WP-18 (episodic store), WP-1 (effect keys) and WP-19 (contrastive features): channels from replay to the episodic store; a separation stage (contrastive keys) before writing; periodic summarisation of clusters of similar episodes into semantic entries (effect keys and typical outcomes); retrieval back into the policy input. The knowledge graph is either fed by effect keys through this chain or removed (WP-15).
4. **Thoughts, alternatives and risks.** Semantic entries must be learned summaries (cluster outcomes), not engine-derived facts. Prompt-context lines (CTX) are not applicable to this system and are excluded from implementation.
5. **Testing procedure.** Unit: separation keeps near-duplicates apart; summarisation recovers known clusters. Perturbation: repeated similar episodes create one semantic entry. A/B on repeated-situation performance.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-18, WP-1, WP-19.

### RW-5. Executive control

**Lines in scope (19):** SY-10, SY-11, SY-12, SY-13, SY-14, SY-28, CP-29, CP-30, ME-08, SY-30, SY-31, ME-15, ST-31, CP-40, CP-41, ID-19, ID-18, ID-14, CP-44. **Status of those lines now:** Missing 8, Not established 9, Deliberate 1, Partial 1. **Blocks needed first:** WP-4.

1. **Theoretical perspective.** Prefrontal cortex exerts top-down control over working memory, planning and inhibition (Miller and Cohen), with gating of what enters working memory (O'Reilly and Frank's prefrontal-basal-ganglia model) and supervisory attention over habit (Norman and Shallice). Evidence strength: B; A for planners and adaptive computation in machine learning.
2. **What the connection should add.** A controller that decides, from calibrated confidence and available time, whether to act on habit, plan, or gather information, and that gates what enters and leaves short-term context.
3. **What will be added or changed explicitly (to be re-read).** After WP-4: an arbitration channel from uncertainty to the choice between the policy and the planner; a gating channel for which latents are held in the policy's short memory window (WP-7); inhibition as the existing action masks driven by the controller; the controller's output logged per decision.
4. **Thoughts, alternatives and risks.** The whole controller must fit the real-time decision budget; the arbitration itself must be nearly free (ADR-005). The first version arbitrates only during the learning phase and standby (WP-4).
5. **Testing procedure.** Unit: arbitration switches to the planner when uncertainty is high and budget allows, and to habit otherwise, on scripted inputs. Latency test. Cut tests. A/B AB-05.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-4, RW-1.

### RW-6. Metacognition and strategy selection

**Lines in scope (21):** CP-33, CP-34, CP-35, CP-36, CP-37, ST-38, ST-39, ST-40, ST-41, MO-10, MO-11, MO-14, ST-12, ST-13, SO-28, ID-23, MO-12, MO-13, MO-27, MO-30, MO-07. **Status of those lines now:** Missing 11, Not established 9, Deliberate 1. **Blocks needed first:** WP-4, WP-8.

1. **Theoretical perspective.** Learners monitor their own knowledge and control their study (Nelson and Narens' monitoring-and-control model; judgements of learning), and choose strategies by expected benefit. Calibration of confidence is measurable. Evidence strength: B; A for calibration and for bandit selection of training recipes.
2. **What the connection should add.** Self-monitoring (confidence, progress estimates) that feeds a selector over training recipes (replay scheduling, sampling, augmentation, practice allocation).
3. **What will be added or changed explicitly (to be re-read).** After WP-4 and WP-8: a calibrated-confidence channel; credit assignment feeding the competence tracker; a strategy selector (a bandit over a small set of existing recipes such as replay mix and sampling schedule) driven by the progress of each; the selector's choices logged. Feedback signals from humans (FB) are deliberately excluded.
4. **Thoughts, alternatives and risks.** 'Strategy selection' must choose among learner-side options, not declare goals. Keep the set of strategies small to keep the bandit stable in one hour.
5. **Testing procedure.** Unit: the selector finds the best of several synthetic recipes. Calibration metric tests. Cut tests. A/B on progress rate.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-4, WP-8, RW-5.

### RW-7. From motivation to objective and attention

**Lines in scope (18):** MO-08, MO-09, MO-22, ST-20, ST-19, MO-03, MO-29, MO-01, MO-02, MO-28, SO-14, SO-15, SO-18, ME-17, ME-16, ME-24, SO-30, ID-12. **Status of those lines now:** Not established 5, Partial 7, Missing 5, Deliberate 1. **Blocks needed first:** WP-3, WP-5, WP-8.

1. **Theoretical perspective.** Interest and motivation follow perceived competence and progress (self-determination theory; flow, Csikszentmihalyi; goal-setting theory, Locke and Latham), and exploration is directed at what can be learned (intrinsically motivated goal exploration, Oudeyer and colleagues). Evidence strength: B; A for learning-progress-driven exploration.
2. **What the connection should add.** An objective in which progress, competence and learnability determine what the agent pursues, and goals arise from experience, not from a list.
3. **What will be added or changed explicitly (to be re-read).** After WP-3, WP-5, WP-8: channels from the learnability frontier to the objective mixture; from competence to goal choice; from exploration to information gain; goals formed from effect keys (WP-1), never declared. The identity, culture and belonging lines (IDENT, CULT, BELONG) have no learned analogue and go to RES-009.
4. **Thoughts, alternatives and risks.** The danger is declared goals by another name. Every goal here must be derived from the agent's own effects and progress.
5. **Testing procedure.** Unit: goal choice follows progress on synthetic contexts. A declared-knowledge review item. Cut tests. A/B.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-3, WP-5, WP-8, RW-6.

### RW-8. Abstraction and transfer

**Lines in scope (26):** CP-25, CP-28, CP-26, CP-27, CP-10, ME-25, ST-25, ST-30, ST-05, ST-06, ST-27, CP-32, OU-11, OU-06, OU-07, ME-21, ST-09, ST-28, CP-42, OU-12, OU-13, OU-16, OU-15, OU-14, OU-09, OU-08. **Status of those lines now:** Partial 2, Not established 18, Missing 5, Deliberate 1. **Blocks needed first:** WP-19, WP-6, WP-1.

1. **Theoretical perspective.** Transfer depends on shared structure (Thorndike's identical elements; Gentner's structure mapping), is promoted by varied practice, and rests on abstracted schemas. Evidence strength: B; A for invariance, augmentation, contrastive representations and domain randomisation; C for analogy.
2. **What the connection should add.** A route from varied experience, through abstracted features, to reuse in an unfamiliar environment: the line the owner's success metric actually measures.
3. **What will be added or changed explicitly (revised 2026-10-21; Amendment 3).** First register learner-side WP-19 representation-loss measurements only: predictive loss (CP-25), contrastive loss (ST-30), and auxiliary loss (CP-10). They are activity measures, not evidence of abstraction or transfer. Do not wire held-out evaluation outcomes into the learner. Do not yet add policy inputs, a transferable skill library, or a novelty-driven learning-rate change: the transfer probe is offline, the target is undecided, and no learner-side cross-game novelty definition or kill criteria for changing the policy is specified. Revisit those mechanisms in a separate plan revision after their inputs, owner, bounds, feedback loop and separation from evaluator signals are defined. Analogical mapping has no reliable method and remains in RES-009.
4. **Thoughts, alternatives and risks.** This cluster is the direct route to the main metric, but current representation losses and an offline frozen-encoder probe cannot show that the acting policy reuses a representation in an unfamiliar game. Avoid implying a transfer result from proxy-map classification or channel coverage alone.
5. **Testing procedure.** For the observation-only first pass: verify the three channels report the corresponding WP-19 metrics when available and skip absent metrics rather than recording zero; confirm no consumer or training value changes. The later transfer mechanisms require a target, learner-side transfer design, shuffled-task controls and the adaptability metric. A/B AB-15 depends on that later design.
6. **Inclusion procedure.** Gates P0 to P10. Prerequisite: WP-19, WP-6, WP-1, Gate G1.

### RW-9. Social and peer learning

**Lines in scope (15):** SY-33, SY-34, SO-04, SO-05, ST-29, ME-12, ME-13, NM-13, NM-14, SO-06, SO-07, ME-14, SO-01, SO-02, SO-03. **Status of those lines now:** Missing 10, Not established 1, Deliberate 4. **Blocks needed first:** WP-12.

1. **Theoretical perspective.** Learning from others is a major human channel (Bandura's observational learning; Tomasello on cultural learning and shared intentionality). Evidence strength: B; A for imitation and multi-agent learning; C for undeclared peer observation across separate instances.
2. **What the connection should add.** Each stream's learning is informed by what its peers do and what follows, within the rules of ADR-006.
3. **What will be added or changed explicitly (Amendment 5, 2026-10-21).** RW-9 is staged. **Stage A (this build):** retain WP-12's allow-listed view/action board and separate predictor; save its per-segment predictor-versus-baseline metrics and view diversity metrics as `_last_peer_metrics`; register read-only SO-04 (peer-board availability/context count), SY-33 (action prediction diagnostic), and SY-34 (next-view prediction diagnostic) channels. These channels report activity and baseline comparisons only; they do not claim imitation benefit or alter the shared encoder, reward, policy, or skill library. **Stage B (deferred):** a shared-encoder auxiliary loss needs an explicit representation ownership/training interface and a separate stability gate; peer-derived skill candidates need a stable effect identity and a route into observer-owned WP-1 probation that never imports peer success. Neither interface exists in this build. Do not use peer success, reward, or hidden game state as supervision. SO-01 to SO-03, SO-05 to SO-07, ME-13 to ME-14, NM-13 to NM-14 and ST-29 remain excluded or deferred under the existing rule register.
4. **Error encountered and reason for staging.** WP-12's `PeerPredictor` is trained separately from the shared encoder, and `_peer_phase` prints metrics without retaining them for registry observation. Its hypotheses contain only an action and normalised view-change vector; they lack an effect key and a WP-1 probation candidate interface. Directly implementing the original wording would either change representation training without an ownership/stability contract or treat a peer's observed change as a verified skill. Stage A makes the already-computed learner-side evidence observable. Stage B requires the missing interfaces before any behavior change.
5. **Thoughts, alternatives and risks.** Observation of a peer's actions must not become “copy this,” which would declare it good. The signal is prediction and outcome, not imitation reward. Prediction accuracy is not evidence of better policy performance.
6. **Testing procedure.** Stage A tests to write later: all three channels skip unavailable peer state; they match saved learner-side metrics; off leaves training and policy inputs untouched; predictor metrics retain their repeat-action and no-change baselines. Stage B requires representation stability, observer-only probation, leakage isolation and cut tests before AB-13.
7. **Inclusion procedure.** Stage A: P1, P3 to P6; observations only, no A/B. Stage B: full P0 to P10 and AB-13 after its interfaces and safety gates are specified. Prerequisite: WP-12, RW-8.

### RW-10. Risk, safety and their effect on cognition

**Lines in scope (11):** SY-15, SY-16, SY-17, MO-25, ME-03, SO-17, PH-13, SO-19, MO-18, MO-19, SO-16. **Status of those lines now:** Not established 7, Missing 3, Partial 1. **Blocks needed first:** WP-11.

1. **Theoretical perspective.** Threat learning (LeDoux) biases attention, memory consolidation and caution; excessive anxiety narrows attention and impairs working memory (attentional control theory). Evidence strength: B; A for risk-sensitive and constrained reinforcement learning.
2. **What the connection should add.** Learned danger that modulates attention, consolidation and exploration, and a guard that prevents the modulation from becoming paralysing.
3. **What will be added or changed explicitly (Amendment 6, 2026-10-21).** RW-10 is an attribution and boundedness pass. **Implemented by existing cross-cluster channels:** SY-16 (WP-11 risk to the RW-3 consolidation factor) is bounded by the shared consolidation-strength mixer; SO-30 (WP-11 risk to WP-5 information gain) has a configured hard floor `1-safe_amp` and releases to base when switched off. RW-10 records these as its implementation rather than duplicating channels. **Deferred:** a risk-to-attention gain control has no line in RW-10's RES-008 scope, so adding it here would create an untracked coupling; SY-17 risk penalties would create a second penalty path despite WP-11's input-only design; the remaining lines lack a specified learner-side producer/consumer contract. Keep existing rewind/stuck handling as its current infra behavior; do not make it a reward or risk label.
4. **Error encountered and reason for amendment.** RW-10's original implementation instruction did not match its line list: the requested attention-gain connection is not one of the 11 RW-10 map lines. Meanwhile its stated consolidation and exploration effects already exist as SY-16 and SO-30 in earlier clusters. Implementing an extra attention or penalty coupling would either invent a map relationship or change WP-11's no-penalty contract. The workaround is to complete RW-10 by documenting and cross-referencing the two bounded existing controls, and defer the unsupported lines to RES-009 D-22 until a map-consistent source and owner are defined.
5. **Thoughts, alternatives and risks.** The existing channels are bounded but not yet empirically validated. SO-30's floor prevents curiosity from being silenced by this coupling; it does not prove safe exploration. SY-16 is a factor in a product with RW-3's other factors; the product's global bounds and slew limit remain the guard.
6. **Testing procedure.** No new code test is added in this attribution pass. The later RW-3 and RW-7 tests must cover saturated risk, bounded output, switch-off release and death-rate/progress outcomes. An attention path or risk penalty needs its own map-consistent design and tests first.
7. **Inclusion procedure.** Existing channels remain governed by RW-3 and RW-7 inclusion gates. No new RW-10 mechanism is promoted. Prerequisite: WP-11, RW-2, RW-3.

### RW-11. Skills, practice and capability

**Lines in scope (20):** OU-03, OU-04, ST-15, ST-16, OU-05, ME-12, OU-08, ST-21, ID-01, ID-02, ID-03, ID-04, ID-05, ST-33, ST-10, ST-17, ST-18, CP-07, CP-08, CP-09. **Status of those lines now:** Partial 2, Missing 6, Connected 1, Deliberate 7, Not established 4. **Blocks needed first:** WP-1, WP-3.

1. **Theoretical perspective.** Skill follows a power law of practice (Newell and Rosenbloom), passes through cognitive, associative and autonomous stages (Fitts and Posner), and is improved by deliberate practice on weaknesses (Ericsson). Evidence strength: B; A for skill libraries and distillation of deliberate behaviour into habit.
2. **What the connection should add.** A route from practised behaviour to automatic skill, with practice directed at measured weaknesses and transferable priors derived from learned skill outcomes.
3. **What will be added or changed explicitly (Amendment 4, 2026-10-21).** Build RW-11 in two stages. **Stage A (this build):** (a) extend each learned skill's invocation record with up to five completed, non-overlapping windows of 20 eligible `asked_log` outcomes while preserving the existing trailing 20-outcome mastery rule; (b) define *falling competence* as at least two complete windows where the newest success fraction is at least 0.15 below the immediately preceding window; (c) expose a read-only `SkillBank` trend summary containing falling-skill count/fraction, mean current mastery, validated-skill count, and bound option count; (d) add read-only registry channels for falling-skill fraction (OU-03) and bound option count (OU-04); (e) keep these readings out of rewards, action selection, curriculum choices and policy updates. **Stage B (separate gated changes):** use the Stage A record to add an option-practice scheduler that boosts only already learned, currently eligible skill options with falling competence, by a bounded additive logit bias no greater than 0.25; distil a mastered skill policy into the default actor only on replay states for which the skill is bound, the state is inside its recorded initiation set, and the teacher/student action masks agree; use successful learned effect-key records as next-game priors only as candidate skill-selection initialisation, never by copying a game-specific policy or injecting evaluator outcomes. Each Stage B mechanism gets its own T1/T2 switch and kill criterion. No Stage B behavior change is part of the present build.
4. **Error encountered and reason for staging.** The existing `Skill.asked_log` is a trailing window of 20 outcomes; it supports current mastery but contains no earlier window from which decline can be computed. `CurriculumScheduler` receives game-adapter contexts and calls the adapter's `set_context`; it has no learner-skill consumer. The current dream distiller has a dream-policy teacher, not a skill-policy teacher. Existing skill warm-start copies an entire saved policy after shape checks and does not provide a game-agnostic learned prior. Implementing the original wording directly would either produce an inert curriculum edge, treat current mastery as a trend, or copy a game-specific skill policy as a prior. The registry records one scalar or pair per channel, so the four aggregate diagnostics are exposed in the SkillBank summary while only two accurately matching map lines receive registry channels. Stage A supplies the missing evidence contract without changing behavior; Stage B names the required interfaces and guards before any acting or policy update changes.
5. **Thoughts, alternatives and risks.** The 0.15 decline threshold and 20-invocation windows are fixed measurement choices for this first implementation and must be compared with A/A variation before use in Stage B. Small or infrequently invoked skills may never qualify; this is reported as insufficient evidence, not zero competence. A learned effect key is a candidate identity, not proof that a capability transfers to another game.
6. **Testing procedure.** Stage A tests to write later: persistence across skill-bank save/load; only complete windows enter history; rolling fraction and decline threshold are correct; old registries without the new fields load as empty history; relation channels skip absent data and never change model/reward/action values. Stage B tests: bounded practice bias applies only to eligible, learned option slots; teacher distillation rejects incompatible masks/architectures and changes only actor parameters; prior initialization uses effect-key identity and resets safely when no matching learned key exists. Then T1 for each mechanism and T2 for its relation under the standard protocol.
7. **Inclusion procedure.** Stage A: P1, P3 to P6; read-only, no A/B. Stage B: full P0 to P10 and separately frozen T1/T2 protocols. Prerequisites: WP-1, WP-3 and RW-8 Stage A.

### RW-12. Learning stage and plasticity

**Lines in scope (17):** ID-17, NS-06, NS-07, NS-03, NS-04, NS-05, NS-21, NS-12, NS-16, NS-17, NS-11, NS-01, NS-02, NS-08, NS-18, NS-19, NS-20. **Status of those lines now:** Not established 12, Deliberate 4, Missing 1. **Blocks needed first:** WP-9, WP-10.

1. **Theoretical perspective.** Plasticity is regulated across development: sensitive periods open and close, structure is pruned, and capacity grows (Hensch; Elman's starting small). Deep networks also lose plasticity unless maintained (Dohare et al.). Evidence strength: B; A for schedules; C for engineered critical periods.
2. **What the connection should add.** Stage-dependent plasticity: schedules and refresh tied to the learner's stage, and growth tied to evidence, with consistent handling of the policy across both.
3. **What will be added or changed explicitly (Amendment 7, 2026-10-21).** RW-12 is an attribution pass against WP-9 and WP-10. Record the existing step-based LR schedules, dormant-unit monitor/optional refresh and drift response from WP-9; record WP-10's evidence-based growth decision and preservation of policy weights, plus optional expansion of the policy Adam moments. Do not describe these as one stage-dependent controller. **Deferred:** a progress-derived stage has no operational definition shared by WP-8, WP-9 and the autoscaler; WP-8's progress is a per-context relative loss-change signal and WP-9 schedules are lifetime-step functions. World-model optimizer state is not persisted. Do not add a stage gate or alter growth until stage semantics, persistence, bounds, and missing optimizer-state behavior are specified. Critical-period, receptor, glia, pruning and other biological lines remain in RES-009 D-05, D-06, D-18 and D-23.
4. **Error encountered and reason for amendment.** The original text treats progress, developmental stage, plasticity schedule and growth as if they shared one state. They do not: WP-8 emits smoothed per-context progress; WP-9 uses the lifetime step count and has optional function-preserving refresh; WP-10 chooses growth from compute and learning signals, and only the policy optimizer's saved state can be expanded. Connecting them directly would add an uncalibrated behavioral controller and overstate state preservation. The workaround is to record the exact existing mechanisms and defer the cross-controller stage until a single bounded stage definition and optimizer-state contract exist.
5. **Thoughts, alternatives and risks.** This pass provides no new developmental-stage behavior. The one-hour protocol is insufficient for the proposed long-run plasticity and growth claims; the optional policy optimizer sidecar also remains untested on a real optimizer state.
6. **Testing procedure.** No new unit test is specified for this attribution pass. Existing WP-9/WP-10 tests remain pending. Any later stage controller needs monotonicity/response, source-loss release, bounded LR/refresh/growth perturbation, growth output preservation, and a long-run protocol.
7. **Inclusion procedure.** Existing WP-9 and WP-10 mechanisms retain their own gates and remain default OFF. No new RW-12 mechanism is promoted. Prerequisite for a stage controller: an agreed operational stage definition and state-persistence contract.

### RW-13. Evaluation to training, and outcome feedback

**Lines in scope (19):** ST-01, SO-27, SO-12, ST-23, LC-01, LC-02, LC-03, LC-04, LC-05, LC-06, LC-07, LC-08, LC-09, LC-10, LC-11, OU-18, OU-10, OU-17, OU-19. **Status of those lines now:** Not established 9, Partial 3, Missing 6, Deliberate 1. **Blocks needed first:** WP-14, WP-8.

1. **Theoretical perspective.** The testing effect (Roediger and Karpicke): retrieval and test results strengthen memory and guide study; knowledge of results shapes learning. Outcomes feed back into the learner's priors and self-model. Evidence strength: B; A for validation-driven training.
2. **What the connection should add.** Learner-side evaluation (fresh-data loss, forgetting probes, calibration) feeding training choices, and outcome measures feeding the self-model and priors.
3. **What will be added or changed explicitly (Amendment 8, 2026-10-21).** RW-13 is a cross-cluster attribution and isolation pass. The learner-side validation path is already implemented: ST-02 (RW-3) uses fresh-data loss to scale consolidation; RW-6's strategy selector is credited by fresh-data loss; WP-8 measures learner-side progress and trains its self-model head. WP-14's owner-set checkpoints remain evaluator-only and do not enter these producers. Do not build an evaluator-to-training channel. Outcome summaries into learned priors remain deferred under RW-11 Stage B / RES-009 D-20 because the skill-prior interface is not defined.
4. **Error encountered and reason for amendment.** The original wording grouped learner-side validation and owner-set checkpoints under “evaluation,” although they have different data ownership. Fresh-data measurements already have learner-side consumers; evaluator checkpoint outcomes are isolated by ADR-002; and there is no safe candidate-prior interface for outcome summaries. Routing checkpoint data into training would violate the boundary, while adding duplicate fresh-loss consumers would create redundant owners. The workaround is to count and cross-reference existing learner-side paths, retain evaluator isolation, and defer prior initialization until D-20's interface exists.
5. **Thoughts, alternatives and risks.** The current fresh-data signal can be noisy and the existing selectors are untested. Their presence does not establish improved progress. Evaluator isolation remains a hard rule, not a tunable switch.
6. **Testing procedure.** No new code tests are added in this attribution pass. Existing WP-14 isolation checks and RW-3/RW-6 perturbation and cut tests remain pending. Any future prior flow must prove evaluator-state exclusion and candidate identity/compatibility.
7. **Inclusion procedure.** Existing RW-3, RW-6, WP-8 and WP-14 gates remain in force. No new RW-13 behavior is promoted. Prerequisite for priors: RW-11 Stage B design and ADR-002 isolation evidence.

### RW-14. Resources and capacity

**Lines in scope (9):** SO-09, SO-10, SO-11, SO-26, PH-11, ID-07, CP-44, ID-06, SO-21. **Status of those lines now:** Not established 5, Deliberate 1, Missing 2, Partial 1. **Blocks needed first:** WP-10, WP-17.

1. **Theoretical perspective.** Resource-rational analysis (Lieder and Griffiths): cognition allocates limited computation where it pays. Working-memory capacity limits what can be held. Evidence strength: B; A for compute-aware machine learning.
2. **What the connection should add.** Explicit links from the compute budget to model size, context length and planning depth, in ADR-005's order.
3. **What will be added or changed explicitly (Amendment 9, 2026-10-21).** RW-14 is a resource-path attribution pass. Record the existing WP-10 VRAM and acting-latency measurements feeding its autoscaler and its ADR-005 response order (model, streams, stepping rate); add a read-only ID-06 channel for the configured/active WP-7 context-window width. Do not claim that memory/GPU/latency currently resize the attention context or planning horizon: neither consumer exposes a safe live resizing interface. Do not label fresh-data loss as “early data quality”; it is a learner-side generalisation measurement, not a data-quality score. Keep the configured context and planning horizons fixed. SO-11 and SO-21 remain deliberate exclusions; other unsupported lines go to RES-009 D-24.
4. **Error encountered and reason for amendment.** The original instructions promise measured-resource links to context length and planning depth, but WP-10 only changes model size, stream count and stepping rate. Attention windows and planner horizons are constructed from configuration and have no resizing contract; changing them mid-run could invalidate state or exceed latency bounds. The code also has no defined early-data-quality scalar. The workaround is to record the exact working autoscaler inputs/outputs and defer dynamic context/planning controls until each runtime state transition and release behavior is specified.
5. **Thoughts, alternatives and risks.** Resource signals and scaling recommendations are not evidence that context shortening or shallower planning improves the learner. Current scope is limited to the ADR-005 autoscaler and its measurements.
6. **Testing procedure.** No new code tests are added in this attribution pass. Existing WP-10 tests remain pending. Any dynamic context/planning mechanism needs a state-preserving resize test, latency budget check, bounded controls, and release/cut coverage before AB-12.
7. **Inclusion procedure.** Existing WP-10 and WP-17 gates remain in force. No new RW-14 behavior is promoted. Prerequisite for additional controls: live resize contracts for attention and planning plus a defined resource/quality signal.

## 8. Coverage of the lines

Every missing, partial and unverified line in RES-008 is either inside a cluster
above or recorded below.

| Class | Lines in a cluster | Lines outside every cluster |
|---|---|---|
| Connected | 36 | 0 |
| Partial | 29 | 15 |
| Missing | 98 | 8 |
| Deliberate | 21 | 5 |
| Not established | 104 | 10 |

- **Missing lines outside every cluster (8):** NM-15, PH-15, MS-12, ST-32, ST-37, SO-29, ID-15, ID-22. They join blocks that are deferred ([RES-007](../research/res-007-hard-and-deferred-differences.md)) or deliberately excluded (language, human feedback, extrinsic reward) and are carried by [RES-009](../research/res-009-hard-and-deferred-relationships.md).
- **Partial lines outside every cluster (15):** NM-16, SY-22, SY-23, CP-05, CP-06, CP-38, MS-02, MS-03, ME-02, ME-07, ST-24, ST-26, SO-08, ID-09, ID-10. They are existing partial flows between blocks that are not the focus of any one cluster (for example the reward model to the reward signal, the encoder to working context, and the multimodal-embedding links). Each is completed in the cluster whose block it touches, or hardened by RW-15 instrumentation, as R0 classifies it.
- **Not-established lines outside every cluster (10):** SY-05, SY-06, SY-26, PH-17, CP-43, ST-08, ST-36, SO-20, ID-11, ID-20. They are covered by phase R0 and are assigned to a cluster as the audit classifies them.
- **Deliberate lines (5 outside a cluster):** ME-10, ME-18, ST-11, SO-13, ID-16. They stay excluded unless the owner reopens the rule.

## 9. Resources and schedule

**Revised 2026-10-21:** with shared runs (section 4A), a relation test adds about 5 hours (only the connected arm), so 14 clusters add about 70 hours, not 140; the total across both layers is roughly 220 to 290 hours. The paragraph below states the earlier, unshared estimate.

Each cluster's A/B is two arms (line connected, line cut) times five seeds times one
hour: **10 hours**. With 14 clusters this is about **140 hours** of continuous
running, on top of the roughly 190 hours in GOV-009, plus the noise-floor
re-estimates. Clusters whose blocks are deferred, or whose effect needs longer than
an hour (RW-12), add a long-run protocol instead. As in GOV-009, a cluster that
fails its dead-wire or perturbation checks in the mock run is stopped before its
10-hour test.

## 10. Risks

- **Over-wiring.** More lines mean more places for a silent fault. The registry,
  cut tests and dead-wire alarms exist for this reason.
- **Interactions.** Lines modulate each other (attention, consolidation, risk all
  touch the same signals). Promote one cluster at a time and re-run the noise floor
  after every three promotions; run the leave-one-out check at the end.
- **Guards that become latches.** Several lines are guards or gates (risk, attention,
  inhibition). Every one needs a stated, reachable release condition (CLAUDE.md
  section 4.1).
- **Declared knowledge by another route.** A connection that hard-wires "this signal
  means this" is a declaration. Each channel's content must come from measured
  quantities.
- **Evaluator leakage.** Evaluation-to-training lines (RW-13) are the most exposed;
  the leakage test is mandatory.
- **Compute.** Channels add computation; ADR-005 decides what yields.

## 11. Thoughts

- Relations are built once their blocks exist, before promotion, so that a block that is inert without its lines can be judged with them. The cost is that a relation's test may be run on a block that is later rejected; section 4A handles that.
- The attention and consolidation hubs account for about a third of all missing
  lines, so WP-7 and WP-2 are the highest-value blocks, and RW-2 and RW-3 the
  highest-value clusters.
- R0 may change this plan: if many "not established" lines turn out to be connected,
  the real gap is smaller than RES-008 suggests; if they are missing, larger.
- The registry makes the owner's map a living specification. It also gives an honest
  progress measure: lines connected, alive and tested.

## 12. Documents this plan will produce

| Record | Class | Timing |
|---|---|---|
| Revision of RES-008 after R0 (classification evidence) | Record revision | After R0 |
| Relation-registry design (architecture record, next free `ARC-` ID) | Proposed | With R1 |
| One research design and one protocol per cluster, and a result record | Proposed, Protocol, Record | Per cluster |
| Updates to RES-009 as lines are deferred or built | Revision | Continuous |
| Updates to GOV-008 (change log), GOV-009 and RES-006 | Revisions | After each decision |

## 13. Open items and questions for the owner

**Update 2026-10-21:** the owner approved the scope of this plan (R0, R1, RW-1 to RW-15) and the two test types. Items 1 to 5 below are resolved except that the clusters' order stays provisional. No code is authorised until the owner's first instruction.

1. **Approve the two-layer rule:** (resolved 2026-10-21) relations are built once both blocks exist, before promotion; see section 4A.
2. **Approve the relation registry** as the first relational work, including a
   machine-readable copy of `ai_learning.mmd` generated from the owner's file.
3. **Audit first (R0)?** The plan proposes auditing the not-established lines for
   each cluster just before building it, not all at once.
4. **Line-level versus cluster-level tests:** the plan tests a cluster as one A/B
   with every line in it connected versus cut. Do you want any individual line
   tested on its own?
5. **Map changes:** if you revise `ai_learning.mmd`, the registry's copy and RES-008
   are regenerated and this plan's line IDs change.

## 14. Change log

| Date | Change |
|---|---|
| 2026-10-21 | Relational plan drafted on top of GOV-009; no code changed |
| 2026-10-21 | Owner: two test types (block, with/without relationship); a relationship is built once its two blocks exist, before promotion. Section 4A added with assistant judgement rules; sections 1, 5, 9, 11, 13 revised |
| 2026-10-21 | Amendment 3 narrows RW-8's first build to learner-side WP-19 representation measurements; policy transfer, transferable skills and novelty-driven adaptation are deferred pending a concrete learner-side design and target. |
| 2026-10-21 | Amendment 4 stages RW-11: define a persistent competence-trend record and observation channels now; defer option-practice bias, skill-policy distillation and learned prior initialization to separately switched and tested mechanisms. |
| 2026-10-21 | Amendment 5 stages RW-9: expose WP-12 predictor and diversity evidence as read-only channels; defer shared-encoder training and peer-derived skill candidates until their ownership and probation interfaces exist. |
| 2026-10-21 | Amendment 6 aligns RW-10 with its map scope: attribute the existing bounded SY-16 and SO-30 controls; defer the unlisted attention coupling and undefined penalty/environment flows to RES-009 D-22. |
| 2026-10-21 | Amendment 7 aligns RW-12 with actual WP-8/9/10 state: record existing step schedules and policy growth preservation; defer a shared progress-stage controller and biological mechanisms until stage and optimizer-state contracts are explicit. |
| 2026-10-21 | Amendment 8 aligns RW-13 with learner/evaluator ownership: attribute existing learner-side validation paths, preserve evaluator isolation, and defer outcome priors to RW-11 D-20. |
| 2026-10-21 | Amendment 9 aligns RW-14 with actual resource actuators: attribute WP-10's VRAM/latency path; defer context/planning resizing and undefined early-data-quality measures to RES-009 D-24. |

## 15. Related records

[RES-008](../research/res-008-relationship-gap-analysis.md),
[RES-009](../research/res-009-hard-and-deferred-relationships.md),
[GOV-009](gov-009-implementation-plan.md),
[GOV-008](gov-008-human-like-learning-motivation.md),
[RES-006](../research/res-006-human-like-learning-gap-analysis.md),
[RES-007](../research/res-007-hard-and-deferred-differences.md),
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md),
[`ai_learning.mmd`](../../assets/ai_learning.mmd).

## 16. Amendment 2 (2026-10-21): execution mode

Source: the owner's instruction to follow through on this plan after GOV-009 without running
tests or scanning code. Written before implementation. The rules of
[GOV-011](gov-011-execution-record-gov-009.md) apply; the execution record is
[GOV-012](gov-012-execution-record-gov-010.md). Differences: R0 is done only where a relation
build requires reading its endpoints; a relation is built once both of its blocks are *built*
(state "Built, untested"), consistent with section 4A; T1, T2 and unit tests, the dead-wire
and perturbation tests, and all promotion are deferred. Channels are default OFF.

## 17. Amendment 4 (2026-10-21): RW-11 staging and operational definitions

The implementation blocker and workaround are recorded in RW-11 above. Stage A is
measurement only. The complete history stores the last five completed, non-overlapping
windows of 20 eligible invocation outcomes per learned skill; a partial window is not
compared. `falling competence` is true only when the newest completed window's success
fraction is at least 0.15 below the immediately preceding completed window. It does not
change the existing mastery definition, which continues to use the trailing 20 outcomes.
Stage B is not authorized by this amendment's Stage A implementation; its switches,
interfaces and T1/T2 results are separate requirements before any behavior changes.

## 18. Amendment 5 (2026-10-21): RW-9 observability and deferred consumers

WP-12 already owns a peer view/action buffer, a separate next-action/next-view
predictor, and view-diversity measures. Stage A retains the predictor/baseline and
diversity numbers for relation-registry observation. It does not train the shared
encoder or consume hypotheses as skills. The missing encoder ownership contract
and peer-to-WP-1 effect-identity/probation route are recorded in RW-9 and RES-009
D-21; these must be defined before Stage B changes representation training or
skill discovery.

## 19. Amendment 6 (2026-10-21): RW-10 scope alignment

RW-10's bounded risk-to-consolidation and risk-to-information-gain mechanisms
already exist in RW-3 and RW-7 as SY-16 and SO-30. The earlier instruction to
add risk-to-attention does not identify a line in RW-10's line list; adding it
would create an untracked coupling. Penalty and environment-derived links also
lack a specified learner-side signal and conflict in part with WP-11's input-only
contract. This pass counts the existing controls, records their bounds and
release behavior, and places unsupported lines in RES-009 D-22. No code behavior
is added for RW-10.

## 20. Amendment 7 (2026-10-21): RW-12 interface and state alignment

WP-8's per-context progress is not a persistent developmental stage, WP-9's
learning-rate schedules use lifetime steps, and WP-10 preserves policy weights
with optional policy-Adam expansion only. World-model optimizer state is not
persisted. RW-12 records these exact implemented interfaces but does not join
them into a new controller. A later stage mechanism needs a shared operational
definition, persistence semantics, bounds, release behavior, and a protocol long
enough to evaluate its effects. Biological mechanisms remain explicitly deferred
in RES-009.

## 21. Amendment 8 (2026-10-21): RW-13 data ownership

Fresh-data validation already feeds RW-3 consolidation and RW-6 strategy
selection, while WP-8 measures learner-side progress for the self-model. WP-14
checkpoints remain evaluator-only under ADR-002. RW-13 adds no duplicate consumer
or evaluator channel; outcome-derived priors remain deferred under RW-11 Stage B
and RES-009 D-20 until that candidate interface is defined.

## 22. Amendment 9 (2026-10-21): RW-14 resource actuator scope

WP-10 already measures VRAM and acting latency and applies ADR-005 to model,
stream and stepping-rate decisions. RW-14 adds a read-only ID-06 observation of
the active WP-7 context width. It does not resize attention context or planning
depth; those values are fixed at construction and have no safe runtime resize
contract. Fresh-data loss is not a data-quality score. Other unsupported
resource/context links and early-data-quality measures are deferred under
RES-009 D-24.

## 23. Amendment 10 (2026-10-21): RW-1 confidence source validity

**Authority.** Owner-requested plan improvements and implementation; paired with
[GOV-009 Amendment 6](gov-009-implementation-plan.md#21-amendment-6-2026-10-21-wp-4-confidence-integrity).

**Error.** `policy_uncertainty` currently reads WP-4's `last.conf` without checking
calibration. That is confidence on planning start states drawn from replay, not
the recent-window measurement described by RW-1. Missing or failed calibration
can therefore modulate exploration, attention and consolidation.

**Implementation contract.** WP-4 publishes `calibration_conf` and
`policy_calibration_ok` for its current call. RW-1 consumes `1 - calibration_conf`
only when the flag passes and confidence is finite and within [0, 1]. Otherwise
it returns `None`, allowing the existing bounded release toward neutral to run.
Old producers without the fields also return `None`. Keep channel IDs, coefficient
ownership, switches and neutral values unchanged. Describe the source as calibrated
action predictability; do not label it outcome confidence or competence.

**Evidence.** P5 checks must perturb the validity flag as well as the numerical
source, reject NaN/out-of-range inputs, demonstrate release after evidence vanishes,
and demonstrate recovery. T2 still compares the relation connected versus cut with
identical blocks and data. A live channel or passing calibration diagnostic alone
does not establish a useful causal role. All checks run on `main`, and the frozen
Doom A/A snapshot remains unchanged.

## 24. Amendment 11 (2026-10-21): coherence across relation clusters

The owner confirmed category testing within and between functional groups.
[GOV-013](gov-013-categorical-system-testing-plan.md) overlays functional categories
on the existing map and WP/RW packages without changing node or line identities.
[ASS-012](../assurance/ass-012-categorical-system-testing-protocol.md) adds composed
consumer-effect, controller-ownership, invalid-input, state/recovery and boundary
tests. These extend P5/P6; a live registry channel alone is insufficient evidence.
Partial and observation-only paths retain their limitations. Category coherence
does not promote a relationship: T2 and final combined behavioral evidence remain
required. Tests are serialized on `main` after the active frozen A/A queue.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-006](../architecture/arc-006-relation-registry.md): Relation registry
- [ASS-011](../assurance/ass-011-main-evaluation-readiness.md): Main evaluation readiness and run record
- [ASS-015](../assurance/ass-015-ab00-interim-analysis.md): AB-00 Doom A/A: interim analysis (16 of 20 runs)
<!-- backlinks:end -->
