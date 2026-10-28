# Audit of the Adjustments branch documentation and plan

> **Document control:** [ASS-008](../README.md#assurance-and-evidence) · Assurance / branch audit record · **State:** Record · **Custodian:** Repository maintainer · Revision history: Git.

**Original header (2026-10-21):** Branch audit · Review of the local `Adjustments` diff · **State:** Review record · **Date:** 2026-10-21 · **Scope:** Documentation, decisions, research records, plans and diagram assets added or changed on the branch.

## 1. Purpose and scope

This audit records questions and potential issues found in the branch's
documentation diff. It covers the plan's motivation, evaluation design,
implementation sequence, governance and consistency across the new records.
It does not edit or validate the proposed implementation against the codebase.

The review was limited to the current branch diff and the accompanying
documentation and diagram source. No repository scan or tests were run. The
untracked PNG assets were not independently assessed for visual accuracy; this
review treats the linked map source and its written descriptions as the basis for
the plan.

## 2. Overall assessment

The documentation is unusually explicit about uncertainty, null results,
evaluation leakage, reversibility and the distinction between a human-learning
analogy and a proven improvement. The central direction is understandable:
measure transfer, develop blocks, wire their relationships, and test both.

The main concern is that the program is not yet operationally ready. Its primary
metric depends on an undecided target game and checkpoints; the uncertainty and
promotion rules do not yet support the strength of the claims proposed; and the
two-layer testing sequence has unresolved baseline and cost assumptions. Several
aspirational concepts also risk being treated as implementation requirements
before their meaning has been made measurable. These are plan-quality issues,
not findings that the proposed mechanisms cannot work.

## 3. Findings

### 3.1 High priority: the primary metric cannot yet be run

**Where:** GOV-008 §§2, 6, 10; GOV-009 §§1, 5, 6; ASS-007 §§9, 12.

ASS-007 says Minecraft is deferred, leaves an interim target undecided, and
requires owner-set checkpoint detectors. GOV-009 nevertheless makes the
adaptability metric the intended success measure and places evaluation machinery
and a five-seed A/A run early in the sequence. Until the target, checkpoint list,
detectors, source content and frozen protocol are settled, AB-00 and later
package A/Bs cannot establish the program's stated outcome. Mechanism measures
and held-out Doom maps may be useful proxies, but they cannot answer whether a
learner adapts to an unfamiliar game.

**Question:** What work is allowed to proceed before the target and checkpoint
protocol are fixed, and which results are explicitly development evidence rather
than evidence for the main goal?

### 3.2 High priority: the one-hour and uncertainty rules need a statistical
definition that matches the data

**Where:** ASS-007 §§3–6, 12; GOV-009 §§3–4; GOV-010 §4A.

The proposed primary score is an area under a checkpoint-attainment curve. Its
observations within a seed are repeated, ordered events from one evolving run,
not independent samples. A bootstrap over checkpoint times or curve segments
could therefore give an overconfident interval unless the resampling unit and
dependence structure are specified. The A/A design uses five seeds and the
protocol refers both to a per-seed noise floor and a false-positive rate; five
paired A/A outcomes provide very little information for estimating either
quantity reliably. The claim that the all-five-seeds probability is the
per-seed rate to the fifth power additionally assumes seed independence, which
the protocol says is not verified.

The fixed 40-minute source / 20-minute target split is an assistant default, not
an owner decision, and leaves only 20 minutes to observe adaptation. A single
complete one-hour run also leaves unclear what counts as a time segment and how
the score handles a checkpoint that is impossible, already achieved at target
entry, or dependent on earlier checkpoints. The crash-and-rerun rule can further
alter the effective sample if failures are related to a particular arm.

**Questions:** What is the independent statistical unit? How will the A/A
calibration estimate uncertainty with enough precision? What effect size and
power can this design detect? How are censoring, checkpoint order, seed
restarts, and multiple package comparisons handled? Is 40/20 an acceptable
evaluation budget for the intended transfer question?

### 3.3 High priority: promotion criterion and stated objective may diverge

**Where:** GOV-008 §§2.3, 8, 10; ASS-007 §12.2; GOV-009 §§1, 3, 8.

The promotion rule means a change is adopted if it improves quicker learning
within one hour on all five seeds. That can favor short-horizon gains even when
they reduce later retention or transfer; the records acknowledge this for
long-horizon effects, but still make one-hour promotion the default across
packages. The overall goal is adaptability in unfamiliar environments, while
many individual packages are judged on mechanism measures, Doom-only probes, or
different outcomes such as death rate and repeated-situation performance. The
plan does not yet state what evidence is sufficient to promote a block when its
main adaptability outcome is unavailable, nor how a later transfer result
reverses an earlier mechanism-based promotion.

**Question:** Is promotion an interim engineering decision or evidence that a
change advances the program's success metric? Define the distinction and the
reversal rule.

### 3.4 High priority: T1/T2 ordering and shared-run accounting are ambiguous

**Where:** GOV-008 §2.4; GOV-009 amendment 1; GOV-010 §§1, 4A–5; ASS-007 §13.

The owner requires a relationship to be built once both blocks exist and before
either is promoted. GOV-010 then orders the work as block mechanism checks, T1
for each block, relationship construction, then T2. It is unclear when the
second block is considered to “exist” relative to its T1, and whether either T1
can be interpreted when its minimal wiring set includes an unpromoted block or
relationship.

The cost-saving rule says the T2 cut arm can reuse the changed T1 arm. For a T2
whose two blocks are both absent from the baseline, the T2 cut arm contains both
blocks, while a single-block T1 arm contains only one. Those are not the same
configuration, so run sharing does not apply without a more specific definition
of the development baseline and minimal wiring set. The 220–290 hour estimate
depends on this reuse, and may undercount the actual test matrix and additional
re-baselining.

**Questions:** What exact configurations define T1, T2 and unit arms when neither
block is promoted? Under what conditions are runs genuinely identical and
reusable? How are conditional results retained if a block is rejected?

### 3.5 High priority: “learned, not declared” needs operational boundaries

**Where:** GOV-008 §4; GOV-009 §§3, 7, 9; GOV-010 §§3, 7; ADR-006 amendment 1.

The rule is appropriately strong, but the plans sometimes describe fixed
architectural choices as if they straightforwardly satisfy it. Examples include
turning competence, belonging, curiosity, goals, danger, importance and
“usefulness” into learner-side quantities; deriving goals from effect keys;
assigning hand-designed meanings to signals; or adding an intrinsic/reward term
for a selected proxy. A signal computed from experience is not automatically a
learned concept, and an agent can optimize a proxy without the intended meaning.
The documentation recognizes proxy farming in places, but does not define a
review test that distinguishes learned representations from designer-declared
semantics across all packages.

**Question:** For each mechanism, what exactly is learned from data, what is
fixed by design, and what observation would show that the agent learned the
intended distinction rather than exploiting its proxy?

### 3.6 Medium priority: the human-learning map is a hypothesis source, not a
validated target specification

**Where:** GOV-008 §§2, 5; RES-006; RES-008; GOV-009 §§2, 8; GOV-010 §§3–7.

The documents do say human-likeness is a hypothesis, but the map's edges and
labels appear to drive a large implementation backlog and a registry intended
to measure conformance. A line-by-line structural match can create a false sense
of scientific validation: biological labels such as “attention,” “belonging,”
“sleep,” or “threat” may not map cleanly to the proposed machine mechanism, and
the presence of a channel does not show the function or benefit is equivalent.
RES-008 also derives some line statuses mechanically from RES-006 block status;
that is useful bookkeeping, but does not validate the line's causal direction
or the completeness of the map.

**Question:** Should the registry measure implementation coverage only, with
separate evidence for causal function and outcome, rather than treating map
conformance as progress toward human-like learning?

### 3.7 Medium priority: “one difference at a time” conflicts with the layered
work graph

**Where:** GOV-008 §§2.2, 4–6, 8; GOV-009 §§5, 7; GOV-010 §§1, 5, 7.

The governance record says differences are taken one by one and none are
bundled. The plans instead contain prerequisite chains, minimal wiring sets,
clusters that connect multiple lines, shared development baselines, and
conditional promotion of blocks plus relationships as a unit. This may be the
right experimental design, but it is not literally one difference at a time.
The plan needs to distinguish one-at-a-time implementation, one-at-a-time
causal testing, and multi-component prerequisite wiring. Without that
distinction, package boundaries and what each result supports remain unclear.

### 3.8 Medium priority: implementation, test, and compute requirements exceed
the stated review scope

**Where:** GOV-009 §§3–6, 9; GOV-010 §§3–6.

The common gates require unit and contract tiers, documentation checks,
isolated-root mock runs, GPU and memory measurements, import-graph checks,
conformance tests, dead-wire alarms and a second-reader reproduction. Several
packages also propose synthetic games, detectors, generated registries and
long-run protocols. These are substantial deliverables and sometimes depend on
host capabilities or a target game that is not yet available. The schedule
counts mostly A/B run hours and does not estimate implementation, review,
re-runs, protocol freezing, storage, recovery, or maintaining parallel
development and promoted baselines.

**Question:** Which gates are mandatory for every package, which are specific to
infrastructure or risk, and what is the expected calendar/compute budget once
these non-run costs are included?

### 3.9 Medium priority: compute-pressure order may change the treatment being
tested

**Where:** ADR-005 amendment 1; GOV-008 §4; GOV-009 §§3–4, 7; ASS-007 §6.

Reducing model size, then streams, then stepping rate is a clear development
rule, but each action changes data quantity, diversity, capacity or temporal
resolution. The plan says frozen evaluation configurations do not change during
a comparison, yet the development process may reach different configurations
for baseline and changed systems as compute pressure varies. The protocols need
to say whether both arms use the same reduced configuration, whether the
reduction itself becomes a tested difference, and how the “10% worse” shrink
tolerance is measured and checked for task-specific regressions.

### 3.10 Medium priority: peer observation is underspecified and may violate
the intended test boundary

**Where:** ADR-006; GOV-008 WP-12 and AB-13; GOV-010 RW-9; RES-009 D-03.

Peer actions may be observed and imitation is not directly rewarded, but
GOV-010 proposes peer outcomes, peer effect keys and a skill library candidate
path. These can become an indirect teacher signal or declared “good action” if
the peer's observed success is used as a label. The documents do not define the
observation geometry (whether peers share a world or separate instances), what
counts as a visible outcome, how peer identities or timing are represented, or
how diversity is measured without using engine state. RES-009 also defers shared
world cooperation, while the social cluster's planned mechanism is across
separate instances.

**Question:** What does a stream actually observe, in what setting, and how will
peer-derived labels be prevented from turning into an implicit declared reward?

### 3.11 Medium priority: removals and “universal” translation need an auditable
decision boundary

**Where:** ADR-004 amendment 1; GOV-008 WP-15; GOV-009 WP-15.

The removal directive gives the assistant discretion to classify each
Minecraft-only item as interface, translated, or removed. “Can be re-derived
from pixels, sound, own-body senses, buttons” is a useful principle, but is not
yet a reproducible test for whether a component is universal. The examples
include shaping and reward paths whose removal or translation could alter
learning, while WP-15 is treated as a no-behavior-change refactor with no A/B.
Configuration parity and a short mechanism mock can catch some regressions, but
may not establish behavioural equivalence or identify which deleted component
caused a difference.

### 3.12 Medium priority: the work package list and map coverage need a
consistency pass

**Where:** GOV-008 §5; GOV-009 §§5, 8; GOV-010 §§5, 7; RES-007; RES-009.

Some map differences are explicitly out of scope or deferred, yet the coverage
table treats every recorded difference as assigned. GOV-009 lists WP-18 and
WP-19 as added by the plan while GOV-008's summary and original package table
describe the scope as owner-directed; the open items ask whether those packages
are accepted. GOV-010 has a provisional cluster order and unresolved choices
about auditing “not established” lines and line-level tests. This makes the
coverage claim stronger than the currently approved scope. Relationship entries
also cite block deferrals in RES-007, so changes to one register can leave
cross-references or classifications stale.

### 3.13 Documentation consistency: stale branch and plan-status statements

**Where:** GOV-008 opening paragraph; GOV-008 §§7, 9; `CLAUDE.md` section 0.

GOV-008's opening says the branch was `Improv` and says no implementation plan
exists yet and that it is the next record. The branch is now `Adjustments`, and
GOV-009 is present and logged later in GOV-008. These statements may be intended
as a historical snapshot, but the header does not mark them as historical. This
creates a misleading live description of the branch and plan state.

**Additional note:** `CLAUDE.md` correctly distinguishes its branch override,
but its line that the work order is “implementation plan first” should be read
alongside the already-existing GOV-009 and its explicit approval/open items, to
avoid confusion about whether planning, approval, or code changes are the next
authorized step.

### 3.14 Lower priority: relationship registry risks turning the map into a
second source of truth

**Where:** GOV-010 §§3, 5–6.

The registry proposes a generated map copy, per-edge switches, producer and
consumer metadata, statistics, status, dead-wire alarms and evaluator-source
tags. This could improve auditability, but a switch for every line and
per-segment metrics may add broad code surface and runtime overhead. Some lines
are descriptive/causal hypotheses rather than literal software data channels,
so representing all as `rel.<FROM>.<TO>` may force artificial wiring. A
conformance test that reports the lines already in the registry may prove
bookkeeping consistency without proving the map-to-code interpretation.

## 4. Questions to resolve before implementation

1. What target game and checkpoint ladder make the main metric executable, and
   what proxy-only results are permitted until then?
2. What is the statistical unit, validated uncertainty method, A/A calibration
   plan, power target, and handling for censoring, crashes and multiple tests?
3. What is the exact arm matrix and development-baseline rule for T1, T2 and unit
   tests, and which runs can actually be shared?
4. What are the operational tests for learned versus designer-declared
   semantics and for evaluator/peer information leakage?
5. Which differences and work packages are approved scope, and what does
   “addressed” mean when the metric cannot be run or the result is null?
6. What evidence is required for promotion, and how are short-horizon mechanism
   wins revisited against later transfer and retention outcomes?
7. Which statements in GOV-008 are current policy and which are historical
   snapshots?

## 5. Review boundary

This report is an issue list for the owner to consider. It does not claim that
the proposed implementation has been inspected or that any listed risk has
occurred. No source code was changed as part of this audit. The only file added
by the audit is this report.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-009](ass-009-audit-response-adjustments-branch.md): Audit response: documentation and plan on the Adjustments branch
<!-- backlinks:end -->
