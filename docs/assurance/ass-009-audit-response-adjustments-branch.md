# Audit response: documentation and plan on the Adjustments branch

> **Document control:** [ASS-009](../README.md#assurance-and-evidence) · Assurance / audit response record · **State:** Record · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Responds to:** [ASS-008](ass-008-audit-adjustments-branch.md)
(a review of the branch's documentation diff). **Scope:** documentation only; no code was
changed. The audit lists issues and questions; this record states, for each finding, whether
it was accepted, what was changed and where, and what the owner still has to decide. The
audit record (ASS-008) has its body unchanged; only its header was brought into the register's format.

## 1. Overall view

The audit is largely right. Its central point is that the plans were not yet operationally
ready: the primary metric cannot run, the statistics were too thin for the claims, the
sharing of runs was wrong in the two-new-blocks case, "learned, not declared" lacked a
review test, and several status statements had gone stale. None of the findings says a
mechanism cannot work; they are plan-quality issues. All findings were accepted in whole or
in part. Where a fix is a design decision that belongs to the owner, it is listed in
section 3 and not decided here.

## 2. Disposition of each finding

| Finding | Disposition | What changed (documents only) |
|---|---|---|
| 3.1 Primary metric cannot yet be run | **Accepted.** Real: target game, checkpoints and detectors are undecided. | ASS-007 14.1 separates **development evidence** from **program evidence** and states what may proceed meanwhile; GOV-008 2.5 |
| 3.2 Statistical definition | **Accepted.** The bootstrap unit, A/A size and the all-five calculation were weak. | ASS-007 14.2 to 14.7: seed as the independent unit; moving-block bootstrap; ten pooled A/A seeds and an empirical all-five rate; power statement; checkpoint edge cases; crash rule; multiplicity. The 40/20 split stays an assistant default with a sensitivity pilot required (needs owner) |
| 3.3 Promotion vs objective | **Accepted.** | ASS-007 14.8: provisional versus confirmed promotion, a no-regression guard, and a symmetric reversal rule using leave-one-out |
| 3.4 T1/T2 ordering and run sharing | **Accepted; the sharing claim was wrong** when both blocks are new. | GOV-010 4B: definition of "exists", sequential development baseline, an arm matrix, corrected sharing, retention of conditional results, corrected cost (250 to 330 hours plus non-run costs) |
| 3.5 Learned, not declared needs operational boundaries | **Accepted.** | GOV-009 17.3: a five-question review record in every research design, and a proxy-exploitation test |
| 3.6 Map is a hypothesis source | **Accepted.** The registry could be mistaken for validation. | GOV-010 section 3 and RES-008 addendum: coverage is not function; typed channels; descriptive lines are not wired |
| 3.7 "One at a time" vs layered graph | **Accepted as a wording problem.** | GOV-010 4B and GOV-008 2.5: three distinct meanings |
| 3.8 Gates and non-run costs | **Accepted.** | GOV-009 17.1 (which gates are mandatory) and 17.2 (non-run cost assumptions, storage) |
| 3.9 Compute order may change the treatment | **Accepted.** | ADR-005 Amendment 2: both arms run the same configuration; the reduction is recorded as its own difference; how the tolerance is measured |
| 3.10 Peer observation underspecified | **Accepted.** | ADR-006 Amendment 2: separate instances, what is and is not observed, peer effects as hypotheses verified by probation, diversity measure, leakage tests; GOV-010 RW-9 adjusted |
| 3.11 Removal boundary not auditable | **Accepted.** WP-15 is not all behaviour-neutral. | ADR-004 Amendment 2: a three-check test for "universal"; WP-15 split into 15a and 15b; attribution by small steps |
| 3.12 Coverage and scope consistency | **Accepted.** | GOV-009 17.5 (the coverage table is a mapping, not an approval); GOV-008 2.5 scope approval status and the definition of "addressed"; cross-reference maintenance notes in RES-007 and RES-009 |
| 3.13 Stale branch and plan statements | **Accepted; verified:** the branch is `Adjustments`. | GOV-008 header and section 8; root `CLAUDE.md` bullet now says the plans exist and await approval, and no code changes before approval |
| 3.14 Registry as a second source of truth | **Accepted in part.** | Covered with 3.6; overhead limits kept (per-segment statistics, switches only on data and control channels) |

## 3. Needs the owner

1. The **target game and checkpoint ladder** (also the interim proxy, if any).
2. Whether **ten A/A seeds** (ten extra hours) are acceptable, and whether **40/20** is a
   sound budget (a pilot is proposed before freezing).
3. Whether **provisional versus confirmed promotion** and the **reversal rule** (ASS-007 14.8)
   match the owner's intent. They are the assistant's judgement.
4. **Approval of the work packages** marked as proposals (GOV-008 2.5), and of the relation
   clusters.
5. The meaning of **"addressed"** in the merge gate as defined in GOV-008 2.5.

## 4. Not changed, and why

- The audit's body is left as written; it is a review record. Its header was changed to the register's control-line format and the file moved to `docs/assurance/` as ASS-008.
- Statements about code were not re-verified (the audit did not verify them either); the
  plans still require a "verify before edit" step.

## 5. Related records

[GOV-008](../governance/gov-008-human-like-learning-motivation.md),
[GOV-009](../governance/gov-009-implementation-plan.md),
[GOV-010](../governance/gov-010-relational-implementation-plan.md),
[ASS-007](ass-007-adaptability-evaluation-protocol.md),
[ASS-008](ass-008-audit-adjustments-branch.md),
[ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md),
[ADR-006](../decisions/adr-006-streams-may-observe-one-another.md),
[RES-008](../research/res-008-relationship-gap-analysis.md),
[RES-009](../research/res-009-hard-and-deferred-relationships.md).

## 6. Owner decisions (2026-10-21) on section 3

1. **Target game:** Minecraft. **Checkpoint ladder:** progress through the game combined
   with survivability and thrivability, to be set later. No interim proxy for now.
2. **Ten A/A seeds and the 40/20 split (with a sensitivity pilot before freezing):** accepted.
3. **Provisional versus confirmed promotion and the reversal rule:** approved.
4. **Work packages and relation clusters:** approved as scope.
5. **"Addressed" in the merge gate:** approved as defined.

Recorded in GOV-008 2.6, ASS-007 15, GOV-009 18. The only item still open from section 3 is
the final checkpoint ladder. Approving scope does not authorise code changes; the owner's
first instruction does.
