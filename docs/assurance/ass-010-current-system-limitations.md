# Current system limitations (baseline for the human-like learning program)

> **Document control:** [ASS-010](../README.md#assurance-and-evidence) · Assurance / limitations record · **State:** Record · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Plan reference:** GOV-009 P0-1 ([GOV-009](../governance/gov-009-implementation-plan.md)).
**Status:** a dated statement of what the system **could not do** on the date above, with
the evidence for each limitation. It is the baseline every later result is compared with.
It is written from evidence gathered in the working sessions of 2026-10-17 to 2026-10-21 and
the records ASS-003, ASS-006, RES-006 and RES-008; no code was scanned and no test or run was
made for it. It does not claim any limitation is permanent, and it is not changed when later
work removes one: a later record notes the change and this one stays as the dated baseline.

**Basis codes:** **M** measured on the training host in a run named here; **R** recorded in a
repository record; **D** a standing rule; **U** not established (no claim).

## 1. Learning outcomes

| # | Limitation | Evidence | Basis |
|---|---|---|---|
| L-01 | SkyBot never reached an exit or left the first Doom map in any controlled run | Five runs of 120 minutes, tiny x 8 Doom, 2026-10-18 to 2026-10-20; the evaluator showed one map and zero exits in every run | M ([ASS-003](ass-003-policy-learning.md)) |
| L-02 | A change in play from the credit fix was not established | Behaviour counters (kills, items, deaths) disagreed between two seeds; the behaviour criterion was not met | M (ASS-003) |
| L-03 | The policy stayed close to random over valid actions | Entropy over valid actions 0.87-0.94 of maximum at the end of the runs (0.97-0.98 at the start) | M (ASS-003) |
| L-04 | Chopping a tree in Minecraft has never been learned | Historical scoreboard: 399 blocks broken in host history gave one log, and later runs 13,305 breaks gave 35 logs; the log skills stayed at zero of twenty | R (root `CLAUDE.md` section 9) |
| L-05 | No evidence of adaptability to an unfamiliar game | No cross-game run exists; the target is Minecraft after Doom, deferred | R ([ASS-007](ass-007-adaptability-evaluation-protocol.md)) |
| L-06 | The main metric cannot be run | The target game's return is deferred and the checkpoint ladder is not yet set | R (ASS-007 section 15) |

## 2. What the system does not have (blocks)

The full term-by-term list is [RES-006](../research/res-006-human-like-learning-gap-analysis.md)
section 8. The limitations that most affect learning, each rated by what the system cannot do:

| # | Limitation | Evidence | Basis |
|---|---|---|---|
| L-10 | Cannot create reusable skills in Doom | Goal discovery fired only on an extrinsic reward spike, and Doom pays none; no skill was minted | R, M (run logs: zero options offered) |
| L-11 | Has no knowledge store in use in Doom | The knowledge graph density was 0.00 in the Doom runs | M |
| L-12 | Has no queryable episodic memory | The episodic record showed "never" for sightings and breaks in Doom | M |
| L-13 | Cannot consolidate or protect what it learned | No consolidation phase and no forgetting protection for the policy; forgetting was measured only | R (RES-006) |
| L-14 | Cannot plan at decision time | Acting is a feed-forward policy; imagination trains and shapes but is not searched when choosing an action | R (RES-006) |
| L-15 | Has no attention or long-context mechanism in the policy | Slot attention existed in the world model but was off in every run | R |
| L-16 | Cannot judge its own confidence | No calibration and no uncertainty estimate that drives behaviour | R |
| L-17 | Has no curriculum | Nothing chooses what to practise; one map in every run | M, R |
| L-18 | Cannot learn from peers | The eight streams never observed one another | R |
| L-19 | Cannot adapt its own learning dynamics | Fixed learning-rate, entropy and discount settings; no schedule for the policy | R |
| L-20 | Has no learned danger estimate or risk-sensitive exploration | No risk module | R |
| L-21 | Scaling is advice only | The autoscaler gave advice and one test run actuated a resize; growth of the policy was verified only in unit tests | M, R |

## 3. What the system does not connect (relationships)

[RES-008](../research/res-008-relationship-gap-analysis.md): of 326 lines compared with the
map, 36 were connected, 44 partial, 106 missing because a block is absent, 26 deliberately
excluded and 114 not established. The control and selection ring (attention, consolidation,
executive control, metacognition, curriculum, strategy selection) was missing. Basis: R.

## 4. Reward and signal limitations

| # | Limitation | Evidence | Basis |
|---|---|---|---|
| L-30 | One income term dominated | Episodic novelty supplied 72-89 % of income in the runs | M |
| L-31 | The learning-progress term paid almost nothing before the action-credit fix | About 0.0003 per step; action attribution 0.0076 | M ([ASS-006](ass-006-realtime-action-lag-incident.md)) |
| L-32 | The intrinsic weight decayed toward an extrinsic reward Doom never pays | Fell from 0.66 to 0.53 within two hours in the unfixed arm | M (ASS-003) |
| L-33 | No information-gain, competence or self-model signal competes with novelty | No such term existed | R |

## 5. Operational limitations

| # | Limitation | Evidence | Basis |
|---|---|---|---|
| L-40 | Runs of the universal brain were at most two hours | The longest controlled run was 120 minutes | M |
| L-41 | One game at a time on the host | Peak process memory about 11 GB of 16 GB; arms ran serially | M |
| L-42 | Seeds are not bit-reproducible | Run-to-run noise had to be estimated, not assumed | R (ASS-007) |
| L-43 | Recovery from host loss is manual | The host was unreachable for about eleven hours on 2026-10-19; the cause was not established | M |
| L-44 | Minecraft-only code and a legacy launch path remained in the tree | Recorded in ADR-004; removal is WP-15 | R |

## 6. What this record does not say

- It does not say a limitation can or cannot be removed; that is the program's work.
- Rows coded U in RES-006 are not listed as limitations until verified.
- Values are those of the named runs on the named configuration; they are not a general
  measure of the architecture.

## 7. Related records

[GOV-009](../governance/gov-009-implementation-plan.md),
[GOV-008](../governance/gov-008-human-like-learning-motivation.md),
[RES-006](../research/res-006-human-like-learning-gap-analysis.md),
[RES-008](../research/res-008-relationship-gap-analysis.md),
[ASS-003](ass-003-policy-learning.md), [ASS-006](ass-006-realtime-action-lag-incident.md),
[ASS-007](ass-007-adaptability-evaluation-protocol.md).
