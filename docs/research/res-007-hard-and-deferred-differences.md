# Hard and deferred differences: what cannot yet be built reliably

> **Document control:** [RES-007](../README.md#research) · Research / deferred-work register · **State:** Proposed · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Parent:** [GOV-009](../governance/gov-009-implementation-plan.md)
and [GOV-008](../governance/gov-008-human-like-learning-motivation.md).
**Status:** a register of differences from the human-learning map
([`ai_learning.mmd`](../../assets/ai_learning.mmd), compared in
[RES-006](res-006-human-like-learning-gap-analysis.md)) that the implementation plan
does not build now, with the reason, what an interim approximation would be, and
what would let us revisit each. The owner asked for these to be recorded so the
system can be built for them later. Nothing here is a claim that the item is
impossible; each entry says what is missing.

## 1. How to read this register

Each entry gives: the **difference**; **why it is not built now** (one of: no
established method, evidence only at a different scale, needs a capability the
host or the project lacks, conflicts with a standing rule, or cannot be shown in a
one-hour seed); an **interim approximation** if one is safe; and a **revisit
trigger**. The classes are:

- **K** missing knowledge: no reliable method is known for this setting.
- **S** scale or horizon: it needs more than an hour, more compute or more tasks than
  the standard allows.
- **F** facility: it needs a game mode, hardware or a second game that is not
  available.
- **R** rule conflict: it conflicts with learned-not-declared, one universal brain
  or the other standing rules, so a faithful version cannot be built.

## 2. The register

| # | Difference | Class | Why not now | Interim approximation | Revisit when |
|---|---|---|---|---|---|
| H-01 | Critical or sensitive learning periods (engineered) | K | Hensch-style windows are observed in animals; in deep networks they appear as an emergent property (Achille et al.), and no reliable recipe exists for *engineering* beneficial ones | WP-9 plasticity maintenance and schedules | There is evidence from WP-9 that early-training changes alter final adaptability |
| H-02 | Pruning and sparsity of the policy | K, S | Synaptic pruning is well described; benefit for a small online policy is unproven and a one-hour seed cannot show it | Report parameter and activation sparsity as measurements | Model growth (WP-10) is live and has a long-run protocol |
| H-03 | Excitatory/inhibitory weight split | K | The map's "positive and negative weights" has no established machine counterpart with a measured benefit | None | A concrete architecture with evidence appears |
| H-04 | Activity-dependent retrograde plasticity tuning (the endocannabinoid link) | K | No established machine analogue | Per-parameter learning-rate adaptation via the optimiser | A published, tested analogue exists |
| H-05 | Social bonding signals (the oxytocin link) | K, R | No machine analogue; a bonding reward would be declared, not learned | WP-12 observation only | Never as a reward; revisit only for a learned analogue |
| H-06 | Relatedness and belonging as need-satisfaction | K, R | Same; the streams share one brain, so "belonging" is not well defined | WP-12 peer observation; diversity measure | A defensible, learned operational definition exists |
| H-07 | Wellbeing | K | No measurable machine analogue beyond learning-progress and stability | Report stability and progress together | A measurable definition is agreed |
| H-08 | Model editing | K | Editing a policy or world model in place without retraining is research-stage, and gives the agent no new knowledge it did not learn | Fine-tune through ordinary learning | Editing methods mature and a use case arises |
| H-09 | Few-shot analogy | K | No reliable method for analogical transfer between games with an unsupervised world model | WP-19 invariant features and a transfer probe | A method with a demonstrated win on a related benchmark |
| H-10 | Full causal structure discovery | K, S | Agency detection (WP-1) is a start; learning a causal graph of the game from interaction is unsolved at this scale | WP-1 effect keys and attribution | WP-1 results show key purity high enough to build on |
| H-11 | Cooperation in a shared world | F | Needs several agents in one game world (a multiplayer engine mode) and a cooperative task; the streams currently run separate instances | WP-12 observation of other instances | A multiplayer mode is adopted and onboarded (WP-16) |
| H-12 | Acting-time decision search at eight streams | S | The Doom loop already uses most of its decision interval; search inside it does not fit | WP-4 planning in the learning phase and during standby | Latency headroom or fewer streams (ADR-005) permit it |
| H-13 | Meta-learning proper (gradient-based or recurrent) | K, S | Needs many tasks; there is one game and one-hour seeds | WP-13 online hyperparameter adaptation | At least three onboarded games and a long-run protocol |
| H-14 | Population-based training across streams | R | It needs separate learners; the streams are one brain | None | Never under ADR-001 |
| H-15 | Federated learning | R | Moot for one shared brain | None | Never under ADR-001 |
| H-16 | True sleep-stage structure and hippocampal-cortical mapping | K | The biology is not specified well enough to copy; WP-2 builds the function (replay, consolidation), not the anatomy | WP-2 | Evidence from WP-2 on which components mattered |
| H-17 | Benefits that need more than one hour (growth, consolidation, curriculum, plasticity) | S | The standard judges a package on one-hour seeds | Separate long-run protocols | A long-run protocol is frozen |
| H-18 | Minecraft as the transfer target | F | Deferred by the owner | An interim proxy (labelled), if chosen at Gate G1 | Minecraft returns to the universal brain |
| H-19 | Adaptive sensor resolution and active perception | K, F | Foveation exists in the senses, but learned, resource-aware resolution control is not planned | Fixed resolution | After attention (WP-7) results |
| H-20 | Embodied robotics | F | Out of scope for games | None | A physical platform exists |
| H-21 | Language-model reasoning, chain-of-thought and self-reflection in language | R | Language is outside the learning loop by the owner's rules | Non-linguistic self-evaluation (WP-8) | The owner reopens language |
| H-22 | Human feedback, demonstrations, imitation of a teacher | R | Deliberate: learned, not declared | Peer observation (ADR-006) | The owner reopens it by decision record |
| H-23 | Graph neural networks for knowledge | K | Disabled in all measured runs; unproven | Removal or replacement by WP-18 | After WP-18 |
| H-24 | A single, principled "objective function" | K | The map implies one; SkyBot uses a mixture whose weights are hand-set | WP-5 and WP-8 make the mixture more principled | A learned or derived mixture has evidence |
| H-25 | Reproducing the 5.13 DNE items as built features | K, R | See H-04 to H-07 | None | As above |

## 3. Reading rules

- An entry is not a promise to build it. A **revisit trigger** is a condition under
  which it becomes worth re-opening; the owner decides.
- A deferral is not a negative result. A negative result is a tested change that did
  not help and is recorded in its own result record.
- When an entry is revisited, add a dated note under it. Do not rewrite it.

## 4. Related records

[GOV-009](../governance/gov-009-implementation-plan.md),
[GOV-008](../governance/gov-008-human-like-learning-motivation.md),
[RES-006](res-006-human-like-learning-gap-analysis.md),
[RES-005](res-005-skill-discovery.md),
[ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md),
[ADR-001](../decisions/adr-001-shared-universal-brain.md),
[ADR-005](../decisions/adr-005-compute-pressure-order.md),
[ADR-006](../decisions/adr-006-streams-may-observe-one-another.md).

## Cross-reference maintenance (2026-10-21)

RES-007 (blocks) and RES-009 (lines) cite each other's entries. When a deferral in either
changes, update the entries in the other that cite it, in the same change, and note the
date. RES-006 and RES-008 are revised in the same change when a block's status moves.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [GOV-010](../governance/gov-010-relational-implementation-plan.md): Relational implementation plan: connecting the learning blocks as the map does
- [RES-009](res-009-hard-and-deferred-relationships.md): Hard and deferred relationships: lines that cannot yet be built reliably
<!-- backlinks:end -->
