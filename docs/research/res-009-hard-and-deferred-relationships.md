# Hard and deferred relationships: lines that cannot yet be built reliably

> **Document control:** [RES-009](../README.md#research) · Research / deferred-work register · **State:** Proposed · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Companion to:** [RES-007](res-007-hard-and-deferred-differences.md),
which records *blocks* that cannot be built reliably. This record does the same for
*lines* ([RES-008](res-008-relationship-gap-analysis.md)). **Parent:**
[GOV-010](../governance/gov-010-relational-implementation-plan.md).

## 1. How to read this register

Each entry names the lines (by RES-008 identifier), why they are not built now,
an interim approximation if one is safe, and a trigger to revisit. The classes are
those of RES-007: **K** missing knowledge, **S** scale or horizon, **F** facility,
**R** rule conflict. An entry is not a promise to build it and not a negative
result; a deferral is recorded so the system can be built for it later.

## 2. The register

| # | Subject | Lines | Class | Why not now | Interim approximation | Revisit when |
|---|---|---|---|---|---|---|
| D-01 | Excitatory and inhibitory weight lines and the receptor line | NM-10, NM-11, NM-12, NS-08 | K | RES-007 H-03: no machine counterpart with a measured benefit | Per-parameter learning-rate adaptation by the optimiser | A concrete architecture with evidence appears |
| D-02 | Retrograde plasticity tuning (endocannabinoid link) | NM-15 | K | RES-007 H-04: no established analogue | Optimiser-side adaptation | A published, tested analogue exists |
| D-03 | Bonding, belonging and relatedness lines | NM-13, NM-14, SO-06, SO-18, SO-05 | K, R | RES-007 H-05, H-06: no learned analogue; a bonding reward would be declared | Peer observation (RW-9) with a diversity measure | A defensible learned definition exists; never as a declared reward |
| D-04 | Wellbeing lines | OU-17, OU-19, LC-08 | K | RES-007 H-07: no measurable machine analogue | Report stability and progress together | A measurable definition is agreed |
| D-05 | Engineered critical-period lines | ID-17, NS-06, NS-03 | K | RES-007 H-01: no recipe for engineering beneficial windows | Stage variable from learning progress (RW-12) as measurement | WP-9 results show early-training changes matter |
| D-06 | Pruning lines | NS-05, NS-12, NS-21 | K, S | RES-007 H-02: benefit unproven; cannot be shown in one hour | Report sparsity as a measurement | Growth (WP-10) is live with a long-run protocol |
| D-07 | Analogical transfer lines | CP-26, CP-27 | K | RES-007 H-09: no reliable method for unsupervised analogical mapping | Invariant features and a transfer probe (RW-8) | A method with a demonstrated win on a related benchmark |
| D-08 | Sleep-to-creativity line | PH-06 | K | A hypothesis in neuroscience; no operational machine test | None | A measurable creativity metric and a mechanism are proposed |
| D-09 | Mirror-system lines | SY-33, SY-34 | K | The function of the mirror system is contested; the machine analogue is speculative | RW-9 peer observation with an auxiliary prediction task | Evidence from WP-12 that action prediction helps |
| D-10 | Teacher, human-feedback and language lines | SO-01, SO-02, SO-03, ST-12, ST-13, ST-16, SO-07, SO-13, ME-18, ME-10, ME-14, ST-09, ST-31 | R | RES-007 H-21, H-22: excluded by learned-not-declared and the no-language rule | Learner-side self-evaluation (RW-6, RW-13) | The owner reopens the rule by decision record |
| D-11 | Identity and culture lines | MO-28, SO-14, SO-15, SO-29 | K, R | No learned analogue; any version would declare identity or goals | Goals derived from effects (RW-7) | Never as declared; revisit only for a learned analogue |
| D-12 | Prompt and context phenomena | SO-24, SO-28, SO-22, SO-21, MS-12, ID-22 | R | They describe language-prompt behaviour; this system has no prompt | None | If a language component is ever added |
| D-13 | Text-only memory aids | ST-32, ST-37 | R | Scratchpad notes and summarisation are linguistic | None | If language is reopened |
| D-14 | Architecture-diversity lines | ID-13, ID-14, ID-15 | K, S | SkyBot is one architecture; a diversity of architectures would need parallel learners, which conflicts with one brain | Ensemble heads inside one brain (WP-4) | A decision to run alternative architectures |
| D-15 | Embodiment line | PH-15 | F | RES-007 H-20: no physical platform | Simulated body senses | A physical platform exists |
| D-16 | Full causal-structure lines | ME-25, ME-17 | K, S | RES-007 H-10: agency detection (WP-1) is a start; structure learning is unsolved at this scale | WP-1 effect keys and attribution | WP-1 key purity is high enough to build on |
| D-17 | Federated-learning line | ST-29 | R | RES-007 H-15: moot for one shared brain | None | Never under ADR-001 |
| D-18 | Lines that need more than an hour to show an effect | ID-18, NS-16, NS-17, NS-01, NS-11 | S | RES-007 H-17: the one-hour standard cannot judge them | Long-run protocols (RW-12, RW-14) | A long-run protocol is frozen |
| D-19 | Cross-environment abstraction, skill reuse and novelty-driven adaptation | CP-28, CP-32, OU-06, OU-07, OU-11, OU-12, OU-16, ST-06, ST-27, ME-21 | K, F | WP-19 supplies representation losses and an offline probe, but no learner-side measure of cross-game novelty or demonstrated path from representation/skills into policy adaptation; held-out outcomes remain evaluator-only | Record WP-19 loss signals (RW-8); keep transfer probes evaluation-only | A target game and a learner-side transfer mechanism with explicit inputs, ownership, bounds and kill criteria are defined |
| D-20 | Skill-directed practice, skill-policy distillation and learned cross-game priors | OU-03, OU-04, ST-15 | K, F | The skill ledger has only a trailing mastery window; the curriculum has no learner-skill consumer; dream distillation has no skill-policy teacher; current warm-start copies shape-dependent policies. A behavior change now would invent decline evidence or cross incompatible policies | RW-11 Stage A persists non-overlapping outcome windows and reports read-only trend/binding diagnostics; no behavior change | Stage A persistence and measurement are verified, then each Stage B consumer has an explicit interface, compatibility guard, switch, and T1/T2 protocol |
| D-21 | Peer-derived capability hypotheses | ME-12 | K, F | WP-12 hypotheses contain an action and normalised view change, not a stable effect key or an observer-owned WP-1 probation candidate; peer outcomes cannot be treated as verified skill evidence | Keep hypotheses in the peer board as unconsumed telemetry; expose no skill candidate from them | Define an effect-identity contract and an observer-owned probation route that excludes peer reward, success and engine state |
| D-22 | Risk penalties, risk-to-attention and environment-derived safe-exploration links | SY-15, SY-17, ME-03, SO-16, SO-17, PH-13, SO-19, MO-18, MO-19, MO-25 | K, R | RW-10's requested attention link is absent from its map-line scope; risk penalties would add a reward/penalty path outside WP-11's input-only contract; the remaining lines lack a bounded, learner-side producer/consumer definition | Keep the existing bounded SY-16 and SO-30 controls; leave rewind/stuck handling in its current infra path | A map-consistent line and a specific learner-side signal, owner, bound, neutral behavior and release condition are defined |
| D-23 | Neurobiological plasticity and growth mechanisms without machine counterparts | NS-01, NS-02, NS-04, NS-07, NS-08, NS-11, NS-18, NS-19, NS-20 | K | WP-9 supplies optimiser schedules and optional function-preserving unit refresh; WP-10 supplies policy growth, but neither supplies neurotrophin, receptor, myelin, glial or synaptic mechanisms. Treating those analogies as implemented would overstate the code | Report existing optimizer, refresh and policy-growth controls separately; keep critical periods, pruning, and long-horizon items in D-05, D-06 and D-18 | A machine mechanism with an explicit state variable, bounded effect, function-preservation rule, and long-run evidence is specified |
| D-24 | Resource-driven context/planning changes and early-data-quality links | SO-09, SO-10, SO-26, PH-11, ID-07, CP-44 | K, F | WP-10 measures VRAM and acting latency but only changes model size, stream count and stepping rate; attention window and planner horizon have no safe live-resize contract; no early-data-quality scalar is defined | Keep the ADR-005 autoscaler as implemented; expose the fixed WP-7 context width via ID-06; leave context and planner settings fixed | Define a measured resource/quality signal, safe resize/state migration, latency bound and release behavior for each consumer |

## 3. The lines, by name

- **D-01** — NM-10: Positive weights → Activation functions (ReLU, GELU); NM-11: Negative weights → Critical learning periods; NM-12: Negative weights → Attention mechanism; NS-08: Activation functions (ReLU, GELU) → Weight updates (gradient descent)
- **D-02** — NM-15: Endocannabinoids (DNE) → Weight updates (gradient descent)
- **D-03** — NM-13: Oxytocin (DNE) → Base model lineage; NM-14: Oxytocin (DNE) → Multi-agent learning; SO-06: Base model lineage → Oxytocin (DNE); SO-18: Belonging (DNE) → Objective function; SO-05: Peer agents → Belonging (DNE)
- **D-04** — OU-17: Intrinsic reward → Wellbeing (DNE) and continual learning; OU-19: Policy updates → Wellbeing (DNE) and continual learning; LC-08: MACHINE LEARNING durable change in parameters, capability or policy → Wellbeing (DNE) and continual learning
- **D-05** — ID-17: Training stage (epoch) → Critical learning periods; NS-06: Critical learning periods → Frozen layers; NS-03: Pretraining → Critical learning periods
- **D-06** — NS-05: Pretraining → Pruning and sparsity; NS-12: Weight updates (gradient descent) → Pruning and sparsity; NS-21: Pruning and sparsity → Parametric memory
- **D-07** — CP-26: Hierarchical feature abstraction → Few-shot analogy; CP-27: Few-shot analogy → Transfer learning
- **D-08** — PH-06: Generative replay → Generative diversity
- **D-09** — SY-33: Behavior cloning network → Multi-agent learning; SY-34: Behavior cloning network → Imitation learning
- **D-10** — SO-01: Teacher model quality → Curriculum learning; SO-02: Teacher model quality → Feedback signals (RLHF); SO-03: Teacher model quality → Prompt framing effects; ST-12: Feedback signals (RLHF) → Self-evaluation; ST-13: Feedback signals (RLHF) → Model calibration; ST-16: Targeted fine-tuning → Feedback signals (RLHF); SO-07: Base model lineage → Training corpus; SO-13: Training corpus → Language modeling; ME-18: Language modeling → Knowledge graph; ME-10: Empirical risk minimization → Language modeling; ME-14: Imitation learning → Language modeling; ST-09: Self-generated rationales → Generalization; ST-31: Self-reflection and critique → Meta-learning
- **D-11** — MO-28: Persona and system prompt → Objective function; SO-14: Data distribution norms → Goal-conditioned policies; SO-15: Data distribution norms → Persona and system prompt; SO-29: Open-source community → Persona and system prompt
- **D-12** — SO-24: Prompt context dependence → Retrieval-augmented generation; SO-28: Prompt framing effects → Self-evaluation; SO-22: Prompt injection and noise → Attention mechanism; SO-21: Frameworks and tooling → Prompt injection and noise; MS-12: Implicit parametric knowledge → Prompt priming; ID-22: Learning styles MYTH (DNE) → Teacher-student training
- **D-13** — ST-32: Scratchpad notes → Input encoding and embedding; ST-37: Context summarization → Data augmentation
- **D-14** — ID-13: Architecture diversity → Attention mechanism; ID-14: Architecture diversity → Planning and tool use; ID-15: Architecture diversity → Feature extraction
- **D-15** — PH-15: Embodied robotics → Action-conditioned learning
- **D-16** — ME-25: Causal discovery → Representation learning; ME-17: Exploration → Causal discovery
- **D-17** — ST-29: Federated learning → Multi-agent learning
- **D-18** — ID-18: Training stage (epoch) → Planning and tool use; NS-16: Adaptive optimizers (Adam, momentum) → Dynamic network expansion; NS-17: Dynamic network expansion → Contrastive learning; NS-01: Weights → Pretraining; NS-11: Learning rate scheduling → Weight updates (gradient descent)
- **D-20** — OU-03: Procedural learning → Skill library; OU-04: Amortized inference → Skill library; ST-15: Targeted fine-tuning → Capability development
- **D-21** — ME-12: Multi-agent learning → Capability
- **D-22** — SY-15: Safety and risk module → Reward shaping; SY-17: Safety and risk module → Risk penalties; ME-03: Supervised learning → Safety and risk module; SO-16: Training environment → Safe exploration; SO-17: Safe exploration → Distribution shift degradation; PH-13: Training instability and noise → Distribution shift degradation; SO-19: Bias-induced degradation → Distribution shift degradation; MO-18: Distribution shift degradation → Context window; MO-19: Distribution shift degradation → Computational load; MO-25: Risk penalties → Safety and risk module
- **D-23** — NS-01: Genes → BDNF; NS-02: Development → Critical learning periods; NS-04: Development → Dendritic spine formation; NS-07: NMDA receptors → Long-term potentiation; NS-08: Long-term potentiation → Hebbian learning; NS-11: Long-term potentiation → Dendritic spine formation; NS-18: Glial cells → Myelination; NS-19: Myelination → Processing speed; NS-20: Dendritic spine formation → Long-term memory
- **D-24** — SO-09: Compute budget → Low-quality early data; SO-10: Low-quality early data → Training instability and noise; SO-26: Real-world data diversity → Training instability and noise; PH-11: Training instability and noise → Context window; ID-07: Model scale → Chain-of-thought reasoning; CP-44: Throughput → Context window

## 4. Reading rules

- A line listed here is not built until its revisit condition is met and the owner
  agrees. Lines whose blocks are deliberately excluded by a standing rule are
  recorded for completeness and stay excluded.
- A deferral is not a finding about the line. Findings come from tested changes and
  are recorded in result records.
- When an entry is revisited, add a dated note beneath it; do not rewrite it.

## 5. Related records

[GOV-010](../governance/gov-010-relational-implementation-plan.md),
[RES-008](res-008-relationship-gap-analysis.md),
[RES-007](res-007-hard-and-deferred-differences.md),
[RES-006](res-006-human-like-learning-gap-analysis.md),
[GOV-009](../governance/gov-009-implementation-plan.md),
[ADR-001](../decisions/adr-001-shared-universal-brain.md),
[ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md),
[`ai_learning.mmd`](../../assets/ai_learning.mmd).

## Cross-reference maintenance (2026-10-21)

RES-007 (blocks) and RES-009 (lines) cite each other's entries. When a deferral in either
changes, update the entries in the other that cite it, in the same change, and note the
date. RES-006 and RES-008 are revised in the same change when a block's status moves.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-006](../architecture/arc-006-relation-registry.md): Relation registry
- [ASS-009](../assurance/ass-009-audit-response-adjustments-branch.md): Audit response: documentation and plan on the Adjustments branch
- [GOV-008](../governance/gov-008-human-like-learning-motivation.md): Human-like learning program: motivation, scope and change record
- [GOV-012](../governance/gov-012-execution-record-gov-010.md): Execution record: relational implementation plan (GOV-010)
<!-- backlinks:end -->
