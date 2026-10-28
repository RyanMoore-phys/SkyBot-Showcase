# Relationship gap analysis: SkyBot's learning connections against the AI-counterpart map

> **Document control:** [RES-008](../README.md#research) · Research / relationship gap analysis · **State:** Record · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-21. **Companion to:** [RES-006](res-006-human-like-learning-gap-analysis.md),
which compared the *blocks* (terms) of [`ai_learning.mmd`](../../assets/ai_learning.mmd)
with SkyBot. This record compares the **relationships** (the lines): which
information flows the ideal learner has between its blocks, and which exist in
SkyBot. It is the factual basis for [GOV-010](../governance/gov-010-relational-implementation-plan.md).

## 1. Purpose and scope

The owner observed that the block-level report does not say how the blocks are
meant to be connected. In the map the lines carry the logic of learning: what
feeds what. A block that exists but is not connected as the map shows cannot
behave as it does in the ideal learner. This record lists every line in the map
and says whether SkyBot has it.

**Excluded, by the owner's instruction:** every line that touches a node the map
marks N/A (Cortisol, Aerobic exercise, Hydration, Circadian rhythm, Inflammation
and illness, Aging, Hormones, Caffeine/alcohol/nicotine/drugs, Mental health):
20 lines. Lines touching DNE nodes are **included**, as in RES-006.

**Method.** The lines were read from the map file itself (346 lines, grouped as
the file groups them); 326 are compared. The status of a line is derived from the
status of its two blocks in RES-006 plus the author's knowledge of specific flows
from the working sessions of 2026-10-17 to 2026-10-21. No code was scanned.

## 2. Status definitions

| Status | Meaning | Certainty |
|---|---|---|
| **Connected** | A working flow of the kind the line describes exists. | Observed or read in the working sessions |
| **Partial** | A flow exists but covers part of the idea, is weak, is advice only, or is off or empty in the measured game. | As above |
| **Missing (an end is absent)** | The line cannot exist because one of its blocks has no counterpart in SkyBot (RES-006). This is derived mechanically and is certain given RES-006. | Certain given RES-006 |
| **Deliberate** | An end is excluded by a standing owner rule (pretraining, language, human feedback, extrinsic reward in Doom, and similar). | Owner rule |
| **Not established** | Both blocks exist or are unverified, but the session gave no basis to say whether the flow exists. **No claim is made.** These are verification items, not findings. | None |

"Not established (no known flow)" is the subset where both blocks are present but
no flow between them is known; it must **not** be read as "absent".

## 3. Summary

| Status | Lines |
|---|---|
| Connected | 36 |
| Partial | 44 |
| Missing (an end is absent) | 106 |
| Deliberate (an end is excluded by rule) | 26 |
| Not established | 51 |
| Not established (no known flow) | 63 |
| **Compared** | **326** |

By group of the map:

| Group | Connected | Partial | Missing | Deliberate | Not established | Total |
|---|---|---|---|---|---|---|
| Model parameters and architecture (section 1 nodes) | 4 | 0 | 2 | 4 | 11 | 21 |
| Learning-signal modulators (section 2) | 2 | 1 | 9 | 0 | 5 | 17 |
| Model components and networks (section 3) | 7 | 6 | 12 | 1 | 8 | 34 |
| Hardware and infrastructure (section 4) | 2 | 2 | 6 | 0 | 7 | 17 |
| Computational processes (section 5) | 2 | 7 | 17 | 0 | 18 | 44 |
| Memory and storage (section 6) | 4 | 5 | 1 | 0 | 3 | 13 |
| Learning algorithms (section 7) | 9 | 4 | 4 | 3 | 5 | 25 |
| Objectives and reward signals (section 8) | 4 | 5 | 10 | 1 | 11 | 31 |
| Training strategies (section 9) | 0 | 4 | 20 | 6 | 11 | 41 |
| Data and environment (section 10) | 0 | 3 | 10 | 5 | 12 | 30 |
| Model differences (section 11) | 1 | 4 | 6 | 5 | 7 | 23 |
| Capabilities (section 12) | 1 | 3 | 6 | 0 | 9 | 19 |
| Links to and from the central node | 0 | 0 | 3 | 1 | 7 | 11 |

## 4. What the pattern shows

1. **The reward-to-update core is connected.** The 36 connected lines sit almost
   entirely in one loop: reward signal, temporal-difference error, weight update,
   policy, habit and behaviour; observation, encoding, prediction error and
   representation. SkyBot learns from reward and prediction error.
2. **The control and selection ring is missing.** The lines that decide *what to
   attend to, what to consolidate, what to practise and which strategy to use*
   are mostly missing because their blocks (attention, consolidation, executive
   control, metacognition, curriculum, strategy selection) do not exist (RES-006).
   A block added by [GOV-009](../governance/gov-009-implementation-plan.md) is
   useless until it is connected, which is the reason for GOV-010.
3. **Hub blocks.** Counting missing lines by block, the ones whose absence breaks
   the most lines are:

| Block | Missing lines it takes down |
|---|---|
| Attention mechanism [ATT] | 24 |
| Weight consolidation (EWC) [CONS] | 11 |
| Reasoning controller [PFC] | 6 |
| Multi-agent learning [OBS] | 5 |
| Training strategy selection [STRAT_ENTRY] | 5 |
| Capability [SKILL] | 5 |
| Input encoding and embedding [ENC] | 4 |
| Meta-learning [META] | 4 |
| Specialist model [EXPT] | 4 |
| Oxytocin (DNE) [OXY] | 3 |
| Context window [WM] | 3 |
| Safety and risk module [AMY] | 3 |
| Sparse and long-context attention [SELATT] | 3 |
| Model calibration [CALIB] | 3 |

   The **attention** block alone accounts for about a quarter of all missing lines.
   The **consolidation** block is second. Both are single blocks that, once built,
   unlock many lines at once.
4. **Many lines are simply not established.** About a third of the lines are in
   this class. They must be checked (GOV-010 phase R0) before anyone assumes they
   are missing or present.
5. **A line is only as good as its flow.** Several Partial lines are partial because
   the flow is empty in Doom (for example the knowledge store and episodic record)
   or advice-only (the autoscaler).

## 5. The lines

Identifiers are the group abbreviation and a number in the order the lines appear
in the map file. A solid line is a data or control flow; a dotted line is a
modulating or "feeds back" relation. Block names are the map's, with the node id
in brackets for lookup.

### NS. Model parameters and architecture (section 1 nodes)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| NS-01 | Weights [GEN] | Pretraining [DEV] | solid | Deliberate (an end is excluded by rule) | D |
| NS-02 | Weights [GEN] | Adaptive optimizers (Adam, momentum) [BDNF] | solid | Not established (no known flow) | U |
| NS-03 | Pretraining [DEV] | Critical learning periods [CRIT] | solid | Deliberate (an end is excluded by rule) | D |
| NS-04 | Pretraining [DEV] | Inference optimization [MYEL] | solid | Deliberate (an end is excluded by rule) | D |
| NS-05 | Pretraining [DEV] | Pruning and sparsity [SPINE] | solid | Deliberate (an end is excluded by rule) | D |
| NS-06 | Critical learning periods [CRIT] | Frozen layers [PNN] | solid | Not established | U |
| NS-07 | Frozen layers [PNN] | Weight updates (gradient descent) [LTP] | solid | Not established (no known flow) | U |
| NS-08 | Activation functions (ReLU, GELU) [NMDA] | Weight updates (gradient descent) [LTP] | solid | Not established | U |
| NS-09 | Weight updates (gradient descent) [LTP] | Backpropagation [HEB] | solid | Connected | M/R |
| NS-10 | Weight updates (gradient descent) [LTP] | Update rules (SGD) [STDP] | solid | Connected | M/R |
| NS-11 | Learning rate scheduling [METAP] | Weight updates (gradient descent) [LTP] | solid | Missing (an end is absent) | RES-006 |
| NS-12 | Weight updates (gradient descent) [LTP] | Pruning and sparsity [SPINE] | solid | Not established | U |
| NS-13 | Weight updates (gradient descent) [LTP] | Weight checkpointing [IEG] | solid | Connected | M/R |
| NS-14 | Weight checkpointing [IEG] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| NS-15 | Adaptive optimizers (Adam, momentum) [BDNF] | Weight updates (gradient descent) [LTP] | solid | Connected | M/R |
| NS-16 | Adaptive optimizers (Adam, momentum) [BDNF] | Dynamic network expansion [NGEN] | solid | Not established (no known flow) | U |
| NS-17 | Dynamic network expansion [NGEN] | Contrastive learning [PATSEP] | solid | Not established | U |
| NS-18 | Normalization layers [GLIA] | Weight updates (gradient descent) [LTP] | solid | Not established | U |
| NS-19 | Normalization layers [GLIA] | Inference optimization [MYEL] | solid | Not established | U |
| NS-20 | Inference optimization [MYEL] | Throughput [SPEED] | solid | Not established (no known flow) | U |
| NS-21 | Pruning and sparsity [SPINE] | Parametric memory [LTM] | solid | Not established | U |

### NM. Learning-signal modulators (section 2)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| NM-01 | Reward signal [DA] | Weight updates (gradient descent) [LTP] | solid | Connected | M/R |
| NM-02 | Reward signal [DA] | TD error [RPE] | solid | Connected | M/R |
| NM-03 | Attention weights [ACH] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| NM-04 | Attention weights [ACH] | Input encoding and embedding [ENC] | solid | Missing (an end is absent) | RES-006 |
| NM-05 | Gain and temperature scaling [NE] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| NM-06 | Gain and temperature scaling [NE] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| NM-07 | Gain and temperature scaling [NE] | Exploration temperature [AROUS] | solid | Not established (no known flow) | U |
| NM-08 | Discount factor [SER] | Reward normalization [EMOREG] | solid | Not established (no known flow) | U |
| NM-09 | Discount factor [SER] | Learning rate scheduling [METAP] | solid | Missing (an end is absent) | RES-006 |
| NM-10 | Positive weights [GLU] | Activation functions (ReLU, GELU) [NMDA] | solid | Not established | U |
| NM-11 | Negative weights [GABA] | Critical learning periods [CRIT] | solid | Not established | U |
| NM-12 | Negative weights [GABA] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| NM-13 | Oxytocin (DNE) [OXY] | Base model lineage [PARENT] | solid | Missing (an end is absent) | RES-006 |
| NM-14 | Oxytocin (DNE) [OXY] | Multi-agent learning [OBS] | solid | Missing (an end is absent) | RES-006 |
| NM-15 | Endocannabinoids (DNE) [ENDO] | Weight updates (gradient descent) [LTP] | solid | Missing (an end is absent) | RES-006 |
| NM-16 | Reward model [VTA] | Reward signal [DA] | solid | Partial | M/R |
| NM-17 | Uncertainty estimator [LC] | Gain and temperature scaling [NE] | solid | Not established (no known flow) | U |

### SY. Model components and networks (section 3)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| SY-01 | Replay buffer [HIP] | Input encoding and embedding [ENC] | solid | Connected | M/R |
| SY-02 | Replay buffer [HIP] | Episodic logs [EPI] | solid | Partial | M/R |
| SY-03 | Replay buffer [HIP] | Continual learning pipeline [SYSCON] | solid | Partial | M/R |
| SY-04 | Replay buffer [HIP] | Contrastive learning [PATSEP] | solid | Not established | U |
| SY-05 | Positional encodings [ENT] | Replay buffer [HIP] | solid | Not established | U |
| SY-06 | Embedding layers [PERI] | Replay buffer [HIP] | solid | Not established (no known flow) | U |
| SY-07 | Replay buffer [HIP] | Experience replay [SWR] | solid | Partial | M/R |
| SY-08 | Deep network layers [NEO] | Knowledge graph [SEM] | solid | Not established (no known flow) | U |
| SY-09 | Deep network layers [NEO] | Feature learning [PERCL] | solid | Connected | M/R |
| SY-10 | Reasoning controller [PFC] | Context window [WM] | solid | Missing (an end is absent) | RES-006 |
| SY-11 | Reasoning controller [PFC] | Planning and tool use [EF] | solid | Missing (an end is absent) | RES-006 |
| SY-12 | Reasoning controller [PFC] | Meta-learning [META] | solid | Missing (an end is absent) | RES-006 |
| SY-13 | Reasoning controller [PFC] | Guardrails and masking [INHIB] | solid | Missing (an end is absent) | RES-006 |
| SY-14 | Reasoning controller [PFC] | Representation learning [SCHEMA] | solid | Missing (an end is absent) | RES-006 |
| SY-15 | Safety and risk module [AMY] | Reward shaping [EMO] | solid | Not established (no known flow) | U |
| SY-16 | Safety and risk module [AMY] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| SY-17 | Safety and risk module [AMY] | Risk penalties [FEAR] | solid | Missing (an end is absent) | RES-006 |
| SY-18 | Policy network [BG] | Learned policy [PROC] | solid | Connected | M/R |
| SY-19 | Policy network [BG] | Policy gradient [OPER] | solid | Connected | M/R |
| SY-20 | Policy network [BG] | Default policy [HABIT] | solid | Connected | M/R |
| SY-21 | Policy network [BG] | Reinforcement learning [RL] | solid | Connected | M/R |
| SY-22 | Supervised error-correction layers [CB] | Control policy learning [MOTOR] | solid | Partial | M/R |
| SY-23 | Supervised error-correction layers [CB] | Error-driven gradient learning [ERRL] | solid | Partial | M/R |
| SY-24 | Routing and gating [THAL] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| SY-25 | Loss function [ACC] | Error-driven gradient learning [ERRL] | solid | Connected | M/R |
| SY-26 | Loss function [ACC] | Long training runs [PERS] | solid | Not established (no known flow) | U |
| SY-27 | Multimodal integration layers [PAR] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| SY-28 | Orchestration and control layer [FPN] | Planning and tool use [EF] | solid | Not established (no known flow) | U |
| SY-29 | Importance weighting [SAL] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| SY-30 | World model (internal simulation) [DMN] | Self-reflection and critique [REFL] | solid | Deliberate (an end is excluded by rule) | D |
| SY-31 | World model (internal simulation) [DMN] | Grokking [INSIGHT] | solid | Not established | U |
| SY-32 | World model (internal simulation) [DMN] | Continual learning pipeline [SYSCON] | solid | Partial | M/R |
| SY-33 | Behavior cloning network [MIRR] | Multi-agent learning [OBS] | solid | Missing (an end is absent) | RES-006 |
| SY-34 | Behavior cloning network [MIRR] | Imitation learning [IMIT] | solid | Missing (an end is absent) | RES-006 |

### PH. Hardware and infrastructure (section 4)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| PH-01 | Offline training phase [SLEEP] | Knowledge distillation [SWS] | solid | Partial | M/R |
| PH-02 | Offline training phase [SLEEP] | Generative replay [REM] | solid | Not established (no known flow) | U |
| PH-03 | Knowledge distillation [SWS] | Experience replay [SWR] | solid | Partial | M/R |
| PH-04 | Experience replay [SWR] | Offline reinforcement learning [REPL] | solid | Missing (an end is absent) | RES-006 |
| PH-05 | Offline reinforcement learning [REPL] | Continual learning pipeline [SYSCON] | solid | Missing (an end is absent) | RES-006 |
| PH-06 | Generative replay [REM] | Generative diversity [CREAT] | solid | Not established | U |
| PH-07 | Generative replay [REM] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| PH-08 | Offline training phase [SLEEP] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| PH-09 | Offline training phase [SLEEP] | Reward normalization [EMOREG] | solid | Not established (no known flow) | U |
| PH-10 | Power and energy supply [NUTR] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| PH-11 | Training instability and noise [STRESS] | Context window [WM] | solid | Not established (no known flow) | U |
| PH-12 | Training instability and noise [STRESS] | Retrieval-augmented generation [RETR] | solid | Not established (no known flow) | U |
| PH-13 | Training instability and noise [STRESS] | Distribution shift degradation [ANX] | solid | Not established | U |
| PH-14 | Sensor quality and input resolution [SENS] | Feature extraction [PERC] | solid | Connected | M/R |
| PH-15 | Embodied robotics [EMBOD] | Action-conditioned learning [GEST] | solid | Missing (an end is absent) | RES-006 |
| PH-16 | Action-conditioned learning [GEST] | Input encoding and embedding [ENC] | solid | Connected | M/R |
| PH-17 | Baseline model quality [BASEL] | Offline training phase [SLEEP] | solid | Not established | U |

### CP. Computational processes (section 5)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| CP-01 | Feature extraction [PERC] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| CP-02 | Attention mechanism [ATT] | Sparse and long-context attention [SELATT] | solid | Missing (an end is absent) | RES-006 |
| CP-03 | Sparse and long-context attention [SELATT] | Input encoding and embedding [ENC] | solid | Missing (an end is absent) | RES-006 |
| CP-04 | Input buffer [SENSM] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| CP-05 | Input encoding and embedding [ENC] | Context window [WM] | solid | Partial | M/R |
| CP-06 | Context window [WM] | Computational load (FLOPs) [CL] | solid | Partial | M/R |
| CP-07 | Computational load (FLOPs) [CL] | Task complexity [ICL] | solid | Not established | U |
| CP-08 | Computational load (FLOPs) [CL] | Irrelevant context [ECL] | solid | Not established | U |
| CP-09 | Computational load (FLOPs) [CL] | Useful gradient signal [GCL] | solid | Not established (no known flow) | U |
| CP-10 | Useful gradient signal [GCL] | Representation learning [SCHEMA] | solid | Not established (no known flow) | U |
| CP-11 | Context window [WM] | Tokenization [CHUNK] | solid | Missing (an end is absent) | RES-006 |
| CP-12 | Tokenization [CHUNK] | Parametric memory [LTM] | solid | Missing (an end is absent) | RES-006 |
| CP-13 | Context window [WM] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| CP-14 | Weight consolidation (EWC) [CONS] | Continual learning pipeline [SYSCON] | solid | Missing (an end is absent) | RES-006 |
| CP-15 | Continual learning pipeline [SYSCON] | Parametric memory [LTM] | solid | Partial | M/R |
| CP-16 | Parametric memory [LTM] | Retrieval-augmented generation [RETR] | solid | Partial | M/R |
| CP-17 | Retrieval-augmented generation [RETR] | Model editing [RECON] | solid | Not established | U |
| CP-18 | Model editing [RECON] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| CP-19 | Retrieval-augmented generation [RETR] | Catastrophic forgetting [FORG] | solid | Not established (no known flow) | U |
| CP-20 | Catastrophic forgetting [FORG] | Cache eviction [DECAY] | solid | Not established (no known flow) | U |
| CP-21 | Catastrophic forgetting [FORG] | Retrieval-augmented generation [RETR] | solid | Not established (no known flow) | U |
| CP-22 | Next-token prediction [PRED] | Error-driven gradient learning [ERRL] | solid | Connected | M/R |
| CP-23 | Next-token prediction [PRED] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| CP-24 | Next-token prediction [PRED] | Representation learning [SCHEMA] | solid | Connected | M/R |
| CP-25 | Representation learning [SCHEMA] | Hierarchical feature abstraction [ABSTR] | solid | Partial | M/R |
| CP-26 | Hierarchical feature abstraction [ABSTR] | Few-shot analogy [ANALOG] | solid | Not established | U |
| CP-27 | Few-shot analogy [ANALOG] | Transfer learning [TRANSFER] | solid | Not established | U |
| CP-28 | Representation learning [SCHEMA] | Concept drift adaptation [CONCEPT] | solid | Not established (no known flow) | U |
| CP-29 | Planning and tool use [EF] | Context window [WM] | solid | Not established (no known flow) | U |
| CP-30 | Planning and tool use [EF] | AutoML [SRL] | solid | Not established (no known flow) | U |
| CP-31 | Guardrails and masking [INHIB] | Sparse and long-context attention [SELATT] | solid | Missing (an end is absent) | RES-006 |
| CP-32 | Task adaptation [SHIFT] | Transfer learning [TRANSFER] | solid | Not established | U |
| CP-33 | Meta-learning [META] | Confidence estimates [JOL] | solid | Missing (an end is absent) | RES-006 |
| CP-34 | Confidence estimates [JOL] | Model calibration [CALIB] | solid | Missing (an end is absent) | RES-006 |
| CP-35 | Model calibration [CALIB] | AutoML [SRL] | solid | Missing (an end is absent) | RES-006 |
| CP-36 | Meta-learning [META] | AutoML [SRL] | solid | Missing (an end is absent) | RES-006 |
| CP-37 | AutoML [SRL] | Training strategy selection [STRAT_ENTRY] | solid | Missing (an end is absent) | RES-006 |
| CP-38 | Multimodal embeddings [DUAL] | Input encoding and embedding [ENC] | solid | Partial | M/R |
| CP-39 | Contrastive learning [PATSEP] | Episodic logs [EPI] | solid | Not established | U |
| CP-40 | Amortized inference [AUTO] | Context window [WM] | solid | Not established (no known flow) | U |
| CP-41 | Learned policy [PROC] | Amortized inference [AUTO] | solid | Partial | M/R |
| CP-42 | Chain-of-thought reasoning [REASON] | Generalization [UNDER] | solid | Missing (an end is absent) | RES-006 |
| CP-43 | Scheduled tasks and reminders [PROSP] | Default policy [HABIT] | solid | Not established | U |
| CP-44 | Throughput [SPEED] | Context window [WM] | solid | Not established (no known flow) | U |

### MS. Memory and storage (section 6)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| MS-01 | Input buffer [SENSM] | KV cache [STM] | solid | Connected | M/R |
| MS-02 | KV cache [STM] | Context window [WM] | solid | Partial | M/R |
| MS-03 | KV cache [STM] | Parametric memory [LTM] | solid | Partial | M/R |
| MS-04 | Parametric memory [LTM] | Knowledge base [EXPL] | solid | Partial | M/R |
| MS-05 | Parametric memory [LTM] | Implicit parametric knowledge [IMPL] | solid | Connected | M/R |
| MS-06 | Knowledge base [EXPL] | Episodic logs [EPI] | solid | Not established (no known flow) | U |
| MS-07 | Knowledge base [EXPL] | Knowledge graph [SEM] | solid | Not established (no known flow) | U |
| MS-08 | Episodic logs [EPI] | Knowledge graph [SEM] | solid | Not established (no known flow) | U |
| MS-09 | Episodic logs [EPI] | Interaction history logs [AUTOB] | solid | Partial | M/R |
| MS-10 | Episodic logs [EPI] | Data provenance [SRCM] | solid | Partial | M/R |
| MS-11 | Implicit parametric knowledge [IMPL] | Learned policy [PROC] | solid | Connected | M/R |
| MS-12 | Implicit parametric knowledge [IMPL] | Prompt priming [PRIM] | solid | Missing (an end is absent) | RES-006 |
| MS-13 | Implicit parametric knowledge [IMPL] | Learned input-output mappings [CONDM] | solid | Connected | M/R |

### ME. Learning algorithms (section 7)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| ME-01 | Adaptive gradient scaling [HAB] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| ME-02 | Supervised learning [CLASS] | Learned input-output mappings [CONDM] | solid | Partial | M/R |
| ME-03 | Supervised learning [CLASS] | Safety and risk module [AMY] | solid | Not established (no known flow) | U |
| ME-04 | Policy gradient [OPER] | Default policy [HABIT] | solid | Connected | M/R |
| ME-05 | Reinforcement learning [RL] | TD error [RPE] | solid | Connected | M/R |
| ME-06 | TD error [RPE] | Reward signal [DA] | solid | Connected | M/R |
| ME-07 | Reinforcement learning [RL] | Model-based vs model-free RL [MB] | solid | Partial | M/R |
| ME-08 | Model-based vs model-free RL [MB] | Reasoning controller [PFC] | solid | Missing (an end is absent) | RES-006 |
| ME-09 | Empirical risk minimization [STAT] | Self-supervised learning [IMPLL] | solid | Connected | M/R |
| ME-10 | Empirical risk minimization [STAT] | Language modeling [LANG] | solid | Deliberate (an end is excluded by rule) | D |
| ME-11 | Self-supervised learning [IMPLL] | Implicit parametric knowledge [IMPL] | solid | Connected | M/R |
| ME-12 | Multi-agent learning [OBS] | Capability [SKILL] | solid | Missing (an end is absent) | RES-006 |
| ME-13 | Multi-agent learning [OBS] | Off-policy learning [SOCLEARN_NOTE] | solid | Missing (an end is absent) | RES-006 |
| ME-14 | Imitation learning [IMIT] | Language modeling [LANG] | solid | Deliberate (an end is excluded by rule) | D |
| ME-15 | Grokking [INSIGHT] | Generalization [UNDER] | solid | Not established | U |
| ME-16 | Exploration [PLAY] | Information gain [CUR] | solid | Partial | M/R |
| ME-17 | Exploration [PLAY] | Causal discovery [CAUS] | solid | Not established (no known flow) | U |
| ME-18 | Language modeling [LANG] | Knowledge graph [SEM] | solid | Deliberate (an end is excluded by rule) | D |
| ME-19 | Control policy learning [MOTOR] | Learned policy [PROC] | solid | Connected | M/R |
| ME-20 | Feature learning [PERCL] | Feature extraction [PERC] | solid | Connected | M/R |
| ME-21 | Concept drift adaptation [CONCEPT] | Generalization [UNDER] | solid | Not established | U |
| ME-22 | Error-driven gradient learning [ERRL] | Next-token prediction [PRED] | solid | Connected | M/R |
| ME-23 | Variational inference [ACTINF] | Next-token prediction [PRED] | solid | Connected | M/R |
| ME-24 | Variational inference [ACTINF] | Information gain [CUR] | solid | Partial | M/R |
| ME-25 | Causal discovery [CAUS] | Representation learning [SCHEMA] | solid | Not established (no known flow) | U |

### MO. Objectives and reward signals (section 8)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| MO-01 | Objective function [MOT] | Intrinsic reward [INTR] | solid | Partial | M/R |
| MO-02 | Objective function [MOT] | Extrinsic reward [EXTR] | solid | Deliberate (an end is excluded by rule) | D |
| MO-03 | Autonomy competence relatedness (DNE) [SDT] | Intrinsic reward [INTR] | solid | Partial | M/R |
| MO-04 | Information gain [CUR] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| MO-05 | Information gain [CUR] | Reward signal [DA] | solid | Connected | M/R |
| MO-06 | Relevance scoring [INT] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| MO-07 | Relevance scoring [INT] | Long training runs [PERS] | solid | Not established (no known flow) | U |
| MO-08 | Goal-conditioned policies [GOAL] | Objective function [MOT] | solid | Not established (no known flow) | U |
| MO-09 | Value function [EXPV] | Objective function [MOT] | solid | Partial | M/R |
| MO-10 | Self-evaluation [SELFE] | Long training runs [PERS] | solid | Not established (no known flow) | U |
| MO-11 | Self-evaluation [SELFE] | AutoML [SRL] | solid | Not established (no known flow) | U |
| MO-12 | Continual learning capacity [MIND] | Long training runs [PERS] | solid | Not established | U |
| MO-13 | Continual learning capacity [MIND] | Error-driven gradient learning [ERRL] | solid | Not established | U |
| MO-14 | Credit assignment [ATTRIB] | Self-evaluation [SELFE] | solid | Not established (no known flow) | U |
| MO-15 | Reward shaping [EMO] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| MO-16 | Reward shaping [EMO] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| MO-17 | Exploration temperature [AROUS] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| MO-18 | Distribution shift degradation [ANX] | Context window [WM] | solid | Not established | U |
| MO-19 | Distribution shift degradation [ANX] | Computational load (FLOPs) [CL] | solid | Not established | U |
| MO-20 | Data saturation [BORE] | Attention mechanism [ATT] | solid | Partial | M/R |
| MO-21 | Curriculum sweet spot [FLOW] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| MO-22 | Curriculum sweet spot [FLOW] | Objective function [MOT] | solid | Missing (an end is absent) | RES-006 |
| MO-23 | Reward function [REWD] | Reward signal [DA] | solid | Connected | M/R |
| MO-24 | Reward function [REWD] | Default policy [HABIT] | solid | Connected | M/R |
| MO-25 | Risk penalties [FEAR] | Safety and risk module [AMY] | solid | Missing (an end is absent) | RES-006 |
| MO-26 | Reward normalization [EMOREG] | Distribution shift degradation [ANX] | solid | Not established | U |
| MO-27 | Long training runs [PERS] | Targeted fine-tuning [DP] | solid | Missing (an end is absent) | RES-006 |
| MO-28 | Persona and system prompt [IDENT] | Objective function [MOT] | solid | Missing (an end is absent) | RES-006 |
| MO-29 | Policy control [SENSE] | Autonomy competence relatedness (DNE) [SDT] | solid | Partial | M/R |
| MO-30 | Intrinsic reward [INTR] | Long training runs [PERS] | solid | Not established (no known flow) | U |
| MO-31 | Extrinsic reward [EXTR] | Policy gradient [OPER] | solid | Connected | M/R |

### ST. Training strategies (section 9)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| ST-01 | Evaluation and validation [RP] | Retrieval-augmented generation [RETR] | solid | Not established (no known flow) | U |
| ST-02 | Evaluation and validation [RP] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| ST-03 | Spaced replay [SP] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| ST-04 | Spaced replay [SP] | Catastrophic forgetting [FORG] | solid | Missing (an end is absent) | RES-006 |
| ST-05 | Data shuffling [INTL] | Hierarchical feature abstraction [ABSTR] | solid | Not established (no known flow) | U |
| ST-06 | Data shuffling [INTL] | Transfer learning [TRANSFER] | solid | Not established | U |
| ST-07 | Data augmentation [ELAB] | Knowledge graph [SEM] | solid | Not established | U |
| ST-08 | Data augmentation [ELAB] | Input encoding and embedding [ENC] | solid | Not established | U |
| ST-09 | Self-generated rationales [SELFEX] | Generalization [UNDER] | solid | Deliberate (an end is excluded by rule) | D |
| ST-10 | Generative training [GEN2] | Input encoding and embedding [ENC] | solid | Not established (no known flow) | U |
| ST-11 | Feedback signals (RLHF) [FB] | Error-driven gradient learning [ERRL] | solid | Deliberate (an end is excluded by rule) | D |
| ST-12 | Feedback signals (RLHF) [FB] | Self-evaluation [SELFE] | solid | Deliberate (an end is excluded by rule) | D |
| ST-13 | Feedback signals (RLHF) [FB] | Model calibration [CALIB] | solid | Missing (an end is absent) | RES-006 |
| ST-14 | Hard negative mining [DD] | Retrieval-augmented generation [RETR] | solid | Not established | U |
| ST-15 | Targeted fine-tuning [DP] | Capability [SKILL] | solid | Missing (an end is absent) | RES-006 |
| ST-16 | Targeted fine-tuning [DP] | Feedback signals (RLHF) [FB] | solid | Missing (an end is absent) | RES-006 |
| ST-17 | Few-shot examples [WE] | Task complexity [ICL] | solid | Deliberate (an end is excluded by rule) | D |
| ST-18 | Few-shot examples [WE] | Computational load (FLOPs) [CL] | solid | Deliberate (an end is excluded by rule) | D |
| ST-19 | Curriculum learning [SCAF] | Learnability frontier [ZPD] | solid | Missing (an end is absent) | RES-006 |
| ST-20 | Learnability frontier [ZPD] | Objective function [MOT] | solid | Not established (no known flow) | U |
| ST-21 | Convergence thresholds [MAST] | Pretrained priors [PK] | solid | Deliberate (an end is excluded by rule) | D |
| ST-22 | Zero-shot evaluation [PRET] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| ST-23 | Zero-shot evaluation [PRET] | Input encoding and embedding [ENC] | solid | Missing (an end is absent) | RES-006 |
| ST-24 | Indexing and hashing [MNEM] | Input encoding and embedding [ENC] | solid | Partial | M/R |
| ST-25 | Graph neural networks [CMAP] | Representation learning [SCHEMA] | solid | Missing (an end is absent) | RES-006 |
| ST-26 | Multimodal training [MULTI] | Multimodal embeddings [DUAL] | solid | Partial | M/R |
| ST-27 | Domain randomization [VAR] | Transfer learning [TRANSFER] | solid | Not established | U |
| ST-28 | Teacher-student training [TEACH] | Generalization [UNDER] | solid | Partial | M/R |
| ST-29 | Federated learning [COLL] | Multi-agent learning [OBS] | solid | Missing (an end is absent) | RES-006 |
| ST-30 | Contrastive loss [CONTR] | Hierarchical feature abstraction [ABSTR] | solid | Not established | U |
| ST-31 | Self-reflection and critique [REFL] | Meta-learning [META] | solid | Missing (an end is absent) | RES-006 |
| ST-32 | Scratchpad notes [NOTE] | Input encoding and embedding [ENC] | solid | Missing (an end is absent) | RES-006 |
| ST-33 | Simulated rollouts [MENT] | Control policy learning [MOTOR] | solid | Partial | M/R |
| ST-34 | Early stopping [BREAK] | Weight consolidation (EWC) [CONS] | solid | Missing (an end is absent) | RES-006 |
| ST-35 | Early stopping [BREAK] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| ST-36 | Overfitting [OVERL] | Amortized inference [AUTO] | solid | Not established (no known flow) | U |
| ST-37 | Context summarization [SUMM] | Data augmentation [ELAB] | solid | Missing (an end is absent) | RES-006 |
| ST-38 | Training strategy selection [STRAT_ENTRY] | Evaluation and validation [RP] | solid | Missing (an end is absent) | RES-006 |
| ST-39 | Training strategy selection [STRAT_ENTRY] | Spaced replay [SP] | solid | Missing (an end is absent) | RES-006 |
| ST-40 | Training strategy selection [STRAT_ENTRY] | Data shuffling [INTL] | solid | Missing (an end is absent) | RES-006 |
| ST-41 | Training strategy selection [STRAT_ENTRY] | Data augmentation [ELAB] | solid | Missing (an end is absent) | RES-006 |

### SO. Data and environment (section 10)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| SO-01 | Teacher model quality [TEACHER] | Curriculum learning [SCAF] | solid | Missing (an end is absent) | RES-006 |
| SO-02 | Teacher model quality [TEACHER] | Feedback signals (RLHF) [FB] | solid | Deliberate (an end is excluded by rule) | D |
| SO-03 | Teacher model quality [TEACHER] | Prompt framing effects [EXPECT] | solid | Deliberate (an end is excluded by rule) | D |
| SO-04 | Peer agents [PEER] | Multi-agent learning [OBS] | solid | Not established (no known flow) | U |
| SO-05 | Peer agents [PEER] | Belonging (DNE) [BELONG] | solid | Missing (an end is absent) | RES-006 |
| SO-06 | Base model lineage [PARENT] | Oxytocin (DNE) [OXY] | solid | Missing (an end is absent) | RES-006 |
| SO-07 | Base model lineage [PARENT] | Training corpus [LANGENV] | solid | Deliberate (an end is excluded by rule) | D |
| SO-08 | Base model lineage [PARENT] | Random seed and initialization [TEMP] | solid | Partial | M/R |
| SO-09 | Compute budget [SES] | Low-quality early data [NUTRI] | solid | Not established (no known flow) | U |
| SO-10 | Low-quality early data [NUTRI] | Training instability and noise [STRESS] | solid | Not established (no known flow) | U |
| SO-11 | Low-quality early data [NUTRI] | Pretraining [DEV] | solid | Deliberate (an end is excluded by rule) | D |
| SO-12 | Compute budget [SES] | Benchmarks and curriculum design [SCHOOL] | solid | Not established (no known flow) | U |
| SO-13 | Training corpus [LANGENV] | Language modeling [LANG] | solid | Deliberate (an end is excluded by rule) | D |
| SO-14 | Data distribution norms [CULT] | Goal-conditioned policies [GOAL] | solid | Not established (no known flow) | U |
| SO-15 | Data distribution norms [CULT] | Persona and system prompt [IDENT] | solid | Missing (an end is absent) | RES-006 |
| SO-16 | Training environment [CLIM] | Safe exploration [SAFE] | solid | Partial | M/R |
| SO-17 | Safe exploration [SAFE] | Distribution shift degradation [ANX] | solid | Not established | U |
| SO-18 | Belonging (DNE) [BELONG] | Objective function [MOT] | solid | Missing (an end is absent) | RES-006 |
| SO-19 | Bias-induced degradation [STEREO] | Distribution shift degradation [ANX] | solid | Not established | U |
| SO-20 | Bias-induced degradation [STEREO] | Context window [WM] | solid | Not established (no known flow) | U |
| SO-21 | Frameworks and tooling [TECH] | Prompt injection and noise [DISTR] | solid | Not established | U |
| SO-22 | Prompt injection and noise [DISTR] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| SO-23 | Sensor and data noise [PHYS2] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| SO-24 | Prompt context dependence [CTX] | Retrieval-augmented generation [RETR] | solid | Not established | U |
| SO-25 | Real-world data diversity [NATURE] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| SO-26 | Real-world data diversity [NATURE] | Training instability and noise [STRESS] | solid | Missing (an end is absent) | RES-006 |
| SO-27 | Benchmarks and curriculum design [SCHOOL] | Evaluation and validation [RP] | solid | Partial | M/R |
| SO-28 | Prompt framing effects [EXPECT] | Self-evaluation [SELFE] | solid | Not established | U |
| SO-29 | Open-source community [COMM] | Persona and system prompt [IDENT] | solid | Missing (an end is absent) | RES-006 |
| SO-30 | Safe exploration [SAFE] | Information gain [CUR] | solid | Not established (no known flow) | U |

### ID. Model differences (section 11)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| ID-01 | Pretrained priors [PK] | Input encoding and embedding [ENC] | solid | Deliberate (an end is excluded by rule) | D |
| ID-02 | Pretrained priors [PK] | Computational load (FLOPs) [CL] | solid | Deliberate (an end is excluded by rule) | D |
| ID-03 | Pretrained priors [PK] | Representation learning [SCHEMA] | solid | Deliberate (an end is excluded by rule) | D |
| ID-04 | Pretrained priors [PK] | Negative transfer [EXPERT] | solid | Deliberate (an end is excluded by rule) | D |
| ID-05 | Negative transfer [EXPERT] | Curriculum learning [SCAF] | solid | Missing (an end is absent) | RES-006 |
| ID-06 | Context window size [WMC] | Context window [WM] | solid | Partial | M/R |
| ID-07 | Model scale (parameters) [IQ] | Chain-of-thought reasoning [REASON] | solid | Missing (an end is absent) | RES-006 |
| ID-08 | Model scale (parameters) [IQ] | Throughput [SPEED] | solid | Connected | M/R |
| ID-09 | Hyperparameter configuration [PERSN] | Training stability [CONSC] | solid | Partial | M/R |
| ID-10 | Hyperparameter configuration [PERSN] | Exploration rate (epsilon) [OPEN] | solid | Partial | M/R |
| ID-11 | Training stability [CONSC] | Long training runs [PERS] | solid | Not established (no known flow) | U |
| ID-12 | Exploration rate (epsilon) [OPEN] | Information gain [CUR] | solid | Partial | M/R |
| ID-13 | Architecture diversity [NEURO] | Attention mechanism [ATT] | solid | Missing (an end is absent) | RES-006 |
| ID-14 | Architecture diversity [NEURO] | Planning and tool use [EF] | solid | Missing (an end is absent) | RES-006 |
| ID-15 | Architecture diversity [NEURO] | Feature extraction [PERC] | solid | Missing (an end is absent) | RES-006 |
| ID-16 | Training language mix [LANGB] | Language modeling [LANG] | solid | Deliberate (an end is excluded by rule) | D |
| ID-17 | Training stage (epoch) [AGEG] | Critical learning periods [CRIT] | solid | Not established | U |
| ID-18 | Training stage (epoch) [AGEG] | Planning and tool use [EF] | solid | Not established (no known flow) | U |
| ID-19 | Gradient clipping [SELFREG] | Planning and tool use [EF] | solid | Not established (no known flow) | U |
| ID-20 | Gradient clipping [SELFREG] | Default policy [HABIT] | solid | Not established (no known flow) | U |
| ID-21 | Random seed and initialization [TEMP] | Reward normalization [EMOREG] | solid | Not established (no known flow) | U |
| ID-22 | Learning styles MYTH (DNE) [STYLES] | Teacher-student training [TEACH] | dotted | Missing (an end is absent) | RES-006 |
| ID-23 | Data coverage gaps [GENDER] | Self-evaluation [SELFE] | solid | Not established (no known flow) | U |

### OU. Capabilities (section 12)

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| OU-01 | Episodic logs [EPI] | Learned knowledge [KNOW] | solid | Not established (no known flow) | U |
| OU-02 | Knowledge graph [SEM] | Learned knowledge [KNOW] | solid | Not established (no known flow) | U |
| OU-03 | Learned policy [PROC] | Capability [SKILL] | solid | Partial | M/R |
| OU-04 | Amortized inference [AUTO] | Capability [SKILL] | solid | Missing (an end is absent) | RES-006 |
| OU-05 | Default policy [HABIT] | Policy updates [BEHAV] | solid | Connected | M/R |
| OU-06 | Generalization [UNDER] | Transfer learning [TRANSFER] | solid | Not established | U |
| OU-07 | Learned knowledge [KNOW] | Generalization [UNDER] | solid | Not established | U |
| OU-08 | Capability [SKILL] | Specialist model [EXPT] | solid | Missing (an end is absent) | RES-006 |
| OU-09 | Learned knowledge [KNOW] | Specialist model [EXPT] | solid | Missing (an end is absent) | RES-006 |
| OU-10 | Parametric memory [LTM] | Weight retention [RET] | solid | Partial | M/R |
| OU-11 | Representation learning [SCHEMA] | Transfer learning [TRANSFER] | solid | Not established | U |
| OU-12 | Transfer learning [TRANSFER] | Domain adaptation [ADAPT] | solid | Not established | U |
| OU-13 | Specialist model [EXPT] | Domain adaptation [ADAPT] | solid | Missing (an end is absent) | RES-006 |
| OU-14 | World model (internal simulation) [DMN] | Generative diversity [CREAT] | solid | Not established | U |
| OU-15 | Hierarchical feature abstraction [ABSTR] | Generative diversity [CREAT] | solid | Not established | U |
| OU-16 | Generative diversity [CREAT] | Domain adaptation [ADAPT] | solid | Not established | U |
| OU-17 | Intrinsic reward [INTR] | Wellbeing (DNE) and continual learning [WELL] | solid | Missing (an end is absent) | RES-006 |
| OU-18 | Weight retention [RET] | Learned knowledge [KNOW] | solid | Partial | M/R |
| OU-19 | Policy updates [BEHAV] | Wellbeing (DNE) and continual learning [WELL] | solid | Missing (an end is absent) | RES-006 |

### LC. Links to and from the central node

| ID | From | To | Kind | Status | Basis |
|---|---|---|---|---|---|
| LC-01 | Learned knowledge [KNOW] | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | solid | Not established (no known flow) | U |
| LC-02 | Capability [SKILL] | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | solid | Missing (an end is absent) | RES-006 |
| LC-03 | Default policy [HABIT] | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | solid | Not established (no known flow) | U |
| LC-04 | Generalization [UNDER] | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | solid | Not established | U |
| LC-05 | Transfer learning [TRANSFER] | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | solid | Not established | U |
| LC-06 | Policy updates [BEHAV] | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | solid | Not established (no known flow) | U |
| LC-07 | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | Specialist model [EXPT] | solid | Missing (an end is absent) | RES-006 |
| LC-08 | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | Wellbeing (DNE) and continual learning [WELL] | solid | Missing (an end is absent) | RES-006 |
| LC-09 | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | Pretrained priors [PK] | dotted | Deliberate (an end is excluded by rule) | D |
| LC-10 | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | Self-evaluation [SELFE] | dotted | Not established (no known flow) | U |
| LC-11 | MACHINE LEARNING durable change in parameters, capability or policy [LEARN] | Weight updates (gradient descent) [LTP] | dotted | Not established (no known flow) | U |

## 6. Limits of this record

- Statuses derive from RES-006's block statuses; if a block's status changes, the
  lines touching it change. RES-006 and this record must be revised together.
- "Connected" and "Partial" rest on flows observed or read in the working sessions;
  a flow that exists but was not seen is recorded as not established, not as missing.
- A connected line is not a verdict on how well the flow works.
- Lines that touch N/A nodes are excluded; their absence is not a difference.

## 7. Related records

[RES-006](res-006-human-like-learning-gap-analysis.md),
[RES-009](res-009-hard-and-deferred-relationships.md),
[GOV-010](../governance/gov-010-relational-implementation-plan.md),
[GOV-009](../governance/gov-009-implementation-plan.md),
[GOV-008](../governance/gov-008-human-like-learning-motivation.md),
[`ai_learning.mmd`](../../assets/ai_learning.mmd).

## 8. Addendum (2026-10-21, response to the branch audit)

- The map is a **source of hypotheses**, not a validated specification. Line statuses are
  derived from RES-006 block statuses plus known flows; they do not validate a line's causal
  direction or the map's completeness.
- A line being implemented, alive and tested is **implementation coverage**, not evidence
  of human-like function. Evidence of benefit comes only from the T1/T2 results planned in
  [GOV-010](../governance/gov-010-relational-implementation-plan.md).
- Some lines are descriptive hypotheses with no literal software channel; they are recorded
  and not wired.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-006](../architecture/arc-006-relation-registry.md): Relation registry
- [ASS-009](../assurance/ass-009-audit-response-adjustments-branch.md): Audit response: documentation and plan on the Adjustments branch
- [ASS-010](../assurance/ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [GOV-012](../governance/gov-012-execution-record-gov-010.md): Execution record: relational implementation plan (GOV-010)
- [GOV-013](../governance/gov-013-categorical-system-testing-plan.md): Categorical system testing: coherence within and between learning groups
<!-- backlinks:end -->
