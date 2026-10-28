# Human-like learning gap analysis: SkyBot against the AI-counterpart learning map

> **Document control:** [RES-006](../README.md#research) · Research / gap analysis · **State:** Record · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-20. **Status:** a dated comparison, not a plan and not a
claim that any gap should be closed. It states what differs between SkyBot as
it stands and the learning system described by
[`ai_learning.mmd`](../../assets/ai_learning.mmd) (rendered in
[`ai_learning_full.png`](../../assets/ai_learning_full.png)).

## 1. Purpose and scope

The owner generated `ai_learning.mmd` by translating the dependencies of human
learning into their AI counterparts. The goal behind it is to give SkyBot the
most human-like learning achievable. This record compares that map with SkyBot's
current learning processes, term by term, and lists the relationship-level
differences.

**Terms excluded, by the owner's instruction.** Terms marked N/A in the map
describe things an AI system does not need, so they are not compared: Cortisol,
Aerobic exercise, Hydration, Circadian rhythm, Inflammation and illness, Aging,
Hormones (puberty and menopause), Caffeine / alcohol / nicotine / drugs, and
Mental health.

**Terms included, by the owner's instruction (revision of 2026-10-20).** Terms
marked DNE (does not exist) are **included**: the goal of this record is to list
everything SkyBot does not have except what it does not need. They are
compared in section 5.13. Because the map itself says a DNE term has no AI
counterpart, SkyBot's lack of one is expected; each row records only what
nearest analogue, if any, SkyBot has. Section 8 consolidates every absence in
one list.

**What this record does not do.** It does not change any code, does not scan
the repository, and does not run any test. It does not say any gap is a defect:
several gaps are deliberate (section 3).

## 2. Evidence rules

The comparison uses only what was directly observed in the working session that
produced it (2026-10-17 to 2026-10-20). Each row carries one basis code.

| Code | Meaning |
|---|---|
| **M** | Measured in a run on the training host during that session; the value is quoted. |
| **R** | Read in code, a configuration file or a log by the author of this record. |
| **A** | Reported with file citations by a code-exploration step in that session and not independently re-read by the author. |
| **D** | A design or standing rule recorded in the repository; the statement is about intent. |
| **U** | **Not established.** The session gave no basis for a statement about SkyBot, so none is made. The row is a verification item (section 8), not a finding. |

"Gap" below is a difference from the map's ideal, not a ruling. The status words
are **Present** (a working counterpart), **Partial** (a counterpart exists but
covers part of the idea, is off in the measured game, or is advice only),
**Absent** (no counterpart; basis stated) and **Not established** (U).

## 3. Differences that are deliberate

These follow standing rules of the project and should not be read as
shortcomings. They are still differences from the map.

| Map term(s) | SkyBot | Basis |
|---|---|---|
| Pretraining, Pretrained priors, Base model lineage, Teacher model quality, Behavior cloning network, Imitation learning | None. SkyBot learns from scratch with no demonstrations, videos or recipes; a new universal brain starts fresh. | D (CLAUDE.md §1, §10) |
| Feedback signals (RLHF), Self-reflection and critique, Persona and system prompt, Chain-of-thought, Language modeling, Training corpus (text) | No human feedback and no language in the learning loop. The vision-language model exists in the codebase but was off in every Doom run. | D, R (symbolic grounding disabled at launch) |
| Reward signals computed from engine state | Excluded from the universal brain's reward. Evaluator counters are measurement only. | D (ADR-002, CLAUDE.md §10) |
| Extrinsic reward | Doom pays 0 by design. | D, R |

## 4. Where SkyBot stands (summary, with measured values)

The latest controlled comparison is in
[ASS-003](../assurance/ass-003-policy-learning.md); the incident behind it is
[ASS-006](../assurance/ass-006-realtime-action-lag-incident.md). In brief, on
the training host, tiny x 8 Doom, 120 minutes per run, five runs:

- The replay lag probe found the old code mislabelling real-time transitions
  (best shift -1 on 8 of 8 streams) and the fixed code correctly aligned (shift
  0 on 8 of 8 streams), in two seeds. **M**
- Action attribution rose from about 0.02 to 0.15-0.20; the dependence of the
  action on the state (I(S;A)) rose to 0.23-0.32 nats against 0.14-0.19; the
  entropy over valid actions fell to 0.87 against 0.92-0.94. **M**
- Play did not consistently improve: no run reached an exit or left the first
  map, and the behaviour counters disagreed between seeds. **M**
- The world model was healthy by its own measures (information in the latent
  grew from about 10 to 36 nats; fresh-data loss about 0.01). **M**

So the learning *machinery* now measurably runs; what is largely missing,
against the map, is most of the structure that turns experience into durable,
transferable, self-directed knowledge. The sections below detail it.

## 5. Term-by-term comparison

Rows follow the map's twelve groups and its term names. "Counterpart in
SkyBot" names the nearest existing thing.

### 5.1 Model parameters and architecture

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Weights | World model, policy, curiosity model, dream actor | Present | R, M | None in kind. |
| Pretraining | None (fresh start) | Absent | D | Deliberate (section 3). |
| Critical learning periods | No mechanism identified | Not established | U | Unknown whether early-life plasticity windows exist. |
| Weight updates (gradient descent) | Adam on the world model; PPO updates | Present | R, A | None in kind. |
| Backpropagation | Through every learned component | Present | R | None. |
| Update rules (SGD) | Adam variants, not plain SGD | Present | R, A | Different rule from the map's named example. |
| Learning rate scheduling | World model: not established. Policy: constant, no scheduler | Absent (policy) | A | No decay or warm-up for the policy; world-model schedule unverified. |
| Pruning and sparsity | None identified | Not established | U | |
| Dynamic network expansion | Function-preserving growth of the world model; growth of the policy's input layers built; autoscaler decides but its actuation is off | Partial | R, M | Growth was exercised once live (tiny to small, outputs matched to about 1e-6); no live run has grown by its own decision beyond the test, and a grow still costs the policy its saved head unless the policy is grown too (built, verified only in unit tests). |
| Inference optimization | Separate inference copy published every 4 gradient steps; bf16 | Present | R, M | None in kind. |
| Normalization layers | Not established | Not established | U | |
| Adaptive optimizers (Adam, momentum) | Adam, fused on the world model | Present | R, M | None. |
| Activation functions | Not established | Not established | U | |
| Frozen layers | A frozen view encoder for novelty; frozen teachers when kickstart is on (off) | Partial | R | No staged freezing of learning layers by training phase. |
| Weight checkpointing | Periodic and final checkpoints, atomic writes, backups before a resize | Present | R, M | Interval-based (every 25,000 steps plus a final one); a power or network loss between checkpoints loses that span. |

### 5.2 Learning-signal modulators

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Reward signal | Mixed intrinsic and extrinsic reward | Present | R, M | The mixture is dominated by one novelty term (episodic novelty 80-90 % of income). |
| Attention weights | World-model slot attention exists but was off in the runs; the policy trunk is a plain feed-forward network | Absent (live) | R | No learned attention gating what the policy attends to. |
| Gain and temperature scaling | Entropy bonus, boosted after a feature-only resume; logit range bound | Partial | R, A | Fixed coefficients; not state- or uncertainty-dependent. |
| Discount factor | Fixed gamma and lambda; reward discount across option steps | Present | A | Fixed, not adapted to task horizon. |
| Positive weights / Negative weights | No explicit excitatory-inhibitory split identified | Not established | U | |

### 5.3 Model components and networks

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Replay buffer | Multi-stream block buffer with a memory budget, persistence, relabelling and stream-cut merging | Present | R, M | Used by the world model only; the policy trains on-policy and discards its rollout after each update. |
| Reasoning controller | Vision-language symbolizer, off for Doom | Absent (live) | R | No reasoning component participates in the measured runs. |
| Safety and risk module | Guards (sensor guard, action mask, farm damp, rewind on stagnation); no learned risk model | Partial | R, A | No learned estimate of danger; a death is penalised only through lost future reward. |
| Policy network | Actor and critic over the world-model latent | Present | R, M | None in kind. |
| Supervised error-correction layers | Inverse-dynamics and reward heads trained on the agent's own data | Partial | R | No external teacher signal (deliberate). |
| Deep network layers | Present | Present | R | None. |
| Routing and gating | Knowledge conditioner gate; action masks; option executor | Present | R | Gating inputs are mostly empty in Doom (the knowledge vector carries nothing there). |
| Reward model | World-model reward head | Partial | R | Used for imagination only, not for the policy's reward. |
| Uncertainty estimator | Latent information (z-info) and reconstruction error are measured; ensembles exist only as a shadow measurement path | Partial | R, M | Uncertainty does not drive exploration or caution directly. |
| Loss function | Several (reconstruction, KL, reward, continue, flow, inverse, horizon, forward model, PPO) | Present | R | None. |
| Multimodal integration layers | Pixels, body senses, motion, light, tempo, sound enter the world model | Present | R | Quality of the integration is unmeasured. |
| World model (internal simulation) | Recurrent state-space model; imagination | Present | R, M | Not used for planning at decision time (acting uses the policy; imagination trains and shapes). |
| Orchestration and control layer | The developmental loop, lifelong controller, supervisor | Present | R | None in kind. |
| Importance weighting | Prioritised-replay columns exist; advantages are normalised and clipped | Partial | R | Whether prioritisation is active in the measured runs is not established. |
| Positional encodings | Not applicable to the current architecture | Not established | U | |
| Embedding layers | Encoder embedding (128 wide at the smallest size) | Present | R | None. |
| Behavior cloning network | None | Absent | D | Deliberate (section 3). |

### 5.4 Hardware and infrastructure

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Offline training phase | The loop alternates a collection segment and a learning phase; no separate consolidation phase | Partial | R, M | No sleep-like phase devoted to consolidation. |
| Knowledge distillation | Dream-to-real distillation; kickstart teacher (off) | Partial | R, A | The distillation target was near uniform in measured runs; the effective weight was negligible. |
| Generative replay | Imagination rollouts train a dream actor and feed imagination curiosity | Partial | R | Not used to protect old knowledge from forgetting. |
| Experience replay | World-model replay only | Partial | R | The policy has none. |
| Power and energy supply | No handling beyond checkpoints | Absent | M | The training host was unreachable for about 11 hours on 2026-10-19; the cause was not established. |
| Training instability and noise | Measured and partly guarded (KL trust region; free-nats floor) | Partial | M, R | Past incidents (posterior collapse, divergence on one accelerator) were fixed individually, not by a general stability controller. |
| Sensor quality and input resolution | Fixed 128 x 128 pixels plus small fovea and sense channels | Present | R | No adaptive resolution. |
| Embodied robotics | Simulated bodies in games only | Absent | D | |

### 5.5 Computational processes

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Feature extraction | Encoder | Present | R | None. |
| Attention mechanism | None live | Absent | R | See 5.2. |
| Sparse and long-context attention | None | Absent | R | The recurrent state is the only long-range carrier. |
| Input encoding and embedding | Present | Present | R | None. |
| Context window | Recurrent state; training windows of 32 steps | Partial | R | No explicit long context or retrieval over it. |
| Computational load (FLOPs) | Phase timing and act latency are logged; the autoscaler weighs latency and memory | Present | M | None in kind. |
| Task complexity / Irrelevant context / Useful gradient signal | No direct instruments; explained variance and I(S;A) partly proxy the last | Partial | M | The first two are not measured at all. |
| Tokenization | No chunking of observations; options are the only temporal chunking and none were offered in Doom (0 of 16,384 decisions) | Absent (live) | M | |
| Weight consolidation (EWC) | Forgetting is measured by paired probes; no consolidation penalty | Absent | R, M | Measured, not prevented. |
| Continual learning pipeline | Lifelong loop, resume, growth | Partial | R | No forgetting protection for the policy; world model protected only by replay. |
| Model editing | None identified | Not established | U | |
| Retrieval-augmented generation | A knowledge graph feeding a knowledge vector | Partial | R, M | Empty in Doom (graph density 0.00 in the runs). |
| Catastrophic forgetting | Measured; mitigated by replay (world model) | Partial | R, M | No mitigation for the policy beyond restoring it. |
| Next-token prediction | Next-frame and latent prediction | Present | R | None in kind. |
| Representation learning | Encoder and latent; latent information measured | Present | M | The learned representation's usefulness for the policy is unproven. |
| Hierarchical feature abstraction | Two latent prediction horizons (4 and 16) | Partial | R | One shallow temporal hierarchy. |
| Few-shot analogy | None identified | Not established | U | |
| Planning and tool use | Imagination exists; acting does not plan at decision time | Partial | A | No search or lookahead over imagined futures when choosing an action. |
| Guardrails and masking | Action mask, logit bound, sensor guard | Present | R | None in kind. |
| Task adaptation | One shared brain across games by design; only one game ran live | Partial | D, M | Cross-game adaptation unproven. |
| Meta-learning | None | Absent | U | The autoscaler is a meta-level only in advice mode. |
| Confidence estimates / Model calibration | None for the policy | Absent | U | |
| AutoML | Autoscaler for size and streams (advice only) | Partial | R, M | Hyperparameters are hand-set. |
| Multimodal embeddings | Present in the world-model input | Partial | R | Joint embedding quality not measured. |
| Contrastive learning | None identified | Not established | U | |
| Cache eviction | Replay overwrites oldest when capped; per-life view memory capped | Present | R, M | |
| Scheduled tasks and reminders | None identified | Not established | U | |
| Throughput | About 5.4 decisions per second per stream with 8 streams | Present | M | |
| Chain-of-thought reasoning | None | Absent | D | Deliberate. |
| Amortized inference | The policy; dream distillation into it | Partial | R | |

### 5.6 Memory and storage systems

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Input buffer | Recent frames and the rewind ring | Partial | R | |
| KV cache | The recurrent state | Present | R | |
| Parametric memory | Weights | Present | R | |
| Knowledge base / Knowledge graph | Knowledge graph, empty in Doom | Partial | M | Nothing is asserted or retracted in the measured game. |
| Episodic logs | The replay buffer; an episodic record store for sightings and breaks, empty in Doom | Partial | M | No event-level memory the agent can query. |
| Implicit parametric knowledge | Present | Present | R | |
| Learned policy | Present; a skill bank of policy copies exists but nothing was minted in Doom | Partial | M, D | The skill route is closed in Doom until skill discovery (RES-005) is built. |
| Prompt priming | None | Absent | D | |
| Learned input-output mappings | Forward model, inverse head | Present | R | |
| Interaction history logs | Telemetry and incident bundles, not fed back into learning | Partial | R | Logged, not learned from. |
| Data provenance | Replay carries an action-semantics stamp, sensor-layout hash and manifest; policy carries a reward fingerprint | Present | R | |

### 5.7 Learning algorithms

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Adaptive gradient scaling | Adam; gradient clipping | Present | A | |
| Supervised learning | Heads trained on own data | Partial | R | No external labels (deliberate). |
| Policy gradient / Reinforcement learning / TD error | PPO with advantage estimation | Present | R, A | On-policy only. |
| Model-based vs model-free RL | Hybrid: a model-free policy over a learned latent, with imagination-trained auxiliaries | Partial | A | No model-based planning for action choice. |
| Empirical risk minimization | Present | Present | R | |
| Self-supervised learning | Reconstruction, KL, flow, inverse and forward models | Present | R | |
| Multi-agent learning | Eight parallel streams of one brain; no interaction between agents in Doom | Partial | R | Parallel data, not social learning. |
| Imitation learning | None | Absent | D | Deliberate. |
| Grokking | Not instrumented | Not established | U | |
| Exploration | Intrinsic curiosity terms plus the policy's own entropy | Present | M | One term dominates; exploration is not directed at learnability. |
| Language modeling | None | Absent | D | Deliberate. |
| Control policy learning / Feature learning / Error-driven gradient learning | Present | Present | R | |
| Concept drift adaptation | Non-stationary stream handled by recency in replay; foundation drift detectors exist as shadow machinery | Partial | R | No drift-triggered adaptation. |
| Variational inference | KL term of the world model | Present | R | |
| Causal discovery | Action attribution is measured (0.15-0.20 after the fix); no causal structure learning | Partial | M | Skill discovery by "agency" is planned (RES-005), not built. |
| Offline reinforcement learning | None | Absent | A | The replay is not used to train the policy. |

### 5.8 Objectives and reward signals

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Objective function | A composite of heterogeneous terms | Partial | R | No single stated objective; the mixture's weights are hand-set. |
| Intrinsic reward | Learning progress, episodic novelty, view novelty, imagination, empowerment | Present | M | One term (episodic) carries 80-90 % of income; learning progress paid about 0.0003 per step before the credit fix. |
| Extrinsic reward | None in Doom | Absent | D | Deliberate. |
| Information gain | Approximated by learning progress and imagination | Partial | M | True information gain is not computed. |
| Relevance scoring | Boring-view and habituation damping | Partial | R | |
| Goal-conditioned policies | A goal vector exists; in Doom one goal was generated and none completed | Partial | M | |
| Value function | Critic; explained variance 0.95-0.98 after the fix | Present | M | |
| Self-evaluation | Competence prediction and telemetry exist; no self-model used in Doom | Partial | R, M | |
| Continual learning capacity | Not measured | Not established | U | |
| Credit assignment | GAE, plus real-time credit delay; attribution measured | Present | M | Held to the ASS-003 result: correct labelling, unproven benefit to play. |
| Reward shaping | Potentials for reach, persistence, gaze; empowerment | Present | R | Most are about zero in Doom. |
| Exploration temperature | Entropy coefficient and a one-time boost | Partial | R | Not adapted over training. |
| Distribution shift degradation | Not measured | Not established | U | |
| Data saturation | View novelty saturates; per-life memory resets | Partial | R | |
| Curriculum sweet spot | None | Absent | A | Nothing selects tasks by difficulty. |
| Reward function | Present | Present | R | |
| Risk penalties | None explicit | Absent | A | |
| Reward normalization | Learning-progress normaliser; advantage normalisation; no return normaliser | Partial | A | |
| Long training runs | The longest controlled universal-brain run was 120 minutes | Partial | M | Long-horizon behaviour of the universal brain is unmeasured. |
| Policy control | Action attribution and empowerment | Partial | M | |

### 5.9 Training strategies that work

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Evaluation and validation | Fresh-data loss on never-trained windows; a pre-registered comparison protocol | Partial | R, M | No held-out tasks or environments. |
| Spaced replay | None | Absent | R | Replay is sampled uniformly or by priority. |
| Data shuffling | Random sequence sampling | Present | R | |
| Data augmentation | Not established | Not established | U | |
| Self-generated rationales | None | Absent | D | |
| Generative training | Dream training | Present | R | |
| Feedback signals (RLHF) | None | Absent | D | Deliberate. |
| Hard negative mining | None identified | Not established | U | |
| Targeted fine-tuning | None | Absent | U | |
| Few-shot examples | None | Absent | D | |
| Curriculum learning | None | Absent | A | |
| Learnability frontier | Learning progress as a proxy | Partial | R | Not used to pick what to practise. |
| Convergence thresholds | Autoscaler plateau test; no mastery criterion | Partial | R | Mastery is unmeasured (CLAUDE.md §4.6). |
| Zero-shot evaluation | None | Absent | U | |
| Indexing and hashing | Hash buckets for view novelty; nearest-neighbour memory | Partial | R | |
| Graph neural networks | Disabled in the runs (the graph library was not installed) | Absent (live) | M | |
| Multimodal training | Present | Partial | R | |
| Domain randomization | None identified | Not established | U | |
| Teacher-student training | Dream-to-real distillation | Partial | R | |
| Federated learning | None | Absent | U | |
| Contrastive loss | None identified | Not established | U | |
| Self-reflection and critique | None | Absent | D | |
| Scratchpad notes | None | Absent | U | |
| Action-conditioned learning | Executed-action labelling; attribution gate; inverse dynamics | Present | M | New; first controlled evidence in ASS-003. |
| Simulated rollouts | Present | Present | R | |
| Early stopping | A KL-based stop on actor updates; no early stopping of training | Partial | R | |
| Overfitting | Fresh-to-train gap measured (0.74-1.77); autoscaler flagged memorising | Partial | M | Advice only. |
| Context summarization | None | Absent | U | |

### 5.10 Data and environment

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Teacher model quality | None | Absent | D | Deliberate. |
| Peer agents | Eight same-brain streams, not interacting | Partial | R | |
| Base model lineage | Fresh brains; the starting size is recorded | Partial | R | |
| Compute budget | 8 GB GPU, 16 GB RAM; a replay ceiling of 5.5 GB | Present | M | Tight; one game at a time. |
| Low-quality early data | Early play is near random | Present | M | |
| Training corpus | The agent's own experience | Present | R | |
| Data distribution norms | Non-stationary, shaped by the policy itself | Present | R | |
| Training environment | One engine, one map in every measured run | Partial | M | |
| Bias-induced degradation | Dominant novelty term raises a farming risk (flagged, not proven) | Partial | M | |
| Frameworks and tooling | Test framework version differs between CI and the host | Partial | R | Recorded. |
| Prompt injection and noise | Not applicable | Not established | U | |
| Sensor and data noise | Game rendering noise; a noisy-signal protection in the learning-progress term | Partial | D | |
| Real-world data diversity | One game live; one map | Absent | M | The widest gap in this group. |
| Benchmarks and curriculum design | A pre-registered protocol (ASS-003); no benchmark suite | Partial | R | |
| Open-source community | Public repository | Present | D | |
| Safe exploration | Rewind on stagnation and life boundaries; no constraint on risk | Partial | R | |
| Prompt context / framing | Not applicable | Not established | U | |

### 5.11 Model differences

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Model scale | World model 12.5 M parameters at the smallest size; policy about 1.7-2.4 M | Present | M | |
| Context window size | 32-step windows | Present | R | |
| Hyperparameter configuration | Hand-set | Present | R | |
| Training stability | Monitored | Partial | M | |
| Exploration rate (epsilon) | None; entropy-based | Partial | R | |
| Architecture diversity | One architecture | Absent | R | |
| Training stage (epoch) | A development-stage label ("explore") appears in logs | Partial | M | |
| Gradient clipping | Policy 0.5 | Present | A | |
| Random seed and initialization | A seed can now be set at launch; two seeds tested | Partial | R, M | |
| Negative transfer | Not measured | Not established | U | |
| Data coverage gaps | One map | Present (gap) | M | |
| Baseline model quality | Not established | Not established | U | |
| Training language mix | Not applicable | Not established | U | |

### 5.12 Capabilities

| Map term | Counterpart in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Learned knowledge | Empty knowledge store in Doom | Partial | M | |
| Capability | No exits in any run; Minecraft tree chopping has never been learned | Absent | D, M | CLAUDE.md §9. |
| Default policy | Near uniform over valid actions (entropy 0.87-0.94 of maximum) | Partial | M | |
| Generalization | Unmeasured beyond fresh-data loss | Not established | U | |
| Transfer learning | Unproven; no cross-game run | Not established | M | |
| Specialist model | None | Absent | R | |
| Weight retention | Checkpoint and resume; the whole policy restored in the measured relaunch | Present | M | |
| Policy updates | Present | Present | M | |
| Generative diversity | Unmeasured | Not established | U | |
| Domain adaptation | Unproven | Not established | U | |

### 5.13 Terms the map marks DNE

| Map term | Nearest analogue in SkyBot | Status | Basis | Difference from the ideal |
|---|---|---|---|---|
| Oxytocin (DNE; links to Base model lineage and Multi-agent learning: bonding, trust) | None. No modulator tied to trust, bonding or caregiver signals. | Absent | D | SkyBot has no social-bonding signal; its streams do not interact. |
| Endocannabinoids (DNE; links to weight updates: retrograde, activity-dependent plasticity tuning) | None identified. Plasticity is set by the optimizer and fixed coefficients. | Absent | D | No activity-dependent, retrograde adjustment of how readily a weight changes. |
| Autonomy / competence / relatedness (DNE; feeds Intrinsic reward) | Autonomy: action attribution and empowerment are measured. Competence: a competence predictor exists for skills; learning progress proxies it. Relatedness: none. | Partial (autonomy, competence); Absent (relatedness) | M, R | These are used as separate reward terms, not as a unified need-satisfaction model; relatedness has no meaning with non-interacting streams. |
| Belonging (DNE; links to Peer agents and Objective function) | None. | Absent | R | Peers (the parallel streams) do not observe or influence one another. |
| Learning styles (marked a myth; no evidence for matching teaching to style) | None. | Absent | D | The map itself says there is nothing to build; listed for completeness. |
| Wellbeing (DNE) | None. No measure of the agent's own state of "wellbeing" is computed or optimised. | Absent | D | The map ties this to long-run continual learning; the continual-learning half is in 5.12 and 5.8. |

## 6. Relationship-level differences

The map is also a set of dependencies. The ones below are present in the map
and absent or only partly present in SkyBot. Each arrow is named by the map's
own node ids.

| Map relationship | In SkyBot | Basis |
|---|---|---|
| SLEEP to SWS / REM to SWR to REPL to SYSCON (offline consolidation drives continual learning) | No consolidation loop. Dream training and distillation run each segment, but nothing replays old experience into the policy. | R |
| CONS to SYSCON to LTM, FORG to DECAY (forgetting is managed) | Forgetting is measured only; the policy has no protection. | R, M |
| PFC to EF / META / INHIB (a controller plans, monitors and inhibits) | The developmental loop orchestrates, but no learned planner or metacognitive monitor exists. | R |
| META to JOL to CALIB to SRL (confidence and calibration guide strategy) | None. | U |
| SRL to STRAT_ENTRY to RP / SP / INTL / ELAB (strategy selection) | No strategy selector; one fixed training recipe. | R |
| SCAF to ZPD to MOT, FLOW to MOT (a curriculum keeps tasks at the learnability frontier) | None. Learning progress is computed but not used to choose tasks. | R |
| MAST to PK (mastery updates priors) | Mastery is unmeasured; no prior update path. | D |
| SELFE to PERS, SELFE to SRL (self-evaluation sustains effort and tunes strategy) | Self-evaluation is not wired into effort or strategy. | R |
| ATTRIB to SELFE (credit assignment feeds self-evaluation) | Credit feeds PPO; it does not feed a self-model. | R |
| TEACHER to SCAF / FB (teacher supplies curriculum and feedback) | No teacher (deliberate). | D |
| PEER to OBS (observation of peers) | Streams do not observe one another. | R |
| EMBOD to GEST to ENC (embodiment shapes encoding) | The simulated body's own senses reach the encoder; no embodied robot. | R |
| DMN to REFL / INSIGHT / SYSCON (internal simulation supports reflection and insight) | Imagination serves curiosity and the dream actor, not reflection or consolidation. | R |
| AMY to EMO / CONS / FEAR (a risk module shapes consolidation and caution) | None. | R |
| GCL to SCHEMA (useful gradient builds schemas) | Not instrumented. | U |
| PRED to ATT (prediction error steers attention) | No attention to steer. | R |
| OXY to PARENT / OBS (bonding shapes lineage and observation) | None. | D |
| ENDO to LTP (retrograde tuning of plasticity) | None. | D |
| SDT to INTR (need satisfaction drives intrinsic reward) | Intrinsic reward has separate terms; no need-satisfaction model sits above them. | R |
| BELONG to MOT, PEER to BELONG (belonging shapes motivation) | None. | R |
| INTR / BEHAV to WELL (intrinsic reward and behaviour feed wellbeing) | None; there is no wellbeing node to feed. | D |
| STYLES to TEACH (the map marks this link as unsupported) | None, and none is needed. | D |
| LEARN back to PK, SELFE, LTP (learning reshapes priors, self-model, plasticity) | Weights are updated; priors do not exist, and no self-model is updated. | R |
| STRESS to ANX to WM (noise degrades working memory) | Not modelled; instability is handled by guards. | R |

## 7. Ranked principal differences

Ordered by how much of the map each gap leaves unrealised. Only statements
backed by a basis code above are used.

1. **No durable-skill route in Doom.** Skills cannot be minted without an
   extrinsic reward spike or engine-derived keys; skill discovery is planned
   ([RES-005](res-005-skill-discovery.md)). The knowledge graph and episodic
   record stay empty in Doom. The map's knowledge, capability and transfer
   outcomes therefore have no source.
2. **No consolidation or forgetting protection for the policy.** The map's
   offline phase, replay into the policy and weight consolidation have no
   counterpart; the policy learns on-policy and is not protected across
   relaunches except by restoring it.
3. **No curriculum or learnability frontier.** Nothing chooses what to practise;
   data diversity is one game and one map.
4. **No planning or metacognition at decision time.** Acting is a feed-forward
   policy; there are no confidence estimates, calibration or strategy selection.
5. **A single dominant intrinsic signal.** Episodic novelty is 80-90 % of
   income, learning progress was near zero before the credit fix, and no
   information-gain or competence signal competes with it.
6. **Autoscaling and growth are advice and tests, not a running control loop.**
7. **Evidence of improved play is missing.** The learning machinery measurably
   runs; behaviour did not consistently improve in 120-minute runs.

## 8. Complete inventory of what SkyBot does not have (N/A items excluded)

This section consolidates every absence stated above, so it can be read alone.
Each item names its row; the basis is the one given there. Section 3's items
are deliberate and are marked (D).

**8.1 No counterpart at all (status Absent)**

- Learning and architecture: pretraining (D); learning-rate scheduling for the
  policy; attention in the policy, and long-context or sparse attention;
  architecture diversity.
- Teaching and feedback: behavior cloning (D); imitation learning (D);
  feedback signals (RLHF) (D); a teacher or teacher model (D); few-shot
  examples (D); targeted fine-tuning; prompt priming (D).
- Reasoning and language: a reasoning controller in the measured game;
  chain-of-thought (D); language modeling (D); self-reflection and critique
  (D); self-generated rationales (D); scratchpad notes; context summarization.
- Memory and consolidation: weight consolidation (EWC); spaced replay; offline
  reinforcement learning (the policy never trains from replay); any chunking
  of observations into reusable units in the measured game (no options were
  offered).
- Control and safety: risk penalties; a learned safety and risk module;
  confidence estimates and model calibration; meta-learning; a curriculum;
  zero-shot evaluation; federated learning.
- Data and environment: real-world data diversity (one game live, one map);
  graph neural networks in the measured runs; a specialist model.
- Hardware and body: resilience to power loss beyond interval checkpoints;
  embodied robotics.
- The DNE terms: Oxytocin; Endocannabinoids; Relatedness; Belonging; Learning
  styles (the map marks it a myth); Wellbeing.
- Extrinsic reward in Doom (D).

**8.2 Present only in part (what is missing)**

- Dynamic network expansion: no growth decided by the system itself in a live
  run; the policy-growth path is verified in unit tests only.
- Frozen layers: no phase-based freezing of learning layers.
- Gain and temperature scaling: fixed coefficients, not state-dependent.
- Discount factor: fixed, not adapted to the task.
- Importance weighting: activity of prioritised replay not established.
- Uncertainty estimation: measured, but not driving exploration or caution.
- Reward model: used for imagination only.
- Offline training phase and generative replay: no consolidation phase, and
  imagination does not protect old knowledge.
- Planning: imagination exists but acting does not search it.
- Model-based learning: no model-based action choice.
- Retrieval and knowledge graph: present but empty in Doom.
- Episodic memory: no event-level memory the agent can query.
- Skills: the bank exists but nothing can be minted in Doom.
- Goal-conditioned behaviour: one goal generated, none completed in Doom.
- Self-evaluation: not wired into effort or strategy.
- Information gain: approximated, not computed.
- Autonomy and competence: separate terms with no unified model.
- Exploration temperature: one-time boost, not adapted.
- Convergence criteria: plateau tests only; no mastery criterion.
- Learnability frontier: a proxy that does not select tasks.
- Evaluation and validation: no held-out tasks or environments.
- Overfitting control: measured, mitigation is advice only.
- Multi-agent and peer learning: parallel streams that never interact.
- Concept-drift adaptation: no drift-triggered adaptation.
- Concept formation across games: transfer unproven.
- Safe exploration: no risk constraint.
- Long training runs: the longest controlled run was 120 minutes.

**8.3 Not established**

Rows coded U in the tables are not claimed as absences or presences; they are
listed in section 9.

## 9. Verification backlog (rows coded U)

To turn the "Not established" rows into findings, each needs a direct check
against code or a run. Not done here, by instruction: critical-period
mechanisms, pruning, normalization and activation choices, positional encoding,
positive/negative weight split, model editing, few-shot analogy, contrastive
objectives, scheduled tasks, data augmentation, hard-negative mining,
domain randomization, targeted fine-tuning, zero-shot evaluation, federated
learning, scratchpad notes, context summarization, grokking, drift
sensitivity, negative transfer, continual-learning capacity, baseline model
quality, generalization and transfer measurements, and whether prioritised
replay is active in the measured configuration.

## 10. Limits of this record

- It compares *names and roles*; it does not assess how well any present
  counterpart works. Where a counterpart is present, quality is covered by the
  evidence in ASS-003 and ASS-006 only.
- Codes **A** and **U** mean some rows rest on exploration reports or have no
  basis; the author did not re-read those parts of the code for this record.
- It reflects the working tree and runs as of the date above. Any later change
  to the learning code or the map makes rows stale; update the affected rows in
  the same change.

**Revision note (2026-10-20).** This record was first written excluding DNE
terms. The owner asked for them to be included; sections 1, 5.13, 6 and 8 were
revised accordingly. The original comparison was not otherwise changed.

## 11. Related records

- [`ai_learning.mmd`](../../assets/ai_learning.mmd) and
  [`ai_learning_full.png`](../../assets/ai_learning_full.png): the target map.
- [`skybot-learning-dependencies.mmd`](../../assets/skybot-learning-dependencies.mmd):
  SkyBot's own learning dependencies.
- [ASS-003](../assurance/ass-003-policy-learning.md),
  [ASS-006](../assurance/ass-006-realtime-action-lag-incident.md),
  [RES-001](res-001-central-behavioral-problem-and-plan.md),
  [RES-004](res-004-rssm-size-experiment.md),
  [RES-005](res-005-skill-discovery.md).
- [ADR-001](../decisions/adr-001-shared-universal-brain.md),
  [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md): Adaptability evaluation protocol (the program's main metric)
- [ASS-010](../assurance/ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [ASS-015](../assurance/ass-015-ab00-interim-analysis.md): AB-00 Doom A/A: interim analysis (16 of 20 runs)
- [GOV-008](../governance/gov-008-human-like-learning-motivation.md): Human-like learning program: motivation, scope and change record
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [GOV-010](../governance/gov-010-relational-implementation-plan.md): Relational implementation plan: connecting the learning blocks as the map does
- [GOV-011](../governance/gov-011-execution-record-gov-009.md): Execution record: block implementation plan (GOV-009)
- [RES-007](res-007-hard-and-deferred-differences.md): Hard and deferred differences: what cannot yet be built reliably
- [RES-008](res-008-relationship-gap-analysis.md): Relationship gap analysis: SkyBot's learning connections against the AI-counterpart map
- [RES-009](res-009-hard-and-deferred-relationships.md): Hard and deferred relationships: lines that cannot yet be built reliably
<!-- backlinks:end -->
