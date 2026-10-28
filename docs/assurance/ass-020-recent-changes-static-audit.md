# Recent changes: static implementation and documentation audit

> **Document control:** [ASS-020](../README.md#assurance) · Assurance / audit record · **State:** Record; static findings, runtime unverified · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-25. **Baseline:** `f4164b4` (2026-10-20), compared with the current staged, unstaged and untracked working tree. **Purpose:** assess whether the recent implementation should operate, identify errors, and distinguish implemented capacity from research claims. **Method:** source and diff inspection only; no tests, training, benchmarks, remote inspection or deployments. Documentation generation and link validation are separate from application testing.

## 1. Assessment and limits

**Some changes should operate, but this tree cannot be endorsed as correct as a whole.** The default RSSM path remains available, the renamed trainer adds the repository root to its import path, and the PC core has a coherent constructor, observation, loss, local-update and checkpoint-buffer path. These are reasonable implementation foundations, not proof of execution or better learning.

The highest-priority problems are in the machinery deciding whether the PC experiment is valid: a failed prerequisite suite does not block its comparison, and the open-loop metric feeds the wrong action for each predicted frame. Optional consolidation also has an incorrect bootstrap, and PC probes can mutate model state. Resolve these before interpreting affected experimental results.

This is a risk-focused audit of a large, mixed working tree, **not a line-by-line certification of every added module**. Detailed review covered world-model/PC integration, the PC pilot and its campaign dependencies, replay/consolidation, measurement callers, growth restrictions, trainer launch migration and documentation control. Other new learning and relation modules were inventoried or sampled; their combined behavior remains unverified. Historical host results are not evidence that this exact working tree passes.

Severity: **P1** threatens experiment validity or required gating; **P2** is a conditional correctness/operational defect; **P3** is documentation debt. Findings below are source-supported unless explicitly called a risk. No runtime failure was reproduced.

## 2. Findings

### F-01 — P1: failed PC suites do not gate the pilot or A/B

**Evidence:** `tools/campaign.py:180–201` defines `pc_tests` with `on_fail="continue"`. `pc_boot_check` requires only `pc_snapshot`; `pc_pilot` requires the snapshot and boot check; `t1_wp20_pc_core` requires the pilot and boot check. The later generic boot-check dependency does not add `pc_tests`. The scheduler's dependency logic is at `tools/campaign.py:465–474`.

**Reasoning:** stage ordering is not a success dependency. If a PC/config/growth suite fails but the short boot and pilot succeed, the five-seed comparison remains eligible. A failed implementation check can therefore coexist with an apparently authorized experimental result.

**Correction:** make the pilot and/or its boot stage depend on successful `pc_tests`, with a transitive dependency reaching the A/B. Preserve failure records rather than treating the suites as informational. This is especially important because the PC change is explicitly unverified.

### F-02 — P1: the PC pilot inherits an off-by-one action in open-loop evaluation

**Evidence:** `tools/rssm_size_experiment.py`, `collect()` stores observation `o_t` with action `a_t` before stepping. `build()` enables causal alignment by default. `WorldModel.observe_sequence()` (`developmental_ai/world_model/rssm.py:1912`) explicitly defines `imagine_step(state_t, a_t)` as predicting `o_(t+1)`. However, `evaluate()` (`tools/rssm_size_experiment.py:207–214`) starts from state `context-1` and first feeds action `context`, then compares its prediction with observation `context`.

**Reasoning:** the first imagined transition needs action `context-1`; every subsequent action is likewise one too late. Sticky random actions can conceal the error during repeated-button periods, but action changes measure the wrong transition. Both arms being misaligned does not cancel it: models with different action sensitivity can be penalized differently. The error predates the new pilot, but the new G2/G3 gates now inherit it.

**Correction:** use action index `context-1+k` for the prediction compared with frame `context+k` under the causal convention. Treat affected open-loop gate results as requiring recalculation; do not repair a frozen result silently.

### F-03 — P2: AWR bootstraps from the current state's value at truncation

**Evidence:** `developmental_ai/policy/consolidation.py`, `Consolidator._awr_targets()`, sets `boot = v` at the last sampled row and when the following return is unavailable. Here `v` is the stored value of the current decision, not the successor state. `_awr_loss()` accepts finite targets produced this way.

**Trigger:** consolidation with AWR enabled, on a nonterminal sampled-window boundary or a gap in available policy rewards/values.

**Reasoning:** the target becomes `r_t + gamma * V(s_t)` where it should bootstrap from `V(s_(t+1))`. A rising value trajectory is underestimated and a falling one overestimated. A missing successor can become an accepted training target despite the method's claim that unknown chains are invalid. Terminal rows with continuation zero are unaffected by this specific error.

**Correction:** provide a real successor value, sample one extra transition and train on the preceding rows, or reject boundaries whose successor cannot be evaluated. Retain explicit terminal handling. This does not affect default-off AWR.

### F-04 — P2: an observational probe changes the PC precision state

**Evidence:** `developmental_ai/core/curriculum.py:298–331`, `measure_streams()`, calls `world_model.compute_loss()` under `torch.no_grad()` without switching to evaluation mode. `developmental_ai/world_model/predictive_coding.py:262–275`, `generative_loss()`, updates the persistent `obs_var` whenever `self.training` is true. The loop reuses `measure_streams()` for curriculum, competence and adaptation phases.

**Trigger:** PC is selected and a measurement caller uses a model still in training mode.

**Reasoning:** disabling gradients does not disable this explicit buffer write. Reading diagnostics changes the precision used by later settling and learning. Enabling another probe consumer can change PC behavior even if that consumer makes no control decision. This undermines the meaning of observational relationships and held-out measurement.

**Correction:** give measurement an explicitly read-only loss path, or save/restore training mode and use evaluation mode in a `try/finally` block under the existing model lock. Audit other stateful readouts too; restoring just this scalar is a narrow workaround.

### F-05 — P2: the backprop ablation also changes network architecture

**Evidence:** `developmental_ai/world_model/predictive_coding.py:165–173`: the Hebbian arm replaces the prior, posterior and prediction networks with `LocalPCMLP`. That class is linear–ELU–linear, without LayerNorm. The backprop arm retains inherited RSSM prior/posterior networks and creates a prediction network containing LayerNorm. `tools/pc_pilot.py` describes the ablation as separating iterative inference from local learning.

**Reasoning:** the ablation changes normalization/architecture as well as the weight-update rule. A difference in results cannot be attributed solely to Hebbian versus backprop learning. The whole-core PC/RSSM comparison is still a legitimate package comparison, subject to F-01/F-02; the narrower causal interpretation is unsupported.

**Correction:** implement a backprop version of the same layer topology and initialization, or describe this as a comparison of two implementations rather than an isolated learning-rule ablation. Preserve the original protocol and disclose the difference separately.

### F-06 — P2: launcher singleton protection misses the frozen old trainer

**Evidence:** `scripts/launch_skybot.sh:73` refuses only a process matching `scripts/run[.]py`. `tools/campaign.py:66` deliberately matches both `(run_minecraft|scripts/run)[.]py`. The current `CLAUDE.md` entry-point migration note says the frozen host campaign still uses the old name.

**Trigger:** the updated launcher is used while an old-name trainer is running, and no other preflight independently prevents launch.

**Reasoning:** the explicit trainer guard passes even though a trainer exists. Other game-process checks may sometimes stop the launch, but they do not restore the missing ownership guarantee. Concurrent trainers can contend for compute and shared output/checkpoint paths. This is a migration risk, not a claim that two trainers are currently running.

**Correction:** retain both entry-point names during migration, or use a shared process lock/PID ownership mechanism across both launch paths.

### F-07 — P3: current guidance contains stale or over-broad claims

* `docs/operations/ops-002-game-server.md:214` recommends `configs/minecraft_lifelong.yaml`, which is absent. The retired variant is under `legacy/`; it should not be offered as an active launch procedure. Plain code-formatted paths escape the Markdown link checker.
* `CLAUDE.md` retains a dated build-only statement that nothing has been tested alongside later campaign records. Historical dates help, but the current operational summary should explicitly supersede that blanket statement while preserving the history.
* The root README still leads with a Minecraft-only description, while the branch authority specifies the universal brain and deferred Minecraft return. It needs a short current-scope note, with historical project motivation retained as such.
* The PC file describes fixed-point local updates as equaling backpropagation. Its own implementation notes more carefully describe an approximation as hidden precision grows. With six finite relaxation iterations and finite precision, exact equality to the feed-forward backprop gradient is not established by this implementation. Qualify the stronger wording.

These are distinct from broken Markdown links. A structurally valid document can still prescribe an absent file or overstate a mathematical/operational claim.

## 3. What should work, and what capacity actually exists

| Area | Static assessment | Practical boundary |
|---|---|---|
| Default RSSM training | Existing core retained; new constructor arguments default to RSSM. Main-loop wiring reaches both parallel collection bodies. | No boot, regression or numerical equivalence was established here. |
| PC observation and learning | Iterative posterior inference, straight-through connection to the amortised path, a generative term and post-backward local updates are wired. Learned buffer weights participate in state dictionaries. | This is a hybrid: encoder, recurrent dynamics, decoder and other heads still use backprop. It does not establish human-like learning or better transfer. |
| Nonfinite world-model gradient guard | Refuses the optimizer step and clears PC collected samples when the gradient norm is nonfinite. | This is not a transaction over every forward-pass buffer mutation, nor a guarantee that every auxiliary optimizer is protected. |
| Replay policy columns | Optional allocation, save/load and NaN-for-unknown behavior are present. | Mixed policy reward is written only for stream 0; scout metadata alone does not supply complete scout AWR targets. F-03 remains. |
| Planning | Learning-phase planning and optional policy distillation exist. | Acting-time planning is explicitly not wired; turning on its setting does not provide an online planner. |
| Peer observation | Restricted view/action context and a prediction subsystem exist. | Peer prediction is not evidence of shared-representation learning; sound is not carried in this build. |
| Compute adaptation | Model/stream ordering and bounded advice exist. PC growth is explicitly rejected rather than silently attempting an unsupported transform. | Stepping-rate actuation is absent. A PC run cannot assume the RSSM grow/shrink path is available. |
| Evaluation | Separate evaluator modules, paired-seed analysis and explicit no-claim conditions are useful foundations. | No complete end-to-end isolation audit or current campaign-state verification was performed; F-01/F-02 block confidence in the new PC comparison. |

**Compute inference:** PC performs additional energy evaluations and reverse differentiation with respect to latent logits at each observed timestep, even inside outer `no_grad` contexts. With the default four settling iterations and sequence length T, it executes roughly 4T such derivative calculations per observation sequence, plus final energy readouts and local learning. This is extra sequential work, not a claim of a fourfold total slowdown. Decoder/encoder costs, batching, device and tensor sizes determine the total.

The Hebbian path avoids Adam moments for its buffer-held weights, but still retains ordinary model autograd and cached local-learning inputs. `hebbian_rows=2048` caps rows used for updates only **after** all cached rows are concatenated (`predictive_coding.py:300–308`); it is not a peak-memory cap. The pilot counts buffer weights, which is appropriate for capacity comparison. Its acting timing covers perception for two streams, not complete environment/policy/rendering latency. No trustworthy FPS, maximum stream count, VRAM fit or throughput gain can be inferred numerically from this review.

## 4. Disposition

1. Repair the campaign dependency and causal evaluation indexing before relying on PC gate results.
2. Repair AWR successor targets and read-only PC measurements before enabling the affected combinations.
3. Resolve the ablation interpretation and dual-entry-point migration guard.
4. Update current operational summaries and absent-file instructions while preserving historical protocols/results.
5. When the owner authorizes execution separately, obtain runtime evidence on the intended host. Nothing in this report constitutes a passed test or a promotion decision.

This audit changes documentation only. It does not apply the proposed code corrections or alter existing experimental results.

## 6. Resolution record (2026-10-25, appended; the findings above are unchanged)

Each finding was re-checked against the code before any change. Corrections are in [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 11b; the protocol disclosure is [ASS-019](ass-019-predictive-coding-core-comparison-protocol.md) section 5a.

| Finding | Confirmed | Disposition |
|---|---|---|
| F-01 | Yes | Fixed in part. `pc_tests` now holds the required suites (predictive-coding, pilot-gate, config-key) and the boot check and pilot require it. The wider list moved to `pc_tests_info`, required by nothing: making every suite a prerequisite would recreate the guard-becomes-latch pattern (CLAUDE.md section 4.1) because some of them can be red for unrelated reasons |
| F-02 | Yes: `a[:, t]` is the action taken at frame `t`, so predicting frame `context + k` needs `a[:, context - 1 + k]` | Fixed in `tools/rssm_size_experiment.py`; regression test with a recording fake model. The older [RES-004](../research/res-004-rssm-size-experiment.md) sweep used the old pairing; not recalculated |
| F-03 | Yes, minor (last row of a window; AWR is off by default) | **Open**, deliberately not changed here |
| F-04 | Yes | Fixed: the precision moves only on a gradient-enabled training pass; regression test. Counters (`refused`) still change on a probe; they are telemetry and do not feed behaviour |
| F-05 | Yes | Fixed for topology and initialisation (`PlainMLP`); the optimiser difference is disclosed |
| F-06 | Yes | Fixed: the launcher guard, `eval_run.sh`, `policy_learning_arms.sh` and `deploy_skybot.sh` accept both entry-point names; contract E in `_clean_exit_smoke` |
| F-07 | Yes | [OPS-002](../operations/ops-002-game-server.md) no longer offers the absent config (it points at the archived file and says it is history); the repository README has a current-scope note; CLAUDE.md's build-only status is marked superseded; the "equals backprop" wording is qualified in the code, [RES-010](../research/res-010-predictive-coding-core.md) and [GOV-016](../governance/gov-016-predictive-coding-core-plan.md) |

**Not verified at runtime.** These are source corrections and new tests; none has run (tests are queued on `main`, owner instruction 2026-10-23).

## 5. Related records

- [ASS-019 — PC comparison protocol](ass-019-predictive-coding-core-comparison-protocol.md): experiment whose measurement and gating are affected.
- [ASS-018 — historical campaign progress](ass-018-campaign-progress-2026-10-23.md): prior host evidence, not verification of this snapshot.
- [GOV-016 — PC implementation plan](../governance/gov-016-predictive-coding-core-plan.md): implementation intent and pilot criteria.
- [OPS-002 — game-server operations](../operations/ops-002-game-server.md): stale operational path cited above.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md): Corrective plan for the categorical-harness findings and the host contract-tier failures
<!-- backlinks:end -->
