# Implementation plan: predictive-coding core versus the RSSM (WP-20)

> **Document control:** [GOV-016](../README.md#governance) · Governance / implementation plan and execution record · **State:** Owner instruction 2026-10-25, revised the same day to local (Hebbian) learning (section 11a); built, **nothing run yet**; tests queued on `main` (section 7) · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-25. **Parents:** [RES-010](../research/res-010-predictive-coding-core.md) (motivation, design, hypotheses),
[ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md) (both cores stay), [GOV-009](gov-009-implementation-plan.md) (one package, one difference),
[ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md) (the protocol). **Authority:** the owner, 2026-10-25: build the
predictive-coding core, keep the RSSM beside it until the tests return, make the pilot a gate, write the documents, and add the tests to the queue.

## 1. Scope

One difference: `world_model.core: pc` against `rssm`. Everything else in an arm is identical. It is a **block** test (T1) in the [GOV-009](gov-009-implementation-plan.md) sense, called WP-20.
It adds code behind a default-off switch and queues tests; it changes no default, no existing config and no frozen code.

## 2. What was built, and the problem each part answers

| # | Change | File | Why |
|---|---|---|---|
| 1 | `PredictiveCodingCore(RSSM)`: amortised guess, then `pc_iters` energy-descent steps with an estimated precision; refine-by-stop-gradient so the value is settled and the gradient is the amortised path's; per-sample refusal of a non-finite step; `generative_loss`; readouts. **Local learning (section 11a):** `LocalPCMLP` (weights are buffers) learns `prior_net`, `posterior_net` and `obs_pred_net` by error x activity updates, collected during the loss pass and applied once after each optimiser step (`begin_collect`, `hebbian_update`); clipped and finite-guarded; `pc_learning: backprop` keeps the first version as an ablation | `world_model/predictive_coding.py` (new) | The thing under test |
| 2 | `WorldModel(core, pc_iters, pc_step, pc_weight, pc_prior_precision, pc_learning, pc_hebbian_lr, pc_hebbian_iters)`; builds the chosen core; adds the `pc` loss term only when the core has one; starts the local collection around the loss pass and runs the local update after a finite optimiser step (drops the collection on a refused step); unknown core or learning mode refused by name | `world_model/rssm.py` | The switch; with `core="rssm"` the loss dict, total and train step are exactly the old ones |
| 3 | Reads the eight keys and passes them | `core/developmental_loop.py` | The loop builds the world model |
| 4 | Defaults `core: rssm`, `pc_learning: hebbian` and the numbers | `configs/universal.yaml` | `tools/campaign_arms.py` validates arm keys against this file (a typo would be a silent no-op arm) |
| 5 | Growth refuses a non-RSSM core by name | `world_model/growth.py` | The growth rules know only the RSSM's columns; a guess would drop the generative head |
| 6 | `build(core, pc)` and `train(cores, pc)`; `--cores rssm,pc,pc_bp`; results keyed `size:core` for a comparison and unchanged for an RSSM-only run; PC refusals recorded; **parameter counts add the locally learned buffers** (otherwise G0 would flatter the PC core) | `tools/rssm_size_experiment.py` | The pilot reuses the harness that measured size in [RES-004](../research/res-004-rssm-size-experiment.md) |
| 7 | The pilot and its pre-registered gate (`verdict()` is a pure function of the results file); it runs `rssm`, `pc` (the gated candidate) and `pc_bp` (reported, never gated) | `tools/pc_pilot.py` (new) | Section 5 |
| 8 | The WP-20 arm and `OWN_CODE` (runs on its own snapshot, with its own baseline) | `tools/campaign_arms.py` | A boot check or an A/B on the frozen copy would run the old model: an A/A in disguise |
| 9 | Stages `pc_snapshot`, `pc_tests`, `pc_boot_check`, `pc_pilot`, `t1_wp20_pc_core`, `an_t1_wp20_pc_core`; `ev()` takes a code root; WP-20 is skipped by the shared-baseline loop and by the frozen-code boot check | `tools/campaign.py` | Section 7 |
| 10 | Suites `_predictive_coding_smoke`, `_pc_pilot_gate_smoke`, registered in `tests/run_all.py` | `tests/` | Section 6 |
| 11 | **Found while doing this (before the audit):** process-match patterns written with a bracket (`"[r]un_minecraft.py --config ..."`) in `tools/eval_run.sh`, `tools/policy_learning_arms.sh` and `scripts/deploy_skybot.sh` were missed by the entry-point rename (a plain search for the old name does not see them) and would have made every eval on a renamed snapshot think the trainer never started. Now `scripts/[r]un.py`. The frozen copy is unchanged and still correct for itself | those three files | [GOV-015](gov-015-corrective-plan-categorical-findings.md) section 9.5 rename |
| 12 | The open-loop evaluation feeds `a[:, context - 1 + k]`, not `a[:, context + k]`, to predict frame `context + k` ([ASS-020](../assurance/ass-020-recent-changes-static-audit.md) F-02) | `tools/rssm_size_experiment.py` | The corrected pairing follows the causal convention of `observe_sequence`; the old one scored every prediction one frame late and fed the pilot's G2 and G3. It also affects the older [RES-004](../research/res-004-rssm-size-experiment.md) sweep; those numbers are not recalculated here |
| 13 | The PC precision moves only on a gradient-enabled training pass (F-04) | `world_model/predictive_coding.py` | A measurement (curriculum, competence, adaptation) runs the loss under `no_grad`, possibly in training mode, and changed the precision |
| 14 | The backprop ablation now builds the same three networks (`PlainMLP`: linear, ELU, linear, the same initial values) as ordinary parameters (F-05) | `world_model/predictive_coding.py` | The first ablation also changed the layer norms, so a difference could not be attributed to the learning rule. It still differs in the optimiser (Adam) |
| 15 | `pc_tests` (required) and `pc_tests_info` (informational) replace the single `pc_tests`; the boot check and the pilot require `pc_tests` (F-01) | `tools/campaign.py` | A failed implementation suite no longer coexists with an authorised comparison |
| 16 | The launcher's one-trainer guard, `eval_run.sh`, `policy_learning_arms.sh` and `deploy_skybot.sh` accept both entry-point names (F-06) | `scripts/`, `tools/` | The frozen copy on `main` keeps the old name; a guard blind to it is a second trainer |

## 3. Why the design keeps the RSSM's interface

The policy, replay, every head, the autoscaler's surprise readout and the loop read only `{h, z}`, the latent width and the prior-divergence of logits.
Subclassing keeps all of it, so the comparison tests the inference rule and not a rewrite. Alternatives considered and not chosen: a from-scratch PC network with
local learning (largest, riskiest, confounded; deferred, [RES-010](../research/res-010-predictive-coding-core.md) section 5); precision weighting only as an auxiliary (too small to count as a swap).

## 4. Where the code runs

The frozen campaign copy (`<root>/code`) is **not touched**. `pc_snapshot` copies the staged tree (`gov015_code`, which carries the [GOV-015](gov-015-corrective-plan-categorical-findings.md) fixes and the PC core)
to `<root>/pc_code` when it starts, hashes it, and every WP-20 stage runs on that copy. The A/B has **its own baseline arm on the same snapshot**, so it compares one difference;
`ab_baseline` (frozen code, without the [GOV-015](gov-015-corrective-plan-categorical-findings.md) fixes) is not used for it.

## 5. The pilot gate (written before any result)

On fixed Doom data (12,000 frames of the real adapter), tiny size, 3,000 steps, the same batches in the same order, the real `WorldModel` with the Doom run's composed settings,
held-out open-loop evaluation, both cores. The PC core passes only if **all** of:

| Gate | Condition |
|---|---|
| G0 | parameters(pc) <= 1.25 x parameters(rssm) |
| G1 | no divergence; every logged value finite (settle refusals are reported) |
| G2 | final open-loop error beats "the frame never changes" (relative error < 1.0) |
| G3 | final open-loop error <= 1.25 x the RSSM's, and reconstruction <= 1.25 x the RSSM's |
| G4 | seconds per training step <= 3 x, acting time <= 3 x, peak VRAM <= 1.25 x the RSSM's |

`pc_pilot` exits 1 on a fail, so `t1_wp20_pc_core` (which requires it) is **blocked**, and the campaign continues. A pass says only that the core is worth a five-seed test.

## 6. Tests added

- `tests/_predictive_coding_smoke.py`: A default unchanged, unknown core or learning mode refused; B same interface, no new parameters (the weights are buffers and are counted by `local_values()`), and the backprop ablation's extra parameters are the generative head only; C the backprop ablation has the Hebbian core's exact topology, initial weights and outputs, and no LayerNorm; M a no_grad loss pass in training mode moves neither the precision nor the local weights; D settling moves the logits and lowers the energy; E the gradient reaches the observation through the local nets, the value is the settled belief, and the local nets hold no parameters and get no autograd gradient; F a non-finite settle step is refused and counted; G it trains: the generative error falls, the local weights move and stay finite, the precision moves, the backprop ablation trains with the optimiser; **H the local update matches minus the autograd gradient (hidden-layer cosine > 0.95, output > 0.995), the hidden layer learns nothing without the relaxation (falsifier), and the softmax output layer's update equals minus the cross-entropy gradient exactly**; I the local rule learns a regression and lr=0 changes nothing; J a non-finite target is dropped and counted, a huge target moves no matrix by more than `clip` of its norm; K state round-trips and growth refuses; L settling-disabled falsifier.
- `tests/_pc_pilot_gate_smoke.py` (also: the required and informational suites are split and the A/B reaches the required ones through the pilot; the open-loop evaluation is causally aligned, checked with a fake model that records the actions it is given, with the old pairing as a falsifier): the verdict passes a good result, reports the backprop ablation without gating it, each gate can fail alone, a missing or diverged or NaN run never passes; the queue obeys the gate, the A/B runs both arms on the snapshot and not on `ab_baseline`, the frozen-code boot check excludes the arm; the arm's keys exist in `configs/universal.yaml`.

## 7. Queue (all on `main`; nothing is run on the development machine)

| Stage | What | Est. h |
|---|---|---|
| `pc_snapshot` | copy and hash the staged tree | 0.05 |
| `pc_tests` | the **required** suites: `_predictive_coding`, `_pc_pilot_gate`, `_config_keys`. The boot check and the pilot wait for it ([ASS-020](../assurance/ass-020-recent-changes-static-audit.md) F-01) | 0.1 |
| `pc_tests_info` | the unit tier, `_campaign`, `_clean_exit`, `_incident_bundle`, `_growth`: recorded, **required by nothing** (an unrelated red suite must not latch the experiment shut, CLAUDE.md section 4.1) | 0.1 |
| `pc_boot_check` | boot the PC arm 4 minutes on the snapshot (a traceback blocks the A/B) | 0.1 |
| `pc_pilot` | the gate (three runs: `rssm`, `pc`, `pc_bp`) | 1.6 |
| `t1_wp20_pc_core` | **both arms** (RSSM baseline and PC), five seeds each, one hour each, on the snapshot | 10.0 |
| `an_t1_wp20_pc_core` | score and compare | 0 |

About 2.0 h to the gate; 10 h more only if it passes. The stages sit after `gov015_rerun` and before the remaining T1 list, so the answer arrives in days, not weeks; reorder in `tools/campaign.py` if the owner prefers.

## 8. Predictions (written before the run)

- **P-1.** `_predictive_coding_smoke` and `_pc_pilot_gate_smoke` pass on `main` on the first run (an unrun build usually has a defect; a failure here is a build fix, not a finding).
- **P-2.** The Hebbian PC core trains without divergence (G1) but is **less likely than the first version to pass G2 and G3**: its posterior can no longer be shaped by the decoder's gradient, and local learning is usually slower. `pc_bp` in the same pilot says whether a shortfall is local learning or the settling.
- **P-3.** G4 is the gate most likely to fail: `pc_iters` extra gradient evaluations per perception step on an acting-latency-limited host.
- **P-4.** If the gate passes, the A/B is **more likely null than positive** for adaptability: the world model's size has not mattered at this stage ([RES-004](../research/res-004-rssm-size-experiment.md)), so a different inference rule may not either. A null result is still informative.

## 9. Open owner decisions

1. Landing the winning core (or the [GOV-015](gov-015-corrective-plan-categorical-findings.md) fixes) in the frozen experiment code needs a re-staged `ab_baseline` ([GOV-015](gov-015-corrective-plan-categorical-findings.md) section 5.6).
2. Whether a positive result should be followed by the `pc_iters = 0` ablation (auxiliary alone) and by local learning as a separate package.
3. Queue position of the A/B (section 7).

## 10. Execution record (2026-10-25)

Built in the development tree and staged to `main`. **Nothing has been run**: the syntax of every changed file was checked and the stage list was built and parsed, no suite was executed (owner instruction: tests are queued on `main`).
The staged tree was synced to `<campaign root>/gov015_code` and the six WP-20 stages were accepted by the driver (code snapshot `ee987bdf8479`; the running T1 seed was re-attached, not restarted). The campaign estimate went from 148.6 to 160.1 host-hours (the WP-20 chain adds 11.5 h, of which 10 h is the A/B that the gate can cancel). The WP-20 snapshot itself (`pc_code`) is taken when `pc_snapshot` starts, so any fix made to the staged tree before then is included.
After the owner's correction to local learning (section 11a) the staged tree was re-synced and the driver reloaded (snapshot `36a56366a2c1`; the pilot now estimates 1.6 h). Still nothing run.
Results are appended here as sections 11 onward, not edited into the plan.

## 11a. The learning rule is Hebbian (owner correction, 2026-10-25)

The first build kept backpropagation for the core's weights. The owner pointed out that predictive learning, not backprop, was the request. The core's three networks (prior, posterior, top-down) now learn by local
error x activity updates (section 2, item 1; [RES-010](../research/res-010-predictive-coding-core.md) section 2). **Limits, stated plainly:** the encoder, the GRU, the decoder and every other head still use backpropagation;
making them local is a separate, larger package ([RES-010](../research/res-010-predictive-coding-core.md) section 5), and no choice about it is made here. The first version remains as the `pc_bp` ablation so a result can be attributed to local learning or to iterative inference.
The pilot's gate, its numbers and its pass conditions are unchanged. Nothing in this section has been run.

## 11b. Audit corrections (2026-10-25)

[ASS-020](../assurance/ass-020-recent-changes-static-audit.md) is a source-only audit of the recent changes. Each finding was checked against the code. The corrections that touch this plan were made **before any WP-20 stage ran**,
so no result is affected and the pre-registered gate in section 5 keeps its numbers; its measurement is now correct.

| Finding | Disposition | Where |
|---|---|---|
| F-01 | Fixed in part: the required suites gate the pilot; the wider list is informational and gates nothing (the audit's wording would have made every suite required, which can latch) | section 2 item 15, section 7 |
| F-02 | Fixed; regression test with a fake model | item 12 |
| F-04 | Fixed; regression test | item 13 |
| F-05 | Fixed (topology and initialisation); the optimiser difference remains and is disclosed in [ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md) | item 14 |
| F-06 | Fixed; contract E in `_clean_exit_smoke` | item 16 |
| F-03 | **Not changed** (AWR is off by default and the effect is on the last row of a window); open | [ASS-020](../assurance/ass-020-recent-changes-static-audit.md) section 6 |
| F-07 | Documentation items fixed ([OPS-002](../operations/ops-002-game-server.md) path, README current-scope note, CLAUDE.md status supersession, the "equals backprop" wording); see [ASS-020](../assurance/ass-020-recent-changes-static-audit.md) section 6 | [ASS-020](../assurance/ass-020-recent-changes-static-audit.md) section 6 |

## 11c. First run: the required suite failed and the gate held (2026-10-26)

The WP-20 chain ran for the first time on `main` at 02:44 UTC (snapshot `pc_snapshot`, done). Facts, from `stages/pc_tests/stage.log`:

- **`pc_tests` failed (exit 1).** `_pc_pilot_gate_smoke` and `_config_keys_smoke` passed. `_predictive_coding_smoke` passed contracts A-F (21 checks) and **failed contract G, check 22: "the generative error falls on a fixed batch (0.0379 -> 1.5309)"**. The error rose about 40-fold over 41 steps; nothing else in the suite ran after it.
- **The gate worked as written.** `pc_boot_check` and `pc_pilot` are blocked on `pc_tests`, so `t1_wp20_pc_core` is blocked too. Nothing was bypassed and no number in section 5 has changed. The pilot has therefore **not run: there is no pilot result yet.**
- `pc_tests_info` passed (exit 0): the unit tier, the campaign, entry-point, incident-bundle and growth suites are green on the snapshot, so the F-06 and F-01 changes did not break them.

**What the failure does and does not show.** It is a real finding about the build, not about the hypotheses H-1..H-4 (which are about the pilot and the A/B). Two explanations fit the number and predict different things:

| Explanation | What it predicts | Status |
|---|---|---|
| H-drift: the head's target is the encoder's embedding, which the optimiser changes in the same steps, so "a fixed batch" is not a fixed problem | the embedding's scale grows; the backprop ablation shows the same rise in raw error; error relative to the embedding's size does not rise | **untested** |
| H-rule: the local rule does not converge (for example a step normalised to the clip never settles) | an isolated network given a truly fixed target does not converge either; the rise is larger than the ablation's and is present in relative terms | **untested** |

I favour H-drift on reading the code (the embedding is detached but not frozen, and Adam at 3e-3 moves a 32-wide encoder a long way in 41 steps), but that is a reading, not a measurement, and contract G stays as written until a measurement says what it should have asked.

**Queued (all on `main`, none run on the development machine).** `pc_hebbian_diag` (`tools/pc_hebbian_diag.py`, about 6 minutes, runs on the staged tree, required by nothing, cannot pause the campaign). Part 1 trains an isolated `LocalPCMLP` on a fixed target over a grid of learning rates and clips, beside the same topology under Adam. Part 2 repeats contract G's own run for the Hebbian core (learning rate 0.05 and the 0.01 default) and the backprop ablation, logging per step the raw error, the embedding's mean square, the relative error and the weight norms. The queue contract in `_pc_pilot_gate_smoke` D now asserts the diagnostic is required by nothing. The driver snapshot was accepted (924237a08a35 to 70eedc47798d) and the service restarted; the running A/B was re-attached, not restarted.

**Next, in order, decided by the diagnostic.** (1) If H-drift: change contract G to compare the head's error against the embedding's own size and against the ablation under the same drift, add an isolated-network convergence contract, and say so in [ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md) section 5a as a revision made after a failure, with the failing numbers quoted. (2) If H-rule: fix the rule (candidates: a clip that shrinks the step as the error falls; a smaller default rate) and keep contract G as is. (3) Either way a new snapshot and a re-run of the required suites precede the boot check and the pilot; the gate in section 5 is not touched. Which contract changes is recorded here with the diagnostic's numbers before any pilot number exists.

**Candidate fixes explored 2026-10-26 (none applied; the diagnostic decides).** Reading `LocalPCMLP.learn`: the update is scaled so that no weight matrix moves by more than `clip` (5 %) of its norm in one step, and that scaling saturates whenever the raw step is larger, so near an optimum the step does not shrink with the error. Two candidates follow, each tied to one diagnostic outcome. If the isolated runs show the rule failing to settle on a fixed target: replace the saturating clip by a step that shrinks with the error (the clip kept only as a ceiling), or lower the default rate. If they show the rule converging and the in-model rise matching the ablation's: change contract G to measure error relative to the embedding's own mean square, and add an isolated-convergence contract, as in the table above. Neither is built, because choosing before the measurement would be fitting the test to the failure.

### 11c.1 Diagnostic result and the change to contract G (2026-10-26, `pc_hebbian_diag`, rc 0)

The diagnostic ran on `main` (stage `pc_hebbian_diag`, `stages/pc_hebbian_diag/diag.json`). It separates the two hypotheses of the table above.

- **Isolated rule (Part 1):** on a fixed regression the Hebbian head's relative error falls monotonically at every setting tried (lr 0.05: 1.209 to 0.042; lr 0.2: 0.008); Adam reaches 0.002 at lr 0.01. The clip made no difference. The local rule converges; it is slower than Adam at equal step size.
- **In the model (Part 2, contract G's run, 41 steps):** the embedding the head predicts grew from a mean square of 0.0016 to 75.8 (lr 0.05) while the head's error **relative to it** fell from 23.7 to 0.020; at the default lr 0.01 the relative error fell to 0.076. **The backprop ablation's raw error rose too (0.0379 to 28.59)**, so the rise is not a property of the local rule.
- **Reading:** H-drift holds, H-rule does not. The raw error is measured against a target that the encoder is still scaling up, so it is the wrong quantity. Nothing in the predictive-coding code was changed.
- **Change made (a test change, stated as one):** contract G in `tests/_predictive_coding_smoke.py` now requires the relative error (raw error divided by the embedding's mean square at that step) to fall below half its first value; it still prints the raw numbers. The weights-move, finite, no-refusal, variance and backprop-ablation checks are unchanged. The corrected suite passes on `main` (relative 23.701 to 0.020).
- **Caution:** the change makes the contract pass on a result the first contract called a failure, so the pilot gate (G0-G4, section 5) is the real test and is unchanged. One consequence the pilot must show is not hidden by this: the encoder's embedding scale grows by about 4 orders of magnitude over 41 steps in the model, so whether the pilot's fixed-data comparison is meaningful depends on G0-G4, not on this contract. If the embedding scale keeps growing in the pilot, record it as a finding.
- **Queue correction (same day):** the first `resume pc_tests` did not persist (see [GOV-015](gov-015-corrective-plan-categorical-findings.md) section 9.8); it was redone with the driver stopped and read back. `pc_tests` runs after `t1_wp7_attention`, then `pc_boot_check` and `pc_pilot`. No separate runs were added: the pilot already measures the RSSM, the Hebbian core and the backprop ablation on one fixed dataset, and the A/B shares one snapshot for both arms.
- **Re-run:** `pc_snapshot`'s tree was refreshed by hand with the stage's own command (the stage was already done and could not be resumed), `pc_tests` was resumed, and the chain `pc_boot_check` then `pc_pilot` then `t1_wp20_pc_core` follows if it passes. A 10 second check of the corrected suite was run directly on `main`.

## 11d. Second run (2026-10-27): the required suites passed and the pilot gate FAILED

Read from `stages/pc_pilot/verdict.json` and `results.json` on `main` (snapshot `6914fff6f146`, tiny, 12,000 frames, 3,000 steps, `pc_hebbian_lr` 0.01, `pc_iters` 4, `pc_step` 0.5). The gate of section 5 is unchanged and was applied as written.

| Gate | Result | Value |
|---|---|---|
| G0 parameters | pass | 1.007 x the RSSM's |
| G1 no divergence | pass | 0 settle refusals, all values finite |
| G2 beats "the frame never changes" | **FAIL** | PC open-loop relative error 1.213 (needs < 1.0) |
| G3 within 1.25 x the RSSM | **FAIL** | open-loop 1.213 against the RSSM's 0.822; reconstruction 0.0279 against 0.0177 |
| G4 cost | **FAIL** | training step 1.98 x (ok), VRAM 1.008 x (ok), **acting time 4.78 x (needs <= 3)** |

The backprop ablation (`pc_bp`, same settling, optimiser-trained networks, reported and never gated): open-loop 0.873, reconstruction 0.0186. It would pass G2 and G3.

**Reading (descriptive).**
- The required suites passed on the first run after the contract G change (`pc_tests` rc 0) and `pc_boot_check` passed. The build runs; the pilot's G1 shows the Hebbian core does not diverge.
- Against the predictions of section 8: **P-2 held** (trains without divergence, fails G2 and G3, and the ablation separates the cause). **P-3 partly held** (acting time is the cost that failed; the step time and VRAM did not). **P-1** held only after the contract G correction of section 11c.1. **P-4** is moot because the gate failed.
- Because `pc_bp` is close to the RSSM (0.873 against 0.822) while the Hebbian core is not (1.213), the shortfall comes mostly from the local learning rule at this setting, not from the settling or the architecture.
- `t1_wp20_pc_core` is blocked by the gate, as designed. The campaign continued.

**Not established.** Whether the Hebbian core is simply under-trained at 3,000 steps with `pc_hebbian_lr` 0.01: the diagnostic of section 11c.1 found the isolated rule reaches a relative error of 0.258 at lr 0.01 against 0.042 at 0.05 and 0.008 at 0.2 in the same number of steps, so the default may be too slow. This is a hypothesis, not a finding; the gate failed at the registered setting and that result stands.

**Rule for any second attempt.** A second pilot is a new, separately registered attempt: one difference (the learning rate, or the number of steps), its own stage id, a prediction written before it runs, the same gate, and the first result kept in this section. It must not be presented as the first pilot passing. Choosing the setting after seeing the failure is a tuning step; record the number of attempts, so a pass is read in that light. Acting time (4.78 x) would also have to fall by about a third, and a learning-rate change does not address it. Decision: owner.

## 11e. Second pilot attempt and the learning-rate diagnostic (registered 2026-10-27, before it runs)

**What the first pilot's curves say** (`stages/pc_pilot/results.json`, same data and batches for all three cores). The Hebbian core was not simply slow:

| Step | RSSM open-loop (x copy-last) | Hebbian PC open-loop | Hebbian PC recon | RSSM recon | PC KL | RSSM KL |
|---|---|---|---|---|---|---|
| 250 | 1.159 | **0.816** | 0.0432 | 0.0306 | 0.202 | 0.491 |
| 1000 | 0.784 | 0.891 | 0.0344 | 0.0206 | 0.315 | 1.043 |
| 2000 | 0.756 | 0.984 | 0.0287 | 0.0181 | 0.415 | 1.320 |
| 3000 | 0.822 | **1.213** | 0.0279 | 0.0177 | 0.467 | 1.480 |

Two separate things: (1) **reconstruction improves, but slowly**, and the latent carries little (KL 0.2-0.5 against 1.5): consistent with an under-trained local rule at lr 0.01, as the isolated test of section 11c.1 suggests (0.258 at lr 0.01 against 0.042 at 0.05 for the same steps). (2) **The open-loop error rises after step 1500** (best 0.816 at step 250, 1.213 at the end) while reconstruction keeps improving: that is not "slow", it is the imagined rollout (which uses the prior) getting worse. Under-training does not explain a rise; this is the more important finding and the diagnostic is built to test it. The backprop ablation shows no such rise (0.82-0.97, flat).

**Hypotheses for the slow learning (none established):**
- **H-lr:** lr 0.01 is too small for 3,000 steps. Predicts that a larger rate improves reconstruction and KL.
- **H-clip:** the per-step 5% clip of the weight norm binds. The isolated test found the clip irrelevant, so this is expected to fail.
- **H-prior:** the locally trained prior drifts from the posterior as the posterior's target moves, so imagination degrades while reconstruction (which uses the posterior) improves. Predicts that the open-loop rise persists at any rate and the prior network's error does not fall.
- **H-embed:** the generative target is the encoder's own moving embedding (11c.1), which also affects the posterior and the precision (`obs_var` ended at 2.5 for the Hebbian core and 186 in the ablation).

**Registered arms** (one run, `pc_pilot2`, `tools/pc_pilot2.py`; the same 12,000 frames copied from the first pilot, the same batches, the same seed, tiny, 3,000 steps; RSSM reference re-run in the same run):

| Arm | Setting | Role |
|---|---|---|
| p2a | `pc_hebbian_lr` 0.05, `pc_iters` 4 | **registered, gated.** One difference from attempt 1: the learning rate. |
| p2b | `pc_hebbian_lr` 0.05, `pc_iters` 2 | **registered, gated.** p2a with half the settling steps, aimed at G4's acting time (4.78 x in attempt 1). |
| d_lr001 | lr 0.01, `pc_iters` 4 | diagnostic: replicates attempt 1, so the first failure's stability is known |
| d_lr02 | lr 0.2, `pc_iters` 4 | diagnostic: a faster rate |
| d_clip | lr 0.05, `pc_iters` 4, `hebbian_clip` 0.5 | diagnostic: is the 5% clip binding |

The gate (section 5) is applied **unchanged** to p2a and p2b only; a diagnostic arm can never make the verdict pass. Each Hebbian arm also logs, at every evaluation, the three local networks' output error, relative step (a step equal to the clip means the clip is binding), weight norms, `obs_var` and refusals. Several measurements per run was a deliberate choice to save host time.

**Attempt count.** The first pilot is attempt 1; p2a and p2b are attempts 2 and 3. A pass by either is only a licence to run the five-seed A/B (the real test), and is to be read in the light of three tries at the same gate. A second A/B overlay (the new rate) would be needed and is decided after the result. A failed attempt is recorded here, not removed.

**Predictions (written before the run).**
- **P2-1.** `d_lr001` reproduces attempt 1 (open-loop 1.1-1.3 at step 3000, recon 0.027-0.030). If it does not, the first failure was not stable and is reported as such.
- **P2-2.** `p2a` improves reconstruction (below 0.0279) and KL above 0.467. Whether G3 passes (open-loop and reconstruction each within 1.25 x the RSSM's) is **less likely than not**.
- **P2-3.** The open-loop rise after step 1500 is **not** removed by a larger rate (H-prior over H-lr): `p2a`'s final open-loop is more than 25% above its own best evaluation. If it is removed, H-lr is supported.
- **P2-4.** `p2a` fails G4 on acting time (about 4.8 x). `p2b` reduces acting time roughly in proportion to the iterations (estimated 2.5-3.2 x, **borderline** for the 3 x limit).
- **P2-5.** `d_clip` is within noise of `p2a` (the clip is not binding). `d_lr02` is faster early and either unstable or no better at the end.
- **P2-6.** Overall: the more likely outcome is that no registered arm passes the whole gate; the run's value is the explanation it gives.

**Queue.** `pc_pilot2_tests` (a few-step real CPU run of each arm kind plus the registration and gate contracts, [`tests/_pc_pilot2_smoke.py`](../../tests/_pc_pilot2_smoke.py)) gates `pc_pilot2` so a defect costs a minute; `pc_pilot2` gates nothing and is required by no stage. Queued 2026-10-27 behind `t1_wp11_risk`.

### 11e.1 Result (2026-10-27, `pc_pilot2` rc 1: no registered arm passed the gate)

Read from `stages/pc_pilot2/verdict.json` and `results.json`. The RSSM reference in this run: open-loop 0.859, reconstruction 0.0179, KL 1.46.

| Arm | G2 open-loop (< 1.0) | G3 (within 1.25 x the RSSM) | G4 cost | Final recon | Final KL |
|---|---|---|---|---|---|
| **p2a** (lr 0.05, 4 iters) | pass 0.783 | **fail**: recon 0.0402 against 0.0179 (2.2 x) | **fail**: acting 4.85 x (step 1.98 x, VRAM 1.006 x pass) | 0.0402 | **0.11** |
| **p2b** (lr 0.05, 2 iters) | pass 0.809 | **fail**: recon 0.0372 (2.1 x) | **fail**: acting **3.17 x** (limit 3; step 1.55 x pass) | 0.0372 | 0.59 |
| d_lr001 (attempt 1's setting) | 0.920 | not gated | not gated | 0.0289 | 0.49 |
| d_lr02 | 0.813 | not gated | not gated | 0.0340 | 0.93 |
| d_clip (clip 0.5) | 0.794 | not gated | not gated | 0.0406 | 0.11 |

**Against the predictions of 11e.** P2-1 **failed**: the replicate reproduces attempt 1's reconstruction (0.0289 against 0.0279) but not its open-loop error (0.920 against 1.213), so attempt 1's G2 failure was largely noise in the open-loop measure (64 held-out windows; the same arm's curve swings 0.85-1.04, the RSSM's 0.78-0.86). P2-2 **failed**: p2a's reconstruction is worse (0.0402) and its KL is 0.11, the free-nats floor, not above 0.467. P2-3 **failed**: p2a has no open-loop rise (0.78 against its best 0.77), so the rise in attempt 1 was not systematic and H-prior is not supported. P2-4 held (p2a 4.85 x; p2b 3.17 x, borderline and just over). P2-5 held for the clip (0.794 against 0.783, recon 0.0406 against 0.0402, KL 0.11 both) and partly for lr 0.2 (better reconstruction than p2a, one spike to 1.09 and KL to 1.68). P2-6 held. Three of the six predictions held (P2-4, P2-5, P2-6) and three failed (P2-1, P2-2, P2-3); the largest surprise is the latent collapse at the higher rate.

**What the diagnostic arms show.**
- **The gap is reconstruction, not imagination.** Open-loop error is at or better than the RSSM's in every Hebbian arm (0.78-0.92 against 0.86); reconstruction is 1.6-2.3 x worse (0.029-0.041 against 0.018). The gate failed on reconstruction (G3) and on acting time (G4), not on G2 once the measure is read with its noise.
- **A faster rate collapses the latent.** KL at the end is 0.11 (the floor; CLAUDE.md: "`kl 0.110` is not a healthy number") for p2a and d_clip, 0.49 at lr 0.01. The posterior carries almost nothing at lr 0.05, so reconstruction cannot be better than the decoder's prior-only guess. This answers the owner's question about the rate: raising it makes the latent worse.
- **The prior and posterior networks learn; the generative head does not.** At lr 0.05 the prior and posterior networks' output errors fall to 0.001-0.003 with relative steps far below the clip; the generative head's error stays at 0.04-0.2 (0.21 at the end of d_clip), its relative step is pinned at the 0.05 clip at every evaluation in p2a, p2b and d_lr02, and its weight norms grow (W1 9 to 16 in p2a, 10 to 135 in d_clip; W2 11 to 96).
- **A looser clip did not help** (d_clip is no better; a tighter clip was not tried here, see 11i), and the running error variance `obs_var` is 177-3,463 in the faster arms (1.6 at lr 0.01), i.e. the head is chasing a target of very large scale.
- **Acting time scales with the settling iterations** (5.0 ms at 4 iterations, 3.26 ms at 2, against the RSSM's 1.03): p2b misses the 3 x limit by 0.17 x. One iteration would likely pass, at some cost to settling that has not been measured.

**Attempt count:** 3 (the first pilot and p2a, p2b). None passed. `t1_wp20_pc_core` stays blocked.

## 11f. The generative-target scale, a fix and its tests (2026-10-27, registered before it runs)

**Hypothesis (H-embed, not established).** The generative head predicts the encoder's own embedding, whose scale grows by orders of magnitude while it trains (11c.1); a local rule with a fixed step cannot track a target that large (its step is pinned at the clip, its weights grow, its error does not fall), the precision it feeds is wrong, and the settled posterior then carries little (KL at the floor). **A caveat that comes from reading the code:** the energy's observation term is precision times error, and the running variance tracks that error, so the scale cancels in the energy; the effect of a large target is therefore expected to act through the head's learning dynamics (the pinned step and growing weights), not through a mis-weighted energy. If the diagnostic shows no improvement, the hypothesis is rejected for the latent collapse.

**Change made.** `world_model.pc_target_norm` (default **false**; the core's constructor argument `target_norm`): when on, the target embedding is divided by its running RMS (statistics only, no gradient; set by the first training pass and moved by training passes only, as `obs_var` is) in the energy, the generative loss and the local update of the generative head. Off, every number is unchanged and no buffer is added (default state dicts are identical). The posterior network's own input is not normalised (it lives in the shared RSSM path); that is a stated limitation. Files: `developmental_ai/world_model/predictive_coding.py`, `rssm.py`, `core/developmental_loop.py` (pass-through), `configs/universal.yaml`.

**Tests.** [`tests/_pc_target_norm_smoke.py`](../../tests/_pc_target_norm_smoke.py): off changes nothing (A); on has the state and a measurement does not move it (B); a falsifier with a target growing 270-fold (error variance reaches orders of magnitude off, stays small on) (C); the local rule sees the normalised target (D); the energy and the loss share one scale (E); wiring (F). The required PC suites, the config-key suite and the pilot suites re-run on the refreshed snapshot (`pc_norm_tests`), because the core changed after `pc_tests` passed.

**Diagnostic run** (`pc_pilot3_diag`, **diagnostic only**: no registered arm, nothing gated, no attempt counted, nothing requires it): the RSSM reference and two normalised Hebbian arms, `d_norm` (lr 0.05, 4 iterations) and `d_norm01` (lr 0.01, 4 iterations), on the same data and batches as pilot 2; read against pilot 2's `p2a` and `d_lr001`.

**Queued** 2026-10-28 (behind `t1_wp12_peers`): the staged tree changed, so the PC snapshot was refreshed by hand with the stage's own command (new snapshot hash `6c6e62034e23`; the first pilot ran on `6914fff6f146`, the second on `f025f97db62d`). The default-off switch leaves the RSSM baseline unchanged.

**Predictions (written before the run).**
- **P3-1.** `d_norm` improves on `p2a`: reconstruction below 0.034 (p2a 0.0402) and KL above 0.3 (p2a 0.11). If reconstruction stays near 0.04 and KL near 0.11, H-embed is rejected for the collapse.
- **P3-2.** The generative head is no longer pinned at the clip (relative step below 0.05 at the later evaluations) and its weight norms stay below 30 (p2a: W2 40).
- **P3-3.** `d_norm01` improves on `d_lr001`'s reconstruction (0.0289).
- **P3-4.** Acting time is unchanged (about 5 ms); this change does not address G4.
- **Overall:** more likely than not a modest improvement in the head's behaviour and, at most, a partial one in reconstruction; the RSSM's 0.018 is not expected to be reached. If rejected, the next suspect is the coupling of the settled posterior to the Hebbian posterior network (the KL floor), measured by z-info.

### 11f.1 Result (2026-10-28, `pc_pilot3_diag`, diagnostic only, rc 0)

Same data, batches and seed as pilot 2. RSSM reference: reconstruction 0.0179, KL 1.48, open-loop 0.821.

| Arm | Recon | KL | Open-loop | Acting | Generative head: relative step / error (normalised units) / weight norms |
|---|---|---|---|---|---|
| `d_norm` (lr 0.05, norm on) | **0.0350** (p2a, same without norm: 0.0402) | **0.326** (p2a 0.11) | 0.766 | 5.09 ms | step **0.05 (pinned)** at every evaluation / 0.39 / W1 9.6, W2 11.0 (p2a W2 40) |
| `d_norm01` (lr 0.01, norm on) | **0.0268** (d_lr001: 0.0289) | 0.446 | 0.793 | 5.13 ms | step **0.05 (pinned)** / **0.85** / W1 10.6, W2 10.2 |

**Against the predictions of 11f.** P3-1: the KL part held (0.326 above 0.3) and the reconstruction part missed by 0.001 (0.0350 against below 0.034). P3-2: **half held**: the weights no longer grow (W2 about 11 against 40, and 135 in the loosest earlier arm) but the head's step is still pinned at the clip at every evaluation. P3-3 held (0.0268 below 0.0289). P3-4 held (acting unchanged, 5.1 ms). The overall prediction (a modest improvement, the RSSM's 0.018 not reached) held.

**Reading.** The growth of the target's scale was real (the running error variance is now 0.3-0.7 against 177-3,463 before, and the head's weights stay bounded) and accounts for some of the latent collapse (KL 0.11 to 0.33) and a modest part of the reconstruction gap. It does **not** make the head learn: its error relative to the (unit-scale) target stays at 0.4-0.85, so it still predicts the embedding poorly, and its step is saturated every update. H-embed is supported in part and is not sufficient. Reconstruction is still 1.5-2 x the RSSM's. The normalisation stays off by default; it changes nothing the campaign runs.

**New hypothesis (H-sat, not established).** With the step clipped to 5% of the weight norm on every update, the head moves by a constant fraction regardless of the error, which is sign-gradient-like noise that cannot settle (a loosened clip made it worse in pilot 2; the prior and posterior networks, whose steps are far below the clip, converge). A smaller clip should let it settle. Registered in 11i.

## 11g. Correction: the PC arm did not boot, and the boot check hid it (found 2026-10-28)

**Correction to earlier reports.** I reported `pc_boot_check` as passed. It was not. The stage ended `done (rc 0)` because `tools/arm_boot_check.py` exits 0 whatever it finds, but its own record says `wp20_pc_core: FAIL exit code 3`, and `analysis/arm_boot_checks.json` holds `{"ok": false, "reason": "exit code 3"}`. The launcher's exit 3 is "the agent never came up": the Hebbian PC arm did not start in the real trainer within the launcher's wait (the check boots for four minutes), on the snapshot. That record is what keeps `t1_wp20_pc_core` **blocked** ("arm wp20_pc_core failed its boot check; fix, then `resume arm_boot_checks`"), so a passing pilot would not have unblocked the five-seed A/B. I read the stage status and not the record. The pilot itself ran because `pc_pilot` required the stage to be done, not to have passed.

**Why the cause is unknown.** The tool deleted the booted root (about 10 GB) after each arm, and the launcher exits before copying logs when the agent never comes up, so nothing was kept. The PC core has run in the model-level pilot and unit tests but has not been shown to start in the real loop; the loop's own handling of the `pc` core (construction, resume, growth and the autoscaler refuse or ignore it) is untested.

**Changes.** `tools/arm_boot_check.py` now keeps the supervisor tail, the run log tail, the first traceback and the crash history in `<exp>/<run>.diag.txt` before deleting the root, and takes `--strict` (exit 1 when a named arm failed; the multi-arm check stays exit 0 so a failing arm still blocks only itself and does not pause the campaign). Suite: [`tests/_arm_boot_check_smoke.py`](../../tests/_arm_boot_check_smoke.py) (a fake launcher: failed boot leaves its evidence; the record is right; strict exit codes; no logs is not an error; a late traceback still fails the arm).

**Queued** 2026-10-28 behind `t1_wp12_peers` (the staged tree changed, so the snapshot was refreshed by hand with the stage's own command first; hash `b9fca824af99`): `pc_boot_tests`, then `pc_boot_check2` (strict, the same 4-minute boot, keeping the logs). Its record rewrites the arm's entry, so a pass releases the A/B block and a failure leaves the diagnosis in `stages/pc_boot_check2/bootchk_wp20_pc_core.diag.txt`. **Open:** the fix itself, which depends on what that file shows; no change to the core is made until it does.

## 11h. Where the Hebbian core stands (2026-10-28; a reading of 11c-11e, not a new result)

**What is promising.** It trains stably (no divergence and no refused update in any of the eight Hebbian runs of pilots 1 to 3: one, five and two; the refusal counts are in each run's `results.json`); its prior and posterior networks converge (output errors 0.001-0.003 with steps below the clip); the isolated rule converges on a fixed regression (relative error 0.008 at lr 0.2) and, in the unit tests, matches the backprop direction (cosine above 0.95 hidden layer, above 0.995 output layer); and its open-loop prediction is competitive with the RSSM's (0.78-0.92 against 0.86; this measure is noisy, the same setting gave 1.21 and 0.92).

**What is not.** Reconstruction is 1.6-2.3 x the RSSM's (0.029-0.041 against 0.018); the latent carries much less (KL 0.11-0.93 against 1.46, with the floor reached at lr 0.05); the generative head does not converge; training costs about 2 x per step and acting 3-5 x. **Slow learning is not the main problem:** a faster rate made the latent worse, not the head better.

**Reading.** The evidence so far points at the local learning of one head rather than the settling: the backprop ablation with the same settling was near the RSSM (0.873 and 0.0186 in the first pilot). That is one ablation, one seed and 3,000 steps, so it is a lead, not a conclusion. The Hebbian core is partly promising and not competitive. It says nothing about adaptability, which is what the program is judged on; the pilot measures representation quality on fixed data at tiny size, one seed, 3,000 steps.

**Why the RSSM stays the default regardless.** (1) [ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md) keeps both cores selectable and the RSSM as the default until the comparison decides, and the comparison that decides is the five-seed A/B (all five positive, ASS-007), not the pilot; a pilot pass is only a licence to run it. (2) Every other test in the campaign (T1, T2, the fresh control, AB-00) is read against the RSSM baseline, so changing the default would change what all of them mean. (3) The RSSM is the system with measured behaviour, resume, growth and autoscaling support (growth refuses `pc`). (4) One difference at a time: the PC core is a candidate for one A/B, not a replacement. Even a clean A/B win would be followed by an owner decision, not an automatic switch.

## 11i. A smaller step clip for the normalised head (registered 2026-10-28, before it runs; diagnostic only)

**Question.** Is the saturated step (11f.1) why the generative head does not converge? **Arms** (`pc_pilot4_diag`, `tools/pc_pilot2.py`, same data, batches and seed; RSSM reference re-run; **no registered arm, nothing gated, no attempt counted**): `d_norm_c01` and `d_norm_c002`, each `d_norm` (lr 0.05, normalisation on, 4 iterations) with `hebbian_clip` 0.01 and 0.002 (the default is 0.05).

**Predictions (written before the run).**
- **P4-1.** In at least one arm the head's relative step is below its clip at the later evaluations (not pinned).
- **P4-2.** In at least one arm the head's relative error falls below 0.3 by step 3,000 (`d_norm`: 0.39).
- **P4-3.** In at least one arm reconstruction is below 0.030 (`d_norm`: 0.035); reaching the RSSM's 0.018 is unlikely.
- **P4-4.** If the head's error is unchanged at the smaller clips (the step was not the cause), H-sat is rejected and the next suspect is the coupling between the settled posterior and the head's target, not the step size.
- **Overall:** a modest improvement is more likely than not; closing the gap is not expected.

**Queue.** `pc_pilot4_diag`, behind the running stage, after `pc_norm_tests` (done). Queued 2026-10-28 with the snapshot refreshed by hand (hash `4938fb6c74cf`); `pc_boot_tests` was reset with the driver stopped (its first run failed on a defect in my report-tools test, GOV-015 9.13), so `pc_boot_check2` (11g) follows it. A fix to the rule itself would be a code change with its own suite; none is made before this reports.

## 11j. Tests added, tests proposed, and what is queued (2026-10-28)

**Added in this stretch** (all queued on `main`, none run on the development machine): [`tests/_pc_pilot2_smoke.py`](../../tests/_pc_pilot2_smoke.py) (registration, the unchanged gate, a real CPU run of each arm kind, the queue); [`tests/_pc_target_norm_smoke.py`](../../tests/_pc_target_norm_smoke.py) (the normalisation switch, with a falsifier); [`tests/_arm_boot_check_smoke.py`](../../tests/_arm_boot_check_smoke.py) (the boot tool keeps its logs and can fail loudly); [`tests/_report_tools_smoke.py`](../../tests/_report_tools_smoke.py) (the two report tools stay two tools); and [`tests/_pc_loop_smoke.py`](../../tests/_pc_loop_smoke.py), **the first test that runs the PC core inside the real loop**: it builds `DevelopmentalAI` with `core: pc` and the async trainer's inference copy, acts for three segments, trains through the loop's own `_train_world_model`, checks that every local network moved and none was refused, that `_publish_wm` carries the local weights (buffers) to the acting copy, and that a checkpoint and its architecture file name the core; with a `core: rssm` falsifier. It exists because the arm failed to boot in the real trainer and no earlier test could have said why. Stage `pc_loop_tests` runs it before the boot re-check.

**Proposed, not built** (each is a decision for the owner or waits on a result):
1. **Open-loop measure noise.** The pilot's open-loop error swung 0.85-1.04 within one arm and 1.21 against 0.92 between two runs of the same setting (11e.1), from 64 held-out windows. A suite that scores one model on several disjoint evaluation samples, and a rule that a gate reads a mean, would stop a gate from failing on noise. Changing what G2 reads is a change to a pre-registered gate, so it needs a stated, dated amendment.
2. **Acting-time test in the loop.** G4 measures acting cost on the model alone; the loop's per-decision time with the PC core (settling inside the lock, with the async trainer) is not measured. A short loop-level timing run would give it.
3. **Resume and grow with `core: pc`.** Growth refuses the PC core by design; a test that a resumed `pc` checkpoint restores its local networks (they are buffers and need saving, `persistent=True`) and that a `pc` checkpoint is refused by a grow with a clear error would pin both.
4. **Stage-record consistency in the campaign driver.** A stage that exits 0 while its own record says FAIL (the boot check) should be caught by a driver-level check; `--strict` fixes this one tool, not the class.
5. **A distinct latent-information gate.** The pilot reads open-loop error and reconstruction; the floor KL of 0.11 (a collapsed latent) is visible only in the logs. A gate input on z-information would turn the collapse the diagnostics found into a pass or fail. Again an amendment to the registered gate.

**Snapshot:** refreshed by hand again with the stage's own command (hash `0b5ae9647443`).

**Reading rule for any amendment (added 2026-10-28).** Items 1 and 5 above would change the pre-registered gate of section 5 after it failed. Any such amendment is dated, applies to future attempts only, and cannot reclassify attempts 1 to 3, which stay on record as FAIL under the gate as written. It would also be tested on data from before the change before it is used, so it cannot be tuned to make the current arm pass.

**Noise across runs (added 2026-10-28).** The RSSM reference is not bit-identical between runs on the same data, batches and seed: open-loop 0.859 and KL 1.46 in pilot 2, 0.821 and 1.48 in pilot 3. Small differences between arms in different pilots (for example 0.0350 against 0.0402) are therefore read against that run-to-run spread, which is not yet measured; only within-run comparisons against the reference of the same run are used for the gate.

**What is queued, in order, behind the running `t1_wp19_representation`:** `gov015_diag4` (BND-04 trajectories and every contract suite), `gov015_verify2` (reset, re-runs the corrected BND-04 scenario and everything else), `pc_loop_tests`, `pc_boot_tests` (reset), `pc_boot_check2`, `pc_pilot4_diag`. About 1.5 hours of host time. Then the T2 blocks (`t2_rw1` to `t2_rw5` and `t2_rw7`, about 10 hours each by the driver's estimate; the campaign status lists them).

## 12. Related records

[RES-010](../research/res-010-predictive-coding-core.md), [ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md), [ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md),
[GOV-015](gov-015-corrective-plan-categorical-findings.md), [GOV-009](gov-009-implementation-plan.md), [ASS-013](../assurance/ass-013-unattended-test-campaign.md), [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md).
