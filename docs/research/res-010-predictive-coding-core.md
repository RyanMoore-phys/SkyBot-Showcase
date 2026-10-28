# Predictive coding as the world model's latent core

> **Document control:** [RES-010](../README.md#research) · Research / design note and hypotheses · **State:** Proposed; built behind a default-off switch (2026-10-25), untested · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-25. **Authority:** the owner, 2026-10-25: build a predictive-coding alternative to the RSSM, keep both in the
repository until a head-to-head test returns, and make the pilot a gate. **Decision record:**
[ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md). **Plan:** [GOV-016](../governance/gov-016-predictive-coding-core-plan.md).
**Test protocol:** [ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md).
This note makes **no behavioural claim**. Where it says a design "should" do something, that is a hypothesis with a stated way to be wrong (section 4).

## 1. The question and why it is asked

The program's aim is learning that is as human-like as can be achieved, judged by adaptability to unfamiliar games
([GOV-008](../governance/gov-008-human-like-learning-motivation.md), [ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md)).
The world model is a DreamerV3-style RSSM ([ARC-001](../architecture/arc-001-architecture.md)): a recurrent state `h` and a categorical latent `z`,
where `z` is inferred in **one amortised pass**, `posterior_net(h, embedding)`, and trained by reconstruction plus a KL term between
that posterior and the model's prior.

**Predictive coding** (PC) is a theory of cortical computation in which a generative model predicts the input top-down, the
mismatch is a **precision-weighted prediction error**, and perception is the **iterative settling** of the belief that reduces it.
The owner asked whether swapping the RSSM's inference for this is an option and whether it learns more like the behaviour the program wants.

What the two already share: both learn by prediction error. The RSSM's posterior-versus-prior KL is a prediction error in latent space;
reconstruction is one in pixel space. What differs is **how the latent is inferred and how the weights learn**:

| | RSSM (today) | PC core (WP-20) |
|---|---|---|
| Inference of `z` | one forward pass of an encoder-like network | that guess, then `pc_iters` energy-descent steps |
| What the error is weighted by | fixed loss weights | an **estimated precision** (inverse running variance of the observation error) |
| Belief vs prior | a KL penalty in the loss | a term of the same energy the belief settles on, at inference time |
| Learning rule of the latent core | backpropagation of one loss through every weight | **local, Hebbian-form updates**: a synapse changes by (the prediction error at its output) x (the activity at its input); no error signal is carried back through the network |

**Owner correction (2026-10-25).** The first build of this core kept backpropagation for its weights. The owner's intent was predictive *learning*, i.e. the local rule,
so the core's own networks now learn locally ([GOV-016](../governance/gov-016-predictive-coding-core-plan.md) section 2). The earlier version is kept as the `pc_bp` ablation, rebuilt after [ASS-020](../assurance/ass-020-recent-changes-static-audit.md) F-05 so that it has the same three networks (same topology and initial values) as ordinary parameters; it still differs in the optimiser.

**What is not known.** That this makes behaviour more human-like is a hypothesis from the theory, not a measured fact in this system. The test is whether the PC core learns at
least as well and adapts better under the program's own standard (five seeds, one hour each, positive on all five).

## 2. The design (what was built)

`world_model/predictive_coding.py`, `PredictiveCodingCore(RSSM)`. Everything the RSSM does is inherited (state `{h, z}`, imagination, the KL training, the unclamped
`prior_divergence` readout that the surprise-based skill trigger and the autoscaler read). What changes:

1. **Inference.** `h` and the prior are computed as before, and the amortised guess `l0 = posterior_net([h, embedding])` is made. The belief is then refined for `pc_iters` steps of gradient descent on

   `E(l) = 0.5 * Pi_o * mean_d (x_d - g(h, softmax(l))_d)^2  +  Pi_p * KL(softmax(l) || prior)`

   where `x` is the observation embedding, `g` is the top-down head (`obs_pred_net`), and `Pi_o = 1 / (running variance of the observation error)`, clamped. Each step is clipped per
   sample; a sample with a non-finite gradient stays at its amortised guess and is counted.
2. **What the model sees.** The logits used are `l0 + stopgrad(l_K - l0)`: the value is the settled belief; the gradient is the amortised path's.
3. **Local learning of the three latent-core networks** (each a `LocalPCMLP`, weights held as buffers, never parameters):

   | Network | Maps | Target it learns toward | Local error |
   |---|---|---|---|
   | `prior_net` | `h` -> prior logits | the settled posterior | `post_probs - softmax(prior)` (delta rule on the softmax) |
   | `posterior_net` | `[h, embedding]` -> amortised logits | the settled logits | `settled - amortised` |
   | `obs_pred_net` | `[h, z]` -> embedding | the observation | `Pi_o * (embedding - prediction)` |

   Each network is two layers whose hidden layer is a predictive-coding node (a value and an error): it is relaxed for a few steps, then `dW_l = lr * e_l * a_(l-1)^T` with `e_l` the
   layer's own error. At equilibrium that approaches the backpropagation gradient, built from local terms only, as the hidden precision grows ([Whittington and Bogacz 2017](#8-external-references); [Millidge, Tschantz and Buckley 2022](#8-external-references)). The test suite checks the cosine numerically (not equality) and checks that it fails when the relaxation is switched off. Updates are clipped relative to the weight norm; a non-finite update is dropped and counted.
4. **A generative loss** still pushes the top-down prediction's error back into `h` and `z`, so the GRU and the encoder receive that signal (autograd flows *through* the local networks to their inputs, a conduit, but never onto their weights).

**Local and not local, plainly.** Local (Hebbian-form): the weights of `prior_net`, `posterior_net`, `obs_pred_net`. **Still backpropagation:** the encoder, the GRU sequence model, the decoder and
every other head (reward, continue, flow, inverse dynamics, ...), and the latent horizon term's effect on `h`. Making those local is a larger, separate change (section 5).

Unchanged by construction: the latent width, the policy and its inputs, every head, the replay buffer, the loop, the relations. `pc_iters = 0` gives the RSSM-shaped core
with local learning but nothing to learn the posterior toward, so it is not a useful arm; the ablations are `pc_bp` (settling, ordinary networks) and `rssm`.
First-arm values (`pc_iters 4`, `pc_step 0.5`, `pc_weight 1.0`, `pc_prior_precision 1.0`, `pc_hebbian_lr 0.01`, `pc_hebbian_iters 6`) are the simplest choice, **not tuned**.

## 3. How it joins the current system

One switch, `world_model.core: rssm | pc` (default `rssm`), plus `world_model.pc_learning: hebbian | backprop` (default `hebbian`), read at world-model construction in the loop and recorded in
`world_model_arch.json`. The local update runs once after each optimiser step, from the steps the loss just saw. Both cores stay in the tree until the verdict
([ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md)). They are **not checkpoint-interchangeable**, and growth refuses a PC core (no growth rule yet).
The locally learned weights are buffers, so parameter counts must add them (`local_values()`); the pilot does.

## 4. Hypotheses and how each can fail (pre-registered; thresholds in GOV-016 section 5 and ASS-019)

| ID | Hypothesis | Fails if |
|---|---|---|
| H-1 | The PC core trains stably on real Doom frames | it diverges, a non-finite value appears in its losses, or many settle or local-update samples are refused |
| H-2 | At equal size and data it models the dynamics about as well | its held-out open-loop error is worse than 1.25 times the RSSM's, or it does not beat "the frame never changes" |
| H-3 | Its cost is affordable on the 8 GB host | seconds per step or acting time exceed 3 times the RSSM's, or VRAM exceeds 1.25 times |
| H-4 | It improves adaptability to held-out maps | the target-phase measure is not above the noise floor and the same-snapshot RSSM baseline on **all five** seeds |

H-1 to H-3 are the **pilot gate**. H-4 is the **five-seed A/B**. A null or negative H-4 is recorded as such and is a useful result.

## 5. Out of scope, deliberately

- **Local learning of the encoder, the GRU, the decoder and the other heads.** The GRU in particular needs a local temporal rule (the hidden state would itself become an inferred
  quantity with its own prediction error); the CNN encoder and decoder need a convolutional local rule. Each is a separate package after this one.
- **Learned per-dimension precision, or a hierarchy of settling layers.** A single scalar precision keeps the change small; richer precision is the first follow-up if H-2 is met and H-4 is close.
- **Growth of the PC core** and **PC for the policy**.

## 6. Risks

- **Compute.** `pc_iters` extra gradient evaluations of a small network per perception step, on a host where acting latency limits the step rate ([ADR-005](../decisions/adr-005-compute-pressure-order.md) order applies). Measured by the pilot.
- **Non-finite values and latches.** The class of defect found by [ASS-014](../assurance/ass-014-categorical-harness-first-run-findings.md). The core refuses a bad sample (it keeps the amortised guess, so it recovers the next time the gradient is finite) and counts it; the test suite injects a non-finite value to prove it.
- **Local learning may simply learn slower.** The latent core's posterior can no longer be shaped by the decoder's gradient (only by the settled belief); the encoder can adapt to it. Expect H-2 to be the gate most at risk, with G4 (cost) next. The `pc_bp` run in the pilot says whether any shortfall comes from local learning or from the settling.
- **Confounds.** The generative loss and the settling are both additions. `pc_bp` separates the learning rule from iterative inference (the optimiser still differs); the A/B itself changes both at once.
- **An unrun build.** Nothing here has been executed on the host yet.

## 7. Related records

[ADR-007](../decisions/adr-007-two-selectable-world-model-cores.md), [GOV-016](../governance/gov-016-predictive-coding-core-plan.md),
[ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md), [RES-004](res-004-rssm-size-experiment.md) (the harness the pilot reuses),
[GOV-008](../governance/gov-008-human-like-learning-motivation.md), [ARC-001](../architecture/arc-001-architecture.md), [ADR-005](../decisions/adr-005-compute-pressure-order.md).

## 8. External references

These motivate the design. None validates this implementation; identifiers should be confirmed against the publisher before any use outside this repository.

- Rao and Ballard (1999), *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects*, Nature Neuroscience 2(1), 79-87, [doi:10.1038/4580](https://doi.org/10.1038/4580): the hierarchical prediction-error model.
- Friston (2005), *A theory of cortical responses*, Philosophical Transactions of the Royal Society B 360, 815-836, [doi:10.1098/rstb.2005.1622](https://doi.org/10.1098/rstb.2005.1622): precision-weighted prediction error and inference as settling.
- Whittington and Bogacz (2017), *An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity*, Neural Computation 29(5), 1229-1262, [doi:10.1162/NECO_a_00949](https://doi.org/10.1162/NECO_a_00949): the local update and its relation to backpropagation.
- Millidge, Tschantz and Buckley (2022), *Predictive coding approximates backprop along arbitrary computation graphs*, Neural Computation 34(6), 1329-1368, [arXiv:2006.04182](https://arxiv.org/abs/2006.04182): the same result for general graphs.
- Hafner, Pasukonis, Ba and Lillicrap (2023), *Mastering Diverse Domains through World Models* (DreamerV3), [arXiv:2301.04104](https://arxiv.org/abs/2301.04104): the RSSM this core replaces.
