# Central behavioral problem and proposed plan

> **Document control:** [RES-001](../README.md#research) · Research / diagnosis and plan · **State:** Proposed · **Custodian:** Repository maintainer · Revision history: Git.

**Status:** diagnosis and proposal, based on repository documentation, code, and
existing telemetry. No tests or new training runs were run for this document.

## Summary

The main behavioral problem is a weak action-to-consequence learning loop. The
agent does not consistently receive a useful signal that connects an action it
took to an outcome it caused and can learn to predict or reproduce. In real-time
games, transitions were also associated with the wrong action. At the same time,
episodic visual novelty became a dominant reward source, even though a new view
does not necessarily represent useful progress. These conditions favor random
movement and novelty-seeking over stable, reusable skills.

The proposed work is to make action credit reliable, make intrinsic motivation
depend more on learnable and controllable change, and make skill discovery
independent of game-specific reward events. Each change should earn its place
through behavior measured outside the reward it changes.

## What the evidence says

The current record supports the diagnosis, while leaving the latest fixes
unproven:

- [`ass-003-policy-learning.md`](../assurance/ass-003-policy-learning.md) reports that Doom's policy remained
  about 97–98% of maximum entropy, did not leave MAP01, and had action
  attribution around 0.0076. The identified timing defect paired outcomes from
  real-time games with the next decision's action rather than the action that
  drove the observed interval.
- [`realtime_credit.py`](../../developmental_ai/core/realtime_credit.py) now
  implements executed-action labels, policy lookahead, and delayed PPO credit.
  This is a plausible fix for the timing defect; the pre-registered live
  evaluation in `ass-003-policy-learning.md` has no results yet.
- [`episodic_novelty.py`](../../developmental_ai/curiosity/episodic_novelty.py)
  defines a bonus for views unlike those recently seen. Its own grading reports
  that spinning can earn nearly as much per visit as exploring. That makes it a
  useful exploration signal with a clear path to proxy-seeking.
- The seven records currently in `runlogs/learning.jsonl` (private run log, not published)
  are short segments, not a long-run evaluation. They show policy entropy near
  the uniform level, zero paid learning-progress reward, and no recorded task
  outcomes. This is corroborating telemetry, not a conclusion about the
  performance of the pending code changes.
- [`res-005-skill-discovery.md`](res-005-skill-discovery.md) describes a game-agnostic,
  action-caused effect trigger and skill probation. It is marked planned, not
  built. The existing reward-spike and Minecraft effect-key route cannot
  produce the same skill-discovery behavior in reward-free Doom.

The immediate bottleneck is therefore not a shortage of model size or
subsystems. It is whether the learning process can reliably identify which
actions cause which learnable changes, then reinforce useful repeatable
behavior.

## Goal alignment: keep evaluation distinct from training

The repository combines two aims that should be stated separately:

1. **Developmental learning:** exploration, learning progress, self-directed
   goals, skill acquisition, and transfer should arise from experience.
2. **A concrete capability target:** learn to chop trees in Minecraft, with
   logs and block events used to judge whether that happened.

If tree-chopping counters remain evaluator-only, they can show whether the
agent learned to chop trees, but they cannot tell the learner that chopping is
preferable. That is consistent with a reward-free developmental experiment,
but it does not guarantee tree chopping. If tree chopping is a training target,
then some task feedback is needed and should be described as such. This is a
research choice, not an implementation detail.

The universal-brain goal also needs one distinction: sharing one learner and
one learning process across games does not by itself require every game to
produce an identical scalar reward. The common learning rules can consume
consistent, player-observable signals while adapters translate each game's
sensory and action consequences into the shared interface. Avoiding hidden
engine state is compatible with this; silently treating different game
outcomes as the same objective is not required.

## Proposed changes

### 1. Close the action-credit loop

Every learned transition should identify the action that actually drove the
observed interval, the state from which that action was selected, the interval
duration, the resulting observation, and the reward assigned to that interval.
This applies consistently to the world model, curiosity and attribution
models, empowerment, PPO, replay, and evaluation traces.

The real-time credit implementation already addresses the one-decision lag.
Treat it as a candidate fix until live evidence confirms:

- The lag probe selects zero shift on replay from the fixed run.
- Action attribution rises from the documented 0.0076 baseline.
- The policy's action distribution becomes meaningfully state-dependent and
  continues to update across segments and restarts.
- Variable decision durations are accounted for consistently in reward
  discounting and action credit.

The current `grad_norm_actor` telemetry should be reviewed with the actor-freeze
and KL-stop events. The logged value is reduced from the final PPO epoch, so
zero there can mean that the actor froze earlier; it does not alone prove that
the actor received no gradients during the update.

### 2. Make curiosity favor learnable, controllable change

Keep intrinsic motivation, but make the main learning signal reflect progress
in predicting effects the agent's actions can influence. Raw novelty and
unfamiliar views can be auxiliary exploration signals; they should not dominate
the policy objective when the agent can maximize them through spinning or other
non-progress behavior.

The preferred direction is an action-conditioned signal based on learning
progress or information gain: reward the agent when experience improves its
model of action consequences, with a control or agency check that discounts
changes that would have occurred regardless of the agent's action. Keep the
noisy-TV protection: unpredictable observations with no learnable or controllable
structure should not produce persistent reward.

Before tuning a new weight, diagnose why the current learning-progress signal
records zero paid reward. The significance gate may be correctly suppressing
noise, or it may be suppressing useful progress. Measure both cases and retain
the existing noise-farming falsifier. Any episodic bonus should be bounded,
reported separately, and evaluated against camera-only changes and repeated
movement loops.

### 3. Discover skills from repeatable effects

Replace reward-spike-only discovery with the proposed game-agnostic pipeline in
[`res-005-skill-discovery.md`](res-005-skill-discovery.md): detect surprising changes that are
better explained by the agent's own action, cluster before/after effects, and
create skill candidates without paying extra reward for surprise itself.
Probation should retain a skill only when invoking it reproduces its effect
above the base policy's rate; otherwise retire it with a defined re-entry path.

Use game counters, kills, pickups, exits, logs, and similar measures as
evaluation answer keys when the experiment requires them. They must not leak
into the discovery trigger or reward if the claim is that the learner acquired
the skill from player-observable experience alone.

### 4. Establish a narrow behavioral proving ground

For the next learning experiment, keep the learner and relevant signal path
fixed, and vary one proposed change at a time. A small controlled environment
can establish action credit and causal skill discovery cheaply; a live-game run
is still needed before claiming that the behavior transfers to Doom or
Minecraft.

Avoid adding model capacity, new reward terms, or additional organs during this
diagnostic phase unless a measured bottleneck requires them. Preserve policy
state across ordinary restarts when the observation and action contracts match,
and report when a model resize or schema change necessarily invalidates a
checkpoint component.

## Evidence gates

Use behavior and mechanism measures side by side. A reward increase alone is
not a pass.

| Gate | Evidence required | What failure means |
|---|---|---|
| Transition integrity | Executed action, observation, timing, and reward refer to the same interval; real-time lag probe selects zero shift. | Do not interpret policy or curiosity learning results yet. |
| Action dependence | Action attribution and policy information about state increase over the run; action entropy is interpreted over valid actions. | The policy is not demonstrating a learned state-action mapping. |
| Useful intrinsic signal | Reward tracks improvement on controllable prediction and is low on uncontrollable noise, camera-only novelty, and repeated farms. | The intrinsic objective remains proxyable or inert. |
| Skill validity | A discovered skill reproduces its effect above the base-policy rate on held-out invocations. | The bank is storing labels or policy copies without demonstrated capability. |
| Behavioral progress | A preregistered evaluator-only measure improves against a baseline over adequate duration and multiple seeds. | No claim of task competence or transfer. |
| Persistence and transfer | Learning survives restart and, separately, accelerates a harder held-out environment. | Report only within-run learning; do not claim cumulative development. |

The current policy-learning protocol provides a starting point for transition
and policy gates. New reward or discovery experiments should freeze their
hypothesis, primary metric, baseline, seeds, and failure conditions before
looking at scored results, consistent with [`FND-001`](../foundation/fnd-001-ab-protocol.md).

## Recommended order

1. Run the already-pre-registered real-time credit comparison and report its
   mechanism and behavior results separately.
2. If transition integrity passes but the policy remains near-uniform, inspect
   PPO actor updates, bootstrap inputs, reward dependence on action, and
   checkpoint continuity before changing intrinsic reward weights.
3. Measure the current learning-progress signal's zero-payment behavior and
   episodic novelty's response to spinning, camera changes, and repeat loops.
4. Build and evaluate generic effect discovery and skill probation behind a
   configuration switch, with no discovery reward.
5. Re-evaluate on the stated behavioral target. Keep task outcomes evaluator
   only for reward-free claims; if task feedback is introduced, state that
   change and test it as a separate condition.
6. Only after a reliable single-environment result, test shared training across
   games and forgetting/transfer.

## Claims this plan does not support

Passing these gates would support claims about measured functions such as
action-conditioned learning, self-directed exploration, repeatable skills, or
transfer under specified conditions. It would not establish subjective
experience, general intelligence, or that a reward-free learner will
spontaneously select a human-chosen task such as tree chopping.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-006](../assurance/ass-006-realtime-action-lag-incident.md): Real-time action-lag incident (2026-10-18)
- [GOV-008](../governance/gov-008-human-like-learning-motivation.md): Human-like learning program: motivation, scope and change record
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [RES-003](res-003-plurality-roadmap.md): The plurality roadmap — what shipped, Phases 1–7
- [RES-006](res-006-human-like-learning-gap-analysis.md): Human-like learning gap analysis: SkyBot against the AI-counterpart learning map
<!-- backlinks:end -->
