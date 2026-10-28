# Real-time action-lag incident (2026-10-18)

> **Document control:** [ASS-006](../README.md#assurance-and-evidence) · Assurance / incident record · **State:** Record · **Custodian:** Repository maintainer · Revision history: Git.

**Scope:** this record covers the defect in which real-time Doom streams paired
each observed interval with the wrong action, and the fixes built for it. It
states what was measured, what was inferred from code, and what has not been
tested live. The live evaluation is governed by
[ASS-003](ass-003-policy-learning.md) and has no results yet.

## 1. Summary

In every Doom run on `main` from 2026-10-17 to 2026-10-18 (Stage C, the mock
deployment, the autoscaler scale test) the policy ended as random as it
started. The run log reported policy entropy of 2.68–2.73 nats and labelled it
"70 % of max". Against the 16 actions Doom exposes (ln 16 = 2.77) that is
97–98 % of uniform. No stream left MAP01 and none reached an exit.

The primary cause found in code is that real-time environments returned the
frame produced by the previous decision's action. Every consumer of a
transition therefore paired the interval with the wrong action. Several
secondary defects, listed in section 4, reduced learning further.

## 2. Evidence

| Observation | Source | Basis |
|---|---|---|
| Policy entropy 2.68–2.73 nats over 40–120 min of play | run logs from the Stage C, mock-deploy and scale-test runs | Measured |
| Printed maximum was ln 48 (3.87); valid actions were 16 (ln 16 = 2.77) | `developmental_loop.py` readout; `configs/games/doom.yaml` controller | Measured and read from code |
| Action attribution 0.0076 ("very low") | run log, curiosity diagnostics | Measured |
| The step function returns before any tic of the new action has run | `vizdoom_env.py` `_step_realtime`, `_rt_collect`, `_clock_loop`; `player/adapter.py` `step` | Read from code |
| The loop takes about 165 ms per decision (about 5.7 tics at 35 Hz) | run log "Loop timing" | Measured |
| `late_tics` is the clock thread's own overrun (about 1 %), not a late agent | `vizdoom_env.py` `_clock_loop` | Read from code |

The causal claim (the interval returned by `step(a_t)` was driven by
`a_{t-1}`) comes from reading the code path, not from an experiment on a live
run. The fix was confirmed against a fake environment and the real ViZDoom
engine in the unit-level checks listed in section 6. Whether it changes
learning is untested.

## 3. Mechanism

1. `step(a_t)` sets the held buttons to `a_t` and returns the newest frame.
2. The game clock runs on its own thread and, between decisions, played
   every tic with the previously held action `a_{t-1}`.
3. The returned interval from `o_t` to `o_{t+1}` was therefore driven by
   `a_{t-1}`. The effect of `a_t` first appeared in `o_{t+2}`.
4. Each of the following consumers stored the interval against `a_t`:
   - replay and the world model, including inverse dynamics, which had to
     predict `a_t` from frames containing no trace of it;
   - the curiosity forward model and its action-attribution gate, which
     compares the chosen action's prediction with the no-op prediction and so
     paid about nothing;
   - empowerment;
   - PPO rows and rewards, so each decision received the reward of the
     previous one's consequences.

Synchronous environments (Minecraft, and ViZDoom with `realtime: false`)
execute `a_t` before returning and were not affected.

## 4. Contributing defects found in the same investigation

| Defect | Location | Effect |
|---|---|---|
| Entropy readout divided by ln 48 | `developmental_loop.py` entropy print | Hid how uniform the policy was |
| KL stop judged on each 32-row minibatch | `actor_critic.py` train step | First noisy batch froze the actor (about 0.02 nats of movement per segment) |
| Bootstrap value computed without proprio | `developmental_loop.py` segment end | Biased value for the last rows of each stream |
| Masked logits counted toward the 5.0 logit range | `actor_critic.py` logit bound | Compressed the gaps among the 16 live actions |
| Distillation toward a near-uniform dream actor, over 48 actions | `developmental_loop.py` distillation | A step per segment that Adam does not scale down |
| Intrinsic weight decayed toward an extrinsic reward Doom never pays | `actor_critic.py` `RewardMixer.decay` | Reward reached 0 after about 137 segments; reset at every launch |
| Seven of eight streams used a different reward function | `developmental_loop.py` `_scout_mixed_reward` | One critic fit two reward functions |
| Farm damp read the engine position | `developmental_loop.py` farm-damp block | Violated the no-engine-reads rule (ADR-002) |
| Action head and critic restarted fresh at every launch | `policy_features` resume | Learning lost at each relaunch |

Episodic novelty supplied 72–89 % of income. It depends little on the action.
Its weight was not changed; that is a separate decision for the owner.

Ruled out by reading the code: world-model drift inside the PPO ratio
(act-time latents are stored), off-policy data from the standby actor (it
stores nothing), and the "Exploration ratio" figure (it only scales curiosity).

## 5. Changes made

All changes sit behind configuration switches in `configs/universal.yaml`.
With the switches off, or for synchronous streams, behaviour is unchanged.

- **Real-time credit** (`realtime_credit.*`, `core/realtime_credit.py`):
  environments report the action actually held during the interval;
  replay, curiosity, the acting belief and empowerment use it; the policy
  chooses from the state its action will start in; PPO credit is delayed one
  interval. Old real-time replay can be relabelled after the lag probe
  (`tools/replay_lag_probe.py`) confirms it is lagged.
- **PPO** (`policy.*`, `dream_training.*`): whole-rollout KL check, valid-only
  logit bound, proprio in the bootstrap, thresholded and masked distillation,
  valid-action entropy readout.
- **Reward:** intrinsic weight held while no extrinsic reward has been
  earned; scouts share stream 0's reward function; farm damp keyed on the view
  (`infra.farm_key: view`).
- **Persistence:** whole-policy resume under an unchanged reward fingerprint
  (`policy_full`), policy growth with the world model, a lifetime step
  counter, replay kept across a stream cut, one replay write per stop.
- **Measurement only:** information between state and action on a fixed probe
  set, KL from the first policy, actor gradient norm over applied updates,
  freeze trip point, decision duration in tics.

## 6. Verification status

| Check | Result |
|---|---|
| New contract tests for the fixes (`_realtime_credit_smoke`, `_ppo_fixes_smoke`, extensions to the ViZDoom, Player, standby, resume and growth suites) | Pass; each new contract was shown to fail on the old behaviour |
| Unit tier and contract tier | Pass, except `_incident_bundle_smoke` (known macOS-only failure, no `flock`) |
| Live comparison, OFF against ON, 120 minutes each | **Not run.** `main` was unreachable on 2026-10-19 |

No claim is made that the fixes improve learning. The pass criteria,
including the lag probe, action attribution, critic explained variance and the
state-dependence measure, are fixed in
[ASS-003](ass-003-policy-learning.md) and apply unchanged.

## 7. Open items

- Run the pre-registered comparison when `main` is reachable.
- Decide the episodic novelty weight after the first results.
- Skill discovery ([RES-005](../research/res-005-skill-discovery.md)) is
  unchanged and still unbuilt.

## 8. Related records

- [ASS-003](ass-003-policy-learning.md): protocol and amendments.
- [RES-001](../research/res-001-central-behavioral-problem-and-plan.md): the
  behavioural diagnosis and evidence gates.
- [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md):
  evaluator state is not a learning input.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-002](ass-002-action-stall-audit.md): SkyBot action-stall audit (2026-08-18)
- [ASS-010](ass-010-current-system-limitations.md): Current system limitations (baseline for the human-like learning program)
- [GOV-008](../governance/gov-008-human-like-learning-motivation.md): Human-like learning program: motivation, scope and change record
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [RES-006](../research/res-006-human-like-learning-gap-analysis.md): Human-like learning gap analysis: SkyBot against the AI-counterpart learning map
<!-- backlinks:end -->
