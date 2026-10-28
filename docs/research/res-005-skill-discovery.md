# Skill discovery from the agent's own experience — plan (2026-10-18)

> **Document control:** [RES-005](../README.md#research) · Research / technical plan · **State:** Proposed · **Custodian:** Repository maintainer · Revision history: Git.

Status: **planned, not built.** Design chosen by the owner on 2026-10-18;
measurement first, then build. Universal brain only — the legacy
Minecraft-only brain and its `skill_bank_mc_curiosity/` are not touched.

## Why

* **Doom can never mint a skill.** Goal discovery fires only on a spike of
  the game's EXTRINSIC reward (`achievement_goals._is_spike`), and Doom pays
  0 by design. The `GATE OVERDUE: 'learned_options'` alarm was about an
  empty bank (CLAUDE.md §10).
* **Minecraft's skills are keyed by engine counters.** A slot is named
  `break_<block>` from MineRL's `mine_*` stats; anything else is a
  `discovered_N` fossil and is refused at mint. That is an engine read, so
  under ONE LEARNING PROCESS FOR EVERY GAME (CLAUDE.md §1) the universal brain
  must discover skills the same way in every game. **Owner's decision:
  Minecraft uses the same trigger and keys as Doom.** The mining counters
  become the evaluation answer key, never an input.
* **Kept from today's pipeline:**
  * the repeatability gate (an effect must recur in `mint_min_repeats`
    distinct episodes before it becomes a skill);
  * grounded-effect dedup (`_slot_for_effect`), working-set paging and the
    competence predictor;
  * the fossil guard against `discovered_N`.
* **Replaced:** the trigger and the source of the effect key.

## The design (owner's choices)

**Trigger = surprise x agency.** A step is a candidate unlock when:

1. **Surprise spikes.** `s_t` = unclamped KL(posterior || prior) summed over
   the latent: the same quantity as the autoscaler's z-info, but per step and
   per stream.
   * It is z-scored against that stream's running baseline (EMA mean and
     variance, ~1,000-step horizon).
   * A spike is `z_t >= z_thr` and `s_t >= s_floor`.
   * The baseline decays, so one freak step cannot pin the bar high: no
     latch, §4.1.
2. **The agent's own action caused it.** Using the action-AVERAGED prior
   `p̄ = mean_a' prior(· | h_{t-1}, z_{t-1}, a')`:
   * `agency_t = KL(post || p̄) − KL(post || prior(· | a_t))`
   * Positive means the action actually taken explains the outcome better
     than "any action would have".
   * A monster walking into view, a door someone else opened, or screen
     flicker is surprising under every action, so its agency is ~0 and it is
     filtered out.
   * The trigger fires when `agency_t >= agency_frac x s_t`.
   * Cost: one batched prior evaluation over all actions, ONLY on
     surprise-spike steps (bounded by the spike rate, not every step).
3. **Excluded:** any step at or within `k_guard` steps after a life boundary
   (death, reset, rewind, `env_restarted`, stream heal) — a new life is
   maximally surprising and caused by nothing.

**Acceptance = both:**

1. **Latent effect key.** At a trigger, the effect vector is
   `e = key(view_{t+k}) − key(view_{t-1})`.
   * `key()` is the frozen view encoder that episodic novelty already uses
     (`DevelopmentalAI._view_key`), normalised; `k` covers the action repeat.
   * Online leader clustering by cosine (>= `tau`) assigns `effect_N`.
     Centroids are EMA-refined and saved with the goal state.
   * The new `effect_N` key replaces `break_<blk>`, and goes through the
     existing `_slot_for_effect` path, so dedup, recall and paging work
     unchanged.
   * The fossil guard keeps refusing `discovered_N` and admits `effect_*`.
   * `required_effects` (hard-coded `["break"]` in the loop and
     `infra/stack.py`) is derived from the keys actually seen.
   * **Encoder drift:** the frozen key encoder refreshes periodically. Each
     key stores up to 8 before/after frame pairs, and on a refresh the
     centroids are re-embedded from them, so `effect_3` keeps meaning the
     same thing.
2. **Probation for every learned skill.** A minted skill gets `K` invocations.
   * Success = its own effect key fires (a trigger in that cluster) while the
     option runs.
   * Baseline = the rate that key fires per equal number of steps under the
     base policy, from the trigger log.
   * **Keep** if successes beat the baseline (one-sided binomial, p < 0.05,
     and at least 2 successes).
   * **Otherwise retire it:** free the option slot, and mark the key retired
     with a backoff.
   * **Escape (§4.1):** a retired key becomes eligible again after it recurs
     in N more distinct episodes, with N doubling per retirement — never
     "until a human notices".
   * This also gives the success-rate measurement that "mastery" never had
     (§4.6). Skills are byte copies of the policy (§4.6), so probation is the
     falsifier: a copy whose slot conditioning does nothing fails it and is
     retired, instead of fossilising.

**Discovery pays no reward.** The trigger only creates goal candidates. It is
not a curiosity term: a reward on "surprising and caused by me" would be a new
farm (flicking a light switch forever).

## Step 0 — measure the trigger before building it (main, training paused)

Owner's choice: pause training for the window (`touch runlogs/STOP`, ~2-3 h,
like Stage C).

* **Brains:**
  * `~/stageC/C_tiny_x8`: tiny, the size the live brain now starts at.
  * `~/stageC/W_fresh`: live size, about 70 min of world-model training.
  * Grading both shows whether the thresholds depend on size.
* **Play:**
  * 8 Doom streams, ~45 min per brain: the brain's own policy, plus a
    sticky-random arm, so agency is not confounded with one policy's habits.
  * Logged per step: `s_t`, the full action-conditioned prior KL vector on
    spike steps, the view keys, the action, and life-boundary flags.
  * Logged as the ANSWER KEY (evaluation only, never fed back): the
    evaluator's kills, items, secrets, health, armor, ammo, deaths and exits.
* **Answer-key events:** kill (KILLCOUNT up); pickup (ITEMCOUNT, or health /
  armor / ammo up without damage); damage taken; secret; exit; death
  (excluded from triggers, kept as a check).
* **Tool:** `tools/skill_trigger_eval.py` (offline). Thresholds `z_thr`,
  `agency_frac` and `tau` are fitted on half the recording and scored on the
  other half; no tuning on the scored half.
* **Compared against:**
  1. surprise alone, no agency;
  2. a random trigger at the same rate;
  3. the old extrinsic spike (expected: 0 fires).

**Pre-registered decision rules:**

| outcome on the scored half | decision |
|---|---|
| precision >= 0.5 (a fire within +/-3 steps of an agent-caused event), >= 2x the random trigger, recall >= 0.3 on kills + pickups, and effect-key purity >= 0.6 against event type | **build it as designed** |
| as above, but agency removes < 30 % of surprise-only false fires, or costs > 20 % of recall | build with **surprise alone**; agency stays measurement-only; report |
| precision < 2x the random trigger | **stop**: the world model's surprise does not mark events yet; report and re-measure after more training |
| effect keys mix event types (purity < 0.6) but the trigger passes | build the trigger and probation; revisit the key (e.g. the latent instead of the view embedding) before relying on names |

## Step 1 — build (behind a switch)

* `goals.discovery: surprise_agency` in `configs/universal.yaml`. The old
  reward-spike path stays the legacy brain's default.
* **New `developmental_ai/skills/effect_discovery.py`:**
  * `SurpriseAgencyTrigger`: per-stream baselines, life-boundary guard.
  * `EffectKeyClusterer`: centroids, exemplars, re-embed on encoder
    refresh, save/load.
* **`world_model/rssm.py`:** `prior_all_actions(h, z)`, the batched prior
  over every action slot.
* **`core/developmental_loop.py`:**
  * ONE helper, `_skill_trigger_step(...)`, called from BOTH stepping bodies
    (§4.2: grep count == 2). It feeds
    `broadcaster.update(env, stream, effect_key=..., triggered=True)`.
  * Per-step KL comes from the perception step that already runs.
* **`core/achievement_goals.py`:**
  * `update()` takes `triggered` in place of the reward spike when the
    switch is on.
  * `effect_` joins the namespaced prefixes.
  * The fossil guard admits `effect_*`.
* **`required_effects`:** derived from the keys seen (loop and
  `infra/stack.py`).
* **Probation:** in the option executor / skill bank. `K`, the baseline,
  retire with backoff, and state saved with the bank.
* **Telemetry:** one `infra/skills:` line per segment: trigger fires per
  minute per stream, agency pass rate, key count and top keys, candidates,
  mints, probation passes / failures, retired keys and their backoff.

## Step 2 — contracts (`tests/_skill_discovery_smoke.py`, CI + run_all)

* **A.** An agent-caused change fires the trigger; the same change caused by
  the world (every action predicts it) does not.
* **B.** Life boundaries never fire; `k_guard` holds.
* **C.** The baseline adapts: a period of high surprise raises the bar, and
  it decays back (no latch).
* **D.** Keys:
  * the same effect from different views goes to one key;
  * two different effects get two keys;
  * a key survives an encoder refresh (re-embedded).
* **E.** The fossil guard refuses `discovered_N` and admits `effect_N`;
  `required_effects` follows the keys.
* **F.** Probation:
  * a skill that reproduces its effect is kept;
  * one at baseline is retired;
  * a retired key is re-eligible after its backoff (the escape path).
* **G.** Wired once in both bodies. Every game composes the identical
  `goals.discovery` block (extends `_no_engine_reward_smoke`). No reward is
  paid by the trigger.

## Step 3 — live on main

1. **Doom, tiny x 8:** watch `infra/skills:`.
   * Success: at least one skill passes probation within the run, and the
     `learned_options` gate opens on a non-empty bank.
   * Judged against the answer key: what do the kept skills DO (kills per
     invocation, pickups, exits)? Per §9, not the reward number.
2. **Minecraft (enable + `host → connect`, 2 streams):**
   * The same code; `break_<blk>` is now only the grader.
   * Key purity against block types, and whether a log-felling skill
     appears — the original objective, never learned (§9).
3. Record the results in this file and in CLAUDE.md §10.

## Defaults to start from (fitted in step 0, then frozen)

| knob | start | meaning |
|---|---|---|
| `z_thr` | 3.0 | surprise spike, in baseline standard deviations |
| `s_floor` | 1.0 nat | ignore spikes of a near-collapsed latent |
| `agency_frac` | 0.25 | share of the surprise the own action must explain |
| `k_guard` | 8 steps | ignore after a life boundary |
| `tau` | 0.8 | cosine to join an existing effect key |
| `K` | 10 | probation invocations |
| backoff | 2 -> 4 -> 8 ... distinct episodes | re-eligibility after retirement |

## Risks, stated up front

* **A young world model is surprised by everything.** That is why step 0
  uses trained brains and why `s_floor` exists. Discovery on the live tiny
  brain will be quiet for its first hours, and that is expected.
* **Near-uniform policies** (all Stage C / warm-start policies were, at about
  2.4 of 2.77 nats) make probation baselines noisy. `K` may need to grow.
* **Slot pressure:** `max_slots` is 32; the repeatability gate and probation
  are what keep the bank from filling with noise.
* **Cost:** the batched prior on spike steps only; measured in step 0 (act ms
  per decision) before it reaches the live loop.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-006](../assurance/ass-006-realtime-action-lag-incident.md): Real-time action-lag incident (2026-10-18)
- [ASS-015](../assurance/ass-015-ab00-interim-analysis.md): AB-00 Doom A/A: interim analysis (16 of 20 runs)
- [ASS-018](../assurance/ass-018-campaign-progress-2026-10-23.md): Campaign progress check, 23 October 2026 (evening)
- [GOV-008](../governance/gov-008-human-like-learning-motivation.md): Human-like learning program: motivation, scope and change record
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md): Corrective plan after the AB-00 interim analysis: source diversity, intrinsic-signal repairs and a fresh control
- [RES-001](res-001-central-behavioral-problem-and-plan.md): Central behavioral problem and proposed plan
- [RES-006](res-006-human-like-learning-gap-analysis.md): Human-like learning gap analysis: SkyBot against the AI-counterpart learning map
- [RES-007](res-007-hard-and-deferred-differences.md): Hard and deferred differences: what cannot yet be built reliably
<!-- backlinks:end -->
