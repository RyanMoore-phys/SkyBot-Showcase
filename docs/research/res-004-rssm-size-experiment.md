# Does world-model size matter? — and the growing-brain "pyramid"

> **Document control:** [RES-004](../README.md#research) · Research / experiment protocol · **State:** Protocol; in progress · **Custodian:** Repository maintainer · Revision history: Git.

Status: **stage A VALID (2026-10-17): size does not matter yet (rule 1).** Posterior collapse fixed; autoscaler built; stage D built with actuation OFF. Stages B and C not started.
Owner question (2026-10-17): can SkyBot's RSSM start tiny and grow as it
masters its worlds — same brain throughout — with many small SkyBots early
(horizontal) and fewer, larger ones later (vertical): a *pyramid*? Before any
of that is built, test whether size matters at all.

## The idea, stated so it can be tested

* **Vertical (plasticity):** grow the world model's capacity in place —
  function-preserving (Net2Net-style: new units enter with zero outgoing
  weight, so the grown model computes exactly what it did) — when a measured
  signal says it is out of room. Fresh units are also a known remedy for
  *loss of plasticity* in long-lived networks, which a forever-learning agent
  is exposed to.
* **Horizontal (pyramid):** while the brain is small, spend the compute on
  MORE streams (more experience per hour); as it grows, shift compute back
  from streams to the model. Still **one brain** (non-negotiable): "more
  SkyBots" means more streams feeding it, not more brains.
* **The counter-argument to test, not assume:** DreamerV3 reported larger
  models are *more sample-efficient*, not only better at the end. A tiny brain
  could learn less per step while running more steps per second. Which wins
  per wall-clock hour on `main` (RTX 5050, 8 GB) is the actual question.

## Stage A — offline world-model size sweep (`tools/rssm_size_experiment.py`)

One fixed Doom dataset (sticky-random play of the real ViZDoom adapter, the
exact game config), split **by episode** into train / held-out. The REAL
`WorldModel` with the Doom run's composed `world_model` settings; only
`deterministic_size`, `stochastic_size × classes`, `encoder_hidden` change.
Every size gets the same batches in the same order.

| size | deterministic | stochastic | encoder | params |
|---|---|---|---|---|
| tiny | 256 | 16×16 | 128 | see results |
| small | 512 | 24×24 | 256 | 25.0 M |
| medium | 1024 | 32×32 | 512 | 52.7 M |
| live (today on `main`) | 2048 | 48×48 | 768 | 123 M (129 M with the flow head) |

Measured on held-out sequences only:

* **open-loop / copy-last**: observe 16 real steps, imagine 12 from the real
  actions, decode, pixel MSE — divided by the MSE of "the last frame never
  changes". **< 1.0 = the model predicts motion better than no model.** This
  is the quantity curiosity and dreaming live on.
* reconstruction and KL (what training minimises), seconds per step.

Mac deviations (recorded in the results JSON as `overrides`): Apple's MPS
backend cannot run the flow head (non-divisible area pooling, `grid_sample`
border padding), so **the flow head is off for every size alike**; proprio
and image sensors are not fed (pixels + actions only). The faithful version
is the same script on `main` with `--device cuda`.

### Decision rules (fixed BEFORE reading the results)

1. If tiny's open-loop/copy-last at equal steps is within **10 %** of live's,
   size does not matter at this stage of learning → prefer small + more
   streams now; growth is a later-stage tool.
2. If live beats tiny by **> 10 %** at equal steps but tiny reaches live's
   step-1000 number in less wall-clock → it is a throughput trade: stage C
   decides it.
3. If live beats tiny by > 10 % at equal steps AND at equal wall-clock →
   size matters; build the growing RSSM (stage D) and start it no smaller
   than the size where the gain flattens.
4. If NO size gets below 1.0× copy-last, the sweep says nothing about size:
   the world model is not yet learning dynamics on this data, and that is
   the finding to chase first.

## Stage B — live behaviour A/B on `main` (not started)

Two fresh universal brains, Doom-only, identical config except world-model
size (tiny vs live), same hours each. **Each arm in its own brain directory**
— never the live `brain_universal/`. Scoreboard (evaluator-only, RED): map
exits, highest map, kills per life, deaths per hour, distinct cells reached
(from the engine, for measurement only); plus steps/s, VRAM, WM loss.
Precondition: the coverage reward issue below is resolved first, or both
arms will mostly measure who farms coverage better.

## Stage C — the pyramid at fixed compute (main, 2026-10-18)

Same GPU, same 40 minutes of real-time Doom each, one after another; fresh
universal brain per arm in its own directory; the composed universal config
with only size and stream count changed; viewer off. Engine counts (deaths,
exits, cells) are measurement only. **One seed per arm.**

| | tiny x 8 | medium x 4 | live x 2 |
|---|---|---|---|
| world model params | 12.5 M | 54.8 M | 124.8 M |
| peak VRAM (nvidia-smi, whole GPU) | **3.4 GB** | 3.9 GB | 5.6 GB |
| transitions / s (all streams) | **57.1** | 40.0 | 23.9 |
| decisions / s per stream | 7.1 | 10.0 | 12.0 |
| seconds per decision (autoscaler, last 8 segs) | 155 ms | 108 ms | 91 ms |
| deaths / hour per stream | 55.5 | 54.6 | 55.9 |
| map exits | 0 | 0 | 0 |
| true cells reached per stream (sum / streams) | 13.9 | 9.0 | 16.0 |
| WM fresh-data loss, mean of last 8 segments | 0.0112 | **0.0091** | 0.0157 |
| WM training loss, same | 0.0097 | 0.0062 | 0.0069 |
| fresh / train (memorising when >= 1.5) | **1.18** | 1.54 | 2.32 |
| z-info (nats / frame) | 22 | 50 | 82 |
| replay ratio (frames trained per frame collected) | 15.6 | 24.6 | 39.3 |

* **No scoreboard separation in 40 minutes**, as expected: no arm exited
  MAP01, and deaths per stream-hour are identical (~55). Behaviour cannot
  decide the pyramid at this length.
* **What the pyramid buys is data per GPU-hour:** tiny x 8 collects 2.4x the
  experience of live x 2 at 61 % of the VRAM, and its world model GENERALISES
  (fresh/train 1.18) where live x 2 memorises (2.32: each frame reused 39x).
  medium x 4 had the lowest loss on fresh data. Caveat: each arm's "fresh"
  window is its own newest play, not one shared held-out set.
* **The cost is reaction time:** 8 streams share one process, so each acts
  at 7 decisions/s instead of 12. Doom tolerates that; a faster game may not
  (the autoscaler's latency budget rule exists for this).
* **The autoscaler's advice matched the data:** live x 2 and medium x 4 ->
  "memorising: more streams"; tiny x 8 -> once "out of room: grow" (a noisy
  0.58 gap; the 8-segment mean is 1.18). It also exposed a bug, fixed: in
  advice mode it advanced its own state as if obeyed, so live x 2 later
  logged "live x 3 -> live x 4". Now only an actuated change moves the state
  (`_autoscaler_smoke` G2).
* **All three arms ran WITHOUT episodic novelty** — a bug found by this run,
  not a design choice: the helper passed a CUDA tensor row to `np.asarray`,
  which raised inside `_view_key`'s bare except, so the term paid exactly 0
  on the GPU (it had only run on CPU). Income was imagination 84-91 %,
  view novelty 8-14 %, empowerment 0-2 %. Fixed (`_view_key` takes tensors,
  logs a failure once; `_no_engine_reward_smoke` G fails on the old code).
  The size/throughput/world-model rows above do not depend on it; the
  behaviour rows were measured under the wrong reward.
* Reading: consistent with stages A and grow-and-continue — start small with
  many streams, grow when the fresh-data loss plateaus while fresh ~= train.
  A medium x 4 start is the strongest single number here.

## Batch 12 vs batch 8 in the full loop (main, live x 2, 2026-10-18)

Offline, batch 12 + fused Adam + TF32 trained 46 % more frames/s. In the real
loop (15 min fresh brain, compared with C_live_x2's first 9 segments):

| | batch 8 (C_live_x2) | batch 12 |
|---|---|---|
| frames trained per frame collected | 33.2 | **56.5** |
| WM fresh-data loss | **0.0141** | 0.0178 |
| fresh / train | **1.40** | 1.89 |
| seconds per decision | **75 ms** | 93 ms |
| transitions / s | **23.9** | 21.8 |
| peak VRAM (whole GPU / torch) | **5.6 / 4.8 GB** | 6.0 / 5.3 GB |

**Not adopted.** The extra training went into memorising the same frames;
the world model got worse on new ones. With 2 streams the limit is data, not
training compute — the same conclusion as Stage C. Batch size is worth
revisiting only with more streams. One seed.

## Warm start: fresh vs policy_features vs kickstart (main, 2026-10-18)

Each arm resumes the SAME brain (C_live_x2's checkpoint, 40 min of play
under the old, episodic-less reward) for 30 min at live x 2, with episodic
novelty now working: so the reward CHANGED at the resume, the case warm
starting is for. World model and the other base components resume in all
three; only the policy differs. One seed.

| | fresh policy | policy_features (+x3 entropy, 20 updates) | kickstart |
|---|---|---|---|
| deaths / hour (2 streams) | 128 | **52** | 123 |
| true cells reached (sum) | 27 | 31 | 31 |
| map exits | 0 | 0 | 0 |
| FARM? flags in the log | 54 | **11** | 20 |
| policy entropy at the end (max 2.77) | 2.37 | 2.46 | 2.52 |
| feature layers' distance to the source (fresh init ~1.41) | 1.40 | **0.375** | 1.41 |
| action head's distance to the source | 1.41 | 1.43 | 1.42 |

* **policy_features did what it was built to do**: the feature layers stayed
  near the source (0.375) while the action head started fresh (1.43), and
  the arm died 2.5x less often (27 vs 66 / 64 deaths) and farmed less.
  One seed and 30 minutes — a strong hint, not proof.
* **Kickstarting showed nothing, for a measurable reason:** the teacher had
  learned almost nothing yet. On probe inputs every policy is near uniform
  (KL from the source 0.02-0.03 nats; 4-8 % same top action), so a pull toward
  it is a pull toward noise. Kickstarting can only be judged from a teacher
  that has actually learned something — a long-trained brain.
* **Decision:** keep `policy_features` in the universal resume list (it
  already is); kickstart stays built and OFF.
* `ledger_shares` in these arms' result.json are lifetime shares (measured
  before the ledger-reset fix); episodic was 74 % of them.

## Stage D — the growing RSSM (built 2026-10-17 at the owner's request; see below)

Minimal first version: grow the stochastic categoricals and the GRU width
with zero-initialised outgoing weights; grow every consumer of the latent in
the same step (PPO policy input, dream actor, curiosity heads, inference and
standby copies; skills are byte copies of the policy, CLAUDE.md §4.6); expand
Adam's moments; record the size in the checkpoint so resume rebuilds the grown
shape instead of tripping the size-mismatch refusal. LayerNorm makes growth
not exactly function-preserving — measure the output change, do not assume 0.
Trigger on a MEASURED signal (held-out WM error plateaued while novelty is
still arriving), never on "mastery" (unmeasurable, §4.6). Hard VRAM ceiling
with a stated behaviour at the ceiling (§4.1: growth is one-way).

## Grow-and-continue: the same brain, made bigger (main, 2026-10-18)

The owner's case: train a small brain, grow it, keep training — no new brain,
no lost training. Same 40k Doom frames, same batch order, fixed losses.
Held-out open-loop / copy-last:

| arm | @5000 | at the grow | @8000 / 9000 / 10000 | mean of last 3 |
|---|---|---|---|---|
| A: tiny 5000 -> grown to live (exact) -> +5000 | 0.749 | **0.749 -> 0.749** | 0.735 / 0.758 / 0.818 | 0.770 |
| B: live from scratch, 10000 | 0.818 | | 0.787 / 0.814 / 0.754 | 0.785 |
| C: tiny only, 10000 | | | 0.731 / 0.752 / 0.727 | 0.737 |

* **The grow loses nothing**: identical score before and after (max output
  difference 4.2e-4 = GPU float noise; cuDNN uses TF32 convolutions by
  default). 12.1 M -> 94.0 M parameters (the encoder keeps tiny's width).
* **The grown brain matches one that was big from the start** (0.770 vs
  0.785), having spent half its training at a tenth of the size.
* Consistent with stage A: at this stage size does not matter (tiny-only
  0.737). Run-to-run noise ~ +/-0.03; one seed, sticky-random data.
* Two bugs found on the way, both fixed: growing a model on CUDA mixed GPU
  and CPU tensors; the arch file failed to save NumPy integers from the live
  loop.

## Replacing coverage / gaze (graded on real Doom, 2026-10-17)

Coverage read the engine's player position (85 % of Doom income, farmed by a
strafe/walk/run loop); gaze read the engine's pitch. The user's rule (CLAUDE.md
§1) is one learning process for every game and nothing a player could not
see. Candidates were graded on 12,000 frames of Doom (sticky-random play
interleaved with scripted standing, spinning and the farm loop), with the
engine position recorded ONLY as the answer key. "Finds new places" = AUC of
the bonus for steps that truly entered unseen territory (0.5 = chance); the
other columns = bonus paid relative to exploring (lower = harder to farm).
Encoder: a tiny world model trained with the posterior-collapse fix.

| candidate | finds new places | standing | spinning | strafe loop | ms/step |
|---|---|---|---|---|---|
| A. view-novelty per step (live) | 0.68 | 0.38 | 0.79 | 0.60 | 0.002 |
| A'. view-novelty per visit | 0.66 | 0.12 | 0.92 | 0.59 | 0.001 |
| B. pixel-hash per visit | 0.70 | 0.22 | 1.01 | 0.77 | 0.002 |
| C. dead reckoning (body senses) | 0.49 | 0.05 | 0.02 | 0.16 | 0.006 |
| **D. episodic kNN (NGU) — chosen** | 0.60 | 0.15 | 0.98 | **0.375** | 0.18 |
| E. place-hash (8-step mean) | 0.54 | 0.16 | 0.82 | 0.41 | 0.005 |

Chosen by the user: **D** for coverage (least farmable on the loop that
farmed it), and **gaze removed** (every pixel option already pays for looking
around: spinning ~0.8-1.0x). Both OFF for the whole universal brain, Minecraft
included; the composer's per-game exception is deleted. Note C: the agent's
own dead-reckoned position is at CHANCE (0.49) in Doom — that body sense does
not track where it is (matches the ORACLE's ~95 % drift), a separate defect.
A first grading accidentally used a collapsed encoder (the deployed old code
was on the import path); its numbers were 0.75/0.67/0.82/0.96 for A and
0.64/0.18/1.01/0.40 for D — the ranking on the farm loop did not change.

## Stage A results — VALID (main, CUDA, posterior-collapse fix, 2026-10-17)

Same 40k-frame Doom data, live config (flow head on, no overrides),
`recon_reduction: sum`, `inverse_input: embed`, 5,000 steps per size, batch
8 x 32. Open-loop / copy-last on held-out windows (lower is better):

| size | params | s/step | @1000 | @2500 | @5000 | best | recon @5000 | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| tiny | 12 M | 0.147 | 0.807 | 0.731 | 0.758 | 0.731 | 0.0119 | 1.06 GB |
| small | 25 M | 0.153 | 0.873 | 0.807 | 0.781 | 0.778 | 0.0109 | 1.26 GB |
| medium | 54 M | 0.163 | 0.752 | 0.771 | 0.773 | 0.724 | 0.0112 | 1.72 GB |
| live | 123 M | 0.203 | 0.823 | 0.774 | 0.798 | 0.750 | 0.0106 | 2.97 GB |

**Decision rule 1 applies: size does not matter at this stage of learning.**
tiny is within 10 % of live at equal steps (ahead, 0.758 vs 0.798); the
spread between sizes is inside the run-to-run noise (live moved 0.757 ->
0.798 between steps 4000 and 5000). Bigger models reconstruct a little better
(live 11 % below tiny) but do not imagine better. tiny trains 1.38x more steps
per second; acting costs ~1 ms for 2 streams at every size. Caveats: one
seed, sticky-random play, 5,000 steps. Consequence: start the universal brain
SMALL with MORE streams (the pyramid's base) and let stage D grow it when the
autoscaler measures it out of room.

## Kernel-overhead options (main, live size, batch 8 unless stated)

| option | s/step | frames/s | peak | |
|---|---|---|---|---|
| baseline | 0.204 | 1,256 | 3.79 GB | |
| fused Adam | 0.184 | 1,390 | 3.79 GB | **shipped** (`world_model.fused_adam`) |
| TF32 + cudnn.benchmark | 0.201 | 1,272 | 3.79 GB | +1 %, global numerics: not taken |
| torch.compile(observe_step) | 0.242 | 1,059 | 5.68 GB | slower, +1.9 GB |
| CUDA graphs (reduce-overhead) | 0.242 | 1,058 | 3.95 GB | slower |
| fused + TF32, batch 12 | 0.210 | 1,832 | 4.85 GB | +46 % — needs ~1 GB headroom in the full loop |

## Stage D — built (2026-10-17), actuation OFF

* `developmental_ai/world_model/growth.py`: function-preserving growth of the
  RSSM (h duplicated k x with zero-sum split weights; new categoricals with zero
  outgoing weight; new classes never fire), every latent consumer transformed,
  any tensor without a rule REFUSED. Measured: outputs identical to ~3e-7 for
  h x2, h x2 + categoricals/classes, h x4 on a trained full-feature model; the
  duplicated units diverge under training (new capacity, not a clone).
* `tools/grow_brain.py`: grow (exact, verified on saved replay frames) or
  shrink (relearned from the saved replay buffer, accepted only within
  `--shrink-tol` of the bigger model's held-out error); backups; scale file;
  latent-sized resumed components moved aside.
* The composer builds the scale file's size and stream count at every launch;
  the loop (with `infra.autoscaler.actuate: true`) writes runlogs/SCALE_RESTART
  and stops gracefully; the supervisor relaunches `kind=scale` (never a crash);
  the launcher resizes before composing. Contracts: `tests/_growth_smoke.py`.
* Encoder width does not grow (the RSSM's capacity does): a brain grown to the
  `live` rung keeps its starting encoder.

## The KL floor / posterior collapse (diagnosed on `main`, 2026-10-17)

The first CUDA sweep (4 sizes x 5,000 steps) was declared invalid by the
owner: KL sat at its floor (0.110) for small and live the whole run. The
diagnosis measured **z-info** = unclamped KL(posterior || prior) summed over
latents = how many nats the latent carries about a frame.

Tiny model, same 40k-frame data, step 3,000 (held-out windows):

| variant | open-loop / copy-last | z-info | reconstruction |
|---|---|---|---|
| baseline (live losses) | 0.940x | **0.06 nats** | 0.0354 |
| **reconstruction SUMMED over pixels** | **0.764x** | **19.9 nats** | **0.0132** |
| inverse-dynamics head off | 0.939x | 0.02 | 0.0355 |
| DreamerV3 KL (sum, 1-nat free bits, 0.5/0.1) | 0.985x | 0.51 | 0.0374 |
| summed reconstruction + DreamerV3 KL | 0.782x | 17.9 | 0.0129 |

**Cause:** reconstruction was the MEAN over 49,152 pixel values (~0.035, a
gradient to match). The KL was clamped at its free-nats floor (zero
gradient) and the inverse-dynamics head dominated the total — so nothing paid
z for carrying information, and the posterior collapsed onto the prior.
DreamerV3 sums the image log-likelihood. Turning the inverse head off or
switching the KL alone does nothing; summing the reconstruction fixes it,
and adding the DreamerV3 KL on top did not help further, so only the
summed reconstruction shipped.

**Second defect, found on the way:** the inverse-dynamics head read
`[h, z]`, and `h_{t+1}` is computed FROM `a_t` — its loss fell to 0.000 by
reading the answer off its own input. Now it reads the encoder's observation
features (`inverse_input: embed`), which never see the action.

Both are config keys (`world_model.recon_reduction: sum`,
`world_model.inverse_input: embed`) set only in `configs/universal.yaml`; the
legacy Minecraft brain keeps the old model. Contracts:
`tests/_posterior_collapse_smoke.py`. A pre-fix universal world-model
checkpoint is refused cleanly at resume (the resume pre-flight now checks
tensor SHAPES as well as keys — before this, a reshaped checkpoint was
copied partway and left a partial hybrid).

## Python overhead (measured 2026-10-17)

* **World-model training step** is launch/overhead-bound, not GPU-bound, at
  the live batch (8 x 32): tiny 0.144 s/step, live 0.203 s/step on the RTX
  5050. Only ~16 Python-level GPU syncs per step; the cost is thousands of
  small kernels from unrolling 32 timesteps. Batch scaling (frames trained
  per second): tiny 1,780 (b8) -> 2,688 (b16) -> 3,434 (b32); live 1,263
  (b8, 3.6 GB) -> 1,983 (b16, 4.9 GB) -> OOM (b32). **The live run leaves ~half
  its training throughput unused at batch 8** — not changed yet: the live
  training process already used 5.7 GB of 8 GB at batch 8.
* **Acting loop** (cProfile, one real Doom segment, Mac CPU — proportions,
  not absolute times): ICM curiosity training 27 % (one gradient step every
  acting step — DELIBERATE: `curiosity.train_every` documents that batching
  it cost 48-79 % of the drive), world-model perception 11 %, infra per-step
  (mostly empowerment's imagined rollouts) 11 %, **viewer 9 % (fixed: panel
  composed at <= 10 Hz while watched, every 2 s when not)**, learning-progress
  reward 9 %, flow/senses/spatial ~10 %.

## Stage A results (Mac, 2026-10-17) — INVALID, re-run on `main`

* Data: 20,000 frames, 33 episodes of sticky-random play (MAP01), ~440 fps
  collection; 5 episodes held out.
* **tiny (11.5 M params) and small (25.0 M) both diverged to NaN before step
  200 on the Apple GPU (MPS).** A repeat on the same batches put the moment at
  step 107, every loss at once after a normal curve. **On CPU the same batches
  stayed finite past step 140**, so this is most likely MPS numerics, not the
  world model — not proven beyond 140 steps. The sweep was stopped; no size
  conclusion can be drawn from it.
* Seconds per training step on this M2: tiny 1.2 (MPS) / 5.5 (CPU), small 1.35
  (MPS), medium ~2.4 (MPS), live 3.4 (MPS) / 10.6 (CPU, with flow head).
  The Mac cannot run this sweep at a useful length: it moves to `main`
  (`--device cuda`). The script now stops a size at its first non-finite loss
  and records `diverged_at`.
* Seen on the way: the **inverse-dynamics loss sat at ~2.78 = ln 16**, chance
  for 16 actions, for all 140 CPU steps — about 95 % of the total loss. Worth
  watching in the `main` sweep: a head at chance contributes gradient noise,
  not dynamics.

## The reward issue found alongside (2026-10-17) — fixed

Doom's `coverage` income (85 % of all income, ledger ALARM for 52 segments,
a farmed strafe/walk/run loop) was computed from the ENGINE's player
position, and `gaze` (3 %) from its pitch — RAM reads the universal-brain
rule forbids for rewards. The RED sensor guard protects inputs only. The
first fix (the composer zeroing both outside Minecraft) was replaced the same
day under the one-learning-process rule: coverage and gaze are off for the
whole universal brain and episodic novelty pays every stream of every game
(see "Replacing coverage / gaze" above). Doom's coverage still counts visits
for measurement. Contract: `tests/_no_engine_reward_smoke.py`.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md): Predictive-coding core versus the RSSM: comparison protocol
- [ASS-020](../assurance/ass-020-recent-changes-static-audit.md): Recent changes: static implementation and documentation audit
- [GOV-005](../governance/gov-005-claims-evidence-register.md): Claims and evidence register
- [GOV-016](../governance/gov-016-predictive-coding-core-plan.md): Implementation plan: predictive-coding core versus the RSSM (WP-20)
- [RES-006](res-006-human-like-learning-gap-analysis.md): Human-like learning gap analysis: SkyBot against the AI-counterpart learning map
- [RES-010](res-010-predictive-coding-core.md): Predictive coding as the world model's latent core
<!-- backlinks:end -->
