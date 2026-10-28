# CLAUDE.md — working notes for agents on this repo

Read this before touching anything. `README.md` tells you where files are; this
tells you how to work here and what has already gone wrong. Nearly every rule
below was paid for by a multi-day run that produced nothing.

---

## 0. BRANCH OVERRIDE (owner, 2026-10-20) — read this first

This branch is a **different program** from the rest of this file's history:
the human-like learning program ([`docs/governance/gov-008-human-like-learning-motivation.md`](docs/governance/gov-008-human-like-learning-motivation.md)).
Where the sections below disagree with this block, **this block wins**. The
sections below stay as history and evidence.

- **LEARNED, NOT DECLARED — ALWAYS.** Meaning, skill, goals and checkpoints are
  earned from experience and never declared, scripted or told to the agent.
  This holds for every change on this branch, with no exception unless the owner
  explicitly changes it.
- **Universal brain only.** There is no legacy Minecraft-only brain on this
  branch ([ADR-004](docs/decisions/adr-004-retire-the-legacy-brain-on-this-branch.md)).
  Statements below that treat the legacy brain, its state, `runlogs/games.enabled`
  absent-means-legacy, "legacy unchanged" or `Minecraft -> legacy` as live no
  longer apply. Minecraft stays a valid *game* for the universal brain; its
  return is deferred until after Doom.
- **Everything is open to change and optimisation.** The standing principles in
  §1 (learned not declared, one universal brain, no engine reads, no steering)
  stay in force unless the owner changes them.
- **Success is adaptability to unfamiliar environments**, measured at
  owner-set checkpoints that are evaluation-only, never rewards
  ([ASS-007](docs/assurance/ass-007-adaptability-evaluation-protocol.md)).
- **A behaviour claim needs five seeds of one hour, positive on all five**, each
  above its region of uncertainty. Record null and negative results honestly.
- **When a change needs more compute:** reduce model size, then streams, then
  stepping rate ([ADR-005](docs/decisions/adr-005-compute-pressure-order.md)).
- **Streams may observe one another** within the limits of ADR-001 and ADR-002
  ([ADR-006](docs/decisions/adr-006-streams-may-observe-one-another.md)).
- **Two layers of change:** blocks ([GOV-009](docs/governance/gov-009-implementation-plan.md)) first, then the relationships between them ([GOV-010](docs/governance/gov-010-relational-implementation-plan.md)); a relationship is built once both blocks **exist**, before either is promoted. **Two test types, both A/B:** T1 tests blocks, T2 tests with and without the relationship ([GOV-010](docs/governance/gov-010-relational-implementation-plan.md) §4A).
- **Work one difference at a time** (one package per change; each A/B changes one thing, apart from declared minimal wiring). **The implementation plans now exist** ([GOV-009](docs/governance/gov-009-implementation-plan.md), [GOV-010](docs/governance/gov-010-relational-implementation-plan.md)); their **scope was approved by the owner on 2026-10-21**. **Do not change code until the owner issues the first instruction.** The target game is Minecraft (checkpoints: progress, survivability, thrivability, set later); see ASS-007 §15. Development evidence is not program evidence ([ASS-007](docs/assurance/ass-007-adaptability-evaluation-protocol.md) §14.1).
- **Build status (2026-10-21):** both plans are BUILT BUT UNTESTED in build-only mode. Every new behaviour is behind a default-OFF switch; nothing has been run, tested or promoted. Records: [GOV-011](docs/governance/gov-011-execution-record-gov-009.md), [GOV-012](docs/governance/gov-012-execution-record-gov-010.md) (they list the tests, mock runs and A/Bs still to do). The next step is testing, at the owner's instruction. **Superseded as a status statement** by the Testing-queue bullet below (2026-10-22 onward): the testing phase has begun on `main`, and no behavioural claim exists yet.
- **Testing queue (2026-10-22):** all A/A, T1/T2 and categorical testing is scheduled unattended on `main` by `tools/campaign.py` ([ASS-013](docs/assurance/ass-013-unattended-test-campaign.md)); its own contracts are `tests/_campaign_smoke.py`. The categorical system tests (CS-0..CS-2, `tests/run_all.py categorical`, [ASS-012](docs/assurance/ass-012-categorical-system-testing-protocol.md) §8) found 28 unfixed defects on their first run ([ASS-014](docs/assurance/ass-014-categorical-harness-first-run-findings.md)); two of them (F-13, F-24) invalidate queued A/Bs until fixed. Add a new test or arm there, not by hand-starting runs. **Tests are queued on `main`, not run on the development machine (owner, 2026-10-23).** The 24 corrections to the ASS-014 findings ([GOV-015](docs/governance/gov-015-corrective-plan-categorical-findings.md)) are in the code but unverified until the queued `gov015_tests` stages finish.
- **Entry point renamed (2026-10-25):** the trainer's entry script is `scripts/run.py` (was `run_minecraft.py` in the repo root; it was never Minecraft-specific). The frozen campaign code on `main` still has the old name and is not changed; `tools/campaign.py` matches both. Older docs and audits keep the old name as history.
- **WP-20, predictive-coding core (2026-10-25):** `world_model.core: pc` swaps the RSSM's amortised posterior for an iteratively settled, precision-weighted one whose prior, posterior and top-down networks learn by LOCAL Hebbian-form updates (`world_model/predictive_coding.py`; the encoder, GRU, decoder and heads still use backprop; `pc_learning: backprop` is the ablation); `rssm` stays the default and **both stay in the tree until the comparison returns** ([ADR-007](docs/decisions/adr-007-two-selectable-world-model-cores.md), [RES-010](docs/research/res-010-predictive-coding-core.md), [GOV-016](docs/governance/gov-016-predictive-coding-core-plan.md), [ASS-019](docs/assurance/ass-019-predictive-coding-core-comparison-protocol.md)). Built; status as of 2026-10-28 (details in GOV-016 sections 11c to 11j): `pc_tests` passed after contract G was corrected (the contract was wrong, not the rule); **the pilot gate FAILED twice** (attempt 1: G2, G3, G4; attempt 2, arms p2a/p2b: G3 reconstruction 2.1-2.2 x the RSSM's and G4 acting time 3.2-4.9 x; the first open-loop failure was largely measurement noise, so the failures that stand are reconstruction and cost); a faster learning rate collapses the latent, so slow learning is not the main problem; the generative head's local rule does not converge (diagnostics `pc_pilot3_diag`, `pc_pilot4_diag`); **the PC arm did not boot in the real trainer** (`pc_boot_check` ended `done` while its record said FAIL, GOV-016 11g; `pc_loop_tests` and `pc_boot_check2` are queued). A pilot pass would only license the five-seed A/B; the RSSM stays the default until that returns (GOV-016 11h). `t1_wp20_pc_core` stays blocked.
- **GOV-015 follow-up (2026-10-26):** the re-run of the red categorical scenarios found one real code defect (WP-8 scored a curriculum context change as lost progress; fixed by `CompetenceController.note_context_change`, [GOV-015](docs/governance/gov-015-corrective-plan-categorical-findings.md) section 9.7) and test defects in the rest; `gov015_verify` confirmed the fix (new green, old red) but five categorical scenarios and a contract-tier traceback stayed red; causes found (macOS `._*` files left in the staged trees by `tar` made a source-walking scenario read a binary stub; two scenarios whose `brk` flag did nothing; two test edits of mine), fixes made and unverified, `gov015_diag2` queued ([GOV-015](docs/governance/gov-015-corrective-plan-categorical-findings.md) sections 9.9 and 9.10). **Staging rule: `COPYFILE_DISABLE=1` for every `tar`, and copying `tools/campaign.py` into the frozen `code/` changes the evaluator's recorded hash (re-sync it).** The telemetry suite was red because two tools shared the name `tools/learning_report.py` (the per-segment one is now `tools/segment_report.py`; GOV-015 section 9.12). **The WP-20 arm did not boot in the real trainer** (`pc_boot_check` ended `done` while its record said `FAIL exit code 3`; GOV-016 section 11g): read a stage's record, not only its status; `pc_boot_check2` is queued with the logs kept. The older failed stages are reviewed in ASS-017 section 8.
- **Corrective items (2026-10-22):** after the AB-00 interim analysis ([ASS-015](docs/assurance/ass-015-ab00-interim-analysis.md)) the owner chose source diversity (WP-6), the F-13/F-04 repairs and a fresh-control arm; plan, predictions and execution record are [GOV-014](docs/governance/gov-014-corrective-plan-after-ab00.md). The metric change and the random-policy reference were declined.
- **Documentation protocol:** follow `docs/README.md` (IDs, naming, register,
  `tools/check_docs.py`). **Cross-references are reciprocal**: after changing links run
  `python3 tools/doc_backlinks.py`; never edit a generated "Referenced by" block. Retired files live in `legacy/`
  and `docs/legacy/` (nothing there is a current instruction). Results are appended, never edited into a protocol.
- **One hour = one complete seed run** (source training, target exposure and
  everything), so each seed gives enough data to judge a change on short
  segments and to promote changes that make SkyBot learn quicker
  ([ASS-007](docs/assurance/ass-007-adaptability-evaluation-protocol.md) §12).
- **Minecraft-only code is removed** unless it is literally the game interface
  (the per-game scripts and configuration every game needs). Anything else that
  cannot be made universal is removed on judgement; what can be is translated
  ([ADR-004 Amendment 1](docs/decisions/adr-004-retire-the-legacy-brain-on-this-branch.md)).
- **Streams may observe a peer's frames, sound and chosen actions**, never its
  engine state (ADR-006 Amendment 1).
- **Adding a game must be easy** (GOV-008 WP-16): only a game interface and a
  checkpoint detector, with no change to the learning code.
- §3 still applies: no commits, branches or deployment unless the owner asks.

---

## 1. What this is

A curiosity-driven neurosymbolic RL agent ("SkyBot") that learns Minecraft
**from scratch — no demos, no videos, no recipes**. The stack:

| Piece | Where | Role |
|---|---|---|
| DreamerV3-style RSSM world model | `world_model/rssm.py` | predicts latents; the dream substrate |
| PPO actor-critic | `policy/actor_critic.py` | the shared policy all skills copy from |
| Options / SMDP | `policy/options.py` | skills as temporally-extended actions |
| ICM + learning-progress curiosity | `curiosity/` | the intrinsic drive |
| Vision magnet + fovea | `llm/vision_scaffold.py` | curiosity-ranked visual seeking |
| Local VLM (qwen2.5vl:7b via Ollama) | `llm/vlm_symbolizer.py` | symbol grounding, unstuck advice |
| Knowledge graph | `knowledge_graph/` | asserted/retracted facts |
| General infrastructure | `infra/` (wired by `infra/stack.py`) | gates, ledger, signal health, empowerment, episodic, affordance, advisor |
| The loop that owns all of it | `core/developmental_loop.py` (**9.4k lines**) | reward assembly, stepping, training |

**The second standing principle (user, 2026-10-17): ONE LEARNING PROCESS FOR
EVERY GAME.** Every learning component — each curiosity and reward term, every
world-model loss, every infra drive — runs identically in every game and ROM
the universal brain plays. No per-game omissions, no `if game == ...` reward
forks. When a term only works in one game because it reads that game's engine
(position, pitch, RAM), the fix is to re-derive it from what any player has —
pixels, sound, own-body senses, buttons — so it runs everywhere, NOT to switch
it off per game. The composer's "coverage/gaze = 0 outside Minecraft"
stopgap (2026-10-17) violated this and is gone: coverage and gaze are off for
the whole universal brain and episodic novelty pays every game (§10). The rule
covers TELEMETRY too: a Minecraft-only `if` around shared income lines hid
them, and broke the ledger, in every other game (§10, 2026-10-18).

**The standing principle:** meaning and skill are *earned from experience, never
declared*. The agent gets pixels and buttons and has to find out what they do.
This is enforced by `tests/_no_scripted_skills_smoke.py`, which fails the build
if a scripted option reappears in the config. If you think you need a scripted
macro — you don't; see §4.1.

---

## 2. Running things

**Always use the repo venv.** System `python3` lacks `yaml`, `torch`, everything.

```bash
cd <repo root>                                     # your local checkout
./venv/bin/python tests/_gui_farm_smoke.py        # Python 3.10 (was 3.9.6; measured 3.10.22 on 2026-10-14)
```

**Known Mac-only red:** `_incident_bundle_smoke` contract I (singleton
supervisor) times out on macOS because there is no `flock`; it fails the same
way on an untouched HEAD. It is meant to pass on Linux. It used to leave its
fake `scripts/run.py` agent running even when it PASSED (contract I's
`release` file was deleted with the temp dir before the agent saw it): three
such stubs sat on `main` (2026-10-18), and any one makes `launch_skybot.sh`
refuse a real launch with "run alive". It now reaps every process in its temp
roots (`/proc` cwd on Linux, `lsof` on macOS); verified clean on `main`.
If in doubt: `pgrep -fa "incident_smoke|supervise_skybot"`.

Tests are **standalone `__main__` scripts**, not pytest. Each prints numbered
contract lines and ends with `[name] ALL PASS`. Since 2026-09-19 there IS a
runner, `tests/run_all.py`, in three tiers — but it only shells out to those
same scripts, so running one directly still works and is still the right move
when you are iterating:

```bash
PYTHONPATH=. python tests/run_all.py unit       # parts: shapes, bounds, ~3s
PYTHONPATH=. python tests/run_all.py contract   # design arguments, ~10s
PYTHONPATH=. python tests/run_all.py legacy     # needs gymnasium -> host only
PYTHONPATH=. python tests/run_all.py all        # exit code = failing suites
```

**The two tiers mean different things.** A `unit` failure is a bug in a
function. A `contract` failure means a DESIGN CLAIM stopped holding — those
docstrings name the live incident and the measured numbers, and one going red
is a finding, not a chore. A `.pth` file in the venv makes `import developmental_ai`
work from any cwd, but prefix `PYTHONPATH=.` anyway — that's the documented form
in every test docstring, and it's what works on the training host.

Training does **not** run on the Mac. It runs on the owned host `main` (not a rented pod)
(`scripts/launch_skybot.sh`, kept alive by `scripts/supervise_skybot.sh`).
See [`docs/operations/ops-005-operations.md`](docs/operations/ops-005-operations.md).

### Stopping a run

`touch runlogs/STOP` — graceful, and the supervisor won't restart it. Do **not**
`kill` the python process: the supervisor treats that as a crash and relaunches.
Also note `pkill -f "tailscale nc <ip>"` will kill the **socat bridge** too
(socat's cmdline contains that string) — this has stranded a run before.

---

## 3. Standing user instructions

- **No git commits, no branches, no deployment ceremony** unless explicitly
  asked. Work and edit in place. (The repo has exactly one commit by design.)
- **Only commit training host-proven changes** — if it hasn't run live, it isn't proven.
- **Never wipe `skill_bank_mc_curiosity/`** on the training host. It is the agent's
  accumulated developmental memory.
- Peer learning between SkyBots is a *feature*, not contamination. Multiple
  agents on one server is a deliberate social-learning experiment.

---

## 4. The recurring bug classes

These have each bitten multiple times. Check for them **by name** in review.

### 4.1 Guard-becomes-latch (≥9 occurrences)

A guard added to stop a bad behaviour becomes a permanent trap because nothing
can ever satisfy it again.

- The competence gate froze at an inherited bias of −5.68 → **all 16 skills, 0
  invocations, forever**.
- `disable_macros: [10]` removed `inventory` — the *only* GUI exit — while
  `use` (which opens villager/chest screens) stayed enabled. SkyBot sat in a
  wandering villager's trade menu for **10,149 consecutive steps**.
- The magnet's cold-start BUDGET latch caused **19 hours of zero reward**.
- The degenerate-signal gate counted *labels* instead of *positives* and
  disabled the magnet live.

- **(2026-10-14, four more, all in the universal-games / Actions work):**
  - A SIGKILLed trainer left its seats' Xvfb running; `seat.py` refused any
    locked display, so after one hard crash **both doom_player seats refused
    on every relaunch**. Fixed: each Xvfb carries `SKYBOT_SEAT_OWNER=<trainer
    pid>` and `Seat.clear_stale_display()` clears dead/orphaned/own locks,
    refusing only another live trainer's or a non-SkyBot X server.
  - A Doom/Player engine or screen capture that died left its stream **dead
    for the whole run** (frozen frame + restart flag every step): the
    lifelong body assumed every env "rebuilds its own world" — MineRL only.
    Fixed: `_maybe_heal_stream` (see §4.2).
  - A failed game `install` with `restart` on stopped training and then
    skipped the relaunch: **training stayed off until a human noticed.**
    Fixed: relaunch runs when stop + sync succeeded, whatever the action did.
  - The opposite edge: the stop step succeeds instantly when nothing runs, so
    any game action with `restart` on **started training on a host someone
    had stopped on purpose.** Fixed: only a run that was live is relaunched.

**Rule:** every guard needs an escape path that the agent can actually reach,
and you must state what re-opens it. If the answer is "a human notices," it's a
latch.

### 4.2 Duplicated-body drift

`developmental_loop.py` steps the environment in **two places**:

- `_run_episode_parallel` (~line 3382)
- `_collect_segment` (~line 4282) ← this is the one SkyBot actually runs

Reward logic is duplicated between them. An edit that lands in one and not the
other is silent — it just doesn't apply live. Use `replace_all: true`, then
**grep to confirm the count is 2**. Several tests assert exactly this, e.g.
`assert src.count("_gui_now2 = bool(") == 2`.

**A body can also share a wrong ASSUMPTION (2026-10-14).** The lifelong body
treated every raised step as "the wrapper rebuilt its own world" — true for
MineRL only. A ViZDoom or Player stream whose engine died raised forever and
was never reset. Now `_collect_segment` counts raised steps per stream
(`_stream_fails`) and calls `_maybe_heal_stream` (bounded rebuild, sliding
`parallel_envs.heal_max_per_hour`, default 6) after
`heal_after_failures` (default 3) or at once when a reset raises. The
episodic body already heals at its boundary via `_parallel_reset`.
`tests/_dead_stream_smoke.py` asserts the wiring.

**A Minecraft-only gate around SHARED code (2026-10-18).** In the progress
printer, every per-term income line AND the per-segment reset of its sum
(coverage, novelty, episodic, imagination) sat inside `if _nb or _bbt:` —
the chop diagnosis, true only when Minecraft break counters exist. In Doom
none of it ran: no line, and the sums were never reset, so the ledger read
lifetime totals as one segment's. Now its own block, outside the gate
(`_no_engine_reward_smoke` H). When you indent new code, check what the
enclosing `if` is actually about.

**There is a THIRD body, and it is unwired** (found 2026-09-22): `_run_episode`
(~line 3645), the single-env path. It has **none** of Waves 1–2 or Phases 1–7 —
no `info["sensors"]`, no `_augment_proprio`, no `_wm_proprio_batch`, no
`_flow_senses`, no `_spatial_step`, no `_phase_mark`. Reaching it is easy and
silent: `parallel_envs.num_envs <= 1` sets `_use_parallel_envs = False`, which
raises if `lifelong.enabled` is true and otherwise drops straight into it.
**So `num_envs: 1` is not "the same agent with one client"** — it is the
pre-Wave-1 agent. Keep `num_envs: 2`, or port the wiring first.

### 4.3 The γ-discounting defect in state costs

A potential written the textbook way, `F = w(γΦ′ − Φ)`, pays **`−w(1−γ)Φ` every
step Φ is held constant**. For a *cost* potential (Φ < 0) that flips the sign:
sitting still becomes a **wage**.

This shipped once and the log showed it: `Gaze level: +0.00014/step` — the agent
was being *paid* to stare at the pitch clamp.

**Rule:** telescoping potentials are for *progress*. A **state cost** must use
the **plain difference** (`γ = 1`), so it charges on entry, refunds on exit, and
pays exactly **0** while pinned. `tests/_gui_farm_smoke.py` encodes both forms —
the flawed one is kept as a regression witness.

Corollary: a telescoping potential **structurally cannot discourage dwelling**.
If you want dwelling to hurt, you need a genuine per-step cost, not a potential.

### 4.4 Reward-channel confusion

`intrinsic` is **zeroed while a GUI is open**. Any penalty you put there is
erased exactly when you need it. GUI/dwell costs must land in `prim_extrinsic`,
*before* the zeroing line. Meanwhile the magnet's shaping was added to
`prim_extrinsic` and so kept paying through an occluded camera — that leak was
77% of all income during the villager incident.

**Before adding any reward term, answer:** which channel, is it gated, does it
survive occlusion, and what does it pay while the agent does nothing?

### 4.5 One-way doors in the action space

Never offer an action that *opens* a state without the action that *closes* it.
Enforced by `test_no_one_way_doors` in `_gui_farm_smoke.py`.

### 4.6 Skills are copies, not memories

A minted skill is a **byte copy of the shared policy** (measured cosine
0.869–1.000; three pairs bit-identical; slot signal attenuated 20×). So:

- **Never key effects on a skill's `name`** — the name carries no weight.
- Merging the skill bank **does not stick**; skills re-derive from goal
  `slot_keys`.
- "Mastery" is currently unmeasurable (3 of its 4 ingredients don't exist).

---

## 5. Debugging discipline

**Measure before theorizing.** This project has burned days on plausible
theories. Two examples worth internalizing:

- The 19-hour zero-reward stall "was curiosity exhaustion." It was a cold-start
  budget latch.
- The `places` counter looked like it proved wasted no-ops. It lists
  `iron_axe: 2281` and `acacia_door: 3004` — the heuristic is **garbage**. Don't
  build an argument on a counter you haven't validated.

**Falsify your own fix.** The gaze-level potential above was caught by a test
written for a *different* bug. Write the test that would catch you being wrong,
not the one that confirms you're right.

**When you find a real failure, encode it as a test.** The convention here is a
long docstring naming the live incident, the measured numbers, and the contracts
— see `_gui_farm_smoke.py` and `_no_scripted_skills_smoke.py`. That's why
reversing a principle requires deleting a test that explains itself, rather than
quietly appending four lines of YAML.

**Don't trust `STATE.md`.** It has been stale and sent a session chasing a
terminated host while the live one ran elsewhere. Verify against the running
system.

---

## 6. Live infrastructure

- **HOST ACCESS: read [OPS-008](docs/operations/ops-008-host-access.md) before connecting to anything. Never guess a login, key, address or port; the real values are only in the untracked CLAUDE.local.md and the runner `.env`. Everything below that mentions `root@`, a rented pod, `POD_*` or a rotating port is history.**
- **Training host (since 2026-09-22): the user's own main computer**, not a
  rented training host. Ubuntu Server at its LAN address (`MAIN_HOST` in the runner `.env`;
  real IPs live in the untracked `CLAUDE.local.md`, never in this public repo), on the **LAN and the tailnet
  at once**. RTX 5050 (8 GB), Ryzen 5 5500 (6c/12t), **16 GB RAM**, 256 GB NVMe
  — far tighter than the training host it replaced (16 GB VRAM, 128 cores, 125 GB RAM),
  and several config values were cut to fit (see below). Everything still
  installs at `/workspace/devai` and deploys as `MAIN_USER` (the login recorded in CLAUDE.local.md; never assume `root` or `skybot`, see OPS-008), deliberately: the box
  is shaped like a training host so almost nothing needed changing. **One thing did, and
  it cost a failed deploy (2026-09-22): `/workspace` itself.** On a rented GPU host that
  was the platform-mounted network volume, present before any script ran —
  nothing in this repo has ever created it, and `provision_host.sh` opens with
  `cd /workspace/devai || exit 1`, so it cannot be what makes it either. On an
  owned box nobody mounts it, and `rsync` creates only the LAST path component,
  so the deploy died on `mkdir "/workspace/devai" failed: No such file or
  directory (2)` — **exit code 11, which reads like a permissions problem and
  is not one.** `deploy_skybot.sh` now creates it and prints the filesystem and
  free space, because a silently-created `/workspace` on the root disk is also
  how you would learn far too late that a data volume failed to mount.
  The former rented GPU host is off. Its IPs, once listed in an external
  runbook, are **stale**.
- **Everything targets the host via the runner's `.env`**, never a value in the
  tree: `MAIN_HOST`, `MAIN_SSH_PORT`, `MAIN_SSH_KEYFILE`, `MAIN_USER`,
  `MC_SERVER_TS_IP`. To
  move hosts again, edit `.env` — not the repo. See `docs/operations/ops-001-ci-setup.md`.
  **These were `POD_*` until 2026-09-22** (138 occurrences across 14 files).
  Renamed because the name had stopped describing the thing: there is no training host,
  the target is the owned box `main`, and the port no longer rotates. There is
  deliberately **no `POD_*` fallback** — every consumer guards with `:?` and
  names the missing variable, so a stale `.env` fails loudly at the preflight
  instead of expanding to an empty-user `@host` and reporting `Could not resolve hostname`.
- **`MAIN_USER` (added 2026-09-22) — the login account, was hardcoded `root`.**
  a rented GPU host only ever gave you root; `main` is a normal Ubuntu box, so it is
  `skybot`. Default stays `root`, so a root host is unchanged. **A non-root
  `MAIN_USER` REQUIRES PASSWORDLESS SUDO on the target** — provisioning installs
  apt packages and creates `/workspace` under `/`. `provision_host.sh` now routes
  every privileged command through `$SUDO` (empty when already root) and aborts
  at the top with a `sudoers.d` recipe if `sudo -n true` fails; it runs detached
  under `nohup` with no tty, so a password prompt would not block visibly — it
  would just fail stages silently amid `set -x` noise.
- **Keep `MAIN_HOST` an IP, not the name `main`.** The runner on `node1` is
  **dockerized**, and `docker/runner/docker-compose.yml` sets no `extra_hosts`
  and no `dns`. A container does not inherit `node1`'s `/etc/hosts`, and mDNS
  (`main.local`) needs an avahi client it does not have — so a name that
  resolves perfectly in your shell fails inside the container as `Could not
  resolve hostname`. Use `<login>@main` (login in CLAUDE.local.md) in hand-typed commands; leave
  the `.env` on the address. To use the name there too, add
  `extra_hosts: ["main:${MAIN_HOST}"]` to the runner service first.
- **`podlogs/` is now `runlogs/`, and every `pod_*` file is `host_*` (2026-09-22).**
  267 references across 61 files; `host.yml` became `host.yml` (safe — `deploy.yml`
  triggers on `workflows: ["ci"]`, never on this one). **THE TRAP IS BRAIN STATE,
  NOT CODE.** `break_memory_path` lives under that directory, so any backup taken
  before this rename carries `podlogs/` paths: restoring one onto a renamed tree
  silently starts the break memory EMPTY rather than erroring, and the agent
  re-opens every mastered block tier at full worth. On restore, `mv podlogs
  runlogs` FIRST. This was safe to do now only because `main` was unprovisioned
  and the old host was already off — there was no live state to orphan.
- **`workflow_run` RUNS THE WORKFLOW FILE FROM THE DEFAULT BRANCH (2026-09-22).**
  `deploy.yml` triggers on `workflow_run: workflows: ["ci"]`, so GitHub executes
  **`main`'s copy of `deploy.yml`**, never the branch you pushed. Its
  `actions/checkout` still pins the CODE to the triggering sha — so you get the
  new scripts running under the OLD workflow, which is a genuinely confusing
  split. It cost a red deploy right after the `POD_*`->`MAIN_*` rename: `main`
  still demanded `POD_HOST`, and the error text was the pre-rename wording,
  which is the tell. **Editing `deploy.yml` on a feature branch does nothing
  until it lands on the default branch.** The stopgap is putting BOTH old and
  new names in the runner `.env`; the fix is merging the workflow to `main`.
- **The runner lives on `node1`, a DIFFERENT machine from the training host.**
  `node1` runs the self-hosted Actions runner and holds the only `.env`; `main`
  (address: runner `.env`) is the box everything deploys to. Its `hostname` really does
  print `main`, which is the quickest way to tell which one you are sitting on.
- **Game server:** the user's own Paper server, on **another LAN machine**,
  reached through `socat` → `127.0.0.1:25565` exactly as before. The bridge now
  has **two far sides** and picks by address family: RFC1918 goes over plain
  TCP on the LAN, everything else (tailnet 100.x, MagicDNS names) keeps
  `tailscale nc` untouched. **Tailscale is fully retained** — it is the path
  again as soon as ethernet is back, via one `.env` value. The endpoint the
  agent sees is `127.0.0.1:25565` either way, which is why
  `environment.remote_server` never changes. The *server* is the client-count
  bottleneck, not the host: 4 MineRL clients produced 0 segments in 17 minutes.
  2 is the proven number, and `remote_server_scope: all` puts **both** streams
  on the Paper server (no stream runs a local generated world).
- **What the 16 GB box forced (2026-09-22).** `buffer_growth` was capped at
  **2 GB** (`hard_max_gb` 16 → 2, `max_ram_frac` 0.6 → 0.15) and
  `block_transitions` 25000 → **5000**. That last one is not cosmetic: the
  **initial block is allocated per stream WITHOUT a budget request** —
  `ReplayBuffer.__init__` allocates outright and only `maybe_grow()` calls
  `MEMORY_BUDGET.request()` — so real RSS exceeds `hard_max_gb` by
  `num_streams × block_transitions` always. At 25k blocks a "2 GB" ceiling
  measured **3.70 GB**. Revisit `block_transitions` whenever `hard_max_gb`
  moves. `OMP_NUM_THREADS` is 6, not 16.
  **FIXED 2026-10-18 (mock deploy):** 8 Doom streams held **5.6 GB under a
  "2.0 GB" budget** — the first blocks AND the proprio column (30 % of a
  row, missing from every grow request). The budget now counts both
  (`MEMORY_BUDGET.charge`, `bytes_per_row`) and N streams' first blocks are
  shrunk to fit it, so `hard_max_gb` is the real ceiling
  (`_replay_budget_smoke`, census contract C). The legacy brain's replay is
  now a true 2 GB; the universal brain sets 5.5 GB (§10).
- **Ollama** serves the VLM on the same GPU. `fovea_interval: 12` cost **28% of
  the step rate**; 30 is the tuned value. VLM frequency trades directly against
  training throughput.
- **Backups:** brain state (`world_model.pt`, `symbolizer.pt`, `familiarity.pt`,
  `magnet.pt`, `knowledge_graph.json`, `options_state.json`, break memory, skill
  bank) lives **only on the training host**. A training host died with ~11 days of
  unbacked state. Moving to owned hardware removes the *training host-terminated* risk
  and replaces it with *disk failure* — 256 GB, one NVMe, no redundancy — so
  the habit still stands: **rsync the brain host→Mac periodically during long
  runs.** Code always flows Mac→host, so code is never at risk.
  **`pull_brain.sh` does not exist** despite `deploy_host.sh`'s header citing
  it; nothing automates this yet.

---

## 7. MineRL / environment gotchas

- `mine_block` stats **restart at 0 on every reset** — a new world per mission.
  Use per-episode marks; never baseline against a running total.
- `FlatInventoryObservation` gives item→count with **no slot indices**. There is
  no equip action and `equipped_items` is dead in MineRL 1.0 — mainhand must be
  derived from evidence (see `_last_placed_item` in `minerl_env.py`).
- World persistence is **impossible in this fork** (measured). Each episode is a
  fresh world; anything that must persist has to live in the brain, not the map.
- Multi-agent usernames go through `handlers.MultiplayerUsername` on a
  **per-instance** spec. Slot 0 keeps the bare name `SkyBot` (the offline UUID
  owns that player's data).
- Boot hangs are solved by **VirtualGL EGL + per-core `taskset`** (an NVIDIA/AMD
  memcpy race); this is baked into `provision_host.sh`.

---

## 8. Where to look

| Question | File |
|---|---|
| Why isn't it acting? | `docs/assurance/ass-002-action-stall-audit.md` — 30+ ranked issues |
| Domain-independent redesign | `docs/architecture/arc-003-general-infrastructure.md` — 57 changes |
| Open findings | `docs/governance/gov-003-audit-findings.md` |
| Depth/flow from motion | `docs/research/res-002-perspective-learning.md` — waves 1-2 |
| Sensors, slots, action space | `docs/research/res-003-plurality-roadmap.md` — phases 1-7 |
| How any of it gets proven | `docs/assurance/ass-005-testing-plan.md` — stages 0-5 |
| What's next | `docs/governance/gov-002-next-objectives.md`, `docs/governance/gov-001-research-roadmap.md` |
| Host ops | `host_repository/docs/{RUNNING,PROVISIONING,TRANSFER,RECREATE}.md` |
| Live config | `configs/minecraft_skybot.yaml` (the only one that matters) |
| One brain, many games (Doom, …) | `docs/architecture/arc-004-universal-games.md`; `configs/universal.yaml` + `configs/games/*.yaml` |
| Steam / browser / exec launchers | `docs/architecture/arc-004-universal-games.md` §11.1; `developmental_ai/environments/player/launchers.py`; examples in `configs/games/examples/` |
| Installing a game, the game depot | `docs/architecture/arc-004-universal-games.md` §12, `docs/operations/ops-005-operations.md` "Game depot"; `tools/game_install.py`, `scripts/depot_sync.sh` |
| Which Actions workflow does what | §10 below |
| Skill discovery (surprise x agency, effect keys, probation) | `docs/research/res-005-skill-discovery.md` |
| World-model size, the pyramid, growth | `docs/research/res-004-rssm-size-experiment.md` |

---

## 9. The honest scoreboard

Keep this in view; it's the point of the project.

- **399 blocks broken in host history → exactly 1 log, ever.** Later runs reached
  13,305 breaks with **35 logs**; all log skills sit at **0/20**.
- Chopping a tree — the original objective — **has never been learned**.
- Most historical "progress" was the agent finding a way to get paid for doing
  nothing: staring at the sky (96% of drive), sitting in a menu (77% of income),
  holding attack against an unreachable trunk (96% of option activity, 0 logs).

When you evaluate a change, ask what it does to *that* scoreboard — not to the
reward number. The reward number has been wrong every single time.

---

## 10. GitHub Actions, games and installs (2026-10-14)

### The five workflows

| Workflow | Trigger | Runs on | What it does |
|---|---|---|---|
| `ci` | every push / PR | GitHub-hosted | the test gate. Since 2026-10-14 it also runs the universal suites (`_universal*`, `_vizdoom`, `_rewind`, `_player_adapter`, `_launchers`, `_game_install`, `_dead_stream`) — before that NONE ran on Linux, though CI installed vizdoom "for" them. |
| `deploy` | after `ci` passes, or by hand | node1 runner | sync code + 4 host-side smokes. **Refuses while training is live** (a red `deploy` after a push during a run is that guard working). |
| `host` | by hand | node1 runner | `status` / `connect` / `launch` / `stop` / `provision`. |
| `Doom` | by hand | node1 runner | `engine`: `vizdoom` (game `doom`) or `player` (game `doom_player`, real Chocolate Doom). `status` / `install` / `enable` / `disable`. |
| `Minecraft` | by hand | node1 runner | `status` / `enable` / `disable` / `legacy`. |

Removed on 2026-10-14: `pipeline` (stop→deploy→connect→launch + a daily
cron health check — it duplicated `host` + `deploy`), and `game-doom` +
`game-doom_player` (merged into `Doom`); `game-minecraft` became
`Minecraft`. The "Dependency graph" run by `dependabot[bot]` is **not a file
in this repo** — it is GitHub's *Automatic dependency submission* setting
(Settings → Code security), and can only be turned off there.

**Order:** once per bare box `host → provision`; once per game `Doom →
install` (**install never stops a run** since 2026-10-16: it uses the code the
last `deploy` put on the host; only `enable`/`disable` stop + relaunch,
because the games file is read at launch); per code change `host → stop` (if live) → push (`ci` → `deploy`) →
`host → connect` (if Minecraft is enabled) → `host → launch`. A game action
with `restart` on does stop → sync → change → relaunch itself, and relaunches
**only if training was running**. Before enabling Minecraft, `host → connect`.

### Legacy vs universal brain

`runlogs/games.enabled` decides at every launch. **Absent** = the legacy
Minecraft-only brain (`configs/minecraft_skybot.yaml`, its accumulated state).
**Present** = the universal brain under `brain_universal/`, which **starts
fresh**. The first `enable` of any game creates the file; `Minecraft →
legacy` deletes it (kept as `.bak`) and is the only way back. Enabling Doom
from a legacy host gives a Doom-only universal brain — Minecraft is not added
for you. `host → launch` always starts ONE trainer; which games it plays is
that file.

**The legacy brain's STATE IS NO LONGER ON `main` (owner, 2026-10-18).**
Archived to the Mac (`~/SkyBot_archive/legacy_brain_main_2026-10-18/`,
2,939 files, sha256-verified on both machines, `RESTORE.txt` inside), then
removed: `logs/checkpoints/`, `skill_bank_mc_rssm/`,
`runlogs/foundation_shadow/`, `runlogs/breaks_by_type.json` and the legacy
infra files in `runlogs/` (anticipation/consequence state, decision traces,
help requests) — 4.2 GB, nothing the universal brain reads. So `Minecraft →
legacy` now starts an EMPTY legacy brain unless the archive is restored
first. `configs/minecraft_skybot.yaml` stays: it is the BASE the universal
config is composed on top of, and `mc-build/` stays: every brain needs it
to play Minecraft. Also deleted: `logs/mc_928.log` + `mc_929.log`, 6.1 GB of
MineRL client stdout (2026-10-08, before the stat-name filter loaded);
`logs/mc_*.log` is now capped at 500 MB each by `scripts/cap_log.sh` (glob
mode) whenever Minecraft is enabled (`_cap_log_smoke`).

### Installing games: `tools/game_install.py` and the depot

- What a game needs lives in its YAML `game.install` (`apt`, `apt_optional`,
  `debs`, `pip`, `roms`, `files`, `note`); `game_ctl.sh <game> install` and
  provisioning STAGE 4d/4e both call the installer — **one install path**.
- **The failure that started this:** `Doom → player → install` on `main`
  fetched 79/83 packages at ~127 kB/s, then four from `esm.ubuntu.com`
  (Ubuntu Pro) timed out **over IPv6** (`Could not wait for server fd`).
  Commit 11277c6 "IPv6 fix" only added retries — and its run on 2026-10-15
  failed **identically** (same four packages, same IPv6 address, 12 min with
  5 retries): the failure is DETERMINISTIC, `main` cannot reach esm.ubuntu.com
  over IPv6. Now: `Acquire::ForceIPv4` (`SKYBOT_APT_IPV4=0` disables), short
  timeouts, 3 retries, one `apt-get update` + `--fix-missing` retry, then a
  last install **without the Ubuntu Pro (ESM) sources** — those four are ESM
  security rebuilds of archive packages, so the archive versions satisfy it
  (loud WARN; ESM builds return on the next `apt upgrade` that reaches ESM;
  `SKYBOT_APT_ESM_FALLBACK=0` disables). Already-installed packages are
  skipped before apt runs. Host-level fix worth doing once on `main`:
  `curl -6 -sI https://esm.ubuntu.com` vs `curl -4 ...` — if only -6 fails,
  `echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4`.
- **Why a stop is slow (~13 min measured on the old host; on `main`,
  tiny x 8 Doom, 196 s incl. an 11 s save of 80k transitions — 2026-10-18):** `touch runlogs/STOP` is read only BETWEEN segments
  (`segment_steps: 1024`), then `close()` writes the checkpoint and the replay
  buffer (`buffer_persist_max_gb: 8`). Never `kill` it — the supervisor
  relaunches a killed run.
- **HOST-PROVEN (2026-10-15):** that same failed run relaunched training
  afterwards (`run pid: 347953 (LIVE)`) — relaunch-after-a-failed-action works
  on `main`.
- **Deploys never re-download games.** `deploy_skybot.sh` rsyncs named
  folders without `--delete`; apt packages and `roms/` survive every deploy.
- **The game depot:** `<repo>/depot/` on the host (gitignored, never synced
  by deploy) = apt's download cache, pip wheels, hand-dropped `.deb`s, game
  files. The runner keeps the LAN copy at `/home/runner/_work/depot` (zero
  setup) and `scripts/depot_sync.sh` syncs it to the host before an install
  and back after, so a rebuilt host installs over the LAN. Neither sync can
  fail an install. Optional visible folder: `docs/operations/ops-005-operations.md` "Game depot".
- `fetch_roms.sh` takes the first source whose sha256 matches (depot →
  vizdoom wheel → apt `freedoom` → GitHub zip via Python `zipfile`). It used to
  need `unzip`, which nothing installs.

### Launchers: exec, Steam, browser

`game.launcher` (`launchers.py`; old `launch:` = exec, unchanged).
- **Steam client mode** (`steam -applaunch`): ONE per Linux user — a second
  launch is forwarded to the running client and opens on ITS screen. So
  streams must be 1 (compose refuses more), and a seat refuses — without
  crashing, re-checked every restart — while a Steam client it did not start
  runs. Needs a one-time human login on the host. **Direct mode** (`exe:`):
  any streams, DRM-free only.
- **Browser** (Chrome/Chromium/Firefox, kiosk app, private profile per seat):
  Ctrl/Alt chords and F-keys are browser shortcuts that leave the game — a
  one-way door (§4.5), refused unless in `launcher.allow_keys`.
- Every game process carries `SKYBOT_SEAT=<display>`; teardown kills by that
  marker because Steam and browsers re-parent children out of the seat's
  process group.
- `game.requires:` lists files the game cannot start without; the launch
  preflight refuses if one is missing (doom_player without its WAD otherwise
  trained on a black screen). The preflight now fails CLOSED.

### Rewards: no engine reads outside Minecraft (2026-10-17)

User rule for the universal brain: rewards only as a player could get them.
A live Doom run had `coverage` at **85 % of income** (ledger ALARM 52
segments, a farmed strafe/walk/run loop, still MAP01 / 0 exits) — computed
from the engine's player position; `gaze` from its pitch. **The RED sensor
guard only covers INPUTS; a reward computed in an env adapter never passes
through the sensor bus.** `compose_run_config.py` now forces
~~`curiosity.coverage_weight` and `gaze_weight` to 0 whenever a non-Minecraft
game is stream 0~~ — **replaced the same day** under the one-learning-process
rule (§1): coverage and gaze are OFF for the whole universal brain
(Minecraft included; the composer's per-game branch is DELETED), and
**episodic novelty** (`curiosity/episodic_novelty.py`, NGU-style kNN over the
agent's own view, memory per stream per life) pays EVERY stream of EVERY
game, from one helper in both stepping bodies. Chosen by the user from a
graded comparison on real Doom (`docs/research/res-004-rssm-size-experiment.md`, "Replacing
coverage"): 0.375x on the farm loop vs 0.96x for the old coverage-era view
term. `tests/_no_engine_reward_smoke.py` asserts every game composes the
IDENTICAL curiosity block. The legacy Minecraft-only brain is unchanged.
Before adding ANY reward term, ask: could a player see this, in every game?
- **Dead reckoning was integrating sideways motion BACKWARDS — fixed
  2026-10-17, in BOTH games.** Both adapters wrote `lat = dx*fz - dz*fx`
  (LEFT-positive) while DeadReckoner and the occupancy map
  (`spatial/memory.py`: "strafe right l -> dx = +l") read it RIGHT-positive.
  Measured in Doom: strafing right moved the agent -3.95 blocks, the
  estimate +3.94; graded, the dead-reckoned place was at chance (AUC 0.49),
  and the occupancy map smeared the same way. Now ONE projection,
  `sensors/heading.body_components`, feeds both adapters, and a round-trip
  unit case pins it (every older test moved straight ahead, lat = 0, which
  is how the sign survived). Real-engine contract: `_vizdoom_smoke` K, worst
  error 0.000 blocks over walk/turn/strafe both ways. The lateral SENSE
  changed sign: the legacy Minecraft world model sees one input flip.
- **Experiment scripts on `main` must NOT `sys.path.insert(0, ".")`** from
  `/workspace/devai`: that imports the DEPLOYED code ahead of the staged
  `~/size_exp/code`. It silently graded one comparison with an old encoder.
- **Off the LAN, reach `main` over Tailscale** (`ssh <host-login>@main` via
  MagicDNS; the login is in CLAUDE.local.md); the LAN address times out.

### World-model size experiment (2026-10-17)

`tools/rssm_size_experiment.py`, plan in `docs/research/res-004-rssm-size-experiment.md`
(pre-registered decision rules; stages A-D toward a growing RSSM). **Do not
trust Apple-GPU (MPS) runs of the world model:** tiny and small diverged to
NaN by step ~107-200 on MPS while the same batches stayed finite on CPU;
MPS also cannot run the flow head at all. The live world model is **129 M
parameters**. The real sweep needs `main` (`--device cuda`) with training
paused.

### THE WORLD MODEL'S LATENT WAS COLLAPSED (2026-10-17) — a bug class

Measured on `main`, same data: the posterior carried **0.00-0.06 nats per
frame** (z-info = unclamped KL(post||prior) summed over latents), KL sat at
its free-nats floor (`kl 0.110` in every log) and imagination barely beat
"the frame never changes" (0.94x). Every world-model size scored the same,
because none was using its latent. **Cause: reconstruction was a MEAN over
49,152 pixel values** — too weak a gradient to pay z for carrying anything.
Summing it (DreamerV3) gave 19.9 nats and 0.764x at step 3,000; inverse-head
off or DreamerV3-KL alone changed nothing. Also: the inverse-dynamics head
read `[h, z]` and `h_{t+1}` is built FROM `a_t`, so its 0.000 loss was the
answer read off its own input. Fixes are config keys set only in
`configs/universal.yaml` (`world_model.recon_reduction: sum`,
`inverse_input: embed`); the legacy Minecraft brain is unchanged.
`tests/_posterior_collapse_smoke.py`. **Lesson: `kl 0.110` is not a healthy
number — it is the floor. Watch z-info, not the clamped KL.** The autoscaler
logs z-info every segment and holds while it is below 1 nat.

- **Resume pre-flight now checks tensor SHAPES, not only keys.** A checkpoint
  with the same key names but a different width used to pass, then torch
  copied every matching tensor and raised at the end — the "FAILED mid-load
  ... partial hybrid" line in the deploy smoke log. Now refused before any
  copy; the module starts cleanly fresh (`_resume_smoke` 3c). The first
  universal launch after the fix therefore starts a FRESH world model.
- **The autoscaler** (`developmental_ai/infra/autoscaler.py`, universal brain
  only, `infra.autoscaler` in `configs/universal.yaml`): world-model size x
  stream count, **recommendations only** — an `infra/autoscale:` line per
  segment and `brain_universal/infra/autoscale.json`; nothing grows or shrinks
  the model yet (stage D). Signals: fresh-data loss on each stream's newest
  window (never sampled for training — `replay_buffer.newest_sequence`),
  training loss, z-info, replay ratio, seconds per decision, VRAM.
  `tests/_autoscaler_smoke.py`.
- **Overhead, measured:** world-model training is kernel-launch-bound at the
  live batch 8 (batch 16 = +57 % frames trained/s for live, 4.9 GB; batch 32
  OOM) — not changed, the live process was at 5.7 GB of 8. The acting loop's
  biggest cost is ICM training every step (27 %, deliberate — see
  `curiosity.train_every`). The viewer composed its panel every step (9 %); it
  now composes at <= 10 Hz while watched, every 2 s otherwise.
- **Size sweep, valid (with the fix):** tiny 0.758x / small 0.781x / medium
  0.773x / live 0.798x open-loop at 5,000 steps — **size does not matter at
  this stage** (pre-registered rule 1); tiny trains 1.38x faster. Start small,
  more streams; grow later (`docs/research/res-004-rssm-size-experiment.md`).
- **Stage D is built, actuation OFF** (`infra.autoscaler.actuate: false`):
  `world_model/growth.py` grows the RSSM function-preservingly (verified to
  ~3e-7), `tools/grow_brain.py` grows exactly or shrinks by relearning (only
  within tolerance), `brain_universal/scale.json` is what the composer builds
  at every launch, `runlogs/SCALE_RESTART` + supervisor `kind=scale` relaunch
  without counting a crash. Checkpoints now carry `world_model_arch.json` —
  a brain saved before this cannot be resized until it has run once.
  `tests/_growth_smoke.py`.
- **Growth bug found on CUDA:** growing a model built on the GPU mixed GPU
  and CPU tensors and raised; the tool and the Mac tests ran on CPU only.
  Fixed (both sides moved to CPU); verified on main's GPU (max diff 1.9e-7).
- **Fused Adam** (`world_model.fused_adam`, universal brain): +11 % frames
  trained/s, same memory. torch.compile / CUDA graphs were SLOWER (0.242 vs
  0.204 s/step). Batch 12 = +46 % but ~1 GB more — measure in the full loop.
- **PPO warm start, built (2026-10-17, universal brain):** resume component
  `policy_features` restores the policy's FEATURE layers (`actor.shared`,
  the knowledge conditioner, a conv encoder) and leaves the action head and
  critic FRESH, then boosts the entropy coefficient (x3, decaying linearly
  to x1 over 20 PPO updates; `loop.policy_features_boost`). The whole
  `policy` is still refused by name. GATED on world_model restoring (the
  features read its latent): the first launch after the inverse-head change
  starts a fresh world model, so it is correctly SKIPPED then and first
  applies the launch after. A reshaped stored policy (stage D growth, sensor
  change) is refused before any copy. Verified in the real Doom loop both
  ways. `tests/_policy_features_smoke.py`. Kickstarting (a KL pull toward the
  old policy) remains an untested alternative.
- **KICKSTARTING, built as the comparison arm (2026-10-17, OFF):** resume
  component `policy_kickstart` makes the stored actor a FROZEN TEACHER; the
  policy starts fresh and PPO adds w x KL(teacher || student), w decaying
  linearly to 0 over `loop.policy_kickstart.updates`, then the teacher is
  dropped. Gated on world_model like policy_features. Not in any resume list
  until the live comparison (fresh vs features vs kickstart) decides.
  Contracts in `_policy_features_smoke` F.
- **`GATE OVERDUE: 'learned_options'` on Doom — investigated (2026-10-17):
  Doom can NEVER mint a learned skill.** Goal discovery triggers only on
  spikes of the game's EXTRINSIC reward (`achievement_goals._is_spike`), and
  Doom pays 0 by design; and a slot without a Minecraft effect key
  (`break_<blk>`, from `mine_*` counters) is refused at mint time as a
  `discovered_N` fossil. So the whole skill/option system is Minecraft-only
  — a violation of the one-learning-process rule (§1). The alarm itself was
  false: it fired on an EMPTY bank, pointing at competence/probation gates.
  Fixed so far (telemetry only): an empty bank now reports a separate
  `skills_minted` gate, `learned_options` closes only when skills exist; and
  the learned-slot index is recomputed every decision (it was cached at the
  first act(), so skills bound later were never counted — also in Minecraft).
  `tests/_skills_gate_smoke.py`.
  **CHOSEN DESIGN (user, 2026-10-18), NOT built yet — full plan with
  pre-registered decision rules in `docs/research/res-005-skill-discovery.md`. Minecraft uses
  the SAME trigger and keys in the universal brain (owner's choice); its
  `mine_*` counters become the evaluation answer key. Step 0 (measure the
  trigger on recorded Doom play) runs with training PAUSED:**
  - **Trigger = surprise x agency**, identical in every game: a spike of
    world-model surprise (unclamped KL(posterior || prior), z-scored against
    the stream's running baseline) that the agent's OWN action caused (the
    action-conditioned prior differs from the action-averaged one). Life
    boundaries, respawns and rewinds excluded. Replaces the extrinsic-reward
    spike (`achievement_goals._is_spike`), which never fires in Doom.
  - **Acceptance = both:** (1) a game-agnostic LATENT EFFECT KEY — a
    cluster of the before->after latent change at the trigger, `effect_N`,
    replacing `break_<blk>`; the fossil guard keeps refusing `discovered_N`
    but admits `effect_*`; `required_effects` derived from the keys seen
    instead of hard-coded "break". (2) PROBATION for every learned skill:
    keep it only if invoking it reproduces its effect above chance within K
    attempts, else retire the slot (so junk cannot fossilise).
  - **Order:** measure first — grade the trigger on recorded Doom play against
    an engine answer key (kills, pickups, doors/switches, damage, exits;
    evaluation only) vs noise (flicker, monsters moving) — then build behind a
    config switch, both stepping bodies via one helper, contracts, and a live
    run on main in Doom AND Minecraft.

### Stage C, batch 12 and the warm-start comparison (main, 2026-10-18)

Full tables in `docs/research/res-004-rssm-size-experiment.md`. One seed each, real-time Doom.
- **Stage C (40 min each, fresh brains):** tiny x 8 collected **2.4x** the
  experience of live x 2 at 3.4 vs 5.6 GB, and its world model generalised
  (fresh/train 1.18) where live x 2 memorised (2.32); medium x 4 had the
  lowest fresh-data loss (0.0091). Behaviour did NOT separate: 0 exits
  anywhere, ~55 deaths per stream-hour in every arm. Cost of 8 streams:
  7 decisions/s per stream instead of 12.
- **Batch 12 at live x 2: NOT adopted.** +70 % frames trained per frame
  collected, but WORSE on fresh data (0.0178 vs 0.0141 over the same 9
  segments), more memorising, +24 % acting latency, +0.4 GB. With 2 streams
  the bottleneck is data, not training compute.
- **Warm start (30 min each from the same brain, reward changed at the
  resume):** policy_features died 2.5x less (52 vs 128 / 123 deaths/h) and
  farmed less (11 FARM? flags vs 54 / 20); its feature layers stayed near
  the source (0.375 vs ~1.41 for a fresh init) with a fresh head. Kickstart =
  fresh, because the 40-minute teacher was itself near uniform (KL 0.02
  nats) — it needs a long-trained teacher to be judged. Kept: policy_features
  ON, kickstart OFF. One seed.
- **A NEW universal brain starts at tiny x 8** (owner, 2026-10-18):
  `universal.initial_scale` in `configs/universal.yaml`, applied only at the
  launch (`compose_run_config.py --record-scale`) when there is no scale file
  and no saved world model, then recorded in `brain_universal/scale.json` —
  editing it never resizes an existing brain. The scale record now carries
  the ENCODER width too (without it "tiny" built live's 768-wide encoder, ~6x
  the measured tiny), and `grow_brain.py` keeps it. `main` already had a
  (live-size, pre-fix) brain, so its `scale.json` was written by hand at
  tiny x 8; the old world model is refused at resume and a fresh tiny one
  starts. **Deploy before launching**: the old composer ignores the encoder
  width. `_growth_smoke` I.
- **Three bugs this run found, all fixed:**
  1. **Episodic novelty paid EXACTLY 0 on the GPU** — `_view_key` called
     `np.asarray` on a CUDA tensor row inside a bare `except`, returning
     None. Every Stage C arm ran without it (imagination took 84-91 %). It had
     only ever run on CPU. Now takes tensors and warns once
     (`_no_engine_reward_smoke` G uses a tensor that refuses `np.asarray`).
  2. **The income lines/resets were Minecraft-gated** (§4.2 above).
  3. **The autoscaler advanced its own state in advice mode**, so a live x 2
     run logged "live x 3 -> live x 4". Only an actuated change moves it now
     (`_autoscaler_smoke` G2).
- **With episodic working, it is 66-96 % of Doom income** at
  `episodic_weight: 0.07` (Mac repro 94-96 % in the first segments; main
  W_fresh 66-74 %, measured before the ledger fix) and the farm detector
  flagged loops at +0.03/step. Dominant, not proven farmed — watch it on
  the first live run before trusting the weight.

### Mock deployment on `main` (2026-10-18)

The whole Actions path, run by hand against a SEPARATE root
(`~/mock_deploy/devai` on `main`, venv symlinked, a copy of the real brain),
so `/workspace/devai` is never touched: `ci` on Linux in a fresh python3.10
venv built exactly as `ci.yml` does, from a git commit of the files a push
would carry (`git ls-files -co --exclude-standard`); `deploy` =
`scripts/deploy_skybot.sh` with only the root path swapped, run from that
checkout; `host` launch / stop / status = the workflow step bodies, root
swapped (`SKYBOT_ROOT` for launcher + supervisor). The scripts live in the
session scratchpad, not the repo. Measured: CI tiers 452-465 s of the
1800 s step limit, core smokes ~60 s; deploy 19-21 s; launch to a live agent
26 s; stop 196 s; RSS 9.2-9.6 GB steady (peak 11.2 GB at a resume), 4.2 GB
minimum free; 5.4 steps/s per stream (43.7 transitions/s, vs 57.1 in the
Stage C harness).
Found and fixed:
- **Replay budget undercount** (§6): 5.6 GB held under "2.0 GB". The
  universal brain now sets `world_model.buffer_growth` to 5.5 GB / 0.35 so
  it keeps the replay every Stage C arm trained with, truthfully (verified
  live: reserved == held == 5.25 GB). **Enabling Minecraft in the universal
  brain adds two Java clients (~5 GB): lower it to 2 first.**
- **`launch_skybot.sh` restarted `ollama serve`** right after stopping it
  for a VLM-off config (0 procs before, 1 after); also printed a stray `0`
  (`pgrep -c ... || echo 0`).
- **`deploy_skybot.sh` printed `MAIN_USER`'s real value** (the host login)
  into the public Actions log; now prints the variable's name.
- **`host → status`** said `socat bridge DOWN` and `java clients: 0 (want
  2)` for a Doom-only brain, counted 2 "restarts" per launch, and printed
  nothing (not even its fallback) when crashes.log / dmesg were clean.
- **`_brain_mirror_smoke` 17 depended on the test machine's free disk**
  (`main`'s /tmp is a 7.5 GB tmpfs < MIN_FREE_GB 20); free space is now
  injected unless a contract asks for the real guard (G).
Seen, NOT changed: the ledger ALARMs `DOMINANT source=episodic` (80-89 %)
from segment 3 on — the episodic weight is the owner's call;
`_foundation_ab_verify_smoke` asserts runtime < 60 s (31.7 s on `main`,
58.7 s on an idle Mac, red when the Mac runs two suites at once); CI pins
torch 2.6.0 CPU while `venv_mc` runs 2.11.0+cu128 (the "mirrors the host"
comment in `ci.yml` is only true for numpy); the autoscaler's fresh-loss
window is not persisted, so every relaunch waits 8 segments (~27 min) for a
verdict; the progress block still prints Minecraft-only lines (`Chop
budget`, `Reach`, `Pitch`) in Doom.

### Stage D actuation, run for real (mock root on `main`, 2026-10-18)

Fresh brain (seeded tiny x 8 from `universal.initial_scale`, verified),
`infra.autoscaler.actuate: true` in the mock config only. Live it stays OFF.
- **The actuated path works end to end:** 40 min in, the loop wrote
  `runlogs/SCALE_RESTART` and stopped (replay 71k saved in 10 s), the
  supervisor logged `kind=scale` and relaunched 2 s later, `grow_brain
  --apply-pending` ran, `scale.json` and the composed config followed, and
  the brain resumed (world model, familiarity, replay, policy features).
- **A real tiny -> small grow** (request written by hand, same format,
  through the real launcher): verified function-preserving (max |diff|
  6.6e-7 recon, 2.9e-6 continue), backups kept, resumed at small x 7 with no
  loss spike (fresh 0.0073-0.011 vs ~0.015 before), z-info 26 -> 36 nats,
  same 5.4-5.6 decisions/s per stream, 3.4 GB VRAM, 9.2 GB RSS. The encoder
  stays 128 wide (growth resizes the RSSM only; the ladder's small is 256).
- **What a resize COSTS (by design, not fixed):** after a GROW the whole
  policy restarts fresh — `policy_features` is refused because the latent
  widened (512 -> 1088). After ANY relaunch the action head + critic are
  fresh. A stream cut drops the removed stream's saved replay (5k of 71k).
- **Fixed (autoscaler.py, `_autoscaler_smoke` E + G3, all three red on the
  old code):** (1) the cooldown did not survive the restart that applied the
  change — every launch reset it, so a verdict could follow after 8
  segments, not 16; `autoscale.json` state is now read back at launch.
  (2) "memorising at the max stream count" at the smallest model asked for
  FEWER streams (the VRAM/latency fallback) — less data, more memorising;
  live it turned tiny x 8 into tiny x 7. Now: shrink, or hold at the
  smallest. (3) the gap was ONE segment's (it swung 0.62 -> 1.72 between
  neighbours); now the window's average.
- **Not fixed:** `total_timesteps` restarts at 0 on every launch (the run
  after a 106k-step brain checkpointed "at 7168 timesteps");
  `progress_curiosity` works around it, but the spatial-memory validity
  `age = max(0, total_timesteps - rec["step"])` (loop ~11927) makes old
  records look new if they outlive the run.

### Why the policy was near random, and the fixes (2026-10-18/19)

Full diagnosis and the pre-registered live test: `docs/assurance/ass-003-policy-learning.md`
(ASS-003). Doom entropy of 2.7 "of 3.87" was **97-98 % of uniform** over the
16 real actions (the readout divided by ln 48). Causes, all fixed behind
switches (off = old behaviour; Minecraft and every synchronous game
byte-identical), all built and tested on the Mac, **none yet confirmed on a
live run**:
- **Real-time action lag (the main cause).** ViZDoom and Player return the
  frame the PREVIOUS action produced, so replay, the world model, inverse
  dynamics, curiosity's action gate (attribution 0.0076) and PPO all paired
  each interval with the wrong action. `realtime_credit.*` +
  `core/realtime_credit.py`: executed-action labels, the policy sees the
  state its action will start in, PPO is paid one interval later
  (`_realtime_credit_smoke`; `tools/replay_lag_probe.py` finds the shift in
  any saved replay).
- **PPO:** the KL stop is judged on the whole rollout (it fired on the first
  noisy 32 rows); masked logits no longer count toward the logit cap; the
  bootstrap value gets proprio; distillation is skipped when its weight is
  negligible (Adam cancels the scale) and masked to executable actions
  (`_ppo_fixes_smoke`). New telemetry: I(S;A) on a fixed probe, KL from the
  first policy, actor grad norm over applied updates, freeze trip point.
- **Reward:** the intrinsic weight no longer anneals toward an extrinsic
  reward the game never pays; the 7 scout streams use stream 0's reward
  function; the farm damp keys on the view, not the engine position.
- **Persistence:** `policy_full` restores the whole policy when the reward
  fingerprint is unchanged; a grow carries the policy; `lifetime_timesteps`
  survives relaunches; a stream cut keeps its replay; one replay write per
  stop; the entropy readout uses the valid-action maximum.
**Live result (2026-10-20, `main`, 5 runs of 120 min; full table in ASS-003):**
the lag probe finds -1 on 8/8 streams with the fixes off and 0 on 8/8 with
them on, in two seeds; action attribution 0.15-0.20 vs 0.02; state
dependence I(S;A) 0.23-0.32 vs 0.14-0.19; entropy over valid actions 0.87
vs 0.92-0.94; the intrinsic weight stays 0.7 (OFF fell to 0.53); the
policy survives a relaunch. **Behaviour did NOT improve consistently** (ON
better in seed 1, not in seed 0; no exits, still MAP01), so do not claim
better play. The first ON run also exposed a scout-imagination crash
(9,515 tracebacks, bonus silently 0), fixed and tested. Next question:
whether the objective rewards anything action-dependent enough (episodic
novelty is still 80-90 % of income).

### Still unproven on `main` (do these before trusting the player path)

~~`tools/player_spike.py`~~ **PASSED on main 2026-10-17** (grab 10,212/s,
key latency 27 ms, mouse 8.9 px, audio non-silent, 26 % CPU / 172 MB per
game; run from a login shell, so audio under the supervisor is still
unverified); openbox decorations cropping the 640×480 grab; PulseAudio for
a non-root `skybot` after ssh exits (`loginctl show-user skybot -p Linger`);
`/usr/games` on the supervisor's PATH; RAM/VRAM for 2 MC + 2 Doom streams on
16 GB / 8 GB; the dead-stream heal against a real engine death; Steam and
browser launchers have only been tested against fakes.

### Other operational facts from this work

- **Actions logs are public and `.env` values are NOT masked** (only GitHub
  secrets are). Two leaks fixed: (1) `host_exec.sh`, `deploy_skybot.sh`,
  `deploy.yml`, `host.yml` pass `-o LogLevel=ERROR`, because with
  `UserKnownHostsFile=/dev/null` every call printed `Permanently added
  '<address>'`; (2) `deploy_skybot.sh` / `deploy.yml` echoed `$HOST` /
  `$MAIN_HOST` in five messages (`launch with: ... <address> ...`) — they now
  print the variable's NAME. Never `echo` a host value in anything a workflow
  runs. Old run logs that show it can be deleted from the Actions tab.
- A failed Actions run cannot be read from the Mac (no `gh` CLI): ask for
  the step log. **`main` is reachable from the Mac as `<host-login>@<MAIN_HOST>` (login in CLAUDE.local.md)
  only when the user has added the Mac's key** (temporary, 2026-10-17);
  the old `skybot@` login is refused. Deploy does NOT manage `authorized_keys` — a removed
  node1 key must be re-added by hand, or deploy fails at "host reachable".
- **Experiments on `main` run from an isolated copy** (`~/size_exp/code`,
  rsynced from the Mac), never by editing `/workspace/devai` outside deploy.
- `_incident_bundle_smoke` **passes on `main`** (Linux `flock`), confirming
  the Mac failure is the missing `flock` only.
