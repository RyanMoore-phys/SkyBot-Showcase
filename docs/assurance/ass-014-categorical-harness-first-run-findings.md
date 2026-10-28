# Categorical harness: first-run findings

> **Document control:** [ASS-014](../README.md#assurance) · Assurance / result record · **State:** Record; no finding fixed · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-22. **Protocol:** [ASS-012](ass-012-categorical-system-testing-protocol.md) section 9.
**Plan:** [GOV-013](../governance/gov-013-categorical-system-testing-plan.md). **Evidence class:** fixture
(real module objects, injected numbers and faults, CPU, this Mac). Nothing was run on `main`, no game was
started, and **no production code was changed**.

## 1. What was run

`PYTHONPATH=. ./venv/bin/python tests/run_all.py categorical`: the CS-0 inventory, seven within-group files
(CAT-01 to CAT-07) and the boundary file (BND-01 to BND-07), about 66 s in total. Counts at the time of
writing (passed / blocked / failed scenarios, where one contract id can carry several scenarios):

| File | Passed | Blocked | Failed |
|---|---|---|---|
| inventory (CS-0) | 3 | 0 | 0 |
| cat01 perception and prediction | 6 | 0 | 2 |
| cat02 memory, attention, consolidation | 14 | 0 | 6 |
| cat03 planning, uncertainty, risk | 6 | 0 | 2 |
| cat04 motivation, exploration, skills | 23 | 0 | 5 |
| cat05 curriculum and adaptation | 22 | 0 | 7 |
| cat06 diversity, peers, transfer | 24 | 0 | 3 |
| cat07 runtime and isolation | 24 | 0 | 1 |
| boundaries (BND-01..07 + two combinations) | 50 | 6 | 15 |

Every passing scenario ran twice: as built, and with a named deliberate break that the harness confirmed was
detected. A failing scenario fails in its normal run, so its break is not exercised; the oracle was checked
against a monkeypatched fix in the cases noted by the authors (F-13, F-01 to F-03). **All failures were
triaged as real defects or design gaps**; fixture errors found on the way were corrected in the tests.

## 2. Findings (28, deduplicated)

Classes: **G** missing finiteness guard (a NaN or infinity enters learner state); **L** latch or missing
release (CLAUDE.md 4.1); **P** persistence; **W** wrong behaviour; **D** design gap needing the owner's
decision. "Where" is the production site. None is fixed.

| # | Class | Where | What the harness observed | Contracts |
|---|---|---|---|---|
| F-01 | G | `core/uncertainty.py:63` `EnsembleHeads.train_step` | one NaN transition batch leaves the ensemble and its scale NaN for good; planning arbitration stays invalid | cat01 CT-03, cat03 CT-06, BND-01 |
| F-02 | G | `world_model/rssm.py:2554` `WorldModel.train_step` | one non-finite observation makes every parameter NaN; two valid batches do not recover it | cat01 CT-06 |
| F-03 | P | `core/planning.py:452` `PlanningPhase.load_state_dict` | a saved ensemble of another width is partly copied (9 tensors) before the error; "starts fresh" is false (reachable on a stage-D grow) | cat03 CT-07, BND-01, BND-07 |
| F-04 | W | `policy/consolidation.py:400,413` | `Consolidator.strength` scales a loss that Adam then normalises: update norm 0.00198 at strength 0.25, 1 and 4. The RW-1 to consolidation chain cannot change the update | cat02 CT-01 · **fixed 2026-10-22, GOV-014 B2** |
| F-05 | G | `core/outcome_memory.py:72,113,150` | one infinite surprise freezes the outcome scale at inf; a NaN view key poisons every retrieval vector | cat02 CT-03, BND-03 |
| F-06 | G | `core/attention_memory.py:275-316` | one NaN latent makes the summary and the layer weights NaN for good | cat02 CT-03 |
| F-07 | G | `relations_consolidation.py:170`, `relations_modulation.py:71`, `relations_attention.py:121` | a NaN input pays the maximum factor once, then the z-score tracker stays NaN and the channel is dead | cat02 CT-03 |
| F-08 | W | `core/attention_memory.py:292` | the window is cleared at an episode end and then the old episode's last latent is pushed: 7 of 11 stored pairs span two episodes | cat02 CT-05 |
| F-09 | L, D | `core/relations.py:358`, `relations_attention.py:163`, `developmental_loop.py:11913,11943,11977` | one exception idles a wiring module or controller (WP-8, WP-9, WP-13, RW-1, RW-2) for the rest of the run; no re-arm. The code says "idled for the run" on purpose; ASS-012 CT-06 requires recovery | cat02/04/05 CT-06 |
| F-10 | G | `curiosity/intrinsic_balance.py:48` `ShareCap.limit` | one NaN income gives NaN payments for ever | cat04 CT-03 |
| F-11 | G | `core/competence.py` `update` | a NaN or infinite fresh loss makes `loss_ema` NaN permanently; intrinsic gain drifts (0.776 to 0.711) and the replay recipe consumed by consolidation changes | cat04 CT-03, BND-04, BND-06 |
| F-12 | G | `skill_bank/effect_discovery.py:184` | `mag < min_change` is False for NaN: a NaN embedding mints `effect_0` with a NaN centroid | cat04 CT-03 |
| F-13 | **W** | `curiosity/action_surprise.py:146` with `intrinsic_balance.py:73`, `effect_discovery.py:98` | **`GuardClock` returns True for rows within `k_guard` steps of a life boundary, its docstring says those are ignored, and both consumers read True as eligible.** Information gain is paid only just after a boundary; the WP-1 trigger fires on no mid-life spike. Inverting the polarity made both oracles pass | cat04 CT-05 · **fixed 2026-10-22, GOV-014 B1** |
| F-14 | G | `policy/adaptive_optim.py:95` | only NaN is filtered; `+inf` and `-inf` move the controller | cat05 CT-03 |
| F-15 | G | `core/meta_adapt.py:156` | one NaN loss leaves `noise = nan` and disables acceptance for good (0 accepts against 8) | cat05 CT-03 |
| F-16 | G | `core/relations_strategy.py:208` | `min(1.0, nan)` is 1.0: a NaN fresh loss is booked as the maximal reward | cat05 CT-03 |
| F-17 | L | `policy/adaptive_optim.py:345` `release()` | restores lr and the return normaliser but not `policy.entropy_coef` (base 0.01, after release 0.0222) | cat05 CT-06 |
| F-18 | L, D | `core/meta_adapt.py:119`, `relations_strategy.py` `apply` | losing the measurement source does not release a trial value or recipe (held 4+ segments); the code releases on failure, not on source loss | cat05 CT-06 |
| F-19 | P | `policy/adaptive_optim.py:336` | the base lr is captured lazily from the live param group; after a restore the scaled lr becomes the base, so a schedule compounds on every relaunch (step 800: 8.4e-05 uninterrupted, 4.62e-05 restored; config 3e-4) | cat05 CT-07 |
| F-20 | G | `core/peer_observation.py:109` `PeerBoard.publish` | a NaN or infinite view is stored; the peer context and the predictor weights go NaN | cat06 CT-03, BND-05 |
| F-21 | W | `core/peer_observation.py:125` | the width check runs only with a non-empty ring and no break; after an absent view or a break, a view of another width enters the ring and later breaks `torch.stack` | cat06 CT-05 |
| F-22 | W | `core/peer_observation.py:144` | the first context read after the peer set changes reshuffles with the board's own RNG: the RNG state depends on when it is read | cat06 CT-08 |
| F-23 | P | `infra/autoscaler.py:174` `restore` | the ADR-005 stepping scale is saved but not restored (saved 2.0, rebuilt 1.0); bites only once a stepping actuator exists | cat07 CT-07, BND-07 |
| F-24 | **W** | `policy/consolidation.py:155` `Consolidator._sample` | with `spaced_fraction > 0` the `index_select` test compares against `observations`' already-shrunk row count, so the other tensors keep 11 rows against 4: `Consolidator.run` raises "Sizes of tensors must match" and the phase fails every later segment. Hit by RW-6 arms 1 and 3 and by `replay.spaced_fraction` | boundaries BND-06, COMBO-B |
| F-25 | L | `core/competence.py:129` | when `measure_streams` returns nothing, the stale smoothed state stays active: after 8 segments (bound 6) the gain is still 0.70 and RW-7 `frontier_gain` 1.092 | BND-04 |
| F-26 | D | `competence.py:137` vs `curriculum.py:248` | WP-3 resets its baseline on a context switch; WP-8 scores the post-switch loss jump as lost progress (lp -0.213 against +0.029). Needs the owner to confirm intent | BND-04 |
| F-27 | W | `core/peer_observation.py:115` | `publish` counts a stamp before validating, so a refused wrong-width view burns a stamp and the next pair is discarded | BND-05 |
| F-28 | G | `infra/autoscaler.py:240` | `act_s = +inf` or `vram_frac = +inf` gives SHRINK, an actuated request; infinity is invalid evidence (ASS-012 CT-03) | BND-07 |

**Blocked, not failed (6):** acting-time planning is not wired (GOV-011 D-30); the RW-4 semantic store has no
policy consumer; RW-12 and RW-13 have no module; peer auxiliary training does not train the shared encoder;
RW-8 is measurements only. These are the GOV-013 section 3 constraints, confirmed in code.

## 3. What passed (summary)

Cut/neutral parity, single ownership of the shared coefficients (five writers, 60 segments, bounded and
one write each), evaluator isolation at the real entry points (registry, peer channel, held-out maps, import
scan of every learner module), stream and episode separation in replay and the peer board, release bounds for
RW-1, and the modulation to consolidation to attention to entropy chain (BND-02, all seven contracts).

## 4. Consequences for the queued runs (ASS-013)

- **Two findings make queued A/Bs meaningless if left as they are.** F-13: `t1_wp5_info_gain` (and WP-1 and
  RW-7, which read the same clock) would test an information-gain term that pays only right after a life
  boundary. F-24: `t2_rw6_strategy` arms with spaced replay would run with consolidation raising every
  segment, silently. Both would burn 5 to 10 host-hours each and report an empty effect as a null result.
- **F-01 to F-03 and the guards (G class)** are rare-event defects; they matter for the long runs and for any
  arm that produces a NaN, and they are the defects ASS-012 5.2 says block the categorical pilots.
- **A fix changes the code snapshot** (ASS-012 6): the campaign's frozen copy must then be re-staged and the
  affected tests re-run. A fix that changes the baseline's behaviour only in a NaN case does not change the
  baseline's expected results but does change its file hash, so it should land before `ab_baseline`.
- **Decision needed from the owner:** which findings to fix, in what order, and the three design calls (F-09,
  F-18, F-26). Until then the categorical pilots stay blocked (`cat_harness` fails), which is the intended
  effect of the gate, and the T1/T2 queue is unaffected.

## 4A. Resolution status (updated 2026-10-22)

Following [ASS-015](ass-015-ab00-interim-analysis.md) the owner chose to repair two findings first
([GOV-014](../governance/gov-014-corrective-plan-after-ab00.md)): **F-13** (item B1) and **F-04** (item B2).
**Both are fixed (2026-10-22).** The pre-registered prediction held: the cat04 `D-GUARD` and cat02 `strength` scenarios
flipped to pass and no other verdict changed (failing scenarios 41 to 39). cat04's fixtures had also encoded the old
polarity and were corrected (GOV-014 section 4). Every other finding
stays open, and F-24 additionally blocks `t2_rw6_strategy` in the queue.

## 4B. Resolution (2026-10-23)

On the owner's instruction of 2026-10-23, 24 of the 26 findings still open were corrected
([GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md), where section 9.1 lists the change and the
issue it solves for every finding): F-01 to F-03, F-05 to F-12, F-14 to F-25, F-27 and F-28 (F-04 and F-13 were fixed
earlier). **F-26 is not changed**: it is a design question for the owner. F-09 and F-18 were flagged as design gaps; they
were corrected with the bounded recovery that ASS-012 CT-06 requires, and the owner may still prefer another design.
**Status: fixed in the code, not yet verified.** Verification is queued on `main` (stages `gov015_tests` and
`gov015_tests_before`, GOV-015 section 9.4); until they finish this record claims no scenario flips. Several harness
scenarios had to be changed with the fixes (their deliberate breaks assumed the defect was present; one asserted the
defective window behaviour); each change is listed in GOV-015 section 9.2.

## 5. Limits of this record

Fixture evidence only; the real loop composes these modules with more state than a namespace carries. The
authors confirmed a fix by monkeypatch for F-01 to F-03, F-10 and F-13 only. Counts are scenario counts, not
independent replications, and overlap between files is not replication (ASS-012 7). Nothing here is a
behavioural claim or a promotion decision.

## 6. Related records

[ASS-012](ass-012-categorical-system-testing-protocol.md),
[ASS-013](ass-013-unattended-test-campaign.md),
[GOV-013](../governance/gov-013-categorical-system-testing-plan.md),
[GOV-011](../governance/gov-011-execution-record-gov-009.md),
[GOV-012](../governance/gov-012-execution-record-gov-010.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-017](ass-017-host-test-stages-and-queue-incidents.md): Host test stages and two queue incidents (2026-10-23)
- [ASS-018](ass-018-campaign-progress-2026-10-23.md): Campaign progress check, 23 October 2026 (evening)
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [RES-010](../research/res-010-predictive-coding-core.md): Predictive coding as the world model's latent core
<!-- backlinks:end -->
