# Relation registry

> **Document control:** [ARC-006](../README.md#architecture) · Architecture / design record · **State:** In progress (built, untested) · **Custodian:** Repository maintainer · Revision history: Git.

**Date opened:** 2026-10-21. **Implements:** [GOV-010](../governance/gov-010-relational-implementation-plan.md) section 3 and phases R1 and RW-15.
**Status:** built and **untested** ([GOV-012](../governance/gov-012-execution-record-gov-010.md), entries R1 and RW-15). Nothing here has run in the loop.
Bound by [ADR-001](../decisions/adr-001-shared-universal-brain.md) and [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md).

## 1. What it is, in one sentence

Every line of the owner's map (`assets/ai_learning.mmd`) is an auditable object, a named **channel** `rel.<FROM>.<TO>` (the map's node ids), so the code can be compared with the map line by line, any line can be cut for an A/B arm, and a line that has gone silent is reported.

## 2. What it measures, and what it does not

It measures **implementation coverage only**: whether a line has a software channel, whether the channel carries a signal, whether it is alive. A line being present is not evidence that its function equals the human one; benefit is decided only by the T1 and T2 tests (GOV-010 4A). The conformance report is labelled accordingly. RES-008 statuses derive from block status and do not validate causal direction or completeness of the map.

## 3. Parts

| Part | File | Role |
|---|---|---|
| Registry, channel, statistics, dead-wire, leakage guard | `developmental_ai/core/relations.py` | The only logic. Imported lazily; never imported while `relations.enabled` is false. |
| Connected-core declarations (RW-15) | `developmental_ai/core/relations_connected.py` | 36 channels with read-only probes over state the loop already holds. Later clusters add a module and list it in `_BUILDERS`. |
| Cluster modules (RW-1 to RW-9, RW-11 and RW-14 Stage A) | `relations_modulation.py`, `relations_consolidation.py`, `relations_semantic.py`, `relations_attention.py`, `relations_executive.py`, `relations_strategy.py`, `relations_motivation.py`, `relations_transfer.py`, `relations_skills.py`, `relations_peers.py`, `relations_resources.py` (+ `semantic_store.py`) | Each registers its channels (`register`) and, for WIRED lines, exposes `apply` / `finalize` / `release`, called once per segment by `RelationRegistry.wire_all` from the existing `_relations_phase` (no new stepping-body hook). Wired channels use `send()` (off = neutral constant) and one-owner coefficients (`core/coefficients.py`). RW-8 registers three representation metrics; RW-9 registers three peer diagnostics; RW-11 Stage A registers two skill diagnostics; RW-14 Stage A registers active context width. These are read-only and do not establish the hypothesised benefit. Entries: GOV-012 RW-1 to RW-14. |
| Map copy (generated) | `assets/ai_learning_map.json`, from `tools/map_to_registry.py` | Nodes and lines with their RES-008 ids and the sha256 of the map. **Never hand-edited**; regenerate on demand; `--check` detects staleness. |
| Conformance report | `tools/relation_conformance.py` | Line by line: channel, descriptive or none, next to the RES-008 status; checks the generated ids against the RES-008 tables, the channels against the map, and that every Connected line has a channel. |
| Loop helper | `_relations_phase` in `developmental_loop.py` | ONE helper, called once from each stepping body (`_collect_segment`, `_run_episode_parallel`; the grep count of `self._relations_phase()` is 2). Prints `infra/relations:` and stashes the segment summary into the metrics record (`relations`). |

## 4. Channel fields

| Field | Meaning |
|---|---|
| name, line id | `rel.<FROM>.<TO>`; the RES-008 id (for example `NM-01`) is the **status** |
| producer, consumer | what sends and what uses (learner-side) |
| signal, units, time scale | what flows |
| type | `data` (a value flows), `control` (a signal gates or scales), `descriptive` (a hypothesis with no literal channel: recorded, never wired, no switch, no statistics) |
| switch | `relations.<FROM>.<TO>.enabled`, default off; `relations.connected_core.enabled` is the default for the RW-15 group; `relations.enabled` is the master |
| neutral | what `send()` returns when the switch is off (the previous behaviour) |
| statistics | per **segment**, not per step: mean, spread (within-segment standard deviation), last, n; pair channels add a rolling correlation over segments |
| source tag | `learner`; registering `evaluator` raises `EvaluatorLeakError` |

Descriptive rule (kept in `is_descriptive_line`): every line to or from the central node and every dotted line (12 lines now).

## 5. Rules

1. **Off reproduces the earlier behaviour.** Observation channels read state and write nothing back. A wired channel (a later cluster) calls `send()`, which returns the neutral constant when off.
2. **No evaluator or engine value (ADR-002).** An evaluator-tagged source raises; an unknown tag raises; a probe defined under `developmental_ai.evaluator` or the game-template detector package raises. The runtime leakage test is therefore a registration test.
3. **Dead-wire alarm (guard-becomes-latch review, CLAUDE.md 4.1).** The consumer-side effect |v| averaged over the last `window` segments at or below `eps` times the channel's decayed peak raises a flag; a channel never nonzero over a full window is flagged. **The alarm gates nothing** and is a pure function of the window, so it **releases by itself** when the flow returns; the peak decays (`peak_decay` per segment) so a transient spike cannot make normal flow look dead for ever. Segments with no sample (probe unavailable) do not count toward the window. Nothing is persisted; a relaunch starts with an empty window.
4. **Cost.** One probe call per enabled channel per segment, each a dictionary read; no per-step work. Overhead to be measured on the host against ADR-005 before use in a long run.
5. **No second source of truth.** The map JSON is generated from the owner's file; the channels are declared in code; the report checks one against the other.

## 6. Known limits

- Several RW-15 lines share one underlying quantity; "alive and nonzero" is the claim, not the line's causal role.
- Two ledger-share probes (MO-05, MO-31) match source names by substring and are unverified.
- `_last_ppo` is set by `_collect_segment` only, so PPO-based probes give no sample from the other body (the existing duplicated-body gap, CLAUDE.md 4.2).
- A channel for an extrinsic reward reads zero in a game that pays none; the dead-wire flag there is true, not a fault.

## 7. Related records

[GOV-010](../governance/gov-010-relational-implementation-plan.md), [GOV-012](../governance/gov-012-execution-record-gov-010.md), [RES-008](../research/res-008-relationship-gap-analysis.md), [RES-009](../research/res-009-hard-and-deferred-relationships.md), [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ASS-012](../assurance/ass-012-categorical-system-testing-protocol.md): Categorical system testing protocol
- [GOV-013](../governance/gov-013-categorical-system-testing-plan.md): Categorical system testing: coherence within and between learning groups
<!-- backlinks:end -->
