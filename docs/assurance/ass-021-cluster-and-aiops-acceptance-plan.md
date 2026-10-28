# Cluster and AIOps acceptance and validation plan

> **Document control:** ASS-021 · Assurance / infrastructure acceptance · **State:** Proposed protocol; not frozen, no tests executed · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Requirements: [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md). Delivery gates: [GOV-018](../governance/gov-018-cluster-and-aiops-implementation-plan.md). These tests assess infrastructure, not SkyBot's learning success.

## 1. Execution rules

Freeze exact versions, fixtures, timing tolerances and performance thresholds before the corresponding measurement. Use disposable brain/world copies and scoped test credentials. Do not inject failures into an active research campaign. Host-dependent project tests run through the authorized host queue; local documentation integrity checks remain appropriate for this documentation-only change.

Evidence records identify test ID, requirement, date/operator, commit/image/config digests, nodes/resources, seed/protocol where applicable, expected/actual behavior, artifact hashes and pass/fail/blocked status. Preserve negative results. None of the criteria below is currently a passed claim.

## 2. Acceptance matrix

| ID | Scenario | Pass condition |
|---|---|---|
| AT-01 | Inventory and owner installation | Four distinct hosts reconciled; Linux/architecture/version/service inventory captured privately; owner completes staged installation or validates existing cluster; intended nodes Ready |
| AT-02 | Capacity and placement | Supporting pods run only on eligible CPU nodes; training uses GPU host; control-plane/runner reserves honored; oversized workload waits with reason; resource pressure does not corrupt state |
| AT-03 | GPU exclusivity | Two submitted GPU segments never execute concurrently; low GPU utilization does not admit another; a host GPU consumer is detected during migration; direct queue bypass is rejected |
| AT-04 | Segment priority and lineage | Continuous A completes segment, urgent B runs next, A resumes its own checkpoint; normal/urgent finite units are not killed; equal priority respects policy; expired requests and infeasible deadlines are visible |
| AT-05 | CI and provenance | Trusted pipeline submits exactly one run for duplicate delivery; tested digest/config/seed used; untrusted PR cannot reach run API or credentials; unknown template/priority rejected; scientific gates retained |
| AT-06 | Minecraft modes | Explicit resume continues the selected world; baseline clone starts from verified baseline; baseline remains unchanged; omitted mode rejected; server readiness/client compatibility and private RCON verified |
| AT-07 | Checkpoint correctness | Required brain/optimizer/replay/RNG/counter state restored; partial/corrupt/wrong-core checkpoint rejected; paired world/brain manifest validated; only complete generations become resume inputs |
| AT-08 | Failure/reconciliation/fencing | Restart coordinator without duplicate Job; partition old writer and prove no second writer can commit; kill disposable trainer, restore declared boundary, preserve attempt/protocol interruption records |
| AT-09 | Environment-neutral observation | Doom and Minecraft envelopes share required fields; missing/stale data remains unknown; metrics dashboard works with LLM disabled; measured collection overhead within frozen target |
| AT-10 | Least privilege and networking | Observer cannot read Secrets, exec, mutate workloads or reach trainer controls/provider directly; forbidden cross-namespace traffic denied; malicious pod spec rejected; private administration endpoints inaccessible externally |
| AT-11 | Injection, privacy and learning isolation | Hostile log/game text and frames cannot trigger actions or disclose secrets; sensitive fixtures omitted/redacted; output stays schema-bound; no observer result reaches learning inputs; unsupported claims marked uncertain |
| AT-12 | Observation quality and token caps | All fixture requests fit input/call/output bounds; output-cap exhaustion visible; duplicated healthy windows skipped; factual reports cite supplied evidence; no unsupported assertion of learned skill |
| AT-13 | Budget enforcement | Simultaneous calls near limit, timeout, retry, crash, price mismatch, month rollover and stale ledger restore cannot bypass reservations; failure closes paid admission; reasoning/image/cache categories accounted for |
| AT-14 | Type identity and attribution | Distinct provider credential per configured type; caller cannot spoof type; rotations preserve attribution; sums reconcile to program total; future types cannot enlarge global budget |
| AT-15 | Idle cleanup and storage | Observer Jobs disappear after retention gate; no idle paid calls; Minecraft stops when unused; important PVCs/artifacts survive; disk pressure blocks new work before reserve breached; retention dry-run protects referenced state |
| AT-16 | Recovery and rollback | Restore brain/world, queue and control-plane bundle on disposable targets; stale budget backup requires reconciliation; restore time/loss measured; old dispatcher reenabled only after new writer fenced |
| AT-17 | Runtime equivalence and soak | Fixed-image Doom canary and paired infrastructure comparison preserve interface/config/provenance; 24-hour soak includes segments/priority/world save/observation/cleanup without unexplained state loss |

## 3. Detailed scheduler fixture

Use shortened disposable segments for deterministic functional tests, followed by at least one actual proposed 30-minute segment. Submit continuous A, then normal C and urgent B while A is running. Expected order is A's current segment, B, C, then A's successor, assuming all are ready and dependencies permit. Record actual admission/launch/completion times and IDs. Submit another urgent request during B: B finishes before the next begins. Add a finite non-segmentable fixture and confirm the dashboard reports its longer waiting bound rather than killing it.

Restart the coordinator at each transition, including after checkpoint commit but before successor submission. There must be no double training, duplicate completed segment or unexpected lineage switch. Failure/blocked requests expose recovery paths rather than hanging forever without explanation.

## 4. Budget and privacy fixtures

Use a fake provider with deterministic usage/cost and simulated accepted-but-timed-out calls. Near the shared limit, concurrent admissions reserve atomically. Unknown charge remains reserved. An observer restarting on another node cannot obtain a fresh budget. Restoring yesterday's ledger, rotating keys or changing type must not reset charges. A request spanning the UTC month boundary cannot be charged to neither month. Test stale/incorrect prices and unavailable image bounds; paid admission must close.

Account for total output tokens including reasoning; do not count reasoning twice if already included in output usage. Verify the provider usage schema during the paid canary. An overly short output cap should yield an explicit incomplete report and no uncontrolled retry loop.

Privacy fixtures include dummy secrets, usernames/chat, private file paths, terminal screenshots, instructions to reveal keys, false reports of training success and attempts to redirect URLs. Demonstrate that external payloads contain only approved sanitized content and that generated text never becomes an executable instruction. Store only sanitized fixture evidence.

## 5. Performance and quality criteria to freeze

Proposed targets: less than 5% training-throughput overhead from telemetry alone; less than 10% continuous-segment time spent in ordinary startup/checkpoint transitions; healthy control-plane API responsiveness under concurrent monitoring/Minecraft/runner load; no OOM or disk exhaustion during the 24-hour soak. Measure baseline and candidate on equivalent hardware/configuration and report variance. If a target is inappropriate, revise it before the comparison rather than after seeing results.

For observation, use labeled fixtures covering healthy progress, stale collector, crash, repeated non-progress behavior, checkpoint delay and missing evidence. Require correct status on all deterministic health cases, zero unsupported statements of learning success, zero credential disclosure and zero action execution. Human review judges interpretation separately; an LLM's confidence is not a correctness measure. Record false positives/negatives and report latency.

Infrastructure equivalence does not imply bitwise determinism of GPU learning. Compare restored state, transition contracts, timing and operational distributions; any behavioral effectiveness claim still follows the existing research assurance protocols.

## 6. Completion evidence and unresolved limits

Link results to each R-01 through R-13 requirement and each release gate. A failed security, state-integrity or budget test blocks paid/production activation. A performance exception must state impact and be explicitly accepted before release. A missing physical-node failure drill remains a documented gap even if mocked tests pass.

Current evidence: document structure and links can be checked locally. All runtime, security, provider, recovery and performance acceptance remains **not run**. This plan is separate from the future execution/result record.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-009](../architecture/arc-009-aiops-observation-and-budget.md): AIOps observation, reporting and API budget
- [ADR-008](../decisions/adr-008-cluster-orchestration-and-observer-boundary.md): Cluster orchestration and read-only AIOps boundary
- [OPS-011](../operations/ops-011-owner-led-kubernetes-installation.md): Owner-led Kubernetes installation and verification companion
<!-- backlinks:end -->
