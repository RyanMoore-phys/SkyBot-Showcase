# Implementation plan: containerization, Kubernetes and AIOps

> **Document control:** GOV-018 · Governance / phased implementation plan · **State:** Proposed; documentation only, no implementation started · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Requirements and package entry point: [GOV-017](gov-017-cluster-and-aiops-proposal.md). Architecture choice: [ADR-008](../decisions/adr-008-cluster-orchestration-and-observer-boundary.md). This plan describes future changes; it does not authorize executing them in the documentation task.

## 1. Delivery method and responsibilities

The owner executes Kubernetes installation and administration for practice. Implementation assistance prepares reviewable application/configuration changes when later requested, explains each installation step and interprets sanitized results. CI automates trusted build and run submission after activation. The maintainer owns credentials, private inventory, retention decisions and release acceptance. AIOps has no approval or execution authority.

Proceed by evidence gates, not calendar promises. Each work package records commit/image identity, changes, tests, limitations, rollback and the owner's installation actions. Host-dependent testing follows the existing queued-host rule. Do not interrupt current research campaigns to complete an infrastructure milestone.

## 2. Work packages and exit gates

| WP | Work and future deliverables | Dependencies | Exit evidence |
|---|---|---|---|
| CP-0 | Inventory four nodes, existing services/cluster, network/storage/GPU, backups and versions; reconcile runner location | None | G0: capacity and private inventory complete; disposable test state available |
| CP-1 | Owner installs or validates control plane, workers, runtime, CNI and security baseline using OPS-011 | CP-0 | G1: AT-01, AT-02, initial AT-10; all intended nodes Ready, policy enforcement proven |
| CP-2 | Reproducible Doom trainer image around `scripts/run.py`, non-root runtime, asset handling, signal/checkpoint audit | CP-0; cluster canary after CP-1 | G2: AT-07 and AT-17 on disposable state; image boots and restores |
| CP-3 | NVIDIA integration, exclusive GPU test, Kueue, run schema/coordinator, locks and segment scheduling | CP-1, CP-2 | G3: AT-03, AT-04, AT-08; one writer and correct priority order |
| CP-4 | Trusted image publication and idempotent run API submission; campaign stage mapping | CP-3 | G4: AT-05; automatic request creates exactly one correct run |
| CP-5 | Minecraft server/client packaging, readiness, explicit world modes and paired persistence | CP-2, CP-3 | G5: AT-06, AT-07; both world choices and continuation demonstrated |
| CP-6 | Adapt metrics/logs/frame envelopes, bounded collectors, dashboards, local deterministic alerts | CP-1, CP-2 | G6: AT-09, AT-11, AT-15; live view independent of model |
| CP-7 | Budget gateway, type identity/key mapping, ledger, mocked provider and reconciliation | CP-6 | G7: AT-13, AT-14; no paid access before budget tests pass |
| CP-8 | Read-only observer Job, fixed prompts/schema, local fixtures, `gpt-6-luna` medium canary | CP-7 | G8: AT-10 to AT-14; quality/cost caps and outage fallback pass |
| CP-9 | Backup/restore drill, load/soak, legacy handoff rehearsal and rollback | CP-4 to CP-8 | G9: AT-08, AT-15, AT-16, AT-17; operational acceptance |
| CP-10 | Owner accepts cutover; old dispatcher drained; CI scheduling activated; operating record opened | CP-9 | G10: one admission authority, observed canary, rollback available |

CP-5 and CP-6 may be developed independently once prerequisites pass, but GPU validation remains serialized. No parallel agent work is implied. A Doom-only interim release can pass its applicable gates while Minecraft remains explicitly unavailable; it is not completion of the full proposal.

## 3. Expected implementation surfaces

Future changes likely include trainer/environment/service Dockerfiles under `docker/`; a new controlled deployment-manifest directory; CI workflows under `.github/workflows/`; a run submission/coordinator service; checkpoint/signal changes only where the audit finds gaps; telemetry adapter/export code; gateway/observer services; and meaningful integration tests. Exact file names are selected in each future work package. No placeholder executable files are added by this documentation package.

Preserve existing public interfaces until an explicit migration is ready. Do not simultaneously change learning algorithms, rewards, default world-model core or research thresholds to make an infrastructure test pass. Reuse the existing campaign's gates and lineage metadata rather than replacing them with approximate equivalents.

## 4. Checkpoint and scheduling milestone

The first useful end-to-end milestone is a disposable Doom run that builds from a fixed digest, waits for the GPU, trains for a bounded segment, commits complete state, releases resources, admits an urgent task and resumes the original lineage. Demonstrate correct restart and seed/timing records before continuous training is allowed.

Thirty minutes is the proposed continuous-segment duration. Measure startup, checkpoint and world save cost; target combined boundary overhead below 10% of segment time. If it exceeds that, optimize state transfer or propose a longer segment with its increased urgent-task latency made explicit. Never shorten a frozen evaluation to meet the target. A checkpoint cannot be called complete solely because its files exist.

## 5. Observation milestone

Start with local deterministic reports and offline sanitized fixtures. Validate that the observer cannot act and that prompt injection in evidence cannot escape report output. Exercise budget race, crash, timeout, rollover, stale-backup and type-spoofing cases with a fake provider. Only then enable a small paid canary charged to the same USD 20 program budget.

Observe one active run with the proposed 30-minute cadence and bounded payloads. Record actual input/output/reasoning/image usage, report completeness, factual support and CPU overhead. If medium reasoning exhausts the output cap, report incompleteness and revise the tested settings deliberately; do not silently increase limits or switch models.

## 6. Installation and operational handoff

[OPS-011](../operations/ops-011-owner-led-kubernetes-installation.md) is the owner's command companion. Resolve actual versions, addresses and existing-cluster status before executing its installation steps. The owner performs the commands in small groups and checks expected outcomes. Implementation supplies reviewed manifests before commands that reference them become executable. No pipe-to-shell installer or guessed live-host values are part of this plan.

Document the old-to-new campaign mapping, final legacy checkpoint, control authority switch and rollback checkpoint. Retain source/image versions necessary to restore the old path. Make normal operation automatic only after that handoff, not by giving the CI runner unrestricted cluster administration.

## 7. Validation, release and completion definition

[ASS-021](../assurance/ass-021-cluster-and-aiops-acceptance-plan.md) is the acceptance plan. A short boot test alone is insufficient. Run a proposed 24-hour disposable soak spanning multiple continuous segments, an urgent task, observer cycles, world save/resume, cleanup and controlled service failure. Perform node failure separately on disposable state. Freeze performance criteria before measuring; document any acceptance exception with its limitation.

The program is complete when all R-01 through R-13 requirements have linked evidence, the owner has performed/understood installation, CI submits reproducible runs automatically, priority works at safe boundaries, both Minecraft world modes restore, observer access is proven read-only, per-type budget accounting enforces the shared ceiling, idle workers disappear without losing state, and recovery/rollback has been rehearsed. Research success is not a completion criterion for infrastructure.

## 8. Current execution record

| Item | State as of 2026-10-28 |
|---|---|
| Requirements and architecture documents | Written for review |
| Cluster inspection / installation | Not performed |
| Application, manifests and workflow changes | Not performed |
| Runtime/security/recovery acceptance | Not run |
| Provider credentials / paid API canary | Not provisioned / not run |
| Existing training and services | Not contacted or modified by this task |

Future work appends evidence and status here or links a separate execution record. Do not rewrite proposed targets as observed results.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [OPS-009](../operations/ops-009-cluster-delivery-and-operations.md): Cluster delivery, automatic training and routine operations
<!-- backlinks:end -->
