# Containerization, Kubernetes and AIOps proposal

> **Document control:** GOV-017 · Governance / infrastructure proposal · **State:** Proposed; owner requirements recorded, implementation not authorized by this document · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. **Authority:** owner's documentation-only request and subsequent answers in the same conversation. No cluster inspection, installation, deployment, API purchase or application change was performed to produce this proposal.

## 1. Purpose and scope

Make SkyBot experiments reproducible, queue training on the existing GPU host, use spare capacity on the existing CPU machines, and provide inexpensive observation without giving an LLM control of training. Kubernetes administration is also a learning objective: the owner performs the installation using the staged instructions, with assistance interpreting results.

This package covers the proposed infrastructure program end to end. It does not replace the existing learning program or claim to redesign every learner subsystem. The universal brain and evaluator isolation remain governed by [ADR-001](../decisions/adr-001-shared-universal-brain.md), [ADR-002](../decisions/adr-002-evaluator-state-is-not-a-learning-input.md) and [ADR-004](../decisions/adr-004-retire-the-legacy-brain-on-this-branch.md).

## 2. Reading map and authority

| Document | Controlling subject |
|---|---|
| GOV-017, this document | Requirements, scope, decisions, risks and unresolved measurements |
| [GOV-018](gov-018-cluster-and-aiops-implementation-plan.md) | Work packages, sequencing, deliverables and release gates |
| [ADR-008](../decisions/adr-008-cluster-orchestration-and-observer-boundary.md) | Proposed durable architecture choices and alternatives |
| [ARC-007](../architecture/arc-007-cluster-platform.md) | Nodes, placement, capacity, cluster services and network paths |
| [ARC-008](../architecture/arc-008-training-run-lifecycle.md) | Run contract, queue, priority, checkpoints and environment lifecycle |
| [ARC-009](../architecture/arc-009-aiops-observation-and-budget.md) | Observation interface, token limits, credentials and monthly spending |
| [SEC-002](../security/sec-002-cluster-and-aiops-threat-model.md) | Threat model, permissions, data disclosure and security controls |
| [OPS-009](../operations/ops-009-cluster-delivery-and-operations.md) | CI/CD, automatic submission, daily operation, upgrades and rollback |
| [OPS-010](../operations/ops-010-cluster-storage-and-recovery.md) | Storage, retention, backups, restoration and failure recovery |
| [OPS-011](../operations/ops-011-owner-led-kubernetes-installation.md) | Owner-led installation command sequence and prerequisites |
| [ASS-021](../assurance/ass-021-cluster-and-aiops-acceptance-plan.md) | Traceable acceptance scenarios and evidence requirements |

All new documents are proposed. Numeric defaults are engineering starting points, not measurements or owner-approved service guarantees. The requirements below take precedence over those defaults. Change a controlling requirement explicitly rather than silently relaxing it in an implementation file.

## 3. Confirmed requirements

| ID | Owner requirement | Design / acceptance |
|---|---|---|
| R-01 | Existing GPU host performs training; no new cloud capacity assumed | ARC-007; AT-01, AT-03 |
| R-02 | Use available CPU capacity on `skybot`, `node1`, `node2`; each has 2 CPUs and 8 GB RAM with substantial storage | ARC-007; AT-01, AT-02 |
| R-03 | `skybot` is separate from the GPU host and currently runs the GitHub Actions worker; all three CPU nodes have Linux and Kubernetes commands installed | OPS-011; AT-01 |
| R-04 | Owner performs Kubernetes installation for practice, following instructions | OPS-011; G1 in GOV-018 |
| R-05 | AIOps initially observes and reports only; allow future separate types | ARC-009, SEC-002; AT-10, AT-11 |
| R-06 | Minimize observation frequency, input tokens and output tokens | ARC-009; AT-12 |
| R-07 | Prefer `gpt-6-luna` with medium reasoning through an external API | ARC-009; AT-12, AT-13 |
| R-08 | Total API spending limited to USD 20 per month; separate API credential per AIOps type for attribution | ARC-009; AT-13, AT-14 |
| R-09 | Active work finishes before another training task starts; higher-priority work precedes continuous training at a checkpointed segment boundary | ARC-008; AT-03, AT-04 |
| R-10 | Training sessions submitted automatically through CI/CD | OPS-009; AT-05 |
| R-11 | Each Minecraft run explicitly selects persistent-world resume or baseline-world clone | ARC-008, OPS-010; AT-06 |
| R-12 | Idle optional workers removed; important state retained | ARC-007, OPS-010; AT-07, AT-15 |
| R-13 | Full formal proposal and implementation plan; present work changes documents only | GOV-018; document integrity check |

The owner confirmed bounded checkpointed segments for continuous training. Proposed default: 30 minutes of active training per segment. The owner did not specify a duration or authorize forced interruption of a finite research run.

## 4. Repository baseline and discrepancies

The repository is public (owner, 2026-10-28) and [OPS-001](../operations/ops-001-ci-setup.md) requires private repositories for a self-hosted runner; that conflict is recorded in OPS-001 and bears on every CI-to-cluster path in [OPS-009](../operations/ops-009-cluster-delivery-and-operations.md). Source inspection found container definitions for the runner and monitoring under `docker/`, SSH-based deployment scripts, training through `scripts/run.py`, and an existing serial unattended campaign described by [ASS-013](../assurance/ass-013-unattended-test-campaign.md). These are migration inputs, not evidence that Kubernetes is currently operational.

[OPS-008](../operations/ops-008-host-access.md) calls the GPU host `main` and places the runner on `node1`. The owner's current statement places the worker on `skybot`; use that statement for this proposal. Preserve the older document until the actual operational inventory is reconciled. Do not copy addresses, login names or credentials into this public document set. Installed command-line tools do not prove that nodes have joined an initialized cluster.

Three existing facts are not reconciled by this package and must be settled at CP-0. (1) The existing Minecraft server is a Paper server on a separate LAN machine reached through a local bridge ([OPS-002](../operations/ops-002-game-server.md)), not a machine in the node list above; R-11's world modes and ARC-007's in-cluster Minecraft server assume the server moves into the cluster, which is a migration, not a given. (2) [CLAUDE.md](../../CLAUDE.md) section 7 records that world persistence was measured impossible in the MineRL fork's own server (each episode is a fresh world); persistent-world resume therefore depends on a standalone server, and AT-06 must prove it rather than assume it. (3) A brain mirror already pulls brain state to a CPU machine ([OPS-005](../operations/ops-005-operations.md)); OPS-010's backup copies must not collide with it or be mistaken for it.

Existing monitoring, campaign gates and research protocols must survive migration. Doom remains the first integration environment; Minecraft follows. Existing historical state must not be deleted to make the new platform fit.

## 5. Proposed operating model

CI produces an immutable image and a validated run request. A deterministic coordinator and Kubernetes queue admit one GPU segment at a time. Supporting resources use eligible CPU workers with measured headroom. The trainer persists state and exports telemetry. Short-lived observer Jobs read bounded, sanitized evidence through an API and write reports through a restricted reporting interface. Only the API gateway possesses provider credentials and admits paid requests against a shared budget.

Training continues when observation or the external provider is unavailable. Observation never supplies rewards, labels, actions, policy parameters or automatic research verdicts. Routine lifecycle automation belongs to the coordinator, not to the observer.

## 6. Scope limits and alternatives

Initial scope excludes distributed training, GPU time sharing, multiple simultaneous GPU trainers, cloud autoscaling, automatic model promotion, LLM shell access and autonomous code changes. Future AIOps types require separate permissions, credentials, sub-budgets and acceptance evidence. Their existence in this architecture is not permission to activate them.

Proposed platform is a small kubeadm-managed Kubernetes cluster with one control plane, containerd, a policy-enforcing CNI, NVIDIA device plugin and Kueue. This exposes standard administration tasks for the owner's practice. K3s remains an alternative if measured overhead is unacceptable. A single control plane and one GPU host are explicit availability limitations. Large free disks do not combine into a resilient storage pool automatically.

## 7. Risks and mitigations

| Risk | Consequence | Required response |
|---|---|---|
| CPU nodes already busy | Slow control plane, Minecraft ticks or metrics | Inventory existing services; admit by measured capacity; queue rather than crowd |
| Incomplete checkpoint state | Restart changes the experiment | Resume validation and declared reset semantics before continuous scheduling |
| Two controllers own the GPU | Conflicting trainers or corrupt state | Drain old campaign authority before enabling the new dispatcher |
| Priority backlog | Continuous training starvation or missed deadlines | Expose age and estimated starts; cap urgent submissions; report infeasible deadlines |
| Shared storage or power failure | Lost work and unavailable artifacts | Independent backups and restoration drills; no HA claim |
| Prompt injection or private screenshots | Unsafe output or data disclosure | No action tools, evidence filtering and external-input policy |
| Billing race, timeout or credential misuse | Budget exceeded | Serialized reservations, retained uncertain charges, gateway-only keys and egress |
| New runtime changes research timing | Misleading learning comparisons | Freeze images and compare infrastructure equivalence separately from learning |

## 8. Decisions still requiring measurements

Before installation, record OS and CPU architecture, installed Kubernetes/runtime versions, any existing cluster, real allocatable capacity, GPU model/count/driver, storage health/free space, network routes and speed, current services and required ports. Select compatible pinned component versions from current official documentation at implementation time. Proposed placement and resource envelopes in ARC-007 must be revised if inventory contradicts them.

Before enabling paid observation, verify account model access, actual pricing/usage fields, sanitized screenshot policy, retention settings and the gateway's charge bounds. Budget accounting uses UTC calendar months by proposal; the USD 20 limit covers this program's provider API usage, not electricity, other account workloads or taxes. If the owner intends an invoice-total limit including tax, reserve that overhead within the same USD 20 before enabling calls.

## 9. Change record

2026-10-28: initial documentation-only package. Owner requirements transcribed; checkpointed segments and owner-led installation confirmed. Implementation, performance, recovery and security outcomes remain unverified.
