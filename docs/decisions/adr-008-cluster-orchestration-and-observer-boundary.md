# Cluster orchestration and read-only AIOps boundary

> **Document control:** ADR-008 · Architecture decision / infrastructure · **State:** Proposed · **Custodian:** Repository maintainer · Revision history: Git.

**Record date:** 2026-10-28. **Source:** owner's proposal/documentation request and answers, recorded in [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md). Confirmed requirements are transcribed there; this record does not imply that all implementation choices are accepted or deployed. Format follows [GOV-006](../governance/gov-006-decision-record-process.md).

## 1. Context

SkyBot training currently depends on host processes and a serial campaign, while runner and monitoring containers already exist. The owner wants queued GPU training, automatic pipeline submission, supporting workloads on three existing small CPU nodes, and low-cost observation. The owner also wants to perform Kubernetes installation personally for practice.

## 2. Proposed decision

Use a small Kubernetes cluster, initially kubeadm with one control plane and containerd. Keep training exclusive on the existing GPU worker. Use Kueue for resource admission and a deterministic coordinator for run/world/checkpoint lifecycle. Continuous runs yield after checkpointed segments; priority never forcibly preempts an active segment. Support explicit persistent-world resume and baseline clone modes.

Deploy AIOps as short-lived read-only observers on eligible CPU workers. All paid model traffic passes through an authenticated budget gateway with a distinct provider credential per agent type and a shared USD 20 monthly ceiling. Use the owner's `gpt-6-luna` medium preference subject to account/pricing validation. Installation remains owner-operated and subsequent training submission becomes automatic through trusted CI.

Keep observation outside learning, preserving [ADR-002](adr-002-evaluator-state-is-not-a-learning-input.md). Container lifetime never defines research-state lifetime. Retain durable provenance, checkpoints, worlds and independently tested backups.

## 3. Alternatives considered

| Alternative | Benefit | Reason not selected initially |
|---|---|---|
| Continue Compose plus SSH | Lowest migration effort | Does not meet the Kubernetes practice and cross-node admission objective |
| K3s | Smaller packaged operational footprint | Retained fallback; kubeadm exposes more standard installation steps for practice |
| Three control planes | Better control-plane availability | Adds resource/maintenance cost on 2-CPU nodes; measure before expanding |
| Full GPU Operator | Integrated GPU lifecycle management | Existing single GPU host may only need toolkit/plugin; avoid unnecessary initial components |
| GPU time sharing or automatic preemption | Potentially lower waiting time | Conflicts with exclusive stateful training and finish-current-segment requirement |
| Autonomous LLM administrator | Faster speculative remediation | Outside confirmed observe/report authority and research-isolation boundary |
| Full GitOps/workflow platform immediately | Declarative delivery and rich workflow features | Start with a narrow run API and existing CI; add only when justified |

## 4. Consequences

More explicit run contracts, image builds, state persistence and operational checks are required. The GPU remains a throughput bottleneck. Segmentation can change environment behavior and must be validated. Single-control-plane and local-storage failure modes remain. API budget enforcement is an application responsibility with conservative failure behavior, not a provider-dashboard assumption.

No existing campaign, live trainer, world or monitoring service is changed by accepting this document set for review. Implementation follows [GOV-018](../governance/gov-018-cluster-and-aiops-implementation-plan.md), architecture [ARC-007](../architecture/arc-007-cluster-platform.md), and acceptance [ASS-021](../assurance/ass-021-cluster-and-aiops-acceptance-plan.md).
