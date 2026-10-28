# Cluster storage, retention and recovery plan

> **Document control:** OPS-010 · Operations / persistence and recovery · **State:** Proposed; backups and recovery targets unverified · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Supports [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md), [ARC-008](../architecture/arc-008-training-run-lifecycle.md) and [ARC-009](../architecture/arc-009-aiops-observation-and-budget.md). Existing baseline: [OPS-005](ops-005-operations.md).

## 1. Storage layout

| Data | Proposed primary | Independent recovery copy |
|---|---|---|
| Active trainer workspace/replay | Fast local persistent volume on `main` | Verified completed checkpoint generations on CPU-node storage |
| Committed run artifacts | Authenticated artifact service on selected CPU node | Separate physical disk/node; versioned, protected from routine deletion |
| Minecraft active world | Local persistent volume on its selected CPU worker | Paired save/checkpoint generations on another node |
| Immutable world baselines | Artifact store, immutable identifiers | Independent copy with hash verification |
| Coordinator/queue metadata | Transactional store on a stable CPU node | Frequent backup plus immutable transition journal |
| Budget ledger | Single active transactional writer on stable storage | Durable journal and backup; provider reconciliation before recovery |
| Metrics and reports | Bounded monitoring volumes and artifact store | Reports retained; raw metrics backup optional by declared retention |
| Control-plane state and PKI | Control-plane node | Encrypted datastore/PKI recovery bundle kept off that node |

Select actual directories, storage drivers and capacities after inventory. Do not publish private paths/addresses. Initial design avoids a distributed filesystem; local persistence plus verified artifact transfer is simpler on this hardware. Kubernetes PVs/PVCs preserve data independently of pod lifetime only according to their configured storage/reclaim behavior. Node-local volumes remain tied to a node. [Kubernetes persistent volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/).

Brain/world volumes use an explicit retain policy. Avoid owner references that delete important PVCs with a run or namespace. Application single-writer fencing remains mandatory even with a ReadWriteOnce volume; access mode is not an application lock. Backups are not verified until restoration succeeds.

## 2. Commit and transfer protocol

Write a new generation; flush and validate all required files; record hashes, sizes, image/config/core identity, parent checkpoint and world generation; then publish its complete manifest. Consumers accept only complete verified manifests. Keep staging files separate from committed generations. Upload to another physical node and read back/hash required data before marking the segment durably complete.

GPU compute can stop after local checkpoint completion, but the run remains archiving or recovery-required until its mandatory durable copy is verified. Do not promote an unverified generation as the resume input. Avoid retaining the GPU allocation merely to retry long artifact uploads: the future coordinator should finish the training Job after safe local commit and perform transfer through a bounded storage task while preventing dependent continuation until verification. Test this transition against ARC-008's completion rules.

## 3. Proposed retention and storage admission

| Class | Starting retention |
|---|---|
| Active brain/world and immutable baselines | Until explicit owner retirement; never age-pruned while referenced |
| Resume checkpoints | Latest three verified generations per active lineage plus all protected protocol baselines |
| Successful-run raw logs | 14 days, bounded per-run size; retain derived evidence longer |
| Failure logs/artifacts | 30 days minimum, longer while an incident is open |
| Routine local frames | 24-hour bounded ring; no continuous archival video by default |
| Incident frames | 7 days unless explicitly retained as evidence |
| Metrics | 14 days initially, with cardinality/size limits |
| Reports, manifests, protocol evidence | 180 days minimum; protected research evidence retained until owner review |
| Usage ledger and security audit | 13 months minimum, access restricted |
| Completed Kubernetes Jobs | TTL target 24 hours after completion, only after durable evidence is available |

These are proposed operational defaults, not permission to delete existing historical artifacts. Set actual byte quotas before activation. Admission requires room for the active workspace, one complete new checkpoint generation, transfer staging and a free-space reserve. Proposed reserve is the larger of 20% of the volume or two measured checkpoint generations. Measure replay growth separately. Prune only eligible completed data, never the only good backup or an input referenced by queued work.

Run a dry-run retention report first, showing paths, generations, references and reclaimable sizes. Research evidence under a frozen protocol overrides routine TTLs. A Job TTL is not a storage retention policy. [Kubernetes finished Job cleanup](https://kubernetes.io/docs/concepts/workloads/controllers/ttlafterfinished/).

## 4. Recovery objectives

Targets to validate, not promises: normal continuous-run work loss at most the current 30-minute segment after an abrupt GPU failure, provided the previous segment is durably committed; ordinary worker/service recovery within two hours of operator availability; full control-plane restoration within four hours of operator availability. Finite runs and frozen protocols have their own replay boundaries and may lose longer intervals.

An off-node copy protects against one disk/node failure, not site-wide power, theft or fire. At least one offline or offsite encrypted backup is recommended for irreplaceable research state; destination/cost are not selected by this proposal. Do not claim disaster recovery for the site until such a copy is restored successfully.

## 5. Failure runbooks

| Failure | Immediate action | Safe reopening condition |
|---|---|---|
| GPU worker crash | Mark attempt interrupted; stop dependent admission; preserve last verified generation | Old writer confirmed dead/fenced; compatible checkpoint and environment restored |
| Minecraft server crash | Stop useful training progression; record incident; do not treat reconnect as seamless resume | Restore paired world/brain state or replay declared unit with deviation recorded |
| Storage unavailable/full | Stop new stateful admissions; retain local state; alert operator | Required durable writes and free-space checks pass |
| Coordinator restart | Reconcile existing Jobs and journal; do not recreate blindly | Identity, lock and completion state agree |
| Network partition | No new conflicting writers; keep ambiguous ownership blocked | Old owner fenced or positively stopped; state reconciled |
| Control-plane loss | Existing workload may continue; freeze operational changes | Restore datastore/PKI and verify actual running Jobs before admitting work |
| API gateway/ledger loss | Deny paid requests; training/telemetry continue | Restore and reconcile spend, including uncertain/in-flight requests |
| Observer crash/provider outage | Keep deterministic local alerts; mark observation unavailable | Bounded retry policy and budget permit future observation |

Do not use a restart loop to hide a persistent failure. Every blocking state has a documented diagnostic and operator recovery path. Recovery may replay work; record this rather than claiming uninterrupted learning.

## 6. Restore drill and backup handling

On disposable destinations, restore a checkpoint, its paired world, queue metadata and control-plane backup. Verify hashes, identities, dependencies, key availability and a short resumed run. Separately restore a stale budget backup and prove paid calls remain blocked until reconciliation. Back up keys needed to decrypt backups separately and restrict access. Record elapsed time, data loss window, failures and evidence paths under the acceptance record.

Never rehearse destructive recovery on the sole live brain/world. The first GPU-loss drill is a disposable canary with no active campaign. Maintenance and rollback coordination are in [OPS-009](ops-009-cluster-delivery-and-operations.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-007](../architecture/arc-007-cluster-platform.md): Cluster platform and resource placement
- [OPS-011](ops-011-owner-led-kubernetes-installation.md): Owner-led Kubernetes installation and verification companion
- [SEC-002](../security/sec-002-cluster-and-aiops-threat-model.md): Cluster and AIOps threat model
<!-- backlinks:end -->
