# Cluster and AIOps threat model

> **Document control:** SEC-002 · Security / proposed infrastructure controls · **State:** Proposed; controls unimplemented and unverified · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Extends [SEC-001](sec-001-security-policy.md) for [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md). This is a project threat model, not a security certification.

## 1. Assets and trust boundaries

Protect brain checkpoints and replay, Minecraft worlds, scientific provenance, private logs/frames, cluster administration credentials, CI identities, API keys, budget ledger, host files and the availability of training. Boundaries are: public repository/PR to trusted CI; CI to run API; application namespaces to node/cluster administration; trainer to evaluator; telemetry to observer; gateway to external provider; primary storage to backup.

Treat PR code, third-party images, game chat/text, log content and model output as potentially hostile. A compromised node or cluster administrator can bypass pod boundaries; containers are not a substitute for host security. The initial cluster is a single-owner trusted laboratory, not a multi-tenant hostile-code service.

## 2. Permission matrix

| Identity | Permitted | Denied |
|---|---|---|
| Hosted PR CI | Build/test without production secrets | LAN, cluster credentials, private artifacts, training submission |
| Trusted delivery identity | Submit allowlisted run templates and read their status | Cluster-admin, arbitrary pod specs, arbitrary commands, Secret reads |
| Coordinator | Manage approved run resources in selected namespaces; run-specific locks/state | Cluster roles, arbitrary host mounts, unrelated Secrets |
| Trainer | Read declared inputs, write own artifact prefix, export telemetry | Kubernetes mutation, API keys, other lineage writes, evaluator feedback |
| Telemetry collector | Read selected metrics/logs and publish sanitized envelopes | Trainer execution/control, bulk credential/config scraping |
| Observer | Read assigned envelopes; append reports via scoped service | Secret reads, pod exec/attach, run mutation, direct provider access |
| Budget gateway | Read type-specific provider keys; write budget ledger; call allowlisted API | Trainer control, arbitrary URL fetching, user-chosen key selection |
| Human operator | Audited administrative and recovery operations | No automatic sharing of admin credentials with agents or CI |

Resource creation can itself enable privilege escalation. Validate submitted templates through admission controls: allowed service accounts, images/digests, volumes, resource requests and security contexts. RBAC alone does not make arbitrary Job creation safe. See [Kubernetes RBAC good practices](https://kubernetes.io/docs/concepts/security/rbac-good-practices/).

## 3. Threat and control register

| ID | Threat | Required control | Evidence |
|---|---|---|---|
| S-01 | Malicious PR reaches cluster | Hosted untrusted builds; separate trusted trigger; no secret-bearing execution of unreviewed code | AT-05 |
| S-02 | Observer follows instructions in logs/screenshots | Untrusted evidence framing, no tools, schema-only reports, no machine execution of output | AT-11 |
| S-03 | Provider keys leak or types impersonate each other | Gateway-only keys, workload identity mapping, redacted logs, rotation and egress restriction | AT-10, AT-14 |
| S-04 | Unexpected external disclosure | Allowlisted metrics, redaction, game-window-only captures, image suppression when uncertain | AT-11 |
| S-05 | Overprivileged pod escapes intended role | Restricted pod profile, no host sockets/files, admission enforcement, minimal service account | AT-10 |
| S-06 | Open offline-mode Minecraft or RCON | Internal service, private routing, firewall and per-workload NetworkPolicy | AT-06, AT-10 |
| S-07 | Duplicate writer or stale controller corrupts state | Fencing and confirmed old-writer shutdown; atomic generation manifests | AT-07, AT-08 |
| S-08 | Compromised or mutable dependency | Digest pinning, provenance, SBOM, scanning, reviewed exceptions, asset checksums | AT-05 |
| S-09 | Excessive API calls or billing races | Transactional reservations, caps, no bypass, fail closed | AT-13 |
| S-10 | Disk fill or noisy-neighbor denial of service | Quotas, resource limits, log caps, bounded retention and free-space admission | AT-02, AT-15 |
| S-11 | AIOps contaminates learning evidence | One-way evaluation exports, no training feedback, protocol-specific reports | AT-11 |
| S-12 | Stolen bootstrap/admin credentials | Private join tokens, short expiry, encrypted backups, separate admin account | AT-10, AT-16 |
| S-13 | Public repository reaches the self-hosted runner or leaks host values in public Actions logs | Prefer a private repository ([OPS-001](../operations/ops-001-ci-setup.md) precondition); otherwise only `workflow_dispatch` on self-hosted jobs, fork-PR approval for all outside collaborators, no secret-bearing trigger on a pull request, no host values in logs | AT-05 |

## 4. Pod, host and network baseline

Application namespaces enforce the Restricted Pod Security Standard where supported: non-root, no privilege escalation, dropped capabilities, seccomp RuntimeDefault, read-only root filesystem where practical, and explicitly writable scratch volumes. Disable service-account token automount unless a workload needs it. Any GPU/CNI/node collector exception is narrowly scoped to its system namespace and reviewed; do not weaken all application namespaces. [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/).

Do not migrate the existing privileged GPU-metrics container as an ordinary observer. Inventory its privileges and replace or isolate it. The Actions worker has its own host/container resource limits and no Docker socket or cluster-admin mount. Separate operator access from CI service credentials.

Default-deny ingress and egress, then permit only documented DNS, telemetry, artifact, game and gateway routes. Authenticate internal APIs even on the LAN. Keep control plane, kubelet, runtime sockets, etcd, dashboards and RCON off the public internet. Verify the CNI actually denies test traffic; an accepted NetworkPolicy object alone proves nothing. Cover host-network components with host firewall rules. See [ARC-007](../architecture/arc-007-cluster-platform.md).

## 5. Secrets and external data handling

Use Kubernetes Secrets with encryption at rest configured and tested, tight RBAC and protected backups, or an equivalent external secret store. Base64 encoding is not encryption. Keep key values, network addresses and login details out of tracked documentation and public CI logs. Use mounted secret files with minimal readers; never embed keys into images or command-line arguments.

Before enabling paid observation, review the provider's current data retention and account settings, send only the approved telemetry subset, and disable persistent conversation storage where supported. Do not claim that a storage setting eliminates all provider logging. Exclude host desktop captures, terminal content, chat/usernames unless explicitly allowed, credentials, private paths and entire raw logs. Sanitize before external transmission and before public artifact publication. If image redaction is uncertain, omit the image.

## 6. Incident response and authority expansion

For suspected key exposure: disable paid egress, revoke the affected type key, preserve sanitized audit/budget records, rotate the credential, reconcile charges, then re-enable after the cause is contained. For suspected node compromise: isolate the node, stop new admissions, preserve evidence, restore through the operator process and rotate exposed cluster/workload identities. Do not delete sole checkpoint copies during containment.

For misleading reports: mark them retracted, preserve their evidence references, disable the offending prompt/model version and rerun offline evaluation. Never let a report retract or promote a research result automatically.

Future action-capable AIOps requires a new scope decision, explicit allowed actions, separate executor identity, bounded rollback, audit records and additional tests. Nothing in this proposal authorizes it. Recovery runbooks are in [OPS-010](../operations/ops-010-cluster-storage-and-recovery.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-009](../architecture/arc-009-aiops-observation-and-budget.md): AIOps observation, reporting and API budget
- [OPS-009](../operations/ops-009-cluster-delivery-and-operations.md): Cluster delivery, automatic training and routine operations
<!-- backlinks:end -->
