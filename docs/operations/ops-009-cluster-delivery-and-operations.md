# Cluster delivery, automatic training and routine operations

> **Document control:** OPS-009 · Operations / future delivery procedure · **State:** Proposed; not an instruction to alter the live deployment now · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Scope: [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md). Work packages: [GOV-018](../governance/gov-018-cluster-and-aiops-implementation-plan.md). Existing CI reference: [OPS-001](ops-001-ci-setup.md).

## 1. Separation of installation and operation

The owner installs and verifies Kubernetes using [OPS-011](ops-011-owner-led-kubernetes-installation.md). Once the platform and application release gates pass, approved pipeline events automatically submit training. Owner-led installation does not mean manually starting every experiment. This proposal itself changes neither workflows nor cluster state.

## 2. Trusted CI/CD sequence

1. Hosted CI checks source, document integrity and appropriate tests without cluster credentials. Preserve the repository's current rule that host-dependent project testing is queued on `main`; do not move GPU tests onto a development laptop.
2. A trusted build stage builds versioned trainer, environment and service images. Pin base/dependency versions, generate an SBOM/provenance record, scan for vulnerabilities and retain reviewed exceptions. Build only redistributable assets; game ROMs or licensed client content are provisioned through authorized private paths.
3. Publish images to an authenticated registry by digest. A private registry such as GHCR is a candidate, not a mandatory new local service. Verify that all eligible CPU architectures have an image. Keep GPU-specific binaries in the GPU image.
4. After required checks, the trusted delivery stage submits a schema-validated request to the run API. It has no kubeconfig allowing arbitrary pod creation. The API maps approved template names to reviewed workloads and records caller, workflow and digest.
5. CI reports the durable run ID and queue link. It need not occupy a runner for the full training duration. Retries with the same idempotency key return the existing run. Training status is tracked by the platform after CI exits.
6. Only after acceptance may an approved continuous-training schedule create successor segments automatically. Frozen runs never change images because a newer build appears.

Proposed triggers: manual `workflow_dispatch` with validated parameters; an owner-configured schedule for continuous training; optionally a successful protected-branch build for a predefined finite validation task. Do not make every PR/push start production training. While the repository is public, a job that can reach the run API or the cluster runs only on `workflow_dispatch` or a protected-branch event by an owner-controlled identity, never on `pull_request`, `pull_request_target` or fork-originated events ([OPS-001](ops-001-ci-setup.md) precondition and 2026-10-28 note; [SEC-002](../security/sec-002-cluster-and-aiops-threat-model.md) S-13). Making the repository private would remove this constraint. A protected-branch trigger must identify the tested commit, not fetch whatever HEAD happens to be at dispatch time.

## 3. Request defaults and policy

Normal finite validation is the default priority. Only the owner's trusted trigger can request urgent priority or a deadline. Continuous schedules submit one logical run and one successor at a time. Duplicate CI events must not produce duplicate learning sessions.

All requests specify brain lineage, seed/config/protocol, image digest and world mode as defined in [ARC-008](../architecture/arc-008-training-run-lifecycle.md). A missing world choice or unavailable checkpoint is a validation error, not an invitation to start fresh. No workflow input is interpolated into an arbitrary shell command. Reject unsafe paths and unapproved image/template combinations.

## 4. Existing campaign migration

[ASS-013](../assurance/ass-013-unattended-test-campaign.md) contains important serial execution, frozen snapshot and research-gate behavior. Preserve the campaign graph and evidence. First support a dry-run mapping from existing stages to proposed requests, preserving stage identity, dependencies, seed/arm, informational versus blocking failures and manual scientific gates.

During cutover, stop new legacy admissions at a natural boundary, let the active stage finish, export and back up its queue state, confirm no legacy trainer/supervisor can restart, then enable the new dispatcher. Exactly one system owns admission. A campaign integration may submit to the run API, but it must not separately launch host processes. Preserve the old deployment and state for rollback without running both paths at once.

Automatic infrastructure recovery must not approve thresholds, change preregistered seeds, accept code changes into a frozen campaign, or mark blocked scientific stages passed. Those existing gates remain visible.

## 5. Daily operator view

The dashboard should show queued/running/blocked/completed requests with reasons; priority and queue age; active segment and next boundary; CPU/RAM/GPU allocation; server readiness; checkpoint generation/age; storage free space; collector freshness; observer status; and per-type settled/reserved/month-to-date API cost. Display failed and unknown states explicitly.

Proposed routine: inspect failed admissions and storage headroom daily; review spend, stale backups and observer quality weekly; test restoration and review component versions monthly. No email/chat integration is assumed. Reports stay in the authenticated dashboard/artifact store unless the owner separately configures a notification destination.

## 6. Operator actions and effects

| Action | Required behavior |
|---|---|
| Submit urgent task | Queue ahead of lower-priority eligible work; preserve active segment |
| Pause continuous run | Prevent successor admission; active segment completes and checkpoints |
| Cancel queued run | Remove eligibility; preserve its request and reason |
| Stop active run safely | Request declared safe boundary; display drain progress |
| Emergency stop | Operator-only termination; record interruption and recovery status |
| Disable paid observation | Close gateway admission; keep local telemetry and training |
| Maintain GPU worker | Stop new admissions, drain/checkpoint active work, verify outputs, then maintain |
| Rotate agent key | Gateway credential rotation with attribution history; no observer pod receives the key |

Application commands/endpoints for these actions will be delivered in implementation. Do not document invented working CLI commands before they exist. Kubernetes inspection commands are supplied in OPS-011.

## 7. Updates and rollback

Pin component versions and deploy one change at a time. Test new images on disposable state, measure runtime equivalence, then permit new run submissions to use the digest. Active/frozen runs retain the old image. Schema migrations need backups, compatibility checks and a tested reversal or forward-recovery plan.

For a failed service release, stop new admissions, preserve evidence, restore the previous image/config and reconcile locks and ledger before reopening. An image rollback is not a state rollback. A newer checkpoint may be incompatible with older software; use a compatible prior generation or block rather than force-load it. Never roll a budget ledger backwards and resume paid traffic without reconciliation.

For full platform rollback, stop Kubernetes training admission, finish or explicitly interrupt the active segment, verify durable artifacts, fence the Kubernetes writer, and only then re-enable the previous host dispatcher. Keep run/protocol records for any interrupted work. Restoration and rollback targets are in [OPS-010](ops-010-cluster-storage-and-recovery.md).
