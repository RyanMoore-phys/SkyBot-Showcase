# Documentation control and index

**Document ID:** DOC-001  
**Classification:** Repository documentation governance  
**Custodian:** Repository maintainer  
**Status:** Active  
**Revision control:** Git history is the authoritative revision history.

This index defines how the repository's authored documentation is organized,
identified, linked, reviewed, and retained. It is a repository practice inspired
by controlled-document methods; it does not claim compliance with a government
records regime or an external certification standard.

## 1. Authority and scope

The repository contains design intent, implementation references, operational
procedures, research protocols, experimental results, incident records, and
agent instructions. These records do not all have the same authority.

Use the following order when statements disagree:

1. A frozen protocol governs the experiment it names. Its results are reported
   in a separate record and do not silently amend the protocol.
2. Current code and configuration describe implemented behavior. Architecture
   and operations documents explain that behavior, but may lag behind it.
3. A document marked proposed, planned, draft, or unverified describes intent
   only; it is not evidence that a feature exists.
4. Audit, incident, and experiment records remain historical evidence. Later
   corrections should be appended or linked, not silently rewritten into the
   original account.
5. `README.md` is the public entry point. `CLAUDE.md` is the repository's
   agent-facing working and incident record; neither replaces this register.

When a document conflicts with executable behavior, record the discrepancy and
update the document or code in the same change where practical. For an
experiment, preserve the preregistration and make the deviation explicit.

## 2. Document identification and naming

Canonical authored documents under `docs/` use a permanent identifier and a
lowercase kebab-case filename:

```text
<category>-<three-digit-sequence>-<short-description>.md
```

For example, `ops-005-operations.md` has the permanent ID `OPS-005`. The ID is
the stable citation; the title and filename may be clarified, but an assigned
ID is not reused. A move or rename must update this register and all repository
links in the same change.

Repository entry-point names such as `README.md` and `CLAUDE.md` retain their
conventional names. Machine-generated experiment reports retain their
experiment identifiers, timestamps, hashes, and generated naming because those
names carry provenance. They are records, not canonical authored guidance.

## 3. Lifecycle and document control fields

Each controlled document begins with a control line that identifies its ID,
class, lifecycle state, and custodian. Git commits provide the revision history;
no independent approval board, signed release, or version number is implied.

Lifecycle states used in this repository:

| State | Meaning |
|---|---|
| **Active reference** | Maintained guidance for a current design or operating practice; verify time-sensitive steps before use. |
| **In progress** | Implementation or evaluation is underway; completion claims remain bounded by the stated evidence. |
| **Proposed** | Design intent only; not an implementation claim. |
| **Protocol** | Defines a planned or frozen procedure. Changes after results exist must be disclosed. |
| **Record** | Historical audit, report, result, or decision evidence. Preserve its original scope and date. |
| **Register** | A maintained inventory of assumptions, milestones, telemetry, or controlled items. |

Status is descriptive, not an approval. Where the document itself contains a
more specific status, the more specific statement governs.

## 4. Directory map

| Directory | Document class | Intended use |
|---|---|---|
| [`governance/`](governance/) | Roadmaps, objectives, program plans, and audit findings | What the project is trying to do, what is next, and what evidence or decisions constrain it. |
| [`architecture/`](architecture/) | System architecture and design | How components fit together and what is proposed or implemented. |
| [`operations/`](operations/) | Setup, deployment, runtime operations, and monitoring | How to provision, run, and observe the system. |
| [`assurance/`](assurance/) | Protocols, evaluations, audits, and system reports | How claims are checked and what evidence currently supports them. |
| [`research/`](research/) | Research questions, technical proposals, and experiment plans | Why a research direction is being pursued and how it can be falsified. |
| [`foundation/`](foundation/) | Shared foundation design, assumptions, and telemetry registers | Cross-cutting contracts and supporting research infrastructure. |
| [`decisions/`](decisions/) | Architecture decision records | Durable decisions, their context, and consequences. |
| [`security/`](security/) | Security policy and threat model | Security boundaries, handling practices, and vulnerability reporting. |
| [`legacy/`](legacy/) | Superseded or retired records, in the same category subfolders (`legacy/decisions/`, `legacy/assurance/`) | Kept for history. An ID is permanent, so a moved record keeps its name, ID and register row; its state says it is legacy. See also the repository-level [`../legacy/`](../legacy/README.md) for retired files that are not documents. |
| [`../experiments/`](../experiments/) | Preregistrations, generated reports, and experiment records | Immutable or provenance-bearing evidence outputs. |

## 5. Controlled document register

The register is the authoritative lookup for canonical `docs/` documents.
Lifecycle states are intentionally conservative; a `Record` does not mean a
claim is currently true, and an `Active reference` does not certify that every
line has been recently revalidated.

### Governance

| ID | Canonical document | State | Scope |
|---|---|---|---|
| GOV-001 | [Research roadmap](governance/gov-001-research-roadmap.md) | Record / roadmap | Research ladder, claims, evidence status, and strategic direction. |
| GOV-002 | [Next objectives](governance/gov-002-next-objectives.md) | In progress | Near-term work items and their stated completion/evidence state. |
| GOV-003 | [Audit findings](governance/gov-003-audit-findings.md) | Record | Chronological incident and failure findings; retain corrections and retractions. |
| GOV-004 | [Improvement plan](governance/gov-004-improvement-plan.md) | Proposed / in progress | Cross-cutting plan and its staged work. |
| GOV-005 | [Claims and evidence register](governance/gov-005-claims-evidence-register.md) | Register | Material project claims, evidence scope, and limitations. |
| GOV-006 | [Decision record process](governance/gov-006-decision-record-process.md) | Active reference | How durable decisions are recorded and superseded. |
| GOV-007 | [Contributor and change guide](governance/gov-007-contributor-and-change-guide.md) | Active reference | Change, documentation, evidence, and security practices. |
| GOV-008 | [Human-like learning program: motivation](governance/gov-008-human-like-learning-motivation.md) | Proposed | Why SkyBot's learning is being changed, the owner's decisions, candidate work packages, required A/B tests and future documents, and a change log. |
| GOV-009 | [Implementation plan: human-like learning program](governance/gov-009-implementation-plan.md) | In progress (build-only; untested) | Per-difference theory, intended change, explicit code changes, testing and inclusion procedure; schedule, risks and coverage of every recorded difference. |
| GOV-010 | [Relational implementation plan](governance/gov-010-relational-implementation-plan.md) | In progress (build-only; untested) | Second layer built on GOV-009: a relation registry and per-cluster plans to connect the learning blocks as the map does. |
| GOV-011 | [Execution record: block plan](governance/gov-011-execution-record-gov-009.md) | In progress | Package-by-package record of the GOV-009 build (build-only mode): state, files, switches, deviations, tests to run later. |
| GOV-012 | [Execution record: relational plan](governance/gov-012-execution-record-gov-010.md) | In progress | Package-by-package record of the GOV-010 build (build-only mode). |
| GOV-013 | [Categorical system testing plan](governance/gov-013-categorical-system-testing-plan.md) | Proposed execution plan; scope confirmed | Functional groups, within-group coherence, cross-group boundaries, readiness gates and serialized host schedule. |
| GOV-014 | [Corrective plan after the AB-00 interim analysis](governance/gov-014-corrective-plan-after-ab00.md) | Approved by owner 2026-10-22; documents revised before any change; execution recorded in section 10 | Source diversity (WP-6), repairs to the intrinsic-signal findings F-13 and F-04, and a fresh-control arm; tests and queue changes. |
| GOV-015 | [Corrective plan for the categorical-harness findings and the host contract-tier failures](governance/gov-015-corrective-plan-categorical-findings.md) | Owner instruction 2026-10-23; plan written, execution recorded in section 9 | Fixes for 24 open findings of ASS-014 (guards, latches, persistence, wrong behaviour), the host contract-tier triage, the new tests, predictions. |
| GOV-016 | [Implementation plan: predictive-coding core versus the RSSM (WP-20)](governance/gov-016-predictive-coding-core-plan.md) | Owner instruction 2026-10-25; built, nothing run; tests queued on `main` | Second selectable latent core, the pilot gate, the A/B, queue, predictions. |
| GOV-017 | [Containerization, Kubernetes and AIOps proposal](governance/gov-017-cluster-and-aiops-proposal.md) | Proposed | Owner requirements, document map, scope, risks and unresolved inventory for the cluster program. |
| GOV-018 | [Cluster and AIOps implementation plan](governance/gov-018-cluster-and-aiops-implementation-plan.md) | Proposed | Phased work packages, owner-led installation, migration, release gates and completion criteria. |

### Architecture

| ID | Canonical document | State | Scope |
|---|---|---|---|
| ARC-001 | [Architecture](architecture/arc-001-architecture.md) | Active reference | Main learner topology, subsystems, reward structure, actions, and configuration. |
| ARC-002 | [Flow maps](architecture/arc-002-flow-maps.md) | Active reference | Current data and control paths, tied to observed implementation. |
| ARC-003 | [General infrastructure](architecture/arc-003-general-infrastructure.md) | Mixed: implemented and proposed | Infrastructure rationale and implementation ledger; use its per-item status. |
| ARC-004 | [Universal games](architecture/arc-004-universal-games.md) | In progress | Shared agent and game-adapter design; live-host claims are explicitly bounded. |
| ARC-005 | [Game onboarding standard](architecture/arc-005-game-onboarding-standard.md) | In progress (built, untested) | The game interface and evaluator-side detector a new game supplies, the shared environment contract and both timing styles, and the evaluator-isolation rule. |
| ARC-006 | [Relation registry](architecture/arc-006-relation-registry.md) | In progress (built, untested) | Each line of the owner's map as a switchable, measured channel `rel.<FROM>.<TO>`; generated map copy, conformance report, dead-wire alarm and evaluator-leakage guard. RW-8 adds representation metrics, RW-9 peer diagnostics, RW-11 skill diagnostics, and RW-14 context-width telemetry. |
| ARC-007 | [Cluster platform](architecture/arc-007-cluster-platform.md) | Proposed | Existing GPU and CPU nodes, placement, resource envelopes, networking and availability limits. |
| ARC-008 | [Training run lifecycle](architecture/arc-008-training-run-lifecycle.md) | Proposed | Run contract, exclusive GPU admission, priority at checkpointed boundaries and world lifecycle. |
| ARC-009 | [AIOps observation and budget](architecture/arc-009-aiops-observation-and-budget.md) | Proposed | Read-only observation, token limits, separate type credentials and shared USD 20 monthly budget. |

### Decisions

| ID | Canonical document | State | Scope |
|---|---|---|---|
| ADR-001 | [One shared brain for universal games](decisions/adr-001-shared-universal-brain.md) | Accepted | One shared policy and world model across supported games. |
| ADR-002 | [Evaluator state is not a learning input](decisions/adr-002-evaluator-state-is-not-a-learning-input.md) | Accepted | Keep privileged engine state outside learning inputs and targets. |
| ADR-003 | [Preserve legacy brain isolation](legacy/decisions/adr-003-preserve-legacy-brain-isolation.md) | Superseded on this branch (ADR-004) | Keep legacy Minecraft state separate from the universal path. |
| ADR-004 | [Retire the legacy brain on this branch](decisions/adr-004-retire-the-legacy-brain-on-this-branch.md) | Accepted | The universal brain is the only target on this branch; legacy compatibility is out of scope. |
| ADR-005 | [Order of response when a change needs more compute](decisions/adr-005-compute-pressure-order.md) | Accepted | Reduce model size first, then streams, then stepping rate; thresholds still to define. |
| ADR-006 | [Parallel streams may observe one another](decisions/adr-006-streams-may-observe-one-another.md) | Accepted | Peer observation is permitted within ADR-001 and ADR-002; mechanism undecided. |
| ADR-007 | [Two selectable world-model cores until a comparison decides](decisions/adr-007-two-selectable-world-model-cores.md) | Accepted | The RSSM and the predictive-coding core both stay in the tree until WP-20 returns. |
| ADR-008 | [Cluster orchestration and observer boundary](decisions/adr-008-cluster-orchestration-and-observer-boundary.md) | Proposed | Kubernetes orchestration choices, alternatives and read-only AIOps authority. |

### Security

| ID | Canonical document | State | Scope |
|---|---|---|---|
| SEC-001 | [Security policy and threat model](security/sec-001-security-policy.md) | Active reference | Repository security boundaries, handling, and vulnerability reporting. |
| SEC-002 | [Cluster and AIOps threat model](security/sec-002-cluster-and-aiops-threat-model.md) | Proposed | Trust boundaries, identities, permissions, disclosure controls and incident response. |

### Operations

| ID | Canonical document | State | Scope |
|---|---|---|---|
| OPS-001 | [CI setup](operations/ops-001-ci-setup.md) | Active reference | Self-hosted CI setup and security assumptions. |
| OPS-002 | [Game server](operations/ops-002-game-server.md) | Active reference | External Minecraft server setup and connectivity. |
| OPS-003 | [MineRL build](operations/ops-003-minerl-build.md) | Active reference | MineRL provisioning and build failure diagnosis. |
| OPS-004 | [Monitoring](operations/ops-004-monitoring.md) | Active reference | Metrics stacks, dashboards, and extension. |
| OPS-005 | [Operations](operations/ops-005-operations.md) | Active reference | Runtime procedures, status, recovery, and backups. |
| OPS-006 | [Replication guide](operations/ops-006-replication.md) | Active reference | Reprovisioning from clean machines. |
| OPS-007 | [Runner mini](operations/ops-007-runner-mini.md) | Active reference | Self-hosted runner deployment on the Linux Mac mini. |
| OPS-008 | [Host access](operations/ops-008-host-access.md) | Active reference | How to reach `main`, where each value lives (never in the repository), what to ignore as stale, and what to do when a connection fails. |
| OPS-009 | [Cluster delivery and operations](operations/ops-009-cluster-delivery-and-operations.md) | Proposed | Trusted CI submission, campaign handoff, daily operation, maintenance and rollback. |
| OPS-010 | [Cluster storage and recovery](operations/ops-010-cluster-storage-and-recovery.md) | Proposed | Durable state, retention, backup, restoration and failure procedures. |
| OPS-011 | [Owner-led Kubernetes installation](operations/ops-011-owner-led-kubernetes-installation.md) | Proposed | Staged command companion, prerequisites, expected checks and stop conditions for owner execution. |

### Assurance and evidence

| ID | Canonical document | State | Scope |
|---|---|---|---|
| ASS-001 | [Continuous-control ablation](legacy/assurance/ass-001-ablation-continuous-control.md) | Record | Reported developmental-machinery ablation on Hopper-v4. |
| ASS-002 | [Action-stall audit](assurance/ass-002-action-stall-audit.md) | Record | Historical diagnosis and subsequent recovery account. |
| ASS-003 | [Policy-learning protocol](assurance/ass-003-policy-learning.md) | Protocol / in progress | Preregistered policy-learning diagnosis and evaluation. |
| ASS-004 | [System report](assurance/ass-004-system-report.md) | Record | Dated system status summary with links to detailed records. |
| ASS-005 | [Testing plan](assurance/ass-005-testing-plan.md) | Protocol | Verification plan for the foundation machinery. |
| ASS-006 | [Real-time action-lag incident](assurance/ass-006-realtime-action-lag-incident.md) | Record | Real-time transitions paired with the wrong action; contributing defects, fixes, and verification status. |
| ASS-007 | [Adaptability evaluation protocol](assurance/ass-007-adaptability-evaluation-protocol.md) | Protocol (draft, not frozen) | The program's main metric (adaptability by owner-set checkpoints) and the five-seed, one-hour evidence standard. |
| ASS-008 | [Branch documentation audit](assurance/ass-008-audit-adjustments-branch.md) | Record | A reviewer's audit of the branch's documentation and plan: issues and questions, 2026-10-21. |
| ASS-009 | [Audit response: Adjustments branch](assurance/ass-009-audit-response-adjustments-branch.md) | Record | Disposition of each finding in the branch documentation audit, what was changed, and what the owner must decide. |
| ASS-010 | [Current system limitations](assurance/ass-010-current-system-limitations.md) | Record | Dated baseline of what the system could not do on 2026-10-21, with the evidence for each limitation. |
| ASS-011 | [Main evaluation readiness and run record](assurance/ass-011-main-evaluation-readiness.md) | Doom A/A development calibration running on `main` | Records Doom held-out-map evaluation setup, AB-00 schedule, and results. |
| ASS-012 | [Categorical system testing protocol](assurance/ass-012-categorical-system-testing-protocol.md) | Draft template; harness built, first run recorded in ASS-014; no category run frozen under it | Integration contracts, boundary/fault scenarios, resource criteria and reporting; complements A/A and T1/T2. |
| ASS-013 | [Unattended test campaign](assurance/ass-013-unattended-test-campaign.md) | Record; installed on `main` and running | The queue that runs A/A, tests, pilots, T1/T2 A/Bs and categorical tests with no operator; its rules and escapes. |
| ASS-014 | [Categorical harness first-run findings](assurance/ass-014-categorical-harness-first-run-findings.md) | Record; no finding fixed | The 28 defects and design gaps the categorical harness found in its first run, with evidence and the consequences for the queued runs. |
| ASS-015 | [AB-00 Doom A/A interim analysis](assurance/ass-015-ab00-interim-analysis.md) | Record; interim, AB-00 still running | Sixteen finished A/A runs: operational health, learning signals, why the checkpoint metric is empty, the noise floor that exists, and recommended structural changes. |
| ASS-016 | [AB-00 final result and the fresh-control comparison](assurance/ass-016-ab00-final-and-fresh-control.md) | Record; AB-00 complete, fresh control complete | Twenty A/A runs and ten fresh-control runs: no transfer visible on the held-out maps, the ten-pair noise floor, and what it means. |
| ASS-017 | [Host test stages and two queue incidents](assurance/ass-017-host-test-stages-and-queue-incidents.md) | Record; contract tier re-queued, five suites unexplained | First host run of the unit and contract tiers (13 red suites classified), the latch that blocked every A/B, and the restart hazard. |
| ASS-018 | [Campaign progress check, 23 October 2026](assurance/ass-018-campaign-progress-2026-10-23.md) | Record; queue running, no behavioural result yet | Queue state, WP-6 coverage pass, contract tier second run (9 red), categorical harness first complete run, time left. |
| ASS-019 | [Predictive-coding core versus the RSSM: comparison protocol](assurance/ass-019-predictive-coding-core-comparison-protocol.md) | Protocol; pre-registered, no result yet | Pilot gate, same-snapshot A/B, measures, decision rules, limits. |
| ASS-020 | [Recent changes: static implementation and documentation audit](assurance/ass-020-recent-changes-static-audit.md) | Record; runtime unverified | Source-only audit of PC, evaluation gating, consolidation, launch migration and documentation; no tests run. |
| ASS-021 | [Cluster and AIOps acceptance plan](assurance/ass-021-cluster-and-aiops-acceptance-plan.md) | Proposed protocol; not frozen, not run | Requirement-linked infrastructure, scheduling, security, budget, recovery and performance validation. |

### Research

| ID | Canonical document | State | Scope |
|---|---|---|---|
| RES-001 | [Central behavioral problem and plan](research/res-001-central-behavioral-problem-and-plan.md) | Proposed | Action-credit diagnosis, proposed changes, and evidence gates. |
| RES-002 | [Perspective learning](research/res-002-perspective-learning.md) | In progress | Learning spatial structure from motion and visual observations. |
| RES-003 | [Plurality roadmap](research/res-003-plurality-roadmap.md) | In progress | Phased work toward richer observation structure. |
| RES-004 | [RSSM size experiment](research/res-004-rssm-size-experiment.md) | Protocol / in progress | Evidence plan for model-size and growth decisions. |
| RES-005 | [Skill discovery](research/res-005-skill-discovery.md) | Proposed | Game-agnostic discovery and validation of repeatable effects. |
| RES-006 | [Human-like learning gap analysis](research/res-006-human-like-learning-gap-analysis.md) | Record | Dated term-by-term and relationship-level comparison of SkyBot against the AI-counterpart learning map (`assets/ai_learning.mmd`). |
| RES-007 | [Hard and deferred differences](research/res-007-hard-and-deferred-differences.md) | Proposed | Register of map differences that cannot be built reliably now: reason, interim approximation, and revisit trigger. |
| RES-008 | [Relationship gap analysis](research/res-008-relationship-gap-analysis.md) | Record | Every line of `assets/ai_learning.mmd` (326 compared) against SkyBot: connected, partial, missing, deliberate or not established. |
| RES-009 | [Hard and deferred relationships](research/res-009-hard-and-deferred-relationships.md) | Proposed | Register of map lines that cannot be built reliably now: reason, interim approximation, revisit trigger. |
| RES-010 | [Predictive coding as the world model's latent core](research/res-010-predictive-coding-core.md) | Proposed; built behind a default-off switch, untested | Motivation, design mapped onto the RSSM, hypotheses with ways to fail, out of scope. |

### Foundation

| ID | Canonical document | State | Scope |
|---|---|---|---|
| FND-001 | [A/B protocol](foundation/fnd-001-ab-protocol.md) | Protocol | Frozen comparisons, preregistration, and reporting rules. |
| FND-002 | [Assumption register](foundation/fnd-002-assumptions.md) | Register | Explicit assumptions and their code counterparts. |
| FND-003 | [Baseline audit](foundation/fnd-003-baseline-audit.md) | Record | Code-reading baseline audit and unresolved live evidence. |
| FND-004 | [Experience reconciliation](foundation/fnd-004-experience-reconciliation.md) | Design | Responsibilities and boundaries for experience stores. |
| FND-005 | [Learning-process improvements](foundation/fnd-005-learning-process-improvements.md) | Record / design | Implemented scope and limits of foundation improvements. |
| FND-006 | [Foundation milestones](foundation/fnd-006-milestones.md) | Register | Milestone and evidence status by foundation stage. |
| FND-007 | [Native runtime](foundation/fnd-007-native-runtime.md) | Decision record | Evidence threshold for considering a native runtime. |
| FND-008 | [Learning telemetry inventory](foundation/fnd-008-telemetry-inventory-learning.md) | Record | Dated inventory of learning telemetry and measurement limits. |
| FND-009 | [Operations telemetry inventory](foundation/fnd-009-telemetry-inventory-operations.md) | Record / proposal | Dated operations telemetry audit and proposed work. |

## 6. Change control and maintenance

When adding or materially revising a canonical document:

1. Assign the next unused ID in the appropriate series and use the naming
   protocol above. Never recycle an ID.
2. Add or revise the control line and register entry in the same change.
3. State the purpose, scope, lifecycle state, evidence basis, and known limits
   near the beginning of the document. Separate implemented behavior from
   planned work and measured results from interpretation.
4. Update links from `README.md`, code comments, configuration comments, and
   other documents when a canonical path changes. Prefer a link to a stable
   document over a prose-only filename reference.
5. For protocols, retain the frozen hypothesis, metrics, seeds, and failure
   conditions. Record deviations and results separately.
6. For historical records, append corrections with dates and preserve the
   original finding, including retractions and superseding evidence.
7. **Cross-references are reciprocal.** If a document links another, the other must link back, and no controlled
   document may stand alone. Cite related records in the document's own "Related records" section; the
   generated "Referenced by" block at the end of each file lists the documents that cite it but that it does
   not cite. Never edit that block by hand: run `python3 tools/doc_backlinks.py` after changing links (it is
   idempotent), and `python3 tools/check_docs.py` fails on a stale block or an isolated document.
8. Use the Git commit as the revision event. A substantive change should make
   its reason and affected document IDs clear in the change description.

Review a document when its linked code or configuration changes, a procedure
fails during use, new evidence changes a claim, a protocol is about to be run,
or a security or operational dependency changes. Operational procedures should
be checked against the actual deployment before they are followed. Research
status should identify whether evidence is synthetic, offline, shadow, or live.

## 7. Code-to-document references

Code comments should link to the controlling document ID or canonical path when
they describe a non-obvious invariant, a dated design decision, an experiment
gate, or an operational constraint. Comments should explain the local reason;
the document should carry the broader rationale and evidence. Avoid duplicating
large procedures or mutable status summaries in comments.

This reorganization updates repository references to canonical paths. If a
future move or rename is required, update source comments, configuration
comments, Markdown links, and this register in the same change.

## 8. Records that retain their existing names

The following are deliberately outside the canonical naming scheme:

- `README.md` and `CLAUDE.md` use established tool and repository entry-point
  names.
- Experiment artifacts under `experiments/` preserve experiment IDs, prereg
  hashes, timestamps, and generated-output names for traceability. They are
  evidence records, not canonical authored guidance.
- `roms/README.md` and `scripts/brain_mirror/README.md` are directory-level
  navigation files.

Generated records should not be renamed without updating the producing
workflow and provenance references.

## 9. Documentation integrity check

The `documentation` job in `.github/workflows/ci.yml` runs
`python tools/check_docs.py` on a hosted runner. It checks canonical document
names and IDs, index registration, local Markdown links in the root README
and `docs/`, and reciprocal cross-references (`tools/doc_backlinks.py`). This is a static
documentation integrity check; it does not execute project tests or validate
the factual currency of operational and research statements.

## 10. Security and change guidance

The repository security policy is [SEC-001](security/sec-001-security-policy.md).
Contributor and change practices are maintained in
[GOV-007](governance/gov-007-contributor-and-change-guide.md); no separate
root-level contribution document is maintained.
