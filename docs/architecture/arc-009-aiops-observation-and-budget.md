# AIOps observation, reporting and API budget

> **Document control:** ARC-009 · Architecture / observation and cost control · **State:** Proposed; no API calls or credentials provisioned · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Requirements: [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md). Threat model: [SEC-002](../security/sec-002-cluster-and-aiops-threat-model.md). Acceptance: [ASS-021](../assurance/ass-021-cluster-and-aiops-acceptance-plan.md).

## 1. Authority and extension model

Initial agent type is `observer`: observe training and cluster health, identify uncertainty, and produce a concise report with evidence references. It cannot change code, runs, priorities, rewards, resources, worlds, checkpoints, prompts or its own budget. It has no command execution or action tools. Routine Kubernetes controllers remain deterministic infrastructure, not AIOps agents.

Future types such as investigator, cost analyst or change proposer receive independent identities, provider keys, schema versions and sub-budgets. Each requires a separate permission review and acceptance record. A change proposer would initially write a proposal only; actual change execution remains outside this release. Adding a type never increases the shared USD 20 ceiling automatically.

## 2. Observation contract

Collectors normalize existing metrics, logs and adapter frame exports into a versioned envelope:

| Group | Required content |
|---|---|
| Provenance | Schema version, run/segment/attempt ID, environment and image/config identities |
| Time | Collection time, observation time, window start/end, sequence number, staleness |
| Operations | State, heartbeat, elapsed active training, steps/sec, CPU/RAM/VRAM, checkpoint age, restart count |
| Training | Explicitly named scalar summaries with units, sample counts and missing-value flags |
| Evidence | Bounded redacted log excerpts, event counts, artifact hashes and authorized local references |
| Vision | Optional sampled game frames with timestamps, dimensions, privacy profile and input-cost bound |
| Integrity | Collector version, dropped-record counts, truncation flags and evidence availability |

Use per-environment adapters under the existing [ARC-005](arc-005-game-onboarding-standard.md) boundary. Operational fields are shared; game-specific observations must name their provenance. An absent metric is unknown, not zero. Distinguish a stale collector, a dead process and a learner making little progress.

Evaluator results are read only through an evaluation-side export. Generated observations never re-enter learning inputs, labels, rewards or actions. Every report distinguishes directly measured facts from model interpretation. A screenshot showing a tree or a high reward is not evidence that a skill was learned.

## 3. Token and cadence policy

Proposed defaults, to be calibrated with recorded fixtures before paid activation:

| Control | Starting value |
|---|---|
| Local metrics scrape / heartbeat | Every 30 seconds; no LLM call |
| Local deterministic aggregation | Five-minute windows, bounded ring buffer |
| Routine LLM summary | At most every 30 minutes during active training; skip unchanged healthy windows |
| Event-triggered summary | State transition or deduplicated anomaly; one per incident per 30-minute cooldown |
| Shared observer call limit | 72 attempts per UTC day across routine/event/final reports and retries |
| Concurrent paid requests | 1 initially |
| Total text input per request | At most 2,000 tokens including instructions, schema and prior compact state |
| Log input within that budget | At most 20 selected lines / 4 KiB, then token truncation |
| Prior compact state | At most 200 tokens; never append an unbounded conversation |
| Visible report target | At most 150 tokens, short structured fields |
| Total model output limit | Initially 2,000 tokens including reasoning; calibrate for incomplete responses |
| Vision | Disabled for routine summaries; at most two sanitized frames per incident and six vision requests/day |

All caps apply together. If input cannot be made safe and bounded, omit it and record the omission. Local aggregation performs deduplication, severity classification and numeric deltas before invoking the model. Avoid paying for an LLM to compress a large raw transcript into its own prompt. Fixed instructions can benefit from provider caching, but the budget never assumes a cache hit.

Continuous dashboards show local metrics and sampled frames independently of LLM cadence. Critical deterministic alerts are immediate and do not wait for the next model summary. When nothing is running and there is no incident, no observer Job or paid call is needed. A small trigger service remains alive to launch bounded Jobs; coalesce repeated events rather than creating one pod for every log line.

## 4. Model selection and report behavior

The owner preference is OpenAI `gpt-6-luna`, medium reasoning. Official documentation checked on 2026-10-28 lists this identifier, medium reasoning support and image input. Account access and the exact API behavior must be verified during implementation; this document does not infer access from a model name visible in a coding assistant. Proposed integration uses the Responses API with no tools and no persistent conversation. [GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).

Reports contain status, at most three evidence references, one short finding, uncertainty and an optional suggested human investigation. Schema validation rejects unexpected fields and executable payloads. Incomplete or malformed output is marked unavailable; no automatic retry merely to obtain prettier prose. Provider unavailability, budget denial or missing evidence falls back to the deterministic local summary. There is no automatic switch to a more expensive model.

Medium reasoning can consume tokens not visible in the report. The API output cap includes reasoning, so the 150-token visible target is not a billing cap. [OpenAI reasoning and cost controls](https://developers.openai.com/api/docs/guides/reasoning).

## 5. Credentials and accounting identity

Create one provider API key per AIOps type, not per pod or run. Store the observer key in the gateway's protected secret store; observer pods authenticate to the gateway using their own workload identity. The gateway derives agent type from that identity and selects its key. A caller-supplied type string cannot select another key. This preserves separate attribution without distributing provider keys to untrusted observers.

Maintain a non-secret credential ID and rotation history for each type. Usage records include type, run, incident, provider/model, request ID, price-table version, input/cached/output/reasoning counts, image cost, reserved cost, reconciled cost and status. Never record the key value. Reconcile provider usage with the local ledger through a separate administrative process; do not give the observer organization-administration credentials.

## 6. Enforcing the USD 20 monthly ceiling

All types share a single durable budget ledger with transactional reservations. Proposed month boundary is UTC. Initially allocate USD 16 to observer use, USD 2 unallocated for future types, and USD 2 held as a non-spendable uncertainty cushion. No future type is activated by this allocation. This deliberately targets less than USD 20 in ordinary API charges.

Before sending a request, authenticate its type, validate the model/settings, bound total billable input including images and cache writes, and reserve the maximum possible output charge using the enforced generation cap. Atomically admit only when settled charges plus outstanding reservations plus this request fit both the type allocation and the global spendable balance. Apply the daily call limit as a separate guard. Use integer monetary units with conservative rounding.

After success, reconcile the reservation to observed usage. On timeout or ambiguous provider acceptance, retain the worst-case reservation until reconciled; retries require a fresh reservation and count as new attempts. Do not retry a timed-out request automatically. Request IDs prevent duplicate local processing but are not assumed to guarantee provider-side billing idempotency.

The gateway is the only allowed external model egress path and holds all program provider keys. If the ledger, price table, input-cost bound or credential mapping is unavailable, deny paid calls while training and local monitoring continue. Restoring an older ledger cannot reset spent money: freeze calls until provider reconciliation establishes a safe balance. Rollover must account for in-flight requests and uncertain charges; conservatively reserve them against the new period until resolved.

Dashboard alerts or provider account budgets are not the program's enforcement mechanism. Under the controlled gateway path and verified pricing bounds, the reservation design limits authorized spend. It cannot guarantee an account invoice if keys are used outside the gateway, prices change undetected, charges arrive outside the modeled rules, or unrelated projects spend money. Disable calls on a pricing discrepancy; keep the cushion, rotate compromised keys, and reconcile before resuming. Clarify tax treatment before activation if the USD 20 must include it.

## 7. Cost estimate, not a spending entitlement

Official Standard pricing checked on 2026-10-28 lists USD 0.10 per million input tokens and USD 0.50 per million output tokens for `gpt-6-luna`. Reverify before activation; other service tiers, cache writes and regional processing can differ. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing).

For a conservative text-only illustration, 72 attempts/day for a 31-day month gives 2,232 attempts. At the proposed ceilings of 2,000 input and 2,000 total output tokens, uncached Standard text cost is:

```text
input:  2,232 × 2,000 × $0.10 / 1,000,000 = $0.4464
output: 2,232 × 2,000 × $0.50 / 1,000,000 = $2.2320
total text ceiling under these assumptions = $2.6784
```

This excludes image input, cache-write premiums, tax and pricing changes. The gateway reserves the actual applicable categories rather than relying on this estimate. Images require a tested model-specific token/cost bound; if none is available, keep external vision disabled. The cheap projected text cost is not a reason to loosen the input or call caps.

## 8. Quality, storage and failure visibility

Keep report provenance, bounded sanitized payload hashes, usage and budget denial reasons. Default report/ledger retention is in [OPS-010](../operations/ops-010-cluster-storage-and-recovery.md). Do not request chain-of-thought; store the concise report and evidence IDs only. Show observation-disabled, provider-failed and budget-exhausted states explicitly. An observer that is unavailable must never leave a stale green health indicator.
