# Two selectable world-model cores until a comparison decides

> **Document control:** [ADR-007](../README.md#decisions) · Decision · **State:** Accepted (owner, 2026-10-25); to be superseded by the verdict of the WP-20 comparison · **Custodian:** Repository maintainer · Revision history: Git.

**Decision source:** the owner, 2026-10-25: build a predictive-coding alternative to the RSSM, "include both the RSSM and predictive
learning methods in the repo until the tests come back so we can decide what to do from there", and make the pilot a gate.

## Context

The world model's latent core is the RSSM ([ARC-001](../architecture/arc-001-architecture.md)). [RES-010](../research/res-010-predictive-coding-core.md)
proposes a predictive-coding core in its place and asks whether it learns better under the program's standard
([ASS-007](../assurance/ass-007-adaptability-evaluation-protocol.md)). The question cannot be answered without both implementations in one tree, run on the same code snapshot.

## Decision

1. `world_model.core` selects the latent core: `rssm` (default, today's model) or `pc` (`PredictiveCodingCore`, whose own networks learn by local Hebbian-form updates; `world_model.pc_learning: backprop` is kept as an ablation). The default never changes without a superseding record.
2. **Both stay in the repository** until the comparison ([GOV-016](../governance/gov-016-predictive-coding-core-plan.md), [ASS-019](../assurance/ass-019-predictive-coding-core-comparison-protocol.md)) returns. Nothing is deleted or rewritten to make room for the other.
3. The PC core is **additive**: it subclasses the RSSM class and changes one method, so everything that reads the latent is untouched and a baseline run on the same snapshot is the unchanged RSSM.
4. After the verdict a superseding record chooses one of: keep both selectable; promote `pc`; retire `pc`. The tests and the evidence stay either way.

## Consequences and limits

- **Checkpoints are not interchangeable** between cores (the PC core's weights are buffers and it adds a precision buffer). The core and its learning mode are recorded in `world_model_arch.json`.
- **The PC core is local only in its latent core.** The encoder, GRU, decoder and heads still use backpropagation ([RES-010](../research/res-010-predictive-coding-core.md) section 2); this record does not authorise changing them.
- **Growth and autoscaler actuation do not support `pc`:** the growth tool refuses it by name. This is a limit of the comparison, not a finding.
- **Maintenance:** two cores to keep green until the verdict; the new suites (`_predictive_coding_smoke`, `_pc_pilot_gate_smoke`) guard the second.
- **No promotion by this record.** It authorises building and testing only; the default behaviour of every run is unchanged.
- Compute follows [ADR-005](adr-005-compute-pressure-order.md); both arms of the comparison run the same configuration.
