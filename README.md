# SkyBot — a developmental learning agent, and the test discipline around it

SkyBot is a research codebase for an agent that **learns from pixels and buttons alone**: no demonstrations, no scripted skills, no
hand-written goals. What counts as meaningful, which skills exist and what the checkpoints are must all be *earned from experience*,
never declared to the agent. One universal brain plays every game; Doom is the development proxy and Minecraft is the later target.

The success measure is **adaptability to unfamiliar environments**, judged at evaluation-only checkpoints that are never rewards. A
behavioural claim needs five seeds of one hour, positive on all five. Null and negative results are recorded.

> **Status: active research, nothing here is a behavioural claim.** The test campaign is still running on private hardware, and this
> repository is a read-only view of the code and the records. No change has yet been shown to make the agent learn faster.

## What to look at first

| If you care about | Read |
|---|---|
| The research program and its rules | [GOV-008](docs/governance/gov-008-human-like-learning-motivation.md), [ASS-007](docs/assurance/ass-007-adaptability-evaluation-protocol.md) |
| How results are recorded, including failures | [GOV-016 §11](docs/governance/gov-016-predictive-coding-core-plan.md) (two failed pilots, predictions written first, corrections), [GOV-015](docs/governance/gov-015-corrective-plan-categorical-findings.md) |
| The testing system | [ASS-013](docs/assurance/ass-013-unattended-test-campaign.md) (an unattended, resumable campaign driver), [ASS-014](docs/assurance/ass-014-categorical-harness-first-run-findings.md) (a harness that must pass as built and detect a deliberately broken version) |
| Incidents and what they taught | [ASS-017](docs/assurance/ass-017-host-test-stages-and-queue-incidents.md) |
| Architecture | [ARC-001](docs/architecture/arc-001-architecture.md), [ARC-004](docs/architecture/arc-004-universal-games.md) |
| A design decision with alternatives | [ADR-007](docs/decisions/adr-007-two-selectable-world-model-cores.md), [RES-010](docs/research/res-010-predictive-coding-core.md) |

## An example of the method

A second world-model core, with a predictive-coding settle and Hebbian (local) learning, was built beside the standard RSSM
(`developmental_ai/world_model/predictive_coding.py`). It was compared on a pre-registered gate:

- Two pilot attempts **failed** the gate on reconstruction quality and acting cost. The records keep both failures.
- Predictions were written before each run; several were wrong, and the documents say which.
- A claim that slow learning was the problem was tested and **rejected**: a faster rate collapsed the latent.
- Another claim, that a test passed, was **corrected** after its own record showed a failure.

That habit (write the prediction, run it, keep the failure, correct the record) is the point of this repository as much as the model.

## Layout

| Path | Contents |
|---|---|
| `developmental_ai/` | the agent: world model, policy, curiosity, evaluator, the learning loop |
| `configs/` | the universal configuration and per-game files |
| `tests/` | contract and categorical suites; each names the incident or design claim it protects |
| `tools/` | the campaign driver, analysis and report tools, documentation checks |
| `docs/` | governance, architecture, decisions, assurance protocols, research notes |
| `experiments/` | recorded experiment results |
| `scripts/run.py` | the training entry point |

## What is not here

Training runs on private hardware that is not part of this repository: its host, deployment and supervision scripts, the runner
and monitoring containers, game ROMs and all brain state and logs are left out on purpose. Documents here may refer to them
(for example OPS-005 and OPS-008); the values they describe are not published. The tests are designed to run on that host, so
this repository's CI checks only syntax and documentation integrity.

## License

MIT, see [LICENSE](LICENSE).
