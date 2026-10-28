# Retire the legacy brain on this branch

> **Document control:** ADR-004 · Accepted · Custodian: Repository maintainer

**Decision source:** the owner, 2026-10-20, in the working session that opened
[GOV-008](../governance/gov-008-human-like-learning-motivation.md). This record
transcribes that instruction. **It supersedes [ADR-003](../legacy/decisions/adr-003-preserve-legacy-brain-isolation.md)
on this branch only**; ADR-003 stays as the historical record of the earlier
decision.

## Context

ADR-003 kept the Minecraft-only ("legacy") brain isolated and unchanged while
the universal brain was introduced. The owner has since stated that this branch
is new, that copies of the legacy brain are stored elsewhere, and that nothing
legacy should remain on this branch. The universal brain, one shared learner for
every game, is the only target.

## Decision

1. This branch supports the **universal brain only**.
2. The legacy brain, its state, its launch path, its switches and every
   requirement to keep its behaviour unchanged are **out of scope** and may be
   changed or removed.
3. "Legacy" means the Minecraft-only brain. **Minecraft as a game is not
   retired**: it remains a valid game for the universal brain, and its return to
   the universal brain is deferred by the owner until after Doom.
4. A change no longer has to leave a legacy configuration identical. A/B
   comparison switches are still used so a baseline arm can be reproduced (see
   [GOV-008](../governance/gov-008-human-like-learning-motivation.md) section 4).

## Consequences and limits

- Several existing documents and the root `CLAUDE.md` describe the legacy brain
  as live. They remain accurate as history; the root `CLAUDE.md` carries a
  branch override that states which parts no longer apply.
- Parts of the current universal configuration are composed on top of the
  legacy configuration file. Removing the legacy brain therefore needs a
  deliberate step (it is a work package in GOV-008), not a delete. This record
  authorises that work; it does not do it.
- Older evidence about the legacy brain (for example the scoreboard in the root
  `CLAUDE.md`) is not invalidated, only no longer a compatibility constraint.
- The owner holds copies of the legacy brain outside this repository; this
  record is not a backup procedure.

## Alternatives recorded

Keeping the legacy brain as a comparison arm or compatibility target was
rejected by the owner for this branch.

## Amendment 1 (2026-10-20, same day, clarifying scope)

Source: the owner's second round of answers in the same session.

**Rule for Minecraft-only code.** Remove all Minecraft-only code from this
branch **unless it is literally the method by which SkyBot interacts with
Minecraft**. What is kept is the *game interface*: the game-dependent scripts
and configuration files that every game needs in order to be run by SkyBot
(for example a per-game adapter, launcher, install step and game
configuration), each of the same shape as the equivalent for Doom and any other
game.

**Judgement for everything else.** For any Minecraft-only file that is not the
game interface, the assistant decides, one item at a time, by this test:

1. *Can the idea be made universal?* If it can be re-derived from what any
   player has (pixels, sound, own-body senses, buttons) and run identically in
   Doom and every other game, it is **translated**, not kept as Minecraft code.
2. *If it cannot,* it is **removed**. Example given by the owner: Minecraft-only
   reward files.

Each decision (keep as interface, translate, remove) is recorded with its reason
in the implementation plan and in the change log of
[GOV-008](../governance/gov-008-human-like-learning-motivation.md), so the owner
can overrule it. The owner holds copies of the legacy brain, so a removal is
recoverable from outside this repository.

This rule is the scope of work package WP-15. It is an instruction; nothing has
been removed yet.

## Amendment 2 (2026-10-21, response to the branch audit)

Source: [ASS-009](../assurance/ass-009-audit-response-adjustments-branch.md), finding
3.11. The assistant's judgement; revisable by the owner.

**A reproducible test for "universal".** An item is classed by three checks, in order:

1. **Interface check.** Is it the method by which SkyBot runs or reads that game
   (adapter, launcher, install step, game configuration, checkpoint detector)? If so
   it is **kept** as game interface and must have the same shape as the equivalent
   for the other games.
2. **Input check.** Do all its inputs come from the shared contract (pixels, sound,
   body senses from the shared sensor bus, buttons)? A reference to a game-specific
   field or concept name (a block, a log, a crafting recipe) fails.
3. **Run check.** Does it run unchanged in Doom and in the synthetic game of WP-16?

An item passing 2 and 3 is **universal** (kept, or **translated** if it needed
rewriting). An item failing either is **removed**.

**WP-15 is split because not every removal is behaviour-neutral.**

- **WP-15a, legacy path removal** (launch and state paths, compatibility logic,
  legacy configuration inheritance): expected to change no behaviour. Evidence:
  composed-configuration diff for Doom; a parity mock run inside the noise floor.
- **WP-15b, removal or translation of Minecraft-only learning terms** (reward,
  shaping, diagnostics that act on learning): first measure each term's income share
  in Doom runs. A term at essentially zero is removed under the parity checks. A term
  that pays in Doom is a **behaviour change**: it is treated as a block change with a
  T1 test, not as a refactor.

**Attribution.** Removals are made in small steps; after each, the unit and contract
tiers, the composed-configuration diff and a per-term income comparison are run, so a
difference can be traced to the step that caused it.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [ARC-005](../architecture/arc-005-game-onboarding-standard.md): Game onboarding standard
- [GOV-009](../governance/gov-009-implementation-plan.md): Implementation plan: human-like learning program
- [GOV-011](../governance/gov-011-execution-record-gov-009.md): Execution record: block implementation plan (GOV-009)
- [GOV-015](../governance/gov-015-corrective-plan-categorical-findings.md): Corrective plan for the categorical-harness findings and the host contract-tier failures
- [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md): Containerization, Kubernetes and AIOps proposal
<!-- backlinks:end -->
