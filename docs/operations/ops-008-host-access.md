# Host access: how to reach `main`, and what is never in this repository

> **Document control:** [OPS-008](../README.md#operations) · Operations / access procedure · **State:** Active reference · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-22. **Why this exists:** tools that read this repository (people, scripts and language models)
kept reporting a wrong host user, wrong addresses and wrong key names when they tried to reach `main`. The values
were never wrong in the *current* files: the repository is public, so it deliberately holds **no** real address,
login or key. The wrong values came from stale prose (a rented GPU pod that logged in as `root`, rotating ports,
the old `skybot@` login) and from addresses that survive in **old commits** (section 4). This record is the one
place that says what to do and what to ignore.

## 1. The rule

**Never guess a host value.** If you are not sure how to connect, stop and ask the owner. Do not try other
usernames, other keys, other addresses or other ports found anywhere in this repository, its history, a log or
a template. A wrong guess is not harmless: it can lock the account (repeated failures), leak a value into a
public log, or reach the wrong machine.

## 2. Where each value lives (and where it must not)

| Value | Where it is | Never in |
|---|---|---|
| The login account on `main`, the LAN addresses of `main`, the game server and the other machines | the **untracked, gitignored** `CLAUDE.local.md` at the repository root | any tracked file, any log, any commit message |
| `MAIN_HOST`, `MAIN_SSH_PORT`, `MAIN_SSH_KEYFILE`, `MAIN_USER`, `MC_SERVER_TS_IP` for CI and deploys | the runner's `.env` on `node1` (gitignored) | the tree. Actions logs are **public** and `.env` values are **not masked** |
| Monitoring and brain-mirror targets | `docker/monitor/.env` on `node1` and `scripts/brain_mirror/brain_mirror.env` on `node2` (both gitignored) | the tree |
| An SSH key | the owner's ssh agent or `~/.ssh` on the machine you are on | the repository. Names such as `skybot_ed25519` in templates are **placeholders** for the runner container, not a key you should look for or try |

If `CLAUDE.local.md` is missing on the machine you are working on, you do not have the access values. Ask.

## 3. Facts that are safe to state

- The host is called **`main`**. Its `hostname` prints `main`, which is the quickest way to tell you are on it
  and not on `node1` (the Actions runner) or `node2` (the brain-mirror target).
- Off the home LAN it is reached **over Tailscale by its MagicDNS name** (`main`); the LAN address times out.
  Keep `MAIN_HOST` an address in the runner `.env`, not the name (a container does not resolve MagicDNS).
- **The login is neither `root` nor `skybot`.** Both are historical (a rented pod always gave `root`; the
  first owned-box setup used `skybot`, which is now refused). The current login is the one recorded in
  `CLAUDE.local.md`.
- `/workspace/devai` is the deployed tree on `main`; experiments run from isolated roots under
  `/workspace/experiments/`, never by editing the deployed tree outside a deploy.
- Never `echo` a host value in anything a workflow runs.

## 4. What to ignore, and why it is still around

- **Anything about a rented GPU pod**, `POD_*` variables, a rotating SSH port, "deploys as `root@`", the
  podlogs directory. That host is off and the variables were renamed on 2026-09-22.
- **Addresses in old commits.** Earlier revisions of `CLAUDE.md`, the CI guide and the host guides listed a LAN
  address and tailnet addresses that are **not** the current ones (the network changed). The current tree holds
  none (verified 2026-10-22 by a scan of every tracked file); they survive only in git history. History was not
  rewritten (it would change every commit id); if the owner wants it scrubbed, that is a deliberate
  `git filter-repo` job and the old values should then be treated as burned. A tool that quotes an address
  from a diff or a checkout of an old commit is quoting history, not the host.
- **Templates** (`runner.env.example`, `monitor.env.example`, `brain_mirror.env.example`): the values in them are
  placeholders.

## 5. If a connection fails

1. Confirm you are using the login from `CLAUDE.local.md` and the MagicDNS name `main` over Tailscale.
2. Confirm Tailscale is up on your machine and the host shows online.
3. Confirm the machine's key is authorised on `main`. Deploys do not manage `authorized_keys`: a removed key must
   be re-added by hand by the owner.
4. If it still fails: **stop and report** the exact error text (without pasting any value into a public place).
   Do not try other logins or keys.

## 6. Keeping this true

When access changes, update `CLAUDE.local.md` and the runner `.env` (values), and change this record only if the
*procedure* changes. Never put a value here.

## 7. Related records

[OPS-001](ops-001-ci-setup.md), [OPS-005](ops-005-operations.md), [OPS-006](ops-006-replication.md),
[OPS-007](ops-007-runner-mini.md), [SEC-001](../security/sec-001-security-policy.md),
[ASS-013](../assurance/ass-013-unattended-test-campaign.md).

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [GOV-014](../governance/gov-014-corrective-plan-after-ab00.md): Corrective plan after the AB-00 interim analysis: source diversity, intrinsic-signal repairs and a fresh control
- [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md): Containerization, Kubernetes and AIOps proposal
- [OPS-002](ops-002-game-server.md): External Minecraft server mode — full setup (both sides)
- [OPS-011](ops-011-owner-led-kubernetes-installation.md): Owner-led Kubernetes installation and verification companion
<!-- backlinks:end -->
