# legacy/

Files kept for history only. **Nothing here is a current instruction, configuration or procedure.** Do not
copy a command, path, host name, user name, address or key from this folder to reach or run anything: the
only source for host access is [OPS-008](../docs/operations/ops-008-host-access.md).

Moved on 2026-10-22 (owner instruction, after the AB-00 interim analysis) because each file is related to
SkyBot's past but not to the current program (the universal brain, Doom, the adaptability evaluation).
Nothing in the repository references them (checked by name and path before moving). Restore one with a plain
`mv` back to its original path (given below); nothing was edited.

| Now | Was | Why it is legacy |
|---|---|---|
| `assets/OLD1.png`, `OLD2.png`, two `Screenshot 2026-06-03 ...png` | `assets/` | the first design sketches (Gymnasium, Stable-Baselines3, DreamerV3, Neo4j, PyG, a Hugging Face skill bank); the current map is `assets/ai_learning.mmd` |
| `configs/acrobot.yaml`, `mountaincar.yaml`, `mujoco_halfcheetah.yaml`, `pixel_cartpole.yaml` | `configs/` | classic-control benchmark configs of the first agent; no code, test or script loads them |
| `configs/_archive/minecraft*.yaml` (6) | `configs/_archive/` | the Minecraft-only legacy brain's configs (ADR-004 retires that brain) |
| `scripts/launch_lifelong.sh`, `launch_curiosity_run.sh`, `deploy_arch_v3.sh`, `host_continuous.py` | `scripts/` | launchers and deploy helpers of the legacy brain |
| `scripts/merge_duplicate_skills.py`, `probe_gui_capability.py`, `probe_inventory_trap.py`, `backfill_action_dim.py` | `scripts/` | one-off Minecraft skill-bank and GUI-trap diagnostics and a migration of old checkpoints |
| `tools/visualize_world_model.py`, `viewer/term_viewer.py` | `tools/`, repo root | an early world-model viewer and a terminal viewer; the live viewer is `viewer/brain_viewer.html` |
| `docs/legacy/decisions/adr-003-...md` | `docs/decisions/` | superseded by ADR-004 |
| `docs/legacy/assurance/ass-001-...md` | `docs/assurance/` | the continuous-control ablation of the classic-control era |

Kept in place on purpose (they look old but are live): `experiments/foundation_ab/` (imported by
`developmental_ai/foundation` and four active tests), `configs/lunar_lander.yaml` (read by
`developmental_ai/core/evaluation.py`), `configs/crafter.yaml`, `minigrid_doorkey.yaml`, `craftax.yaml`
(games the universal brain can play and that tests use), `scripts/rcon.py` (the game-server guide uses it),
`scripts/retire_fossil_skills.py` (cited by live code), and the Grafana dashboards.
