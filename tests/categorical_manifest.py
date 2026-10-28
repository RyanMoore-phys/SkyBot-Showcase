"""CS-0 inventory: the category manifests (GOV-013 sections 2-3; ASS-012 2).

DATA ONLY. One entry per functional group, mapping the group to its blocks (work packages), its
relation clusters (RW), the module files that implement them, the config switches that turn them on,
and an honest DISPOSITION for every member. `tests/_categorical_inventory_smoke.py` checks this file
against the code (files exist, switches exist in configs/universal.yaml, every wired channel is
claimed by exactly one category), so the manifest cannot drift from the tree.

Dispositions (GOV-013 3): implemented | partial | observation-only | deferred | excluded | existing.
A member that is not `implemented`/`existing` carries a NOTE saying what is missing; a scenario that
needs the missing part must be declared blocked in its test file, never passed with a zero-valued probe.

The groups overlay the owner's map; they are an engineering boundary, not an extra learner (ADR-001).
"""
from __future__ import annotations

# member: (id, disposition, [module files], [config switch paths], note)
# relation cluster: (id, disposition, relation module (name under developmental_ai/core) or None, note)
CATEGORIES = {
    "CAT-01": {
        "title": "Perception, representation and prediction",
        "blocks": [
            ("RSSM/encoder/replay", "existing", ["developmental_ai/world_model/rssm.py", "developmental_ai/world_model/replay_buffer.py"], [], ""),
            ("WP-19", "implemented", ["developmental_ai/world_model/representation.py"], ["representation.enabled"], "auxiliary loss only; transfer remains deferred (RES-007 H-09)"),
            ("WP-4 ensemble", "implemented", ["developmental_ai/core/uncertainty.py"], ["planning.ensemble.enabled"], "input to planning and calibration"),
        ],
        "relations": [
            ("RW-8", "partial", "relations_transfer", "measurements only; the remaining RW-8 mechanisms are deferred (GOV-012)"),
            ("RW-15", "observation-only", "relations_connected", "36 connected lines instrumented, nothing wired"),
        ],
    },
    "CAT-02": {
        "title": "Memory, attention and consolidation",
        "blocks": [
            ("WP-18", "implemented", ["developmental_ai/core/outcome_memory.py"], ["outcome_memory.enabled"], "widens the policy knowledge vector when ON"),
            ("WP-7", "implemented", ["developmental_ai/core/attention_memory.py"], ["attention_memory.enabled"], "widens the policy knowledge vector when ON"),
            ("WP-2", "implemented", ["developmental_ai/policy/consolidation.py"], ["consolidation.enabled"], "writes policy parameters; the only consolidation writer"),
        ],
        "relations": [
            ("RW-2", "implemented", "relations_attention", "SAL.ATT is the wired gate; PRED/CUR/BORE select its inputs"),
            ("RW-3", "implemented", "relations_consolidation", "four factors into the shared consolidation-strength product"),
            ("RW-4", "partial", "relations_semantic", "EPI.SEM builds semantic entries; NO policy retrieval consumer exists (GOV-013 3)"),
        ],
    },
    "CAT-03": {
        "title": "Planning, uncertainty and risk",
        "blocks": [
            ("WP-4", "partial", ["developmental_ai/core/planning.py", "developmental_ai/core/uncertainty.py"], ["planning.enabled"], "learning-phase planning only; acting-time search deferred (GOV-011 D-30). Action-predictability calibration is not competence calibration"),
            ("WP-11", "implemented", ["developmental_ai/core/risk.py"], ["risk.enabled"], "widens the policy knowledge vector when ON"),
        ],
        "relations": [
            ("RW-5", "implemented", "relations_executive", "PFC.WM working-memory gate"),
            ("RW-10", "excluded", None, "built by attribution to owning clusters; no RW-10 code (GOV-012 RW-10)"),
        ],
    },
    "CAT-04": {
        "title": "Motivation, exploration and skills",
        "blocks": [
            ("WP-5", "implemented", ["developmental_ai/curiosity/intrinsic_balance.py", "developmental_ai/curiosity/action_surprise.py"], ["curiosity.info_gain.weight"], "weights default 0 = off; the switch is a nonzero weight"),
            ("WP-1", "implemented", ["developmental_ai/skill_bank/effect_discovery.py"], ["goals.discovery.enabled"], "built, unmeasured: RES-005 Step 0 not done (GOV-011 D-14)"),
            ("WP-8", "implemented", ["developmental_ai/core/competence.py", "developmental_ai/core/coefficients.py"], ["competence.enabled"], "owner of mixer.intrinsic_gain"),
        ],
        "relations": [
            ("RW-1", "implemented", "relations_modulation", "owns the shared consolidation-strength mixer and three coefficients"),
            ("RW-7", "implemented", "relations_motivation", "ZPD.MOT and SAFE.CUR"),
            ("RW-11", "observation-only", "relations_skills", "read-only skill trend and option-binding channels"),
        ],
    },
    "CAT-05": {
        "title": "Curriculum and learning adaptation",
        "blocks": [
            ("WP-3", "implemented", ["developmental_ai/core/curriculum.py"], ["curriculum.enabled"], "context bandit over map x skill"),
            ("WP-9", "implemented", ["developmental_ai/policy/adaptive_optim.py"], ["adaptive_optim.enabled"], "writes learning-rate, entropy and return-scale coefficients"),
            ("WP-13", "implemented", ["developmental_ai/core/meta_adapt.py"], ["meta_adapt.enabled"], "online hyperparameter adaptation only; meta-learning proper deferred (RES-007 H-13)"),
        ],
        "relations": [
            ("RW-6", "implemented", "relations_strategy", "bandit over consolidation recipes"),
            ("RW-12", "excluded", None, "no module found in the tree (GOV-012); the inventory reports it rather than inventing one"),
        ],
    },
    "CAT-06": {
        "title": "Experience diversity, peers and transfer",
        "blocks": [
            ("WP-6", "implemented", ["developmental_ai/environments/data_diversity.py"], [], "game-YAML keys (map_assignment, held_out_maps), absent by default"),
            ("WP-12", "partial", ["developmental_ai/core/peer_observation.py"], ["peer_observation.enabled"], "peer auxiliary training does not train the shared encoder (GOV-013 3)"),
            ("WP-16", "implemented", ["developmental_ai/game_template/contract.py"], [], "game-onboarding template"),
            ("WP-19", "implemented", ["developmental_ai/world_model/representation.py"], ["representation.enabled"], "shared with CAT-01; one result, not counted twice"),
        ],
        "relations": [
            ("RW-8", "partial", "relations_transfer", "shared with CAT-01"),
            ("RW-9", "observation-only", "relations_peers", "read-only observations of WP-12 signals"),
        ],
    },
    "CAT-07": {
        "title": "Runtime, resources and evaluation isolation",
        "blocks": [
            ("WP-10", "partial", ["developmental_ai/infra/scale_order.py", "developmental_ai/world_model/growth_optim.py", "developmental_ai/infra/autoscaler.py"], ["infra.autoscaler.adr005.enabled"], "long-run protocol deferred (RES-007 H-17)"),
            ("WP-14", "implemented", ["developmental_ai/evaluator/protocol.py", "developmental_ai/evaluator/runner.py", "developmental_ai/evaluator/scoring.py"], [], "evaluator package: must stay outside every learner entry point (ADR-002)"),
            ("WP-17", "implemented", ["tools/campaign.py", "tools/eval_runner.py"], [], "run continuity and tooling"),
        ],
        "relations": [
            ("RW-14", "observation-only", "relations_resources", "read-only configured context capacity"),
            ("RW-15", "observation-only", "relations_connected", "shared with CAT-01"),
            ("RW-13", "excluded", None, "no module found in the tree (GOV-012)"),
        ],
    },
}

# Boundary inventory (ASS-012 4): ids, the groups joined, and the real code path each end uses.
BOUNDARIES = {
    "BND-01": ("CAT-01", "CAT-03", "ensemble/calibration evidence -> planner arbiter"),
    "BND-02": ("CAT-03", "CAT-04", "calibration validity -> modulation (explore / consolidate / attend coefficients)"),
    "BND-03": ("CAT-02", "CAT-03", "retrieval and attention features at policy/planner entry (declared zero-knowledge path)"),
    "BND-04": ("CAT-04", "CAT-05", "competence, curriculum and adaptive optimisation: one owner per coefficient"),
    "BND-05": ("CAT-06", "CAT-01", "peer/diversity provenance through replay and memory; no held-out leakage"),
    "BND-06": ("CAT-02", "CAT-05", "consolidation, replay selection and optimiser update order"),
    "BND-07": ("CAT-07", "ALL", "checkpoint/restart, instrumentation and resource pressure vs each group's state"),
}

DISPOSITIONS = ("implemented", "partial", "observation-only", "deferred", "excluded", "existing")
