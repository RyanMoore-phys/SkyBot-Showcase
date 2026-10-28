"""RW-8 (GOV-010): learner-side representation and transfer measurements.

BUILT, UNTESTED. Observation channels only; they do not wire a transfer policy,
change an optimiser, or read evaluator state. The offline transfer probe remains
an evaluation tool, not a learning input (ADR-002).

These measurements show that WP-19's auxiliary representation losses are
present. They do not establish abstraction, cross-game reuse, or causal benefit.
The remaining RW-8 mechanisms are deferred in GOV-012 and RES-009.
"""
from __future__ import annotations

from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import wm

_S = "per segment"


def register(reg: RelationRegistry) -> None:
    R = reg.register
    g = "transfer"
    # Read-only metrics from learner-side world-model training. Loss values are
    # activity proxies, not evidence of the map's semantic interpretation.
    R("CP-25", "SCHEMA", "ABSTR", "WP-19 predictive representation loss",
      "hierarchical abstraction (proxy only)",
      "predictive representation loss", "loss", _S, "data",
      probe=wm("repr_predictive"), group=g)
    R("ST-30", "CONTR", "ABSTR", "WP-19 contrastive objective",
      "hierarchical abstraction (proxy only)",
      "contrastive representation loss", "loss", _S, "data",
      probe=wm("repr_contrastive"), group=g)
    R("CP-10", "GCL", "SCHEMA", "WP-19 auxiliary training signal",
      "representation learner", "representation auxiliary loss", "loss",
      _S, "data", probe=wm("repr_aux"), group=g)
