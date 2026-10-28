"""RW-9 Stage A: read-only observations of WP-12 peer-learning signals."""
from __future__ import annotations

from typing import Any, Optional

from developmental_ai.core.relations import RelationRegistry


def _metric(key: str):
    def probe(loop) -> Optional[float]:
        metrics: Any = getattr(loop, "_last_peer_metrics", None)
        if not isinstance(metrics, dict):
            return None
        value = metrics.get(key)
        return None if value is None else float(value)
    return probe


def register(reg: RelationRegistry) -> None:
    R = reg.register
    group = "social"
    scale = "per segment"
    R("SO-04", "PEER", "OBS", "WP-12 observed peer streams",
      "peer observation board", "peer stream count", "streams", scale,
      "data", probe=_metric("peer_streams"), group=group)
    R("SY-33", "MIRR", "OBS", "peer next-action predictor diagnostic",
      "WP-12 separate predictor", "action accuracy gain over repeat baseline",
      "accuracy delta", scale, "data",
      probe=_metric("action_prediction_gain"), group=group)
    R("SY-34", "MIRR", "IMIT", "peer next-view predictor diagnostic",
      "WP-12 separate predictor", "view cosine gain over no-change baseline",
      "cosine delta", scale, "data",
      probe=_metric("view_prediction_gain"), group=group)
