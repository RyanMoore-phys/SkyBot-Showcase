"""RW-4: from episodes to semantic knowledge (GOV-010 section 7; GOV-012 entry
RW-4). BUILT, UNTESTED. All channels default OFF; OFF reproduces the earlier
behaviour (this module's wiring runs only when EPI -> SEM is on).

Blocks used (built, GOV-011): WP-18 `OutcomeMemory` (the episodic store; `loop._om`),
WP-19 representation auxiliary loss (contrastive separation, telemetry), the
replay buffer. The semantic store (`core/semantic_store.py`) is the consumer of
EPI -> SEM and is created by this relation (GOV-012 D-R11).

WIRED (data): MS-08 EPI -> SEM. Every `every` segments the episodic store is
summarised into semantic entries (clusters of similar episodes with their typical
learned outcome); the channel carries the number of kept entries (neutral 0).
The store is exposed as `loop._semantic`; nothing reads it yet.

OBSERVATION: SY-02 (rows written to the episodic store), SY-04 (contrastive
loss on replayed rows), CP-39 (near-duplicate pressure on the store: the mean
cosine similarity of the newest keys to their nearest older neighbour; the
contrastive keys are NOT used as store keys, D-R11), OU-01 (retrieval hit rate
of the store into the policy's knowledge vector), OU-02 (share of episodes
inside a kept semantic entry).

NOT BUILT: MS-04, MS-06, MS-07, SY-08, ST-07 (the knowledge base and the
knowledge graph are one block with no distinct flow; D-27 undecided), MS-09,
MS-10 (logs and provenance are telemetry or metadata, not learning paths),
CP-11, CP-12 (no chunking or tokenisation block), CP-16, CP-17 (RECON, model
editing, is absent; no distinct LTM to retrieval flow), ST-01 (no
retrieval-utility measurement exists), ST-14 (no hard-negative mining block),
SO-24 (prompt context, RES-009 D-12), PH-12 (no known mechanism).

Switches: relations.enabled; relations.EPI.SEM.enabled and the observation lines
(relations.HIP.EPI / HIP.PATSEP / PATSEP.EPI / EPI.KNOW / SEM.KNOW .enabled);
relations.semantic.{every 4, tau 0.8, min_support 5, max_entries 256,
max_rows 2000}.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from developmental_ai.core.relations import RelationRegistry
from developmental_ai.core.relations_connected import delta, wm

WIRED = ("rel.EPI.SEM",)
_S = "per segment"


def _om(loop):
    om = getattr(loop, "_om", None)
    return om if om else None


def _writes(loop) -> Optional[float]:
    om = _om(loop)
    return None if om is None else float(om.writes)


_writes_d = delta(_writes, "rw4_writes")
_hits_d = delta(lambda l: None if _om(l) is None else float(_om(l).hits),
                "rw4_hits")
_queries_d = delta(lambda l: None if _om(l) is None else float(_om(l).queries),
                   "rw4_queries")


def hit_rate(loop) -> Optional[float]:
    h, q = _hits_d(loop), _queries_d(loop)
    if h is None or q is None or q <= 0:
        return None
    return h / q


@torch.no_grad()
def duplicate_pressure(loop) -> Optional[float]:
    """Mean cosine similarity of the newest keys to their nearest OLDER key in
    the same ring (1.0 = the store is filling with near-duplicates)."""
    om = _om(loop)
    if om is None:
        return None
    best = None
    for r in getattr(om, "_rings", {}).values():
        if r.n < 64:
            continue
        keys = r.keys[:r.n]
        order = torch.argsort(r.wid[:r.n])                  # oldest .. newest
        newest = order[-32:]
        older = order[:-32]
        if older.numel() == 0:
            continue
        s = keys[newest] @ keys[older].T
        v = float(s.max(dim=1).values.mean())
        best = v if best is None else max(best, v)
    return best


def semantic_coverage(loop) -> Optional[float]:
    s = getattr(loop, "_semantic", None)
    return None if s is None or not s.rows else float(s.coverage)


def register(reg: RelationRegistry) -> None:
    g = "semantic"
    R = reg.register
    R("MS-08", "EPI", "SEM", "WP-18 outcome memory (episodes)",
      "semantic store (clusters of similar episodes)",
      "number of kept semantic entries", "entries", _S, "data", neutral=0.0,
      group=g)
    R("SY-02", "HIP", "EPI", "stepping loop (episodes written)",
      "WP-18 outcome memory", "episodes written to the store", "rows", _S,
      "data", probe=_writes_d, group=g)
    R("SY-04", "HIP", "PATSEP", "replayed rows", "WP-19 contrastive loss",
      "contrastive loss on replayed rows", "loss", _S, "data",
      probe=wm("repr_contrastive"), group=g)
    R("CP-39", "PATSEP", "EPI", "episodic keys", "episodic store",
      "mean similarity of newest keys to their nearest older key", "cosine",
      _S, "data", probe=duplicate_pressure, group=g)
    R("OU-01", "EPI", "KNOW", "episodic store", "policy knowledge vector",
      "retrieval hit rate this segment", "fraction", _S, "data",
      probe=hit_rate, group=g)
    R("OU-02", "SEM", "KNOW", "semantic store", "learned knowledge",
      "share of stored episodes inside a kept semantic entry", "fraction", _S,
      "data", probe=semantic_coverage, group=g)


def apply(loop, reg: RelationRegistry) -> None:
    c = dict(reg.cfg.get("semantic", {}) or {})
    om = _om(loop)
    st = reg.state.setdefault("sem", {"seg": 0, "store": None})
    st["seg"] += 1
    if om is None:
        reg.send("rel.EPI.SEM", 0.0)
        return
    if st["store"] is None:
        from developmental_ai.core.semantic_store import SemanticStore
        st["store"] = SemanticStore(
            float(c.get("tau", 0.8)), int(c.get("min_support", 5)),
            int(c.get("max_entries", 256)), int(c.get("max_rows", 2000)))
        loop._semantic = st["store"]
    if st["seg"] % max(1, int(c.get("every", 4))) == 0 or not st["store"].rows:
        st["store"].summarise(om)
    reg.send("rel.EPI.SEM", float(len(st["store"].entries)))


def release(loop, reg: RelationRegistry) -> None:
    if getattr(loop, "_semantic", None) is not None:
        loop._semantic = None
    reg.state.pop("sem", None)
