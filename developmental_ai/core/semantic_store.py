"""RW-4: semantic entries summarised from episodes (GOV-010 section 7, GOV-012).
BUILT, UNTESTED. Imported lazily by relations_semantic only when the line
EPI -> SEM is on.

A SEMANTIC ENTRY is a learned summary of a cluster of similar episodes held by
the WP-18 outcome memory: the cluster centre (a unit view-embedding key), how
many episodes support it, the mean and spread of their three LEARNED outcome
numbers (view change, world-model surprise, drive paid, each already squashed to
[0, 1)) and the share of each action taken from it. Nothing here is an engine
fact, an event name or a label: the keys are the agent's own frozen view
embeddings and the outcomes are its own learned quantities (ADR-002).

CLUSTERING. Greedy leader clustering on cosine similarity over a bounded
subsample of the stored episodes: the first unassigned row becomes a leader and
claims every unassigned row at least `tau` similar to it; repeat. `tau` is a
placeholder (0.8) for the value the separation test will set; entries with fewer
than `min_support` episodes are not kept (they stay in the episodic store only).
Entries are RECOMPUTED from the store at each call (a summary of what is
stored), so nothing latches and nothing is persisted.

NOT WIRED. The entries are not yet read by the policy or the knowledge graph
(GOV-012 RW-4 lists this as the open question of WP-18 / D-27); `lookup()` is the
retrieval a later consumer would use.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class SemanticStore:
    def __init__(self, tau: float = 0.8, min_support: int = 5,
                 max_entries: int = 256, max_rows: int = 2000, seed: int = 0):
        self.tau, self.min_support = float(tau), int(min_support)
        self.max_entries, self.max_rows = int(max_entries), int(max_rows)
        self._g = torch.Generator().manual_seed(int(seed))
        self.entries: List[Dict[str, Any]] = []
        self.rows = 0
        self.coverage = 0.0          # share of sampled rows inside a kept entry
        self.dispersion = 0.0        # mean within-entry outcome std
        self.new_entries = 0         # change in the number of kept entries

    @torch.no_grad()
    def summarise(self, om) -> int:
        """Rebuild the entries from outcome memory `om`; return how many."""
        rings = [r for r in getattr(om, "_rings", {}).values() if r.n > 0]
        if not rings:
            self.new_entries = -len(self.entries)
            self.entries, self.rows, self.coverage = [], 0, 0.0
            return 0
        keys = torch.cat([r.keys[:r.n] for r in rings]).float().cpu()
        acts = torch.cat([r.acts[:r.n] for r in rings]).cpu()
        outs = torch.cat([r.outs[:r.n] for r in rings]).float().cpu()
        n = keys.shape[0]
        if n > self.max_rows:
            idx = torch.randperm(n, generator=self._g)[:self.max_rows]
            keys, acts, outs, n = keys[idx], acts[idx], outs[idx], self.max_rows
        A = int(getattr(om, "A", 0))
        free = torch.ones(n, dtype=torch.bool)
        entries: List[Dict[str, Any]] = []
        covered = 0
        disp = 0.0
        for _ in range(self.max_entries):
            if not bool(free.any()):
                break
            lead = int(torch.nonzero(free)[0])
            sims = keys @ keys[lead]
            member = free & (sims >= self.tau)
            member[lead] = True
            free &= ~member
            m = int(member.sum())
            if m < self.min_support:
                continue
            ks = keys[member]
            cen = ks.mean(0)
            cen = cen / (cen.norm() + 1e-6)
            o = outs[member]
            hist = None
            if A > 0:
                a = acts[member]
                a = a[(a >= 0) & (a < A)]
                hist = (torch.bincount(a, minlength=A).float()
                        / max(1, a.numel())).tolist()
            entries.append({"centre": cen, "support": m,
                            "outcome_mean": o.mean(0).tolist(),
                            "outcome_std": o.std(0, unbiased=False).tolist(),
                            "action_share": hist})
            covered += m
            disp += float(o.std(0, unbiased=False).mean())
        self.new_entries = len(entries) - len(self.entries)
        self.entries = entries
        self.rows = n
        self.coverage = covered / max(1, n)
        self.dispersion = disp / max(1, len(entries))
        return len(entries)

    def lookup(self, key: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Nearest kept entry to a unit key, with its similarity, or None."""
        if not self.entries:
            return None
        k = key.detach().reshape(-1).float().cpu()
        k = k / (k.norm() + 1e-6)
        C = torch.stack([e["centre"] for e in self.entries])
        s = C @ k
        j = int(s.argmax())
        out = dict(self.entries[j])
        out["similarity"] = float(s[j])
        return out
