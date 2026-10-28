"""Episodic novelty: how unlike everything seen THIS LIFE is what I see now.

Born 2026-10-17 — the player-legal replacement for `coverage`, chosen by the
user from a graded comparison on real Doom play (docs/research/res-004-rssm-size-experiment.md,
"Replacing coverage"). Coverage paid for reaching new 4-block cells of the
ENGINE's player position: 85 % of all Doom income, farmed by a strafe loop,
and a value no player can see. This term sees only what the agent sees.

The form is Never Give Up's episodic bonus (Badia et al., 2020): keep the
embeddings of this life's views; the bonus is high when the current view is
far from its k nearest neighbours, and decays as similar views accumulate.

    d_i   = squared distances to the k nearest remembered views
    d_m   = running mean of those distances (scale-free)
    K_i   = eps / (d_i / d_m + eps)          similarity kernel
    r     = 1 / sqrt(sum_i K_i + c)          in (0, 1/sqrt(c)]
    bonus = r * sqrt(c)                      in (0, 1]

Graded against the engine's true position (answer key only, never reward),
sticky-random Doom play, 12,000 frames, encoder from a world model trained
with the posterior-collapse fix: it picked out real new places with AUC
0.60, paid 0.15x of exploring while standing still, 0.98x while spinning
(new views are new), and 0.375x on the strafe loop that farmed coverage —
the least farmable on that loop of every player-legal option measured
(view-novelty per visit: 0.66 / 0.12 / 0.92 / 0.59; pixel-hash 0.70 / 0.22 /
1.01 / 0.77). A first run graded with a COLLAPSED encoder (the deployed old
code was imported by mistake) gave 0.64 / 0.18 / 1.01 / 0.40 — same ranking.

ONE LEARNING PROCESS (CLAUDE.md §1): the same computation on every stream
of every game; the input is the agent's own frozen view encoder
(`DevelopmentalAI._view_key`), nothing game-specific. The memory belongs to a
LIFE: it is cleared at each death / reset / rewind of that stream, so a new
life can be curious again about places it has seen before.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class EpisodicNovelty:
    def __init__(self, k: int = 10, capacity: int = 2000, eps: float = 1e-3,
                 c: float = 1e-3, dm_decay: float = 0.99,
                 cluster_distance: float = 0.0):
        # cluster_distance (NGU's xi, WP-5 arm; 0 = off = the original form):
        # neighbour distances, in units of their running mean, below this are
        # treated as zero, so near-identical views stop paying.
        self.cluster_distance = float(cluster_distance)
        self.k, self.capacity = int(k), int(capacity)
        self.eps, self.c, self.dm_decay = float(eps), float(c), float(dm_decay)
        self._mem: Dict[int, torch.Tensor] = {}      # stream -> (n, D), normalised
        self._dm: Dict[int, float] = {}

    def reset(self, stream: int) -> None:
        """A new life: forget this stream's views (the running scale is kept)."""
        self._mem.pop(int(stream), None)

    def size(self, stream: int) -> int:
        m = self._mem.get(int(stream))
        return 0 if m is None else int(m.shape[0])

    @torch.no_grad()
    def bonus(self, stream: int, emb: Optional[torch.Tensor]) -> float:
        """Bonus in (0, 1] for this view, then remember it. 0.0 until k views
        have been seen this life (nothing to compare against yet), and for a
        missing embedding (a missing sense reads neutral, never fatal)."""
        if emb is None:
            return 0.0
        e = emb.reshape(1, -1).float()
        e = e / (e.norm(dim=1, keepdim=True) + 1e-6)
        s = int(stream)
        mem = self._mem.get(s)
        out = 0.0
        if mem is not None and mem.shape[0] >= self.k:
            d = ((mem - e) ** 2).sum(1)
            kn = torch.topk(d, self.k, largest=False).values
            km = float(kn.mean())
            dm = self._dm.get(s)
            dm = km if dm is None else self.dm_decay * dm + (1.0 - self.dm_decay) * km
            self._dm[s] = dm
            dn = kn / (dm + 1e-8)
            if self.cluster_distance > 0.0:
                dn = (dn - self.cluster_distance).clamp(min=0.0)
            K = self.eps / (dn + self.eps)
            out = float(self.c ** 0.5 / torch.sqrt(K.sum() + self.c))
        if mem is None:
            mem = e
        else:
            mem = torch.cat([mem, e.to(mem.device)], 0)
            if mem.shape[0] > self.capacity:
                mem = mem[-self.capacity:]
        self._mem[s] = mem
        return out
