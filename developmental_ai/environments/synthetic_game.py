"""A tiny synthetic third game, used ONLY to prove the onboarding template (WP-16).

Not a research environment and not part of any run: nothing imports it except
the game YAML that names it (configs/games/templates/synthetic.yaml) and the
onboarding test. It is built on `GameAdapterTemplate`, so it exercises exactly
the path a real new game takes: files added, no learning module edited.

World: an 8x8 grid drawn as a 32x32 picture. Slots: 0 no-op, 1 up, 2 down,
3 left, 4 right, 5 touch. Touching a cell holding an item collects it (item
cells are chosen from the seed). A life ends after `life_steps` decisions.
Timing: `synchronous` (default) or `realtime` (an interval of `tics` tics in
which the first `lag_tics` still run the PREVIOUSLY held slot, imitating the
real-time action lag the executed-action contract exists for).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from developmental_ai.game_template.adapter_template import GameAdapterTemplate

GRID = 8
CELL = 4
_MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class SyntheticGameAdapter(GameAdapterTemplate):
    GAME_NAME = "synthetic"

    def __init__(self, game, image_size=128, sensors_cfg=None, stream_index=0,
                 seed=0, action_slots=None):
        super().__init__(game, image_size, sensors_cfg, stream_index, seed, action_slots)
        self.life_steps = int(self.game_cfg.get("life_steps", 200))
        self.tics = int(self.game_cfg.get("tics", 3))
        self.lag_tics = int(self.game_cfg.get("lag_tics", 1))
        self.lives = 0
        self.touches = 0
        self.collected = 0
        self._visited: set = set()
        self._rng = np.random.default_rng(seed)
        self._hold = 0
        self._beep = False

    # ------------------------------------------------------------- engine
    def _engine_reset(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(self.seed_value if seed is None else seed)
        self.pos = [GRID // 2, GRID // 2]
        cells = self._rng.permutation(GRID * GRID)[:6]
        self.items = {(int(c) % GRID, int(c) // GRID) for c in cells}
        self.age = 0
        self._hold = 0
        self._visited = {tuple(self.pos)}
        self.lives += 1

    def _apply(self, slot: int) -> None:
        self._beep = False
        if slot in _MOVES:
            dx, dy = _MOVES[slot]
            self.pos[0] = min(GRID - 1, max(0, self.pos[0] + dx))
            self.pos[1] = min(GRID - 1, max(0, self.pos[1] + dy))
            self._visited.add(tuple(self.pos))
        elif slot == 5:
            self.touches += 1
            self._beep = True
            if tuple(self.pos) in self.items:
                self.items.discard(tuple(self.pos))
                self.collected += 1

    def _engine_advance(self, slot: int) -> Dict[str, Any]:
        self.age += 1
        if self.timing == "realtime":
            slots: List[int] = []
            for t in range(self.tics):
                s = self._hold if t < self.lag_tics else slot
                self._apply(s)
                slots.append(s)
            self._hold = slot
            out: Dict[str, Any] = {"slots": slots, "tics": self.tics}
        else:
            self._apply(slot)
            out = {}
        out["events"] = ["life_end"] if self.age >= self.life_steps else []
        return out

    def _engine_frame(self) -> np.ndarray:
        img = np.zeros((GRID * CELL, GRID * CELL, 3), np.uint8)
        for (x, y) in self.items:
            img[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = (200, 160, 0)
        x, y = self.pos
        img[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = (0, 200, 80)
        return img

    def _engine_audio(self):
        n = 441
        if not self._beep:
            return np.zeros((n, 2), np.int16)
        t = np.arange(n) / 22050.0
        tone = (np.sin(2 * np.pi * 880.0 * t) * 8000).astype(np.int16)
        return np.stack([tone, tone], axis=1)

    def _engine_body(self) -> Dict[str, float]:
        # a body sense a player has: how much of this life is left
        return {"life_left": max(0.0, 1.0 - self.age / float(self.life_steps))}

    def _engine_evaluator(self) -> Dict[str, Any]:       # RED
        return {"cells_visited": len(self._visited), "touches": self.touches,
                "collected": self.collected, "lives": self.lives}
