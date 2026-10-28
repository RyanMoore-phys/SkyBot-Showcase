"""Adapter skeleton: the game interface a new game implements (WP-16, ARC-005).

Subclass `GameAdapterTemplate`, implement the `_engine_*` hooks (the ONLY
game-specific code), name the class in the game YAML:

    game:
      backend: plugin
      adapter: "my_pkg.my_game:MyGameAdapter"

`make_env` imports it lazily; no learning module is edited. This base class
owns everything that must be IDENTICAL across games so a game cannot get it
wrong: the fixed action-slot table with slot 0 as the no-op, the valid mask,
invalid-slot handling, the canonical observation, the reward of exactly 0.0,
the RED evaluator channel, life boundaries, and executed-action reporting for
both timing styles (contract.py states the rules).

The hooks (engine side). Nothing returned by them may be derived from state a
player cannot see EXCEPT `_engine_evaluator` (RED, never learned):

    _engine_reset(seed)            start a life
    _engine_advance(slot)          apply the slot's inputs; return a dict:
                                     events: list[str]  "died" | "exit" | "life_end"
                                       (any of them ends the life)
                                     slots:  realtime only, the slot held on each
                                       tic of the returned interval
                                     tics:   realtime only, tic count (default len(slots))
    _engine_frame()                uint8 (H, W, 3) screen
    _engine_audio()                None or (N, 2) stereo samples
    _engine_body()                 dict of BODY senses (health, ...); learned inputs
    _engine_evaluator()            RED dict for scoring/diagnostics only
    _engine_close()

STATUS: built, untested. The sensor-bus wiring below mirrors the ViZDoom
adapter and is exercised only when `sensors_cfg.enabled` is set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from developmental_ai.environments.held_tags import majority_slot
from developmental_ai.game_template.contract import TIMINGS


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """(H, W, 3) uint8 -> flat float32 CHW in [0, 1], box-sampled, numpy only."""
    f = np.asarray(frame, dtype=np.uint8)
    h, w = f.shape[0], f.shape[1]
    ys = np.minimum((np.arange(size) * h) // size, h - 1)
    xs = np.minimum((np.arange(size) * w) // size, w - 1)
    small = f[ys][:, xs]
    return (small.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(-1)


class GameAdapterTemplate(gym.Env):
    PROPRIO_DIM = 13          # must equal stream 0's width (the loop sizes from it)
    TIMING = "synchronous"    # overridden by game_cfg["timing"]
    GAME_NAME = "template"

    def __init__(self, game: Dict[str, Any], image_size: int = 128,
                 sensors_cfg: Optional[Dict] = None, stream_index: int = 0,
                 seed: int = 0, action_slots: Optional[int] = None):
        super().__init__()
        self.game_cfg = dict(game or {})
        self.image_size = int(image_size)
        self.stream_index = int(stream_index)
        self.seed_value = int(seed)
        self.timing = str(self.game_cfg.get("timing", self.TIMING))
        if self.timing not in TIMINGS:
            raise ValueError(f"game timing {self.timing!r} not in {TIMINGS}")
        self.controller: List[Dict[str, Any]] = [
            dict(c or {}) for c in (self.game_cfg.get("controller") or [])]
        if not self.controller or self.controller[0]:
            raise ValueError("controller slot 0 must be {} (the no-op) in every game")
        if action_slots is not None and len(self.controller) > int(action_slots):
            raise ValueError("controller has more slots than universal.action_slots")
        self.valid_mask = np.ones(len(self.controller), dtype=bool)   # override per game
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (3 * self.image_size * self.image_size,), np.float32)
        self.action_space = gym.spaces.Discrete(len(self.controller))
        audio = self.game_cfg.get("audio") or {}
        self.audio_enabled = bool(audio.get("enabled", False))
        self.sample_rate = int(audio.get("sample_rate", 22050))
        self.invalid_steps = 0
        self.steps = 0
        self._held = 0
        self._closed = False
        self._pending_restart = False
        self._last_obs = np.zeros(self.observation_space.shape, np.float32)
        self._sensor_bus = None
        self._last_frame = None
        self._prev_fovea = None
        self._fovea_size = int((sensors_cfg or {}).get("fovea_size", 32))
        if sensors_cfg and sensors_cfg.get("enabled") is not None:
            from developmental_ai.sensors import build_default_bus
            self._sensor_bus = build_default_bus(
                self.PROPRIO_DIM, lambda ctx: self._proprio(ctx["world"]),
                enabled=list(sensors_cfg.get("enabled") or []),
                fovea_size=self._fovea_size, move_scale=1.0)

    # ------------------------------------------------------------- hooks
    def _engine_reset(self, seed: Optional[int]) -> None:
        raise NotImplementedError

    def _engine_advance(self, slot: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _engine_frame(self) -> np.ndarray:
        raise NotImplementedError

    def _engine_audio(self) -> Optional[np.ndarray]:
        return None

    def _engine_body(self) -> Dict[str, float]:
        return {}

    def _engine_evaluator(self) -> Dict[str, Any]:
        return {}

    def _engine_close(self) -> None:
        pass

    def _proprio(self, world: Dict[str, Any]) -> np.ndarray:
        """Fixed-width own-body vector. Default: zeros (neutral). Override to
        fill from BODY senses only (never positions an engine would know)."""
        return np.zeros(self.PROPRIO_DIM, np.float32)

    # ------------------------------------------------------------ shared
    @property
    def held_action(self) -> int:
        return int(self._held)

    def _compose(self, decision_tics: int) -> tuple:
        frame = self._engine_frame()
        obs = resize_frame(frame, self.image_size)
        if self._sensor_bus is not None:
            from developmental_ai.sensors import crop_centre
            self._prev_fovea = getattr(self, "_cur_fovea", None)
            self._cur_fovea = crop_centre(frame, self._fovea_size)
        self._last_frame = frame
        world = dict(self._engine_body())
        world["decision_seconds"] = float(decision_tics) / float(
            self.game_cfg.get("tic_hz", 1.0))
        info: Dict[str, Any] = {"game": self.GAME_NAME, "world": world,
                                "decision_tics": int(decision_tics),
                                "gui_open": False, "env_restarted": False,
                                "rewind": False}
        info["proprio"] = np.asarray(self._proprio(world), np.float32)
        if self.audio_enabled:
            from developmental_ai.environments.universal import stereo_mel
            mel = stereo_mel(self._engine_audio(), self.sample_rate)
            if mel is not None:
                info["audio"] = mel
        if self._sensor_bus is not None:
            ctx = {"world": world, "info": info, "pov_native": frame,
                   "pov_small": obs, "prev_fovea": self._prev_fovea}
            try:
                info["sensors"] = self._sensor_bus.read_policy(ctx)
            except Exception:
                info["sensors"] = self._sensor_bus.neutral_policy()
            info["sensor_layout"] = self._sensor_bus.layout_hash()
        info["evaluator"] = dict(self._engine_evaluator())          # RED
        self._last_obs = obs
        return obs, info

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("adapter closed")
        self._engine_reset(self.seed_value if seed is None else int(seed))
        self._held = 0
        self._pending_restart = False
        obs, info = self._compose(1)
        if self.timing == "realtime":     # nothing has driven an interval yet
            info.update(executed_action=0, executed_purity=1.0, held_action=0)
        return obs, info

    def step(self, action):
        if self._closed:
            raise RuntimeError("adapter closed")
        if self._pending_restart:
            # a forced discontinuity: honoured as env_restarted (CLAUDE.md 4.1:
            # truncated alone would never reset the lifelong body)
            info = {"game": self.GAME_NAME, "rewind": True, "env_restarted": True,
                    "gui_open": False, "world": {}, "proprio": np.zeros(self.PROPRIO_DIM, np.float32)}
            if self.timing == "realtime":
                info.update(executed_action=0, executed_purity=1.0, held_action=0)
            self._held = 0
            return self._last_obs, 0.0, False, True, info
        a = int(action)
        if not 0 <= a < len(self.controller) or not bool(self.valid_mask[a]):
            self.invalid_steps += 1
            a = 0
        prev = self._held
        self._held = a
        out = self._engine_advance(a) or {}
        events = list(out.get("events") or [])
        self.steps += 1
        slots = list(out.get("slots") or [])
        tics = int(out.get("tics") or len(slots) or 1)
        terminated = bool(events)
        obs, info = self._compose(tics)
        if terminated:
            info["world"]["died"] = "died" in events
        if self.timing == "realtime":
            ex, pur = majority_slot(slots, prev)
            info["executed_action"] = int(ex)      # what drove THIS interval
            info["executed_purity"] = float(pur)
            info["held_action"] = int(a)           # what is running from now on
            info["interval_tics"] = int(len(slots))
        return obs, 0.0, terminated, False, info

    def force_restart(self) -> None:
        """Mark the stream discontinuous; the next step reports env_restarted."""
        self._pending_restart = True

    def close(self):
        if not self._closed:
            self._closed = True
            self._engine_close()
