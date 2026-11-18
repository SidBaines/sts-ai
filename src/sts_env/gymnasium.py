"""Optional fixed-space adapter. Import requires the [gym] extra."""
from __future__ import annotations

from copy import deepcopy
import math
import operator
from typing import Any

import gymnasium as gym
import numpy as np

from sts_env.env import SlayTheSpireEnv, SimulatorError


class GymnasiumEnv(gym.Env):
    """Numeric public-state baseline with a fixed menu-index action space.

    This is an intentionally small feature encoder, not a rules-complete state.
    Use info['observation'] to build richer card/enemy/action encoders in an
    experiment wrapper. Action meanings vary with the legal menu each step;
    algorithms should respect observation['action_mask']. Padded, masked indices
    are explicit no-op transitions with configurable reward (zero by default).
    The decision limit counts all attempts, including invalid masked indices.
    """

    metadata = {"render_modes": ["ansi"]}
    FEATURE_NAMES = ("hp", "max_hp", "gold", "act", "floor", "combat",
                     "turn", "energy", "block", "enemy_count", "enemy_hp",
                     "hand_count", "draw_count", "discard_count", "exhaust_count", "ascension")

    def __init__(self, *, max_actions: int = 2048, render_mode: str | None = None,
                 invalid_action_reward: float = 0.0, **kwargs: Any) -> None:
        super().__init__()
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        if render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be None or ansi")
        if not math.isfinite(invalid_action_reward):
            raise ValueError("invalid_action_reward must be finite")
        self.invalid_action_reward = float(invalid_action_reward)
        self._attempts = 0
        self._finished = False
        self._raw_observation: dict[str, Any] | None = None
        self._raw_info: dict[str, Any] | None = None
        self.render_mode = render_mode
        self.max_actions = max_actions
        self.env = SlayTheSpireEnv(**kwargs)
        self.action_space = gym.spaces.Discrete(max_actions)
        self.observation_space = gym.spaces.Dict({
            "features": gym.spaces.Box(-np.inf, np.inf, (len(self.FEATURE_NAMES),), dtype=np.float64),
            "action_mask": gym.spaces.MultiBinary(max_actions),
        })

    def _encode(self, observation: dict[str, Any]) -> dict[str, Any]:
        actions = observation["legal_actions"]
        if len(actions) > self.max_actions:
            raise SimulatorError(f"Legal menu has {len(actions)} actions, exceeds max_actions={self.max_actions}; increase max_actions")
        state = observation["state"]
        combat = state.get("combat", {})
        cards = combat.get("cards", {})
        enemies = [enemy for enemy in combat.get("enemies", []) if enemy.get("alive", True)]
        def pile_count(name: str) -> int:
            pile = cards.get(name, [])
            if isinstance(pile, list):
                return sum(int(card.get("count", 1)) for card in pile)
            return int(pile) if isinstance(pile, int) else 0
        features = np.asarray([
            combat.get("player_cur_hp", state.get("cur_hp", 0)),
            combat.get("player_max_hp", state.get("max_hp", 0)), state.get("gold", 0),
            state.get("act", 0), state.get("floor", 0), int(observation["phase"] == "combat"),
            combat.get("turn", 0), combat.get("player_energy", 0), combat.get("player_block", 0),
            len(enemies), sum(enemy.get("hp", enemy.get("cur_hp", 0)) for enemy in enemies),
            pile_count("hand"), pile_count("draw"), pile_count("discard"), pile_count("exhaust"),
            state.get("ascension", 0),
        ], dtype=np.float64)
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mask[:len(actions)] = 1
        return {"features": features, "action_mask": mask}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if options:
            raise ValueError("Reset options are unsupported; configure via constructor")
        world_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63-1))
        observation, info = self.env.reset(seed=world_seed)
        encoded = self._encode(observation)
        self._attempts = 0
        self._finished = bool(info["terminated"] or info["truncated"])
        self._raw_observation, self._raw_info = deepcopy(observation), deepcopy(info)
        return encoded, {**info, "observation": observation, "action_attempts": 0,
                         "invalid_action": False}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._raw_observation is None:
            raise RuntimeError("Call reset before step")
        if self._finished:
            raise RuntimeError("Episode ended; call reset before step")
        if isinstance(action, (bool, np.bool_)):
            raise ValueError("Action must be an integer in the fixed action space")
        try:
            index = operator.index(action)
        except TypeError as exc:
            raise ValueError("Action must be an integer in the fixed action space") from exc
        if not self.action_space.contains(index):
            raise ValueError("Action is outside the fixed action space")
        invalid = index >= len(self._raw_observation["legal_actions"])
        if invalid:
            observation = deepcopy(self._raw_observation)
            info = deepcopy(self._raw_info)
            reward, terminated, truncated = self.invalid_action_reward, False, False
            # A prior valid action's execution identity does not describe a no-op.
            info.pop("executed_action", None)
        else:
            observation, reward, terminated, truncated, info = self.env.step(index)
        self._attempts += 1
        if self._attempts >= self.env.config["max_steps"] and not terminated:
            truncated = True
            info["termination_reason"] = "max_steps"
        self._finished = terminated or truncated
        info.update(action_attempts=self._attempts, invalid_action=invalid,
                    terminated=terminated, truncated=truncated)
        self._raw_observation, self._raw_info = deepcopy(observation), deepcopy(info)
        encoded = self._encode(observation)
        return encoded, reward, terminated, truncated, {**info, "observation": observation}

    def action_masks(self) -> np.ndarray:
        observation = self.env.observation
        if observation is None:
            raise RuntimeError("Call reset before action_masks")
        return self._encode(observation)["action_mask"].astype(bool)

    def render(self) -> str | None:
        return self.env.render() if self.render_mode == "ansi" else None

    def close(self) -> None:
        self.env.close()
