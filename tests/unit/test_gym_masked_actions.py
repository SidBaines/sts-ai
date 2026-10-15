from __future__ import annotations

import importlib.util
from copy import deepcopy
import unittest
from unittest.mock import patch


class FakeEnv:
    def __init__(self, **kwargs):
        self.config = {"max_steps": kwargs.get("max_steps", 3)}
        self.calls = []
        self.raw = {"state": {"cur_hp": 80, "max_hp": 80}, "phase": "out_of_combat",
                    "legal_actions": [{"index": 0, "description": "first"}]}

    def reset(self, *, seed):
        return self.raw, {"terminated": False, "truncated": False}

    def step(self, index):
        self.calls.append(index)
        return self.raw, 0.0, False, False, {"executed_action": {"index": index}}

    def close(self):
        pass


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional Gymnasium extra absent")
class MaskedActionTests(unittest.TestCase):
    def make_env(self, **kwargs):
        from sts_env.gymnasium import GymnasiumEnv
        with patch("sts_env.gymnasium.SlayTheSpireEnv", FakeEnv):
            return GymnasiumEnv(max_actions=8, **kwargs)

    def test_masked_slot_noop_fresh_observation_and_finite_horizon(self):
        env = self.make_env(max_steps=2)
        initial, _ = env.reset(seed=3)
        first, reward, terminated, truncated, info = env.step(7)
        self.assertEqual(env.env.calls, [])
        self.assertEqual(reward, 0)
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["invalid_action"])
        first["features"][0] = -999
        self.assertEqual(initial["features"][0], 80)
        _, _, terminated, truncated, info = env.step(7)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "max_steps")
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_valid_then_masked_slot_removes_old_execution_identity(self):
        env = self.make_env()
        env.reset(seed=3)
        env.step(0)
        _, reward, _, _, info = env.step(6)
        self.assertEqual(env.env.calls, [0])
        self.assertNotIn("executed_action", info)
        self.assertEqual(info["action_attempts"], 2)

    def test_out_of_fixed_space_remains_invalid(self):
        env = self.make_env()
        env.reset(seed=3)
        for action in (-1, 8, 0.5, True):
            with self.assertRaises(ValueError):
                env.step(action)

    def test_custom_invalid_reward_is_explicit_and_finite(self):
        env = self.make_env(invalid_action_reward=-0.25)
        env.reset(seed=3)
        self.assertEqual(env.step(7)[1], -0.25)
        with self.assertRaises(ValueError):
            self.make_env(invalid_action_reward=float("nan"))

    def test_numeric_pile_counts_include_duplicate_card_instances(self):
        env = self.make_env()
        env.reset(seed=3)
        raw = env.env.raw
        raw["state"]["combat"] = {"cards": {
            "hand": [{"index": 0}, {"index": 1}],
            "draw": [{"name": "Strike", "count": 4}, {"name": "Defend", "count": 3}],
            "discard": [{"name": "Strike", "count": 2}],
            "exhaust": [{"name": "Ascender's Bane", "count": 1}],
        }}
        features = env._encode(raw)["features"]
        positions = dict(zip(env.FEATURE_NAMES, features))
        self.assertEqual(positions["hand_count"], 2)
        self.assertEqual(positions["draw_count"], 7)
        self.assertEqual(positions["discard_count"], 2)
        self.assertEqual(positions["exhaust_count"], 1)

    def test_native_draw_indices_cannot_change_policy_features_or_mask(self):
        import numpy as np

        env = self.make_env()
        observation, _ = env.reset(seed=3)
        self.assertEqual(set(observation), {"features", "action_mask"})
        self.assertEqual(set(env.observation_space.spaces), {"features", "action_mask"})
        first = deepcopy(env.env.raw)
        second = deepcopy(env.env.raw)
        first["legal_actions"][0]["bits"] = 65536
        second["legal_actions"][0]["bits"] = 196608
        encoded_first, encoded_second = env._encode(first), env._encode(second)
        for field in encoded_first:
            np.testing.assert_array_equal(encoded_first[field], encoded_second[field])
        self.assertTrue(env.observation_space.contains(encoded_first))
