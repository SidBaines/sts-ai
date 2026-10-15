from __future__ import annotations

from dataclasses import dataclass
import time
import unittest
from unittest.mock import patch

from sts_env.env import SlayTheSpireEnv, SimulatorError, SimulatorTimeout


@dataclass
class Action:
    index: int = 0
    bits: int = 1
    description: str = "advance"


class FakeBackend:
    def __init__(self, *, outcome="UNDECIDED", cutoff=False, automatic_death=False, act=1):
        self.outcome, self.cutoff = outcome, cutoff
        self.act = act
        self.automatic_death = automatic_death
        self.steps = 0
        self.native_steps = 0
        self.undefined = False
    def seed_native_ooc(self, seed):
        self.seed = seed
    def advance_to_decision(self):
        return 0
    def phase(self):
        return "combat" if self.native_steps else "out_of_combat"
    def is_terminal(self):
        return self.outcome != "UNDECIDED" or (self.cutoff and self.steps > 0)
    def summary(self):
        return {"world_seed": 123, "outcome": "GameOutcome." + self.outcome,
                "act": self.act + 1 if self.cutoff and self.steps else self.act,
                "cur_hp": 80, "max_hp": 80, "undefined_behavior_evoked": self.undefined}
    def legal_actions(self):
        return [Action()]
    def describe_state(self):
        return "state\nAscension: 0"
    def map_graph(self):
        return None
    def _turn_math_lines(self):
        return []
    def step(self, index):
        self.steps += 1
        return Action()
    def advance_native_ooc(self):
        self.native_steps += 1
        if self.automatic_death:
            self.outcome = "PLAYER_LOSS"
        return Action()
    def native_ooc_action(self):
        return 0
    def display_index_for_bits(self, bits):
        return 0
    def search_best_combat_action(self, *args, **kwargs):
        return {"bits": 1}


def stalled_worker(connection, config, seed):
    time.sleep(30)


class EnvironmentTests(unittest.TestCase):
    def make(self, backend=None, **kwargs):
        backend = backend or FakeBackend()
        patcher = patch("sts_env.env._make_backend", return_value=backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = SlayTheSpireEnv(isolated=False, **kwargs)
        self.addCleanup(env.close)
        return env, backend

    def test_invalid_action_never_advances_and_state_copies_are_owned(self):
        env, backend = self.make()
        obs, _ = env.reset(seed=3)
        obs["state"]["cur_hp"] = 0
        obs["legal_actions"].clear()
        for bad in (-1, 1, 0.2, True, "0"):
            with self.assertRaises(ValueError):
                env.step(bad)
        self.assertEqual(backend.steps, 0)
        self.assertEqual(env.observation["state"]["cur_hp"], 80)
        self.assertEqual(len(env.observation["legal_actions"]), 1)
        env.step(0)
        self.assertEqual(backend.steps, 1)

    def test_step_limit_is_truncation_and_reset_reopens_episode(self):
        env, _ = self.make(max_steps=1)
        env.reset(seed=3)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0, False, True))
        self.assertEqual(info["termination_reason"], "max_steps")
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=3)
        env.step(0)

    def test_game_outcomes_are_terminated_with_sparse_reward(self):
        for outcome, reward in (("PLAYER_VICTORY", 1), ("PLAYER_LOSS", 0)):
            env, backend = self.make()
            env.reset(seed=3)
            backend.outcome = outcome
            _, actual, terminated, truncated, _ = env.step(0)
            self.assertEqual((actual, terminated, truncated), (reward, True, False))

    def test_max_act_cutoff_is_not_victory(self):
        env, _ = self.make(FakeBackend(cutoff=True), max_act=1)
        env.reset(seed=3)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0, False, True))
        self.assertEqual(info["termination_reason"], "max_act")

    def test_status_at_act_boundaries(self):
        for max_act, act, outcome, expected in (
            (1, 2, "UNDECIDED", (False, True, "max_act")),
            (2, 3, "UNDECIDED", (False, True, "max_act")),
            (3, 3, "UNDECIDED", (False, False, "")),
            (3, 4, "UNDECIDED", (True, False, "victory")),
            (4, 4, "UNDECIDED", (False, False, "")),
            (4, 5, "UNDECIDED", (False, True, "max_act")),
            (3, 4, "PLAYER_LOSS", (True, False, "defeat")),
            (3, 3, "PLAYER_VICTORY", (True, False, "victory")),
        ):
            with self.subTest(max_act=max_act, act=act, outcome=outcome):
                env, _ = self.make(max_act=max_act)
                env.reset(seed=3)
                observation = {"state": {"act": act, "outcome": "GameOutcome." + outcome},
                               "legal_actions": [Action()] if not any(expected[:2]) else []}
                self.assertEqual(env._episode._status(observation), expected)

    def test_keyed_act3_victory_has_sparse_reward_and_episode_outcome(self):
        env, _ = self.make(FakeBackend(cutoff=True, act=3), max_act=3, max_steps=1)
        env.reset(seed=3)
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
        self.assertEqual(info["termination_reason"], "victory")
        self.assertEqual(info["outcome"], "GameOutcome.PLAYER_VICTORY")
        self.assertEqual(observation["state"]["outcome"], "GameOutcome.UNDECIDED")
        self.assertEqual(observation["legal_actions"], [])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_reverse_hybrid_advances_native_ooc_and_exposes_combat(self):
        env, backend = self.make(mode="combat")
        obs, info = env.reset(seed=123)
        self.assertEqual(obs["phase"], "combat")
        self.assertEqual(info["native_steps"], 1)
        self.assertEqual(backend.seed, 123)
        self.assertEqual(env.native_action(), 0)

    def test_reverse_hybrid_can_terminate_before_first_external_decision(self):
        env, _ = self.make(FakeBackend(automatic_death=True), mode="combat")
        obs, info = env.reset(seed=3)
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated"])
        self.assertEqual(obs["legal_actions"], [])

    def test_undefined_behavior_is_failure_not_loss(self):
        env, backend = self.make()
        env.reset(seed=3)
        backend.undefined = True
        with self.assertRaisesRegex(SimulatorError, "undefined behavior"):
            env.step(0)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_reward_adapter_receives_owned_public_states(self):
        def reward(before, index, after, info):
            before.clear()
            after.clear()
            return -0.25
        env, _ = self.make(reward_fn=reward)
        env.reset(seed=3)
        obs, reward, *_ = env.step(0)
        self.assertEqual(reward, -0.25)
        self.assertIn("state", obs)
        self.assertIn("state", env.observation)

    def test_close_is_idempotent_and_prevents_step(self):
        env, _ = self.make()
        env.reset(seed=3)
        env.close()
        env.close()
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_seed_and_settings_fail_before_loading_native(self):
        for settings in ({"mode": "unknown"}, {"max_steps": 0}, {"step_timeout": float("nan")}, {"ascension": 21}):
            with self.assertRaises(ValueError):
                SlayTheSpireEnv(**settings)
        env = SlayTheSpireEnv()
        self.addCleanup(env.close)
        for seed in (-1, True, 2**64, "3"):
            with self.assertRaises(ValueError):
                env.reset(seed=seed)

    def test_native_startup_timeout_kills_and_reaps_worker(self):
        env = SlayTheSpireEnv(step_timeout=0.25)
        self.addCleanup(env.close)
        with patch("sts_env.env._worker", stalled_worker):
            start = time.monotonic()
            with self.assertRaises(SimulatorTimeout):
                env.reset(seed=3)
        self.assertLess(time.monotonic() - start, 3)
        self.assertIsNone(env._process)
        self.assertIsNone(env._connection)


    def test_failed_spawn_cleans_up_unstarted_process(self):
        env = SlayTheSpireEnv()
        self.addCleanup(env.close)
        # A non-importable target fails before the child starts under spawn.
        with patch("sts_env.env._worker", lambda *args: None):
            with self.assertRaises((AttributeError, TypeError)):
                env.reset(seed=3)
        self.assertIsNone(env._process)
        self.assertIsNone(env._connection)


    def test_world_seed_is_provenance_not_public_observation(self):
        env, _ = self.make()
        obs, info = env.reset(seed=7)
        self.assertNotIn("world_seed", obs["state"])
        self.assertEqual(info["world_seed"], 7)

    def test_visible_map_keeps_route_order_and_is_owned(self):
        backend = FakeBackend()
        graph = {"cur_y": -1, "nodes": [{"x": 0, "y": 0, "room": "MONSTER", "edges": [1]}]}
        backend.map_graph = lambda: graph
        env, _ = self.make(backend)
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs["state"]["map"], graph)
        self.assertIn("(0,0) MONSTER -> (1,1)", obs["text"])
        obs["state"]["map"]["nodes"].clear()
        self.assertEqual(len(graph["nodes"]), 1)

    def test_visible_routes_mark_only_the_burning_elite(self):
        backend = FakeBackend()
        graph = {"cur_y": 4, "burning_elite": {"x": 1, "y": 5}, "nodes": [
            {"x": 1, "y": 5, "room": "ELITE", "edges": [1], "burning_elite": True},
            {"x": 2, "y": 5, "room": "ELITE", "edges": [2], "burning_elite": False},
        ]}
        backend.map_graph = lambda: graph
        env, _ = self.make(backend)
        observation, _ = env.reset(seed=3)
        self.assertIn("(1,5) ELITE (burning: Emerald Key) -> (1,6)", observation["text"])
        self.assertIn("(2,5) ELITE -> (2,6)", observation["text"])
        self.assertEqual(observation["state"]["map"], graph)

    def test_native_seed_effective_width_is_recorded(self):
        env, _ = self.make(ooc_seed=2**32 + 7)
        _, info = env.reset(seed=3)
        self.assertEqual(info["ooc_seed"], 2**32 + 7)
        self.assertEqual(info["ooc_effective_seed"], 7)


    def test_hidden_native_action_bits_do_not_affect_public_observation(self):
        backend = FakeBackend()
        backend.legal_actions = lambda: [Action(bits=123)]
        env, _ = self.make(backend)
        observation, _ = env.reset(seed=3)
        self.assertEqual(set(observation["legal_actions"][0]), {"index", "description"})
        # Draw-selector native bits can name different private draw-vector slots
        # for otherwise identical public cards. They must not reach a policy.
        backend.legal_actions = lambda: [Action(bits=987)]
        self.assertEqual(env._episode._observe(), observation)
        _, _, _, _, info = env.step(0)
        self.assertIn("bits", info["executed_action"])
