from __future__ import annotations

import sys
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sts_env.agents import FirstAgent, NativeAgent, RandomAgent, choose_action, load_agent


class AgentTests(unittest.TestCase):
    observation = {"legal_actions": [{"index": 7}, {"index": 12}]}

    def test_first_and_random_use_legal_indices_not_positions(self) -> None:
        self.assertEqual(FirstAgent().choose_action(self.observation), 7)
        agent = RandomAgent()
        agent.reset(42)
        first = [agent.choose_action(self.observation) for _ in range(20)]
        agent.reset(42)
        self.assertEqual(first, [agent.choose_action(self.observation) for _ in range(20)])
        self.assertTrue(set(first) <= {7, 12})

    def test_plugin_keyword_config_and_reset_protocol(self) -> None:
        module = types.ModuleType("test_policy_plugin")

        class Policy:
            def __init__(self, index: int) -> None:
                self.index = index

            def choose_action(self, observation: dict) -> int:
                return self.index

        module.create = Policy
        with patch.dict(sys.modules, {module.__name__: module}):
            agent = load_agent("test_policy_plugin:create", {"index": 12})
        self.assertEqual(choose_action(agent, self.observation, None), 12)

    def test_malformed_factory_and_invalid_actions_fail_closed(self) -> None:
        module = types.ModuleType("bad_policy_plugin")
        module.create = lambda: object()
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(TypeError, "choose_action"):
                load_agent("bad_policy_plugin:create")
        for value in (True, "7", -1, 0):
            agent = types.SimpleNamespace(choose_action=lambda observation: value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                choose_action(agent, self.observation, None)

    def test_native_baseline_queries_environment_without_observation_hint(self) -> None:
        env = types.SimpleNamespace(native_action=lambda: 12)
        self.assertEqual(choose_action(NativeAgent(), self.observation, env), 12)

    def test_agent_cannot_mutate_canonical_menu_and_integer_protocol_is_supported(self) -> None:
        class Index:
            def __index__(self) -> int:
                return 7

        def decide(observation: dict) -> Index:
            observation["legal_actions"][0]["index"] = 99
            return Index()

        canonical = {"legal_actions": [{"index": 7}]}
        self.assertEqual(choose_action(types.SimpleNamespace(choose_action=decide), canonical, None), 7)
        self.assertEqual(canonical["legal_actions"][0]["index"], 7)

    def test_plugin_module_in_working_directory_loads_from_console_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "cwd_policy_test.py").write_text("class Agent:\n    def choose_action(self, observation):\n        return 7\ndef create():\n    return Agent()\n")
            previous = os.getcwd()
            try:
                os.chdir(directory)
                self.assertEqual(load_agent("cwd_policy_test:create").choose_action(self.observation), 7)
            finally:
                os.chdir(previous)
                sys.modules.pop("cwd_policy_test", None)


if __name__ == "__main__":
    unittest.main()
