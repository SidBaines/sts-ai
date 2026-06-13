import unittest

from sts_ai.agents import FirstLegalAgent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import run_rollout
from sts_ai.schemas import AgentDecision


class BadIndexAgent:
    name = "bad"

    def choose_action(self, state_text, legal_actions):
        return AgentDecision(action_index=999, raw_response="bad index")


class TerminalBoundaryTest(unittest.TestCase):
    def test_act_equal_to_max_act_is_not_terminal(self):
        env = object.__new__(LightspeedHybridEnv)
        env.max_act = 1
        env.sts = type("Sts", (), {"GameOutcome": type("Outcome", (), {"UNDECIDED": "undecided"})})
        env.gc = type("Gc", (), {"outcome": "undecided", "act": 1})
        self.assertFalse(env.is_terminal())

    def test_act_greater_than_max_act_is_terminal(self):
        env = object.__new__(LightspeedHybridEnv)
        env.max_act = 1
        env.sts = type("Sts", (), {"GameOutcome": type("Outcome", (), {"UNDECIDED": "undecided"})})
        env.gc = type("Gc", (), {"outcome": "undecided", "act": 2})
        self.assertTrue(env.is_terminal())


class RolloutFallbackTest(unittest.TestCase):
    def test_out_of_range_action_records_requested_and_executed_actions(self):
        env = LightspeedHybridEnv(seed=1, battle_simulations=50)
        result = run_rollout(env, BadIndexAgent(), max_decisions=1)
        self.assertEqual(len(result.decisions), 1)

        record = result.decisions[0]
        self.assertEqual(record.agent["action_index"], 999)
        self.assertFalse(record.agent["valid"])
        self.assertEqual(record.selected_action["index"], 0)
        self.assertEqual(
            record.agent["metadata"]["fallback_reason"],
            "agent returned out-of-range action",
        )


class SerializerSmokeTest(unittest.TestCase):
    def test_state_uses_screen_name(self):
        env = LightspeedHybridEnv(seed=1, battle_simulations=50)
        self.assertIn("screen EVENT_SCREEN", env.describe_state())

    def test_neow_empty_drawback_has_no_trailing_slash(self):
        env = LightspeedHybridEnv(seed=1, battle_simulations=50)
        descriptions = [action.description for action in env.legal_actions()]
        self.assertIn("bits=1 event option 1: Obtain three potions.", descriptions)
        self.assertNotIn("Obtain three potions. / ", descriptions)


if __name__ == "__main__":
    unittest.main()
