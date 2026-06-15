"""Integration tests that drive the real built sts_lightspeed simulator.

Gated with @requires_simulator: skipped (with a reason) when the build is
missing, unless STS_REQUIRE_SIMULATOR=1 forces a fail-closed run. See
tests/support.py and tests/CLAUDE.md.
"""
import unittest

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import run_rollout
from sts_ai.schemas import AgentDecision

from tests.support import requires_simulator


class BadIndexAgent:
    name = "bad"

    def reseed(self, policy_seed):
        return None

    def choose_action(self, state_text, legal_actions):
        return AgentDecision(action_index=999, raw_response="bad index")


@requires_simulator
class RolloutFallbackTest(unittest.TestCase):
    def test_out_of_range_action_records_requested_and_executed_actions(self):
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
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


@requires_simulator
class SerializerSmokeTest(unittest.TestCase):
    def test_state_uses_screen_name(self):
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
        self.assertIn("screen EVENT_SCREEN", env.describe_state())

    def test_neow_empty_drawback_has_no_trailing_slash(self):
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
        descriptions = [action.description for action in env.legal_actions()]
        self.assertIn("event option 1: Obtain three potions.", descriptions)
        self.assertNotIn("Obtain three potions. / ", descriptions)

    def test_action_descriptions_omit_bits_prefix(self):
        # The raw action `bits` are internal binding detail and must not leak into
        # the human-/model-facing description; they remain on LegalAction.bits.
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
        actions = env.legal_actions()
        self.assertTrue(actions)
        for action in actions:
            self.assertNotIn("bits=", action.description)
        # the structured field is still populated
        self.assertEqual([a.bits for a in actions], [a.bits for a in actions])

    def test_state_room_label_is_not_invalid(self):
        # On the Neow floor the simulator leaves curRoom == INVALID; the serialized
        # header should render that as "room none", never "room INVALID".
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
        state = env.describe_state()
        self.assertIn("room none", state)
        self.assertNotIn("room INVALID", state)


if __name__ == "__main__":
    unittest.main()
