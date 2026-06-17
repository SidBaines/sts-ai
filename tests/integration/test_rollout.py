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
class RolloutAgentInvalidTest(unittest.TestCase):
    def test_out_of_range_action_records_invalid_and_stops_without_execution(self):
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
        result = run_rollout(env, BadIndexAgent(), max_decisions=1)
        self.assertEqual(result.stopped_reason, "agent_invalid")
        self.assertEqual(len(result.decisions), 1)

        record = result.decisions[0]
        self.assertEqual(record.agent["action_index"], 999)
        self.assertFalse(record.agent["valid"])
        self.assertFalse(record.action_executed)
        self.assertEqual(record.selected_action, {})
        self.assertEqual(
            record.agent["metadata"]["invalid_reason"],
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


@requires_simulator
class MapRepresentationTest(unittest.TestCase):
    """The binding exposes the act map as a graph and describeGameState no longer
    dumps the unparseable ASCII grid; glossary renders a per-choice summary."""

    def _advance_to_map_screen(self, env, limit=60):
        for _ in range(limit):
            if env.is_terminal():
                return False
            env.advance_to_decision()
            if "MAP_SCREEN" in env.summary().get("screen_state", ""):
                return True
            env.step(0)
        return False

    def test_map_graph_shape_and_no_ascii_in_describe_state(self):
        from sts_ai import glossary

        env = LightspeedHybridEnv(world_seed=3, battle_simulations=50, max_act=1)
        self.assertTrue(self._advance_to_map_screen(env), "no MAP_SCREEN reached")

        # describeGameState must not carry the old ASCII map any more.
        self.assertNotIn("\nMap:\n", env.describe_state())

        graph = env.map_graph()
        self.assertIsNotNone(graph)
        self.assertIn("cur_y", graph)
        self.assertTrue(graph["nodes"], "map graph has no nodes")
        for node in graph["nodes"]:
            self.assertEqual(set(node), {"x", "y", "room", "edges"})
            self.assertTrue(0 <= node["x"] < 7 and 0 <= node["y"] < 15)
            for child_x in node["edges"]:
                self.assertTrue(0 <= child_x < 7)

        # Off the map screen, map_graph() is None (only populated on MAP_SCREEN).
        la = env.legal_actions()
        lad = [env.action_dict(a) for a in la]
        rendered = glossary.augment(env.describe_state(), lad, env.phase(), map_graph=graph)
        self.assertIn("\nMap:\n", rendered)
        self.assertIn("Your choices (match x= to the LEGAL ACTIONS):", rendered)
        # Every offered map choice's x= appears in the rendered summary.
        import re

        for desc in (a["description"] for a in lad):
            m = re.search(r"choose map node x=(\d+)", desc)
            if m:
                self.assertIn(f"x={m.group(1)}:", rendered)

    def test_map_graph_none_off_map_screen(self):
        # Neow floor (EVENT_SCREEN) is not a map screen.
        env = LightspeedHybridEnv(world_seed=1, battle_simulations=50)
        env.advance_to_decision()
        self.assertNotIn("MAP_SCREEN", env.summary().get("screen_state", ""))
        self.assertIsNone(env.map_graph())


if __name__ == "__main__":
    unittest.main()
