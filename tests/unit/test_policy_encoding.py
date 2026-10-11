from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from sts_policy.encoding import ACTION_DIM, STATE_DIM, encode_observation


def observation():
    return {"phase": "combat", "text": "Battle turn 1\nPlayer HP: 50/80",
            "state": {"cur_hp": 50, "max_hp": 80, "gold": 100, "act": 1, "floor": 3,
                      "combat": {"player_cur_hp": 50, "player_max_hp": 80, "turn": 1,
                                 "player_energy": 3, "player_block": 4,
                                 "player": {"powers": {"Dexterity": 2}},
                                 "cards": {"hand": [{"name": "Defend+", "type": "Skill", "cost_for_turn": 1}]},
                                 "enemies": [{"index": 0, "name": "SENTRY", "cur_hp": 8, "block": 2,
                                              "intent_damage": 9, "intent_hits": 1, "alive": True},
                                             {"index": 1, "name": "SENTRY", "cur_hp": 35, "block": 0,
                                              "intent_damage": 0, "intent_hits": -1, "alive": True}]}},
            "legal_actions": [{"index": 0, "description": "play Defend+ [Skill] (cost 1)"},
                              {"index": 1, "description": "play Strike (cost 1) -> SENTRY [enemy 0] (deal 10)"},
                              {"index": 2, "description": "play Strike (cost 1) -> SENTRY [enemy 1] (deal 10)"},
                              {"index": 3, "description": "end turn"}]}


class PolicyEncodingTests(unittest.TestCase):
    def test_dimensions_and_public_tactical_features(self):
        encoded = encode_observation(observation())
        self.assertEqual(encoded.state_features.shape, (STATE_DIM,))
        self.assertEqual(encoded.action_features.shape, (4, ACTION_DIM))
        self.assertEqual(encoded.state_features.dtype, np.float32)
        self.assertAlmostEqual(encoded.action_features[0, 34], 10 / 30, places=6)
        self.assertAlmostEqual(encoded.action_features[1, 48], 8 / 100, places=6)
        self.assertEqual(encoded.action_features[1, 53], 1)
        self.assertEqual(encoded.action_features[2, 53], 0)
        self.assertTrue(np.isfinite(encoded.state_features).all())

    def test_private_fields_and_menu_indices_never_enter_features(self):
        public = observation()
        augmented = deepcopy(public)
        augmented["info"] = {"native_values": [1, 2, 3], "seed": 999}
        augmented["state"]["world_seed"] = 999
        augmented["state"]["private_card_order"] = ["Secret"]
        augmented["state"]["combat"]["player"]["rng_state"] = "hidden"
        augmented["text"] += "\nprivate_diagnostic: seed 999 hidden card order"
        for i, action in enumerate(augmented["legal_actions"]):
            action.update(index=100 + i, bits=0xDEADBEEF, native_values=1234)
        a, b = encode_observation(public), encode_observation(augmented)
        np.testing.assert_array_equal(a.state_features, b.state_features)
        np.testing.assert_array_equal(a.action_features, b.action_features)
        self.assertEqual(b.action_indices.tolist(), [100, 101, 102, 103])

    def test_menu_permutation_equivariance(self):
        original = observation()
        shuffled = deepcopy(original)
        permutation = [3, 1, 0, 2]
        shuffled["legal_actions"] = [original["legal_actions"][i] for i in permutation]
        a, b = encode_observation(original), encode_observation(shuffled)
        np.testing.assert_allclose(a.state_features, b.state_features, atol=1e-6)
        np.testing.assert_array_equal(a.action_features[permutation], b.action_features)

    def test_ooc_public_deck_and_selection_purpose(self):
        base = {"phase": "out_of_combat", "state": {"cur_hp": 50, "max_hp": 80},
                "text": "Deck: (2): {Strike, Defend,}\nRelics: {Burning Blood:0,}\nPotions: none\nChoose a card to REMOVE from your deck (chosen 0 of 1).",
                "legal_actions": [{"index": 0, "description": "select card index 0: Strike [Attack, Common]"}]}
        a = encode_observation(base)
        base["text"] = base["text"].replace("REMOVE", "UPGRADE")
        b = encode_observation(base)
        self.assertFalse(np.array_equal(a.state_features, b.state_features))
        self.assertEqual(a.coverage["ooc_deck_line"], 1)
        self.assertEqual(a.coverage["known_action_rules"], 1)

    def test_unordered_draw_bag_and_multiselection(self):
        original = observation()
        original["state"]["combat"]["cards"]["draw"] = [{"name": "Strike", "count": 2}, {"name": "Defend", "count": 1}]
        shuffled = deepcopy(original)
        shuffled["state"]["combat"]["cards"]["draw"].reverse()
        np.testing.assert_allclose(encode_observation(original).state_features,
                                   encode_observation(shuffled).state_features, atol=1e-6)
        original["legal_actions"] = [{"index": 0, "description": "exhaust: hand [0] Defend [Skill] (cost 1); hand [2] Strike [Attack] (cost 1)"}]
        encoded = encode_observation(original)
        self.assertAlmostEqual(encoded.action_features[0, 55], 0.2, places=6)

    def test_terminal_empty_menu_and_unknown_coverage(self):
        original = observation()
        original["legal_actions"] = []
        self.assertEqual(encode_observation(original).action_features.shape, (0, ACTION_DIM))
        original["legal_actions"] = [{"index": 0, "description": "event option 0: effects unavailable"},
                                     {"index": 1, "description": "play Mystery (cost 1)"}]
        encoded = encode_observation(original)
        self.assertEqual(encoded.coverage["unknown_event_effects"], 1)
        self.assertEqual(encoded.coverage["unknown_action_rules"], 1)

    def test_map_choices_encode_only_reachable_routes(self):
        public = {"phase": "out_of_combat", "text": "", "state": {"map": {"cur_y": -1,
                  "nodes": [{"x": 0, "y": 0, "room": "MONSTER", "edges": [0]},
                            {"x": 1, "y": 0, "room": "MONSTER", "edges": [1]},
                            {"x": 0, "y": 1, "room": "REST", "edges": []},
                            {"x": 1, "y": 1, "room": "ELITE", "edges": []}]}},
                  "legal_actions": [{"index": 0, "description": "choose map node x=0 room=MONSTER"},
                                    {"index": 1, "description": "choose map node x=1 room=MONSTER"}]}
        encoded = encode_observation(public)
        self.assertFalse(np.array_equal(encoded.action_features[0], encoded.action_features[1]))
        # Numbers/coordinates do not themselves encode different actions. Once
        # reachable rooms are equal, these two labels have identical semantics.
        public["state"]["map"]["nodes"][3]["room"] = "REST"
        equal_routes = encode_observation(public)
        np.testing.assert_array_equal(equal_routes.action_features[0], equal_routes.action_features[1])

    def test_native_potion_prefix_and_card_remove_service(self):
        public = observation()
        public["legal_actions"] = [{"index": 0, "description": "drink potion Block Potion"},
                                    {"index": 1, "description": "drink potion Energy Potion"},
                                    {"index": 2, "description": "buy card remove for 75g"}]
        encoded = encode_observation(public)
        self.assertEqual(encoded.coverage["known_action_rules"], 2)
        self.assertEqual(encoded.coverage["unknown_action_rules"], 0)
        self.assertAlmostEqual(encoded.action_features[0, 34], 12 / 30, places=6)
        self.assertAlmostEqual(encoded.action_features[1, 47], 2 / 3, places=6)
