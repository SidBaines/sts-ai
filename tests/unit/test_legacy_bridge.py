from __future__ import annotations

from copy import deepcopy
import unittest

from sts_policy.legacy_bridge import project_legacy_menu


class LegacyBridgeTests(unittest.TestCase):
    def test_free_instance_restores_old_description_and_maps_first_copy(self):
        obs = {"phase": "combat", "state": {"observation_version": "sts_env_public_v2"},
               "legal_actions": [
                   {"index": 4, "description": "play Strike [Attack] (cost 0) -> FOE (deal 6) [hand 1; free once]",
                    "public": {"kind": "play_card", "card": {"cost_for_turn": 1}}},
                   {"index": 5, "description": "play Strike [Attack] (cost 1) -> FOE (deal 6) [hand 2]",
                    "public": {"kind": "play_card", "card": {"cost_for_turn": 1}}},
                   {"index": 7, "description": "end turn", "public": {"kind": "end_turn"}}]}
        before = deepcopy(obs)
        projected = project_legacy_menu(obs)
        self.assertEqual(projected.current_indices, (4, 7))
        self.assertEqual(projected.observation["legal_actions"][0],
                         {"index": 0, "description": "play Strike [Attack] (cost 1) -> FOE (deal 6)"})
        self.assertEqual(obs, before)

    def test_ooc_duplicate_offers_and_noncontiguous_current_indices_remain_distinct(self):
        obs = {"phase": "out_of_combat", "state": {"observation_version": "sts_env_public_v2"},
               "legal_actions": [{"index": 3, "description": "take card Strike"},
                                 {"index": 9, "description": "take card Strike"}]}
        result = project_legacy_menu(obs)
        self.assertEqual(result.current_indices, (3, 9))
        self.assertEqual(len(result.observation["legal_actions"]), 2)

    def test_missing_mechanics_and_wrong_interface_fail_closed(self):
        obs = {"phase": "combat", "state": {"observation_version": "sts_env_public_v2"},
               "legal_actions": [{"index": 0, "description": "play Strike",
                                  "public": {"kind": "play_card", "card": {}}}]}
        with self.assertRaisesRegex(ValueError, "current cost"):
            project_legacy_menu(obs)
        obs["state"]["observation_version"] = "sts_env_public_v1"
        with self.assertRaisesRegex(ValueError, "public-v2"):
            project_legacy_menu(obs)


if __name__ == "__main__":
    unittest.main()
