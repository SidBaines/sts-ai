"""Unit tests for the pure-Python combat affordances (eval support)."""
from __future__ import annotations

import unittest

from sts_ai.affordances import calc_block, compute


def _combat(enemies, hand_lines, powers="none", energy=3, block=0):
    state = {"combat": {"player_energy": energy, "player_block": block, "enemies": enemies}}
    state_text = (
        "Battle turn 1\n"
        f"Player HP: 50/80, block: {block}, energy: {energy}/3\n"
        f"Player powers: {powers}\n"
        "Enemies:\n  [0] FOE HP 10/10, block 0, intent FOE_X\n"
        "Hand:\n" + "".join(f"  {l}\n" for l in hand_lines)
        + "Piles: draw 1, discard 0, exhaust 0\nPotions: none\n"
    )
    return state, state_text


class CalcBlockTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(calc_block(5, dex=0, frail=False, no_block=False), 5)

    def test_dexterity_adds(self):
        self.assertEqual(calc_block(5, dex=2, frail=False, no_block=False), 7)

    def test_frail_reduces_three_quarters(self):
        self.assertEqual(calc_block(5, dex=0, frail=True, no_block=False), 3)  # 5*3//4

    def test_no_block_zeroes(self):
        self.assertEqual(calc_block(8, dex=3, frail=False, no_block=True), 0)


class ComputeTest(unittest.TestCase):
    def test_out_of_combat_is_empty(self):
        self.assertEqual(compute({}, "Act 1 ...", [], "out_of_combat"), {})

    def test_lethal_detection_tolerates_enemy_index_suffix(self):
        # Two same-named enemies -> the binding appends " [enemy i]"; affordances must
        # still match the name (after stripping the suffix) to spot an available lethal.
        enemies = [
            {"name": "FUNGI_BEAST", "cur_hp": 6, "max_hp": 28, "block": 0, "alive": True,
             "intent_damage": 0, "intent_hits": -1},
            {"name": "FUNGI_BEAST", "cur_hp": 6, "max_hp": 28, "block": 0, "alive": True,
             "intent_damage": 0, "intent_hits": -1},
        ]
        state = {"combat": {"player_energy": 3, "player_block": 0, "enemies": enemies}}
        state_text = (
            "Battle turn 1\nPlayer HP: 50/80, block: 0, energy: 3/3\nPlayer powers: none\n"
            "Enemies:\n  [0] FUNGI_BEAST HP 6/28, block 0, intent X\n"
            "  [1] FUNGI_BEAST HP 6/28, block 0, intent X\n"
            "Hand:\n  [0] Strike (cost 1)\nPiles: draw 1, discard 0, exhaust 0\nPotions: none\n"
        )
        actions = [
            {"index": 0, "description": "play Strike (cost 1) -> FUNGI_BEAST [enemy 1] (deal 6)"},
            {"index": 1, "description": "end turn"},
        ]
        aff = compute(state, state_text, actions, "combat")
        self.assertTrue(aff["single_target_lethal_available"])

    def test_incoming_and_full_block(self):
        enemies = [{"name": "FOE", "cur_hp": 40, "block": 0, "alive": True,
                    "intent_damage": 4, "intent_hits": 2}]  # incoming 8
        state, text = _combat(enemies, ["[0] Defend (cost 1)", "[1] Defend (cost 1)", "[2] Strike (cost 1)"])
        legal = [{"description": "play Defend (cost 1)"}, {"description": "play Strike (cost 1) -> FOE (deal 6)"},
                 {"description": "end turn"}]
        a = compute(state, text, legal, "combat")
        self.assertEqual(a["incoming_damage_total"], 8)
        self.assertEqual(a["max_block_available"], 10)  # 2 Defends @5, energy 3
        self.assertTrue(a["full_block_possible"])       # 10 >= 8
        self.assertTrue(a["end_turn_available"])

    def test_frail_can_block_less(self):
        enemies = [{"name": "FOE", "cur_hp": 40, "block": 0, "alive": True,
                    "intent_damage": 8, "intent_hits": 1}]
        state, text = _combat(enemies, ["[0] Defend (cost 1)", "[1] Defend (cost 1)"], powers="Frail 1")
        a = compute(state, text, [{"description": "play Defend (cost 1)"}], "combat")
        self.assertEqual(a["max_block_available"], 6)   # each Defend 5 -> 3 under Frail; 2 of them
        self.assertFalse(a["full_block_possible"])       # 6 < 8

    def test_lethal_detection(self):
        enemies = [{"name": "FOE", "cur_hp": 10, "block": 0, "alive": True,
                    "intent_damage": 0, "intent_hits": -1}]  # not attacking
        state, text = _combat(enemies, ["[0] Strike (cost 1)"])
        legal = [{"description": "play Strike (cost 1) -> FOE (deal 12)"}, {"description": "end turn"}]
        a = compute(state, text, legal, "combat")
        self.assertEqual(a["incoming_damage_total"], 0)      # intent_hits -1 ignored
        self.assertEqual(a["max_single_target_damage"], 12)
        self.assertTrue(a["single_target_lethal_available"])  # 12 >= 10+0

    def test_not_lethal_when_blocked(self):
        enemies = [{"name": "FOE", "cur_hp": 10, "block": 5, "alive": True,
                    "intent_damage": 0, "intent_hits": -1}]
        state, text = _combat(enemies, ["[0] Strike (cost 1)"])
        legal = [{"description": "play Strike (cost 1) -> FOE (deal 12)"}]
        a = compute(state, text, legal, "combat")
        self.assertFalse(a["single_target_lethal_available"])  # 12 < 10+5

    def test_unknown_card_contributes_no_block(self):
        enemies = [{"name": "FOE", "cur_hp": 40, "block": 0, "alive": True,
                    "intent_damage": 8, "intent_hits": 1}]
        state, text = _combat(enemies, ["[0] Mysterycard (cost 1)"])
        a = compute(state, text, [{"description": "play Mysterycard (cost 1)"}], "combat")
        self.assertEqual(a["max_block_available"], 0)
        self.assertFalse(a["full_block_possible"])

    def test_missing_combat_dict_is_empty(self):
        self.assertEqual(compute({"foo": 1}, "text", [], "combat"), {})


if __name__ == "__main__":
    unittest.main()
