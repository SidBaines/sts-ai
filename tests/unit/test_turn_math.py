from __future__ import annotations

import unittest

from sts_ai.turn_math import TurnMathInputs, turn_math_lines


def _inputs(**overrides) -> TurnMathInputs:
    values = {
        "incoming_damage": 11,
        "player_block": 3,
        "player_metallicize": 2,
        "energy": 3,
        "hand_attacks": [],
        "living_enemies": [("JAW_WORM", 20, 0)],
    }
    values.update(overrides)
    return TurnMathInputs(**values)


class TurnMathLinesTest(unittest.TestCase):
    def test_bounded_knapsack_counts_hand_multiplicity(self):
        lines = turn_math_lines(
            _inputs(
                energy=2,
                hand_attacks=[
                    ("Strike", 1, 6, 2),
                    ("Bash", 2, 8, 1),
                ],
            )
        )
        self.assertEqual(
            lines[1],
            "Max attack damage playable this turn (using shown deal values, "
            "current modifiers only): 12.",
        )

    def test_energy_is_binding_and_zero_cost_cards_are_included(self):
        lines = turn_math_lines(
            _inputs(
                energy=2,
                hand_attacks=[
                    ("Expensive", 2, 9, 1),
                    ("Two cheap copies", 1, 6, 2),
                    ("Free", 0, 3, 1),
                ],
            )
        )
        self.assertIn("current modifiers only): 15.", lines[1])

    def test_unannotated_and_x_cost_attacks_are_listed_as_excluded(self):
        lines = turn_math_lines(
            _inputs(
                hand_attacks=[
                    ("Strike", 1, 6, 1),
                    ("Fiend Fire", 2, None, 1),
                    ("Whirlwind", None, None, 1),
                ]
            )
        )
        self.assertEqual(
            lines[1],
            "Max attack damage playable this turn (using shown deal values, "
            "current modifiers only): 6. Excluded from this total: Fiend "
            "Fire, Whirlwind.",
        )

    def test_unaffordable_attack_is_listed_as_excluded(self):
        lines = turn_math_lines(
            _inputs(
                energy=1,
                hand_attacks=[("Bash", 3, None, 1)],
            )
        )
        self.assertEqual(
            lines[1],
            "Max attack damage playable this turn (using shown deal values, "
            "current modifiers only): 0. Excluded from this total: Bash.",
        )

    def test_blocked_enemy_prevents_false_positive_lethal(self):
        lines = turn_math_lines(
            _inputs(
                energy=1,
                hand_attacks=[("Strike", 1, 6, 1)],
                living_enemies=[("CULTIST", 6, 1)],
            )
        )
        self.assertEqual(
            lines[2],
            "Lethal check vs CULTIST (HP 6, block 1): not lethal this turn.",
        )

    def test_zero_block_lethal_wording_is_regular(self):
        lines = turn_math_lines(
            _inputs(
                energy=1,
                hand_attacks=[("Strike", 1, 6, 1)],
                living_enemies=[("CULTIST", 6, 0)],
            )
        )
        self.assertEqual(
            lines[2],
            "Lethal check vs CULTIST (HP 6, block 0): lethal available this turn.",
        )

    def test_lethal_boundary_includes_enemy_block(self):
        lines = turn_math_lines(
            _inputs(
                energy=1,
                hand_attacks=[("Strike", 1, 6, 1)],
                living_enemies=[("CULTIST", 4, 2)],
            )
        )
        self.assertEqual(
            lines[2],
            "Lethal check vs CULTIST (HP 4, block 2): lethal available this turn.",
        )

    def test_end_turn_projection_floors_at_zero(self):
        lines = turn_math_lines(
            _inputs(incoming_damage=4, player_block=3, player_metallicize=2)
        )
        self.assertEqual(
            lines[0],
            "End-turn projection: you would take 0 damage (incoming 4 - block 3 "
            "- Metallicize 2; minimum 0).",
        )

    def test_multiple_enemies_omit_lethal_line(self):
        lines = turn_math_lines(
            _inputs(living_enemies=[("LOUSE", 5, 0), ("LOUSE", 6, 0)])
        )
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
