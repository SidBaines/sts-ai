from __future__ import annotations

import unittest

from sts_ai.turn_math import TurnMathInputs, turn_math_lines as v3_turn_math_lines
from sts_ai.turn_math_v4 import turn_math_lines


def _inputs(**overrides: object) -> TurnMathInputs:
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


class TurnMathV4Test(unittest.TestCase):
    def test_attack_sum_keeps_costs_and_hand_multiplicity_fixed(self):
        lines = turn_math_lines(
            _inputs(
                energy=2,
                hand_attacks=[
                    ("Strike", 1, 6, 2),
                    ("Bash", 2, 8, 1),
                    ("Free Attack", 0, 3, 1),
                ],
            )
        )
        self.assertIn("within current energy 2", lines[1])
        self.assertIn("holding current costs and shown damage fixed: 15.", lines[1])
        self.assertIn("does not simulate an action sequence", lines[1])
        self.assertIn("target allocation", lines[1])

    def test_unknown_damage_x_cost_and_unaffordable_attacks_are_explicit(self):
        lines = turn_math_lines(
            _inputs(
                energy=1,
                hand_attacks=[
                    ("Strike", 1, 6, 1),
                    ("Fiend Fire", 2, None, 1),
                    ("Whirlwind", None, None, 1),
                    ("Bash", 2, 8, 1),
                ],
            )
        )
        self.assertIn("shown damage fixed: 6.", lines[1])
        self.assertIn(
            "Attacks excluded from this sum: Fiend Fire, Whirlwind, Bash.",
            lines[1],
        )

    def test_zero_energy_still_counts_each_free_attack_copy(self):
        lines = turn_math_lines(
            _inputs(energy=0, hand_attacks=[("Free Attack", 0, 3, 2)])
        )
        self.assertIn("within current energy 0", lines[1])
        self.assertIn("shown damage fixed: 6.", lines[1])

    def test_skill_enabled_win_cannot_receive_negative_lethal_verdict(self):
        # Audit seed_4009_r0_d43 decision 32: energy generation and Flex
        # enabled a same-turn kill absent from the fixed-current attack sum.
        lines = turn_math_lines(
            _inputs(
                energy=3,
                hand_attacks=[("Body Slam", 1, 0, 1), ("Strike", 1, 6, 1)],
                living_enemies=[("MAD_GREMLIN", 7, 0)],
            )
        )
        self.assertIn("shown damage fixed: 6.", lines[1])
        self.assertIn("energy generation or refunds", lines[1])
        self.assertIn("other card effects", lines[1])
        self.assertIn("does not simulate an action sequence", lines[1])
        self.assertNotIn("not lethal this turn", "\n".join(lines))

    def test_high_sum_does_not_claim_an_executable_winning_sequence(self):
        lines = turn_math_lines(
            _inputs(
                energy=3,
                hand_attacks=[("Attack", 1, 20, 3)],
                living_enemies=[("GUARDIAN", 10, 0)],
            )
        )
        self.assertIn("shown damage fixed: 60.", lines[1])
        self.assertNotIn("lethal available", "\n".join(lines))
        self.assertNotIn("Lethal check", "\n".join(lines))
        self.assertIn("subsequent changes to costs, damage, or defenses", lines[1])

    def test_end_turn_examples_are_not_presented_as_hp_forecasts(self):
        # Audit examples: Plated Armor changes 14 -> 10 HP lost;
        # Burn changes 4 -> 6; Combust changes 0 -> 1. These omitted
        # effects are deliberately not inferred from the limited inputs.
        for incoming, block, arithmetic in [(14, 0, 14), (10, 6, 4), (0, 0, 0)]:
            with self.subTest(incoming=incoming, block=block):
                lines = turn_math_lines(
                    _inputs(
                        incoming_damage=incoming,
                        player_block=block,
                        player_metallicize=0,
                    )
                )
                self.assertIn(f") = {arithmetic}.", lines[0])
                self.assertIn("omits Plated Armor, Burn, Combust", lines[0])
                self.assertIn("does not predict HP loss", lines[0])
                self.assertNotIn("you would take", lines[0])
                self.assertNotIn("End-turn projection", lines[0])

    def test_incoming_arithmetic_floors_at_zero(self):
        lines = turn_math_lines(
            _inputs(incoming_damage=4, player_block=3, player_metallicize=2)
        )
        self.assertIn(
            "max(0, shown incoming 4 - current Block 3 - Metallicize 2) = 0.",
            lines[0],
        )

    def test_enemy_count_does_not_add_outcome_claims(self):
        for enemies in [[], [("LOUSE", 5, 0)], [("LOUSE", 5, 0), ("LOUSE", 6, 0)]]:
            with self.subTest(enemies=enemies):
                self.assertEqual(len(turn_math_lines(_inputs(living_enemies=enemies))), 2)

    def test_frozen_v3_rendering_is_unchanged(self):
        inputs = _inputs(
            incoming_damage=4,
            player_block=3,
            player_metallicize=2,
            energy=1,
            hand_attacks=[("Strike", 1, 6, 1)],
            living_enemies=[("CULTIST", 6, 0)],
        )
        self.assertEqual(
            v3_turn_math_lines(inputs),
            [
                "End-turn projection: you would take 0 damage (incoming 4 - block "
                "3 - Metallicize 2; minimum 0).",
                "Max attack damage playable this turn (using shown deal values, "
                "current modifiers only): 6.",
                "Lethal check vs CULTIST (HP 6, block 0): lethal available this turn.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
