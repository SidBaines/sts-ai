from __future__ import annotations

import unittest

from sts_env._backend.glossary_v5 import augment, buff_effect, calculated_block, card_definition


def state(extra: str = "") -> str:
    return ("Battle turn 1\nPlayer HP: 51/80, block: 0, energy: 4/3\n"
            "Player powers: none\nEnemies:\n"
            "  [0] SLIME_BOSS HP 89/140, block 0, intent SLIME_BOSS_SLAM (deal 35)\n"
            "Hand:\n  [0] Defend [Skill] (cost 1)\n  [1] Defend+ [Skill] (cost 1)\n"
            "Piles: draw 1, discard 0, exhaust 0\n"
            "Draw pile contents (unordered): Rampage=5 [Attack] (cost 1)\n"
            "Relics: Burning Blood\nAscension: 0\n" + extra)


class GlossaryV5Test(unittest.TestCase):
    def test_instance_values_and_rampage_bonus_are_unambiguous(self):
        text = augment(state(), [], "combat")
        self.assertIn("Defend: Gain 5 Block.", text)
        self.assertIn("Defend+: Gain 8 Block.", text)
        self.assertNotIn("5 (8)", text)
        self.assertIn("Rampage=5: Deal 13 unmodified damage", text)
        self.assertIn("not total damage", text)
        self.assertIn("Current direct Block calculation for Defend: 5", text)
        self.assertIn("Current direct Block calculation for Defend+: 8", text)

    def test_split_and_round_rules_include_decision_relevant_exceptions(self):
        text = augment(state(), [], "combat")
        for phrase in ("half its maximum HP", "replacing its previous move",
                       "parent's current HP", "Energy A/B", "B is not a cap",
                       "Ice Cream", "Barricade or Blur", "Calipers", "absent Strength"):
            self.assertIn(phrase, text)

    def test_exact_buff_amounts_follow_ascension(self):
        for asc, strength, block, enrage in ((0, 3, 6, 2), (2, 4, 6, 2),
                                             (17, 5, 9, 2), (18, 5, 9, 3)):
            self.assertEqual(buff_effect("FUNGI_BEAST_GROW", asc), f"gains {strength} Strength")
            self.assertEqual(buff_effect("JAW_WORM_BELLOW", asc), f"gains {strength} Strength and {block} Block")
            self.assertIn(f"gains {enrage} Enrage", buff_effect("GREMLIN_NOB_BELLOW", asc))
        raw = state().replace("SLIME_BOSS_SLAM (deal 35)", "GREMLIN_NOB_BELLOW")
        text = augment(raw, [], "combat")
        self.assertIn("no damage; gains 2 Enrage", text)
        self.assertIn("Enrage:", text)

    def test_direct_block_uses_public_modifiers_and_native_iron_wave_formula(self):
        self.assertEqual(calculated_block("Defend", {"Dexterity": 2, "Frail": 1}), 5)
        self.assertEqual(calculated_block("Flame Barrier+", {"Frail": 1}), 12)
        self.assertEqual(calculated_block("Defend+", {"No Block": 1}), 0)
        self.assertEqual(calculated_block("Defend", {"Dexterity": -9}), 0)
        self.assertEqual(calculated_block("Iron Wave", {"Dexterity": 2}), 9)
        self.assertIsNone(calculated_block("Entrench", {}))

    def test_upgraded_effects_change_more_than_numeric_pairs(self):
        self.assertIn("chosen card", card_definition("True Grit+", []))
        self.assertIn("all cards", card_definition("Armaments+", []))
        self.assertNotIn("Exhaust", card_definition("Limit Break+", []))
        self.assertIn("Deal 34 damage", card_definition("Searing Blow+4", []))
        self.assertIn("deal 8 damage", card_definition("Whirlwind+", []))
        self.assertIn("Deal 21 unmodified damage", card_definition("Ritual Dagger=21", []))

    def test_cached_rules_follow_relic_context_and_full_instance_name(self):
        for relics in ([], ["Blue Candle"], ["Burning Blood"], ["Blue Candle"]):
            definition = card_definition("Injury", relics)
            self.assertEqual("Playable with Blue Candle" in definition, "Blue Candle" in relics)
        for name, damage in (("Rampage=5", 13), ("Rampage=9+", 17),
                             ("Ritual Dagger=21", 21), ("Ritual Dagger=25+", 25)):
            self.assertIn(f"{name}: Deal {damage} unmodified damage", card_definition(name, []))
        self.assertIn("Deal 34 damage", card_definition("Searing Blow+4", []))
        self.assertIn("Deal 51 damage", card_definition("Searing Blow+6", []))

    def test_pending_optional_exhaust_selection_is_clear(self):
        text = augment(state("Selection type: EXHAUST_MANY\n"),
                       [{"index": 0, "description": "exhaust no cards"}], "combat")
        self.assertIn("Exhausting no cards finishes", text)
        self.assertNotIn("Current direct Block calculation", text)
        self.assertNotIn("not enough energy", text)

    def test_requires_ascension_and_keeps_noncombat_unchanged(self):
        with self.assertRaisesRegex(ValueError, "ascension"):
            augment(state().replace("Ascension: 0", ""), [], "combat")
        from sts_env._backend.glossary_v4 import augment as previous
        self.assertEqual(augment("Your HP: 80/80", [], "out_of_combat"),
                         previous("Your HP: 80/80", [], "out_of_combat"))
