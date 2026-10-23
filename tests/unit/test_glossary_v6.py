from __future__ import annotations

import unittest

from sts_env._backend import glossary as legacy, glossary_v5 as v5, glossary_v6 as v6
from tests.unit.test_glossary_v5 import state


def render(move, ascension, *, attack=False):
    text = state().replace("Ascension: 0", f"Ascension: {ascension}")
    text = text.replace("SLIME_BOSS_SLAM (deal 35)", move + (" (deal 35)" if attack else ""))
    return v6.augment(text, [], "combat")


class GlossaryV6Tests(unittest.TestCase):
    def test_zero_ascension_keeps_every_legacy_intent_and_prior_renderers_unchanged(self):
        for move in legacy.INTENT_DB:
            text = state().replace("SLIME_BOSS_SLAM (deal 35)", move)
            self.assertEqual(v6.augment(text, [], "combat"), v5.augment(text, [], "combat"), move)
        for phase in ("out_of_combat", "terminal"):
            self.assertEqual(v6.augment("Your HP: 80/80", [], phase),
                             v5.augment("Your HP: 80/80", [], phase))

    def test_numeric_thresholds_replace_old_amounts_for_buffs_and_status_cards(self):
        for move, threshold, before, after in (
                ("MYSTIC_BUFF", 2, "grants 2 Strength", "grants 3 Strength"),
                ("NEMESIS_DEBUFF", 3, "adds 3 Burn", "adds 5 Burn"),
                ("BRONZE_AUTOMATON_BOOST", 4, "gains 3 Strength and 9 Block", "gains 4 Strength and 9 Block"),
                ("BRONZE_AUTOMATON_BOOST", 9, "gains 4 Strength and 9 Block", "gains 4 Strength and 12 Block"),
                ("AWAKENED_ONE_REBIRTH", 9, "revives at 300 HP", "revives at 320 HP"),
                ("BEAR_BEAR_HUG", 17, "Dexterity by 2", "Dexterity by 4"),
                ("LAGAVULIN_SIPHON_SOUL", 18, "by 1 each", "by 2 each"),
                ("SENTRY_BOLT", 18, "adds 2 Dazed", "adds 3 Dazed"),
                ("REPTOMANCER_SUMMON", 18, "summons 1 Dagger", "summons 2 Daggers"),
                ("SLIME_BOSS_GOOP_SPRAY", 19, "adds 3 Slimed", "adds 5 Slimed"),
                ("THE_CHAMP_DEFENSIVE_STANCE", 19, "18 Block and 6 Metallicize", "20 Block and 7 Metallicize")):
            with self.subTest(move=move, threshold=threshold):
                self.assertIn(before, render(move, threshold - 1))
                result = render(move, threshold)
                self.assertIn(after, result)
                self.assertNotIn(before, result)

    def test_attack_riders_keep_damage_and_add_definitions_for_new_debuffs(self):
        for move, ascension, phrase, status in (
                ("FAT_GREMLIN_SMASH", 17, "1 Weak and 1 Frail", "Frail"),
                ("TASKMASTER_SCOURING_WHIP", 18, "3 Wound cards to your discard pile and gains 1 Strength", "Strength"),
                ("TIME_EATER_HEAD_SLAM", 19, "adds 2 Slimed cards", None)):
            with self.subTest(move=move):
                result = render(move, ascension, attack=True)
                line = next(line for line in result.splitlines() if f"intent {move}" in line)
                self.assertIn("(deal 35) (also:", line)
                self.assertIn(phrase, line)
                if status:
                    self.assertEqual(sum(line.startswith(f"  {status}:") for line in result.splitlines()), 1)

    def test_boss_defense_and_debuff_changes_are_visible_before_execution(self):
        guardian = render("THE_GUARDIAN_DEFENSIVE_MODE", 19)
        self.assertIn("no damage; enters Defensive Mode: gains 4 Sharp Hide", guardian)
        self.assertIn("(deals damage back when you attack it)", guardian)
        self.assertIn("gains 32 Block", render("TIME_EATER_HASTE", 19))
        self.assertIn("1 Weak, 1 Vulnerable and 1 Frail", render("TIME_EATER_RIPPLE", 19))
        result = render("DECA_SQUARE_OF_PROTECTION", 19)
        self.assertIn("16 Block and 3 Plated Armor", result)
        self.assertIn("Plated Armor:", result)

    def test_ascenders_bane_explains_opening_hand_and_ethereal_with_blue_candle(self):
        text = state().replace("Defend [Skill]", "Ascender's Bane [Curse]").replace("Ascension: 0", "Ascension: 10")
        for candle in (False, True):
            raw = text.replace("Burning Blood", "Blue Candle") if candle else text
            line = next(line for line in v6.augment(raw, [], "combat").splitlines()
                        if line.startswith("  Ascender's Bane:"))
            self.assertIn("Innate: starts in your opening hand", line)
            self.assertIn("Ethereal: Exhausts", line)
            self.assertEqual("Playable with Blue Candle" in line, candle)
