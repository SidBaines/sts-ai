"""Regressions for the evidence-backed 6 September combat template repairs."""
from __future__ import annotations

import copy
import unittest

from sts_env._backend import glossary, glossary_v4


COMBAT_TEXT = (
    "Battle turn 0\n"
    "Player HP: 56/80, block: 0, energy: 3/3\n"
    "Player powers: none\n"
    "Enemies:\n"
    "  [0] SENTRY HP 30/40, block 0, intent SENTRY_BOLT, Artifact 1\n"
    "Hand:\n"
    "  [0] Strike [Attack] (cost 1)\n"
    "  [1] Defend+ [Skill] (cost 1)\n"
    "Piles: draw 3, discard 0, exhaust 0\n"
    "Relics: none\n"
    "Potions (capacity 2): [0] empty, [1] empty\n"
)
PLAY = [{"description": "play Strike [Attack] (cost 1) -> SENTRY (deal 6)"}]


class CombatV4RegressionTest(unittest.TestCase):
    def render(self, text: str = COMBAT_TEXT, actions: list[dict] | None = None) -> str:
        return glossary_v4.augment(text, PLAY if actions is None else actions, "combat")

    def test_action_is_explicitly_distinct_from_turn(self):
        output = self.render()
        self.assertIn("one legal action, not a whole turn", output)
        self.assertIn("multiple cards in the same turn, one action per response", output)
        self.assertIn("end turn starts end-of-turn effects and the enemy phase", output)

    def test_upgrade_values_are_distinct_from_current_attack_damage(self):
        output = self.render()
        self.assertIn("a + marks an upgraded card", output)
        self.assertIn("6 (9) mean unmodified base (upgraded) values", output)
        self.assertIn("Inline (deal N) annotations on actions give the current", output)
        self.assertIn("Battle turn numbers start at 0", output)

    def test_intent_identifier_excludes_trailing_comma_and_preserves_power(self):
        output = self.render()
        self.assertIn(
            "intent SENTRY_BOLT (no damage; adds 2 Dazed cards to your discard pile), Artifact 1",
            output,
        )
        self.assertIn("Artifact: negates", output)

    def test_nonattack_intent_references_define_future_debuff(self):
        text = COMBAT_TEXT.replace("SENTRY_BOLT", "LAGAVULIN_SIPHON_SOUL")
        output = self.render(text)
        self.assertIn(glossary.INTENT_DB["LAGAVULIN_SIPHON_SOUL"], output)
        self.assertIn("Strength:", output)
        self.assertIn("Dexterity:", output)

    def test_known_intents_with_and_without_powers_have_same_effect(self):
        # Exercise every existing move, not just the reported Sentry example.
        for move, effect in glossary.INTENT_DB.items():
            with self.subTest(move=move):
                line = f"  [0] FOE HP 10/10, block 0, intent {move}"
                for tail in ("", ", Artifact 1", " (deal 6), Strength 1"):
                    annotated, _ = glossary_v4._label_intent(line + tail)
                    self.assertIn(effect, annotated)
                    self.assertIn(tail, annotated)

    def test_nob_skull_bash_retains_damage_and_adds_vulnerable(self):
        text = COMBAT_TEXT.replace(
            "SENTRY_BOLT, Artifact 1",
            "GREMLIN_NOB_SKULL_BASH (deal 6), Enrage 2",
        )
        output = self.render(text)
        self.assertIn("(deal 6), Enrage 2 (also: applies 2 Vulnerable to you)", output)
        self.assertIn("Vulnerable:", output)
        self.assertIn("Enrage:", output)

    def test_unknown_attack_is_not_assigned_an_invented_rider(self):
        line = "  [0] FOE HP 10/10, block 0, intent FUTURE_ATTACK (deal 3), Artifact 1"
        self.assertEqual((line, set()), glossary_v4._label_intent(line))

    def test_ten_audited_relic_mechanics_are_defined(self):
        expected = {
            "Blue Candle": "loses 1 HP and Exhausts it",
            "Captains Wheel": "Battle turn 2), gain 18 Block",
            "Charons Ashes": "Exhausted, deal 3 damage",
            "Gambling Chip": "discard the selected cards and draw that many replacements",
            "Ginger": "Prevents Weak",
            "Incense Burner": "At 6, reset the counter to 0 and gain 1 Intangible",
            "Ornamental Fan": "Every third Attack played in a turn grants 4 Block",
            "Paper Phrog": "75% more attack damage instead of 50%",
            "Stone Calendar": "Battle turn 6), deal 52 damage",
            "Turnip": "Prevents Frail",
        }
        for name, effect in expected.items():
            with self.subTest(relic=name):
                text = COMBAT_TEXT.replace("Relics: none", f"Relics: {name} [counter 4, live]")
                self.assertIn(effect, self.render(text))

    def test_blue_candle_curse_rule_has_no_unplayable_contradiction(self):
        text = COMBAT_TEXT.replace(
            "Strike [Attack] (cost 1)", "Injury [Curse] (cost -2)",
        ).replace("Relics: none", "Relics: Blue Candle")
        output = self.render(text)
        self.assertIn("Injury: Curse. Playable with Blue Candle", output)
        self.assertNotIn("Injury: Curse. Unplayable", output)
        self.assertIn("Injury: Curse. Unplayable", self.render(text.replace("Blue Candle", "none")))

    def test_inactive_relic_statuses_are_defined_before_they_trigger(self):
        expected = {
            "Incense Burner [counter 4, live]": "Intangible: reduces ALL damage and HP loss it takes to 1",
            "Ginger": "Weak: the affected creature deals 25% less attack damage",
            "Turnip": "Frail: the affected creature gains 25% less Block from cards",
            "Paper Phrog": "Vulnerable: affected enemies take 75% more attack damage",
        }
        for relic, definition in expected.items():
            with self.subTest(relic=relic):
                text = COMBAT_TEXT.replace("Relics: none", f"Relics: {relic}")
                self.assertIn("Player powers: none", text)
                self.assertIn(definition, self.render(text))

    def test_paper_phrog_vulnerable_rule_distinguishes_player_from_enemy(self):
        text = COMBAT_TEXT.replace("Player powers: none", "Player powers: Vulnerable 2")
        text = text.replace("Relics: none", "Relics: Paper Phrog")
        output = self.render(text)
        self.assertIn("affected enemies take 75% more attack damage", output)
        self.assertIn("you take 50% more attack damage", output)
        self.assertNotIn("the affected creature takes 50%", output)
        self.assertIn(
            "you take 25% more attack damage",
            self.render(text.replace("Relics: Paper Phrog", "Relics: Paper Phrog, Odd Mushroom")),
        )

    def test_duplication_remains_defined_with_empty_potion_slots(self):
        text = COMBAT_TEXT.replace("Player powers: none", "Player powers: Duplication 1")
        output = self.render(text)
        self.assertIn("Duplication: Your next card is played twice", output)
        self.assertIn("Each original card played consumes 1 charge", output)
        self.assertIn("after the enemy phase, 1 unused charge is removed", output)
        self.assertNotIn("Duplication Potion:", output)

    def test_flame_barrier_upgrade_has_correct_retaliation(self):
        text = COMBAT_TEXT.replace("Strike [Attack]", "Flame Barrier [Skill]")
        text = text.replace("Defend+ [Skill]", "Flame Barrier+ [Skill]")
        output = self.render(text)
        self.assertIn("Flame Barrier: Gain 12 base Block. This turn, deal 4 damage", output)
        self.assertIn("Flame Barrier+: Gain 16 base Block. This turn, deal 6 damage", output)

    def test_selection_does_not_emit_insufficient_energy_warning(self):
        text = COMBAT_TEXT + "(card select in progress)\nSelection type: EXHAUST_ONE\n"
        output = self.render(text, [{"description": "select card for EXHAUST_ONE: Strike"}])
        self.assertIn("A card selection is pending", output)
        self.assertNotIn("not enough energy", output)
        self.assertNotIn("You cannot play any card right now", output)

    def test_gamble_selection_explains_discard_and_none(self):
        text = COMBAT_TEXT + "(card select in progress)\nSelection type: GAMBLE\n"
        output = self.render(text, [{"description": "keep hand (discard no cards)"}])
        self.assertIn("GAMBLE means discard the selected cards", output)
        self.assertIn("Selecting none keeps your hand and completes this selection", output)
        self.assertNotIn("not enough energy", output)

    def test_normal_playability_warning_is_retained(self):
        output = self.render(actions=[{"description": "end turn"}])
        self.assertIn("You cannot play any card right now", output)

    def test_passes_supplied_v4_arithmetic_through_in_order(self):
        output = glossary_v4.augment(COMBAT_TEXT, PLAY, "combat", turn_math_lines=["restricted estimate"])
        self.assertLess(output.index("Incoming attack damage"), output.index("restricted estimate"))
        self.assertLess(output.index("restricted estimate"), output.index("-- KEY"))

    def test_legacy_output_and_reference_tables_are_unchanged(self):
        old_output = glossary.augment(COMBAT_TEXT, PLAY, "combat")
        old_tables = copy.deepcopy((glossary.STATUS_DB, glossary.CARD_DB, glossary.RELIC_DB, glossary.INTENT_DB))
        self.render()
        self.assertEqual(old_output, glossary.augment(COMBAT_TEXT, PLAY, "combat"))
        self.assertEqual(old_tables, (glossary.STATUS_DB, glossary.CARD_DB, glossary.RELIC_DB, glossary.INTENT_DB))
        # Confirm the compatibility arm still has the exact audited punctuation
        # behavior, even after the repaired renderer has run in this process.
        self.assertIn("SENTRY_BOLT, (no attack) Artifact 1", old_output)

    def test_noncombat_delegates_without_changes(self):
        text = "Your HP: 40/80\nRelics: Ginger\nPotions: none"
        actions = [{"description": "take card Strike [Attack, Basic]"}]
        self.assertEqual(
            glossary.augment(text, actions, "reward"),
            glossary_v4.augment(text, actions, "reward"),
        )


if __name__ == "__main__":
    unittest.main()
