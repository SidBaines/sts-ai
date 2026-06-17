"""Unit tests for the effect/status glossary augmentation (pure Python)."""
from __future__ import annotations

import glob
import json
import re
import unittest

from sts_ai import glossary
from sts_ai.glossary import (
    CARD_DB,
    INTENT_DB,
    RELIC_DB,
    STATUS_DB,
    _card_definition,
    augment,
    intent_effect,
    relic_definition,
    status_definition,
)
from sts_ai.rollout_view import to_view

COMBAT_TEXT = (
    "Battle turn 1\n"
    "Player HP: 56/80, block: 0, energy: 3/3\n"
    "Player powers: Strength 2, Weak 1\n"
    "Enemies:\n"
    "  [0] CULTIST HP 50/50, block 0, intent CULTIST_INCANTATION\n"
    "  [1] JAW_WORM HP 40/44, block 0, intent JAW_WORM_CHOMP (deal 11), Vulnerable 2\n"
    "Hand:\n"
    "  [0] Strike (cost 1)\n"
    "  [1] Bash (cost 2)\n"
    "  [2] Defend+ (cost 1)\n"
    "Piles: draw 3, discard 0, exhaust 0\n"
    "Potions: none\n"
)


class StatusDefinitionTest(unittest.TestCase):
    def test_kind_phrasing(self):
        self.assertIn("turns remaining", status_definition("Weak"))
        self.assertIn("persistent", status_definition("Strength"))
        self.assertIn("per-turn", status_definition("Metallicize"))

    def test_unknown_status_is_none(self):
        self.assertIsNone(status_definition("Nonexistent"))


class CardDefinitionTest(unittest.TestCase):
    def test_lookup_and_upgrade_marker(self):
        self.assertTrue(_card_definition("Strike").startswith("Strike: "))
        # an upgraded card resolves to its base entry
        self.assertTrue(_card_definition("Defend+").startswith("Defend: "))

    def test_unknown_card_is_none(self):
        self.assertIsNone(_card_definition("Totally Made Up Card"))


class AugmentCombatTest(unittest.TestCase):
    def setUp(self):
        self.out = augment(COMBAT_TEXT, [], "combat")

    def test_non_attacking_intent_labelled_with_effect(self):
        # A known non-attacking move now spells out its effect (not bare "no attack").
        self.assertIn("intent CULTIST_INCANTATION (no damage; buffs itself with Ritual", self.out)
        self.assertNotIn("CULTIST_INCANTATION (no attack)", self.out)

    def test_intent_effect_statuses_added_to_key(self):
        # Ritual is referenced only by the Cultist's intent (not on the board), but
        # is still defined in the KEY so the model knows what it portends.
        key = self.out[self.out.index("-- KEY"):]
        self.assertIn("Ritual:", key)

    def test_attacking_intent_keeps_damage_and_no_no_attack(self):
        # the pure-attack enemy keeps its (deal …) and gets no "(no attack)"/"(also:"
        self.assertIn("intent JAW_WORM_CHOMP (deal 11)", self.out)
        jaw_line = next(l for l in self.out.splitlines() if "JAW_WORM_CHOMP" in l)
        self.assertNotIn("(no attack)", jaw_line)
        self.assertNotIn("(also:", jaw_line)

    def test_unknown_nonattack_falls_back_to_no_attack(self):
        out = augment(
            "Enemies:\n  [0] FOE HP 5/5, block 0, intent FOE_GLARE\n"
            "Hand: empty\nPiles: draw 1, discard 0, exhaust 0\nPotions: none\n",
            [], "combat",
        )
        self.assertIn("intent FOE_GLARE (no attack)", out)

    def test_attack_rider_appended(self):
        text = (
            "Enemies:\n  [0] SPIKE_SLIME_M HP 28/28, block 0, intent SPIKE_SLIME_M_FLAME_TACKLE (deal 8)\n"
            "Hand: empty\nPiles: draw 5, discard 0, exhaust 0\nPotions: none\n"
        )
        out = augment(text, [], "combat")
        self.assertIn("(deal 8) (also: adds 1 Slimed card to your discard pile)", out)

    def test_non_attack_hp_loss_does_not_claim_no_damage(self):
        text = (
            "Enemies:\n  [0] EXPLODER HP 30/30, block 0, intent EXPLODER_EXPLODE\n"
            "Hand: empty\nPiles: draw 5, discard 0, exhaust 0\nPotions: none\n"
        )
        out = augment(text, [], "combat")
        self.assertIn("intent EXPLODER_EXPLODE (no attack; deals 30 damage to you, then dies)", out)
        self.assertNotIn("EXPLODER_EXPLODE (no damage;", out)

    def test_incoming_damage_note_sums_intents(self):
        # Jaw Worm deals 11; Cultist deals 0 -> total 11.
        self.assertIn("Incoming attack damage this turn: 11 (before your Block)", self.out)

    def test_cant_play_note_for_zero_energy_trap(self):
        text = (
            "Battle turn 0\nPlayer HP: 80/80, block: 5, energy: 0/3\nPlayer powers: none\n"
            "Enemies:\n  [0] JAW_WORM HP 25/42, block 0, intent JAW_WORM_CHOMP (deal 11)\n"
            "Hand:\n  [0] Strike (cost 1)\n  [1] Strike (cost 1)\n"
            "Piles: draw 5, discard 0, exhaust 0\nPotions: none\n"
        )
        out = augment(text, [{"index": 0, "description": "end turn"}], "combat")
        self.assertIn("You cannot play any card right now", out)

    def test_no_cant_play_note_when_a_play_action_exists(self):
        out = augment(
            COMBAT_TEXT,
            [{"index": 0, "description": "play Strike (cost 1) -> CULTIST (deal 6)"},
             {"index": 1, "description": "end turn"}],
            "combat",
        )
        self.assertNotIn("You cannot play any card right now", out)

    def test_key_block_defines_active_statuses(self):
        key = self.out[self.out.index("-- KEY"):]
        for status in ("Strength:", "Weak:", "Vulnerable:"):
            self.assertIn(status, key)

    def test_key_block_defines_hand_cards(self):
        key = self.out[self.out.index("-- KEY"):]
        self.assertIn("Strike:", key)
        self.assertIn("Bash:", key)
        self.assertIn("Defend:", key)  # Defend+ resolved to base

    def test_no_key_when_nothing_recognised(self):
        bare = "Battle turn 0\nPlayer powers: none\nEnemies:\nHand: empty\nPiles: draw 5, discard 0, exhaust 0\nPotions: none\n"
        self.assertNotIn("-- KEY", augment(bare, [], "combat"))

    def test_unknown_names_do_not_crash_or_appear(self):
        text = (
            "Player powers: Bogusbuff 3\n"
            "Enemies:\n  [0] FOE HP 5/5, block 0, intent FOE_GLARE\n"
            "Hand:\n  [0] Mysterycard (cost 1)\nPiles: draw 1, discard 0, exhaust 0\nPotions: none\n"
        )
        out = augment(text, [], "combat")
        self.assertIn("intent FOE_GLARE (no attack)", out)
        self.assertNotIn("Bogusbuff", out[out.index("-- KEY"):] if "-- KEY" in out else "")
        self.assertNotIn("Mysterycard:", out)


class RelicPotionKeyTest(unittest.TestCase):
    def test_ooc_key_defines_relics_and_potions(self):
        state_text = (
            "Act 1, floor 8, screen SHOP_ROOM, room SHOP, end-of-act boss The Guardian\n"
            "Your HP: 34/80, gold: 290\n"
            "Deck: (10): {Strike,}\n"
            "Relics: {Burning Blood:0,Mercury Hourglass:0,}\n"
            "Potions: Fire Potion, Energy Potion\n"
        )
        out = augment(state_text, [], "out_of_combat")
        key = out[out.index("-- KEY"):]
        self.assertIn("Burning Blood: At the end of combat, heal 6 HP.", key)
        self.assertIn("Mercury Hourglass:", key)
        self.assertIn("Fire Potion: Deal 20 damage", key)
        self.assertIn("Energy Potion: Gain 2 energy.", key)

    def test_unknown_relic_is_skipped(self):
        state_text = (
            "Act 1, floor 1, screen MAP_SCREEN, room none, end-of-act boss The Guardian\n"
            "Your HP: 80/80, gold: 99\n"
            "Relics: {Totally Made Up Relic:0,}\n"
            "Potions: none\n"
        )
        out = augment(state_text, [], "out_of_combat")
        # The unknown relic is skipped, so no KEY definition block is produced
        # (the name still appears in the input Relics: line, which is fine).
        self.assertNotIn("-- KEY", out)

    def test_combat_key_defines_potions(self):
        combat_text = (
            "Battle turn 1\nPlayer HP: 50/80, block: 0, energy: 3/3\nPlayer powers: none\n"
            "Enemies:\n  [0] JAW_WORM HP 40/44, block 0, intent JAW_WORM_CHOMP (deal 11)\n"
            "Hand:\n  [0] Strike (cost 1)\nPiles: draw 3, discard 0, exhaust 0\n"
            "Potions: Fire Potion\n"
        )
        out = augment(combat_text, [], "combat")
        key = out[out.index("-- KEY"):]
        self.assertIn("Fire Potion: Deal 20 damage", key)


class Act23GlossaryCoverageTest(unittest.TestCase):
    def test_act2_and_act3_intent_effects(self):
        for move in (
            "CHOSEN_DRAIN",
            "BRONZE_ORB_STASIS",
            "SPIRE_GROWTH_CONSTRICT",
            "TIME_EATER_RIPPLE",
            "AWAKENED_ONE_REBIRTH",
        ):
            self.assertIsNotNone(intent_effect(move), move)
            self.assertNotEqual(intent_effect(move), "", move)

    def test_new_relic_definitions(self):
        for relic in ("White Beast Statue", "Nilrys Codex", "Philosophers Stone", "Pandoras Box"):
            self.assertTrue(relic_definition(relic).startswith(f"{relic}: "), relic)

    def test_new_status_definition(self):
        desc = status_definition("Constricted")
        self.assertIn("lose that much HP", desc)
        self.assertIn("per-turn amount", desc)

    def test_new_entries_keep_neutral_tone(self):
        banned = re.compile(r"\b(dangerous|powerful|be careful|prioritize|threat)\b", re.I)
        new_intents = (
            "CHOSEN_DRAIN",
            "SNECKO_PERPLEXING_GLARE",
            "TIME_EATER_RIPPLE",
            "WRITHING_MASS_IMPLANT",
            "AWAKENED_ONE_REBIRTH",
        )
        new_relics = ("White Beast Statue", "Nilrys Codex", "Philosophers Stone", "Pandoras Box")
        text = "\n".join([INTENT_DB[k] for k in new_intents] + [RELIC_DB[k] for k in new_relics])
        self.assertIsNone(banned.search(text))


class AugmentOocTest(unittest.TestCase):
    def test_explains_offered_cards_stripping_tags_and_price(self):
        actions = [
            {"index": 0, "description": "take gold 11g"},
            {"index": 1, "description": "take card Offering [Skill, Rare]"},
            {"index": 2, "description": "buy card Twin Strike [Attack, Common] for 54g"},
        ]
        out = augment("Act 1, floor 5 ...\n", actions, "out_of_combat")
        key = out[out.index("-- KEY"):]
        self.assertIn("Offering:", key)
        self.assertIn("Twin Strike:", key)
        self.assertNotIn("gold", key)  # non-card choices ignored
        self.assertNotIn("[Skill", key)  # tag stripped from the looked-up name


class RegressionTest(unittest.TestCase):
    def test_augmented_state_text_still_parses_hand(self):
        # The KEY block is appended after Piles/Potions, so the combat-board hand
        # parser (rollout_view) must still see exactly the 3 hand cards.
        record = {
            "phase": "combat",
            "state": {
                "screen_state": "ScreenState.BATTLE",
                "combat": {
                    "turn": 1, "player_cur_hp": 56, "player_max_hp": 80,
                    "player_block": 0, "player_energy": 3,
                    "enemies": [
                        {"index": 0, "name": "CULTIST", "cur_hp": 50, "max_hp": 50,
                         "block": 0, "intent": "CULTIST_INCANTATION", "alive": True},
                    ],
                },
            },
            "state_text": augment(COMBAT_TEXT, [], "combat"),
            "legal_actions": [],
            "selected_action": {"index": 0, "description": "end turn"},
            "agent": {"action_index": 0},
            "after_state": {},
        }
        cv = to_view(record).combat
        self.assertEqual([c.name for c in cv.hand], ["Strike", "Bash", "Defend+"])

    def test_combat_notes_do_not_break_combat_view_parsing(self):
        # The incoming-damage and can't-play notes are appended after Potions; the
        # combat-board parser (rollout_view) must still read hand + enemies cleanly.
        combat_text = (
            "Battle turn 0\nPlayer HP: 80/80, block: 5, energy: 0/3\nPlayer powers: none\n"
            "Enemies:\n  [0] JAW_WORM HP 25/42, block 0, intent JAW_WORM_CHOMP (deal 11)\n"
            "Hand:\n  [0] Strike (cost 1)\n  [1] Strike (cost 1)\n"
            "Piles: draw 5, discard 0, exhaust 0\nPotions: none\n"
        )
        actions = [{"index": 0, "description": "end turn"}]
        record = {
            "phase": "combat",
            "state": {
                "screen_state": "ScreenState.BATTLE",
                "combat": {
                    "turn": 0, "player_cur_hp": 80, "player_max_hp": 80,
                    "player_block": 5, "player_energy": 0,
                    "enemies": [
                        {"index": 0, "name": "JAW_WORM", "cur_hp": 25, "max_hp": 42,
                         "block": 0, "intent": "JAW_WORM_CHOMP", "alive": True},
                    ],
                },
            },
            "state_text": augment(combat_text, actions, "combat"),
            "legal_actions": actions,
            "selected_action": {"index": 0, "description": "end turn"},
            "agent": {"action_index": 0},
            "after_state": {},
        }
        cv = to_view(record).combat
        self.assertEqual([c.name for c in cv.hand], ["Strike", "Strike"])
        self.assertEqual([e.name for e in cv.enemies], ["JAW_WORM"])
        self.assertTrue(cv.chosen_is_end_turn)


class CoverageTest(unittest.TestCase):
    """Every card/status that actually appears in our recorded traces must be in
    the DBs (skips if no traces are present, so a fresh clone stays green)."""

    def test_dbs_cover_names_seen_in_traces(self):
        files = glob.glob("data/rollouts/enriched_qwen_seed*.jsonl")
        if not files:
            self.skipTest("no enriched_qwen rollout traces present")

        cards: set[str] = set()
        statuses: set[str] = set()
        for path in files:
            with open(path) as handle:
                lines = handle.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                state_text = record.get("state_text", "")
                cards.update(glossary._hand_card_names(state_text))
                cards.update(glossary._ooc_card_names(record.get("legal_actions", [])))
                for enemy in record.get("state", {}).get("combat", {}).get("enemies", []):
                    for field, label in (("vulnerable", "Vulnerable"), ("weak", "Weak"),
                                         ("strength", "Strength"), ("poison", "Poison")):
                        if enemy.get(field):
                            statuses.add(label)
                for raw in state_text.splitlines():
                    if raw.startswith("Player powers:"):
                        statuses |= glossary._scan_status_names(raw)

        missing_cards = sorted(c for c in cards if c.rstrip("+") not in CARD_DB)
        missing_statuses = sorted(s for s in statuses if s not in STATUS_DB)
        self.assertEqual(missing_cards, [], f"cards seen in traces but missing from CARD_DB: {missing_cards}")
        self.assertEqual(missing_statuses, [], f"statuses missing from STATUS_DB: {missing_statuses}")


if __name__ == "__main__":
    unittest.main()
