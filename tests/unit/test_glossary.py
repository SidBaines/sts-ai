"""Unit tests for the effect/status glossary augmentation (pure Python)."""
from __future__ import annotations

import glob
import json
import unittest

from sts_ai import glossary
from sts_ai.glossary import (
    CARD_DB,
    STATUS_DB,
    _card_definition,
    augment,
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

    def test_non_attacking_intent_labelled(self):
        self.assertIn("intent CULTIST_INCANTATION (no attack)", self.out)

    def test_attacking_intent_not_labelled(self):
        # the attacking enemy keeps its (deal …) and gets no "(no attack)"
        self.assertIn("intent JAW_WORM_CHOMP (deal 11)", self.out)
        jaw_line = next(l for l in self.out.splitlines() if "JAW_WORM_CHOMP" in l)
        self.assertNotIn("(no attack)", jaw_line)

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
