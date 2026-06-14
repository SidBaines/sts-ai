"""Unit tests for the rollout display model (pure Python, no Streamlit)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_ai.rollout_view import load_rollout, parse_state_text, to_view

STATE_TEXT = (
    "Act 1, floor 6, screen REST_ROOM, room REST, boss The Guardian\n"
    "HP: 54/80, gold: 110\n"
    "Deck: (12): {Strike,Strike,Defend,Bash,Anger,Cleave,}\n"
    "Relics: {Burning Blood:0,Vajra:0,}\n"
    "Potions: Fire Potion, Block Potion\n"
)


def _record() -> dict:
    return {
        "seed": 3,
        "decision_index": 7,
        "state": {
            "act": 1,
            "floor": 6,
            "screen_state": "ScreenState.REST_ROOM",
            "room": "Room.REST",
            "boss": "The Guardian",
            "cur_hp": 54,
            "max_hp": 80,
            "gold": 110,
        },
        "state_text": STATE_TEXT,
        "legal_actions": [
            {"index": 0, "bits": 0, "description": "rest (heal 30% of max HP)"},
            {"index": 1, "bits": 1, "description": "smith (upgrade a card in your deck)"},
        ],
        "selected_action": {"index": 1, "description": "smith (upgrade a card in your deck)"},
        "agent": {
            "action_index": 1,
            "raw_response": "<think>weigh options</think>\n{\"reasoning\": \"upgrade\", \"action_index\": 1}",
            "reasoning": "upgrade",
            "thinking": "weigh options",
            "valid": True,
            "retries": 0,
        },
        "after_state": {"cur_hp": 54, "floor": 6},
    }


class ParseStateTextTest(unittest.TestCase):
    def test_parses_deck_relics_potions(self):
        parsed = parse_state_text(STATE_TEXT)
        self.assertEqual(parsed["deck"][:2], ["Strike", "Strike"])
        self.assertEqual(len(parsed["deck"]), 6)  # trailing comma ignored
        self.assertEqual(parsed["relics"], ["Burning Blood:0", "Vajra:0"])
        self.assertEqual(parsed["potions"], ["Fire Potion", "Block Potion"])

    def test_parses_boss_from_header(self):
        # boss is only in state_text, not the state dict
        self.assertEqual(parse_state_text(STATE_TEXT)["boss"], "The Guardian")

    def test_boss_falls_back_to_state_text_when_absent_from_state(self):
        record = _record()
        del record["state"]["boss"]
        self.assertEqual(to_view(record).boss, "The Guardian")

    def test_potions_none(self):
        parsed = parse_state_text("Potions: none\n")
        self.assertEqual(parsed["potions"], [])

    def test_missing_lines_degrade_gracefully(self):
        parsed = parse_state_text("garbage with no recognizable lines")
        self.assertEqual(parsed, {"deck": [], "relics": [], "potions": [], "boss": ""})


class ToViewTest(unittest.TestCase):
    def test_builds_decision_view(self):
        view = to_view(_record())
        self.assertEqual(view.floor, 6)
        self.assertEqual(view.screen, "REST_ROOM")
        self.assertEqual(view.room, "REST")
        self.assertEqual(view.cur_hp, 54)
        self.assertEqual(view.chosen_index, 1)
        self.assertEqual(len(view.deck), 6)
        self.assertEqual(view.thinking, "weigh options")
        # exactly one action flagged chosen, and it is the smith option
        chosen = [a for a in view.actions if a.chosen]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].index, 1)


class LoadRolloutTest(unittest.TestCase):
    def test_loads_file_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed_3.jsonl"
            path.write_text(json.dumps(_record()) + "\n")
            (Path(d) / "seed_3.error.json").write_text(json.dumps({"message": "boom"}))
            rollout = load_rollout(path)
            self.assertEqual(len(rollout.decisions), 1)
            self.assertEqual(rollout.seed, 3)
            self.assertEqual(rollout.boss, "The Guardian")
            self.assertEqual(rollout.error["message"], "boom")

    def test_missing_file_returns_empty(self):
        rollout = load_rollout("/nonexistent/seed_999.jsonl")
        self.assertEqual(rollout.decisions, [])
        self.assertIsNone(rollout.error)


if __name__ == "__main__":
    unittest.main()
