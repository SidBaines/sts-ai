"""Smoke tests for the pure board-HTML builders (no simulator, no Streamlit)."""
from __future__ import annotations

import unittest

from sts_ai.rollout_view import to_view
from sts_ai.rollout_view_html import BOARD_CSS, render_decision_html


def _ooc_record() -> dict:
    return {
        "world_seed": 3,
        "decision_index": 0,
        "phase": "out_of_combat",
        "state": {"act": 1, "floor": 2, "screen_state": "REWARDS", "room": "MONSTER",
                  "cur_hp": 70, "max_hp": 80, "gold": 99, "boss": "Hexaghost"},
        "state_text": "Your HP: 70/80\nDeck: {Strike, Strike, Defend,}\nRelics: {Burning Blood:0,}\nPotions: none\nboss Hexaghost",
        "legal_actions": [
            {"index": 0, "bits": 1, "description": "take card Anger [Attack, Common]"},
            {"index": 1, "bits": 2, "description": "skip rewards / proceed"},
        ],
        "selected_action": {"index": 0, "bits": 1, "description": "take card Anger [Attack, Common]"},
        "agent": {"action_index": 0, "reasoning": "grab the attack", "valid": True, "raw_response": "{}"},
        "after_state": {"cur_hp": 70, "floor": 2},
    }


def _combat_record() -> dict:
    return {
        "world_seed": 3,
        "decision_index": 5,
        "phase": "combat",
        "state": {
            "act": 1, "floor": 3, "screen_state": "BATTLE", "room": "MONSTER",
            "cur_hp": 64, "max_hp": 80, "gold": 99, "boss": "Hexaghost",
            "combat": {
                "turn": 1, "player_cur_hp": 64, "player_max_hp": 80, "player_block": 0,
                "player_energy": 3,
                "enemies": [
                    {"index": 0, "name": "JAW_WORM", "cur_hp": 40, "max_hp": 44, "block": 0,
                     "intent": "JAW_WORM_CHOMP", "alive": True},
                ],
            },
        },
        "state_text": (
            "Player powers: none\nenergy: 3/3\n"
            "Hand:\n[0] Strike (cost 1)\n[1] Defend (cost 1)\n"
            "Piles: draw 3, discard 0, exhaust 0\nPotions: none"
        ),
        "legal_actions": [
            {"index": 0, "bits": 1, "description": "play Strike (cost 1) -> JAW_WORM (deal 6)"},
            {"index": 1, "bits": 2, "description": "play Defend (cost 1)"},
            {"index": 2, "bits": 4, "description": "end turn"},
        ],
        "selected_action": {"index": 0, "bits": 1, "description": "play Strike (cost 1) -> JAW_WORM (deal 6)"},
        "agent": {"action_index": 0, "reasoning": "chip the worm", "thinking": "I should attack now.",
                  "valid": True, "raw_response": "{}"},
        "after_state": {"cur_hp": 64, "floor": 3},
    }


class RenderDecisionHtmlTest(unittest.TestCase):
    def test_ooc_board_renders(self):
        dv = to_view(_ooc_record())
        out = render_decision_html(dv)
        self.assertIn("sts-board", out)
        self.assertIn("Hexaghost", out)
        self.assertIn("take card Anger", out)
        self.assertIn("sts-action chosen", out)  # the chosen action is highlighted
        self.assertIn("grab the attack", out)  # reasoning shown

    def test_combat_board_renders(self):
        dv = to_view(_combat_record())
        out = render_decision_html(dv)
        self.assertIn("⚔️ Combat", out)
        self.assertIn("Jaw Worm", out)  # prettified enemy name
        self.assertIn("Strike", out)  # hand card tile
        self.assertIn("Thinking", out)  # thinking <details>
        self.assertIn("🎯 target", out)  # targeted enemy highlighted

    def test_board_css_nonempty(self):
        self.assertIn(".sts-card", BOARD_CSS)


if __name__ == "__main__":
    unittest.main()
