"""Unit tests for pure hinting helpers."""
from __future__ import annotations

import unittest

from sts_ai.affordances import action_contributes_block, action_is_single_target_lethal
from sts_ai.hinting import (
    BLOCK_HINT,
    LETHAL_HINT,
    HintConfig,
    action_only_raw_response,
    build_hinted_prompt_suffix,
    build_launder_state_text,
    detect_mistake,
    finalize_hinted_decision,
    launder_guardrail_ok,
    mistake_kind_for,
)
from sts_ai.schemas import AgentDecision


def _combat(player_cur_hp=10):
    return {
        "player_cur_hp": player_cur_hp,
        "player_energy": 3,
        "player_block": 0,
        "enemies": [
            {
                "name": "JAW_WORM",
                "cur_hp": 10,
                "max_hp": 40,
                "block": 0,
                "alive": True,
                "intent_damage": 10,
                "intent_hits": 1,
            }
        ],
    }


def _state_text(powers="none", block=0):
    return (
        "Battle turn 1\n"
        f"Player HP: 99/99, block: {block}, energy: 3/3\n"
        f"Player powers: {powers}\n"
        "Enemies:\n  [0] JAW_WORM HP 10/40, block 0, intent ATTACK\n"
        "Hand:\n"
        "  [0] Defend (cost 1)\n"
        "  [1] Strike (cost 1)\n"
        "Piles: draw 1, discard 0, exhaust 0\n"
        "Potions: none\n"
    )


class AffordanceHelperTest(unittest.TestCase):
    def test_action_is_single_target_lethal(self):
        combat = _combat()

        self.assertTrue(
            action_is_single_target_lethal(
                {"description": "play Strike (cost 1) -> JAW_WORM (deal 10)"},
                combat,
            )
        )
        self.assertFalse(
            action_is_single_target_lethal(
                {"description": "play Strike (cost 1) -> JAW_WORM (deal 9)"},
                combat,
            )
        )

    def test_action_contributes_block(self):
        state_text = _state_text()

        self.assertTrue(
            action_contributes_block({"description": "play Defend (cost 1)"}, state_text)
        )
        self.assertFalse(
            action_contributes_block(
                {"description": "play Strike (cost 1) -> JAW_WORM (deal 6)"},
                state_text,
            )
        )
        self.assertFalse(
            action_contributes_block({"description": "end turn"}, state_text)
        )

    def test_action_contributes_block_respects_no_block(self):
        self.assertFalse(
            action_contributes_block(
                {"description": "play Defend (cost 1)"},
                _state_text(powers="No Block 1"),
            )
        )


class DetectMistakeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = HintConfig(enabled=True)
        self.combat = _combat(player_cur_hp=10)
        self.state_text = _state_text()
        self.end_turn = {"index": 1, "description": "end turn"}
        self.lethal = {
            "index": 0,
            "description": "play Strike (cost 1) -> JAW_WORM (deal 10)",
        }

    def test_lethal_available_chosen_not_lethal(self):
        affordances = {"single_target_lethal_available": True}

        self.assertEqual(
            detect_mistake(
                affordances,
                self.end_turn,
                self.combat,
                self.state_text,
                self.cfg,
            ),
            LETHAL_HINT,
        )

    def test_lethal_taken_is_not_mistake(self):
        affordances = {"single_target_lethal_available": True}

        self.assertIsNone(
            detect_mistake(
                affordances,
                self.lethal,
                self.combat,
                self.state_text,
                self.cfg,
            )
        )

    def test_full_block_available_above_threshold_chosen_end_turn(self):
        affordances = {
            "full_block_possible": True,
            "incoming_damage_total": 10,
        }

        self.assertEqual(
            detect_mistake(
                affordances,
                self.end_turn,
                self.combat,
                self.state_text,
                self.cfg,
            ),
            BLOCK_HINT,
        )

    def test_full_block_available_below_threshold_is_not_mistake(self):
        affordances = {
            "full_block_possible": True,
            "incoming_damage_total": 9,
        }

        self.assertIsNone(
            detect_mistake(
                affordances,
                self.end_turn,
                self.combat,
                self.state_text,
                self.cfg,
            )
        )

    def test_lethal_wins_when_both_hints_would_fire(self):
        affordances = {
            "single_target_lethal_available": True,
            "full_block_possible": True,
            "incoming_damage_total": 10,
        }

        self.assertEqual(
            detect_mistake(
                affordances,
                self.end_turn,
                self.combat,
                self.state_text,
                self.cfg,
            ),
            LETHAL_HINT,
        )

    def test_empty_affordances_are_ignored(self):
        self.assertIsNone(
            detect_mistake(
                {},
                self.end_turn,
                self.combat,
                self.state_text,
                self.cfg,
            )
        )


class MistakeKindTest(unittest.TestCase):
    def test_known_hint_kinds(self):
        self.assertEqual(mistake_kind_for(LETHAL_HINT), "lethal")
        self.assertEqual(mistake_kind_for(BLOCK_HINT), "block")

    def test_unknown_hint_raises(self):
        with self.assertRaises(ValueError):
            mistake_kind_for("some future hint")


class LaunderHelpersTest(unittest.TestCase):
    def test_launder_guardrail_ok(self):
        self.assertTrue(launder_guardrail_ok(AgentDecision(action_index=2), 2))
        self.assertFalse(launder_guardrail_ok(AgentDecision(action_index=1), 2))
        self.assertFalse(
            launder_guardrail_ok(AgentDecision(action_index=2, valid=False), 2)
        )

    def test_trait_neutral_text(self):
        denylist = {
            "risky",
            "risk",
            "dangerous",
            "danger",
            "might die",
            "play safe",
            "safe",
            "rest",
            "aggressive",
            "cautious",
        }
        texts = [
            LETHAL_HINT,
            BLOCK_HINT,
            build_hinted_prompt_suffix(LETHAL_HINT),
            build_hinted_prompt_suffix(BLOCK_HINT),
            build_launder_state_text(
                "Battle turn 1\nPlayer powers: none\n",
                {
                    "index": 0,
                    "description": "play Strike (cost 1) -> JAW_WORM (deal 10)",
                },
            ),
        ]

        for text in texts:
            lower = text.lower()
            for denied in denylist:
                self.assertNotIn(denied, lower)


class FinalizeHintedDecisionTest(unittest.TestCase):
    def setUp(self):
        self.normal = AgentDecision(
            action_index=0,
            raw_response="normal raw",
            reasoning="normal reason",
            thinking="normal thought",
            retries=1,
            metadata={"source": "normal"},
        )
        self.hinted = AgentDecision(
            action_index=2,
            raw_response="hinted raw",
            reasoning="hinted reason",
            thinking="hinted thought",
            valid=True,
            retries=2,
            metadata={"source": "hinted"},
        )

    def test_laundered_path(self):
        laundered = AgentDecision(
            action_index=2,
            raw_response="laundered raw",
            reasoning="laundered reason",
            thinking="laundered thought",
        )

        final = finalize_hinted_decision(
            normal_decision=self.normal,
            hinted_decision=self.hinted,
            laundered_decision=laundered,
            hint_text=LETHAL_HINT,
            mistake_kind="lethal",
            cfg=HintConfig(enabled=True),
        )

        self.assertEqual(final.action_index, 2)
        self.assertEqual(final.raw_response, "laundered raw")
        self.assertEqual(final.reasoning, "laundered reason")
        self.assertEqual(final.thinking, "laundered thought")
        self.assertTrue(final.valid)
        self.assertEqual(final.retries, 0)
        self.assertEqual(final.metadata["hint"]["launder_outcome"], "laundered")

    def test_action_only_fallback(self):
        hinted = AgentDecision(
            action_index=2,
            raw_response=(
                "<|channel|>thought\n"
                "Hinted native reasoning before fallback.\n"
                "<|channel|>final\n"
                '{"action_index": 2}'
            ),
            reasoning="hinted reason",
            thinking="hinted thought",
            valid=True,
            retries=2,
            metadata={
                "source": "hinted",
                "reasoning_format": "gemma_thought",
            },
        )

        final = finalize_hinted_decision(
            normal_decision=self.normal,
            hinted_decision=hinted,
            laundered_decision=None,
            hint_text=BLOCK_HINT,
            mistake_kind="block",
            cfg=HintConfig(enabled=True, on_launder_fail="action_only"),
        )

        self.assertEqual(final.action_index, 2)
        self.assertEqual(final.raw_response, action_only_raw_response(2))
        self.assertEqual(final.reasoning, "")
        self.assertEqual(final.thinking, "")
        self.assertTrue(final.valid)
        self.assertEqual(final.retries, 0)
        self.assertEqual(
            final.metadata["hint"]["launder_outcome"],
            "fallback_action_only",
        )
        self.assertEqual(final.metadata["reasoning_format"], "gemma_thought")
        self.assertNotIn("<|channel", final.raw_response)

    def test_drop_fallback(self):
        final = finalize_hinted_decision(
            normal_decision=self.normal,
            hinted_decision=self.hinted,
            laundered_decision=None,
            hint_text=BLOCK_HINT,
            mistake_kind="block",
            cfg=HintConfig(enabled=True, on_launder_fail="drop"),
        )

        self.assertEqual(final.action_index, self.normal.action_index)
        self.assertEqual(final.raw_response, self.normal.raw_response)
        self.assertEqual(final.metadata["hint"]["launder_outcome"], "drop")


if __name__ == "__main__":
    unittest.main()
