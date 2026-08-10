from __future__ import annotations

import unittest

from sts_ai.prompting import ACTION_ONLY_OUTPUT, render_action_prompt
from sts_ai.schemas import LegalAction


class RenderActionPromptTest(unittest.TestCase):
    def setUp(self):
        self.state_text = "Player HP: 42/80, block: 0, energy: 3/3"
        self.actions = [
            LegalAction(index=0, bits=1, description="play Strike -> Jaw Worm"),
            LegalAction(index=1, bits=2, description="end turn"),
        ]

    def test_induce_reasoning_adds_think_instruction_and_keeps_prompt_content(self):
        prompt = render_action_prompt(
            self.state_text,
            self.actions,
            induce_reasoning=True,
        )

        self.assertIn("think step by step", prompt)
        self.assertIn("<think>...</think>", prompt)
        self.assertIn("</think>", prompt)
        self.assertIn("Do not use markdown fences", prompt)
        self.assertIn("Return exactly one JSON object with this schema:", prompt)
        self.assertIn(
            '{"reasoning": "brief private reasoning", "action_index": 0}',
            prompt,
        )
        self.assertIn("Valid action_index values are: 0, 1.", prompt)
        self.assertIn("do not use hand, enemy, deck, or map indices", prompt)
        self.assertIn("LEGAL ACTIONS\n0: play Strike -> Jaw Worm\n1: end turn\n", prompt)

    def test_default_matches_explicit_false(self):
        self.assertEqual(
            render_action_prompt(self.state_text, self.actions),
            render_action_prompt(
                self.state_text,
                self.actions,
                induce_reasoning=False,
            ),
        )

    def test_default_prompt_remains_byte_compatible(self):
        expected = (
            "You are playing Slay the Spire. Choose one legal action from the list. "
            "Use the game state and action descriptions to make the strongest choice you can.\n\n"
            "Return exactly one JSON object with this schema:\n"
            '{"reasoning": "brief private reasoning", "action_index": 0}\n\n'
            "Valid action_index values are: 0, 1. Use only these LEGAL ACTIONS indices; "
            "do not use hand, enemy, deck, or map indices as action_index.\n\n"
            f"GAME STATE\n{self.state_text}\n\n"
            "LEGAL ACTIONS\n0: play Strike -> Jaw Worm\n1: end turn\n"
        )

        self.assertEqual(render_action_prompt(self.state_text, self.actions), expected)

    def test_action_only_contract_has_matching_schema_and_no_reasoning_field(self):
        prompt = render_action_prompt(
            self.state_text,
            self.actions,
            output_contract=ACTION_ONLY_OUTPUT,
        )

        self.assertIn('Return exactly one JSON object with this schema:\n{"action_index": 0}', prompt)
        self.assertNotIn('"reasoning"', prompt)
        self.assertIn("Valid action_index values are: 0, 1.", prompt)

    def test_unknown_output_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_contract"):
            render_action_prompt(
                self.state_text,
                self.actions,
                output_contract="action-ish",
            )


if __name__ == "__main__":
    unittest.main()
