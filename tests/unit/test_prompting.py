from __future__ import annotations

import unittest

from sts_ai.prompting import render_action_prompt
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

        self.assertIn("think briefly", prompt)
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


if __name__ == "__main__":
    unittest.main()
