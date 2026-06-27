"""Unit tests for the framing / prompt-template manager (no simulator)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sts_ai.interactive.templates import (
    DEFAULT_PROMPT_TEMPLATE,
    NEUTRAL_FRAME,
    TemplateStore,
    render_template,
)
from sts_ai.prompting import render_action_prompt
from sts_ai.schemas import LegalAction


def _legal() -> list[LegalAction]:
    return [
        LegalAction(index=0, bits=1, description="take card Strike [Attack, Common]"),
        LegalAction(index=1, bits=2, description="skip rewards / proceed"),
        LegalAction(index=2, bits=4, description="play Defend (cost 1) -> {self}"),
    ]


class TemplateParityTest(unittest.TestCase):
    """The default template must reproduce render_action_prompt byte-for-byte —
    this is what guarantees framing-only Studio rollouts match the harness."""

    def test_default_template_matches_render_action_prompt_no_reasoning(self):
        legal = _legal()
        state_text = "Your HP: 70/80\nDeck: {Strike, Strike, Defend,}\nPotions: none"
        rendered = render_template(
            DEFAULT_PROMPT_TEMPLATE,
            framing=NEUTRAL_FRAME,
            state_text=state_text,
            legal_actions=legal,
            induce_reasoning=False,
        )
        self.assertEqual(rendered, render_action_prompt(state_text, legal, NEUTRAL_FRAME, False))

    def test_default_template_matches_render_action_prompt_with_reasoning(self):
        legal = _legal()
        state_text = "combat state with {braces} that must survive"
        custom_framing = "Be adventurous and take risks."
        rendered = render_template(
            DEFAULT_PROMPT_TEMPLATE,
            framing=custom_framing,
            state_text=state_text,
            legal_actions=legal,
            induce_reasoning=True,
        )
        self.assertEqual(rendered, render_action_prompt(state_text, legal, custom_framing, True))

    def test_literal_braces_in_state_text_are_not_substituted(self):
        # state_text containing a placeholder-looking token must be passed through
        # verbatim (single-pass substitution, substituted values not re-scanned).
        legal = _legal()
        rendered = render_template(
            DEFAULT_PROMPT_TEMPLATE,
            framing="F",
            state_text="literal {framing} and {state_text} tokens",
            legal_actions=legal,
        )
        self.assertIn("literal {framing} and {state_text} tokens", rendered)

    def test_unknown_placeholder_left_untouched(self):
        out = render_template(
            "head {framing} {unknown} tail",
            framing="F",
            state_text="",
            legal_actions=[],
        )
        self.assertEqual(out, "head F {unknown} tail")


class TemplateStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TemplateStore(Path(self._tmp.name) / "templates")

    def tearDown(self):
        self._tmp.cleanup()

    def test_builtins_present_without_any_files(self):
        self.assertEqual(self.store.get_framing("neutral"), NEUTRAL_FRAME)
        self.assertEqual(self.store.get_prompt_template("default"), DEFAULT_PROMPT_TEMPLATE)
        self.assertIn("neutral", self.store.list_framings())
        self.assertIn("default", self.store.list_prompt_templates())

    def test_save_get_list_delete_roundtrip(self):
        self.store.save_framing("risky", "Take risks.")
        self.assertEqual(self.store.get_framing("risky"), "Take risks.")
        self.assertIn("risky", self.store.list_framings())
        self.assertTrue(self.store.delete_framing("risky"))
        self.assertNotIn("risky", self.store.list_framings())
        # built-ins survive a delete of a user file
        self.assertIn("neutral", self.store.list_framings())

    def test_user_file_overrides_builtin(self):
        self.store.save_framing("neutral", "OVERRIDDEN")
        self.assertEqual(self.store.get_framing("neutral"), "OVERRIDDEN")
        self.store.delete_framing("neutral")
        self.assertEqual(self.store.get_framing("neutral"), NEUTRAL_FRAME)

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save_framing("../escape", "x")
        with self.assertRaises(ValueError):
            self.store.save_framing("", "x")

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.get_framing("nope")


if __name__ == "__main__":
    unittest.main()
