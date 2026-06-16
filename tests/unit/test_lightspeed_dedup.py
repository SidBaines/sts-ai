"""Unit tests for combat action de-duplication + display->raw index mapping.

`LightspeedHybridEnv._action_views`/`legal_actions`/`step` are exercised with a
fake raw-action list via `object.__new__` (per tests/CLAUDE.md), so the dedup
logic and the index mapping are covered without the native simulator build.
"""
from __future__ import annotations

import unittest

from sts_ai.lightspeed import LightspeedHybridEnv


class _FakeRawAction:
    def __init__(self, bits: int, description: str) -> None:
        self.bits = bits
        self._description = description
        self.executed_on = None

    def describe(self, ctx):
        return self._description

    def execute(self, ctx):
        self.executed_on = ctx


def _env(raw, *, combat: bool) -> LightspeedHybridEnv:
    env = object.__new__(LightspeedHybridEnv)
    env.bc = "BATTLE_CTX" if combat else None  # truthy sentinel doubles as the ctx
    env.gc = "GAME_CTX"
    env.raw_actions = lambda: raw  # stub out advance + native enumeration
    env.advance_to_decision = lambda: 0  # stub out the native engine step
    return env


class CombatDedupTest(unittest.TestCase):
    def test_combat_collapses_identical_descriptions(self):
        raw = [
            _FakeRawAction(10, "play Strike (cost 1) -> FOE (deal 6)"),
            _FakeRawAction(11, "play Strike (cost 1) -> FOE (deal 6)"),  # dup (other hand slot)
            _FakeRawAction(12, "play Defend (cost 1)"),
            _FakeRawAction(13, "end turn"),
        ]
        la = _env(raw, combat=True).legal_actions()
        self.assertEqual(
            [a.description for a in la],
            ["play Strike (cost 1) -> FOE (deal 6)", "play Defend (cost 1)", "end turn"],
        )
        self.assertEqual([a.index for a in la], [0, 1, 2])  # contiguous display indices

    def test_chosen_display_index_maps_to_first_raw_action(self):
        raw = [
            _FakeRawAction(10, "play Strike (cost 1) -> FOE (deal 6)"),
            _FakeRawAction(11, "play Strike (cost 1) -> FOE (deal 6)"),
            _FakeRawAction(12, "end turn"),
        ]
        env = _env(raw, combat=True)
        selected = env.step(1)  # display index 1 == "end turn"
        self.assertEqual(selected.description, "end turn")
        self.assertEqual(raw[2].executed_on, "BATTLE_CTX")  # mapped to the raw end-turn
        self.assertIsNone(raw[1].executed_on)  # the collapsed duplicate is never executed

    def test_out_of_combat_keeps_actions_one_to_one(self):
        raw = [
            _FakeRawAction(1, "take card Strike [Attack, Basic]"),
            _FakeRawAction(2, "take card Strike [Attack, Basic]"),  # identical offers kept distinct
            _FakeRawAction(3, "skip rewards / proceed"),
        ]
        env = _env(raw, combat=False)
        self.assertEqual(len(env.legal_actions()), 3)  # no dedup out of combat
        env.step(1)
        self.assertEqual(raw[1].executed_on, "GAME_CTX")


if __name__ == "__main__":
    unittest.main()
