from __future__ import annotations

import unittest

from sts_ai.interactive.replay import resolve_action_index


class _Action:
    def __init__(self, index: int, bits: int, description: str):
        self.index = index
        self.bits = bits
        self.description = description


class _Env:
    def __init__(self, actions):
        self._actions = actions

    def legal_actions(self):
        return self._actions


class ResolveActionIndexTest(unittest.TestCase):
    def test_recorded_index_breaks_exact_duplicate_tie(self):
        env = _Env(
            [
                _Action(0, 134217728, "take gold 10g"),
                _Action(1, 134217728, "take gold 10g"),
                _Action(2, 805306368, "skip rewards / proceed"),
            ]
        )

        self.assertEqual(
            resolve_action_index(
                env,
                134217728,
                "take gold 10g",
                recorded_index=1,
            ),
            1,
        )

    def test_exact_duplicate_without_index_uses_first_equivalent_action(self):
        env = _Env(
            [
                _Action(0, 134217728, "take gold 10g"),
                _Action(1, 134217728, "take gold 10g"),
            ]
        )

        self.assertEqual(resolve_action_index(env, 134217728, "take gold 10g"), 0)


if __name__ == "__main__":
    unittest.main()
