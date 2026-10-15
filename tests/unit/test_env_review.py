"""Independent worker lifecycle regressions using importable spawn targets."""
from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from sts_env.env import SimulatorError, SimulatorTimeout, SlayTheSpireEnv
from sts_env._backend.combat_public_v5 import CombatPublicV5Env
from sts_env._backend.schemas import LegalAction
from tests.unit.test_environment_base import FakeBackend


def _ready(connection) -> None:
    observation = {"state": {}, "text": "ready", "phase": "combat",
                   "legal_actions": [{"index": 0, "bits": 0, "description": "advance"}]}
    connection.send(("ok", (observation, {"terminated": False, "truncated": False})))


def stalled_step_worker(connection, config, seed) -> None:
    _ready(connection)
    try:
        connection.recv()
        time.sleep(30)
    except EOFError:
        pass


def crashed_step_worker(connection, config, seed) -> None:
    _ready(connection)
    connection.recv()
    os._exit(19)


class WorkerReviewTests(unittest.TestCase):
    def test_selection_history_preserves_cards_chosen_before_redraw(self) -> None:
        class SelectionBackend:
            combat_observation = "combat_public_v3"

            def __init__(self):
                self.hand = "Strike"
                self._combat_recent_actions = []

            def phase(self):
                return "combat"

            def legal_actions(self):
                return [LegalAction(0, 1, "{ GAMBLE [0] }")]

            def public_combat_cards(self):
                return {"hand": [{"index": 0, "label": self.hand}],
                        "selection": {"type": "GAMBLE", "candidates": ["{ GAMBLE [0] }"]}}

            def step(self, index):
                action = self.legal_actions()[index]
                self._combat_recent_actions.append(action.description)
                self.hand = "Defend" if self.hand == "Strike" else "Bash"
                return action

            def summary(self):
                return {"ascension": 0, "combat": {"cards": self.public_combat_cards(),
                        "recent_actions": self._combat_recent_actions}}

            def describe_state(self):
                return "Recent actions this turn: " + " -> ".join(self._combat_recent_actions)

        backend = SelectionBackend()
        env = CombatPublicV5Env(backend)
        first = env.step(0).description
        self.assertIn("Strike", first)
        self.assertEqual(backend._combat_recent_actions, ["{ GAMBLE [0] }"])
        self.assertEqual(env.summary()["combat"]["recent_actions"], [first])
        second = env.step(0).description
        self.assertIn("Defend", second)
        self.assertEqual(backend._combat_recent_actions, ["{ GAMBLE [0] }", "{ GAMBLE [0] }"])
        self.assertEqual(env.summary()["combat"]["recent_actions"], [first, second])
        self.assertIn("Strike", env.describe_state())
        self.assertIn("Defend", env.describe_state())
        self.assertNotIn("Bash", env.describe_state())
        backend._combat_recent_actions.clear()
        self.assertEqual(env.summary()["combat"]["recent_actions"], [])

    def test_world_seed_stays_in_info_and_never_enters_public_observation(self) -> None:
        class SeededBackend(FakeBackend):
            def summary(self):
                return {**super().summary(), "world_seed": 123456789}

        env = SlayTheSpireEnv(isolated=False, ooc_seed=2**32 + 17)
        self.addCleanup(env.close)
        with patch("sts_env.env._make_backend", return_value=SeededBackend()):
            observation, info = env.reset(seed=123456789)
            self.assertNotIn("world_seed", observation["state"])
            self.assertNotIn("123456789", observation["text"])
            self.assertEqual(info["world_seed"], 123456789)
            self.assertEqual(info["ooc_seed"], 2**32 + 17)
            self.assertEqual(info["ooc_effective_seed"], 17)
            after, _, _, _, after_info = env.step(0)
            self.assertNotIn("world_seed", after["state"])
            self.assertEqual(after_info["world_seed"], 123456789)

    def test_native_step_timeout_reaps_worker_and_reset_remains_available(self) -> None:
        env = SlayTheSpireEnv(step_timeout=0.5)
        self.addCleanup(env.close)
        with patch("sts_env.env._worker", stalled_step_worker):
            env.reset(seed=3)
            started = time.monotonic()
            with self.assertRaises(SimulatorTimeout):
                env.step(0)
            self.assertLess(time.monotonic() - started, 3)
            self.assertIsNone(env._process)
            self.assertIsNone(env._connection)
            with self.assertRaises(RuntimeError):
                env.step(0)
            # Same API object can recover into a newly isolated episode.
            env.reset(seed=4)
            self.assertIsNotNone(env._process)

    def test_native_crash_raises_failure_and_reaps_worker(self) -> None:
        env = SlayTheSpireEnv(step_timeout=1)
        self.addCleanup(env.close)
        with patch("sts_env.env._worker", crashed_step_worker):
            env.reset(seed=3)
            with self.assertRaisesRegex(SimulatorError, "worker exited"):
                env.step(0)
            self.assertIsNone(env._process)
            self.assertIsNone(env._connection)


if __name__ == "__main__":
    unittest.main()
