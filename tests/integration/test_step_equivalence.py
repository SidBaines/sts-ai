"""Per-step hashes of fixed seeded games, including reset and defeat.

The fixture was recorded from release 0.1.6 with `scripts/benchmark_env_step.py
--record-fixture`. Re-record it only for a deliberate, versioned change to
observations, info or rewards. (Its 0.1.5.1 predecessor certified that the
step-path optimization was byte-identical; 0.1.6 was checked against it with the
potion line converted back to the old format.)

Source/runtime/build identities describe the running code, not gameplay. Pin only
those fields to the recorded identities when hashing info, so this test can check
new source with another compatible build. Live identities are checked separately.
Seeded trajectories are certified within a native platform, not across platforms.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.benchmark_env_step import CONFIG, SEEDS, choose_action, digest, hashes
from sts_env.env import _Episode
from tests.support import requires_simulator


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "env_step_reference.json"


@requires_simulator
class StepEquivalenceTests(unittest.TestCase):
    def test_original_full_games_are_byte_identical(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(CONFIG, fixture["config"])
        self.assertEqual(SEEDS, tuple(game["seed"] for game in fixture["games"]))
        for game in fixture["games"]:
            with self.subTest(seed=game["seed"]):
                episode = _Episode(fixture["config"], game["seed"])
                observation, info = episode.reset()
                identities = {**episode.interface_identity,
                              "native_provenance": episode.env.native_provenance}
                for key, value in identities.items():
                    self.assertEqual(info[key], value)
                self.assertEqual(hashes(observation, info, None, fixture["identities"]),
                                 game["reset"])
                for step, expected in enumerate(game["steps"]):
                    self.assertFalse(episode.finished, f"ended before step {step}")
                    action = choose_action(observation, step)
                    observation, reward, terminated, truncated, info = episode.step(action)
                    for key, value in identities.items():
                        self.assertEqual(info[key], value)
                    self.assertEqual(hashes(observation, info, reward, fixture["identities"])
                                     + [terminated, truncated], expected, f"step {step + 1}")
                self.assertTrue(episode.finished)
                self.assertFalse(truncated, "Fixture games must finish without truncation")

    def test_mutating_episode_results_cannot_change_state_or_later_results(self):
        def corrupt(value):
            if isinstance(value, dict):
                for key, item in list(value.items()):
                    corrupt(item)
                    value[key] = None
            elif isinstance(value, list):
                for item in value:
                    corrupt(item)
                value.clear()

        fixture = json.loads(FIXTURE.read_text())
        game = fixture["games"][0]
        episode = _Episode(fixture["config"], game["seed"])
        provenance = deepcopy(episode.env.native_provenance)
        observation, info = episode.reset()
        for step, expected in enumerate(game["steps"]):
            action = choose_action(observation, step)
            before = digest(observation)
            corrupt(observation)
            corrupt(info)
            self.assertEqual(digest(episode._observe()), before, f"mutation at step {step}")
            self.assertEqual(episode.env.native_provenance, provenance)
            observation, reward, terminated, truncated, info = episode.step(action)
            self.assertEqual(hashes(observation, info, reward, fixture["identities"])
                             + [terminated, truncated], expected, f"step {step + 1}")
        corrupt(observation)
        corrupt(info)
        self.assertEqual(episode.env.native_provenance, provenance)
