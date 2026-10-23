from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sts_env import SlayTheSpireEnv
from tests.support import requires_simulator


@requires_simulator
class PublicEnvironmentTests(unittest.TestCase):
    def test_three_modes_obey_control_contract_and_are_repeatable(self):
        for mode in ("full", "ooc", "combat"):
            with self.subTest(mode=mode), SlayTheSpireEnv(mode=mode, max_steps=8, max_act=1, step_timeout=30) as env:
                first, info = env.reset(seed=3)
                self.assertTrue(info["native_provenance"]["module_sha256"])
                expected_phase = "combat" if mode == "combat" else "out_of_combat"
                self.assertEqual(first["phase"], expected_phase)
                trace = []
                for _ in range(8):
                    observation, reward, terminated, truncated, after = env.step(0)
                    if mode != "full" and observation["legal_actions"]:
                        self.assertEqual(observation["phase"], expected_phase)
                    trace.append((observation, reward, terminated, truncated))
                    if terminated or truncated:
                        break
                again, _ = env.reset(seed=3)
                self.assertEqual(first, again)
                for expected in trace:
                    observation, reward, terminated, truncated, _ = env.step(0)
                    self.assertEqual(expected, (observation, reward, terminated, truncated))

    def test_illegal_index_is_rejected_without_mutating_native_state(self):
        with SlayTheSpireEnv(max_steps=3) as env:
            before, _ = env.reset(seed=3)
            with self.assertRaises(ValueError):
                env.step(len(before["legal_actions"]))
            self.assertEqual(env.observation, before)
            env.step(0)

    def test_native_preview_is_stable_and_does_not_change_public_state(self):
        with SlayTheSpireEnv(mode="full", max_steps=3, step_timeout=30) as env:
            before, _ = env.reset(seed=3)
            a = env.native_action()
            self.assertEqual(a, env.native_action())
            self.assertEqual(env.observation, before)
            self.assertIn(a, [x["index"] for x in before["legal_actions"]])
            env.step(a)

    def test_trace_matches_across_fresh_cli_processes_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"trace{i}.jsonl" for i in range(2)]
            for path in paths:
                completed = subprocess.run([
                    sys.executable, "-m", "sts_env", "run", "--mode", "full",
                    "--agent", "random", "--seed", "3", "--max-steps", "30",
                    "--step-timeout", "30", "--output", str(path),
                ], capture_output=True, text=True, timeout=90)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
            for trajectory in records:
                trajectory[-1].pop("elapsed_seconds")
            self.assertEqual(records[0], records[1])
            replay = subprocess.run([sys.executable, "-m", "sts_env", "replay", str(paths[0])],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            self.assertTrue(json.loads(replay.stdout)["verified_prefix"])
