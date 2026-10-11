from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_nonllm_baselines import matched_tasks, heuristic_action, task_identity
from sts_policy.curriculum import TaskSpec, world_split
from tests.unit.test_policy_encoding import observation


class NonllmBaselineTests(unittest.TestCase):
    def test_run_tasks_deduplicate_battle_bank_worlds(self):
        seed = next(seed for seed in range(1000, 1200) if world_split(seed) == "dev")
        a = TaskSpec(seed=seed, prefix=(0,), seek_combat=False)
        b = replace(a, prefix=(1,))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank.jsonl"
            path.write_text(json.dumps(a.to_dict()) + "\n" + json.dumps(b.to_dict()) + "\n")
            self.assertEqual(len(matched_tasks(bank=path, kind="run", split="dev", episodes=32)), 1)
            battles = matched_tasks(bank=path, kind="battle", split="dev", episodes=32)
            self.assertEqual(len(battles), 2)
            self.assertNotEqual(task_identity(battles[0]), task_identity(battles[1]))

    def test_generated_tasks_are_dev_and_deterministically_matched(self):
        kwargs = dict(bank=None, kind="run", split="dev", episodes=10, seed=7)
        first, second = matched_tasks(**kwargs), matched_tasks(**kwargs)
        self.assertEqual(first, second)
        self.assertTrue(all(world_split(task.seed) == "dev" for task in first))

    def test_tactical_heuristic_prefers_lethal_target(self):
        public = observation()
        self.assertEqual(heuristic_action(public), 1)
