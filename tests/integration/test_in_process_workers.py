"""Exact collection and greedy evaluation parity at the same game-slot count."""
from pathlib import Path
import time
import unittest

import numpy as np
import torch

from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public
from sts_policy.rollout_training import collect_rollouts
from sts_rl.common import close_workers, load_policy, make_workers, phase_forward, train_tasks
from sts_rl.evaluate import evaluate_tasks
from tests.support import requires_simulator

CHAMPION = Path("/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start/"
                "data/rl_runs/fresh_start_20261015/inputs/champion_3d135d9bd53b.pt")


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if key not in {"elapsed_seconds", "elapsed", "wait_seconds", "timings"}}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


@requires_simulator
class InProcessExactnessTests(unittest.TestCase):
    def test_collection_and_greedy_evaluation_match_subprocess(self):
        if not CHAMPION.is_file():
            self.skipTest(f"Collection exactness requires local champion: {CHAMPION}")
        torch.set_num_threads(1)
        model, _ = load_policy(CHAMPION)
        model.eval()
        tasks = train_tasks(7_950_000_000, 8)
        collections, evaluations = [], []
        for in_process in (False, True):
            workers = make_workers(4, dead_action_filter=True, in_process=in_process)
            try:
                started = time.monotonic()
                torch.manual_seed(7)
                result = collect_rollouts(model, workers, tasks, collate=collate_public,
                    forward=phase_forward(combat="greedy"), config=PPOConfig(gamma=1., gae_lambda=1.),
                    max_seconds=3600, deterministic=False, deferred_bootstrap=True)
                print(f"\n{'In-process' if in_process else 'Subprocess'} collection: "
                      f"{time.monotonic() - started:.3f}s", flush=True)
                collections.append(result)
                evaluations.append(evaluate_tasks(model, workers, tasks, mode="greedy"))
            finally:
                close_workers(workers)
        left, right = collections
        self.assertEqual(left["faults"], [])
        self.assertEqual(right["faults"], [])
        self.assertEqual(len(left["episodes"]), 8)
        self.assertEqual(len(right["episodes"]), 8)
        self.assertEqual(len(left["transitions"]), len(right["transitions"]))
        exact = ("world_seed", "action_position", "action_index", "reward", "terminated", "truncated", "behavior_mask")
        approximate = ("behavior_log_prob", "behavior_value", "next_value", "advantage", "returns")
        for index, (a, b) in enumerate(zip(left["transitions"], right["transitions"])):
            for field in exact:
                self.assertEqual(getattr(a, field), getattr(b, field), f"transition {index}: {field}")
            for field in ("state_features", "action_features", "action_indices"):
                np.testing.assert_array_equal(getattr(a.encoded_observation, field),
                    getattr(b.encoded_observation, field), err_msg=f"transition {index}: {field}")
            for field in approximate:
                self.assertAlmostEqual(getattr(a, field), getattr(b, field), delta=1e-6,
                                       msg=f"transition {index}: {field}")
        self.assertEqual(without_timings(left["episodes"]), without_timings(right["episodes"]))
        self.assertEqual(evaluations[0][1], 0)
        self.assertEqual(evaluations[1][1], 0)
        self.assertEqual(len(evaluations[0][0]), 8)
        self.assertEqual(evaluations[0][0], evaluations[1][0])
        print(f"Exact parity: {len(left['transitions'])} transitions, 8 episodes, 8 greedy records", flush=True)


if __name__ == "__main__":
    unittest.main()
