from __future__ import annotations

from pathlib import Path
import unittest

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.local_tasks.elite_fights import LagavulinTask, SentriesTask
from sts_ai.local_tasks.gremlin_nob import GremlinNobTask
from sts_ai.local_tasks.runner import replay_task_start
from tests.support import requires_simulator


SOURCE = Path("data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192")


class LocalTaskReplayIntegrationTest(unittest.TestCase):
    def _assert_replay_reaches_task(self, task, enemy_name: str, expected_count: int):
        if not SOURCE.exists():
            self.skipTest(f"local source data not present: {SOURCE}")
        manifest = task.build_manifest(SOURCE)
        if not manifest["windows"]:
            self.skipTest(f"source data has no {task.task_id} windows")
        window = manifest["windows"][0]
        env = LightspeedHybridEnv(
            world_seed=int(window["world_seed"]),
            combat_control="llm",
            battle_simulations=50,
            max_act=3,
        )

        replay_task_start(env, window, task)

        summary = env.summary()
        self.assertIsNone(task.completion_reason(summary))
        enemies = summary["combat"]["enemies"]
        self.assertEqual(
            sum(enemy["name"] == enemy_name and enemy["alive"] for enemy in enemies),
            expected_count,
        )

    @requires_simulator
    def test_replay_reaches_live_gremlin_nob_start_when_source_data_exists(self):
        if not SOURCE.exists():
            self.skipTest(f"local source data not present: {SOURCE}")
        task = GremlinNobTask()
        manifest = task.build_manifest(SOURCE)
        if not manifest["windows"]:
            self.skipTest("source data has no Gremlin Nob windows")
        window = manifest["windows"][0]
        env = LightspeedHybridEnv(
            world_seed=int(window["world_seed"]),
            combat_control="llm",
            battle_simulations=50,
            max_act=3,
        )

        replay_task_start(env, window)

        summary = env.summary()
        self.assertIsNone(task.completion_reason(summary))
        enemies = summary["combat"]["enemies"]
        self.assertTrue(
            any(enemy["name"] == "GREMLIN_NOB" and enemy["alive"] for enemy in enemies)
        )

    @requires_simulator
    def test_replay_reaches_fixed_lagavulin_start_when_source_data_exists(self):
        self._assert_replay_reaches_task(LagavulinTask(), "LAGAVULIN", 1)

    @requires_simulator
    def test_replay_reaches_fixed_sentries_start_when_source_data_exists(self):
        self._assert_replay_reaches_task(SentriesTask(), "SENTRY", 3)


if __name__ == "__main__":
    unittest.main()
