from __future__ import annotations

from pathlib import Path
import unittest

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.local_tasks.gremlin_nob import GremlinNobTask
from sts_ai.local_tasks.runner import replay_task_start
from tests.support import requires_simulator


SOURCE = Path("data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192")


class LocalTaskReplayIntegrationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
