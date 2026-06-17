from __future__ import annotations

import unittest

from tests.integration.test_streaming_rollout import ScriptedStreamingAgent
from tests.support import requires_simulator


def _make_env(seed: int):
    from sts_ai.lightspeed import LightspeedHybridEnv

    return LightspeedHybridEnv(
        world_seed=seed,
        battle_simulations=50,
        max_act=1,
        combat_control="llm",
    )


@requires_simulator
class RunUntilStreamingSupervisorTest(unittest.TestCase):
    def test_finite_specs_launch_exactly_m_and_drain_all(self) -> None:
        from sts_ai.streaming_rollout import run_streaming_rollouts

        specs = [(3, 0), (4, 0), (5, 0), (6, 0)]

        # This deterministic scripted agent keeps the test focused on the
        # supervisor contract: a finite list of M specs is launched once and all
        # in-flight rollouts are drained, even with concurrency below M.
        results = run_streaming_rollouts(
            specs,
            _make_env,
            ScriptedStreamingAgent(),
            concurrency=2,
            max_decisions=30,
        )

        self.assertEqual(len(results), 4)
        self.assertEqual(
            {(result.world_seed, result.rollout_index) for result in results},
            set(specs),
        )
        for result in results:
            self.assertGreater(len(result.decisions), 0)
            self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
