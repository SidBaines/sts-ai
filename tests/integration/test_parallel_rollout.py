"""Integration test: the batched parallel orchestrator is trace-identical to the
serial path under a deterministic agent, and runs multiple seeds without error.

Gated with @requires_simulator (drives the real engine). See tests/CLAUDE.md."""
from __future__ import annotations

import unittest
from tests.integration.test_combat_control import ScriptedCombatAgent
from tests.support import requires_simulator


class ScriptedBatchAgent(ScriptedCombatAgent):
    """Deterministic agent with a batched API that simply maps the single-decision
    policy over the batch — so parallel and serial must produce identical records."""

    name = "scripted-batch"

    def choose_actions_batch(self, items):
        return [self.choose_action(state_text, legal_actions) for state_text, legal_actions in items]


def _make_env(seed: int):
    from sts_ai.lightspeed import LightspeedHybridEnv

    return LightspeedHybridEnv(world_seed=seed, battle_simulations=50, max_act=1, combat_control="llm")


@requires_simulator
class ParallelParityTest(unittest.TestCase):
    def test_parallel_k1_matches_serial(self):
        from sts_ai.rollout import run_rollout
        from sts_ai.parallel_rollout import run_parallel_rollouts

        max_decisions = 40
        serial = run_rollout(_make_env(3), ScriptedBatchAgent(), max_decisions=max_decisions)
        parallel = run_parallel_rollouts([(3, 0)], _make_env, ScriptedBatchAgent(),
                                         batch_size=1, max_decisions=max_decisions)[0]

        self.assertEqual(serial.stopped_reason, parallel.stopped_reason)
        self.assertEqual(len(serial.decisions), len(parallel.decisions))
        self.assertGreater(len(serial.decisions), 0)
        for a, b in zip(serial.decisions, parallel.decisions):
            self.assertEqual(a.phase, b.phase)
            self.assertEqual(a.selected_action, b.selected_action)
            self.assertEqual(a.affordances, b.affordances)  # shared builder -> identical

    def test_multi_seed_batch_runs(self):
        from sts_ai.parallel_rollout import run_parallel_rollouts

        results = run_parallel_rollouts([(3, 0), (4, 0)], _make_env, ScriptedBatchAgent(),
                                        batch_size=2, max_decisions=30)
        self.assertEqual([r.world_seed for r in results], [3, 4])
        self.assertEqual([r.rollout_index for r in results], [0, 0])
        for r in results:
            self.assertGreater(len(r.decisions), 0)
            self.assertIsNone(r.error)


if __name__ == "__main__":
    unittest.main()
