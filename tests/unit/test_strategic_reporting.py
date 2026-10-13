from __future__ import annotations

import unittest

from scripts.run_strategic_v3 import summarize


class StrategicReportingTests(unittest.TestCase):
    def test_boss_only_wins_and_correlated_continuations_are_distinguished(self):
        def episode(seed, family, choices):
            return {"world_seed": seed, "start_family": family, "terminated": True, "truncated": False,
                    "task_outcome": "victory", "floor": 51, "max_act_reached": 3, "actor_choice_rows": choices}
        result = {"episodes": [episode(1, "fresh", 10), episode(2, "continuation", 0), episode(2, "continuation", 5)],
                  "faults": [], "requested_tasks": 3, "unstarted_tasks": 0, "primitive_steps": 10,
                  "elapsed_seconds": 1., "primitive_steps_per_second": 10.}
        metrics = summarize(result)
        self.assertEqual(metrics["wins_with_actor_choices"], 2)
        self.assertEqual(metrics["wins_without_actor_choices"], 1)
        self.assertIsNone(metrics["win_wilson_95"])
        self.assertIsNotNone(metrics["fresh_win_wilson_95"])
        result["episodes"] = result["episodes"][:1]
        self.assertIsNotNone(summarize(result)["win_wilson_95"])


if __name__ == "__main__":
    unittest.main()
