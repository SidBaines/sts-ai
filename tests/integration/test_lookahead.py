from __future__ import annotations
import unittest
from sts_policy.curriculum import TaskSpec
from sts_policy.lookahead import CounterfactualOOCWorker
from tests.support import requires_simulator

@requires_simulator
class LookaheadTests(unittest.TestCase):
    def test_candidate_order_independent_common_world_returns(self):
        with CounterfactualOOCWorker({"fast_observations": True}) as worker:
            spec = TaskSpec(seed=3, kind="run")
            first = worker.score_ooc(spec, candidates=[0, 1], branches=2, horizon=8)
            reverse = worker.score_ooc(spec, candidates=[1, 0], branches=2, horizon=8)
            self.assertEqual(first["observation"], reverse["observation"])
            self.assertEqual(sorted(first["evaluated_candidates"], key=lambda x: x["index"]),
                             sorted(reverse["evaluated_candidates"], key=lambda x: x["index"]))
            self.assertEqual(first["teacher_index"], reverse["teacher_index"])
            self.assertIn(first["teacher_index"], [0, 1])
            self.assertTrue(all(row["decisions"] <= 8 for row in first["evaluated_candidates"]))

    def test_lookahead_candidate_sample_is_reproducible(self):
        with CounterfactualOOCWorker({"fast_observations": True}) as worker:
            spec = TaskSpec(seed=10000, kind="run")
            first = worker.score_ooc(spec, branches=4, horizon=3)
            second = worker.score_ooc(spec, branches=4, horizon=3)
            self.assertEqual(first, second)
            self.assertIn(first["scripted_index"], [r["index"] for r in first["evaluated_candidates"]])
