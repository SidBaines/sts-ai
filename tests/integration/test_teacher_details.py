"""Native public-search details preserve the original vote protocol."""
import json
import unittest

from sts_policy.curriculum import TaskSpec
from sts_rl.fresh_collection import search_plan
from sts_rl.fresh_panel import FreshPanelWorker
from sts_rl.teacher_votes import votes_from_details
from tests.support import requires_simulator


@requires_simulator
class TeacherDetailsTests(unittest.TestCase):
    def test_details_match_votes_and_repeat_in_fresh_workers(self):
        for seed in (3, 7):
            with self.subTest(seed=seed):
                results = []
                for _ in range(2):
                    with FreshPanelWorker({}, timeout=30.) as worker:
                        observation, info = worker.reset(TaskSpec(seed=seed, kind="battle", battle_simulations=16))
                        self.assertEqual(observation["phase"], "combat")
                        plan = search_plan(info["public_hash"], samples=4)
                        votes = worker.search_votes_public(16, plan["samples"], plan["search_seed"])
                        details = worker.search_details_public(16, plan["samples"],
                                                               plan["effective_search_seeds"], plan["search_seed"])
                        self.assertEqual(votes_from_details(details, "best_line"), votes)
                        self.assertEqual(sum(votes.values()), 4)
                        self.assertEqual(details, json.loads(json.dumps(details, allow_nan=False)))
                        results.append(details)
                self.assertEqual(*results)


if __name__ == "__main__":
    unittest.main()
