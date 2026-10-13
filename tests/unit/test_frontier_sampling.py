from __future__ import annotations

from copy import deepcopy
import unittest

from sts_policy.curriculum import world_split
from sts_policy.frontier_sampling import sample_frontiers


class FrontierSamplingTests(unittest.TestCase):
    def rows(self):
        worlds=[seed for seed in range(100) if world_split(seed)=="train"][:8]
        return [{"seed":world,"start_act":act,"prefix":[index],"decision_family":"card_only_rewards"}
                for world in worlds for act in (2,3) for index in range(3)]

    def test_world_membership_does_not_depend_on_menu_count_or_row_order(self):
        rows=self.rows();a=sample_frontiers(rows,per_act=3,seed=4)
        expanded=deepcopy(rows)
        expanded += [{**rows[0],"prefix":[index]} for index in range(4,204)]
        b=sample_frontiers(list(reversed(expanded)),per_act=3,seed=4)
        self.assertEqual([(row["start_act"],row["seed"]) for row in a],[(row["start_act"],row["seed"]) for row in b])
        self.assertEqual(a,sample_frontiers(list(reversed(rows)),per_act=3,seed=4))
        self.assertEqual(len({(row["start_act"],row["seed"]) for row in a}),6)

    def test_family_filter_duplicate_rows_and_nontrain_boundaries(self):
        rows=self.rows();a=sample_frontiers(rows,seed=1,decision_family="card_only_rewards")
        self.assertEqual(a,sample_frontiers(rows+rows,seed=1,decision_family="card_only_rewards"))
        self.assertEqual(sample_frontiers(rows,decision_family="missing"),[])
        invalid={**rows[0],"seed":next(seed for seed in range(100) if world_split(seed)!="train")}
        with self.assertRaises(ValueError):sample_frontiers(rows+[invalid])
