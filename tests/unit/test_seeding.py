from __future__ import annotations

import unittest

from sts_ai.seeding import derive_batch_seed, derive_policy_seed, derive_stage_seed


class PolicySeedTest(unittest.TestCase):
    def test_policy_seed_is_deterministic(self):
        self.assertEqual(derive_policy_seed(123, 4), derive_policy_seed(123, 4))
        self.assertEqual(derive_policy_seed(123, 4), 8226882720170492154)
        self.assertLess(derive_policy_seed(123, 4), 1 << 63)

    def test_rollout_index_changes_policy_seed(self):
        for world_seed in (-3, 0, 1, 999999):
            seeds = {derive_policy_seed(world_seed, rollout_index) for rollout_index in range(5)}
            self.assertEqual(len(seeds), 5)


class BatchSeedTest(unittest.TestCase):
    def test_batch_seed_is_order_independent(self):
        members = [(3, 0, 2), (1, 0, 5), (3, 1, 0), (2, 0, 4)]
        self.assertEqual(derive_batch_seed(members), derive_batch_seed(reversed(members)))

    def test_batch_seed_is_stable(self):
        members = [(7, 0, 0), (7, 0, 1), (8, 2, 3)]
        self.assertEqual(derive_batch_seed(members), derive_batch_seed(members))
        self.assertEqual(derive_batch_seed(members), 9164810563290509533)
        self.assertLess(derive_batch_seed(members), 1 << 63)


class StageSeedTest(unittest.TestCase):
    def test_stage_seed_is_deterministic_distinct_and_stable(self):
        hinted = derive_stage_seed(7, 1, 2, "HINTED")
        launder = derive_stage_seed(7, 1, 2, "LAUNDER")
        normal = derive_stage_seed(7, 1, 2, "NORMAL")

        self.assertEqual(hinted, derive_stage_seed(7, 1, 2, "HINTED"))
        self.assertEqual(hinted, 8228253589604550278)
        self.assertEqual(len({normal, hinted, launder}), 3)
        self.assertNotEqual(hinted, derive_batch_seed([(7, 1, 2)]))
        self.assertLess(hinted, 1 << 63)


if __name__ == "__main__":
    unittest.main()
