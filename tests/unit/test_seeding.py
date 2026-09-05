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

    def test_salt_zero_is_byte_identical_to_unsalted(self):
        # Frozen-seed contract: the default salt must reproduce the historical
        # stream exactly. 5755142843407948472 is the recorded policy_seed of
        # the ooc_grpo_v1 iter_0 seed_202_r0 rollout (2026-08-17).
        self.assertEqual(derive_policy_seed(202, 0), 5755142843407948472)
        self.assertEqual(derive_policy_seed(202, 0, salt=0), 5755142843407948472)

    def test_salt_changes_policy_seed_deterministically(self):
        unsalted = derive_policy_seed(202, 0)
        salted = {salt: derive_policy_seed(202, 0, salt=salt) for salt in (1, 4, 1004)}
        self.assertNotIn(unsalted, salted.values())
        self.assertEqual(len(set(salted.values())), 3)
        for salt, seed in salted.items():
            self.assertEqual(seed, derive_policy_seed(202, 0, salt=salt))
            self.assertLess(seed, 1 << 63)


class BatchSeedTest(unittest.TestCase):
    def test_batch_seed_is_order_independent(self):
        members = [(3, 0, 2), (1, 0, 5), (3, 1, 0), (2, 0, 4)]
        self.assertEqual(derive_batch_seed(members), derive_batch_seed(reversed(members)))

    def test_batch_seed_is_stable(self):
        members = [(7, 0, 0), (7, 0, 1), (8, 2, 3)]
        self.assertEqual(derive_batch_seed(members), derive_batch_seed(members))
        self.assertEqual(derive_batch_seed(members), 9164810563290509533)
        self.assertLess(derive_batch_seed(members), 1 << 63)

    def test_batch_seed_salt_zero_is_byte_identical_and_salt_changes_seed(self):
        members = [(7, 0, 0), (7, 0, 1), (8, 2, 3)]
        self.assertEqual(derive_batch_seed(members, salt=0), 9164810563290509533)
        salted = derive_batch_seed(members, salt=4)
        self.assertNotEqual(salted, 9164810563290509533)
        self.assertEqual(salted, derive_batch_seed(members, salt=4))
        self.assertNotEqual(salted, derive_batch_seed(members, salt=5))
        # Salting stays order-independent over members.
        self.assertEqual(salted, derive_batch_seed(list(reversed(members)), salt=4))


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
