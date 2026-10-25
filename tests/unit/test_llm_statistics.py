import math
import random
import unittest

from sts_llm.statistics import DEFAULT_MIN_CLUSTERS, paired_cluster_bootstrap


def records(values, worlds=None):
    return [{"prompt_id": str(i), "world_id": str(worlds[i] if worlds else i), "value": v}
            for i, v in enumerate(values)]


class PairedStatisticsTests(unittest.TestCase):
    # Mechanics tests lower min_clusters so small hand-checkable panels still bootstrap.
    def test_zero_and_constant_differences_have_exact_intervals(self):
        for delta in (0., .1, -2.):
            report = paired_cluster_bootstrap(records([0.] * 3), records([delta] * 3), min_clusters=2)
            self.assertEqual(report["difference"], delta)
            self.assertEqual(report["ci95"], [delta, delta])

    def test_reversal_changes_sign_and_reverses_interval(self):
        a, b = records([1., 2., 4.]), records([2., 0., 7.])
        forward = paired_cluster_bootstrap(a, b, seed=3, replicates=100, min_clusters=2)
        reverse = paired_cluster_bootstrap(b, a, seed=3, replicates=100, min_clusters=2)
        self.assertEqual(forward["difference"], -reverse["difference"])
        for actual, expected in zip(forward["ci95"], [-x for x in reversed(reverse["ci95"])]):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_worlds_are_resampled_whole_and_decisions_weighted_equally(self):
        a, b = records([0., 0., 0., 0.], [0, 0, 0, 1]), records([0., 0., 0., 1.], [0, 0, 0, 1])
        got = paired_cluster_bootstrap(a, b, seed=8, replicates=17, min_clusters=2)
        rng = random.Random(8)
        draws = []
        for _ in range(17):
            sampled = rng.choices([[0., 0., 0.], [1.]], k=2)
            states = [v for world in sampled for v in world]
            draws.append(sum(states) / len(states))
        draws.sort()
        quantile = lambda q: draws[math.floor(16*q)] + (draws[math.ceil(16*q)]-draws[math.floor(16*q)])*(16*q-math.floor(16*q))
        self.assertEqual(got["difference"], .25)
        self.assertEqual(got["ci95"], [quantile(.025), quantile(.975)])

    def test_input_order_does_not_change_draws(self):
        a, b = records([0., 1., 2.]), records([1., 1., 4.])
        self.assertEqual(paired_cluster_bootstrap(a, b, replicates=31, min_clusters=2),
                         paired_cluster_bootstrap(a[::-1], b[::-1], replicates=31, min_clusters=2))

    def test_missing_cases_are_reported_and_only_complete_pairs_used(self):
        a, b = records([1., 2.]), records([3., 4., 5.])[1:]
        got = paired_cluster_bootstrap(a, b)
        self.assertEqual(got["n_pairs"], 1)
        self.assertEqual(got["difference"], 2.)
        self.assertEqual(got["missing_left_prompt_ids"], ["2"])
        self.assertEqual(got["missing_right_prompt_ids"], ["0"])
        self.assertEqual(got["pairs"][0]["prompt_id"], "1")

    def test_no_pairs_reports_missing_without_interval(self):
        got = paired_cluster_bootstrap([], records([1.]))
        self.assertIsNone(got["ci95"])
        self.assertIsNotNone(got["interval_unavailable_reason"])
        self.assertEqual(got["missing_left_prompt_ids"], ["0"])

    def test_invalid_records_and_parameters_fail(self):
        valid = records([0.])
        for bad in ([*valid, *valid], records([math.nan]), records([math.inf]), records([True]),
                    [{"prompt_id": "", "world_id": "w", "value": 0.}]):
            with self.assertRaises(ValueError): paired_cluster_bootstrap(bad, valid)
        with self.assertRaises(ValueError): paired_cluster_bootstrap(valid, records([1.], [9]))
        for kw in ({"replicates": 0}, {"replicates": True}, {"seed": 1.2},
                   {"min_clusters": 1}, {"min_clusters": 0}, {"min_clusters": 10.}, {"min_clusters": True}):
            with self.assertRaises(ValueError): paired_cluster_bootstrap(valid, valid, **kw)


class MinimumWorldsTests(unittest.TestCase):
    """Regression: few-world percentile intervals were reported with false precision."""

    def panel(self, n_worlds, states_per_world=2):
        worlds = [w for w in range(n_worlds) for _ in range(states_per_world)]
        left = records([0.] * len(worlds), worlds)
        right = records([(i % 3) * .1 + (w % 2) * .2 for i, w in enumerate(worlds)], worlds)
        return left, right

    def test_default_minimum_is_ten_worlds(self):
        self.assertEqual(DEFAULT_MIN_CLUSTERS, 10)

    def test_one_world_has_no_interval_even_for_constant_differences(self):
        # Previously a constant difference from one world gave a zero-width "interval".
        got = paired_cluster_bootstrap(records([0., 0.], [0, 0]), records([.5, .5], [0, 0]))
        self.assertEqual(got["difference"], .5)
        self.assertEqual(got["n_worlds"], 1)
        self.assertIsNone(got["ci95"])
        self.assertIn("1 paired world", got["interval_unavailable_reason"])
        self.assertEqual(got["used_replicates"], 0)

    def test_two_and_four_worlds_report_estimate_without_interval(self):
        for n_worlds in (2, 4):
            left, right = self.panel(n_worlds)
            got = paired_cluster_bootstrap(left, right)
            self.assertEqual(got["n_worlds"], n_worlds)
            self.assertIsNotNone(got["difference"])
            self.assertIsNotNone(got["left_mean"])
            self.assertIsNone(got["ci95"])
            self.assertEqual(got["used_replicates"], 0)
            self.assertIn(f"Only {n_worlds} paired world", got["interval_unavailable_reason"])
            self.assertIn("at least 10", got["interval_unavailable_reason"])

    def test_ten_or_more_worlds_get_an_interval(self):
        for n_worlds in (10, 25):
            left, right = self.panel(n_worlds)
            got = paired_cluster_bootstrap(left, right, replicates=200)
            self.assertEqual(got["n_worlds"], n_worlds)
            self.assertIsNone(got["interval_unavailable_reason"])
            self.assertEqual(got["used_replicates"], 200)
            low, high = got["ci95"]
            self.assertLessEqual(low, got["difference"])
            self.assertLessEqual(got["difference"], high)
            self.assertLess(low, high)

    def test_threshold_counts_worlds_not_decisions(self):
        # Forty decisions from four worlds are still four clusters.
        left, right = self.panel(4, states_per_world=10)
        got = paired_cluster_bootstrap(left, right)
        self.assertEqual(got["n_pairs"], 40)
        self.assertIsNone(got["ci95"])

    def test_threshold_can_be_set_explicitly(self):
        left, right = self.panel(4)
        self.assertIsNotNone(paired_cluster_bootstrap(left, right, replicates=50, min_clusters=4)["ci95"])
        self.assertIsNone(paired_cluster_bootstrap(left, right, replicates=50, min_clusters=5)["ci95"])


if __name__ == "__main__":
    unittest.main()
