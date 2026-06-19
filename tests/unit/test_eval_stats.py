from __future__ import annotations

import unittest

from sts_ai.eval_stats import bootstrap_ci, paired_floor_summary, sign_test


class SignTestTest(unittest.TestCase):
    def test_all_positive_exact_two_sided_p_value(self) -> None:
        result = sign_test([1.0, 2.0, 3.0])

        self.assertEqual(result["n_pos"], 3)
        self.assertEqual(result["n_neg"], 0)
        self.assertEqual(result["n_zero"], 0)
        self.assertEqual(result["n_effective"], 3)
        self.assertEqual(result["p_value"], 0.25)

    def test_zeros_are_excluded(self) -> None:
        result = sign_test([0.0, 1.0, -1.0, 2.0])

        self.assertEqual(result["n_pos"], 2)
        self.assertEqual(result["n_neg"], 1)
        self.assertEqual(result["n_zero"], 1)
        self.assertEqual(result["n_effective"], 3)

    def test_empty_has_unit_p_value(self) -> None:
        result = sign_test([])

        self.assertEqual(result["n_pos"], 0)
        self.assertEqual(result["n_neg"], 0)
        self.assertEqual(result["n_zero"], 0)
        self.assertEqual(result["n_effective"], 0)
        self.assertEqual(result["p_value"], 1.0)


class BootstrapCiTest(unittest.TestCase):
    def test_is_deterministic_for_seed_and_brackets_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]

        first = bootstrap_ci(values, n_resamples=1000, seed=123)
        second = bootstrap_ci(values, n_resamples=1000, seed=123)

        self.assertEqual(first, second)
        sample_mean = sum(values) / len(values)
        self.assertLessEqual(first[0], sample_mean)
        self.assertGreaterEqual(first[1], sample_mean)

    def test_empty_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_ci([])


class PairedFloorSummaryTest(unittest.TestCase):
    def test_computes_intersection_deltas(self) -> None:
        base = {1: 5.0, 2: 10.0, 3: 100.0}
        trained = {1: 7.0, 2: 9.0, 4: 200.0}

        result = paired_floor_summary(base, trained, n_resamples=200, bootstrap_seed=1)

        self.assertEqual(result["n_seeds"], 2)
        self.assertEqual(result["deltas_by_seed"], {1: 2.0, 2: -1.0})
        self.assertEqual(result["mean_delta"], 0.5)
        self.assertEqual(result["sign_test"]["n_effective"], 2)
        self.assertNotIn(3, result["deltas_by_seed"])
        self.assertNotIn(4, result["deltas_by_seed"])

    def test_empty_intersection_returns_zero_summary(self) -> None:
        result = paired_floor_summary({1: 5.0}, {2: 7.0})

        self.assertEqual(result["n_seeds"], 0)
        self.assertEqual(result["mean_delta"], 0.0)
        self.assertEqual(result["se_delta"], 0.0)
        self.assertEqual(result["sign_test"]["p_value"], 1.0)
        self.assertEqual(result["bootstrap_ci"], [0.0, 0.0])
        self.assertEqual(result["deltas_by_seed"], {})


if __name__ == "__main__":
    unittest.main()
