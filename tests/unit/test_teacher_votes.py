"""Pure vote-rule choices, thresholds and deterministic ties."""
from copy import deepcopy
import math
import unittest

from sts_rl.teacher_votes import RULES, pooled_value_target, votes_from_details


def edge(index, visits, mean):
    return {"index": index, "visits": visits, "evaluation_sum": visits * mean, "mean_evaluation": mean}


class TeacherVotesTests(unittest.TestCase):
    def test_pooled_value_target_merges_edges_and_uses_inclusive_pooled_floor(self):
        details = [{"edges": [edge(2, 60, 1000), edge(2, 40, 1200), edge(7, 6, 1400),
                              edge(7, 4, 1000), edge(9, 4, 9000), edge(1, 0, 9000)]},
                   {"edges": [edge(2, 100, 1100), edge(7, 10, 1200), edge(9, 15, 9000)]}]
        for search in details:
            for item in search['edges']:
                item['mean_evaluation'] = -999  # Only sums and visits determine pooled means.
        original = deepcopy(details)
        target = pooled_value_target(details, 130.)
        self.assertEqual(set(target), {2, 7})
        self.assertAlmostEqual(target[2], 1 / (1 + math.e))
        self.assertAlmostEqual(target[7], math.e / (1 + math.e))
        self.assertEqual(details, original)

    def test_pooled_value_scale_and_stable_softmax(self):
        for scale in (16., 100., 1000.):
            with self.subTest(scale=scale):
                details = [{"edges": [edge(2, 100, 10000), edge(7, 10, 10032)]}]
                target = pooled_value_target(details, scale)
                self.assertAlmostEqual(target[7], 1 / (1 + math.exp(-32 / scale)))
                self.assertAlmostEqual(sum(target.values()), 1.)
        self.assertEqual(pooled_value_target(details, .001), {2: 0., 7: 1.})
        tied = [{"edges": [edge(2, 100, 3), edge(7, 10, 3)]}]
        self.assertEqual(pooled_value_target(tied, 100.), {2: .5, 7: .5})

    def test_pooled_value_empty_unvisited_and_invalid_scale(self):
        for details in ([], [{"edges": []}], [{"edges": [edge(2, 0, 100)]}]):
            self.assertEqual(pooled_value_target(details, 100.), {})
        for scale in (0, -1, float('nan'), float('inf'), True, None, '100'):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'value_scale'):
                pooled_value_target([], scale)

    def test_empty_and_unknown_rule(self):
        for rule in RULES:
            self.assertEqual(votes_from_details([], rule), {})
        for details in ([], [{"chosen": 2}]):
            with self.assertRaisesRegex(ValueError, "Unknown teacher vote rule"):
                votes_from_details(details, "unknown")

    def test_best_line_counts_native_choices(self):
        self.assertEqual(votes_from_details([{"chosen": i} for i in (7, 2, 7)], "best_line"), {7: 2, 2: 1})

    def test_best_line_counts_choices_once_despite_duplicate_edges(self):
        details = [{"chosen": 2, "edges": [edge(7, 6, 3), edge(7, 6, 2), edge(2, 10, 1)]}]
        original = deepcopy(details)
        self.assertEqual(votes_from_details(details, "best_line"), {2: 1})
        self.assertEqual(details, original)

    def test_most_visited_merges_duplicate_indices_before_ranking_and_ties(self):
        for visits, expected in ((10, 7), (12, 2)):
            details = [{"edges": [edge(7, 6, 3), edge(7, 6, 2), edge(2, visits, 1)]}]
            original = deepcopy(details)
            self.assertEqual(votes_from_details(details, "most_visited"), {expected: 1})
            self.assertEqual(details, original)

    def test_mean_value_merges_duplicate_visits_and_weighted_values(self):
        cases = (([edge(2, 100, 1), edge(7, 6, 3), edge(7, 4, 2)], 7),
                 ([edge(2, 50, 1), edge(2, 50, 1), edge(7, 9, 100)], 2),
                 ([edge(7, 1, 100), edge(7, 19, 0), edge(2, 20, 6)], 2))
        for edges, expected in cases:
            details = [{"edges": edges}]
            original = deepcopy(details)
            self.assertEqual(votes_from_details(details, "mean_value"), {expected: 1})
            self.assertEqual(details, original)

    def test_pooled_mean_merges_duplicates_within_and_across_searches(self):
        details = [{"edges": [edge(2, 50, 1), edge(2, 50, 1), edge(7, 6, 3), edge(7, 4, 2)]},
                   {"edges": [edge(2, 1, 100), edge(7, 50, 0), edge(7, 50, 0)]}]
        original = deepcopy(details)
        self.assertEqual(votes_from_details(details, "pooled_mean"), {2: 2})
        self.assertEqual(details, original)

    def test_duplicate_zero_visit_edges_have_zero_mean(self):
        details = [{"chosen": 7, "edges": [edge(7, 0, 100), edge(7, 0, 100), edge(2, 0, -1)]}]
        for rule in RULES:
            with self.subTest(rule=rule):
                self.assertEqual(votes_from_details(details, rule), {7 if rule == "best_line" else 2: 1})

    def test_most_visited_ignores_chosen_and_breaks_ties_by_index(self):
        details = [{"chosen": 9, "edges": [edge(7, 20, 1), edge(2, 20, 0), edge(9, 19, 100)]},
                   {"chosen": 2, "edges": [edge(7, 21, 0), edge(2, 20, 100)]}]
        self.assertEqual(votes_from_details(details, "most_visited"), {2: 1, 7: 1})

    def test_mean_value_threshold_is_inclusive(self):
        details = [{"chosen": 2, "edges": [edge(2, 100, 1), edge(7, 10, 2), edge(9, 9, 100)]}]
        self.assertEqual(votes_from_details(details, "mean_value"), {7: 1})

    def test_mean_value_ties_prefer_visits_then_index(self):
        details = [{"edges": [edge(7, 10, 2), edge(9, 20, 2), edge(2, 20, 2), edge(1, 100, 1)]}]
        self.assertEqual(votes_from_details(details, "mean_value"), {2: 1})

    def test_pooled_mean_weights_by_visits_and_counts_all_searches(self):
        details = [{"edges": [edge(2, 100, 1), edge(7, 10, 6)]},
                   {"edges": [edge(2, 1, 100), edge(7, 100, 0)]}]
        self.assertEqual(votes_from_details(details, "mean_value"), {7: 2})
        self.assertEqual(votes_from_details(details, "pooled_mean"), {2: 2})

    def test_pooled_mean_uses_pooled_threshold(self):
        details = [{"edges": [edge(2, 60, 1), edge(7, 1, 2), edge(9, 4, 100)]},
                   {"edges": [edge(2, 40, 1), edge(7, 9, 2), edge(9, 5, 100)]}]
        self.assertEqual(votes_from_details(details, "pooled_mean"), {7: 2})

    def test_pooled_mean_ties_and_zero_visits(self):
        for edges, expected in (([edge(7, 10, 2), edge(9, 20, 2)], 9),
                                ([edge(7, 20, 2), edge(2, 20, 2)], 2)):
            details = [{"edges": [edge(1, 0, 100), *edges]}]
            self.assertEqual(votes_from_details(details, "pooled_mean"), {expected: 1})


if __name__ == "__main__":
    unittest.main()
