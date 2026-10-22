"""Pure vote-rule choices, thresholds and deterministic ties."""
from copy import deepcopy
import unittest

from sts_rl.teacher_votes import RULES, votes_from_details


def edge(index, visits, mean):
    return {"index": index, "visits": visits, "evaluation_sum": visits * mean, "mean_evaluation": mean}


class TeacherVotesTests(unittest.TestCase):
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
