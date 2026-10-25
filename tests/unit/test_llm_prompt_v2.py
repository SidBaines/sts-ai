from __future__ import annotations

import unittest

from sts_llm.decision_metrics import (centered_advantages, decisive_margin, normalized_teacher_score,
                                      reasoning_token_count, teacher_top)
from sts_llm.generation import K2_REASONING_END_TOKEN_IDS, parse_action
from sts_llm.observations import LABELLED_SYSTEM_MESSAGE, labelled_prompt, menu_labels


OBSERVATION = {'phase': 'combat', 'text': 'Hand:\n  [0] Strike\n  [1] Defend',
               'legal_actions': [{'index': 0, 'description': 'play Strike'},
                                 {'index': 1, 'description': 'play Defend'},
                                 {'index': 3, 'description': 'end turn'}]}
CLOSE = '</ifm|think>'


def row(probabilities=(0.7, 0.2), indices=(0, 1), legal=(0, 1, 3)):
    return {'legal_indices': list(legal), 'teacher': {'legal_indices': list(indices),
            'probabilities': list(probabilities)}}


class LabelledPromptTests(unittest.TestCase):
    def test_labels_are_letters_and_never_digits(self):
        self.assertEqual(menu_labels(3), ['A', 'B', 'C'])
        labels = menu_labels(64)
        self.assertEqual((labels[25], labels[26], labels[27], labels[-1]), ('Z', 'AA', 'AB', 'BL'))
        self.assertEqual(len(set(labels)), 64)
        self.assertTrue(all(label.isalpha() and label.isupper() for label in labels))
        with self.assertRaises(ValueError):
            menu_labels(0)

    def test_menu_follows_the_given_order_and_labels_map_to_native_indices(self):
        prompt = labelled_prompt(OBSERVATION, order=[3, 0, 1])
        system, user = prompt.messages
        self.assertEqual(system, {'role': 'system', 'content': LABELLED_SYSTEM_MESSAGE})
        self.assertTrue(user['content'].startswith(OBSERVATION['text'] + '\n\n'))
        self.assertTrue(user['content'].endswith('A) end turn\nB) play Strike\nC) play Defend'))
        self.assertEqual(prompt.labels, {'A': 3, 'B': 0, 'C': 1})
        self.assertEqual(labelled_prompt(OBSERVATION).labels, {'A': 0, 'B': 1, 'C': 3})

    def test_messages_hold_only_public_text_and_menu(self):
        observation = dict(OBSERVATION, state={'world_seed': 12345}, teacher='secret')
        content = ''.join(m['content'] for m in labelled_prompt(observation).messages)
        self.assertNotIn('12345', content)
        self.assertNotIn('secret', content)

    def test_order_must_be_a_permutation_of_the_menu(self):
        for order in ([0, 1], [0, 1, 2], [0, 0, 3], [0, 1, 3, 3]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                labelled_prompt(OBSERVATION, order=order)


class LabelledParseTests(unittest.TestCase):
    LABELS = {'A': 3, 'B': 0, 'C': 1}

    def parse(self, final, **kwargs):
        return parse_action('thinking' + CLOSE + final, [0, 1, 3], labels=self.LABELS, **kwargs)

    def test_label_answer_returns_native_index(self):
        parsed = self.parse('{"action": "B"}<|ifm|im_end|>')
        self.assertEqual((parsed.valid, parsed.action_index, parsed.error), (True, 0, None))

    def test_wrong_schema_unknown_label_and_truncation_are_invalid(self):
        cases = {'{"action_index": 0}': 'invalid_action_schema', '{"action": 1}': 'invalid_action_schema',
                 '{"action": "D"}': 'illegal_action_label', '{"action": "b"}': 'illegal_action_label',
                 '{"action": "B", "why": "x"}': 'invalid_action_schema', 'B': 'invalid_final_json'}
        for final, error in cases.items():
            with self.subTest(final=final):
                parsed = self.parse(final)
                self.assertFalse(parsed.valid)
                self.assertEqual(parsed.error, error)
        self.assertEqual(self.parse('{"action": "B"}', finish_reason='length').error, 'truncated')
        self.assertEqual(parse_action('{"action": "B"}', [0, 1, 3], labels=self.LABELS).error,
                         'missing_reasoning_boundary')

    def test_labels_must_point_at_legal_indices(self):
        with self.assertRaises(ValueError):
            parse_action(CLOSE + '{"action": "A"}', [0, 1], labels={'A': 7})

    def test_version_one_parsing_is_unchanged(self):
        parsed = parse_action('t' + CLOSE + '{"action_index": 3}', [0, 1, 3])
        self.assertEqual((parsed.valid, parsed.action_index), (True, 3))


class DecisionMetricTests(unittest.TestCase):
    def test_centered_advantages_keep_near_ties_small(self):
        self.assertEqual(centered_advantages([1., 0., 0., 1.]), [0.5, -0.5, -0.5, 0.5])
        near_tie = centered_advantages([0.30, 0.30, 0.30, 0.31])
        self.assertLess(max(map(abs, near_tie)), 0.01)
        for bad in ([], [float('nan')], [True, 0.]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                centered_advantages(bad)

    def test_margin_top_and_filtered_actions(self):
        decisive = row((0.7, 0.3))
        self.assertAlmostEqual(decisive_margin(decisive), 0.4)
        self.assertEqual(teacher_top(decisive), 0)
        # Index 3 is filtered by the teacher: probability zero, still legal.
        self.assertEqual(teacher_top(row((0.0, 1.0))), 1)

    def test_normalized_score_is_zero_for_random_and_one_for_teacher_choice(self):
        r = row((0.7, 0.3))
        mean = 1 / 3
        self.assertAlmostEqual(normalized_teacher_score(r, 0), 1.)
        self.assertAlmostEqual(normalized_teacher_score(r, 3), (0 - mean) / (0.7 - mean))
        self.assertEqual(normalized_teacher_score(r, None), normalized_teacher_score(r, 3))
        expected_random = sum(normalized_teacher_score(r, i) for i in (0, 1, 3)) / 3
        self.assertAlmostEqual(expected_random, 0.)
        flat = {'legal_indices': [0, 1], 'teacher': {'legal_indices': [0, 1], 'probabilities': [0.5, 0.5]}}
        self.assertIsNone(normalized_teacher_score(flat, 0))

    def test_reasoning_tokens_stop_at_the_first_closer(self):
        closer = K2_REASONING_END_TOKEN_IDS[1]
        self.assertEqual(reasoning_token_count([5, 6, 7, closer, 8]), (3, True))
        self.assertEqual(reasoning_token_count([closer, 8]), (0, True))
        self.assertEqual(reasoning_token_count([5, 6]), (2, False))


if __name__ == '__main__':
    unittest.main()
