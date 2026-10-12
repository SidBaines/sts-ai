from __future__ import annotations

from copy import deepcopy
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from sts_policy import encoding
from tests.unit.test_policy_encoding import observation


def reference_bag(target, prefix, text, weight=1.0):
    words = encoding._WORDS.findall(str(text).lower())
    for word in [*words, *(a + ':' + b for a, b in zip(words, words[1:]))]:
        index, sign = encoding._hash(prefix + ':' + word, len(target))
        target[index] += sign * weight


class PublicBagCacheTests(unittest.TestCase):
    def setUp(self):
        encoding._bag_terms.cache_clear()

    def test_collisions_dynamic_weights_and_target_sizes_preserve_every_bit(self):
        text = "Strike+ Defend Strike+ unknown-symbol alpha beta gamma " * 3
        for size in (1, 2, 3, 64, 192):
            expected = np.linspace(-3, 4, size, dtype=np.float32)
            actual = expected.copy()
            for weight in (0., -0., -.2, np.float32(.3), 1e-9, -17.125):
                reference_bag(expected, 'dynamic', text, weight)
                encoding._bag(actual, 'dynamic', text, weight)
                self.assertEqual(expected.tobytes(), actual.tobytes())
        self.assertGreater(encoding._bag_terms.cache_info().hits, 0)

    def test_key_includes_prefix_normalized_content_and_length_not_weight(self):
        target = np.zeros(64, dtype=np.float32)
        encoding._bag(target, 'a', 'UNKNOWN Strike', .2)
        first = encoding._bag_terms.cache_info()
        encoding._bag(target, 'a', 'unknown strike', -2.)
        self.assertEqual(encoding._bag_terms.cache_info().hits, first.hits + 1)
        encoding._bag(target, 'b', 'unknown strike')
        encoding._bag(np.zeros(192, dtype=np.float32), 'a', 'unknown strike')
        encoding._bag(target, 'a', 'unknown defend')
        self.assertEqual(encoding._bag_terms.cache_info().misses, first.misses + 3)

    def test_mutable_text_and_target_arrays_are_never_retained(self):
        class MutablePublicText:
            value = 'alpha'
            def __str__(self): return self.value
        text = MutablePublicText()
        first = np.zeros(64, dtype=np.float32);encoding._bag(first, 'public', text)
        text.value = 'beta'
        second = np.zeros(64, dtype=np.float32);encoding._bag(second, 'public', text)
        self.assertNotEqual(first.tobytes(), second.tobytes())
        ref = weakref.ref(first);del first;gc.collect();self.assertIsNone(ref())
        terms = encoding._bag_terms('public', 'alpha', 64)
        self.assertIsInstance(terms, tuple)
        self.assertTrue(all(isinstance(pair, tuple) and isinstance(pair[0], int) for pair in terms))

    def test_lru_and_long_text_bypass_bound_retained_memory(self):
        target = np.zeros(1, dtype=np.float32)
        for index in range(encoding._BAG_CACHE_SIZE + 10):
            encoding._bag(target, 'public', 'name_' + str(index))
        self.assertEqual(encoding._bag_terms.cache_info().currsize, encoding._BAG_CACHE_SIZE)
        before = encoding._bag_terms.cache_info()
        long_text = 'unknown-word ' * encoding._BAG_CACHE_TEXT_LIMIT
        expected = target.copy();reference_bag(expected, 'long', long_text, -.2)
        encoding._bag(target, 'long', long_text, -.2)
        self.assertEqual(expected.tobytes(), target.tobytes())
        self.assertEqual(before, encoding._bag_terms.cache_info())
        long_prefix = 'public:' * encoding._BAG_CACHE_PREFIX_LIMIT
        expected = target.copy();reference_bag(expected, long_prefix, 'short public text', .3)
        encoding._bag(target, long_prefix, 'short public text', .3)
        self.assertEqual(expected.tobytes(), target.tobytes())
        self.assertEqual(before, encoding._bag_terms.cache_info())

    def test_empty_text_and_zero_length_target_keep_existing_behavior(self):
        encoding._bag(np.zeros(0, dtype=np.float32), 'empty', '1234')
        with self.assertRaises(ZeroDivisionError):encoding._bag(np.zeros(0), 'empty', 'word')
        with self.assertRaises(ZeroDivisionError):reference_bag(np.zeros(0), 'empty', 'word')

    def test_complete_features_private_fields_and_candidate_order_preserved(self):
        public = observation()
        variants = [public, deepcopy(public)]
        variants[1]['legal_actions'].reverse()
        variants[1]['state']['world_seed'] = 999
        variants[1]['info'] = {'native_values': [123]}
        for source in variants:
            with patch.object(encoding, '_bag', reference_bag):expected = encoding.encode_observation(source)
            actual = encoding.encode_observation(source)
            for key in ('state_features', 'action_features', 'action_indices'):
                self.assertEqual(getattr(expected,key).tobytes(), getattr(actual,key).tobytes())
            self.assertEqual(expected.coverage, actual.coverage)
