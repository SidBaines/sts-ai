from __future__ import annotations

import unittest

from scripts.experiments.astra_policy_probe import CandidateTrie, canonical_candidates
from sts_ai.schemas import LegalAction


class TestCandidateTrie(unittest.TestCase):
    def test_shared_prefix_and_terminal_eos(self):
        trie = CandidateTrie([[1, 2], [1, 3, 4]], [9])
        self.assertEqual(trie.allowed([]), [1])
        self.assertEqual(trie.allowed([1]), [2, 3])
        self.assertEqual(trie.allowed([1, 2]), [9])
        self.assertEqual(trie.allowed([1, 2, 9]), [9])
        self.assertEqual(trie.allowed([1, 3]), [4])
        with self.assertRaises(ValueError):
            trie.allowed([1, 4])

    def test_candidate_can_prefix_another(self):
        trie = CandidateTrie([[1], [1, 2]], [9])
        self.assertEqual(trie.allowed([1]), [2, 9])

    def test_canonical_strings_escape_action_text(self):
        actions = [LegalAction(index=0, bits=7, description='choose "yes"')]
        self.assertEqual(canonical_candidates(actions, "action_only"), ['{"action_index":0}'])
        self.assertEqual(canonical_candidates(actions, "action_text"), ['{"action":"choose \\"yes\\""}'])
