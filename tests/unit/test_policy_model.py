from __future__ import annotations

import unittest

import torch

from sts_policy.encoding import encode_observation, collate_observations
from sts_policy.model import CandidateActorCritic, parameter_count
from tests.unit.test_policy_encoding import observation


class CandidateModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_permutation_and_padding_invariance_both_contexts(self):
        encoded = encode_observation(observation())
        state, actions, mask = collate_observations([encoded])
        permutation = torch.tensor([3, 1, 0, 2])
        for context in ("none", "mean"):
            model = CandidateActorCritic(context=context)
            logits, values = model(state, actions, mask)
            changed_logits, changed_values = model(state, actions[:, permutation], mask[:, permutation])
            torch.testing.assert_close(changed_logits, logits[:, permutation], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(changed_values, values)
            padded_actions = torch.cat([actions, torch.full((1, 3, actions.shape[-1]), 100)], dim=1)
            padded_mask = torch.cat([mask, torch.zeros((1, 3), dtype=torch.bool)], dim=1)
            padded_logits, padded_values = model(state, padded_actions, padded_mask)
            torch.testing.assert_close(padded_logits[:, :4], logits, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(padded_values, values)
            self.assertTrue((padded_logits[:, 4:] == -1e9).all())

    def test_shapes_gradient_and_terminal_critic(self):
        public = observation()
        terminal = {**public, "legal_actions": []}
        batch = collate_observations([encode_observation(public), encode_observation(terminal)])
        model = CandidateActorCritic()
        logits, value = model(*batch)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((logits[1] == -1e9).all())
        loss = -torch.log_softmax(logits[0], 0)[1] + value.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_architectures_and_serializable_metadata(self):
        previous = 0
        for size in ("small", "medium", "large"):
            model = CandidateActorCritic(size=size)
            self.assertGreater(parameter_count(model), previous)
            previous = parameter_count(model)
            clone = CandidateActorCritic(**model.metadata())
            clone.load_state_dict(model.state_dict())
        with self.assertRaises(ValueError):
            CandidateActorCritic(size="huge")
