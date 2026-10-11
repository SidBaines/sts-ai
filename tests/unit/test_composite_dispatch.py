from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from sts_policy.encoding import STATE_DIM, ACTION_DIM
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic


def reference(model, state, actions, mask):
    combat_logits, combat_values = model.combat(state, actions, mask)
    ooc_logits, ooc_values = model.ooc(state, actions, mask)
    phase = state[:, 7] > .5
    return torch.where(phase[:, None], combat_logits, ooc_logits), torch.where(phase, combat_values, ooc_values)


class CompositeDispatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        torch.set_num_threads(1)

    def batch(self, phases):
        state = torch.randn(len(phases), STATE_DIM)
        state[:, 7] = torch.tensor(phases)
        actions = torch.randn(len(phases), 5, ACTION_DIM)
        mask = torch.ones(len(phases), 5, dtype=torch.bool)
        mask[:, -2:] = False
        return state, actions, mask

    def model(self, size="small"):
        return CompositeActorCritic(CandidateActorCritic(size=size, context="mean"),
                                    CandidateActorCritic(size=size, context="none"))

    def test_outputs_and_every_parameter_gradient_match_all_phase_patterns(self):
        cases = [(size, phases) for size in ("small", "large")
                 for phases in ([1.] * 8, [0.] * 8, [1., 0., 0., 1., 1., 0., 1., 0.])]
        for size, phases in cases:
            actual = self.model(size)
            original = deepcopy(actual)
            batch = self.batch(phases)
            logits, values = actual(*batch)
            old_logits, old_values = reference(original, *batch)
            torch.testing.assert_close(logits, old_logits, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(values, old_values, atol=1e-5, rtol=1e-5)
            weights = torch.randn_like(logits)
            weights[~batch[2]] = 0
            (logits.mul(weights).sum() + values.square().sum()).backward()
            (old_logits.mul(weights).sum() + old_values.square().sum()).backward()
            for (name, parameter), (_, old_parameter) in zip(actual.named_parameters(), original.named_parameters()):
                self.assertIsNotNone(parameter.grad, name)
                self.assertIsNotNone(old_parameter.grad, name)
                torch.testing.assert_close(parameter.grad, old_parameter.grad, atol=1e-5, rtol=1e-5)

    def test_only_matching_rows_enter_each_network(self):
        for phases in ([1.] * 8, [0.] * 8, [1., 0., 0., 1., 1., 0., 1., 0.]):
            model = self.model()
            counts = {"combat": [], "ooc": []}
            handles = [getattr(model, name).register_forward_pre_hook(
                lambda module, args, name=name: counts[name].append(len(args[0])))
                for name in counts]
            with torch.no_grad():
                model(*self.batch(phases))
            for handle in handles:
                handle.remove()
            self.assertEqual(counts["combat"], [sum(phases)] if any(phases) else [])
            self.assertEqual(counts["ooc"], [len(phases) - sum(phases)] if not all(phases) else [])

    def test_padding_and_candidate_permutation_remain_invariant(self):
        model = self.model().eval()
        state, actions, mask = self.batch([1., 0., 1., 0.])
        permutation = torch.tensor([2, 0, 1, 4, 3])
        with torch.no_grad():
            logits, values = model(state, actions, mask)
            changed_logits, changed_values = model(state, actions[:, permutation], mask[:, permutation])
            padded_actions = torch.cat((actions, torch.randn(4, 3, ACTION_DIM)), 1)
            padded_mask = torch.cat((mask, torch.zeros(4, 3, dtype=torch.bool)), 1)
            padded_logits, padded_values = model(state, padded_actions, padded_mask)
        torch.testing.assert_close(changed_logits, logits[:, permutation], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(changed_values, values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(padded_logits[:, :5], logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(padded_values, values, atol=1e-5, rtol=1e-5)
