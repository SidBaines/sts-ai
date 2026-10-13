from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

import torch

from sts_policy.losses import PPOConfig, masked_distribution
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic, collate_public, forward_policy
from sts_policy.rollout_training import Transition, update_ppo
from tests.unit.test_policy_encoding import observation


class MicroBatchPPOTests(unittest.TestCase):
    """micro_batch accumulates the same minibatch gradient with a lower memory peak."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def buffer(self, model):
        obs = []
        for i in range(24):
            base = observation()
            # Mixed phases and menu sizes, so chunks pad to different widths.
            legal = base["legal_actions"][: 2 + i % 3]
            obs.append(dict(base, phase="combat" if i % 2 else "out_of_combat", legal_actions=legal))
        rows = []
        with torch.no_grad():
            for i, o in enumerate(obs):
                logits, values = model(*collate_public([o]))
                mask = collate_public([o])[2]
                log_prob = float(masked_distribution(logits, mask).log_prob(torch.tensor([1]))[0])
                rows.append(Transition(o, 1, int(o["legal_actions"][1]["index"]), log_prob, float(values[0]),
                                       [True] * len(o["legal_actions"]), 1., True, False, 0., i,
                                       advantage=(1. if i % 3 else -.7) * (1 + i / 24), returns=float(values[0]) + .5))
        return rows

    def run_update(self, model, rows, micro, **config):
        settings = dict(epochs=2, minibatch_size=12, target_kl=10.)
        settings.update(config)
        # SGD, not Adam: Adam rescales the ~0 gradients of softmax-invariant biases
        # into lr-sized steps, so float summation order would dominate the comparison.
        return update_ppo(model, rows, collate=collate_public, forward=forward_policy, seed=3,
                          config=PPOConfig(**settings), micro_batch=micro,
                          optimizer=torch.optim.SGD(model.parameters(), lr=.05))

    def test_micro_batched_update_matches_full_minibatch(self):
        model = CompositeActorCritic(CandidateActorCritic(), CandidateActorCritic())
        rows = self.buffer(model)
        full, micro = deepcopy(model), deepcopy(model)
        a = self.run_update(full, rows, None)
        b = self.run_update(micro, rows, 5)
        self.assertEqual(a["steps"], b["steps"])
        self.assertEqual(a["steps"], 4)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(micro.state_dict()[key], value, rtol=1e-5, atol=1e-6)
        for key in ("approx_kl", "policy_loss", "value_loss", "entropy", "clip_fraction"):
            self.assertAlmostEqual(a[key], b[key], places=5)

    def test_micro_batched_kl_guard_rejects_without_stepping(self):
        model = CompositeActorCritic(CandidateActorCritic(), CandidateActorCritic())
        rows = self.buffer(model)
        before = deepcopy(model.state_dict())
        # Stale behaviour log-probs make the first minibatch's KL exceed a tiny target.
        stale = [replace(r, behavior_log_prob=r.behavior_log_prob - 1.0) for r in rows]
        result = self.run_update(model, stale, 5, target_kl=1e-4, kl_guard_before_update=True)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["rejected_minibatches"], 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
