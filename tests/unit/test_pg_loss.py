"""Unit tests for the torch policy-gradient loss helpers."""
from __future__ import annotations

import unittest

try:
    import torch
    from sts_ai.train.pg_loss import grpo_loss, selective_logps
    _TORCH = True
except Exception:
    _TORCH = False


@unittest.skipUnless(_TORCH, "torch not installed")
class PgLossTest(unittest.TestCase):
    def test_selective_logps_gathers_shifted_tokens(self):
        logits = torch.tensor(
            [
                [
                    [1.0, 2.0, 0.0, -1.0],
                    [0.5, -0.5, 1.5, 2.5],
                    [3.0, 0.0, -1.0, 1.0],
                ]
            ]
        )
        input_ids = torch.tensor([[0, 2, 3]])

        actual = selective_logps(logits, input_ids)
        expected = torch.stack(
            [
                torch.log_softmax(logits[0, 0], dim=-1)[2],
                torch.log_softmax(logits[0, 1], dim=-1)[3],
            ]
        ).unsqueeze(0)

        self.assertEqual(actual.shape, (1, 2))
        self.assertTrue(torch.allclose(actual, expected))

    def test_mu_one_reduces_to_negative_mean_advantage(self):
        logp_new = torch.tensor([[-0.2, -0.4], [-1.0, -1.2]], requires_grad=True)
        logp_old = logp_new.detach()
        advantages = torch.tensor([2.0, -0.5])
        mask = torch.ones_like(logp_new)

        loss, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            kl_beta=0.0,
        )

        expected = -advantages.unsqueeze(1).expand_as(logp_new).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0)
        self.assertAlmostEqual(metrics["clip_fraction"], 0.0)

    def test_gradient_sign_tracks_signed_advantage(self):
        logp_new = torch.tensor([[-0.2, -0.4], [-1.0, -1.2]], requires_grad=True)
        advantages = torch.tensor([1.5, -2.0])
        mask = torch.ones_like(logp_new)

        loss, _ = grpo_loss(
            logp_new,
            logp_new.detach(),
            None,
            advantages,
            mask,
            kl_beta=0.0,
        )
        loss.backward()

        self.assertLess(float(logp_new.grad[0, 0]), 0.0)
        self.assertLess(float(logp_new.grad[0, 1]), 0.0)
        self.assertGreater(float(logp_new.grad[1, 0]), 0.0)
        self.assertGreater(float(logp_new.grad[1, 1]), 0.0)

    def test_clipping_caps_positive_advantage_gradient(self):
        logp_old = torch.tensor([[-2.0, -2.0]])
        logp_new = (logp_old + 4.0).clone().detach().requires_grad_(True)
        advantages = torch.tensor([3.0])
        mask = torch.ones_like(logp_new)

        loss, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            clip_eps=0.2,
            kl_beta=0.0,
        )
        loss.backward()

        self.assertTrue(torch.allclose(logp_new.grad, torch.zeros_like(logp_new)))
        self.assertAlmostEqual(metrics["clip_fraction"], 1.0)
        self.assertGreater(metrics["mean_ratio"], 1.2)

    def test_kl_zero_when_ref_matches_and_positive_kl_increases_loss(self):
        logp_new = torch.tensor([[-0.4, -0.6]], requires_grad=True)
        logp_old = logp_new.detach()
        advantages = torch.tensor([1.0])
        mask = torch.ones_like(logp_new)

        loss_no_kl, _ = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            kl_beta=0.0,
        )
        loss_zero_kl, zero_metrics = grpo_loss(
            logp_new,
            logp_old,
            logp_new.detach(),
            advantages,
            mask,
            kl_beta=0.5,
        )
        loss_with_kl, kl_metrics = grpo_loss(
            logp_new,
            logp_old,
            logp_new.detach() + 1.0,
            advantages,
            mask,
            kl_beta=0.5,
        )

        self.assertTrue(torch.allclose(loss_zero_kl, loss_no_kl))
        self.assertAlmostEqual(zero_metrics["mean_kl"], 0.0)
        self.assertGreater(kl_metrics["mean_kl"], 0.0)
        self.assertGreater(loss_with_kl.item(), loss_no_kl.item())

    def test_completion_mask_ignores_unmasked_tokens(self):
        logp_new = torch.tensor(
            [[-0.2, -0.4], [4.0, 5.0]],
            requires_grad=True,
        )
        logp_old = logp_new.detach()
        advantages = torch.tensor([2.0, 100.0])
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        loss, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            kl_beta=0.0,
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(-2.0)))
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0)
        self.assertAlmostEqual(metrics["clip_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
