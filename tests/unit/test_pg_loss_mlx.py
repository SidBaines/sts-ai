"""Unit tests for the MLX policy-gradient loss helpers."""
from __future__ import annotations

import unittest

import numpy as np

from tests.support import requires_mlx, requires_torch

try:
    import mlx.core as mx
    from sts_ai.train.pg_loss_mlx import grpo_loss, selective_logps

    _MLX = True
except Exception:
    _MLX = False

try:
    import torch
    from sts_ai.train.pg_loss import (
        grpo_loss as torch_grpo_loss,
        selective_logps as torch_selective_logps,
    )

    _TORCH = True
except Exception:
    _TORCH = False


def _mx_allclose(actual, expected, *, atol=1e-6) -> bool:
    result = mx.allclose(actual, expected, atol=atol)
    if hasattr(result, "item"):
        return bool(result.item())
    return bool(result)


def _mx_scalar(value) -> float:
    return float(value.item())


@requires_mlx
@unittest.skipUnless(_MLX, "mlx not installed")
class MlxPgLossTest(unittest.TestCase):
    def test_selective_logps_gathers_shifted_tokens(self):
        logits = mx.array(
            [
                [
                    [1.0, 2.0, 0.0, -1.0],
                    [0.5, -0.5, 1.5, 2.5],
                    [3.0, 0.0, -1.0, 1.0],
                ]
            ]
        )
        input_ids = mx.array([[0, 2, 3]])

        actual = selective_logps(logits, input_ids)
        shifted_logits = logits[:, :-1, :].astype(mx.float32)
        log_softmax = shifted_logits - mx.logsumexp(
            shifted_logits,
            axis=-1,
            keepdims=True,
        )
        expected = mx.take_along_axis(
            log_softmax,
            input_ids[:, 1:][..., None],
            axis=-1,
        ).squeeze(-1)

        self.assertEqual(actual.shape, (1, 2))
        self.assertTrue(_mx_allclose(actual, expected))

    def test_mu_one_reduces_to_negative_mean_advantage(self):
        logp_new = mx.array([[-0.2, -0.4], [-1.0, -1.2]])
        logp_old = logp_new
        advantages = mx.array([2.0, -0.5])
        mask = mx.ones_like(logp_new)

        loss, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            kl_beta=0.0,
        )

        expected = -mx.mean(advantages[:, None] * mx.ones_like(logp_new))
        self.assertTrue(_mx_allclose(loss, expected))
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0)
        self.assertAlmostEqual(metrics["clip_fraction"], 0.0)

    def test_completion_mask_ignores_unmasked_tokens(self):
        logp_new = mx.array([[-0.2, -0.4], [4.0, 5.0]])
        logp_old = logp_new
        advantages = mx.array([2.0, 100.0])
        mask = mx.array([[1.0, 1.0], [0.0, 0.0]])

        loss, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            kl_beta=0.0,
        )

        self.assertTrue(_mx_allclose(loss, mx.array(-2.0)))
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0)
        self.assertAlmostEqual(metrics["clip_fraction"], 0.0)

    def test_kl_zero_when_ref_matches_and_positive_kl_increases_loss(self):
        logp_new = mx.array([[-0.4, -0.6]])
        logp_old = logp_new
        advantages = mx.array([1.0])
        mask = mx.ones_like(logp_new)

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
            logp_new,
            advantages,
            mask,
            kl_beta=0.5,
        )
        loss_with_kl, kl_metrics = grpo_loss(
            logp_new,
            logp_old,
            logp_new + 1.0,
            advantages,
            mask,
            kl_beta=0.5,
        )

        self.assertTrue(_mx_allclose(loss_zero_kl, loss_no_kl))
        self.assertAlmostEqual(zero_metrics["mean_kl"], 0.0)
        self.assertGreater(kl_metrics["mean_kl"], 0.0)
        self.assertGreater(_mx_scalar(loss_with_kl), _mx_scalar(loss_no_kl))

    def test_clipping_caps_positive_advantage_surrogate(self):
        logp_old = mx.array([[-2.0, -2.0]])
        logp_new = logp_old + 4.0
        advantages = mx.array([3.0])
        mask = mx.ones_like(logp_new)

        loss, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            clip_eps=0.2,
            kl_beta=0.0,
        )

        expected = -mx.array(3.0 * 1.2)
        self.assertTrue(_mx_allclose(loss, expected))
        self.assertAlmostEqual(metrics["clip_fraction"], 1.0)
        self.assertGreater(metrics["mean_ratio"], 1.2)

    def test_gradient_sign_tracks_signed_advantage(self):
        logp_new = mx.array([[-0.2, -0.4], [-0.6, -0.8]])
        advantages = mx.array([1.5, -2.0])
        mask = mx.ones_like(logp_new)

        def f(lp):
            return grpo_loss(
                lp,
                mx.stop_gradient(lp),
                None,
                advantages,
                mask,
                kl_beta=0.0,
            )[0]

        grad = mx.grad(f)(logp_new)

        self.assertTrue(bool(mx.all(grad[0] < 0).item()))
        self.assertTrue(bool(mx.all(grad[1] > 0).item()))

    def test_clipping_zeros_positive_advantage_gradient(self):
        logp_old = mx.array([[-2.0, -2.0]])
        logp_new = logp_old + 4.0
        advantages = mx.array([3.0])
        mask = mx.ones_like(logp_new)

        def f(lp):
            return grpo_loss(
                lp,
                logp_old,
                None,
                advantages,
                mask,
                clip_eps=0.2,
                kl_beta=0.0,
            )[0]

        grad = mx.grad(f)(logp_new)
        _, metrics = grpo_loss(
            logp_new,
            logp_old,
            None,
            advantages,
            mask,
            clip_eps=0.2,
            kl_beta=0.0,
        )

        self.assertTrue(_mx_allclose(grad, mx.zeros_like(grad), atol=1e-6))
        self.assertAlmostEqual(metrics["clip_fraction"], 1.0)

    def test_unsupported_loss_type_raises(self):
        logp_new = mx.array([[-0.2, -0.4]])
        advantages = mx.array([1.0])
        mask = mx.ones_like(logp_new)

        with self.assertRaises(ValueError):
            grpo_loss(
                logp_new,
                logp_new,
                None,
                advantages,
                mask,
                loss_type="ppo",
            )


@requires_mlx
@requires_torch
@unittest.skipUnless(_MLX, "mlx not installed")
@unittest.skipUnless(_TORCH, "torch not installed")
class TorchMlxEquivalenceTest(unittest.TestCase):
    def test_selective_logps_and_grpo_loss_match_torch(self):
        logits_new = np.array(
            [
                [
                    [0.2, -1.1, 0.7, 1.3, -0.4],
                    [-0.5, 0.9, 1.1, -0.2, 0.4],
                    [1.5, -0.7, 0.3, 0.0, -1.2],
                    [0.8, 0.1, -0.9, 1.4, -0.6],
                ],
                [
                    [-1.0, 0.4, 0.6, -0.3, 1.2],
                    [0.5, -0.8, 1.7, -1.1, 0.2],
                    [-0.4, 1.0, -1.5, 0.9, 0.3],
                    [1.1, -0.2, 0.5, -0.7, 1.6],
                ],
            ],
            dtype=np.float32,
        )
        delta_old = np.array(
            [
                [
                    [0.10, -0.05, 0.02, -0.03, 0.04],
                    [-0.06, 0.03, -0.04, 0.08, -0.02],
                    [0.05, -0.01, 0.07, -0.06, 0.02],
                    [-0.03, 0.09, -0.02, 0.01, -0.04],
                ],
                [
                    [0.04, -0.02, 0.06, -0.05, 0.01],
                    [-0.01, 0.07, -0.03, 0.02, -0.05],
                    [0.08, -0.04, 0.01, -0.02, 0.03],
                    [-0.02, 0.05, -0.06, 0.04, -0.01],
                ],
            ],
            dtype=np.float32,
        )
        delta_ref = np.array(
            [
                [
                    [-0.04, 0.08, -0.01, 0.03, -0.02],
                    [0.02, -0.06, 0.05, -0.01, 0.04],
                    [-0.03, 0.01, -0.07, 0.06, -0.02],
                    [0.05, -0.02, 0.04, -0.03, 0.01],
                ],
                [
                    [-0.02, 0.04, -0.05, 0.01, 0.03],
                    [0.06, -0.01, 0.02, -0.04, 0.05],
                    [-0.01, 0.03, -0.02, 0.07, -0.06],
                    [0.04, -0.05, 0.03, -0.01, 0.02],
                ],
            ],
            dtype=np.float32,
        )

        cases = [
            {
                "logits_new": logits_new,
                "logits_old": logits_new + delta_old,
                "logits_ref": logits_new + delta_ref,
                "input_ids": np.array([[0, 2, 4, 1], [3, 0, 2, 4]], dtype=np.int64),
                "advantages": np.array([1.25, -0.75], dtype=np.float32),
                "mask": np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
                "clip_eps": 0.2,
                "kl_beta": 0.0,
            },
            {
                "logits_new": logits_new * -0.35 + 0.15,
                "logits_old": logits_new * -0.35 + 0.15 - delta_old,
                "logits_ref": logits_new * -0.35 + 0.15 + delta_ref,
                "input_ids": np.array([[4, 1, 3, 0], [2, 4, 1, 3]], dtype=np.int64),
                "advantages": np.array([-0.5, 2.0], dtype=np.float32),
                "mask": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
                "clip_eps": 0.15,
                "kl_beta": 0.4,
            },
        ]

        for case in cases:
            with self.subTest(clip_eps=case["clip_eps"], kl_beta=case["kl_beta"]):
                torch_ids = torch.tensor(case["input_ids"], dtype=torch.long)
                mlx_ids = mx.array(case["input_ids"], dtype=mx.int32)

                torch_logp_new = torch_selective_logps(
                    torch.tensor(case["logits_new"], dtype=torch.float32),
                    torch_ids,
                )
                torch_logp_old = torch_selective_logps(
                    torch.tensor(case["logits_old"], dtype=torch.float32),
                    torch_ids,
                )
                torch_logp_ref = torch_selective_logps(
                    torch.tensor(case["logits_ref"], dtype=torch.float32),
                    torch_ids,
                )
                mlx_logp_new = selective_logps(
                    mx.array(case["logits_new"], dtype=mx.float32),
                    mlx_ids,
                )
                mlx_logp_old = selective_logps(
                    mx.array(case["logits_old"], dtype=mx.float32),
                    mlx_ids,
                )
                mlx_logp_ref = selective_logps(
                    mx.array(case["logits_ref"], dtype=mx.float32),
                    mlx_ids,
                )

                self.assertTrue(
                    np.allclose(
                        torch_logp_new.detach().cpu().numpy(),
                        np.array(mlx_logp_new),
                        atol=1e-5,
                    )
                )

                torch_loss, torch_metrics = torch_grpo_loss(
                    torch_logp_new,
                    torch_logp_old,
                    torch_logp_ref,
                    torch.tensor(case["advantages"], dtype=torch.float32),
                    torch.tensor(case["mask"], dtype=torch.float32),
                    clip_eps=case["clip_eps"],
                    kl_beta=case["kl_beta"],
                )
                mlx_loss, mlx_metrics = grpo_loss(
                    mlx_logp_new,
                    mlx_logp_old,
                    mlx_logp_ref,
                    mx.array(case["advantages"], dtype=mx.float32),
                    mx.array(case["mask"], dtype=mx.float32),
                    clip_eps=case["clip_eps"],
                    kl_beta=case["kl_beta"],
                )

                self.assertAlmostEqual(
                    float(torch_loss.detach().cpu().item()),
                    _mx_scalar(mlx_loss),
                    delta=1e-5,
                )
                for key in (
                    "mean_kl",
                    "mean_ratio",
                    "clip_fraction",
                    "mean_advantage",
                ):
                    self.assertAlmostEqual(
                        torch_metrics[key],
                        mlx_metrics[key],
                        delta=1e-5,
                    )


if __name__ == "__main__":
    unittest.main()
