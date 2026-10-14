import unittest

try:
    import torch

    from sts_policy.losses import masked_distribution, masked_kl_divergence
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency absent")
class MaskedKLTests(unittest.TestCase):
    def test_actual_underflow_boundary_and_gradients(self):
        reference = torch.tensor([[0.0, -103.29425048828125, 0.0]])
        current = torch.tensor([[0.0, -104.02094268798828, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        legacy = torch.distributions.kl_divergence(
            masked_distribution(reference, mask), masked_distribution(current, mask)
        )
        self.assertTrue(torch.isinf(legacy).all())
        stable = masked_kl_divergence(reference, current, mask)
        self.assertTrue(torch.isfinite(stable).all())
        self.assertTrue((stable >= 0).all())
        stable.sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertEqual(current.grad[0, 2], 0.0)

    def test_regular_kl_and_gradients_match_independent_double_reference(self):
        reference = torch.tensor(
            [[1.0, 2.0, -1.0], [0.0, 2.0, 4.0]], dtype=torch.float64, requires_grad=True
        )
        current = torch.tensor(
            [[2.0, 0.0, -2.0], [1.0, 1.0, 3.0]], dtype=torch.float64, requires_grad=True
        )
        mask = torch.tensor([[True, True, False], [True, True, True]])
        expected = torch.distributions.kl_divergence(
            masked_distribution(reference, mask), masked_distribution(current, mask)
        )
        got = masked_kl_divergence(reference, current, mask)
        self.assertTrue(torch.allclose(got, expected, atol=1e-12, rtol=1e-12))
        expected_grad = torch.autograd.grad(
            expected.sum(), (reference, current), retain_graph=True
        )
        got_grad = torch.autograd.grad(got.sum(), (reference, current))
        for a, b in zip(expected_grad, got_grad):
            self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))

    def test_padding_nan_and_inf_cannot_leak_or_receive_gradients(self):
        ref = torch.tensor([[5.0, float("nan"), -2.0], [7.0, 0.0, float("inf")]])
        current = torch.tensor(
            [[1.0, float("-inf"), -3.0], [20.0, 0.0, float("nan")]], requires_grad=True
        )
        mask = torch.tensor([[True, False, True], [True, False, False]])
        kl = masked_kl_divergence(ref, current, mask)
        self.assertTrue(torch.isfinite(kl).all())
        self.assertEqual(kl[1], 0)
        kl.sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertTrue((current.grad[~mask] == 0).all())

    def test_empty_and_nonfinite_legal_menus_are_rejected(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        x = torch.zeros(1, 2)
        with self.assertRaises(ValueError):
            masked_kl_divergence(x, x, mask & False)
        with self.assertRaises(ValueError):
            masked_kl_divergence(x, x + float("nan"), mask)
        with self.assertRaises(ValueError):
            masked_kl_divergence(x, x, mask.float())

    def test_identical_and_saturated_policies_have_finite_zero_kl(self):
        x = torch.tensor([[1e4, -1e4, -1e3]], requires_grad=True)
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_kl_divergence(x.detach(), x, mask)
        self.assertEqual(result.item(), 0.0)
        result.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))


if __name__ == "__main__":
    unittest.main()
