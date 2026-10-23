"""Numerical contracts for GRPO, token alignment and frozen old-policy values."""
import importlib.util
import math
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from sts_llm.training import Completion, generation_backend_kwargs, load_rows, normalized_advantages


class AdvantageTests(unittest.TestCase):
    def test_constant_group_is_exact_zero(self):
        values, variance = normalized_advantages([.125] * 6)
        self.assertEqual(values, [0.] * 6)
        self.assertEqual(variance, 0.)

    def test_population_normalization_and_shift_invariance(self):
        values, variance = normalized_advantages([1., 2., 3.])
        shifted, _ = normalized_advantages([101., 102., 103.])
        self.assertEqual(values, shifted)
        self.assertAlmostEqual(sum(values), 0.)
        self.assertAlmostEqual(sum(v * v for v in values) / len(values), 1.)
        self.assertAlmostEqual(variance, 2. / 3.)

    def test_nonfinite_reward_rejected(self):
        for rewards in ([], [0., float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                normalized_advantages(rewards)

    def test_chat_template_returns_flat_ids_explicitly(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("return_dict") is not False:
                    raise AssertionError("Transformers5 returns BatchEncoding by default")
                return [2, 4, 6]
        args = SimpleNamespace(benchmark_updates=True, benchmark_generation=False,
                               effort="medium", max_new_tokens=10,
                               max_sequence_length=20, limit=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps({"id":"x", "legal_indices":[0],
                                        "messages":[{"role":"user", "content":"go"}]}) + "\n")
            rows = load_rows(path, Tokenizer(), args)
        self.assertEqual(rows[0]["prompt_ids"], [2, 4, 6])

    def test_prompt_overflow_is_rejected_not_silently_truncated(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **kw): return [2] * 11
        args = SimpleNamespace(benchmark_updates=True, benchmark_generation=False,
                               effort="medium", max_new_tokens=10,
                               max_sequence_length=20, limit=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps({"legal_indices":[0],
                                        "messages":[{"role":"user", "content":"go"}]}) + "\n")
            with self.assertRaisesRegex(ValueError, "exceeds max sequence length"):
                load_rows(path, Tokenizer(), args)

    def test_training_selects_train_worlds_and_never_validation_or_test(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kw):
                return [int(messages[0]["content"])]
        args = SimpleNamespace(benchmark_updates=False, benchmark_generation=False,
                               effort="medium", max_new_tokens=10,
                               max_sequence_length=20, limit=None)
        records = [{"id": split, "split": split, "world_identity": f"world{number}",
                    "legal_indices":[0], "teacher":{"checkpoint_sha256":"champion"},
                    "messages":[{"role":"user", "content":str(number)}]}
                   for number, split in enumerate(("validation", "train", "test"), 1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            rows = load_rows(path, Tokenizer(), args)
        self.assertEqual([row["id"] for row in rows], ["train"])
        self.assertEqual(args.selected_split, "train")
        self.assertEqual(args.selected_row_count, 1)
        self.assertEqual(len(args.selected_selection_sha256), 64)

    def test_unsplit_teacher_data_rejected_even_after_smoke_limit(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **kw): return [2, 4]
        args = SimpleNamespace(benchmark_updates=False, benchmark_generation=False,
                               effort="medium", max_new_tokens=10,
                               max_sequence_length=20, limit=1)
        row = {"id":"train", "split":"train", "legal_indices":[0], "teacher":{},
               "messages":[{"role":"user", "content":"go"}]}
        unsplit = dict(row, id="unsplit"); del unsplit["split"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(unsplit) + "\n")
            with self.assertRaisesRegex(ValueError, "every dataset row to declare split"):
                load_rows(path, Tokenizer(), args)

    def test_resolved_selection_identity_changes_with_limit(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **kw): return [2, 4]
        args = SimpleNamespace(benchmark_updates=False, benchmark_generation=False,
                               effort="medium", max_new_tokens=10,
                               max_sequence_length=20, limit=None)
        row = {"id":"a", "split":"train", "legal_indices":[0], "teacher":{},
               "messages":[{"role":"user", "content":"go"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, id="b")) + "\n")
            load_rows(path, Tokenizer(), args)
            full_identity = args.selected_selection_sha256
            args.limit = 1
            load_rows(path, Tokenizer(), args)
        self.assertNotEqual(full_identity, args.selected_selection_sha256)
        self.assertEqual(args.selected_row_count, 1)

    def test_static_cache_budget_is_checked_as_hard_input_contract(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **kw): return [2] * 9
        args = SimpleNamespace(benchmark_updates=False, benchmark_generation=True,
            static_cache_compile=True, static_cache_size=16, effort="medium",
            max_new_tokens=8, max_sequence_length=100, limit=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps({"legal_indices":[0],
                "messages":[{"role":"user", "content":"go"}]}) + "\n")
            with self.assertRaisesRegex(ValueError, "fixed static cache size"):
                load_rows(path, Tokenizer(), args)

    def test_compile_kwargs_use_supported_top_level_size_and_unmerged_path(self):
        self.assertEqual(generation_backend_kwargs(SimpleNamespace(static_cache_compile=False)), {})
        fake = SimpleNamespace(CompileConfig=lambda **kwargs: kwargs)
        with patch.dict("sys.modules", {"transformers": fake}):
            result = generation_backend_kwargs(SimpleNamespace(static_cache_compile=True, static_cache_size=8192))
        self.assertEqual(result["max_cache_len"], 8192)
        self.assertEqual(result["cache_implementation"], "static")
        self.assertNotIn("cache_config", result)
        self.assertNotIn("past_key_values", result)
        self.assertFalse(result["disable_compile"])
        self.assertEqual(result["compile_config"], {"backend":"inductor", "fullgraph":False,
            "dynamic":False, "mode":"reduce-overhead"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for numerical gradient tests")
class TensorContracts(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def objective(self, new, old, ref, advantage, mask, beta=0., **kwargs):
        from sts_llm.training import clipped_response_loss
        return clipped_response_loss(new, old, ref, advantage, mask,
                                      clip_epsilon=.2, kl_beta=beta, **kwargs)

    def test_clipping_has_correct_sign_and_zero_gradient_regions(self):
        t = self.torch
        for ratio, advantage, expected_loss, expected_gradient in (
            (1.3, 1., -1.2, 0.), (1.3, -1., 1.3, 1.3),
            (.7, 1., -.7, -.7), (.7, -1., .8, 0.)):
            new = t.tensor([[math.log(ratio)]], requires_grad=True)
            loss, metrics = self.objective(new, t.zeros_like(new), t.zeros_like(new),
                                           t.tensor([advantage]), t.ones_like(new, dtype=t.bool))
            loss.backward()
            self.assertAlmostEqual(float(loss), expected_loss, places=5)
            self.assertAlmostEqual(float(new.grad), expected_gradient, places=5)
            self.assertEqual(float(metrics["clip_fraction"]), 1.)

    def test_frozen_old_policy_makes_later_ratio_nonunit(self):
        t = self.torch
        new = t.nn.Parameter(t.tensor([[0.]]))
        old = t.zeros_like(new)
        optimizer = t.optim.SGD([new], lr=.5)
        mask = t.ones_like(new, dtype=t.bool)
        loss, first = self.objective(new, old, old, t.tensor([1.]), mask)
        self.assertEqual(float(first["mean_ratio"]), 1.)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        later_loss, later = self.objective(new, old, old, t.tensor([1.]), mask)
        self.assertGreater(float(later["mean_ratio"]), 1.2)
        self.assertEqual(float(later["clip_fraction"]), 1.)
        later_loss.backward()
        self.assertEqual(float(new.grad), 0.)
        self.assertEqual(float(old), 0.)

    def test_equal_response_weight_and_padding_exclusion(self):
        t = self.torch
        new = t.zeros((2, 3), requires_grad=True)
        old = t.zeros_like(new, requires_grad=True)
        ref = t.zeros_like(new, requires_grad=True)
        mask = t.tensor([[True, False, False], [True, True, True]])
        # Masked extreme reference values must not overflow the objective.
        with t.no_grad(): ref[0, 1:] = 10000.
        loss, _ = self.objective(new, old, ref, t.tensor([1., -1.]), mask)
        loss.backward()
        self.assertAlmostEqual(float(new.grad[0, 0]), -.5)
        self.assertEqual(float(new.grad[0, 1:].abs().sum()), 0.)
        for value in new.grad[1]: self.assertAlmostEqual(float(value), 1. / 6., places=6)
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)

    def test_zero_advantage_is_finite_without_policy_gradient(self):
        t = self.torch
        new = t.zeros((2, 4), requires_grad=True)
        loss, _ = self.objective(new, t.zeros_like(new), t.zeros_like(new),
                                 t.zeros(2), t.ones_like(new, dtype=t.bool))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(new.grad.abs().sum()), 0.)

    def test_fixed_budget_gives_same_per_token_gradient_for_mixed_lengths(self):
        t = self.torch
        mask = t.tensor([[True, False, False], [True, True, True]])
        new = t.zeros((2, 3), requires_grad=True)
        loss, metrics = self.objective(new, t.zeros_like(new), t.zeros_like(new),
            t.tensor([1., 1.]), mask, loss_normalization="fixed_token_budget", loss_token_budget=8)
        loss.backward()
        self.assertAlmostEqual(float(loss), -4. / 16.)
        expected = t.tensor([[-1./16., 0., 0.], [-1./16., -1./16., -1./16.]])
        t.testing.assert_close(new.grad, expected)
        # Ratio diagnostics remain likelihood ratios, not length / budget.
        self.assertEqual(float(metrics["mean_ratio"]), 1.)

    def test_fixed_budget_is_same_constant_across_effort_lengths_and_kl(self):
        t = self.torch
        budget, beta = 2048, .3
        observed = []
        for effort, length in (("low", 2), ("medium", 5), ("high", 9)):
            new = t.zeros((1, length), requires_grad=True)
            ref = t.full_like(new, math.log(2.))
            loss, metrics = self.objective(new, t.zeros_like(new), ref, t.tensor([1.]),
                t.ones_like(new, dtype=t.bool), beta=beta,
                loss_normalization="fixed_token_budget", loss_token_budget=budget)
            loss.backward()
            # k3 = exp(log2)-log2-1; both policy and KL sum / same C.
            expected_kl = length * (1. - math.log(2.)) / budget
            self.assertAlmostEqual(float(metrics["sampled_kl_proxy"]), expected_kl, places=7)
            self.assertAlmostEqual(float(loss), -length / budget + beta * expected_kl, places=7)
            expected_gradient = (-1. - beta) / budget
            for value in new.grad.flatten():
                self.assertAlmostEqual(float(value), expected_gradient, places=7, msg=effort)
            observed.append(float(new.grad[0, 0]))
        self.assertEqual(observed, [observed[0]] * 3)

    def test_mixed_length_generated_token_alignment(self):
        from sts_llm.training import response_logps
        t = self.torch
        class TinyModel(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = t.nn.Parameter(t.arange(49, dtype=t.float32).reshape(7, 7) / 10)
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                self.test_use_cache = use_cache
                return SimpleNamespace(logits=self.weight[input_ids[:, logits_to_keep]])
        model = TinyModel()
        examples = [Completion("a", "combat", [1, 2], [3, 4], 0.),
                    Completion("b", "out_of_combat", [1, 3, 2], [5, 6, 1], 0.)]
        observed = response_logps(model, examples, pad_id=0, temperature=.7, chunk_size=1)
        for item, logps in zip(examples, observed):
            ids = item.prompt_ids + item.output_ids
            expected = []
            for position in range(len(item.prompt_ids), len(ids)):
                expected.append(t.log_softmax(model.weight[ids[position - 1]] / .7, dim=-1)[ids[position]])
            t.testing.assert_close(logps, t.stack(expected))
        self.assertFalse(model.test_use_cache)
        self.assertEqual([len(v) for v in observed], [2, 3])

    def test_expanded_dynamic_cache_is_independent_for_every_microbatch(self):
        from transformers import DynamicCache
        from sts_llm.training import expanded_prompt_cache
        t = self.torch
        prefix = DynamicCache()
        keys = t.arange(4, dtype=t.float32).reshape(1, 1, 2, 2)
        prefix.update(keys, keys + 10, 0)
        first = expanded_prompt_cache(prefix, 3)
        second = expanded_prompt_cache(prefix, 2)
        first.update(t.ones(3, 1, 1, 2), t.ones(3, 1, 1, 2), 0)
        self.assertEqual(prefix.get_seq_length(), 2)
        self.assertEqual(first.get_seq_length(), 3)
        self.assertEqual(second.get_seq_length(), 2)
        self.assertEqual(prefix.layers[0].keys.shape[0], 1)
        self.assertEqual(first.layers[0].keys.shape[0], 3)
        self.assertEqual(second.layers[0].keys.shape[0], 2)
        t.testing.assert_close(prefix.layers[0].keys, keys)
        # No working cache tensor aliases the shared prefix or another batch.
        self.assertNotEqual(prefix.layers[0].keys.data_ptr(), second.layers[0].keys.data_ptr())


if __name__ == "__main__":
    unittest.main()
