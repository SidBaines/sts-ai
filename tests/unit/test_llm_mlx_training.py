"""MLX trainer tests on a tiny random Llama; skipped where mlx-lm is not installed."""
from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

try:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
except ImportError:  # pragma: no cover - depends on the optional MLX runtime
    mx = None

from sts_llm import mlx_training
from sts_llm.dataset import SPLIT_SCHEME_VERSION, canonical_world_identity, world_split
from sts_llm.observations import labelled_prompt


def args(**overrides):
    values = dict(temperature=0.6, kl_beta=0.0, loss_normalization="fixed_token_budget", loss_token_budget=16,
                  max_grad_norm=1e9, invalid_reward=-1.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_policy():
    mx.random.seed(0)
    model = Model(ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=40,
                            tie_word_embeddings=False))
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": list(mlx_training.LORA_KEYS)})
    policy = mlx_training.MLXPolicy.__new__(mlx_training.MLXPolicy)
    policy.mx, policy.model, policy.args = mx, model, args()
    return policy


@unittest.skipIf(mx is None, "mlx-lm is not installed")
class MLXTrainerTests(unittest.TestCase):
    def test_token_logps_score_exactly_the_generated_tokens(self):
        policy = tiny_policy()
        prompt, output = [1, 2, 3, 4], [5, 6, 7]
        logits = policy.model(mx.array([prompt + output]))[0].astype(mx.float32) / 0.6
        logps = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        expected = [logps[3 + i, token].item() for i, token in enumerate(output)]
        actual = policy.token_logps(prompt, output).tolist()
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=4)

    def test_positive_advantage_raises_and_negative_lowers_the_sampled_answer(self):
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                policy = tiny_policy()
                optimizer = optim.SGD(learning_rate=0.5)
                item = {"prompt_ids": [1, 2, 3], "output_token_ids": [9, 10, 11], "advantage": sign}
                before = policy.token_logps(item["prompt_ids"], item["output_token_ids"]).sum().item()
                step = mlx_training.update(policy, optimizer, [item, dict(item, advantage=0.0)], args())
                after = policy.token_logps(item["prompt_ids"], item["output_token_ids"]).sum().item()
                self.assertTrue(step["optimizer_step"])
                self.assertEqual(step["responses_with_gradient"], 1)
                self.assertGreater(sign * (after - before), 0)

    def test_flat_group_takes_no_step_and_adapters_can_be_disabled(self):
        policy = tiny_policy()
        optimizer = optim.SGD(learning_rate=0.5)
        item = {"prompt_ids": [1, 2], "output_token_ids": [3, 4], "advantage": 0.0}
        self.assertFalse(mlx_training.update(policy, optimizer, [item], args())["optimizer_step"])
        base = policy.token_logps([1, 2], [3, 4]).tolist()
        mlx_training.update(policy, optimizer, [dict(item, advantage=1.0)], args())
        self.assertNotEqual(policy.token_logps([1, 2], [3, 4]).tolist(), base)
        with policy.adapters_disabled():
            for a, b in zip(policy.token_logps([1, 2], [3, 4]).tolist(), base):
                self.assertAlmostEqual(a, b, places=5)
        self.assertNotEqual(policy.token_logps([1, 2], [3, 4]).tolist(), base)


class StateAndScoringTests(unittest.TestCase):
    def state(self, seed=11, split=None):
        world = canonical_world_identity(seed)
        return {"id": f"s{seed}", "phase": "out_of_combat", "world_identity": world,
                "split": split or world_split(world), "split_scheme_version": SPLIT_SCHEME_VERSION,
                "observation": {"phase": "out_of_combat", "text": "Rest site",
                                "legal_actions": [{"index": 0, "description": "rest"},
                                                  {"index": 1, "description": "smith"}]},
                "legal_indices": [0, 1],
                "teacher": {"legal_indices": [0, 1], "probabilities": [0.9, 0.1], "excluded_indices": []}}

    def test_states_must_come_from_the_requested_split(self):
        train_seed = next(s for s in range(100) if world_split(s) == "train")
        held_out_seed = next(s for s in range(100) if world_split(s) == "validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.jsonl"
            path.write_text(json.dumps(self.state(train_seed)) + "\n")
            self.assertEqual(len(mlx_training.load_states(path, split="train")), 1)
            path.write_text(json.dumps(self.state(held_out_seed)) + "\n")
            with self.assertRaises(ValueError):
                mlx_training.load_states(path, split="train")
            path.write_text(json.dumps(self.state(held_out_seed, split="train")) + "\n")
            with self.assertRaises(ValueError):
                mlx_training.load_states(path, split="train")

    def test_answers_are_scored_by_native_index_through_shuffled_labels(self):
        state = self.state()
        prompt = labelled_prompt(state["observation"], order=[1, 0])
        good = mlx_training.score_answer(state, prompt, [5, 64030, 6], 'x</ifm|think>{"action": "B"}', "stop", -1.0)
        self.assertEqual((good["action_index"], good["label"], good["reward"]), (0, "B", 0.9))
        self.assertTrue(good["teacher_top_agreement"])
        self.assertFalse(good["first_displayed_option"])
        self.assertTrue(good["native_first_option"])
        self.assertEqual((good["reasoning_tokens"], good["reasoning_closed"]), (1, True))
        capped = mlx_training.score_answer(state, prompt, [5, 6], "x", "length", -1.0)
        self.assertEqual((capped["valid"], capped["reward"], capped["teacher_probability"]), (False, -1.0, 0.0))
        self.assertTrue(capped["capped"])

    def test_menu_orders_are_permutations_and_reproducible(self):
        state = self.state()
        first = [mlx_training.menu_order(random.Random(3), state) for _ in range(3)]
        again = [mlx_training.menu_order(random.Random(3), state) for _ in range(3)]
        self.assertEqual(first, again)
        self.assertTrue(all(sorted(order) == [0, 1] for order in first))

    def test_group_statistics_and_summary(self):
        group = [{"reward": 0.9, "action_index": 0}, {"reward": 0.9, "action_index": 0}]
        mixed = [{"reward": 0.9, "action_index": 0}, {"reward": 0.1, "action_index": 1}]
        stats = mlx_training.group_statistics([group, mixed])
        self.assertEqual(stats["zero_variance_group_fraction"], 0.5)
        self.assertAlmostEqual(stats["mean_group_reward_range"], 0.4)
        self.assertEqual(stats["mean_distinct_actions_per_group"], 1.5)


if __name__ == "__main__":
    unittest.main()
