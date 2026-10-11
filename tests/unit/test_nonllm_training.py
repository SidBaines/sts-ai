from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import random
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from sts_policy.losses import PPOConfig, generalized_advantages, masked_distribution, ppo_objective
    from sts_policy.training import load_checkpoint, save_checkpoint, train_bc
    from sts_policy.rollout_training import Transition, collect_rollouts, update_ppo


@unittest.skipIf(torch is None, "optional torch training extra unavailable")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_gae_terminal_and_timelimit_bootstrap_without_reset_leak(self):
        advantages, returns = generalized_advantages([1., 2., 3.], [0.5, 1., 2.],
            [False, False, True], [True, False, False], [10., 2., 99.], gamma=.9, gae_lambda=1.)
        self.assertAlmostEqual(advantages[0], 9.5)
        self.assertAlmostEqual(advantages[1], 3.7)
        self.assertAlmostEqual(advantages[2], 1.)
        self.assertAlmostEqual(returns[0], 10.)

    def test_clipped_ppo_uses_stored_behavior_and_zero_illegal_mass(self):
        logits = torch.tensor([[3., 0., 99.]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        distribution = masked_distribution(logits, mask)
        self.assertEqual(float(distribution.probs.detach()[0, 2]), 0.)
        old = torch.tensor([-.69314718])
        loss, stats = ppo_objective(logits, mask, torch.tensor([0.]), torch.tensor([0]), old,
                                    torch.tensor([1.]), torch.tensor([0.]), PPOConfig(entropy_weight=0.))
        self.assertGreater(float(stats["ratio_mean"]), 1.8)
        self.assertEqual(float(stats["clip_fraction"]), 1.)
        self.assertAlmostEqual(float(loss.detach()), -1.2, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_checkpoint_restores_policy_optimizer_rng_and_identity(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            random.seed(42)
            torch.manual_seed(42)
            expected = model(torch.ones(1, 2)).detach().clone()
            save_checkpoint(path, model, model_config={"input": 2}, optimizer=optimizer,
                            step=7, metadata={"data_identity": "abc"})
            python_sample, torch_sample = random.random(), torch.rand(3)
            with torch.no_grad():
                model.weight.fill_(99.)
            payload = load_checkpoint(path, model, optimizer=optimizer, restore_rng=True,
                                      expected_data_identity="abc")
            self.assertEqual(payload["step"], 7)
            torch.testing.assert_close(model(torch.ones(1, 2)), expected)
            self.assertEqual(random.random(), python_sample)
            torch.testing.assert_close(torch.rand(3), torch_sample)
            self.assertTrue(optimizer.state)
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(path, expected_data_identity="wrong")
            self.assertEqual(list(Path(directory).glob("*.tmp-*")), [])

    def test_bc_learns_routing_indices_not_positions(self):
        model = TinyModel()
        record = {"observation": observation(), "teacher_index": 9}
        initial = float(torch.softmax(model.weights.detach(), 0)[1])
        result = train_bc(model, [record] * 64, collate=tiny_collate, forward=tiny_forward,
                          learning_rate=.1, batch_size=16, epochs=5)
        self.assertGreater(float(torch.softmax(model.weights.detach(), 0)[1]), initial + .2)
        self.assertEqual(result["steps"], 20)

    def test_collection_fault_excludes_whole_episode_and_dagger_labels(self):
        result = collect_rollouts(TinyModel(), [FakeWorker()], [FakeTask(1), FakeTask(2)],
                                   collate=tiny_collate, forward=tiny_forward,
                                   teacher_simulations=1, max_seconds=10.)
        self.assertEqual(len(result["faults"]), 1)
        self.assertEqual(result["excluded_transitions"], 1)
        self.assertEqual({t.world_seed for t in result["transitions"]}, {1})
        self.assertEqual({r["world_seed"] for r in result["teacher_records"]}, {1})
        self.assertEqual(len(result["transitions"]), 2)
        self.assertTrue(result["transitions"][-1].terminated)
        self.assertEqual(result["transitions"][-1].next_value, 0.)

    def test_ppo_changes_weights_and_ratios_beyond_first_epoch(self):
        model = TinyModel()
        transitions = [Transition(observation(), position, index, -.69314718, 0., [True, True],
                                  reward, True, False, 0., 1, reward, reward)
                       for position, index, reward in [(0, 7, -1.), (1, 9, 1.)] * 4]
        result = update_ppo(model, transitions, collate=tiny_collate, forward=tiny_forward,
                            config=PPOConfig(learning_rate=.05, epochs=3, minibatch_size=8, target_kl=10.))
        self.assertEqual(result["steps"], 3)
        self.assertGreater(float((model.weights[1] - model.weights[0]).detach()), .1)
        self.assertGreater(result["approx_kl"], 0.)

    def test_cli_checkpoint_prediction_is_preserved_after_loading(self):
        from sts_policy.model import CandidateActorCritic
        from sts_policy.model_agent import ModelAgent, collate_public, forward_policy
        obs = observation()
        model = CandidateActorCritic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            from sts_policy.encoding import ENCODER_VERSION
            save_checkpoint(path, model, model_config=model.metadata(), metadata={"encoder_version": ENCODER_VERSION})
            agent = ModelAgent(str(path))
            logits, _, _ = forward_policy(model, collate_public([obs]))
            expected = obs["legal_actions"][int(logits.argmax(-1)[0])]["index"]
            self.assertEqual(agent.choose_action(obs), expected)

    def test_composite_tensor_and_agent_dispatch_use_public_phase(self):
        from sts_policy.model_agent import CompositeActorCritic, CompositeAgent
        class Fixed(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, state, actions, mask):
                return torch.full(mask.shape, self.value), torch.full((len(state),), self.value)
        composite = CompositeActorCritic(Fixed(3.), Fixed(-2.))
        state = torch.zeros(2, 256)
        state[0, 7] = 1.
        logits, values = composite(state, torch.zeros(2, 2, 128), torch.ones(2, 2, dtype=torch.bool))
        torch.testing.assert_close(values, torch.tensor([3., -2.]))
        torch.testing.assert_close(logits[:, 0], values)
        class FixedAgent:
            def __init__(self, index): self.index = index
            def choose_actions(self, observations): return [self.index] * len(observations)
        agent = CompositeAgent.__new__(CompositeAgent)
        agent.combat, agent.ooc = FixedAgent(7), FixedAgent(9)
        combat = observation()
        ooc = {**combat, "phase": "out_of_combat"}
        self.assertEqual(agent.choose_actions([ooc, combat, ooc]), [9, 7, 9])

    def test_cached_tensor_collation_and_gradients_match_including_large_menu(self):
        from sts_policy.model_agent import collate_public, cached_public_collator, forward_policy
        from sts_policy.model import CandidateActorCritic
        from sts_policy.losses import behavioral_cloning_loss
        import copy
        observations = []
        for count in (1, 2, 3, 65):
            obs = observation()
            obs["legal_actions"] = [{"index": i + 7, "description": f"Play strike target {i}"} for i in range(count)]
            observations.append(obs)
        cached = cached_public_collator(observations, tensor_cache=True)
        for selection in (observations, observations[2:3], [observations[3], observations[0]]):
            ordinary_batch, cached_batch = collate_public(selection), cached(selection)
            for ordinary, actual in zip(ordinary_batch, cached_batch):
                torch.testing.assert_close(ordinary, actual, rtol=0., atol=0.)
            first = CandidateActorCritic()
            second = copy.deepcopy(first)
            for model, batch in ((first, ordinary_batch), (second, cached_batch)):
                logits, _, mask = forward_policy(model, batch)
                behavioral_cloning_loss(logits, mask, torch.zeros(len(selection), dtype=torch.long)).backward()
            for expected, actual in zip(first.parameters(), second.parameters()):
                if expected.grad is not None:
                    torch.testing.assert_close(expected.grad, actual.grad, rtol=0., atol=0.)

    def test_metric_batching_preserves_bc_updates_and_statistics(self):
        records = [{"observation": observation(), "teacher_index": 9}] * 65
        first, second = TinyModel(), TinyModel()
        one = train_bc(first, records, collate=tiny_collate, forward=tiny_forward,
                       batch_size=16, epochs=3, seed=4, metric_interval=1)
        many = train_bc(second, records, collate=tiny_collate, forward=tiny_forward,
                        batch_size=16, epochs=3, seed=4, metric_interval=32)
        torch.testing.assert_close(first.weights, second.weights, rtol=0., atol=0.)
        for key in ("loss", "accuracy", "steps", "examples"):
            self.assertEqual(one[key], many[key])

    def test_action_bucket_epoch_keeps_every_complex_menu(self):
        seen = []
        records = []
        for count in (1, 2, 3, 5, 65, 1024):
            obs = observation()
            obs["legal_actions"] = [{"index": i + 7, "description": "Play strike"} for i in range(count)]
            records.append({"observation": obs, "teacher_index": 7})
        def track(batch, device="cpu"):
            seen.extend(id(obs) for obs in batch)
            return tiny_collate(batch, device)
        result = train_bc(TinyModel(), records, collate=track, forward=tiny_forward,
                          batch_size=2, epochs=2, bucket_actions=True)
        self.assertEqual(result["examples"], 12)
        for record in records:
            self.assertEqual(seen.count(id(record["observation"])), 2)

    def test_optimizer_moments_carry_only_compatible_ppo_stage(self):
        from sts_policy.training import warm_optimizer
        model = TinyModel()
        first = torch.optim.AdamW(model.parameters(), lr=.01)
        model.weights.sum().backward()
        first.step()
        reward = {"gamma": 1., "reward_mode": "potential"}
        payload = {"optimizer_state": first.state_dict(), "step": 12,
                   "metadata": {"stage": "ppo", "reward_config": reward, "data_identity": "old-bank",
                                "config": {"task_kind": "battle"}}}
        second = torch.optim.AdamW(model.parameters(), lr=.02)
        result = warm_optimizer(second, payload, method="ppo", task_kind="battle", reward_config=reward,
                                learning_rate=.02)
        self.assertTrue(result["restored"])
        self.assertEqual(second.param_groups[0]["lr"], .02)
        self.assertEqual(float(second.state[model.weights]["step"]), 1.)
        for method, kind, expected in (("bc", "battle", "training_method_changed"),
                                        ("ppo", "ooc", "task_kind_changed")):
            fresh = torch.optim.AdamW(model.parameters())
            outcome = warm_optimizer(fresh, payload, method=method, task_kind=kind,
                                     reward_config=reward, learning_rate=.001)
            self.assertEqual(outcome["reason"], expected)
            self.assertFalse(fresh.state)

    def test_terminal_hp_and_progress_rewards_are_explicit_and_bounded(self):
        from sts_policy.curriculum import reward_components
        previous = {"state": {"cur_hp": 80, "max_hp": 80, "floor": 3}}
        following = {"state": {"cur_hp": 40, "max_hp": 80, "floor": 5}}
        components = reward_components(previous, following, task_outcome="battle_victory", terminated=True,
                                       mode="progress", terminal_hp_weight=.25, progress_floor_weight=.2)
        self.assertEqual(components["terminal_hp"], .125)
        self.assertEqual(components["progress"], .4)
        self.assertEqual(components["terminal"], 1.)
        cutoff = reward_components(previous, following, task_outcome="", terminated=False,
                                   mode="progress", terminal_hp_weight=.25, progress_floor_weight=.2)
        self.assertEqual(cutoff["terminal_hp"], 0.)
        self.assertEqual(cutoff["terminal"], 0.)

    def test_cli_reward_flags_reach_worker_and_logged_configuration(self):
        import contextlib
        import importlib.util
        import io
        from unittest.mock import patch
        script = Path(__file__).resolve().parents[2] / "scripts" / "train_nonllm.py"
        spec = importlib.util.spec_from_file_location("nonllm_cli_test", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        received, logged = [], []
        class Worker:
            def __init__(self, config, timeout): received.append(config)
            def close(self): pass
        result = {"transitions": [], "teacher_records": [], "episodes": [], "faults": []}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = ["train_nonllm.py", "evaluate", "--checkpoint", "unused.pt", "--output", str(Path(directory)/"eval.pt"),
                    "--workers", "1", "--reward-mode", "progress", "--terminal-hp-weight", ".25",
                    "--progress-floor-weight", ".2"]
            with patch("sys.argv", args), patch.object(cli, "source_model", return_value=(TinyModel(), None)), \
                 patch.object(cli, "tasks_for", return_value=[FakeTask(1)]), patch.object(cli, "CurriculumWorker", Worker), \
                 patch.object(cli, "collect_rollouts", return_value=result), \
                 patch.object(cli, "write_metrics", side_effect=lambda path, metrics: logged.append(metrics)):
                cli.main()
        self.assertEqual(received[0]["terminal_hp_weight"], .25)
        self.assertEqual(received[0]["progress_floor_weight"], .2)
        self.assertEqual(logged[-1]["config"]["terminal_hp_weight"], .25)
        self.assertEqual(logged[-1]["config"]["progress_floor_weight"], .2)
        self.assertEqual(logged[-1]["reward_config"]["terminal_hp_weight"], .25)
        self.assertEqual(logged[-1]["reward_config"]["progress_floor_weight"], .2)

    def test_collector_zero_step_budget_preserves_reset_info_and_censors_cohort(self):
        class MetadataWorker(FakeWorker):
            def submit(self, method, *args):
                future = super().submit(method, *args)
                if method == "reset":
                    obs, _ = future.result()
                    future = Future()
                    future.set_result((obs, {"hp_cost": 0., "split": "dev", "native_steps": 4,
                                             "native_provenance": {"build": "abc"}, "reward_version": "test"}))
                return future
        result = collect_rollouts(TinyModel(), [MetadataWorker()], [FakeTask(1), FakeTask(3)],
                                  collate=tiny_collate, forward=tiny_forward, max_seconds=0.)
        self.assertEqual(result["requested_tasks"], 2)
        self.assertEqual(result["unstarted_tasks"], 1)
        self.assertFalse(result["cohort_completed"])
        episode = result["episodes"][0]
        self.assertEqual(episode["steps"], 0)
        self.assertTrue(episode["truncated"])
        self.assertFalse(episode["terminated"])
        self.assertEqual(episode["termination_reason"], "collector_budget")
        self.assertEqual(episode["split"], "dev")
        self.assertEqual(episode["native_steps"], 4)
        self.assertEqual(episode["native_provenance"], {"build": "abc"})

    def test_collector_step_budget_preserves_step_info_and_never_marks_cohort_complete(self):
        from unittest.mock import patch
        elapsed = {"seconds": 0.}
        class StepMetadataWorker(FakeWorker):
            def submit(self, method, *args):
                future = super().submit(method, *args)
                if method == "step":
                    obs, reward, terminated, truncated, _ = future.result()
                    future = Future()
                    future.set_result((obs, reward, terminated, truncated,
                                       {"hp_cost": 5., "split": "dev", "native_steps": 10, "task_outcome": ""}))
                    elapsed["seconds"] = 2.
                return future
        # First decision completes, then the collector wall budget expires.
        with patch("sts_policy.rollout_training.time.monotonic", side_effect=lambda: elapsed["seconds"]):
            result = collect_rollouts(TinyModel(), [StepMetadataWorker()], [FakeTask(1)],
                                      collate=tiny_collate, forward=tiny_forward, max_seconds=1.)
        self.assertFalse(result["cohort_completed"])
        self.assertEqual(result["unstarted_tasks"], 0)
        episode = result["episodes"][0]
        self.assertEqual(episode["termination_reason"], "collector_budget")
        self.assertEqual(episode["steps"], 1)
        self.assertEqual(episode["hp_cost"], 5.)
        self.assertEqual(episode["native_steps"], 10)
        self.assertTrue(episode["truncated"])


def observation():
    return {"phase": "combat", "state": {"cur_hp": 80, "max_hp": 80},
            "legal_actions": [{"index": 7, "description": "End turn"},
                              {"index": 9, "description": "Play strike"}]}


if torch is not None:
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.zeros(2))
            self.value = torch.nn.Parameter(torch.zeros(1))

        def forward(self, count):
            return self.weights[None].expand(count, -1), self.value.expand(count)


def tiny_collate(observations, device="cpu"):
    return len(observations), device


def tiny_forward(model, batch):
    count, device = batch
    logits, values = model(count)
    return logits, values, torch.ones(count, 2, dtype=torch.bool, device=device)


@dataclass
class FakeTask:
    seed: int
    kind: str = "battle"


class FakeWorker:
    def submit(self, method, *args):
        result = Future()
        try:
            if method == "reset":
                self.seed, self.steps = args[0].seed, 0
                output = observation(), {}
            elif method == "teacher_action":
                output = 9
            else:
                self.steps += 1
                if self.seed == 2 and self.steps == 2:
                    raise RuntimeError("simulator fault")
                output = observation(), 1., self.steps == 2, False, {"outcome": "victory"}
            result.set_result(output)
        except Exception as error:
            result.set_exception(error)
        return result
