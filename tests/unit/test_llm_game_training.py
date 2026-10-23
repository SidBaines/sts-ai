"""Full-game credit/failure boundaries over fake clean environments; no GPU runs."""
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sts_llm.game_training import (Episode, CollectingAgent, StopRequested, collect_episode, credit_groups,
    optimize_games, terminal_result, save_checkpoint, settings, validate_world_seeds, validate_full_game_scope)
from sts_llm.training import Completion


def observation(phase="combat", indices=(0, 1)):
    return {"phase": phase, "text": "Public game state", "state": {"outcome": "UNDECIDED"},
            "legal_actions": [{"index": i, "description": f"public action {i}"} for i in indices]}


def summary(outcome, *, truncated=False, error=None):
    return {"status": "terminated" if not truncated else "truncated", "terminated": not truncated,
            "truncated": truncated, "error": error, "info": {"outcome": outcome}}


def completion(number=0, phase="combat", *, text=None, truncated=False):
    return Completion(str(number), phase, [1, 2], [3, 4], 0.,
        raw_text=text if text is not None else '</ifm|think>{"action_index": 1}<|ifm|im_end|>',
        truncated=truncated)


class FakeEnv:
    def __init__(self, *, forced=False, truncation=False, fault=False, two_phases=False):
        self.forced, self.truncation, self.fault, self.two_phases = forced, truncation, fault, two_phases
        self.actions, self.closed = [], False

    def reset(self, seed):
        return observation(indices=(7,) if self.forced else (0, 1)), {"terminated": False,
            "truncated": False, "outcome": "UNDECIDED", "secret": "PRIVILEGED", "world_seed": seed}

    def step(self, action):
        self.actions.append(action)
        if self.fault:
            raise TimeoutError("native operation timed out")
        if self.two_phases and len(self.actions) == 1:
            return observation("out_of_combat"), 0., False, False, {"outcome": "UNDECIDED"}
        outcome = "UNDECIDED" if self.truncation else "PLAYER_VICTORY"
        return observation(), 0. if self.truncation else 1., not self.truncation, self.truncation, {"outcome": outcome}

    def close(self):
        self.closed = True


class GameContracts(unittest.TestCase):
    def test_world_seed_boundaries_match_clean_environment(self):
        from sts_env import SlayTheSpireEnv
        accepted = [0, 2**63 - 1]
        self.assertEqual(validate_world_seeds(accepted), accepted)
        for seed in accepted:
            SlayTheSpireEnv._validate_seed(seed)
        for seed in (-1, 2**63, 2**64 - 1, True, 1.5):
            with self.assertRaises(ValueError): validate_world_seeds([seed])
            with self.assertRaises(ValueError): SlayTheSpireEnv._validate_seed(seed)
        for bad in ([], [1, 1], "1", None):
            with self.assertRaises(ValueError): validate_world_seeds(bad)

    def test_training_and_evaluation_require_all_three_acts(self):
        validate_full_game_scope(3)
        for maximum in (1, 2, 4, True, "3"):
            with self.assertRaisesRegex(ValueError, "training and evaluation require"):
                validate_full_game_scope(maximum)

    def collect(self, env, policy=None):
        with tempfile.TemporaryDirectory() as directory, patch("sts_env.cli.make_env", return_value=env):
            output = Path(directory) / "episode"
            episode = collect_episode(policy or (lambda messages, **kw: completion(phase=kw["phase"])),
                world_seed=5, policy_seed=19, env_config={"mode": "full", "max_steps": 20}, output=output,
                win_reward=1., loss_reward=-1.)
            trace = [json.loads(line) for line in (output / "game.jsonl").read_text().splitlines()]
            responses = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
        self.assertTrue(env.closed)
        return episode, trace, responses

    def test_terminal_outcome_only_and_fault_truncation_never_game_losses(self):
        self.assertEqual(terminal_result(summary("GameOutcome.PLAYER_VICTORY"), win_reward=1, loss_reward=-1),
                         ("complete", "PLAYER_VICTORY", 1.))
        self.assertEqual(terminal_result(summary("PLAYER_LOSS"), win_reward=1, loss_reward=-1),
                         ("complete", "PLAYER_LOSS", -1.))
        for result in (summary("PLAYER_LOSS", truncated=True), summary("PLAYER_LOSS", error={"kind":"SimulatorError"}),
                       summary("UNDECIDED")):
            self.assertIsNone(terminal_result(result, win_reward=1, loss_reward=-1)[2])

    def test_current_policy_controls_combat_and_ooc_and_prompts_stay_public(self):
        messages_seen = []
        def policy(messages, **kwargs):
            messages_seen.append(messages)
            return completion(phase=kwargs["phase"])
        game, trace, responses = self.collect(FakeEnv(two_phases=True), policy)
        self.assertEqual(game.reward, 1.)
        self.assertEqual([d.phase for d in game.decisions], ["combat", "out_of_combat"])
        self.assertNotIn("PRIVILEGED", json.dumps(messages_seen))
        self.assertEqual([r["type"] for r in trace], ["header", "reset", "action", "step", "action", "step", "end"])
        self.assertEqual(len(responses), 2)

    def test_forced_menus_never_generate_or_receive_policy_tokens(self):
        def forbidden(*args, **kwargs): raise AssertionError("forced action generated tokens")
        game, trace, responses = self.collect(FakeEnv(forced=True), forbidden)
        self.assertEqual(game.reward, 1.)
        self.assertEqual(game.decisions, [])
        self.assertEqual(game.forced_actions, 1)
        self.assertEqual(responses[0]["action_index"], 7)
        self.assertEqual(next(r for r in trace if r["type"] == "action")["action"], 7)

    def test_native_fault_and_decision_truncation_remain_unscored(self):
        fault, _, _ = self.collect(FakeEnv(fault=True))
        truncated, _, _ = self.collect(FakeEnv(truncation=True))
        self.assertEqual((fault.status, fault.reward), ("failed", None))
        self.assertEqual((truncated.status, truncated.reward), ("incomplete", None))

    def test_invalid_or_capped_model_output_is_not_executed_or_scored_as_defeat(self):
        for item in (completion(text='<ifm|think>{"action_index":1}'), completion(truncated=True)):
            env = FakeEnv()
            game, trace, responses = self.collect(env, lambda *a, **kw: item)
            self.assertEqual(env.actions, [])
            self.assertEqual((game.status, game.reward), ("failed", None))
            self.assertEqual(game.summary["error"]["kind"], "InvalidDecision")
            self.assertFalse(responses[0]["parse"]["valid"])

    def test_incomplete_group_is_excluded_in_full_even_if_one_game_won(self):
        win = Episode(5, 1, "complete", "PLAYER_VICTORY", 1., {}, [completion()])
        incomplete = Episode(5, 2, "incomplete", None, None, {}, [completion()])
        eligible, report = credit_groups([[win, incomplete]], expected_group_size=2)
        self.assertEqual(eligible, [])
        self.assertFalse(report[0]["eligible"])

    def test_short_group_is_excluded_and_mixed_world_group_is_rejected(self):
        game = Episode(5, 1, "complete", "PLAYER_VICTORY", 1., {}, [completion()])
        self.assertEqual(credit_groups([[game]], expected_group_size=2)[0], [])
        other = Episode(6, 2, "complete", "PLAYER_LOSS", -1., {}, [completion()])
        with self.assertRaises(ValueError): credit_groups([[game, other]], expected_group_size=2)

    def test_stop_between_decisions_never_generates_an_action(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory)/"STOP"; stop.touch()
            def forbidden(*a, **kw): raise AssertionError("generated after STOP")
            agent = CollectingAgent(forbidden, io.StringIO(), stop)
            agent.reset(1)
            with self.assertRaises(StopRequested): agent.choose_action(observation())

    def test_terminal_credit_reaches_every_reasoning_action_response(self):
        win = Episode(5, 1, "complete", "PLAYER_VICTORY", 1., {}, [completion(0), completion(1, "out_of_combat")])
        lose = Episode(5, 2, "complete", "PLAYER_LOSS", -1., {}, [completion(2)])
        eligible, report = credit_groups([[win, lose]], expected_group_size=2)
        self.assertEqual([d.advantage for g in eligible for d in g.decisions], [1., 1., -1.])
        self.assertEqual([d.reward for g in eligible for d in g.decisions], [1., 1., -1.])
        self.assertEqual(report[0]["reward_variance"], 1.)

    def test_old_and_reference_logps_frozen_once_before_all_updates_and_game_scaling(self):
        games = [Episode(5, 1, "complete", "PLAYER_VICTORY", 1., {}, [completion()]),
                 Episode(5, 2, "complete", "PLAYER_LOSS", -1., {}, [completion(i) for i in range(3)])]
        eligible, _ = credit_groups([games], expected_group_size=2)
        events = []
        def freeze(model, cohort, **kwargs):
            events.append("freeze")
            for item in cohort: item.old_logps, item.ref_logps = [-3., -4.], [-5., -6.]
        def update(model, optimizer, cohort, args, pad):
            events.append("update")
            self.assertEqual([d.old_logps for d in cohort], [[-3., -4.]]*4)
            self.assertEqual([d.ref_logps for d in cohort], [[-5., -6.]]*4)
            self.assertEqual([d.advantage for d in cohort], [2., -2., -2., -2.])
            self.assertEqual(args.kl_beta, .4)  # N/G times original .2.
            return {"loss": 0.}
        args = SimpleNamespace(microbatch=1, temperature=.7, kl_beta=.2, update_epochs=2)
        with patch("sts_llm.game_training.freeze_cohort_logps", side_effect=freeze), \
             patch("sts_llm.game_training.optimizer_step", side_effect=update):
            cohort, updates, factor = optimize_games(None, None, eligible, args, 0)
        self.assertEqual(events, ["freeze", "update", "update"])
        self.assertEqual(factor, 2.)
        self.assertEqual(len(updates), 2)
        self.assertEqual(cohort[0].advantage / len(cohort), 1. / len(games))
        self.assertEqual(args.kl_beta, .2)  # Caller settings remain unchanged.

    def test_zero_decision_games_need_no_optimizer_or_probability_scoring(self):
        game = Episode(5, 1, "complete", "PLAYER_VICTORY", 1., {}, [])
        args = SimpleNamespace()
        with patch("sts_llm.game_training.freeze_cohort_logps") as freeze, \
             patch("sts_llm.game_training.optimizer_step") as update:
            self.assertEqual(optimize_games(None, None, [game], args, 0), ([], [], 0.))
        freeze.assert_not_called(); update.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch needed for checkpoint roundtrip")
class CheckpointContracts(unittest.TestCase):
    def test_checkpoint_preserves_optimizer_rng_and_refuses_overwrite(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([1.]))
            def save_pretrained(self, path): Path(path).mkdir(); torch.save(self.state_dict(), Path(path)/"weights.pt")
        class Tokenizer:
            def save_pretrained(self, path): (Path(path)/"tokenizer.json").write_text("{}")
        model, rng = Model(), random.Random(7)
        optimizer = torch.optim.AdamW(model.parameters())
        (model.weight**2).sum().backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(output=Path(directory), name="adapter_initial", next_iteration=0,
                          history=[], invariants={"format_version":1})
            path = save_checkpoint(model, Tokenizer(), optimizer, rng, **kwargs)
            state = torch.load(Path(path)/"game_training_state.pt", weights_only=False)
            self.assertEqual(state["python_rng"], rng.getstate())
            self.assertTrue(torch.equal(state["torch_rng"], torch.get_rng_state()))
            self.assertEqual(state["next_iteration"], 0)
            self.assertEqual(len(state["optimizer"]["state"]), 1)
            with self.assertRaises(FileExistsError): save_checkpoint(model, Tokenizer(), optimizer, rng, **kwargs)


if __name__ == "__main__":
    unittest.main()
