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

from sts_llm.game_training import (Episode, CollectingAgent, ContextBudgetExceeded, StopRequested,
    adapter_training_worlds, collect_episode, credit_groups, evaluation_world_check, optimize_games,
    outcome_counts, terminal_result, save_checkpoint, settings, validate_world_seeds, validate_full_game_scope)
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

    def collect(self, env, policy=None, *, stop_file=None):
        with tempfile.TemporaryDirectory() as directory, patch("sts_env.cli.make_env", return_value=env):
            output = Path(directory) / "episode"
            episode = collect_episode(policy or (lambda messages, **kw: completion(phase=kw["phase"])),
                world_seed=5, policy_seed=19, env_config={"mode": "full", "max_steps": 20}, output=output,
                win_reward=1., loss_reward=-1., invalid_game_reward=-2., stop_file=stop_file)
            trace = [json.loads(line) for line in (output / "game.jsonl").read_text().splitlines()]
            responses = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            result = json.loads((output / "result.json").read_text())
        self.assertTrue(env.closed)
        self.assertEqual(result["status"], episode.status)
        return episode, trace, responses

    def collect_raising(self, env, policy, expected, *, stop_file=None):
        """Collect a game whose policy error must surface; return the retained result.json."""
        with tempfile.TemporaryDirectory() as directory, patch("sts_env.cli.make_env", return_value=env):
            output = Path(directory) / "episode"
            with self.assertRaises(expected):
                collect_episode(policy, world_seed=5, policy_seed=19, env_config={"mode": "full", "max_steps": 20},
                                output=output, win_reward=1., loss_reward=-1., invalid_game_reward=-1.,
                                stop_file=stop_file)
            self.assertTrue(env.closed)
            return json.loads((output / "result.json").read_text())

    def test_terminal_outcome_only_and_fault_truncation_never_game_losses(self):
        self.assertEqual(terminal_result(summary("GameOutcome.PLAYER_VICTORY"), win_reward=1, loss_reward=-1),
                         ("win", "PLAYER_VICTORY", 1.))
        self.assertEqual(terminal_result(summary("PLAYER_LOSS"), win_reward=1, loss_reward=-1),
                         ("loss", "PLAYER_LOSS", -1.))
        cases = ((summary("PLAYER_LOSS", truncated=True), "budget_truncated"),
                 (summary("PLAYER_LOSS", error={"kind": "SimulatorError"}), "simulator_fault"),
                 (summary("UNDECIDED"), "unknown_terminal"),
                 ({**summary("PLAYER_LOSS", truncated=True), "status": "interrupted"}, "stopped"))
        for result, status in cases:
            self.assertEqual(terminal_result(result, win_reward=1, loss_reward=-1, invalid_game_reward=-1),
                             (status, result["info"]["outcome"].rsplit(".", 1)[-1] if status == "unknown_terminal"
                              else None, None))

    def test_agent_terminations_override_the_generic_error_summary(self):
        error = summary("UNDECIDED", truncated=True, error={"kind": "InvalidDecision"})
        self.assertEqual(terminal_result(error, win_reward=1, loss_reward=-1, invalid_game_reward=-3,
                                         termination={"category": "model_invalid"}), ("model_invalid", None, -3.))
        for category in ("budget_truncated", "stopped"):
            self.assertEqual(terminal_result(error, win_reward=1, loss_reward=-1, invalid_game_reward=-1,
                                             termination={"category": category}), (category, None, None))
        with self.assertRaises(ValueError):
            terminal_result(error, win_reward=1, loss_reward=-1, termination={"category": "model_invalid"})

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
        self.assertEqual((fault.status, fault.reward), ("simulator_fault", None))
        self.assertEqual((truncated.status, truncated.reward), ("budget_truncated", None))

    def test_invalid_or_capped_model_output_is_not_executed_and_is_scored_as_model_invalid(self):
        for item in (completion(text='<ifm|think>{"action_index":1}'), completion(truncated=True),
                     completion(text='</ifm|think>{"action_index": 9}<|ifm|im_end|>')):
            env = FakeEnv()
            game, trace, responses = self.collect(env, lambda *a, **kw: item)
            self.assertEqual(env.actions, [])
            self.assertEqual((game.status, game.reward), ("model_invalid", -2.))
            self.assertEqual(game.summary["error"]["kind"], "InvalidDecision")
            self.assertEqual(game.termination["category"], "model_invalid")
            # The invalid completion is retained so it can receive the game's advantage.
            self.assertEqual(len(game.decisions), 1)
            self.assertFalse(game.decisions[0].valid)
            self.assertFalse(responses[0]["parse"]["valid"])
            self.assertEqual(responses[-1]["event"], "termination")

    def test_context_budget_is_an_unscored_budget_truncation(self):
        def too_long(*args, **kwargs): raise ContextBudgetExceeded("prompt_plus_completion_budget_exceeds_context")
        env = FakeEnv()
        game, _, _ = self.collect(env, too_long)
        self.assertEqual((game.status, game.reward, game.decisions), ("budget_truncated", None, []))
        self.assertEqual(env.actions, [])

    def test_policy_crash_is_reraised_instead_of_becoming_an_unscored_game(self):
        def crash(*args, **kwargs): raise RuntimeError("CUDA out of memory")
        result = self.collect_raising(FakeEnv(), crash, RuntimeError)
        self.assertEqual(result["status"], "policy_error")
        self.assertEqual(result["termination"]["kind"], "RuntimeError")
        def interrupt(*args, **kwargs): raise KeyboardInterrupt
        self.assertEqual(self.collect_raising(FakeEnv(), interrupt, KeyboardInterrupt)["status"], "policy_error")

    def test_stop_inside_a_game_is_recorded_as_stopped_and_reraised(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            def stop_after_first(messages, **kwargs):
                stop.touch()
                return completion(phase=kwargs["phase"])
            result = self.collect_raising(FakeEnv(two_phases=True), stop_after_first, StopRequested, stop_file=stop)
        self.assertEqual((result["status"], result["reward"]), ("stopped", None))

    def test_model_invalid_game_stays_in_group_and_its_invalid_completion_gets_the_advantage(self):
        invalid_completion = completion(2, text="no boundary")
        invalid_completion.valid = False
        win = Episode(5, 1, "win", "PLAYER_VICTORY", 1., {}, [completion(0)])
        invalid = Episode(5, 2, "model_invalid", None, -1., {}, [completion(1), invalid_completion])
        eligible, report = credit_groups([[win, invalid]], expected_group_size=2)
        self.assertEqual(eligible, [win, invalid])
        self.assertTrue(report[0]["eligible"])
        self.assertEqual([d.advantage for d in invalid.decisions], [-1., -1.])
        self.assertEqual(invalid_completion.reward, -1.)

    def test_incomplete_group_is_excluded_in_full_even_if_one_game_won(self):
        win = Episode(5, 1, "win", "PLAYER_VICTORY", 1., {}, [completion()])
        for status in ("simulator_fault", "budget_truncated", "unknown_terminal", "stopped"):
            incomplete = Episode(5, 2, status, None, None, {}, [completion()])
            eligible, report = credit_groups([[win, incomplete]], expected_group_size=2)
            self.assertEqual(eligible, [])
            self.assertFalse(report[0]["eligible"])

    def test_short_group_is_excluded_and_mixed_world_group_is_rejected(self):
        game = Episode(5, 1, "win", "PLAYER_VICTORY", 1., {}, [completion()])
        self.assertEqual(credit_groups([[game]], expected_group_size=2)[0], [])
        other = Episode(6, 2, "loss", "PLAYER_LOSS", -1., {}, [completion()])
        with self.assertRaises(ValueError): credit_groups([[game, other]], expected_group_size=2)

    def test_outcome_counts_keep_categories_distinct_and_report_both_win_rates(self):
        games = [Episode(1, i, status, None, reward, {}) for i, (status, reward) in enumerate((
            ("win", 1.), ("loss", -1.), ("model_invalid", -1.), ("simulator_fault", None),
            ("budget_truncated", None)))]
        counts = outcome_counts(games)
        self.assertEqual({k: counts[k] for k in ("win", "loss", "model_invalid", "simulator_fault",
                          "budget_truncated", "unknown_terminal", "stopped")},
                         {"win": 1, "loss": 1, "model_invalid": 1, "simulator_fault": 1,
                          "budget_truncated": 1, "unknown_terminal": 0, "stopped": 0})
        self.assertEqual((counts["attempted"], counts["scored"]), (5, 3))
        self.assertEqual(counts["win_rate_over_attempted"], .2)
        self.assertAlmostEqual(counts["win_rate_over_scored"], 1 / 3)
        self.assertIsNone(outcome_counts([])["win_rate_over_attempted"])
        with self.assertRaises(ValueError):
            outcome_counts([Episode(1, 1, "failed", None, None, {})])

    def test_stop_between_decisions_never_generates_an_action(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory)/"STOP"; stop.touch()
            def forbidden(*a, **kw): raise AssertionError("generated after STOP")
            agent = CollectingAgent(forbidden, io.StringIO(), stop)
            agent.reset(1)
            with self.assertRaises(StopRequested): agent.choose_action(observation())

    def test_terminal_credit_reaches_every_reasoning_action_response(self):
        win = Episode(5, 1, "win", "PLAYER_VICTORY", 1., {}, [completion(0), completion(1, "out_of_combat")])
        lose = Episode(5, 2, "loss", "PLAYER_LOSS", -1., {}, [completion(2)])
        eligible, report = credit_groups([[win, lose]], expected_group_size=2)
        self.assertEqual([d.advantage for g in eligible for d in g.decisions], [1., 1., -1.])
        self.assertEqual([d.reward for g in eligible for d in g.decisions], [1., 1., -1.])
        self.assertEqual(report[0]["reward_variance"], 1.)

    def test_old_and_reference_logps_frozen_once_before_all_updates_and_game_scaling(self):
        games = [Episode(5, 1, "win", "PLAYER_VICTORY", 1., {}, [completion()]),
                 Episode(5, 2, "loss", "PLAYER_LOSS", -1., {}, [completion(i) for i in range(3)])]
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
        game = Episode(5, 1, "win", "PLAYER_VICTORY", 1., {}, [])
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


class HeldOutWorldContracts(unittest.TestCase):
    def test_full_game_adapter_worlds_include_inherited_teacher_worlds(self):
        teacher = adapter_training_worlds({"kind": "teacher_feedback", "path": "t",
            "run_config": {"training_world_identities": ["world:7", "world:8"]}})
        self.assertTrue(teacher["recorded"])
        record = {"kind": "full_game", "path": "g", "settings": {"world_seeds": [1, 2]},
                  "inherited_training_worlds": teacher}
        lineage = adapter_training_worlds(record)
        self.assertEqual(lineage["worlds"], ["world:1", "world:2", "world:7", "world:8"])
        self.assertTrue(lineage["recorded"])

    def test_old_teacher_runs_and_bare_adapters_are_reported_unrecorded(self):
        for record in ({"kind": "teacher_feedback", "path": "t", "run_config": None},
                       {"kind": "unrecorded", "path": "x"}):
            lineage = adapter_training_worlds(record)
            self.assertFalse(lineage["recorded"])
            self.assertEqual(lineage["worlds"], [])
            check = evaluation_world_check([1], lineage, allow_training_worlds=False)
            self.assertFalse(check["training_worlds_recorded"])
            self.assertIn("not recorded", check["result"])

    def test_evaluation_refuses_training_worlds_unless_explicitly_allowed(self):
        lineage = adapter_training_worlds({"kind": "full_game", "path": "g", "settings": {"world_seeds": [3, 4]}})
        with self.assertRaisesRegex(ValueError, "allow-training-worlds"):
            evaluation_world_check([4, 5], lineage, allow_training_worlds=False)
        allowed = evaluation_world_check([4, 5], lineage, allow_training_worlds=True)
        self.assertEqual(allowed["overlapping_worlds"], ["world:4"])
        self.assertTrue(allowed["overlap_allowed_by_flag"])
        clean = evaluation_world_check([5, 6], lineage, allow_training_worlds=False)
        self.assertEqual(clean["overlapping_worlds"], [])
        self.assertEqual(evaluation_world_check([1], None, allow_training_worlds=False)["adapter"], None)


def scripted_completion(prompt_id, phase, action):
    text = (f'</ifm|think>{{"action_index": {action}}}<|ifm|im_end|>' if action is not None
            else "unfinished thought")
    return Completion(prompt_id, phase, [1, 2], [3, 4], 0., raw_text=text)


class ScriptedEnv:
    """World 1 wins, 2 loses, 4 faults, 5 hits the decision limit; otherwise action 0 wins, 1 loses."""
    def __init__(self):
        self.seed, self.closed = None, False

    def reset(self, seed):
        self.seed = seed
        return observation(), {"terminated": False, "truncated": False, "outcome": "UNDECIDED"}

    def step(self, action):
        if self.seed == 4:
            raise TimeoutError("native operation timed out")
        if self.seed == 5:
            return observation(), 0., False, True, {"outcome": "UNDECIDED"}
        won = self.seed == 1 or self.seed not in (1, 2) and action == 0
        return observation(), float(won), True, False, {"outcome": "PLAYER_VICTORY" if won else "PLAYER_LOSS"}

    def close(self):
        self.closed = True


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch needed for the HF policy and iteration loop")
class LoopContracts(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def tiny_model(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.]))
            def save_pretrained(self, path):
                Path(path).mkdir()
                torch.save(self.state_dict(), Path(path) / "weights.pt")
        return Model()

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 50
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]
        def decode(self, tokens, skip_special_tokens):
            return " ".join(map(str, tokens))
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    def args(self, **overrides):
        values = dict(policy_seed=11, evaluate_only=False, learning_rate=1e-3, resume=None,
            output=self.root / "run", stop_file=self.root / "run" / "STOP", max_act=3, max_decisions=20,
            step_timeout=1., ascension=0, build_dir=None, worlds_per_iteration=1, group_size=3, iterations=4,
            model="fake/model", revision="r1", effort="medium", temperature=1., win_reward=1., loss_reward=-1.,
            invalid_game_reward=-1., microbatch=1, kl_beta=0., update_epochs=1, max_no_update_iterations=2,
            device="cpu", max_new_tokens=8, max_sequence_length=64, loss_normalization="response_mean",
            loss_token_budget=2048)
        values.update(overrides)
        values["output"].mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(**values)

    def run_execute(self, args, seeds, policy, *, updates=None):
        from sts_llm.game_training import execute
        def update(model, optimizer, cohort, update_args, pad):
            updates.append([(d.valid, d.advantage) for d in cohort])
            return {"loss": 0.}
        def freeze(model, cohort, **kwargs):
            for item in cohort: item.old_logps, item.ref_logps = [0., 0.], [0., 0.]
        with patch("sts_env.cli.make_env", side_effect=lambda config: ScriptedEnv()), \
             patch("sts_llm.game_training.HFPolicy", return_value=policy), \
             patch("sts_llm.game_training.freeze_cohort_logps", side_effect=freeze), \
             patch("sts_llm.game_training.optimizer_step", side_effect=update):
            return execute(self.tiny_model(), self.Tokenizer(), 99, args, seeds)

    def test_hf_policy_trims_at_first_stop_and_flags_cap_truncation(self):
        from sts_llm.game_training import HFPolicy
        torch = self.torch
        model = self.tiny_model()
        calls = []
        def generate(input_ids, **kwargs):
            calls.append(kwargs)
            return torch.cat([input_ids, torch.tensor([model.script])], dim=1)
        model.generate = generate
        policy = HFPolicy(model, self.Tokenizer(), 99, SimpleNamespace(effort="medium", max_new_tokens=8,
                          max_sequence_length=64, temperature=0.))
        for script, expected, truncated in (([5, 99, 7, 7], [5, 99], False), ([5, 50, 99], [5, 50], False),
                                            ([5, 6, 7], [5, 6, 7], True)):
            model.script = script
            item = policy([{"role": "user", "content": "x"}], seed=1, prompt_id="p", phase="combat")
            self.assertEqual((item.prompt_ids, item.output_ids, item.truncated), ([1, 2, 3], expected, truncated))
            self.assertEqual(item.raw_text, " ".join(map(str, expected)))
        self.assertEqual(calls[0]["eos_token_id"], [99, 50])
        self.assertFalse(calls[0]["do_sample"])
        policy.args.max_sequence_length = 10
        with self.assertRaises(ContextBudgetExceeded):
            policy([{"role": "user", "content": "x"}], seed=1, prompt_id="p", phase="combat")

    def test_invalid_game_is_scored_and_its_completion_trained_in_the_iteration_loop(self):
        actions = iter([0, 1, None])  # win, loss, invalid output in one group of three.
        policy = lambda messages, **kw: scripted_completion(kw["prompt_id"], kw["phase"], next(actions))
        updates = []
        result = self.run_execute(self.args(iterations=1), [7], policy, updates=updates)
        record = result["iterations"][0]
        self.assertEqual((record["wins"], record["losses"], record["model_invalid_games"]), (1, 1, 1))
        self.assertEqual((record["eligible_groups"], record["updated"], record["update_skipped"]), (1, True, None))
        [cohort] = updates
        self.assertEqual([valid for valid, _ in cohort], [True, True, False])
        self.assertGreater(cohort[0][1], 0)
        self.assertLess(cohort[2][1], 0)
        self.assertAlmostEqual(cohort[1][1], cohort[2][1])
        saved = json.loads((self.root / "run" / "collection_0000.json").read_text())
        self.assertEqual(saved["status_counts"]["model_invalid"], 1)

    def test_iterations_without_eligible_groups_are_recorded_and_stop_the_run(self):
        from sts_llm.game_training import NoUpdateLimitReached
        policy = lambda messages, **kw: scripted_completion(kw["prompt_id"], kw["phase"], 0)
        updates = []
        with self.assertRaisesRegex(NoUpdateLimitReached, "2 consecutive iterations"):
            self.run_execute(self.args(), [4], policy, updates=updates)
        self.assertEqual(updates, [])
        output = self.root / "run"
        for iteration in (0, 1):
            collection = json.loads((output / f"collection_{iteration:04d}.json").read_text())
            self.assertEqual((collection["eligible_groups"], collection["update_skipped"]), (0, "no_eligible_groups"))
            self.assertEqual(collection["status_counts"]["simulator_fault"], 3)
        self.assertEqual([r["updated"] for r in json.loads((output / "progress.json").read_text())], [False, False])
        self.assertFalse((output / "collection_0002.json").exists())
        self.assertFalse((output / "summary.json").exists())

    def test_evaluate_only_counts_every_category_and_reports_wins_over_attempted_games(self):
        def policy(messages, **kw):
            return scripted_completion(kw["prompt_id"], kw["phase"], None if "world_3:" in kw["prompt_id"] else 0)
        args = self.args(evaluate_only=True, group_size=1, iterations=1)
        result = self.run_execute(args, [1, 2, 3, 4, 5], policy, updates=[])
        evaluation = result["evaluation"]
        self.assertEqual({k: evaluation[k] for k in ("win", "loss", "model_invalid", "simulator_fault",
                          "budget_truncated", "attempted", "scored")},
                         {"win": 1, "loss": 1, "model_invalid": 1, "simulator_fault": 1, "budget_truncated": 1,
                          "attempted": 5, "scored": 3})
        self.assertEqual(evaluation["win_rate_over_attempted"], .2)
        self.assertFalse(any((self.root / "run").glob("adapter_*")))

    def run_main(self, *extra, policy=None):
        from sts_llm import game_training
        seeds = self.root / "seeds.json"
        seeds.write_text("[3, 4]")
        argv = ["game_training", "--output", str(self.root / "main"), "--seeds-file", str(seeds),
                "--policy-seed", "1", "--effort", "medium", "--temperature", "1", "--max-new-tokens", "8",
                "--max-sequence-length", "64", "--max-decisions", "5", "--max-act", "3", "--ascension", "0",
                "--step-timeout", "1", "--win-reward", "1", "--loss-reward", "-1", "--group-size", "2",
                "--loss-normalization", "response_mean", "--device", "cpu", *extra]
        with patch("sys.argv", argv), \
             patch("sts_llm.game_training.load_model", return_value=(self.tiny_model(), self.Tokenizer(), 99)), \
             patch("sts_env.cli.make_env", side_effect=lambda config: ScriptedEnv()), \
             patch("sts_llm.game_training.HFPolicy", return_value=policy):
            game_training.main()

    def test_runtime_error_inside_the_policy_fails_the_run_with_failure_json(self):
        def crash(*args, **kwargs): raise RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            self.run_main(policy=crash)
        failure = json.loads((self.root / "main" / "failure.json").read_text())
        self.assertEqual((failure["kind"], failure["status"]), ("RuntimeError", "failed"))
        self.assertFalse((self.root / "main" / "summary.json").exists())

    def adapter(self, **settings_overrides):
        import torch
        path = self.root / "trained" / "adapter_0000"
        path.mkdir(parents=True)
        arguments = {"model": "IFM/K2-Horizon-0.9B", "revision": "7c75f2f49e74ac97c1852e62b4c860091d6a00ea",
                     "effort": "medium"}
        arguments.update(settings_overrides)
        from sts_llm.observations import PROMPT_VERSION
        from sts_llm.training import TRAINING_FORMAT_VERSION
        torch.save({"settings": {"world_seeds": [4, 9], "arguments": arguments, "prompt_version": PROMPT_VERSION,
                                 "shared_trainer_format_version": TRAINING_FORMAT_VERSION}},
                   path / "game_training_state.pt")
        return path

    def test_evaluate_only_refuses_the_adapters_training_worlds(self):
        adapter = self.adapter()
        evaluate = ("--evaluate-only", "--group-size", "1", "--warm-start-adapter", str(adapter))
        with self.assertRaisesRegex(ValueError, "allow-training-worlds"):
            self.run_main(*evaluate)
        self.assertIn("allow-training-worlds", json.loads((self.root / "main" / "failure.json").read_text())["message"])
        import shutil
        shutil.rmtree(self.root / "main")
        policy = lambda messages, **kw: scripted_completion(kw["prompt_id"], kw["phase"], 0)
        self.run_main(*evaluate, "--allow-training-worlds", policy=policy)
        summary = json.loads((self.root / "main" / "summary.json").read_text())
        self.assertEqual(summary["evaluation_world_check"]["overlapping_worlds"], ["world:4"])
        self.assertTrue(summary["evaluation_world_check"]["overlap_allowed_by_flag"])

    def test_adapter_with_other_recorded_effort_is_refused_without_override(self):
        adapter = self.adapter(effort="high")
        with self.assertRaisesRegex(ValueError, "effort: adapter 'high'"):
            self.run_main("--warm-start-adapter", str(adapter))
        import shutil
        shutil.rmtree(self.root / "main")
        policy = lambda messages, **kw: scripted_completion(kw["prompt_id"], kw["phase"], 0)
        with patch("sts_llm.game_training.execute") as execute:
            self.run_main("--warm-start-adapter", str(adapter), "--allow-adapter-mismatch", policy=policy)
        config = json.loads((self.root / "main" / "config.json").read_text())
        self.assertTrue(config["adapter_compatibility"]["mismatch_allowed_by_flag"])
        self.assertEqual(execute.call_args.kwargs["inherited_training_worlds"]["worlds"], ["world:4", "world:9"])


if __name__ == "__main__":
    unittest.main()
