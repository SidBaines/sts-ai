"""Pure world/shard/search/bank and panel ordering tests; no native parity claim."""
from copy import deepcopy
from contextlib import redirect_stdout
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sts_env.env import SimulatorError
from sts_policy.curriculum import public_hash
from sts_rl.fresh_collection import (DEFAULT_PROBABILITIES, FightBank, TEACHER_PROTOCOL,
    MAX_SEARCH_SAMPLES, WorldMetrics, is_test_world, play_world, plurality, query_details, query_votes, run_collection, search_plan,
    summarize, task, validate_config)
from sts_rl.fresh_panel import (FreshPanelEpisode, FreshPanelWorker, WorkerProgrammingError,
                               freeze_grid, load_bank, paired_summary, run_panel)
from sts_rl.fresh_study import Interrupted, MIN_DISK, PROTOCOL, RunControl, policy_observation
from sts_rl.fresh_native import native_call
from sts_rl.panel import PanelEpisode, PanelWorker
from sts_rl.teacher_votes import RULES, votes_from_details
from tests.unit.fresh_cohort_fixture import write_cohort

EXPECTED = {"python_interface_sha256": "python", "runtime_python_version": "3.12",
            "environment_version": "0.1.5", "observation_version": "public",
            "native_provenance": {"module_sha256": "native"}}
SOURCE = {"protocol": PROTOCOL, "launch": {"sha256": "parent"}}
SEED = 92_000_000_032  # Predeclared TEST split (hash bucket 1).


def observation(phase="out_of_combat", room="EVENT", act=2, floor=21, hp=37, actions=None):
    return {"phase": phase, "text": "Entire public glossary and visible routes",
        "state": {"act": act, "room": room, "floor": floor, "cur_hp": hp, "max_hp": 80, "gold": 90},
        "legal_actions": actions if actions is not None else ([{"index": 2, "description": "enter boss"}] +
            ([{"index": 7, "description": "end turn"}] if phase == "combat" else []))}


def config(**changes):
    return {"mode": "collect", "teacher_labels": True, "bank": True,
            "teacher_probabilities": {"BOSS2": 1.}, "samples": 24, "simulations": 1000,
            "self_consistency_every": 1, "world_start": 90_500_000_000, "n": 1,
            "workers": 1, "timeout": 1., "max_labels": 0, **changes}


class Control:
    def __init__(self, stop_after=None):
        self.calls, self.stop_after = 0, stop_after

    def check(self):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise Interrupted("STOP")


class Progress:
    def add(self, **kwargs):
        pass


class Policy:
    identity = {"champion": {"path": "fake"}}

    def choose_action(self, value):
        return value["legal_actions"][0]["index"]


class Worker:
    def __init__(self, states, outcome="victory", truncated=False, fault=False):
        self.states, self.outcome, self.truncated, self.fault = states, outcome, truncated, fault
        self.queries, self.last_fault_path, self.position = [], "fault/receipt.json", 0

    def reset(self, spec):
        self.spec = spec
        self.position, self.observation = 0, self.states[0]
        return self.observation, EXPECTED

    def step(self, action):
        if self.fault:
            raise SimulatorError("bounded failure")
        self.position += 1
        self.observation = self.states[self.position]
        last = self.position == len(self.states)-1
        return self.observation, 0., last and not self.truncated, last and self.truncated, {
            **EXPECTED, "task_outcome": self.outcome if last else "",
            "termination_reason": "task_horizon" if self.truncated else self.outcome}

    def search_votes_public(self, simulations, samples, search_seed):
        self.queries.append((simulations, samples, search_seed))
        return {self.observation["legal_actions"][0]["index"]: len(samples)}

    def close(self):
        pass


class DetailsWorker(Worker):
    def __init__(self, states, **kwargs):
        super().__init__(states, **kwargs)
        self.detail_queries, self.played = [], []

    def search_details_public(self, simulations, futures, search_seeds, rng_base):
        self.detail_queries.append((simulations, futures, search_seeds, rng_base))
        return [{"future": future, "search_seed": seed, "chosen": 2,
                 "selection_method": "winning_sequence", "winning_sequence_found": True,
                 "root_simulations": simulations, "best_evaluation": 3., "min_evaluation": 0.,
                 "edges": [{"index": 2, "visits": 10, "evaluation_sum": 10., "mean_evaluation": 1.},
                           {"index": 7, "visits": 20, "evaluation_sum": 40., "mean_evaluation": 2.}]}
                for future, seed in zip(futures, search_seeds)]

    def step(self, action):
        self.played.append(action)
        return super().step(action)


class CliTests(unittest.TestCase):
    def test_collect_cli_teacher_options_and_defaults(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("fresh_collect_cli", root/"scripts"/"rl"/"fresh_collect.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = ["--mode", "collect", "--checkpoint", "fake", "--out", "out",
                "--world-start", str(SEED), "--n", "1"]
        defaults = vars(module.parser().parse_args(args))
        for key in ("vote_rule", "teacher_plays", "record_search"):
            self.assertNotIn(key, defaults)
        for rule in RULES:
            value = vars(module.parser().parse_args([*args, "--vote-rule", rule, "--teacher-plays", "--record-search"]))
            self.assertEqual((value["vote_rule"], value["teacher_plays"], value["record_search"]), (rule, True, True))
            validate_config(value)
        configs = []
        summary = {key: 0 for key in ("completed", "faults", "cuts", "censored", "unstarted", "wins")}
        summary.update(event="done", status="completed", cohort_complete=True)
        for extra in ([], ["--vote-rule", "best_line"], ["--vote-rule", "mean_value"]):
            with patch.object(module, "StudyPolicy"), \
                 patch.object(module, "run_collection", return_value=summary) as run, redirect_stdout(io.StringIO()):
                module.main([*args, *extra])
            configs.append(run.call_args.args[0])
        self.assertEqual(configs[0], defaults)
        self.assertEqual(configs[1], defaults)
        self.assertEqual(configs[2], {**defaults, "vote_rule": "mean_value"})

    def test_panel_cli_accepts_multiple_receipted_directories_and_expected_grid(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("fresh_panel_cli", root/"scripts"/"rl"/"fresh_panel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = vars(module.parser().parse_args(["--checkpoint", "champion", "--combat", "replacement",
             "--out", "out", "--bank", "a", "b", "--bank-world-start", str(SEED), "--bank-n", "2"]))
        self.assertEqual(value["bank"], ["a", "b"])
        self.assertEqual((value["bank_world_start"], value["bank_n"]), (SEED, 2))
        self.assertEqual((value["starts_per_group"]["EVENT1"], value["starts_per_group"]["EVENT2"]), (52, 4))
        self.assertNotIn("EVENT3", value["starts_per_group"])
        self.assertEqual(sum(value["starts_per_group"].values()), 1006)

    def test_both_clis_emit_done_with_explicit_partial_failure_status(self):
        summary = {"event": "done", "status": "failed", "cohort_complete": False,
                   "grid_complete": False, "completed": 0, "valid": 0, "faults": 1,
                   "cuts": 2, "censored": 2, "unstarted": 3, "wins": 0}
        root = Path(__file__).resolve().parents[2]
        for name, runner, args in (
            ("fresh_collect", "run_collection", ["--mode", "eval", "--world-start", "90500000000", "--n", "6"]),
            ("fresh_panel", "run_panel", ["--bank", "fake-bank", "--combat", "fake-combat"])):
            with self.subTest(cli=name):
                spec = importlib.util.spec_from_file_location(name, root/"scripts"/"rl"/(name+".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                output = io.StringIO()
                with patch.object(module, runner, return_value=summary), \
                     patch.object(module, "StudyPolicy", create=True), redirect_stdout(output):
                    module.main(["--checkpoint", "fake", "--out", "unused", *args])
                result = json.loads(output.getvalue())
                for key in ("event", "status", "cohort_complete", "faults", "cuts", "censored", "unstarted"):
                    self.assertEqual(result[key], summary[key])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        # Unit tests never consult laptop power or launch a real sleep assertion.
        self.controls = []
        self.popen = self.enterContext(patch("sts_rl.fresh_study.subprocess.Popen"))
        self.popen.return_value.poll.return_value = None
        def control(*args, **kwargs):
            value = RunControl(*args, **kwargs, ac=lambda: True,
                               disk=lambda _: SimpleNamespace(free=MIN_DISK))
            self.controls.append(value)
            return value
        self.enterContext(patch("sts_rl.fresh_collection.RunControl", side_effect=control))
        self.enterContext(patch("sts_rl.fresh_panel.RunControl", side_effect=control))

    def assert_done(self, out, *, complete, status):
        summary = json.loads((out/"summary.json").read_text())
        receipt = json.loads((out/"completion.json").read_text())
        events = [json.loads(line) for line in (out/"progress.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(sum(e["event"] == "done" for e in events), 1)
        for row in (summary, receipt, events[-1]):
            self.assertEqual(row["cohort_complete"], complete)
            self.assertEqual(row["status"], status)
            for key in ("faults", "cuts", "censored", "unstarted"):
                self.assertEqual(row[key], summary[key])
        from sts_rl.recording import file_identity
        self.assertEqual(receipt["summary"], file_identity(out/"summary.json"))
        self.assertEqual(receipt["source"], SOURCE)
        self.assertTrue(self.controls[-1]._closed)
        self.assertIsNone(self.controls[-1]._caffeinate)
        return summary


class CollectionTests(LifecycleTests):
    def run_world(self, worker, *, world_seed=SEED, **changes):
        count = []
        record = play_world(worker, Policy(), world_seed, config(**changes), Control(), Progress(), SOURCE, count.append)
        return record, count

    def states(self):
        return [observation(), observation("combat", "BOSS"),
                observation("combat", "BOSS", hp=23, actions=[])]

    def test_contiguous_tasks_ignore_legacy_split(self):
        tasks = [task(90_500_000_000+i) for i in range(12)]
        self.assertEqual([t.seed for t in tasks], list(range(90_500_000_000, 90_500_000_012)))
        self.assertTrue(all(t.horizon == 4096 and t.max_act == 3 and t.ascension == 0 for t in tasks))

    def test_completed_raw_labels_prefix_bank_final_boss(self):
        worker = Worker(self.states())
        record, count = self.run_world(worker)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["action_prefix"], [2, 2])
        self.assertEqual(record["action_hash"], public_hash({"actions": [2, 2]}))
        self.assertEqual(len(record["bank"]), 1)
        bank = record["bank"][0]
        self.assertEqual(bank["prefix"], [])
        self.assertEqual(bank["entering_action"], 2)
        self.assertEqual(bank["entering_description"], "enter boss")
        self.assertEqual(bank["first_combat_hash"], public_hash(worker.states[1]))
        self.assertEqual(bank["baseline_hp_lost"], 14)
        self.assertTrue(bank["baseline_win"])
        self.assertEqual(bank["task"]["expected_provenance"], EXPECTED)
        ooc, combat = record["labels"]
        self.assertFalse(ooc["choice"])
        self.assertEqual(ooc["label"], 2)
        self.assertEqual(combat["prefix"], [2])
        self.assertEqual(combat["label_kind"], TEACHER_PROTOCOL)
        self.assertEqual(combat["votes"], {2: 24})
        self.assertFalse(combat["self_consistency"]["use_for_training"])
        self.assertEqual(ooc["observation"]["text"], self.states()[0]["text"])
        self.assertNotIn("world_seed", combat["observation"]["state"])
        self.assertEqual(count, [1])

    def test_fault_cut_and_stop_discard_labels_and_bank(self):
        for worker in (Worker(self.states(), fault=True), Worker(self.states(), truncated=True)):
            row, _ = self.run_world(worker)
            self.assertIn(row["status"], {"fault", "censored"})
            self.assertNotIn("labels", row)
            self.assertNotIn("bank", row)
            self.assertEqual(row["last_fault_path"], "fault/receipt.json")
            self.assertGreater(row["discarded_partial_labels"], 0)
        row = play_world(Worker(self.states()), Policy(), SEED, config(), Control(1), Progress(), SOURCE, lambda _: None)
        self.assertEqual(row["status"], "censored")
        self.assertEqual(row["action_prefix"], [])

    def test_explicit_defaults_keep_entire_record_and_vote_calls_unchanged(self):
        first, second = DetailsWorker(self.states()), DetailsWorker(self.states())
        baseline, _ = self.run_world(first)
        explicit, _ = self.run_world(second, vote_rule="best_line", teacher_plays=False, record_search=False)
        self.assertEqual(explicit, baseline)
        self.assertEqual(first.queries, second.queries)
        self.assertEqual(len(first.queries), 12)
        self.assertEqual(first.detail_queries + second.detail_queries, [])

    def test_alternative_rules_use_details_with_original_seeds_and_repeat_path(self):
        for rule in RULES[1:]:
            with self.subTest(rule=rule):
                worker = DetailsWorker(self.states())
                record, count = self.run_world(worker, vote_rule=rule)
                sample = record["labels"][1]
                self.assertEqual((sample["label"], sample["votes"], sample["vote_rule"]), (7, {7: 24}, rule))
                self.assertNotIn("search_details", sample)
                self.assertEqual(record["action_prefix"], [2, 2])
                self.assertEqual(sample["self_consistency"]["votes"], {2: 24})
                self.assertEqual(len(worker.queries), 6)
                plan = sample["search"]
                self.assertEqual([f for _, futures, _, _ in worker.detail_queries for f in futures], plan["samples"])
                self.assertEqual([s for _, _, seeds, _ in worker.detail_queries for s in seeds], plan["effective_search_seeds"])
                self.assertTrue(all(n == 1000 and base == plan["search_seed"] and len(futures) == 4
                                    for n, futures, _, base in worker.detail_queries))
                self.assertEqual(count, [1])

    def test_record_search_keeps_details_for_default_and_alternative_rules(self):
        for rule, index in (("best_line", 2), ("most_visited", 7)):
            worker = DetailsWorker(self.states())
            record, _ = self.run_world(worker, vote_rule=rule, record_search=True)
            sample = record["labels"][1]
            details = sample["search_details"]
            self.assertEqual(sample["votes"], {index: 24})
            self.assertEqual(len(details), 24)
            self.assertEqual(details, json.loads(json.dumps(details, allow_nan=False)))
            self.assertEqual(sample["votes"], votes_from_details(details, rule))
            self.assertEqual("vote_rule" in sample, rule != "best_line")

    def test_details_abstention_discards_previous_chunks_and_preserves_policy_action(self):
        class Empty(DetailsWorker):
            def search_details_public(self, *args):
                details = super().search_details_public(*args)
                return details if len(self.detail_queries) == 1 else []
        worker = Empty(self.states())
        record, count = self.run_world(worker, bank=False, record_search=True, teacher_plays=True, self_consistency_every=0)
        sample = record["labels"][1]
        self.assertEqual((sample["votes"], sample["search_details"], sample["label"]), ({}, [], None))
        self.assertNotIn("played_by", sample)
        self.assertNotIn("played_action", sample)
        self.assertNotIn("use_for_training", sample)
        self.assertNotIn("BOSS2:teacher_played", record["sampling"])
        self.assertEqual(worker.played, [2, 2])
        self.assertEqual(count, [])

    def test_details_queries_check_control_and_reject_partial_searches(self):
        worker = DetailsWorker(self.states())
        worker.reset(task(SEED))
        plan = search_plan(public_hash(self.states()[1]))
        with self.assertRaisesRegex(Interrupted, "STOP"):
            query_details(worker, self.states()[1], plan, config(), Control(1))
        self.assertEqual(len(worker.detail_queries), 1)
        with patch.object(worker, "search_details_public", return_value=[{"chosen": 2}]):
            with self.assertRaisesRegex(ValueError, "incomplete search details"):
                query_details(worker, self.states()[1], plan, config(), Control())

    def test_teacher_plays_steps_label_marks_samples_and_completion(self):
        worker = DetailsWorker(self.states())
        record, count = self.run_world(worker, bank=False, vote_rule="most_visited", teacher_plays=True)
        sample = record["labels"][1]
        self.assertEqual(worker.played, [2, 7])
        self.assertEqual(record["action_prefix"], [2, 7])
        self.assertEqual((sample["label"], sample["source_action"], sample["played_action"],
                          sample["played_by"], sample["use_for_training"]), (7, 2, 7, "teacher", False))
        self.assertEqual(sample["played_action"], record["action_prefix"][len(sample["prefix"])])
        self.assertEqual(record["sampling"]["BOSS2:teacher_played"], 1)
        self.assertEqual(count, [1])
        class Factory(DetailsWorker):
            def __init__(self, **kwargs):
                super().__init__([observation(), observation("combat", "BOSS"), observation(actions=[])])
        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                summary = run_collection(config(out=directory, bank=False, vote_rule="most_visited", teacher_plays=True),
                                         Policy(), worker_factory=Factory)
            self.assertIs(summary["teacher_plays"], True)
            for name in ("summary.json", "completion.json"):
                self.assertIs(json.loads((Path(directory)/name).read_text())["teacher_plays"], True)

    def test_new_config_validation_accepts_legacy_and_rejects_invalid_options(self):
        original = config()
        validate_config(original)
        self.assertEqual(original, config())
        validate_config(config(bank=False, teacher_plays=True))
        for changes in ({"vote_rule": "unknown"}, {"vote_rule": None},
                        {"teacher_plays": 1}, {"teacher_plays": "true"}, {"record_search": 0},
                        {"record_search": None}, {"teacher_plays": True, "mode": "eval"},
                        {"teacher_plays": True, "teacher_labels": False}, {"teacher_plays": True, "bank": True}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "vote_rule|teacher_plays|record_search"):
                validate_config(config(**changes))

    def test_programming_errors_propagate(self):
        class Bad(Policy):
            def choose_action(self, value):
                raise ValueError("model shape bug")
        with self.assertRaisesRegex(ValueError, "model shape"):
            play_world(Worker(self.states()), Bad(), SEED, config(), Control(), Progress(), SOURCE, lambda _: None)

    def test_eval_does_not_query_teacher(self):
        worker = Worker(self.states())
        record, count = self.run_world(worker, mode="eval")
        self.assertEqual(worker.queries, [])
        self.assertEqual(record["labels"], [])
        self.assertEqual(count, [])

    def test_old_environment_refused_before_policy_or_teacher_actions(self):
        class Old(Worker):
            def reset(self, spec):
                value, info = super().reset(spec)
                return value, {**info, "environment_version": "0.1.4"}
        worker = Old(self.states())
        with self.assertRaisesRegex(RuntimeError, "requires environment 0.1.5"):
            self.run_world(worker)
        self.assertEqual(worker.position, 0)
        self.assertEqual(worker.queries, [])

    def test_probe_is_unlabelled_and_separate_namespace_required(self):
        validate_config(config(world_start=93_000_000_000, teacher_labels=False))
        with self.assertRaisesRegex(ValueError, "93e9"):
            validate_config(config(teacher_labels=False))
        worker = Worker(self.states())
        record, _ = self.run_world(worker, teacher_labels=False)
        self.assertEqual(worker.queries, [])
        self.assertTrue(all(r["label_kind"] == "probe_unlabelled" and "label" not in r for r in record["labels"]))

    def test_empty_teacher_abstains_and_plurality_uses_menu_order(self):
        class Empty(Worker):
            def search_votes_public(self, *args):
                return {}
        record, count = self.run_world(Empty(self.states()))
        self.assertIsNone(record["labels"][1]["label"])
        self.assertEqual(count, [])
        value = observation("combat", actions=[{"index": 9, "description": "A"}, {"index": 3, "description": "B"}])
        self.assertEqual(plurality({3: 12, 9: 12}, value), 9)
        self.assertIsNone(plurality({}, value))
        with self.assertRaises(ValueError):
            plurality({0: 24}, value)

    def test_self_consistency_draw_battle_rng_and_search_seeds_disjoint(self):
        for n in (24, MAX_SEARCH_SAMPLES):
            first, repeat = search_plan("abc123"*11, n), search_plan("abc123"*11, n, repeat=True)
            self.assertEqual(first["samples"], list(range(1, n+1)))
            self.assertEqual(repeat["samples"], list(range(1025, 1025+n)))
            for name in ("samples", "draw_order_seeds", "effective_rng_seeds", "effective_search_seeds"):
                self.assertFalse(set(first[name]) & set(repeat[name]), name)
            for plan in (first, repeat):
                self.assertEqual(plan["draw_order_seeds"], plan["samples"])
                self.assertEqual(plan["effective_rng_seeds"], [plan["search_seed"]*1009+s for s in plan["samples"]])
                self.assertEqual(plan["effective_search_seeds"], [plan["search_seed"]+s for s in plan["samples"]])
            self.assertNotEqual(first["search_seed"], repeat["search_seed"])
            self.assertEqual(first, search_plan("abc123"*11, n))
        for n in (0, MAX_SEARCH_SAMPLES+1, 1025, True, 1.5):
            with self.assertRaises(ValueError):
                search_plan("abc123"*11, n)
            with self.assertRaises(ValueError):
                validate_config(config(samples=n))

    def test_repeat_only_on_test_worlds_with_existing_hash_frequency(self):
        for seed, allowed in ((92_000_000_030, False), (92_000_000_031, False),
                              (92_000_000_032, True), (92_000_000_033, False)):
            self.assertEqual(is_test_world(seed), allowed)
            worker = Worker(self.states())
            record, count = self.run_world(worker, world_seed=seed)
            self.assertEqual(len(worker.queries), 12 if allowed else 6)
            self.assertEqual("self_consistency" in record["labels"][1], allowed)
            self.assertEqual(count, [1])
        for every in (0, int(public_hash(self.states()[1]), 16)+1):
            worker = Worker(self.states())
            record, _ = self.run_world(worker, self_consistency_every=every)
            self.assertEqual(len(worker.queries), 6)
            self.assertNotIn("self_consistency", record["labels"][1])

    def test_repeat_disagreement_never_changes_first_target_or_trajectory(self):
        class Disagree(Worker):
            def search_votes_public(self, simulations, samples, search_seed):
                self.queries.append((simulations, samples, search_seed))
                return {2 if samples[0] < 1025 else 7: len(samples)}
        worker = Disagree(self.states())
        record, count = self.run_world(worker)
        label = record["labels"][1]
        self.assertEqual((label["label"], label["votes"]), (2, {2: 24}))
        self.assertEqual(label["self_consistency"]["votes"], {7: 24})
        self.assertFalse(label["self_consistency"]["use_for_training"])
        self.assertEqual(record["action_prefix"], [2, 2])
        self.assertEqual(count, [1])
        plan = label["search"]
        self.assertEqual([s for _, samples, _ in worker.queries[:6] for s in samples], plan["samples"])
        self.assertTrue(all(seed == plan["search_seed"] for _, _, seed in worker.queries[:6]))
        repeat = label["self_consistency"]["search"]
        self.assertEqual([s for _, samples, _ in worker.queries[6:] for s in samples], repeat["samples"])

    def test_forced_combat_skipped_and_filtered_ooc_raw_single_choice_retained(self):
        ooc = observation(actions=[{"index": 2, "description": "smith"},
                                   {"index": 7, "description": "discard potion Fire Potion (slot 0)"}])
        forced = observation("combat", "BOSS", actions=[{"index": 2, "description": "end turn"}])
        worker = Worker([ooc, forced, *self.states()[1:]])
        record, count = self.run_world(worker)
        self.assertEqual(count, [1])
        self.assertEqual(len(worker.queries), 12)  # Only the informative state + TEST repeat.
        self.assertEqual(len(record["labels"]), 2)
        self.assertEqual(record["labels"][0]["observation"], ooc)
        self.assertFalse(record["labels"][0]["choice"])
        self.assertEqual(record["sampling"]["BOSS2:forced"], 1)
        self.assertEqual(record["sampling"]["BOSS2:choice"], 1)
        self.assertEqual(record["sampling"]["BOSS2:seen"], 2)
        self.assertEqual(record["sampling"]["ooc:forced"], 1)
        self.assertEqual(record["behavior"]["smith_hp<.5"]["opportunities"], 0)
        for actions in ([], forced["legal_actions"]):
            with self.assertRaisesRegex(ValueError, "informative"):
                query_votes(worker, {**forced, "legal_actions": actions}, search_plan("abc123"*11), config(), Control())
        forced_only = Worker([ooc, forced, self.states()[-1]])
        row, counts = self.run_world(forced_only)
        self.assertEqual(forced_only.queries, [])
        self.assertEqual(counts, [])
        self.assertEqual(len(row["labels"]), 1)

    def test_recorded_search_seeds_match_actual_panel_calls(self):
        value, calls = self.states()[1], []
        class Scratch:
            def reseed_rngs(self, seed):
                self.rng_seed = seed
            def search_best_action(self, simulations, *, search_seed, draw_order_seed, include_card_type):
                calls.append((self.rng_seed, search_seed, draw_order_seed))
                return {"bits": 12, "provenance": {"hidden_pile_order_canonicalized": True},
                        "selection_method": "most_visited_root", "winning_sequence_found": False,
                        "root_simulations": simulations, "best_evaluation": 2., "min_evaluation": 0.,
                        "root_edges": [{"bits": 12, "valid": True, "visits": 20,
                                        "evaluation_sum": 40., "mean_evaluation": 2.},
                                       {"bits": 13, "valid": True, "visits": 5,
                                        "evaluation_sum": 15., "mean_evaluation": 3.},
                                       {"bits": 999, "valid": False}]}
        raw = SimpleNamespace(bc=SimpleNamespace(copy=Scratch), combat_observation="unused",
                              advance_to_decision=lambda: None)
        episode = FreshPanelEpisode({})
        episode.episode = SimpleNamespace(env=SimpleNamespace(_env=raw, display_index_for_bits=lambda bits: {12: 2, 13: 2}[bits]),
                                          _observe=lambda: value)
        first, repeat = search_plan(public_hash(value)), search_plan(public_hash(value), repeat=True)
        for plan in (first, repeat):
            self.assertEqual(episode.search_votes_public(1000, plan["samples"], plan["search_seed"]), {2: 24})
            actual = calls[-24:]
            self.assertEqual([r[0] for r in actual], plan["effective_rng_seeds"])
            self.assertEqual([r[1] for r in actual], plan["effective_search_seeds"])
            self.assertEqual([r[2] for r in actual], plan["draw_order_seeds"])
        for column in range(3):
            self.assertFalse({r[column] for r in calls[:24]} & {r[column] for r in calls[24:]})
        for plan in (first, repeat):
            details = episode.search_details_public(1000, plan["samples"], plan["effective_search_seeds"], plan["search_seed"])
            self.assertEqual(votes_from_details(details, "best_line"), {2: 24})
            self.assertEqual([r[0] for r in calls[-24:]], plan["effective_rng_seeds"])
            self.assertEqual([r[1] for r in calls[-24:]], plan["effective_search_seeds"])
            self.assertEqual([r[2] for r in calls[-24:]], plan["draw_order_seeds"])
            self.assertEqual(details[0], {"future": plan["samples"][0], "search_seed": plan["effective_search_seeds"][0],
                "chosen": 2, "selection_method": "most_visited_root", "winning_sequence_found": False,
                "root_simulations": 1000, "best_evaluation": 2., "min_evaluation": 0.,
                "edges": [{"index": 2, "bits": 12, "visits": 20, "evaluation_sum": 40., "mean_evaluation": 2.},
                          {"index": 2, "bits": 13, "visits": 5, "evaluation_sum": 15., "mean_evaluation": 3.}]})
            self.assertEqual(details, json.loads(json.dumps(details, allow_nan=False)))
        episode.search_details_public(1000, [5, 5], [91, 92], 7)
        self.assertEqual(calls[-2:], [(7068, 91, 5), (7068, 92, 5)])

    def test_details_preconditions_abstention_and_canonicalisation(self):
        search = Mock()
        raw = SimpleNamespace(bc=SimpleNamespace(copy=lambda: SimpleNamespace(
                              reseed_rngs=lambda seed: None, search_best_action=search)),
                              combat_observation="unused", advance_to_decision=lambda: None)
        episode = FreshPanelEpisode({})
        episode.episode = SimpleNamespace(env=SimpleNamespace(_env=raw, display_index_for_bits=int),
                                          _observe=lambda: self.states()[1])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            episode.search_details_public(10, [1], [], 0)
        search.side_effect = ValueError("unsupported draw_order_seed")
        self.assertEqual(episode.search_details_public(10, [1], [2], 0), [])
        search.side_effect = ValueError("other failure")
        with self.assertRaisesRegex(ValueError, "other failure"):
            episode.search_details_public(10, [1], [2], 0)
        search.side_effect, search.return_value = None, {"provenance": {}}
        with self.assertRaisesRegex(RuntimeError, "Canonical search"):
            episode.search_details_public(10, [1], [2], 0)
        raw.bc = None
        with self.assertRaisesRegex(RuntimeError, "pending combat decision"):
            episode.search_details_public(10, [1], [2], 0)

    def test_details_mutation_check_and_worker_proxy_submit(self):
        episode = FreshPanelEpisode({})
        episode.episode = SimpleNamespace(_observe=Mock(side_effect=[self.states()[1], self.states()[2]]))
        with patch.object(PanelEpisode, "search_details_public", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "mutated the live decision"):
                episode.search_details_public(10, [1], [2], 0)
        with PanelWorker({}) as worker, patch.object(worker, "_call", return_value=[{"chosen": 2}]) as call:
            result = worker.submit("search_details_public", 10, [1], [2], 0).result()
            self.assertEqual(result, [{"chosen": 2}])
            call.assert_called_once_with("search_details_public", 10, [1], [2], 0)

    def test_boss_passage_secret_portal_and_final_victory(self):
        metrics = WorldMetrics()
        before = observation("combat", "Room.BOSS", floor=23)
        metrics.transition(before, 2, observation("out_of_combat", "MAP", act=3, floor=24))
        self.assertTrue(metrics.bosses["2"]["pass"])
        before = observation("combat", "BOSS", act=3, floor=35)
        metrics.transition(before, 2, observation("combat", "BOSS", act=3, floor=35, hp=1), win=True)
        self.assertTrue(metrics.bosses["3"]["pass"])
        self.assertEqual(metrics.bosses["3"]["entry_hp"], 37)

    def test_behavior_denominators_and_incomplete_summary(self):
        metrics = WorldMetrics()
        value = observation(hp=76, actions=[{"index": 2, "description": "smith"}, {"index": 7, "description": "rest"}])
        metrics.transition(value, 2, value)
        self.assertEqual(metrics.counters["smith_hp>=.85"]["opportunities"], 1)
        self.assertEqual(metrics.counters["smith_hp>=.85"]["chosen"], 1)
        record, _ = self.run_world(Worker(self.states()))
        summary = summarize([record, {"status": "fault"}], 3, {}, 1)
        self.assertFalse(summary["confidence_claims_allowed"])
        self.assertEqual((summary["completed"], summary["faults"], summary["unstarted"]), (1, 1, 1))

    def test_event_fights_have_default_coverage_and_potion_counters_are_combat_only(self):
        for act in (1, 2, 3):
            self.assertEqual(DEFAULT_PROBABILITIES[f"EVENT{act}"], DEFAULT_PROBABILITIES[f"MONSTER{act}"])
        metrics = WorldMetrics()
        menu = [{"index": 2, "description": "leave"},
                {"index": 7, "description": "drink potion Fire Potion (slot 1)"},
                {"index": 19, "description": "discard potion Fire Potion (slot 1)"}]
        ooc = observation(actions=menu)
        metrics.transition(ooc, 2, ooc)
        self.assertEqual(metrics.counters["combat_potion_drink"]["opportunities"], 0)
        combat = observation("combat", actions=menu)
        metrics.transition(combat, 7, combat)
        self.assertEqual(metrics.counters["combat_potion_drink"]["opportunities"], 1)
        self.assertEqual(metrics.counters["combat_potion_drink"]["chosen"], 1)
        self.assertEqual(metrics.counters["combat_potion_discard"]["opportunities"], 1)
        self.assertEqual(metrics.counters["combat_potion_discard"]["chosen"], 0)

    def test_run_writes_only_completed_immutable_world_shards(self):
        class Factory(Worker):
            def __init__(self, **kwargs):
                super().__init__([observation(), observation("combat", "BOSS"), observation("combat", "BOSS", hp=20, actions=[])])
        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                result = run_collection(config(out=directory, n=3, mode="eval", workers=2), Policy(), worker_factory=Factory)
            self.assertEqual(result["completed"], 3)
            paths = sorted((Path(directory)/"worlds").glob("*.gz"))
            self.assertEqual([int(p.name.split(".")[0]) for p in paths], list(range(90_500_000_000, 90_500_000_003)))
            with gzip.open(paths[0], "rt") as source:
                self.assertEqual(json.load(source)["expected_provenance"], EXPECTED)
            self.assertTrue(result["cohort_complete"])
            self.assert_done(Path(directory), complete=True, status="completed")

    def test_public_inventory_is_internal_and_raw_shard_hashes_objects_stay_unchanged(self):
        raw = observation(actions=[
            {"index": 2, "description": "enter boss"},
            {"index": 7, "description": "discard potion Fire Potion (slot 1)"}])
        raw["text"] = "Potions: EMPTY_POTION_SLOT"
        original, digest = deepcopy(raw), public_hash(raw)
        entries, state = tuple(raw["legal_actions"]), raw["state"]
        seen = []

        class PublicPolicy(Policy):
            def choose_action(self, value):
                derived = policy_observation(value)
                seen.append(derived)
                return derived["legal_actions"][0]["index"]

        class Factory(Worker):
            def __init__(self, **kwargs):
                super().__init__([raw, observation("combat", "BOSS", hp=20, actions=[])])

        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                result = run_collection(config(out=directory, workers=1, bank=False),
                                        PublicPolicy(), worker_factory=Factory)
            with gzip.open(next((Path(directory)/"worlds").glob("*.gz")), "rt") as source:
                label = json.load(source)["labels"][0]
            self.assertTrue(result["cohort_complete"])
            self.assertEqual(label["observation"], original)
            self.assertEqual(label["public_hash"], digest)
            self.assertNotIn("adapter_public_ooc_potions", label["observation"])
            self.assertEqual(seen[0]["adapter_public_ooc_potions"], [{"slot": 1, "name": "Fire Potion"}])
            self.assertEqual(raw, original)
            self.assertEqual(public_hash(raw), digest)
            self.assertIs(raw["state"], state)
            self.assertTrue(all(a is b for a, b in zip(raw["legal_actions"], entries)))
            self.assert_done(Path(directory), complete=True, status="completed")

    def test_max_labels_stops_between_complete_worlds_and_reports_overshoot(self):
        class Factory(Worker):
            def __init__(self, **kwargs):
                super().__init__([observation(), observation("combat", "BOSS"),
                                  observation("combat", "BOSS", hp=30),
                                  observation("combat", "BOSS", hp=20, actions=[])])
        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                result = run_collection(config(out=directory, n=5, max_labels=1), Policy(), worker_factory=Factory)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["combat_labels"], 2)
            self.assertEqual(result["max_labels_overshoot"], 1)
            self.assertEqual(result["unstarted"], 4)
            self.assertEqual(result["records"][0]["action_prefix"], [2, 2, 2])
            self.assert_done(Path(directory), complete=False, status="interrupted")

    def test_quarantined_queried_labels_do_not_spend_saved_label_budget(self):
        first_seed = 90_500_000_000
        class Factory(Worker):
            def __init__(self, **kwargs):
                super().__init__([observation(), observation("combat", "BOSS"),
                                  observation("combat", "BOSS", hp=20, actions=[])])
            def step(self, action):
                if self.spec.seed == first_seed and self.position == 1:
                    raise SimulatorError("fault after informative query")
                return super().step(action)
        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                result = run_collection(config(out=directory, n=3, max_labels=1), Policy(), worker_factory=Factory)
            self.assertEqual((result["faults"], result["completed"], result["unstarted"]), (1, 1, 1))
            self.assertEqual(result["queried_combat_labels"], 2)
            self.assertEqual(result["combat_labels"], 1)
            self.assertEqual(result["max_labels_overshoot"], 0)
            with gzip.open(next((Path(directory)/"quarantine").glob("*.gz")), "rt") as source:
                self.assertNotIn("labels", json.load(source))
            self.assert_done(Path(directory), complete=False, status="failed")

    def test_programming_abort_quarantines_partial_world_and_writes_summary(self):
        class Factory(Worker):
            def __init__(self, **kwargs):
                super().__init__([observation(), observation("combat", "BOSS")])
            def step(self, action):
                raise ValueError("programming bug")
        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                with self.assertRaisesRegex(ValueError, "programming bug"):
                    run_collection(config(out=directory), Policy(), worker_factory=Factory)
            result = json.loads((Path(directory)/"summary.json").read_text())
            self.assertEqual(result["faults"], 0)
            self.assertEqual(result["censored"], 1)
            self.assertEqual(result["unstarted"], 0)
            self.assertEqual(result["programming_errors"], 1)
            self.assert_done(Path(directory), complete=False, status="failed")
            quarantine = list((Path(directory)/"quarantine").glob("*.gz"))
            with gzip.open(quarantine[0], "rt") as source:
                self.assertNotIn("labels", json.load(source))

    def test_initial_stop_deadline_and_disk_publish_done_with_no_claims(self):
        for reason in ("STOP", "deadline", "disk_below_6_GiB"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                def prepare(*args):
                    (out/"receipt.json").write_text(json.dumps(SOURCE))
                    if reason == "STOP":
                        (out/"STOP").touch()
                    return SOURCE
                def make_control(*args, **kwargs):
                    value = RunControl(*args, **kwargs, ac=lambda: True,
                        disk=lambda _: SimpleNamespace(free=MIN_DISK-int(reason == "disk_below_6_GiB")))
                    self.controls.append(value)
                    return value
                factory = Mock(return_value=Mock())
                with patch("sts_rl.fresh_collection.prepare_run", side_effect=prepare), \
                     patch("sts_rl.fresh_collection.RunControl", side_effect=make_control):
                    result = run_collection(config(out=directory, mode="eval", n=3,
                        deadline_utc="1970-01-01T00:00:00Z" if reason == "deadline" else None),
                        Policy(), worker_factory=factory)
                self.assertEqual((result["completed"], result["unstarted"]), (0, 3))
                self.assertEqual(result["stop_reasons"], [reason])
                self.assertFalse(result["confidence_claims_allowed"])
                factory.return_value.reset.assert_not_called()
                factory.return_value.close.assert_called_once_with()
                self.assertTrue((out/"receipt.json").exists())
                self.assert_done(out, complete=False, status="interrupted")

    def test_stop_inside_world_publishes_quarantine_and_done(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            class Factory(Worker):
                def __init__(self, **kwargs):
                    super().__init__([observation(), observation("combat", "BOSS"),
                                      observation("combat", "BOSS", actions=[])])
                def step(self, action):
                    result = super().step(action)
                    (out/"STOP").touch()
                    return result
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                result = run_collection(config(out=directory, n=2), Policy(), worker_factory=Factory)
            self.assertEqual((result["completed"], result["censored"], result["unstarted"]), (0, 1, 1))
            self.assertEqual(result["combat_labels"], 0)
            self.assertEqual(list((out/"worlds").glob("*.gz")), [])
            with gzip.open(next((out/"quarantine").glob("*.gz")), "rt") as source:
                row = json.load(source)
                self.assertEqual(row["discarded_partial_labels"], 1)
                self.assertNotIn("labels", row)
            self.assert_done(out, complete=False, status="interrupted")

    def test_eval_faults_and_cuts_never_claim_completed_cohort(self):
        for kind in ("fault", "cut"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                class Factory(Worker):
                    def __init__(self, **kwargs):
                        super().__init__([observation(), observation("combat", "BOSS"),
                                          observation("combat", "BOSS", actions=[])],
                                         fault=kind == "fault", truncated=kind == "cut")
                with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                    result = run_collection(config(out=directory, mode="eval"), Policy(), worker_factory=Factory)
                self.assertEqual((result["faults"], result["cuts"]), (int(kind == "fault"), int(kind == "cut")))
                self.assertFalse(result["confidence_claims_allowed"])
                self.assert_done(Path(directory), complete=False, status="failed" if kind == "fault" else "incomplete")

    def test_horizon_cut_quarantines_then_continues_exact_next_seed(self):
        seeds = []
        class Factory(Worker):
            def __init__(self, **kwargs):
                super().__init__(self_states, truncated=True)
            def reset(self, spec):
                seeds.append(spec.seed)
                self.truncated = len(seeds) == 1
                return super().reset(spec)
        self_states = [observation(), observation("combat", "BOSS"), observation("combat", "BOSS", actions=[])]
        with tempfile.TemporaryDirectory() as directory:
            with patch("sts_rl.fresh_collection.prepare_run", return_value=SOURCE):
                result = run_collection(config(out=directory, mode="eval", n=3), Policy(), worker_factory=Factory)
            self.assertEqual(seeds, list(range(90_500_000_000, 90_500_000_003)))
            self.assertEqual((result["cuts"], result["completed"], result["unstarted"]), (1, 2, 0))
            self.assertEqual(result["stop_reasons"], [])
            self.assert_done(Path(directory), complete=False, status="incomplete")
            with gzip.open(next((Path(directory)/"quarantine").glob("*.gz")), "rt") as stream:
                self.assertNotIn("bank", json.load(stream))


class PanelTests(LifecycleTests):
    @staticmethod
    def panel_receipt(out, *args, **kwargs):
        from sts_rl.fresh_study import atomic_json
        atomic_json(Path(out)/"receipt.json", SOURCE)
        return SOURCE

    def run_fixture(self, out, factory, **changes):
        states = [observation(), observation("combat", "BOSS"), observation("combat", "BOSS", hp=20, actions=[])]
        world = play_world(Worker(states), Policy(), SEED, config(mode="eval"), Control(), Progress(), SOURCE, lambda _: None)
        bank_file = write_cohort(out/"bank", [world])
        model = SimpleNamespace(identity={"champion": {"path": "champion"}, "combat": {"path": "entity"}, "ooc": None})
        with patch("sts_rl.fresh_panel.policy_factory", return_value=model), \
             patch("sts_rl.fresh_panel.prepare_run", side_effect=self.panel_receipt):
            return run_panel({"out": str(out), "bank": str(bank_file), "checkpoint": "champion",
                "combat": "entity", "workers": 1, "timeout": 1., "resamples": 1,
                "starts_per_group": {"BOSS2": 1}, **changes}, worker_factory=factory)

    def rows(self):
        return [{"world_seed": SEED+i, "start_id": start, "group": group,
                 "public_hash": "ooc", "first_combat_hash": "combat", "expected_provenance": EXPECTED,
                 "prefix": [], "task": {}, "entering_action": 2, "entering_description": "enter"}
                for i in range(3) for start, group in ((1, "BOSS2"), (2, "MONSTER2"), (3, "BOSS2"))]

    def test_one_start_per_world_group_selection_and_grid_integrity(self):
        selected, frozen = freeze_grid(self.rows(), {"BOSS2": 2}, 3)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({r["world_seed"] for r in selected}), 2)
        self.assertEqual(frozen["coverage"], {"BOSS2": 2})
        self.assertEqual(frozen["salts"], [-1, 0, 1, 2])
        self.assertEqual(len(frozen["grid"]), 8)
        self.assertEqual(freeze_grid(list(reversed(self.rows())), {"BOSS2": 2}, 3)[1], frozen)
        records = []
        for item in frozen["grid"]:
            for policy in ("champion", "replacement"):
                records.append({**item, "policy": policy, "win": policy == "replacement", "hp_lost": 4})
        summary = paired_summary(records, frozen)
        self.assertTrue(summary["confidence_claims_allowed"])
        metric = summary["groups"]["BOSS2:resampled"]["win"]
        self.assertEqual(metric["worlds"], 2)
        self.assertEqual(metric["paired_difference"], 1.)
        self.assertEqual(metric["se"], 0.)
        records[0]["cut"] = True
        incomplete = paired_summary(records, frozen)
        self.assertFalse(incomplete["confidence_claims_allowed"])
        self.assertTrue(incomplete["groups"]["BOSS2:resampled"]["diagnostic_only"])
        self.assertTrue(incomplete["groups"]["BOSS2:unsalted"]["win"]["diagnostic_only"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_summary(records + records[:1], frozen)
        records[0].pop("cut")
        records[0]["first_combat_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "different first-combat"):
            paired_summary(records, frozen)

    def test_bank_identity_and_completed_shards(self):
        worker = Worker([observation(), observation("combat", "BOSS"), observation("combat", "BOSS", hp=20, actions=[])])
        row = play_world(worker, Policy(), SEED, config(mode="eval"), Control(), Progress(), SOURCE, lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"bank.jsonl"
            file.write_text(json.dumps(row["bank"][0])+"\n")
            with self.assertRaisesRegex(ValueError, "unreceipted"):
                load_bank(file)
            bank_dir = write_cohort(Path(directory)/"bank", [row])
            self.assertEqual(len(load_bank(bank_dir)), 1)
            altered = deepcopy(row["bank"][0])
            altered["observation"]["text"] = "slimmed"
            file.write_text(json.dumps(altered)+"\n")
            with self.assertRaisesRegex(ValueError, "unreceipted"):
                load_bank(file)
            shard = Path(directory)/"world.json.gz"
            row["bank"][0]["prefix"] = [77]
            with gzip.open(shard, "wt") as output:
                json.dump(row, output)
            with self.assertRaisesRegex(ValueError, "unreceipted"):
                load_bank(shard)

    def test_pure_panel_runner_uses_identical_grid_and_separate_policies(self):
        worker = Worker([observation(), observation("combat", "BOSS"), observation("combat", "BOSS", hp=20, actions=[])])
        world = play_world(worker, Policy(), SEED, config(mode="eval"), Control(), Progress(), SOURCE, lambda _: None)
        calls, owners = [], []
        class Factory:
            def __init__(self, **kwargs):
                pass
            def fresh_fight(self, bank, salt, champion, combat, expected_policy):
                owners.append(id(self))
                calls.append((bank["world_seed"], salt, combat, expected_policy["combat"]))
                return {"win": combat is not None, "hp_lost": 2, "first_combat_hash": str(salt), "decisions": 1}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            bank_file = write_cohort(Path(directory)/"bank", [world])
            out = Path(directory)/"out"
            out.mkdir()
            model = SimpleNamespace(identity={"champion": {"path": "champion"}, "combat": {"path": "entity"}, "ooc": None})
            with patch("sts_rl.fresh_panel.policy_factory", return_value=model), \
                 patch("sts_rl.fresh_panel.prepare_run", side_effect=self.panel_receipt):
                report = run_panel({"out": str(out), "bank": str(bank_file), "checkpoint": "champion",
                    "combat": "entity", "workers": 2, "timeout": 1., "resamples": 1,
                    "starts_per_group": {"BOSS2": 1}}, worker_factory=Factory)
            self.assertTrue(report["grid_complete"])
            self.assertEqual(report["received"], 4)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(set(owners)), 1)  # All salts/policies share one child.
            self.assertTrue(all(expected == ({"path": "entity"} if combat else None)
                                for _, _, combat, expected in calls))
            self.assertTrue((out/"grid.json").exists())
            self.assertTrue((out/"summary.json").exists())
            self.assert_done(out, complete=True, status="completed")

    def test_panel_initial_deadline_and_stop_between_policies_publish_done(self):
        for initial in (True, False):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                out, calls = Path(directory), []
                class Factory:
                    def __init__(self, **kwargs):
                        pass
                    def fresh_fight(self, *args):
                        calls.append(args)
                        (out/"STOP").touch()
                        return {"win": True, "hp_lost": 2, "first_combat_hash": "same", "decisions": 1}
                    def close(self):
                        pass
                report = self.run_fixture(out, Factory, deadline_utc="1970-01-01T00:00:00Z" if initial else None)
                self.assertEqual(report["received"], 0 if initial else 2)
                self.assertEqual(report["valid"], 0 if initial else 1)
                self.assertEqual(report["censored"], 0 if initial else 1)
                self.assertEqual(len(calls), 0 if initial else 1)
                self.assertFalse(report["confidence_claims_allowed"])
                self.assert_done(out, complete=False, status="interrupted")

    def test_panel_fault_cut_and_programming_error_release_control_and_publish_done(self):
        for kind in ("fault", "cut", "programming_error"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                class Factory:
                    last_fault_path = "fault/receipt.json"
                    def __init__(self, **kwargs):
                        pass
                    def fresh_fight(self, *args):
                        if kind == "fault":
                            raise SimulatorError("bounded failure")
                        if kind == "programming_error":
                            raise ValueError("panel bug")
                        return {"cut": True, "win": None, "hp_lost": None, "first_combat_hash": "same"}
                    def close(self):
                        pass
                if kind == "programming_error":
                    with self.assertRaisesRegex(ValueError, "panel bug"):
                        self.run_fixture(out, Factory)
                else:
                    self.run_fixture(out, Factory)
                result = self.assert_done(out, complete=False, status="incomplete" if kind == "cut" else "failed")
                self.assertFalse(result["confidence_claims_allowed"])
                self.assertEqual(result["faults"], 4 if kind == "fault" else 0)
                self.assertEqual(result["cuts"], 4 if kind == "cut" else 0)
                self.assertEqual(list((out/"results").glob("*.json")), [])

    def native_fake(self):
        events = []
        class Context(SimpleNamespace):
            def copy(self):
                events.append(("copy", self.floor_num, self.cur_hp, self.gold))
                return deepcopy(self)
        class Battle:
            def exit_battle(self, context):
                raise AssertionError("Real callback must not run")
        sts = SimpleNamespace(ScreenState=SimpleNamespace(BATTLE="BATTLE", MAP_SCREEN="MAP"),
                              BattleContext=Battle, BattleOutcome=SimpleNamespace(PLAYER_VICTORY="WIN"),
                              GameOutcome=SimpleNamespace(UNDECIDED="UNDECIDED", PLAYER_VICTORY="VICTORY", PLAYER_LOSS="LOSS"))
        gc = Context(seed=40, floor_num=20, cur_hp=37, max_hp=80, gold=90, screen_state="MAP", act=2,
                     outcome="UNDECIDED", map_y=14)
        raw = SimpleNamespace(gc=gc, bc=None, sts=sts, _undefined_behavior_evoked=False)
        class Action:
            def execute(self, context):
                events.append(("execute", raw.bc))
                context.floor_num += 1
                context.gold -= 5
                context.screen_state = "BATTLE"
                context.map_y = 15
        raw._action_views = lambda: ([Action()], [SimpleNamespace(index=2, description="enter boss")], [0])
        def advance():
            events.append(("initialize", raw.gc.seed, raw.gc.floor_num, raw.gc.cur_hp, raw.gc.gold))
            raw.bc = SimpleNamespace(outcome="WIN", player_cur_hp=25, undefined_behavior_evoked=False)
        raw.advance_to_decision = advance
        wrapper = SimpleNamespace(_env=raw, _raw_history=[], _public_history=[], _selection_labels={})
        def observe():
            if raw.bc is None and raw.gc.outcome == "UNDECIDED" and raw.gc.map_y == 15:
                raise IndexError("row 15 map moves must never be requested")
            return observation("combat" if raw.bc else "out_of_combat", "BOSS", floor=raw.gc.floor_num, hp=raw.gc.cur_hp)
        def step(action):
            sts.BattleContext.exit_battle(raw.bc, raw.gc)
            raw.bc = None
            return observe(), 0., raw.gc.outcome != "UNDECIDED", False, {}
        episode = FreshPanelEpisode({})
        episode.episode = SimpleNamespace(env=wrapper, _observe=observe, step=step, steps=0, finished=False)
        before = observation(floor=20)
        episode.reset = lambda spec: (before, EXPECTED)
        bank = {"world_seed": SEED, "prefix": [], "expected_provenance": EXPECTED,
                "public_hash": public_hash(before), "entering_action": 2,
                "entering_description": "enter boss", "entry_hp": 37,
                "first_combat_hash": public_hash(observation("combat", "BOSS"))}
        return episode, raw, bank, events

    def test_execute_before_initialization_actual_floor_hp_gold_and_stub_restore(self):
        episode, raw, bank, events = self.native_fake()
        original = raw.sts.BattleContext.exit_battle
        raw, base = episode._preinitialization(bank)
        self.assertEqual(events[0], ("execute", None))
        self.assertIsNone(raw.bc)
        self.assertEqual(raw.gc.screen_state, "BATTLE")
        self.assertEqual(base.floor_num, 21)
        episode._scratch(raw, base, -1, bank)
        result = episode._scratch(raw, base, 2, bank, Policy())
        self.assertTrue(result["win"])
        self.assertEqual(result["hp_lost"], 12)
        initializations = [e for e in events if e[0] == "initialize"]
        self.assertEqual(initializations[0], ("initialize", 40, 21, 37, 85))
        self.assertEqual(initializations[1], ("initialize", 40+1_000_003*3, 21, 37, 85))
        self.assertIs(raw.sts.BattleContext.exit_battle, original)
        self.assertIsNone(raw.bc)

    def test_unsalted_mismatch_and_policy_failure_restore_stub(self):
        episode, raw, bank, events = self.native_fake()
        original = raw.sts.BattleContext.exit_battle
        raw, base = episode._preinitialization(bank)
        with self.assertRaisesRegex(SimulatorError, "Unsalted"):
            episode._scratch(raw, base, -1, {**bank, "first_combat_hash": "wrong"})
        self.assertIs(raw.sts.BattleContext.exit_battle, original)
        class Bad:
            def choose_action(self, obs):
                raise ValueError("inference bug")
        with self.assertRaisesRegex(ValueError, "inference bug"):
            episode._scratch(raw, base, 0, bank, Bad())
        self.assertIs(raw.sts.BattleContext.exit_battle, original)

    def test_salted_entry_hp_is_measured_from_public_combat_state(self):
        episode, raw, bank, _ = self.native_fake()
        original_observe = episode.episode._observe
        def observe():
            value = original_observe()
            if value["phase"] == "combat":
                # Combat initialization can differ from the bank's entry HP;
                # the public combat HP also takes precedence over state.cur_hp.
                value["state"]["combat"] = {"player_cur_hp": 51 if raw.gc.seed == 40+1_000_003 else 61}
            return value
        episode.episode._observe = observe
        raw, base = episode._preinitialization(bank)
        for salt, hp in ((0, 51), (1, 61)):
            result = episode._scratch(raw, base, salt, bank, Policy())
            self.assertEqual(result["entry_hp"], hp)
            self.assertEqual(result["source_entry_hp"], 37)
            self.assertEqual(result["hp_after"], 25)
            self.assertEqual(result["hp_lost"], hp-25)

    def test_prepared_cache_replays_and_verifies_once_for_all_eighteen_fights(self):
        episode, raw, bank, events = self.native_fake()
        bank.update(source=SOURCE, start_id=0)
        raw, base = episode._preinitialization(bank)
        gc_before, episode_before = raw.gc, dict(vars(episode.episode))
        raw._combat_recent_actions, raw._combat_history_turn = ["original"], 99
        old_actions = raw._combat_recent_actions
        wrapper = episode.episode.env
        wrapper._raw_history, wrapper._public_history = ["raw original"], ["public original"]
        wrapper._selection_labels = {"original": "label"}
        wrapper_before = dict(vars(wrapper))
        class MutatingPolicy(Policy):
            identity = {"champion": "champion", "combat": "replacement"}
            def choose_action(self, value):
                self.assert_start()
                raw.gc.gold -= 30
                raw._combat_recent_actions.append("scratch")
                wrapper._selection_labels["scratch"] = "leak"
                return super().choose_action(value)
            def assert_start(self):
                if raw.gc.gold != 85 or wrapper._selection_labels:
                    raise AssertionError("Mutable state leaked across salts/policies")
        model = MutatingPolicy()
        with patch.object(episode, "_preinitialization", return_value=(raw, base)) as prepare, \
             patch.object(episode, "_scratch", wraps=episode._scratch) as scratch, \
             patch("sts_rl.fresh_panel._policy", return_value=model):
            for salt in [-1, *range(8)]:
                for policy in (None, "replacement"):
                    result = episode.fresh_fight(bank, salt, "champion", policy)
                    self.assertTrue(result["win"])
                    self.assertEqual(result["hp_lost"], 12)
                    self.assertIs(raw.gc, gc_before)
                    self.assertIs(raw._combat_recent_actions, old_actions)
                    self.assertEqual(vars(episode.episode), episode_before)
                    self.assertEqual(vars(wrapper), wrapper_before)
                    self.assertEqual((base.gold, base.seed, base.outcome), (85, 40, "UNDECIDED"))
            prepare.assert_called_once()
            self.assertEqual(scratch.call_count, 19)  # One verification + 18 fights.
            self.assertEqual(sum(call.args[4:] == () for call in scratch.call_args_list), 1)
            # Provenance/source/observation/start changes cannot reuse the base.
            for field, value in (("source", {"changed": True}), ("start_id", 1),
                                 ("expected_provenance", {"changed": True}), ("public_hash", "changed")):
                episode.fresh_fight({**bank, field: value}, 0, "champion")
            self.assertEqual(prepare.call_count, 5)

    def test_terminal_boss_and_portal_victory_defeat_restore_context_and_histories(self):
        for won in (True, False):
            with self.subTest(won=won):
                episode, raw, bank, events = self.native_fake()
                raw, base = episode._preinitialization(bank)
                original = raw.sts.BattleContext.exit_battle
                gc, native_state, wrapper_state = raw.gc, dict(vars(raw)), dict(vars(episode.episode.env))
                advance = raw.advance_to_decision
                def initialize():
                    advance()
                    raw.bc.outcome = "WIN" if won else "LOSS"
                    raw.bc.player_cur_hp = 25 if won else 0
                raw.advance_to_decision = initialize
                native_state["advance_to_decision"] = initialize
                result = episode._scratch(raw, base, -1, bank, Policy())
                self.assertEqual(result["win"], won)
                self.assertFalse(result["cut"])
                self.assertEqual(result["hp_after"], 25 if won else 0)
                self.assertEqual(vars(raw), native_state)
                self.assertEqual(vars(episode.episode.env), wrapper_state)
                self.assertIs(raw.gc, gc)
                self.assertEqual(raw.gc.outcome, "UNDECIDED")
                self.assertIs(raw.sts.BattleContext.exit_battle, original)

    def test_narrow_native_errors_have_fault_receipts_python_bugs_still_propagate(self):
        from sts_rl.fresh_panel import _serve_fresh
        for operation in (lambda: native_call("GameContext.legal_actions", [].pop),
                          lambda: native_call("GameAction.execute", [].index, 1)):
            class Episode:
                def reset(self):
                    return operation()
            class Connection:
                response = None
                def recv(self):
                    return "reset", []
                def send(self, value):
                    self.response = value
                def close(self):
                    pass
            connection = Connection()
            with patch("sts_rl.fresh_panel.FreshPanelEpisode", return_value=Episode()):
                _serve_fresh(connection, {})
            self.assertEqual(connection.response[0], "error")
            self.assertIn("Game", connection.response[1]["message"])
            self.assertIn("traceback", connection.response[1])
            worker = FreshPanelWorker(timeout=1.)
            worker._process = object()
            parent = Mock()
            parent.poll.return_value = True
            parent.recv.return_value = connection.response
            worker._connection = parent
            with patch.object(worker, "_kill"), patch.object(worker, "_record_fault") as record:
                with self.assertRaises(SimulatorError):
                    worker._call("reset", {"seed": SEED})
                record.assert_called_once()
                self.assertEqual(record.call_args.args[2], connection.response[1])
            worker._process, worker._connection = None, None
            worker.close()
        def python_bug():
            raise IndexError("Python indexing bug")
        with self.assertRaisesRegex(IndexError, "Python indexing bug"):
            native_call("GameContext.legal_actions", python_bug)
        with self.assertRaises(ValueError):
            native_call("BattleContext.search_best_action", [].index, "draw_order_seed")
        with self.assertRaises(TypeError):
            native_call("GameAction.execute", [].index)

    def test_real_fault_record_contains_native_operation_and_child_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = FreshPanelWorker(config={"fault_dir": temporary}, timeout=1.)
            worker._process = SimpleNamespace(pid=123, exitcode=None)
            connection = Mock()
            connection.poll.return_value = True
            child = {"type": "SimulatorError", "message": "GameAction.execute: ValueError: invalid OOC potion",
                     "traceback": "native boundary traceback"}
            connection.recv.return_value = ("error", child)
            worker._connection = connection
            with patch.object(worker, "_kill"), \
                 patch("sts_env._backend.lightspeed_import.native_metadata", return_value=EXPECTED["native_provenance"]):
                with self.assertRaisesRegex(SimulatorError, "invalid OOC potion"):
                    worker._call("reset", {"seed": SEED, "prefix": [2, 7]})
            record = json.loads(Path(worker.last_fault_path).read_text())
            self.assertEqual(record["task"], {"seed": SEED, "prefix": [2, 7]})
            self.assertEqual(record["exception"]["child"], child)
            self.assertEqual(record["native_provenance"], EXPECTED["native_provenance"])
            self.assertEqual(record["attempted_rpc"]["method"], "reset")
            worker._process, worker._connection = None, None
            worker.close()

    def test_cached_fight_rpc_still_has_finite_timeout_and_fault_record(self):
        worker = FreshPanelWorker(timeout=.25)
        worker._process, worker._connection = object(), Mock()
        worker._connection.poll.return_value = False
        with patch.object(worker, "_kill") as kill, patch.object(worker, "_record_fault") as record:
            from sts_env.env import SimulatorTimeout
            with self.assertRaises(SimulatorTimeout):
                worker.fresh_fight({"world_seed": SEED}, 0, "champion")
            worker._connection.poll.assert_called_once_with(.25)
            record.assert_called_once()
            kill.assert_called_once()
        worker._process, worker._connection = None, None
        worker.close()

    def test_native_init_failure_restores_all_saved_python_context(self):
        episode, raw, bank, events = self.native_fake()
        raw, base = episode._preinitialization(bank)
        original = raw.sts.BattleContext.exit_battle
        def fail_init():
            return native_call("BattleContext.init", [].pop)
        raw.advance_to_decision = fail_init
        states = [(obj, dict(vars(obj))) for obj in (raw, episode.episode.env, episode.episode)]
        with self.assertRaisesRegex(SimulatorError, "BattleContext.init"):
            episode._scratch(raw, base, 0, bank, Policy())
        self.assertIs(raw.sts.BattleContext.exit_battle, original)
        for obj, state in states:
            self.assertEqual(vars(obj), state)

    def test_rpc_python_errors_are_not_simulator_faults(self):
        class Connection:
            def send(self, request):
                pass
            def poll(self, timeout):
                return True
            def recv(self):
                return "python_error", {"type": "ValueError", "message": "shape bug", "traceback": "trace"}
        worker = FreshPanelWorker(timeout=1.)
        worker._process, worker._connection = object(), Connection()
        with patch.object(worker, "_kill"), patch.object(worker, "_record_fault") as record:
            with self.assertRaisesRegex(WorkerProgrammingError, "shape bug"):
                worker._call("reset", {"seed": 3})
            record.assert_not_called()
        worker._process, worker._connection = None, None
        worker.close()


if __name__ == "__main__":
    unittest.main()
