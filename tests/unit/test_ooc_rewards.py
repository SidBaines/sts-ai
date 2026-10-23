from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from sts_policy.curriculum import reward_components
from sts_policy.model_agent import CompositeActorCritic
from sts_rl import common, ooc_trainer, parallel_collect
from sts_rl.ooc_trainer import OOCTrainConfig, ReturnNormalizer


class OOCRewardWorkerTests(unittest.TestCase):
    def test_floor_weight_overrides_both_worker_types_without_changing_defaults(self):
        original = dict(common.WORKER_CONFIG)
        for in_process in (False, True):
            for slim in (False, True):
                with self.subTest(in_process=in_process, slim=slim):
                    workers = common.make_workers(2, in_process=in_process, slim=slim,
                                                  progress_floor_weight=0.0)
                    try:
                        for worker in workers:
                            config = (worker.worker if slim else worker).config
                            self.assertEqual(config["progress_floor_weight"], 0.0)
                    finally:
                        common.close_workers(workers)
                    self.assertEqual(common.WORKER_CONFIG, original)
                    self.assertEqual(common.WORKER_CONFIG["progress_floor_weight"], 0.05)

    def test_none_and_omitted_floor_weight_keep_default_worker_config(self):
        for in_process in (False, True):
            for options in ({}, {"progress_floor_weight": None}):
                with self.subTest(in_process=in_process, options=options):
                    workers = common.make_workers(1, in_process=in_process, slim=False, **options)
                    try:
                        for key, value in common.WORKER_CONFIG.items():
                            self.assertEqual(workers[0].config[key], value)
                    finally:
                        common.close_workers(workers)

    def test_invalid_floor_weights_are_rejected_before_creating_workers(self):
        for weight in (-0.01, float("nan"), float("inf"), -float("inf")):
            with self.subTest(weight=weight), \
                    patch.object(common, "InProcessWorker") as in_process, \
                    patch.object(common, "CurriculumWorker") as isolated:
                with self.assertRaisesRegex(ValueError, "progress_floor_weight"):
                    common.make_workers(1, progress_floor_weight=weight)
                in_process.assert_not_called()
                isolated.assert_not_called()

    def test_zero_floor_weight_leaves_only_terminal_reward_on_floor_change(self):
        previous = {"state": {"floor": 8}}
        observation = {"state": {"floor": 9}}
        for outcome, terminal in (("victory", 1.0), ("act_clear", 1.0), ("defeat", -1.0), ("", 0.0)):
            with self.subTest(outcome=outcome):
                components = reward_components(previous, observation, task_outcome=outcome,
                                               terminated=bool(outcome), mode="progress",
                                               progress_floor_weight=0.0)
                self.assertEqual(components, {"terminal": terminal, "potential": 0.0,
                                              "progress": 0.0, "terminal_hp": 0.0})
                self.assertEqual(sum(components.values()), terminal)
                default = reward_components(previous, observation, task_outcome=outcome,
                                            terminated=bool(outcome), mode="progress")
                self.assertEqual(default["progress"], 0.05)

    def test_collector_pool_forwards_floor_weight_to_worker_factory(self):
        for weight in (None, 0.0, 0.125):
            with self.subTest(weight=weight):
                with patch.object(parallel_collect.CollectorPool, "_launch"):
                    pool = parallel_collect.CollectorPool(2, workers=4, checkpoint="unused",
                                                          progress_floor_weight=weight)
                try:
                    conn = Mock()
                    conn.recv.return_value = ("close", None)
                    with patch.object(parallel_collect, "load_policy", return_value=(torch.nn.Linear(2, 2), {})), \
                            patch.object(parallel_collect, "make_workers", return_value=[]) as factory:
                        parallel_collect._collector_main(conn, *pool._collector_args)
                    self.assertEqual(factory.call_args.args, (2,))
                    self.assertEqual(factory.call_args.kwargs["progress_floor_weight"], weight)
                finally:
                    pool.close()


class OOCRewardTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = CompositeActorCritic(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        inherited = ReturnNormalizer(count=10, mean=20.0, m2=250.0)
        self.template = {"trainer_state": inherited.trainer_state()}
        self.cfg = OOCTrainConfig(out=str(self.root / "run"), checkpoint="unused", games=2,
                                  iterations=1, eval_every=0, health_every=0,
                                  normalize_returns=True, exploration_stats=False, threads=1)
        self.collection_scales = []

    def train(self, cfg, *, resume=False):
        def collect(*args, **kwargs):
            self.collection_scales.append(kwargs["value_scale"])
            rows = [SimpleNamespace(returns=value, advantage=0.0, encoded_observation=object())
                    for value in (2.0, 6.0)]
            return {"kept": [rows], "game_episodes": [{"floor": 5, "task_outcome": "defeat"}],
                    "faults": 0, "fault_records": [], "stats": {}, "collector_restarts": 0}

        def update(*args, **kwargs):
            return {"optimizer": kwargs["optimizer"], "steps": 1, "early_stop": False}

        pool = Mock(restarts=0)
        pool.collect_ooc.side_effect = collect
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(ooc_trainer, "load_policy", return_value=(self.model, self.template)), \
                patch.object(ooc_trainer, "begin_run", return_value={}), \
                patch.object(ooc_trainer, "make_workers", return_value=[]) as factory, \
                patch.object(parallel_collect, "CollectorPool", return_value=pool) as pool_factory, \
                patch.object(ooc_trainer, "evaluate", return_value={"summary": {}, "records": []}), \
                patch.object(ooc_trainer, "collect_ooc_batch", side_effect=collect), \
                patch.object(ooc_trainer, "write_collection"), \
                patch.object(ooc_trainer, "update_ppo", side_effect=update):
            ooc_trainer.train(cfg, resume=resume)
        return factory, pool_factory

    def latest(self, cfg):
        return torch.load(Path(cfg.out) / "latest.pt", weights_only=False)

    def records(self, cfg, event):
        rows = [json.loads(line) for line in (Path(cfg.out) / "log.jsonl").read_text().splitlines()]
        return [row for row in rows if row["event"] == event]

    def test_defaults_and_generated_cli_flags(self):
        cfg = OOCTrainConfig(out="unused")
        self.assertEqual(cfg.progress_floor_weight, 0.05)
        self.assertFalse(cfg.reset_return_normalizer)
        script = Path(__file__).resolve().parents[2] / "scripts" / "rl" / "train_ooc.py"
        with patch.object(sys, "argv", [str(script), "--out", self.cfg.out,
                                       "--progress-floor-weight", "0", "--reset-return-normalizer"]), \
                patch.object(ooc_trainer, "train") as train:
            runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(train.call_args.args[0].progress_floor_weight, 0.0)
        self.assertTrue(train.call_args.args[0].reset_return_normalizer)

    def test_trainer_passes_floor_weight_to_both_collection_paths(self):
        for collectors in (1, 2):
            with self.subTest(collectors=collectors):
                cfg = replace(self.cfg, out=str(self.root / str(collectors)), collectors=collectors,
                              progress_floor_weight=0.0)
                factory, pool_factory = self.train(cfg)
                selected, unused = (factory, pool_factory) if collectors == 1 else (pool_factory, factory)
                self.assertEqual(selected.call_args.kwargs["progress_floor_weight"], 0.0)
                unused.assert_not_called()

    def test_fresh_start_resets_only_when_requested_and_logs_reset(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                cfg = replace(self.cfg, out=str(self.root / str(reset)), reset_return_normalizer=reset)
                self.train(cfg)
                normalizer = ReturnNormalizer() if reset else ReturnNormalizer.restore(self.template, enabled=True)
                normalizer.update([2.0, 6.0])
                self.assertEqual(self.latest(cfg)["trainer_state"], normalizer.trainer_state())
                event = self.records(cfg, "start")[0]
                if reset:
                    self.assertTrue(event["reset_return_normalizer"])
                else:
                    self.assertNotIn("reset_return_normalizer", event)
        self.assertEqual(self.collection_scales, [5.0, 1.0])

    def test_resume_keeps_normalizer_and_uses_current_config_when_iterations_increase(self):
        cfg = replace(self.cfg, iterations=12, progress_floor_weight=0.0, reset_return_normalizer=True)
        self.train(cfg)
        first = self.latest(cfg)
        normalizer = ReturnNormalizer.restore(first, enabled=True)
        self.assertEqual(normalizer.count, 24)
        resumed = replace(cfg, iterations=24, progress_floor_weight=0.125)
        factory, _ = self.train(resumed, resume=True)
        for _ in range(12):
            normalizer.update([2.0, 6.0])
        latest = self.latest(resumed)
        self.assertEqual(latest["trainer_state"], normalizer.trainer_state())
        self.assertEqual(latest["iteration"], 24)
        self.assertEqual(latest["steps_total"], 24)
        self.assertEqual(latest["config"]["iterations"], 24)
        self.assertEqual(latest["config"]["progress_floor_weight"], 0.125)
        self.assertEqual(factory.call_args.kwargs["progress_floor_weight"], 0.125)
        self.assertEqual(self.collection_scales, [1.0] + [2.0] * 23)
        self.assertEqual(len(self.records(resumed, "start")), 1)
        self.assertEqual(self.records(resumed, "resume")[0]["iteration"], 12)
        self.assertEqual([row["iteration"] for row in self.records(resumed, "train")], list(range(1, 25)))


if __name__ == "__main__":
    unittest.main()
