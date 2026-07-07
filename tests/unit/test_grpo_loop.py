from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

from sts_ai.streaming_rollout import run_streaming_rollouts
from sts_ai.train.grpo_loop import (
    _train_metrics,
    floor_stats,
    run_grpo,
    select_iteration_seeds,
)
from tests.unit.test_parallel_rollout import FakeParallelEnv
from tests.unit.test_streaming_rollout import FakeStreamingAgent


class GrpoLoopSeedSelectionTest(unittest.TestCase):
    def test_select_iteration_seeds_rotates_and_wraps(self) -> None:
        train_seeds = [10, 11, 12, 13, 14]

        self.assertEqual(select_iteration_seeds(train_seeds, 0, 2), [10, 11])
        self.assertEqual(select_iteration_seeds(train_seeds, 1, 2), [12, 13])
        self.assertEqual(select_iteration_seeds(train_seeds, 2, 2), [14, 10])

    def test_select_iteration_seeds_returns_all_when_window_covers_split(self) -> None:
        self.assertEqual(select_iteration_seeds([10, 11, 12], 3, 3), [10, 11, 12])
        self.assertEqual(select_iteration_seeds([10, 11, 12], 3, 8), [10, 11, 12])


class TrackingStreamingAgent(FakeStreamingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.adapter_paths: list[str] = []

    def wake(self) -> None:
        self.events.append("wake")

    def sleep(self, level: int = 1) -> None:
        self.events.append("sleep")

    def set_adapter(self, adapter_path: str) -> None:
        self.events.append("set_adapter")
        self.adapter_paths.append(adapter_path)

    def stream_submit(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("generate")
        super().stream_submit(*args, **kwargs)


class GrpoLoopControlFlowTest(unittest.TestCase):
    def test_run_grpo_sleeps_trains_and_hot_swaps_each_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent = TrackingStreamingAgent()
            train_calls: list[dict[str, Any]] = []

            def make_env(world_seed: int) -> FakeParallelEnv:
                return FakeParallelEnv(world_seed=world_seed, decisions=1)

            def build_dataset_fn(
                rollout_dir: Path,
                **kwargs: Any,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                self.assertTrue(list(rollout_dir.glob("seed_*_r*.jsonl")))
                return (
                    [{"prompt": "p", "completion": "c", "advantage": 0.0}],
                    {"advantage_report": {"advantage_mean": 0.0}},
                )

            def train_fn(**kwargs: Any) -> Path:
                agent.events.append("train")
                train_calls.append(kwargs)
                out_adapter_dir = Path(kwargs["out_adapter_dir"])
                out_adapter_dir.mkdir(parents=True, exist_ok=True)
                return out_adapter_dir

            summary = run_grpo(
                agent=agent,
                make_env=make_env,
                base_model="base-model",
                tokenizer=object(),
                tokenizer_id="tokenizer-id",
                framing="framing",
                train_seeds=[10, 11],
                out_dir=root,
                num_iterations=2,
                group_size=2,
                seeds_per_iter=1,
                concurrency=2,
                max_decisions=2,
                build_dataset_fn=build_dataset_fn,
                train_fn=train_fn,
                run_streaming_fn=run_streaming_rollouts,
            )

            iter0_adapter = str(root / "iter_0" / "adapter")
            iter1_adapter = str(root / "iter_1" / "adapter")

            self.assertEqual(
                agent.events,
                [
                    "wake",
                    "generate",
                    "generate",
                    "sleep",
                    "train",
                    "set_adapter",
                    "wake",
                    "generate",
                    "generate",
                    "sleep",
                    "train",
                    "set_adapter",
                ],
            )
            self.assertEqual(train_calls[0]["init_adapter_path"], None)
            self.assertEqual(train_calls[1]["init_adapter_path"], iter0_adapter)
            # Batch-size/grad-accum must reach the trainer (GPU-utilization lever).
            self.assertEqual(train_calls[0]["per_device_batch_size"], 1)
            self.assertEqual(train_calls[0]["grad_accum"], 8)
            self.assertEqual(agent.adapter_paths, [iter0_adapter, iter1_adapter])
            self.assertEqual(summary["final_adapter"], iter1_adapter)
            self.assertEqual(summary["iterations"][0]["n_specs"], 2)
            self.assertEqual(summary["iterations"][0]["n_examples"], 1)
            self.assertEqual(summary["iterations"][1]["current_adapter"], iter1_adapter)
            self.assertTrue((root / "iter_0" / "pg.jsonl").exists())
            self.assertEqual(
                train_calls[0]["manifest_path"],
                root / "iter_0" / "pg.jsonl.manifest.json",
            )


class GrpoLoopResumeTest(unittest.TestCase):
    def test_resume_starts_at_iteration_and_seeds_from_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent = TrackingStreamingAgent()
            train_calls: list[dict[str, Any]] = []

            def make_env(world_seed: int) -> FakeParallelEnv:
                return FakeParallelEnv(world_seed=world_seed, decisions=1)

            def build_dataset_fn(
                rollout_dir: Path, **kwargs: Any
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                return (
                    [{"prompt": "p", "completion": "c", "advantage": 0.0}],
                    {"advantage_report": {"advantage_mean": 0.0}},
                )

            def train_fn(**kwargs: Any) -> Path:
                train_calls.append(kwargs)
                out_adapter_dir = Path(kwargs["out_adapter_dir"])
                out_adapter_dir.mkdir(parents=True, exist_ok=True)
                return out_adapter_dir

            summary = run_grpo(
                agent=agent,
                make_env=make_env,
                base_model="base-model",
                tokenizer=object(),
                tokenizer_id="tokenizer-id",
                framing="framing",
                train_seeds=[10, 11, 12, 13],
                out_dir=root,
                num_iterations=4,
                group_size=2,
                seeds_per_iter=1,
                start_iteration=2,
                init_adapter_path="/hf/iter_1/adapter",
                concurrency=2,
                max_decisions=2,
                build_dataset_fn=build_dataset_fn,
                train_fn=train_fn,
                run_streaming_fn=run_streaming_rollouts,
            )

            # Only iterations 2 and 3 run.
            self.assertEqual([s["iteration"] for s in summary["iterations"]], [2, 3])
            # The first resumed iteration trains from the supplied resume adapter,
            # then the next iteration chains off iter_2's freshly written adapter.
            self.assertEqual(train_calls[0]["init_adapter_path"], "/hf/iter_1/adapter")
            self.assertEqual(
                train_calls[1]["init_adapter_path"], str(root / "iter_2" / "adapter")
            )
            self.assertEqual(summary["final_adapter"], str(root / "iter_3" / "adapter"))
            # Only the resumed iteration dirs exist.
            self.assertTrue((root / "iter_2" / "pg.jsonl").exists())
            self.assertFalse((root / "iter_0").exists())

    def test_invalid_start_iteration_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_grpo(
                agent=TrackingStreamingAgent(),
                make_env=lambda ws: FakeParallelEnv(world_seed=ws, decisions=1),
                base_model="m",
                tokenizer=object(),
                tokenizer_id="t",
                framing="f",
                train_seeds=[1, 2],
                out_dir=Path("/tmp/unused_grpo_resume"),
                num_iterations=2,
                start_iteration=2,  # == num_iterations -> invalid
            )


class FloorStatsTest(unittest.TestCase):
    def _write_meta(self, root: Path, stem: str, **fields: Any) -> None:
        (root / f"{stem}.meta.json").write_text(json.dumps(fields), encoding="utf-8")

    def test_floor_stats_aggregates_reward_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_meta(root, "seed_1_r0", final_floor=10, outcome="DEFEAT")
            self._write_meta(root, "seed_2_r0", final_floor=20, outcome="VICTORY")
            self._write_meta(
                root, "seed_3_r0", final_floor=30, outcome="DEFEAT",
                extra={"budget_truncated": True},
            )
            stats = floor_stats(root)
            self.assertEqual(stats["n_rollouts"], 3)
            self.assertEqual(stats["mean_floor"], 20.0)
            self.assertEqual(stats["median_floor"], 20)
            self.assertEqual(stats["max_floor"], 30)
            self.assertEqual(stats["min_floor"], 10)
            self.assertEqual(stats["n_win"], 1)
            self.assertAlmostEqual(stats["win_rate"], 1 / 3)
            self.assertAlmostEqual(stats["budget_truncated_rate"], 1 / 3)

    def test_floor_stats_empty_dir_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = floor_stats(Path(tmpdir))
            self.assertEqual(stats["n_rollouts"], 0)
            self.assertEqual(stats["mean_floor"], 0.0)
            self.assertEqual(stats["win_rate"], 0.0)

    def test_floor_stats_skips_malformed_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "seed_1_r0.meta.json").write_text("{not json", encoding="utf-8")
            self._write_meta(root, "seed_2_r0", final_floor=5, outcome="DEFEAT")
            stats = floor_stats(root)
            self.assertEqual(stats["n_rollouts"], 1)
            self.assertEqual(stats["mean_floor"], 5.0)


class RunGrpoEntrypointImportTest(unittest.TestCase):
    """Regression: ``python scripts/run_grpo.py`` with ``PYTHONPATH=src`` must be
    able to ``from scripts.run_until import ...``. Run as a script, only the
    ``scripts/`` dir lands on sys.path[0] (not the repo root), so without the
    module-level repo-root bootstrap the entrypoint died with
    ``ModuleNotFoundError: No module named 'scripts'`` before doing any work.
    """

    def test_entrypoint_resolves_scripts_namespace_package(self) -> None:
        env = {**os.environ, "PYTHONPATH": "src"}
        # A non-existent seeds config makes load_split_seeds (which runs *after*
        # the scripts import) fail fast, so this never downloads a model.
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_grpo.py",
                "--base-model", "dummy",
                "--tokenizer", "dummy",
                "--train-seeds-config", "/nonexistent/frozen_seeds.json",
                "--train-split", "smoke",
                "--out-dir", "/tmp/grpo_entrypoint_test",
                "--num-iterations", "1",
            ],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertNotIn("No module named 'scripts'", result.stderr)


class TrainMetricsTest(unittest.TestCase):
    def test_train_metrics_averages_numeric_keys_and_skips_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Path(tmpdir)
            history = [
                {"loss": 1.0, "mean_kl": 0.1, "clip_fraction": 0.2, "epoch": 0.0, "step": 1},
                {"loss": 3.0, "mean_kl": 0.3, "clip_fraction": 0.4, "epoch": 1.0, "step": 2},
            ]
            (adapter / "trainer_log.json").write_text(json.dumps(history), encoding="utf-8")
            metrics = _train_metrics(adapter)
            self.assertAlmostEqual(metrics["loss"], 2.0)
            self.assertAlmostEqual(metrics["mean_kl"], 0.2)
            self.assertAlmostEqual(metrics["clip_fraction"], 0.3)
            self.assertNotIn("epoch", metrics)
            self.assertNotIn("step", metrics)

    def test_train_metrics_missing_log_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(_train_metrics(Path(tmpdir)), {})


if __name__ == "__main__":
    unittest.main()
