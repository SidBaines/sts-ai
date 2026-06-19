from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from sts_ai.streaming_rollout import run_streaming_rollouts
from sts_ai.train.grpo_loop import run_grpo, select_iteration_seeds
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


if __name__ == "__main__":
    unittest.main()
