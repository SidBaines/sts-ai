"""Unit tests for scripts.run_grpo backend selection."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_grpo


class RunGrpoBackendTest(unittest.TestCase):
    def _argv(self, root: Path, *backend_args: str) -> list[str]:
        return [
            *backend_args,
            "--base-model",
            "base/model",
            "--tokenizer",
            "tokenizer/model",
            "--framing",
            "test framing",
            "--train-seeds-config",
            str(root / "seeds.json"),
            "--train-split",
            "train",
            "--out-dir",
            str(root / "out"),
            "--num-iterations",
            "2",
        ]

    def test_cuda_backend_uses_vllm_and_loop_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured_kwargs: dict[str, object] = {}
            vllm_agent = object()
            tokenizer = object()

            def fake_run_grpo(**kwargs: object) -> dict[str, bool]:
                captured_kwargs.update(kwargs)
                return {"ok": True}

            with (
                patch("sts_ai.train.grpo_loop.run_grpo", side_effect=fake_run_grpo) as run_mock,
                patch("sts_ai.agents.VllmJsonAgent", return_value=vllm_agent) as vllm_mock,
                patch("sts_ai.train.mlx_grpo.build_mlx_backend") as mlx_mock,
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                patch("scripts.run_until.load_split_seeds", return_value=[11, 22]),
                patch("sts_ai.lightspeed.LightspeedHybridEnv") as env_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_grpo.main(self._argv(root, "--backend", "cuda"))

            vllm_mock.assert_called_once_with(
                model_id="base/model",
                framing="test framing",
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_retries=1,
                enable_lora=True,
                enable_sleep_mode=True,
                gpu_memory_utilization=0.85,
                output_contract="reasoning_action",
                ooc_output_contract=None,
            )
            mlx_mock.assert_not_called()
            run_mock.assert_called_once()
            env_mock.assert_not_called()
            self.assertIs(captured_kwargs["agent"], vllm_agent)
            self.assertIs(captured_kwargs["tokenizer"], tokenizer)
            self.assertNotIn("run_streaming_fn", captured_kwargs)
            self.assertNotIn("train_fn", captured_kwargs)
            self.assertNotIn("build_dataset_fn", captured_kwargs)
            self.assertEqual(captured_kwargs["wandb_config"]["backend"], "cuda")
            self.assertFalse(captured_kwargs["wandb_config"]["thinking"])

    def test_mlx_backend_wires_mlx_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured_kwargs: dict[str, object] = {}
            mlx_agent = object()
            run_fn = object()
            train_fn = object()
            build_dataset_fn = object()
            tokenizer = object()
            backend = SimpleNamespace(
                agent=mlx_agent,
                run_fn=run_fn,
                train_fn=train_fn,
                build_dataset_fn=build_dataset_fn,
            )

            def fake_run_grpo(**kwargs: object) -> dict[str, bool]:
                captured_kwargs.update(kwargs)
                return {"ok": True}

            with (
                patch("sts_ai.train.grpo_loop.run_grpo", side_effect=fake_run_grpo) as run_mock,
                patch("sts_ai.agents.VllmJsonAgent") as vllm_mock,
                patch("sts_ai.train.mlx_grpo.build_mlx_backend", return_value=backend) as mlx_mock,
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                patch("scripts.run_until.load_split_seeds", return_value=[11, 22]),
                patch("sts_ai.lightspeed.LightspeedHybridEnv") as env_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_grpo.main(
                    self._argv(
                        root,
                        "--backend",
                        "mlx",
                        "--thinking",
                        "--max-seq-len",
                        "1024",
                        "--policy-seed-salt",
                        "1000",
                    )
                )

            mlx_mock.assert_called_once_with(
                base_model="base/model",
                framing="test framing",
                thinking=True,
                temperature=1.0,
                max_seq_len=1024,
                max_retries=1,
                resume_adapter=None,
                output_contract="reasoning_action",
                ooc_output_contract=None,
            )
            vllm_mock.assert_not_called()
            run_mock.assert_called_once()
            env_mock.assert_not_called()
            self.assertIs(captured_kwargs["agent"], mlx_agent)
            self.assertIs(captured_kwargs["tokenizer"], tokenizer)
            self.assertIs(captured_kwargs["run_streaming_fn"], run_fn)
            self.assertIs(captured_kwargs["train_fn"], train_fn)
            self.assertIs(captured_kwargs["build_dataset_fn"], build_dataset_fn)
            self.assertEqual(captured_kwargs["wandb_config"]["backend"], "mlx")
            self.assertTrue(captured_kwargs["wandb_config"]["thinking"])
            self.assertEqual(captured_kwargs["wandb_config"]["max_seq_len"], 1024)
            # Restart-supervisor salt reaches the loop and the config record.
            self.assertEqual(captured_kwargs["policy_seed_salt"], 1000)
            self.assertEqual(captured_kwargs["wandb_config"]["policy_seed_salt"], 1000)

    def test_cuda_backend_rejects_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_grpo.parse_args(self._argv(root, "--backend", "cuda", "--thinking"))


if __name__ == "__main__":
    unittest.main()
