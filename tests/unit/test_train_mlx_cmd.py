"""Unit tests for the pure MLX LoRA command builder."""
from __future__ import annotations

import unittest
from pathlib import Path

from sts_ai.train.train_mlx import build_lora_cmd


class TrainMlxCommandTest(unittest.TestCase):
    def test_base_argv(self):
        cmd = build_lora_cmd(
            python_exe="python",
            base_model="base/model",
            data_dir=Path("data"),
            out_adapter_dir=Path("adapter"),
            num_layers=4,
            iters=9,
            batch_size=2,
            learning_rate=0.0002,
        )

        self.assertEqual(
            cmd,
            [
                "python",
                "-m",
                "mlx_lm",
                "lora",
                "--model",
                "base/model",
                "--train",
                "--data",
                "data",
                "--adapter-path",
                "adapter",
                "--iters",
                "9",
                "--batch-size",
                "2",
                "--num-layers",
                "4",
                "--learning-rate",
                "0.0002",
                "--max-seq-length",
                "8192",
                "--mask-prompt",
            ],
        )
        self.assertIn("--mask-prompt", cmd)
        self.assertNotIn("--report-to", cmd)
        self.assertNotIn("--project-name", cmd)
        self.assertNotIn("--steps-per-eval", cmd)
        self.assertNotIn("--save-every", cmd)

    def test_mask_prompt_can_be_disabled(self):
        cmd = build_lora_cmd(
            python_exe="python",
            base_model="base/model",
            data_dir=Path("data"),
            out_adapter_dir=Path("adapter"),
            num_layers=4,
            iters=9,
            batch_size=2,
            learning_rate=0.0002,
            mask_prompt=False,
        )

        self.assertNotIn("--mask-prompt", cmd)

    def test_optional_wandb_and_eval_flags(self):
        cmd = build_lora_cmd(
            python_exe="python",
            base_model="base/model",
            data_dir=Path("data"),
            out_adapter_dir=Path("adapter"),
            num_layers=4,
            iters=9,
            batch_size=2,
            learning_rate=0.0002,
            wandb_project="P",
            steps_per_eval=10,
            save_every=5,
        )

        report_index = cmd.index("--report-to")
        self.assertEqual(
            cmd[report_index : report_index + 4],
            ["--report-to", "wandb", "--project-name", "P"],
        )
        self.assertEqual(cmd[cmd.index("--steps-per-eval") + 1], "10")
        self.assertEqual(cmd[cmd.index("--save-every") + 1], "5")
        self.assertNotIn("--steps-per-report", cmd)
        self.assertNotIn("--val-batches", cmd)

    def test_max_seq_length_can_be_overridden(self):
        cmd = build_lora_cmd(
            python_exe="python",
            base_model="base/model",
            data_dir=Path("data"),
            out_adapter_dir=Path("adapter"),
            num_layers=4,
            iters=9,
            batch_size=2,
            learning_rate=0.0002,
            max_seq_length=1234,
        )

        self.assertEqual(cmd[cmd.index("--max-seq-length") + 1], "1234")


if __name__ == "__main__":
    unittest.main()
