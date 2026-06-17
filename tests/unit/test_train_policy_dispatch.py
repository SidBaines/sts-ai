"""Unit tests for train_policy backend dispatch."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import train_policy


class TrainPolicyDispatchTest(unittest.TestCase):
    def test_mlx_backend_forwards_paths_manifest_and_mlx_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "sft.jsonl"
            manifest = root / "sft.manifest.json"
            out = root / "adapter"
            dataset.write_text("", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")

            args = train_policy.parse_args(
                [
                    "--backend",
                    "mlx",
                    "--base-model",
                    "mlx/model",
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(out),
                    "--num-layers",
                    "4",
                    "--iters",
                    "9",
                    "--batch-size",
                    "2",
                    "--learning-rate",
                    "0.0002",
                ]
            )

            with patch("sts_ai.train.train_mlx.train", return_value=out) as train_mock:
                result = train_policy.dispatch(args)

            self.assertEqual(result, out)
            train_mock.assert_called_once_with(
                dataset,
                "mlx/model",
                out,
                num_layers=4,
                iters=9,
                batch_size=2,
                learning_rate=0.0002,
                manifest_path=manifest,
            )

    def test_trl_backend_forwards_paths_manifest_and_trl_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "sft.jsonl"
            manifest = root / "manifest.json"
            out = root / "adapter"
            dataset.write_text("", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")

            args = train_policy.parse_args(
                [
                    "--backend",
                    "trl",
                    "--base-model",
                    "hf/model",
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(out),
                    "--manifest",
                    str(manifest),
                    "--lora-r",
                    "8",
                    "--lora-alpha",
                    "16",
                    "--lora-dropout",
                    "0.1",
                    "--epochs",
                    "3",
                    "--per-device-batch-size",
                    "2",
                    "--grad-accum",
                    "4",
                    "--max-seq-len",
                    "1024",
                    "--learning-rate",
                    "0.0003",
                ]
            )

            with patch("sts_ai.train.train_trl.train", return_value=out) as train_mock:
                result = train_policy.dispatch(args)

            self.assertEqual(result, out)
            train_mock.assert_called_once_with(
                dataset,
                "hf/model",
                out,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                epochs=3,
                learning_rate=0.0003,
                per_device_batch_size=2,
                grad_accum=4,
                max_seq_len=1024,
                manifest_path=manifest,
            )


if __name__ == "__main__":
    unittest.main()
