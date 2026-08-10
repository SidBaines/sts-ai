"""Unit tests for train_policy backend dispatch."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import train_policy


MODEL_REVISION = "eec12d0899edea9b738ab1009af9159cdfd70d71"


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
                    "--wandb-project",
                    "P",
                    "--steps-per-eval",
                    "10",
                    "--save-every",
                    "5",
                    "--eval-fraction",
                    "0.0",
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
                wandb_project="P",
                steps_per_eval=10,
                steps_per_report=None,
                save_every=5,
                val_batches=None,
                max_seq_length=8192,
                loss_mask_mode="auto",
                valid_fraction=0.0,
                expected_example_count=None,
                seed=0,
                lora_rank=8,
                lora_scale=20.0,
                lora_dropout=0.0,
                grad_accumulation_steps=1,
                action_token_only=False,
                action_token_weight=1.0,
                preserve_row_order=False,
                model_revision=None,
            )

    def test_eval_fraction_defaults_preserve_each_backend(self):
        common = [
            "--base-model",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--out",
            "adapter",
        ]

        mlx_args = train_policy.parse_args(["--backend", "mlx", *common])
        trl_args = train_policy.parse_args(["--backend", "trl", *common])

        self.assertEqual(mlx_args.eval_fraction, 0.1)
        self.assertEqual(trl_args.eval_fraction, 0.0)

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
                    "--wandb-project",
                    "P",
                    "--run-name",
                    "R",
                    "--eval-fraction",
                    "0.1",
                    "--eval-steps",
                    "20",
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
                max_steps=-1,
                learning_rate=0.0003,
                per_device_batch_size=2,
                grad_accum=4,
                max_seq_len=1024,
                manifest_path=manifest,
                wandb_project="P",
                run_name="R",
                eval_fraction=0.1,
                eval_steps=20,
                loss_mask_mode="auto",
            )

    def test_explicit_action_mask_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "sft.jsonl"
            out = root / "adapter"
            dataset.write_text("", encoding="utf-8")
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
                    "--loss-mask",
                    "action",
                ]
            )
            with patch("sts_ai.train.train_mlx.train", return_value=out) as train_mock:
                train_policy.dispatch(args)
            self.assertEqual(train_mock.call_args.kwargs["loss_mask_mode"], "action")

    def test_mlx_expected_example_count_is_validated_and_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "sft.jsonl"
            out = root / "adapter"
            dataset.write_text("", encoding="utf-8")
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
                    "--expected-example-count",
                    "32",
                    "--mlx-seed",
                    "7",
                    "--mlx-lora-rank",
                    "4",
                    "--mlx-lora-scale",
                    "9",
                    "--mlx-lora-dropout",
                    "0.1",
                    "--mlx-grad-accum",
                    "2",
                    "--mlx-model-revision",
                    MODEL_REVISION,
                    "--mlx-action-token-only",
                    "--mlx-preserve-row-order",
                ]
            )
            with patch("sts_ai.train.train_mlx.train", return_value=out) as train_mock:
                train_policy.dispatch(args)
            self.assertEqual(
                train_mock.call_args.kwargs["expected_example_count"],
                32,
            )
            self.assertEqual(train_mock.call_args.kwargs["seed"], 7)
            self.assertEqual(train_mock.call_args.kwargs["lora_rank"], 4)
            self.assertEqual(train_mock.call_args.kwargs["lora_scale"], 9.0)
            self.assertEqual(train_mock.call_args.kwargs["lora_dropout"], 0.1)
            self.assertEqual(train_mock.call_args.kwargs["grad_accumulation_steps"], 2)
            self.assertTrue(train_mock.call_args.kwargs["action_token_only"])
            self.assertEqual(train_mock.call_args.kwargs["action_token_weight"], 1.0)
            self.assertTrue(train_mock.call_args.kwargs["preserve_row_order"])
            self.assertEqual(
                train_mock.call_args.kwargs["model_revision"],
                MODEL_REVISION,
            )

    def test_expected_example_count_rejects_non_mlx_and_nonpositive_values(self):
        common = [
            "--base-model",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--out",
            "adapter",
            "--expected-example-count",
        ]
        with self.assertRaises(SystemExit):
            train_policy.parse_args(["--backend", "trl", *common, "32"])
        with self.assertRaises(SystemExit):
            train_policy.parse_args(["--backend", "mlx", *common, "0"])

    def test_mlx_knobs_reject_wrong_backend_and_invalid_values(self):
        common = [
            "--base-model",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--out",
            "adapter",
        ]
        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                ["--backend", "trl", *common, "--mlx-seed", "0"]
            )
        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                ["--backend", "trl", *common, "--mlx-action-token-only"]
            )
        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                ["--backend", "trl", *common, "--mlx-preserve-row-order"]
            )
        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                ["--backend", "trl", *common, "--mlx-action-token-weight", "8"]
            )
        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                [
                    "--backend",
                    "trl",
                    *common,
                    "--mlx-model-revision",
                    MODEL_REVISION,
                ]
            )
        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                [
                    "--backend",
                    "mlx",
                    *common,
                    "--batch-size",
                    "2",
                    "--mlx-preserve-row-order",
                ]
            )
        invalid = (
            ("--mlx-seed", "-1"),
            ("--mlx-lora-rank", "0"),
            ("--mlx-lora-scale", "0"),
            ("--mlx-lora-dropout", "1"),
            ("--mlx-grad-accum", "0"),
            ("--mlx-action-token-weight", "0"),
            ("--mlx-action-token-weight", "nan"),
        )
        for flag, value in invalid:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    train_policy.parse_args(
                        ["--backend", "mlx", *common, flag, value]
                    )
        for invalid_revision in ("main", "E" * 40, "e" * 39, "g" * 40):
            with self.subTest(invalid_revision=invalid_revision):
                with self.assertRaises(SystemExit):
                    train_policy.parse_args(
                        [
                            "--backend",
                            "mlx",
                            *common,
                            "--mlx-model-revision",
                            invalid_revision,
                        ]
                    )

    def test_mlx_action_token_weight_is_forwarded_and_conflicts_with_only(self):
        common = [
            "--backend",
            "mlx",
            "--base-model",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--out",
            "adapter",
        ]
        args = train_policy.parse_args(
            [*common, "--mlx-action-token-weight", "8"]
        )
        with patch(
            "sts_ai.train.train_mlx.train",
            return_value=Path("adapter"),
        ) as train_mock:
            train_policy.dispatch(args)
        self.assertEqual(train_mock.call_args.kwargs["action_token_weight"], 8.0)

        with self.assertRaises(SystemExit):
            train_policy.parse_args(
                [
                    *common,
                    "--mlx-action-token-only",
                    "--mlx-action-token-weight",
                    "8",
                ]
            )


if __name__ == "__main__":
    unittest.main()
