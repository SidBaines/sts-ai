from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.decompose_sft_loss import (
    _checkpoint_provenance,
    _resolve_model_snapshot,
    _runtime_tokenizer_provenance,
    _training_report_provenance,
    _write_text_atomically,
    parse_args,
)


class DecomposeSftLossCliTests(unittest.TestCase):
    def test_required_arguments_and_optional_training_report_hash(self):
        args = parse_args(
            [
                "dataset.jsonl",
                "--manifest",
                "manifest.json",
                "--checkpoint",
                "checkpoint",
                "--expected-model-revision",
                "d" * 40,
                "--checkpoint-step",
                "1500",
                "--training-report",
                "native_mlx_data_report.json",
                "--expected-dataset-sha256",
                "a" * 64,
                "--expected-manifest-sha256",
                "b" * 64,
                "--expected-training-report-sha256",
                "c" * 64,
                "--out",
                "report.json",
            ]
        )

        self.assertEqual(args.dataset, Path("dataset.jsonl"))
        self.assertEqual(args.checkpoint, Path("checkpoint"))
        self.assertEqual(args.checkpoint_step, 1500)
        self.assertEqual(args.expected_model_revision, "d" * 40)
        self.assertEqual(args.expected_training_report_sha256, "c" * 64)

    def test_model_snapshot_requires_and_content_addresses_exact_cached_revision(
        self,
    ):
        revision = "eec12d0899edea9b738ab1009af9159cdfd70d71"
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / revision
            snapshot.mkdir()
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (snapshot / name).write_text(
                    f"{name}\n",
                    encoding="utf-8",
                )
            (snapshot / "model-00001-of-00001.safetensors").write_bytes(
                b"weights"
            )

            with patch(
                "huggingface_hub.snapshot_download",
                return_value=str(snapshot),
            ) as download:
                resolved, provenance = _resolve_model_snapshot(
                    "model",
                    expected_revision=revision,
                )

            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(provenance["revision"], revision)
            self.assertEqual(len(provenance["identity_sha256"]), 64)
            self.assertEqual(
                set(provenance["files"]),
                {
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model-00001-of-00001.safetensors",
                },
            )
            download.assert_called_once_with(
                repo_id="model",
                revision=revision,
                local_files_only=True,
            )

        with self.assertRaisesRegex(ValueError, "40-character"):
            _resolve_model_snapshot("model", expected_revision="main")

    def test_runtime_tokenizer_is_content_addressed(self):
        class Backend:
            def to_str(self):
                return '{"vocab":{}}'

        class Tokenizer:
            backend_tokenizer = Backend()
            chat_template = "template"
            name_or_path = "snapshot"
            vocab_size = 10

        provenance = _runtime_tokenizer_provenance(Tokenizer())

        self.assertEqual(provenance["name_or_path"], "snapshot")
        self.assertEqual(provenance["vocab_size"], 10)
        self.assertEqual(len(provenance["backend_tokenizer_sha256"]), 64)
        self.assertEqual(len(provenance["chat_template_sha256"]), 64)

    def test_checkpoint_step_is_proved_against_full_training_run_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source_training_run"
            checkpoint = root / "step1500"
            source.mkdir()
            checkpoint.mkdir()
            schedule_sha256 = "d" * 64
            adapter_config = {
                "model": "model",
                "fine_tune_type": "lora",
                "batch_size": 1,
                "iters": 3000,
                "grad_accumulation_steps": 1,
                "mask_prompt": True,
                "action_token_only": False,
                "preserve_row_order": True,
                "expected_schedule_sha256": schedule_sha256,
                "save_every": 750,
                "adapter_path": str(source),
            }
            mlx_config = {
                "grad_accumulation_steps": 1,
                "action_token_only": False,
                "preserve_row_order": True,
                "expected_schedule_sha256": schedule_sha256,
            }
            adapter_text = json.dumps(adapter_config, sort_keys=True)
            mlx_text = json.dumps(mlx_config, sort_keys=True)
            for directory in (source, checkpoint):
                (directory / "adapter_config.json").write_text(
                    adapter_text,
                    encoding="utf-8",
                )
                (directory / "mlx_lora_config.json").write_text(
                    mlx_text,
                    encoding="utf-8",
                )
            snapshot = b"immutable-step-1500"
            (source / "0001500_adapters.safetensors").write_bytes(snapshot)
            (source / "native_mlx_data_report.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (checkpoint / "adapters.safetensors").write_bytes(snapshot)

            provenance = _checkpoint_provenance(
                checkpoint,
                model_id="model",
                schedule_sha256=schedule_sha256,
                n_rows=3000,
                checkpoint_step=1500,
            )

            self.assertEqual(
                provenance["training_contract"]["iters"],
                3000,
            )
            self.assertEqual(
                provenance["snapshot"]["checkpoint_step"],
                1500,
            )

            (checkpoint / "adapters.safetensors").write_bytes(
                b"not-the-source-snapshot"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _checkpoint_provenance(
                    checkpoint,
                    model_id="model",
                    schedule_sha256=schedule_sha256,
                    n_rows=3000,
                    checkpoint_step=1500,
                )

    def test_training_report_must_match_checkpoint_source_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "native_mlx_data_report.json"
            report.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "does not match the checkpoint's source",
            ):
                _training_report_provenance(
                    report,
                    dataset_sha256="a" * 64,
                    schedule={},
                    expected_source_sha256="b" * 64,
                )

    def test_atomic_writer_refuses_overwrite_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "nested" / "report.json"
            _write_text_atomically(output, '{"complete":true}\n')

            self.assertEqual(output.read_text(), '{"complete":true}\n')
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _write_text_atomically(output, "changed\n")
            self.assertEqual(output.read_text(), '{"complete":true}\n')

    def test_atomic_writer_loses_name_race_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "report.json"

            with patch(
                "scripts.decompose_sft_loss.os.link",
                side_effect=FileExistsError,
            ):
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    _write_text_atomically(output, "complete\n")

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
