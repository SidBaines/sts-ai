"""Unit tests for pure MLX dataset conversion."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from sts_ai.train.train_mlx import (
    _MaskedPreTokenizedDataset,
    _iterate_masked_batches,
    _masked_ce_loss,
    _require_model_snapshot_unchanged,
    _require_dataset_sha256,
    _resolve_cached_model_snapshot,
    _run_mlx_lora_action_masked,
    _validate_search_teacher_training_contract,
    prepare_mlx_data,
    prepare_native_mlx_data,
    train,
)


MODEL_REVISION = "eec12d0899edea9b738ab1009af9159cdfd70d71"


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _messages(index: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"prompt {index}"},
        {"role": "assistant", "content": f"completion {index}"},
    ]


def _write_dataset(
    path: Path,
    n_examples: int,
    *,
    include_messages: bool = True,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(n_examples):
            record: dict[str, Any] = {
                "prompt": f"prompt {index}",
                "completion": f"completion {index}",
                "world_seed": 200 + index,
                "stem": f"seed_{200 + index}_r0",
            }
            if include_messages:
                record["messages"] = _messages(index)
            handle.write(
                json.dumps(record)
                + "\n",
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_teacher_dataset(
    root: Path,
    *,
    base_model: str = "model",
    version: int = 3,
    dataset_sha256: str | None = None,
) -> tuple[Path, Path]:
    dataset = root / "teacher.jsonl"
    row = {
        "prompt": (
            "PROMPT:\nValid action_index values are: 0.\n"
            "Use only the compact action JSON."
        ),
        "completion": '{"action_index":0}',
        "target_action_index": 0,
        "teacher_action_index": 0,
        "window_id": "window_0",
        "assistant_turn_terminator": "<turn|>\n",
        "loss_mask_mode": "action",
        "output_contract": "action_only",
        "observation_version": "combat_public_v2",
        "teacher_selection_rule": "aggregated_root_visits",
        "teacher_privilege": "simulator_full_state",
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = root / "teacher.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "search_teacher_sft",
                "version": version,
                "observation_version": "combat_public_v2",
                "teacher_selection_rule": "aggregated_root_visits",
                "teacher_privilege": "simulator_full_state",
                "loss_mask_mode": "action",
                "output_contract": "action_only",
                "enable_thinking": False,
                "tokenizer_id": base_model,
                "n_examples": 1,
                "dataset_sha256": dataset_sha256 or digest,
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest


def _write_model_snapshot(root: Path) -> Path:
    snapshot = root / MODEL_REVISION
    snapshot.mkdir()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / name).write_text(f"{name}\n", encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return snapshot


def _fake_model_provenance(snapshot: Path) -> dict[str, Any]:
    return {
        "model_id": "model",
        "revision": MODEL_REVISION,
        "snapshot_path": str(snapshot.resolve()),
        "files": {"fixture": {"sha256": "b" * 64, "size_bytes": 1}},
        "identity_sha256": "a" * 64,
    }


class PrepareMlxDataTest(unittest.TestCase):
    def test_zero_valid_fraction_keeps_every_example_in_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "mlx"
            _write_dataset(dataset, 5)
            prepare_mlx_data(dataset, out_dir, valid_fraction=0.0)
            self.assertEqual(len(_read_jsonl(out_dir / "train.jsonl")), 5)
            self.assertEqual(_read_jsonl(out_dir / "valid.jsonl"), [])

    def test_drops_extra_keys_creates_files_and_splits_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_a = root / "mlx_a"
            out_b = root / "mlx_b"
            _write_dataset(dataset, 5)

            result = prepare_mlx_data(
                dataset,
                out_a,
                valid_fraction=0.4,
                shuffle_seed=123,
            )
            prepare_mlx_data(
                dataset,
                out_b,
                valid_fraction=0.4,
                shuffle_seed=123,
            )

            self.assertEqual(result, out_a)
            self.assertTrue((out_a / "train.jsonl").exists())
            self.assertTrue((out_a / "valid.jsonl").exists())
            self.assertEqual(
                (out_a / "train.jsonl").read_text(encoding="utf-8"),
                (out_b / "train.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "valid.jsonl").read_text(encoding="utf-8"),
                (out_b / "valid.jsonl").read_text(encoding="utf-8"),
            )

            train_records = _read_jsonl(out_a / "train.jsonl")
            valid_records = _read_jsonl(out_a / "valid.jsonl")
            self.assertEqual(len(train_records), 3)
            self.assertEqual(len(valid_records), 2)
            self.assertTrue(valid_records)
            for record in train_records + valid_records:
                self.assertEqual(list(record.keys()), ["messages"])
                self.assertEqual(record, {"messages": record["messages"]})
                self.assertEqual(
                    [message["role"] for message in record["messages"]],
                    ["user", "assistant"],
                )
                self.assertNotIn("prompt", record)
                self.assertNotIn("completion", record)

    def test_single_example_stays_in_train_with_empty_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "mlx"
            _write_dataset(dataset, 1)

            prepare_mlx_data(dataset, out_dir)

            self.assertEqual(
                _read_jsonl(out_dir / "train.jsonl"),
                [{"messages": _messages(0)}],
            )
            self.assertEqual(_read_jsonl(out_dir / "valid.jsonl"), [])

    def test_missing_messages_raises_instead_of_falling_back_to_completion_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "mlx"
            _write_dataset(dataset, 1, include_messages=False)

            with self.assertRaisesRegex(ValueError, "messages"):
                prepare_mlx_data(dataset, out_dir)

    def test_gemma_native_thought_channel_raises_instead_of_stripping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "mlx"
            record = {
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {
                        "role": "assistant",
                        "content": "<|channel>thought\nthink\n<channel|>{\"action_index\": 0}",
                    },
                ],
                "completion": "<|channel>thought\nthink\n<channel|>{\"action_index\": 0}",
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strip Gemma native thought"):
                prepare_mlx_data(dataset, out_dir)

    def test_native_mlx_data_preserves_thought_and_drops_truncated_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "native"
            prompt = "PROMPT:"
            good_completion = "<|channel>thought\nok\n<channel|>{\"action_index\": 0}"
            too_long_thinking = (
                "<|channel>thought\n"
                + ("x" * 80)
                + "\n<channel|>{\"action_index\": 1}"
            )
            records = [
                {
                    "prompt": prompt,
                    "completion": good_completion,
                    "messages": _messages(0),
                },
                {
                    "prompt": prompt,
                    "completion": too_long_thinking,
                    "messages": _messages(1),
                },
                {
                    "prompt": prompt,
                    "completion": "<|channel>thought\nunfinished",
                    "messages": _messages(2),
                },
            ]
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = prepare_native_mlx_data(
                dataset,
                out_dir,
                tokenizer=FakeTokenizer(),
                max_seq_length=len(prompt)
                + len(good_completion)
                + len("<turn|>\n"),
                valid_fraction=0.0,
            )

            train_records = _read_jsonl(out_dir / "train.jsonl")
            self.assertEqual(report["n_input_records"], 3)
            self.assertEqual(report["n_kept_records"], 1)
            self.assertEqual(report["skipped_record_counts"]["would_truncate_thinking"], 1)
            self.assertEqual(report["skipped_record_counts"]["source_thinking_truncated"], 1)
            self.assertEqual(len(train_records), 1)
            kept = train_records[0]
            self.assertEqual(kept["offset"], len(prompt))
            self.assertEqual(
                FakeTokenizer().decode(kept["input_ids"][kept["offset"] :]),
                good_completion + "<turn|>\n",
            )

    def test_action_mlx_data_writes_non_contiguous_mask_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "action"
            completion = (
                "<|channel>thought\nprivate\n<channel|>"
                '{"action_index": 2}'
            )
            record = {
                "prompt": "PROMPT:",
                "completion": completion,
                "target_action_index": 2,
                "assistant_turn_terminator": "<turn|>\n",
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = prepare_native_mlx_data(
                dataset,
                out_dir,
                tokenizer=FakeTokenizer(),
                max_seq_length=256,
                valid_fraction=0.0,
                loss_mask_mode="action",
            )
            kept = _read_jsonl(out_dir / "train.jsonl")[0]
            prompt_len = kept["n_prompt_tokens"]
            completion_text = FakeTokenizer().decode(kept["input_ids"][prompt_len:])
            completion_mask = kept["loss_mask"][prompt_len:]
            supervised = "".join(
                char if is_supervised else "·"
                for char, is_supervised in zip(completion_text, completion_mask)
            )
            self.assertEqual(report["loss_mask_mode"], "action")
            self.assertEqual(report["format"], "pretokenized_action_mask")
            self.assertFalse(report["action_token_only"])
            self.assertEqual(report["token_count_totals"]["n_action_tokens"], 1)
            self.assertGreater(
                report["token_count_totals"]["n_supervised_format_tokens"],
                0,
            )
            self.assertEqual(
                report["token_count_totals"]["n_supervised_tokens"],
                sum(kept["loss_mask"]),
            )
            self.assertNotIn("loss_weights", kept)
            self.assertEqual(
                report["loss_weight_representation"],
                "boolean_mask",
            )
            self.assertEqual(
                report["supervised_weight_mass_totals"],
                {
                    "supervised_format_weight_mass": float(
                        kept["n_supervised_format_tokens"]
                    ),
                    "supervised_action_weight_mass": float(
                        kept["n_supervised_action_tokens"]
                    ),
                    "supervised_total_weight_mass": float(
                        kept["n_supervised_tokens"]
                    ),
                },
            )
            self.assertNotIn("private", supervised)
            self.assertIn('"action_index": 2', supervised)
            self.assertTrue(supervised.endswith("<turn|>\n"))
            self.assertEqual(kept["n_action_tokens"], 1)
            self.assertGreater(kept["n_format_tokens"], 0)

    def test_action_token_only_masks_format_but_retains_full_causal_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "action_value"
            completion = (
                "<|channel>thought\nprivate\n<channel|>"
                '{"action_index": 2}'
            )
            record = {
                "prompt": "PROMPT:",
                "completion": completion,
                "target_action_index": 2,
                "assistant_turn_terminator": "<turn|>\n",
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = prepare_native_mlx_data(
                dataset,
                out_dir,
                tokenizer=FakeTokenizer(),
                max_seq_length=256,
                valid_fraction=0.0,
                loss_mask_mode="action",
                action_token_only=True,
            )
            kept = _read_jsonl(out_dir / "train.jsonl")[0]
            prompt_len = kept["n_prompt_tokens"]
            completion_text = FakeTokenizer().decode(kept["input_ids"][prompt_len:])
            completion_mask = kept["loss_mask"][prompt_len:]
            supervised = "".join(
                char
                for char, is_supervised in zip(completion_text, completion_mask)
                if is_supervised
            )

            self.assertEqual(
                completion_text,
                completion + "<turn|>\n",
            )
            self.assertEqual(supervised, "2")
            self.assertEqual(report["format"], "pretokenized_action_value_mask")
            self.assertTrue(report["action_token_only"])
            self.assertEqual(
                report["token_count_totals"]["n_supervised_format_tokens"],
                0,
            )
            self.assertEqual(
                report["token_count_totals"]["n_supervised_action_tokens"],
                1,
            )
            self.assertEqual(report["token_count_totals"]["n_supervised_tokens"], 1)

    def test_semantic_contracts_round_trip_through_both_mlx_data_paths(self):
        cases = (
            (
                "action_text",
                '{"action":"play Strike -> Nob"}',
                "play Strike -> Nob",
            ),
            (
                "turn_plan",
                '{"plan":["play Bash -> Nob","end turn"],'
                '"action":"play Bash -> Nob"}',
                "play Bash -> Nob",
            ),
        )
        for output_contract, completion, description in cases:
            with self.subTest(output_contract=output_contract):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    dataset = root / "dataset.jsonl"
                    record = {
                        "prompt": "PROMPT:",
                        "completion": completion,
                        "messages": [
                            {"role": "user", "content": "PROMPT:"},
                            {"role": "assistant", "content": completion},
                        ],
                        "output_contract": output_contract,
                        "target_action_index": 0,
                        "target_action_description": description,
                        "assistant_turn_terminator": "<turn|>\n",
                    }
                    dataset.write_text(
                        json.dumps(record) + "\n",
                        encoding="utf-8",
                    )

                    prepare_mlx_data(
                        dataset,
                        root / "chat",
                        valid_fraction=0.0,
                    )
                    chat_record = _read_jsonl(root / "chat" / "train.jsonl")[0]
                    self.assertEqual(chat_record["messages"], record["messages"])

                    report = prepare_native_mlx_data(
                        dataset,
                        root / "native",
                        tokenizer=FakeTokenizer(),
                        max_seq_length=512,
                        valid_fraction=0.0,
                        loss_mask_mode="action",
                    )
                    native_record = _read_jsonl(
                        root / "native" / "train.jsonl"
                    )[0]
                    self.assertEqual(report["n_kept_records"], 1)
                    self.assertGreater(native_record["n_action_tokens"], 0)
                    self.assertGreater(native_record["n_format_tokens"], 0)

    def test_strict_teacher_preflight_accepts_semantic_contracts(self):
        cases = (
            ("action_text", '{"action":"hit"}'),
            (
                "turn_plan",
                '{"plan":["hit","end turn"],"action":"hit"}',
            ),
        )
        for output_contract, completion in cases:
            with self.subTest(output_contract=output_contract):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    dataset = root / "teacher.jsonl"
                    row = {
                        "prompt": (
                            "Choose from the LEGAL ACTIONS list below.\n\n"
                            "GAME STATE\nstate\n\n"
                            "LEGAL ACTIONS\n0: hit\n1: end turn"
                            "<turn|>\n<|turn>model\n"
                        ),
                        "completion": completion,
                        "target_action_index": 0,
                        "target_action_description": "hit",
                        "teacher_action_index": 0,
                        "assistant_turn_terminator": "<turn|>\n",
                        "window_id": "window_0",
                        "loss_mask_mode": "action",
                        "output_contract": output_contract,
                        "observation_version": "combat_public_v2",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "teacher_privilege": "simulator_full_state",
                    }
                    dataset.write_text(
                        json.dumps(row) + "\n",
                        encoding="utf-8",
                    )
                    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
                    manifest = {
                        "kind": "search_teacher_sft",
                        "version": 3,
                        "observation_version": "combat_public_v2",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "teacher_privilege": "simulator_full_state",
                        "loss_mask_mode": "action",
                        "output_contract": output_contract,
                        "enable_thinking": False,
                        "tokenizer_id": "model",
                        "n_examples": 1,
                        "dataset_sha256": digest,
                    }

                    self.assertEqual(
                        _validate_search_teacher_training_contract(
                            dataset_path=dataset,
                            manifest=manifest,
                            base_model="model",
                            expected_example_count=1,
                        ),
                        digest,
                    )

    def test_action_token_weight_preserves_scaffold_and_audits_weight_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            out_dir = root / "weighted"
            record = {
                "prompt": "PROMPT:",
                "completion": '{"action_index": 2}',
                "target_action_index": 2,
                "assistant_turn_terminator": "<turn|>\n",
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = prepare_native_mlx_data(
                dataset,
                out_dir,
                tokenizer=FakeTokenizer(),
                valid_fraction=0.0,
                loss_mask_mode="action",
                action_token_weight=8.0,
            )
            kept = _read_jsonl(out_dir / "train.jsonl")[0]
            format_count = kept["n_supervised_format_tokens"]
            action_count = kept["n_supervised_action_tokens"]

            self.assertEqual(report["action_token_weight"], 8.0)
            self.assertEqual(report["format_token_weight"], 1.0)
            self.assertEqual(
                report["loss_normalization"],
                "sum_of_token_weights",
            )
            self.assertEqual(
                report["loss_weight_representation"],
                "numeric_weights",
            )
            self.assertEqual(sum(kept["loss_mask"]), format_count + action_count)
            self.assertEqual(
                kept["supervised_format_weight_mass"],
                float(format_count),
            )
            self.assertEqual(
                kept["supervised_action_weight_mass"],
                float(8 * action_count),
            )
            self.assertEqual(
                kept["supervised_total_weight_mass"],
                float(format_count + 8 * action_count),
            )
            masses = report["supervised_weight_mass_totals"]
            self.assertEqual(
                masses["supervised_format_weight_mass"],
                float(format_count),
            )
            self.assertEqual(
                masses["supervised_action_weight_mass"],
                float(8 * action_count),
            )
            self.assertEqual(kept["loss_weights"].count(8.0), action_count)
            self.assertEqual(kept["loss_weights"].count(1.0), format_count)
            self.assertEqual(
                kept["loss_weights"].count(0.0),
                len(kept["loss_weights"]) - format_count - action_count,
            )

    def test_weighted_ce_normalizes_by_total_weight_mass(self):
        import numpy as np

        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.float32 = np.float32
        fake_nn = types.ModuleType("mlx.nn")
        fake_nn.losses = types.SimpleNamespace(
            cross_entropy=lambda logits, targets: logits
        )
        fake_mlx.core = fake_core
        fake_mlx.nn = fake_nn
        batch = np.array([[1, 2, 3, 4]], dtype=np.int32)
        weights = np.array([[0.0, 1.0, 8.0, 0.0]], dtype=np.float32)

        with patch.dict(
            sys.modules,
            {"mlx": fake_mlx, "mlx.core": fake_core, "mlx.nn": fake_nn},
        ):
            loss, total_weight = _masked_ce_loss(
                lambda inputs: np.array([[2.0, 4.0, 10.0]], dtype=np.float32),
                batch,
                weights,
            )

        self.assertEqual(float(total_weight), 9.0)
        self.assertAlmostEqual(float(loss), (2.0 + 8.0 * 4.0) / 9.0, places=6)

    def test_expected_retained_count_failure_persists_native_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            data_dir = root / "prepared"
            persistent_report = root / "adapter" / "native_mlx_data_report.json"
            record = {
                "prompt": "PROMPT:",
                "completion": '{"action_index":0}',
                "target_action_index": 0,
                "assistant_turn_terminator": "<turn|>\n",
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected=2 actual=1"):
                prepare_native_mlx_data(
                    dataset,
                    data_dir,
                    tokenizer=FakeTokenizer(),
                    valid_fraction=0.0,
                    loss_mask_mode="action",
                    report_path=persistent_report,
                    expected_kept_records=2,
                )

            report = json.loads(persistent_report.read_text(encoding="utf-8"))
            self.assertEqual(report["n_kept_records"], 1)
            self.assertEqual(report["expected_kept_records"], 2)
            self.assertFalse(report["retained_count_matches_expected"])

    def test_zero_retained_failure_persists_native_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            persistent_report = root / "adapter" / "native_mlx_data_report.json"
            dataset.write_text(
                json.dumps({"prompt": "PROMPT:", "completion": ""}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "kept zero examples"):
                prepare_native_mlx_data(
                    dataset,
                    root / "prepared",
                    tokenizer=FakeTokenizer(),
                    valid_fraction=0.0,
                    loss_mask_mode="action",
                    report_path=persistent_report,
                    expected_kept_records=1,
                )

            report = json.loads(persistent_report.read_text(encoding="utf-8"))
            self.assertEqual(report["n_kept_records"], 0)
            self.assertEqual(report["max_total_tokens"], 0)
            self.assertEqual(report["skipped_record_counts"], {"missing_completion": 1})

    def test_tokenization_error_persists_native_report_before_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            persistent_report = root / "adapter" / "native_mlx_data_report.json"
            dataset.write_text(
                json.dumps(
                    {
                        "prompt": "PROMPT:",
                        "completion": "not action JSON",
                        "target_action_index": 0,
                        "assistant_turn_terminator": "<turn|>\n",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "tokenization errors"):
                prepare_native_mlx_data(
                    dataset,
                    root / "prepared",
                    tokenizer=FakeTokenizer(),
                    valid_fraction=0.0,
                    loss_mask_mode="action",
                    report_path=persistent_report,
                    expected_kept_records=1,
                )

            report = json.loads(persistent_report.read_text(encoding="utf-8"))
            self.assertEqual(report["n_kept_records"], 0)
            self.assertEqual(report["tokenization_error_counts"], {"ValueError": 1})

    def test_masked_batch_shuffle_uses_local_seeded_rng(self):
        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.array = lambda value: value
        fake_mlx.core = fake_core
        dataset = [
            ([index, index + 10], [True, True])
            for index in range(6)
        ]

        def order_after_global_noise(global_seed: int) -> list[int]:
            import numpy as np

            np.random.seed(global_seed)
            np.random.permutation(100)
            with patch.dict(
                sys.modules,
                {"mlx": fake_mlx, "mlx.core": fake_core},
            ):
                iterator = _iterate_masked_batches(
                    dataset,
                    batch_size=1,
                    max_seq_length=32,
                    loop=True,
                    seed=0,
                )
                batches = list(itertools.islice(iterator, len(dataset)))
            return [int(tokens[0, 0]) for tokens, _ in batches]

        self.assertEqual(order_after_global_noise(1), order_after_global_noise(999))

    def test_masked_batches_keep_unit_masks_boolean_and_weighted_masks_float(self):
        import numpy as np

        unit_dataset = _MaskedPreTokenizedDataset(
            [{"input_ids": [1, 2, 3], "loss_mask": [False, True, True]}]
        )
        weighted_dataset = _MaskedPreTokenizedDataset(
            [
                {
                    "input_ids": [1, 2, 3],
                    "loss_mask": [False, True, True],
                    "loss_weights": [0.0, 1.0, 8.0],
                }
            ]
        )
        unit_row = unit_dataset.process(unit_dataset[0])
        weighted_row = weighted_dataset.process(weighted_dataset[0])
        self.assertTrue(all(isinstance(value, bool) for value in unit_row[1]))
        self.assertTrue(
            all(isinstance(value, float) for value in weighted_row[1])
        )

        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.array = lambda value: value
        fake_mlx.core = fake_core
        with patch.dict(
            sys.modules,
            {"mlx": fake_mlx, "mlx.core": fake_core},
        ):
            _, unit_mask = next(
                _iterate_masked_batches(
                    [unit_row],
                    batch_size=1,
                    max_seq_length=32,
                )
            )
            _, weighted_mask = next(
                _iterate_masked_batches(
                    [weighted_row],
                    batch_size=1,
                    max_seq_length=32,
                )
            )

        self.assertEqual(unit_mask.dtype, np.dtype(np.bool_))
        self.assertEqual(weighted_mask.dtype, np.dtype(np.float32))
        self.assertEqual(unit_mask.tolist()[0][:3], [False, True, True])
        self.assertEqual(weighted_mask.tolist()[0][:3], [0.0, 1.0, 8.0])
        with self.assertRaisesRegex(
            ValueError,
            "cannot mix boolean masks and numeric weights",
        ):
            with patch.dict(
                sys.modules,
                {"mlx": fake_mlx, "mlx.core": fake_core},
            ):
                next(
                    _iterate_masked_batches(
                        [unit_row, weighted_row],
                        batch_size=2,
                        max_seq_length=32,
                    )
                )

    def test_masked_batch_preserve_row_order_skips_length_sort_and_shuffle(self):
        import numpy as np

        del np  # Keep NumPy loaded outside patch.dict's module-table snapshot.
        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.array = lambda value: value
        fake_mlx.core = fake_core
        dataset = [
            ([3, 30, 31], [True, True, True]),
            ([1], [True]),
            ([2, 20], [True, True]),
        ]
        with patch.dict(
            sys.modules,
            {"mlx": fake_mlx, "mlx.core": fake_core},
        ):
            iterator = _iterate_masked_batches(
                dataset,
                batch_size=1,
                max_seq_length=32,
                loop=True,
                seed=999,
                preserve_row_order=True,
            )
            batches = list(itertools.islice(iterator, 6))
        self.assertEqual(
            [int(tokens[0, 0]) for tokens, _ in batches],
            [3, 1, 2, 3, 1, 2],
        )
        with self.assertRaisesRegex(ValueError, "requires batch_size=1"):
            with patch.dict(
                sys.modules,
                {"mlx": fake_mlx, "mlx.core": fake_core},
            ):
                next(
                    _iterate_masked_batches(
                        dataset,
                        batch_size=2,
                        max_seq_length=32,
                        preserve_row_order=True,
                    )
                )

    def test_native_preparation_preserves_schedule_identity_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "scheduled.jsonl"
            records = [
                {
                    "prompt": f"PROMPT:{index}",
                    "completion": '{"action_index":0}',
                    "target_action_index": 0,
                    "assistant_turn_terminator": "<turn|>\n",
                    "schedule_step": index,
                    "source_identity": f"source-{index % 2}",
                }
                for index in range(4)
            ]
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            report = prepare_native_mlx_data(
                dataset,
                root / "prepared",
                tokenizer=FakeTokenizer(),
                valid_fraction=0.0,
                loss_mask_mode="action",
                preserve_row_order=True,
            )

            prepared = _read_jsonl(root / "prepared" / "train.jsonl")
            self.assertEqual(
                [record["schedule_step"] for record in prepared],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [record["source_identity"] for record in prepared],
                ["source-0", "source-1", "source-0", "source-1"],
            )
            self.assertTrue(
                report["schedule"][
                    "steps_are_zero_based_contiguous_in_prepared_order"
                ]
            )
            self.assertEqual(report["schedule"]["n_unique_source_identities"], 2)

    def test_train_forwards_zero_valid_fraction_to_action_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            fake_mlx_lm = types.ModuleType("mlx_lm")
            with (
                patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
                patch(
                    "sts_ai.train.train_mlx._run_mlx_lora_action_masked"
                ) as runner,
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    loss_mask_mode="action",
                    valid_fraction=0.0,
                )

        self.assertEqual(runner.call_args.kwargs["valid_fraction"], 0.0)

    def test_train_forwards_zero_valid_fraction_to_native_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            fake_mlx_lm = types.ModuleType("mlx_lm")
            with (
                patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
                patch(
                    "sts_ai.train.train_mlx._has_gemma_thought_completion",
                    return_value=True,
                ),
                patch("sts_ai.train.train_mlx._run_mlx_lora_native") as runner,
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    valid_fraction=0.0,
                )

        self.assertEqual(runner.call_args.kwargs["valid_fraction"], 0.0)

    def test_strict_teacher_manifest_is_validated_before_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            out = root / "adapter"
            snapshot = root / MODEL_REVISION
            provenance = _fake_model_provenance(snapshot)
            fake_mlx_lm = types.ModuleType("mlx_lm")
            with (
                patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
                patch(
                    "sts_ai.train.train_mlx._resolve_cached_model_snapshot",
                    return_value=(snapshot, provenance),
                ) as resolver,
                patch(
                    "sts_ai.train.train_mlx._run_mlx_lora_action_masked"
                ) as runner,
            ):
                result = train(
                    dataset,
                    "model",
                    out,
                    manifest_path=manifest,
                    loss_mask_mode="action",
                    expected_example_count=1,
                    seed=7,
                    lora_rank=4,
                    lora_scale=9.0,
                    lora_dropout=0.1,
                    grad_accumulation_steps=2,
                    action_token_only=True,
                    model_revision=MODEL_REVISION,
                )

            self.assertEqual(result, out)
            self.assertTrue(out.is_dir())
            resolver.assert_called_once_with("model", revision=MODEL_REVISION)
            self.assertEqual(
                runner.call_args.kwargs["model_load_path"],
                str(snapshot),
            )
            self.assertEqual(
                runner.call_args.kwargs["base_model_provenance"],
                provenance,
            )
            self.assertEqual(runner.call_args.kwargs["expected_example_count"], 1)
            self.assertEqual(
                runner.call_args.kwargs["preparation_report_path"],
                out / "native_mlx_data_report.json",
            )
            self.assertEqual(
                len(runner.call_args.kwargs["expected_dataset_sha256"]),
                64,
            )
            self.assertEqual(runner.call_args.kwargs["seed"], 7)
            self.assertEqual(runner.call_args.kwargs["lora_rank"], 4)
            self.assertEqual(runner.call_args.kwargs["lora_scale"], 9.0)
            self.assertEqual(runner.call_args.kwargs["lora_dropout"], 0.1)
            self.assertEqual(runner.call_args.kwargs["grad_accumulation_steps"], 2)
            self.assertTrue(runner.call_args.kwargs["action_token_only"])
            self.assertEqual(runner.call_args.kwargs["action_token_weight"], 1.0)
            self.assertEqual(
                json.loads((out / "mlx_lora_config.json").read_text(encoding="utf-8")),
                {
                    "seed": 7,
                    "grad_accumulation_steps": 2,
                    "action_token_only": True,
                    "action_token_weight": 1.0,
                    "format_token_weight": 1.0,
                    "loss_normalization": "sum_of_token_weights",
                    "loss_weight_representation": "boolean_mask",
                    "preserve_row_order": False,
                    "expected_schedule_sha256": None,
                    "base_model_revision": MODEL_REVISION,
                    "base_model_identity_sha256": "a" * 64,
                    "base_model_provenance": provenance,
                    "lora_parameters": {
                        "rank": 4,
                        "scale": 9.0,
                        "dropout": 0.1,
                    },
                },
            )

    def test_strict_teacher_preserve_row_order_is_persisted_and_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            row = json.loads(dataset.read_text(encoding="utf-8"))
            row["schedule_step"] = 0
            row["source_identity"] = "a" * 64
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ordered_schedule = [
                {
                    "schedule_step": 0,
                    "source_identity": "a" * 64,
                }
            ]
            schedule_sha = hashlib.sha256(
                json.dumps(
                    ordered_schedule,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_value["dataset_sha256"] = hashlib.sha256(
                dataset.read_bytes()
            ).hexdigest()
            manifest_value["augmentation"] = {
                "kind": "paired_cyclic_action_order_sft",
                "version": 1,
                "n_output_rows": 1,
                "source_schedule_sha256": "b" * 64,
                "ordered_step_source_sha256": schedule_sha,
            }
            manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
            out = root / "adapter"
            snapshot = root / MODEL_REVISION
            provenance = _fake_model_provenance(snapshot)
            fake_mlx_lm = types.ModuleType("mlx_lm")
            with (
                patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
                patch(
                    "sts_ai.train.train_mlx._resolve_cached_model_snapshot",
                    return_value=(snapshot, provenance),
                ),
                patch(
                    "sts_ai.train.train_mlx._run_mlx_lora_action_masked"
                ) as runner,
            ):
                train(
                    dataset,
                    "model",
                    out,
                    manifest_path=manifest,
                    loss_mask_mode="action",
                    batch_size=1,
                    iters=1,
                    valid_fraction=0.0,
                    preserve_row_order=True,
                    model_revision=MODEL_REVISION,
                )

            self.assertTrue(runner.call_args.kwargs["preserve_row_order"])
            config = json.loads(
                (out / "mlx_lora_config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(config["preserve_row_order"])
            self.assertEqual(config["expected_schedule_sha256"], schedule_sha)
            invalid_cases = (
                (
                    {"valid_fraction": 0.1, "iters": 1},
                    "valid_fraction=0",
                ),
                (
                    {
                        "valid_fraction": 0.0,
                        "iters": 1,
                        "grad_accumulation_steps": 2,
                    },
                    "grad_accumulation_steps=1",
                ),
                (
                    {"valid_fraction": 0.0, "iters": 2},
                    "cannot exceed the explicit schedule",
                ),
            )
            for case_index, (overrides, message) in enumerate(invalid_cases):
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(ValueError, message):
                        train(
                            dataset,
                            "model",
                            root / f"invalid_adapter_{case_index}",
                            manifest_path=manifest,
                            loss_mask_mode="action",
                            batch_size=1,
                            preserve_row_order=True,
                            **overrides,
                        )

    def test_strict_teacher_action_weight_is_persisted_and_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            out = root / "adapter"
            snapshot = root / MODEL_REVISION
            provenance = _fake_model_provenance(snapshot)
            fake_mlx_lm = types.ModuleType("mlx_lm")
            with (
                patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
                patch(
                    "sts_ai.train.train_mlx._resolve_cached_model_snapshot",
                    return_value=(snapshot, provenance),
                ),
                patch(
                    "sts_ai.train.train_mlx._run_mlx_lora_action_masked"
                ) as runner,
            ):
                train(
                    dataset,
                    "model",
                    out,
                    manifest_path=manifest,
                    loss_mask_mode="action",
                    action_token_weight=8.0,
                    model_revision=MODEL_REVISION,
                )

            self.assertEqual(runner.call_args.kwargs["action_token_weight"], 8.0)
            config = json.loads(
                (out / "mlx_lora_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["action_token_weight"], 8.0)
            self.assertEqual(config["format_token_weight"], 1.0)
            self.assertEqual(
                config["loss_normalization"],
                "sum_of_token_weights",
            )
            self.assertEqual(
                config["loss_weight_representation"],
                "numeric_weights",
            )

    def test_preserve_row_order_requires_strict_teacher_and_batch_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "legacy.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "requires a strict search_teacher_sft",
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    preserve_row_order=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            with self.assertRaisesRegex(
                ValueError,
                "paired augmentation manifest metadata",
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                    batch_size=1,
                    iters=1,
                    valid_fraction=0.0,
                    preserve_row_order=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            with self.assertRaisesRegex(ValueError, "requires batch_size=1"):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                    batch_size=2,
                    preserve_row_order=True,
                )

    def test_strict_teacher_manifest_rejects_hash_tokenizer_and_old_version(self):
        cases = (
            ({"dataset_sha256": "0" * 64}, "dataset_sha256 disagrees"),
            ({"base_model": "different/model"}, "tokenizer_id"),
            ({"version": 2}, "version"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    dataset, manifest = _write_teacher_dataset(
                        root,
                        version=int(overrides.get("version", 3)),
                        dataset_sha256=overrides.get("dataset_sha256"),
                    )
                    base_model = str(overrides.get("base_model", "model"))
                    with self.assertRaisesRegex(ValueError, message):
                        train(
                            dataset,
                            base_model,
                            root / "adapter",
                            manifest_path=manifest,
                            loss_mask_mode="action",
                        )
                    self.assertFalse((root / "adapter").exists())

    def test_strict_teacher_manifest_rejects_stale_interface_and_objective(self):
        cases = (
            ("observation_version", "combat_public_v1"),
            ("teacher_selection_rule", "native_selected_action"),
            ("loss_mask_mode", "completion"),
            ("output_contract", "reasoning_action"),
        )
        for field, value in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    dataset, manifest = _write_teacher_dataset(root)
                    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
                    manifest_value[field] = value
                    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        train(
                            dataset,
                            "model",
                            root / "adapter",
                            manifest_path=manifest,
                        )

    def test_strict_teacher_manifest_rejects_row_level_contract_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            row = json.loads(dataset.read_text(encoding="utf-8"))
            row["observation_version"] = "combat_public_v1"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_value["dataset_sha256"] = hashlib.sha256(
                dataset.read_bytes()
            ).hexdigest()
            manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row 0.*observation_version"):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                )

    def test_strict_teacher_manifest_rejects_noncanonical_or_wrong_action_target(self):
        cases = (
            ({"completion": 'private {"action_index":0}'}, "noncanonical"),
            ({"teacher_action_index": 1, "target_action_index": 0}, "mismatch"),
            ({"teacher_action_index": 1, "target_action_index": 1}, "out_of_range"),
        )
        for mutation, message in cases:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    dataset, manifest = _write_teacher_dataset(root)
                    row = json.loads(dataset.read_text(encoding="utf-8"))
                    row.update(mutation)
                    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
                    manifest_value["dataset_sha256"] = hashlib.sha256(
                        dataset.read_bytes()
                    ).hexdigest()
                    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, message):
                        train(
                            dataset,
                            "model",
                            root / "adapter",
                            manifest_path=manifest,
                        )

    def test_dataset_hash_recheck_detects_post_preparation_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.jsonl"
            dataset.write_text("before\n", encoding="utf-8")
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            dataset.write_text("after!\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after preflight"):
                _require_dataset_sha256(dataset, digest)

    def test_cached_model_revision_is_resolved_offline_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = _write_model_snapshot(Path(tmp))
            snapshot_download = Mock(return_value=str(snapshot))
            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.snapshot_download = snapshot_download

            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                resolved, provenance = _resolve_cached_model_snapshot(
                    "model",
                    revision=MODEL_REVISION,
                )

            self.assertEqual(resolved, snapshot.resolve())
            snapshot_download.assert_called_once_with(
                repo_id="model",
                revision=MODEL_REVISION,
                local_files_only=True,
            )
            self.assertEqual(provenance["model_id"], "model")
            self.assertEqual(provenance["revision"], MODEL_REVISION)
            self.assertEqual(provenance["snapshot_path"], str(snapshot.resolve()))
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
            _require_model_snapshot_unchanged(provenance)

            (snapshot / "tokenizer_config.json").write_text(
                "changed tokenizer config\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "base model snapshot changed after training preflight",
            ):
                _require_model_snapshot_unchanged(provenance)

    def test_action_masked_runner_loads_snapshot_but_records_logical_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / MODEL_REVISION
            provenance = _fake_model_provenance(snapshot)
            model = types.SimpleNamespace(
                layers=[object()],
                freeze=Mock(),
            )
            load = Mock(return_value=(model, FakeTokenizer()))
            saved_configs: list[tuple[dict[str, Any], Path]] = []

            mlx = types.ModuleType("mlx")
            mlx.__path__ = []  # type: ignore[attr-defined]
            mx = types.ModuleType("mlx.core")
            mx.random = types.SimpleNamespace(seed=Mock())
            optim = types.ModuleType("mlx.optimizers")
            optim.Adam = Mock(return_value="optimizer")
            mlx.core = mx
            mlx.optimizers = optim

            mlx_lm = types.ModuleType("mlx_lm")
            mlx_lm.__path__ = []  # type: ignore[attr-defined]
            lora = types.ModuleType("mlx_lm.lora")
            lora.CONFIG_DEFAULTS = {
                "steps_per_eval": 10,
                "steps_per_report": 10,
                "save_every": 100,
                "val_batches": 1,
                "grad_checkpoint": False,
            }
            lora.linear_to_lora_layers = Mock()
            lora.print_trainable_parameters = Mock()
            lora.save_config = (
                lambda config, path: saved_configs.append((config, Path(path)))
            )
            trainer = types.ModuleType("mlx_lm.tuner.trainer")
            trainer.CacheDataset = lambda dataset: dataset
            trainer.TrainingArgs = lambda **kwargs: types.SimpleNamespace(**kwargs)
            trainer.train = Mock()
            tuner = types.ModuleType("mlx_lm.tuner")
            tuner.__path__ = []  # type: ignore[attr-defined]
            utils = types.ModuleType("mlx_lm.utils")
            utils.load = load

            modules = {
                "mlx": mlx,
                "mlx.core": mx,
                "mlx.optimizers": optim,
                "mlx_lm": mlx_lm,
                "mlx_lm.lora": lora,
                "mlx_lm.tuner": tuner,
                "mlx_lm.tuner.trainer": trainer,
                "mlx_lm.utils": utils,
            }
            with (
                patch.dict(sys.modules, modules),
                patch(
                    "sts_ai.train.train_mlx.prepare_native_mlx_data"
                ),
                patch(
                    "sts_ai.train.train_mlx._require_dataset_sha256"
                ),
                patch(
                    "sts_ai.train.train_mlx._load_masked_pretokenized_dataset",
                    return_value=[],
                ),
                patch(
                    "sts_ai.train.train_mlx._require_model_snapshot_unchanged"
                ) as recheck,
            ):
                _run_mlx_lora_action_masked(
                    dataset_path=root / "teacher.jsonl",
                    data_dir=root / "data",
                    base_model="model",
                    model_load_path=str(snapshot),
                    base_model_provenance=provenance,
                    out_adapter_dir=root / "adapter",
                    num_layers=1,
                    iters=1,
                    batch_size=1,
                    learning_rate=1e-4,
                    max_seq_length=128,
                )

            load.assert_called_once_with(
                str(snapshot),
                tokenizer_config={"trust_remote_code": True},
            )
            self.assertEqual(recheck.call_count, 3)
            self.assertEqual(len(saved_configs), 1)
            adapter_config, config_path = saved_configs[0]
            self.assertEqual(config_path, root / "adapter" / "adapter_config.json")
            self.assertEqual(adapter_config["model"], "model")
            self.assertEqual(
                adapter_config["base_model_revision"],
                MODEL_REVISION,
            )
            self.assertEqual(
                adapter_config["base_model_identity_sha256"],
                provenance["identity_sha256"],
            )
            self.assertEqual(
                adapter_config["base_model_provenance"],
                provenance,
            )

    def test_strict_teacher_requires_revision_and_non_teacher_rejects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            with self.assertRaisesRegex(
                ValueError,
                "requires model_revision",
            ):
                train(
                    dataset,
                    "model",
                    root / "teacher_adapter",
                    manifest_path=manifest,
                )

            legacy = root / "legacy.jsonl"
            legacy.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "supported only for strict search_teacher_sft",
            ):
                train(
                    legacy,
                    "model",
                    root / "legacy_adapter",
                    model_revision=MODEL_REVISION,
                )

    def test_expected_count_requires_teacher_manifest_and_exact_manifest_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "legacy.jsonl"
            manifest = root / "legacy.manifest.json"
            dataset.write_text("{}\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({"loss_mask_mode": "action"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires a search_teacher_sft"):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                    expected_example_count=1,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            with self.assertRaisesRegex(ValueError, "expected=2 manifest=1"):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                    expected_example_count=2,
                )

    def test_action_token_only_requires_strict_teacher_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "legacy.jsonl"
            manifest = root / "legacy.manifest.json"
            dataset.write_text("{}\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({"loss_mask_mode": "action"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "action_token_only requires a strict search_teacher_sft",
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                    action_token_only=True,
                )
            self.assertFalse((root / "adapter").exists())

    def test_action_token_weight_requires_strict_teacher_mixed_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "legacy.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "action_token_weight requires a strict search_teacher_sft",
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    loss_mask_mode="action",
                    action_token_weight=8.0,
                )
            self.assertFalse((root / "adapter").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest = _write_teacher_dataset(root)
            with self.assertRaisesRegex(
                ValueError,
                "action_token_weight must be 1 with action_token_only",
            ):
                train(
                    dataset,
                    "model",
                    root / "adapter",
                    manifest_path=manifest,
                    loss_mask_mode="action",
                    action_token_only=True,
                    action_token_weight=8.0,
                )
            self.assertFalse((root / "adapter").exists())

    def test_train_requires_fresh_output_but_preserves_legacy_manifest_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "legacy.jsonl"
            manifest = root / "legacy.manifest.json"
            dataset.write_text("{}\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({"loss_mask_mode": "action"}),
                encoding="utf-8",
            )
            out = root / "adapter"
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh, nonexistent"):
                train(
                    dataset,
                    "model",
                    out,
                    manifest_path=manifest,
                )

            fresh_out = root / "fresh_adapter"
            fake_mlx_lm = types.ModuleType("mlx_lm")
            with (
                patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
                patch(
                    "sts_ai.train.train_mlx._run_mlx_lora_action_masked"
                ) as runner,
            ):
                train(
                    dataset,
                    "model",
                    fresh_out,
                    manifest_path=manifest,
                )
            runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
