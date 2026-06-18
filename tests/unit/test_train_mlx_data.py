"""Unit tests for pure MLX dataset conversion."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from sts_ai.train.train_mlx import prepare_mlx_data


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


class PrepareMlxDataTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
