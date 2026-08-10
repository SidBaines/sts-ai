"""Thin MLX LoRA trainer wrapper and data conversion helpers."""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import types
from collections import Counter
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from sts_ai.prompting import ACTION_ONLY_OUTPUT
from sts_ai.provenance import file_sha256
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import validate_teacher_row
from sts_ai.train.sft_format import resolve_loss_mask_mode, tokenize_example

__all__ = ["build_lora_cmd", "prepare_mlx_data", "prepare_native_mlx_data", "train"]


_GEMMA_THOUGHT_MARKERS = ("<|channel>thought", "<channel|>")
_GEMMA_THOUGHT_OPEN = "<|channel>thought"
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_GEMMA_ASSISTANT_TURN_END = "<turn|>\n"
_SEARCH_TEACHER_SFT_KIND = "search_teacher_sft"
_SEARCH_TEACHER_SFT_VERSION = 3
_PAIRED_AUGMENTATION_KIND = "paired_cyclic_action_order_sft"
_PAIRED_AUGMENTATION_VERSION = 1


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _mlx_record(record: dict[str, Any]) -> dict[str, Any]:
    if "messages" not in record:
        raise ValueError(
            "dataset record is missing required 'messages' field for MLX chat "
            "training"
        )
    completion = str(record.get("completion", ""))
    if any(marker in completion for marker in _GEMMA_THOUGHT_MARKERS):
        raise ValueError(
            "MLX chat training would strip Gemma native thought-channel "
            "completions via the model chat template. Use a native-thinking-safe "
            "training path instead of train_mlx.prepare_mlx_data."
        )
    return {"messages": record["messages"]}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_mlx_record(record), ensure_ascii=False) + "\n")


def _write_raw_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _split_records(
    records: list[dict[str, Any]],
    *,
    valid_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0.0, 1.0)")
    if valid_fraction == 0.0:
        return records, []
    if len(records) <= 1:
        return records, []

    n_valid = int(len(records) * valid_fraction)
    n_valid = max(1, min(n_valid, len(records) - 1))
    return records[n_valid:], records[:n_valid]


def prepare_mlx_data(
    dataset_path: Path,
    out_dir: Path,
    *,
    valid_fraction: float = 0.1,
    shuffle_seed: int = 0,
) -> Path:
    """Convert SFT JSONL to mlx-lm's chat JSONL layout.

    The input dataset can contain provenance keys such as ``world_seed`` and
    ``stem``; only ``messages`` is written so mlx-lm selects its ChatDataset.
    For a one-example dataset, the single example stays in train and
    ``valid.jsonl`` is empty so the trainer never sees a duplicated target.
    """
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(dataset_path)
    random.Random(shuffle_seed).shuffle(records)
    train_records, valid_records = _split_records(
        records,
        valid_fraction=valid_fraction,
    )

    _write_jsonl(out_dir / "train.jsonl", train_records)
    _write_jsonl(out_dir / "valid.jsonl", valid_records)
    return out_dir


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return list(token_ids)


def _thinking_close_token_index(
    *,
    prompt_len: int,
    completion: str,
    tokenizer: Any,
) -> int | None:
    lower = completion.lower()
    open_idx = lower.find(_GEMMA_THOUGHT_OPEN)
    if open_idx == -1:
        return None
    close_idx = lower.find(_GEMMA_THOUGHT_CLOSE, open_idx)
    if close_idx == -1:
        return -1
    close_end = close_idx + len(_GEMMA_THOUGHT_CLOSE)
    return prompt_len + len(
        _encode(
            tokenizer,
            completion[:close_end],
            add_special_tokens=False,
        )
    )


def _native_token_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
    loss_mask_mode: str = "completion",
    action_token_only: bool = False,
    action_token_weight: float = 1.0,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = str(record.get("prompt", ""))
    completion = str(record.get("completion", ""))
    if not prompt:
        return None, "missing_prompt"
    if not completion:
        return None, "missing_completion"

    if loss_mask_mode == "action":
        tokenized = tokenize_example(
            record,
            tokenizer,
            loss_mask_mode="action",
        )
        input_ids = list(tokenized["input_ids"])
        labels = list(tokenized["labels"])
        if action_token_only:
            labels = [
                token_id if is_action else -100
                for token_id, is_action in zip(
                    input_ids,
                    tokenized["action_mask"],
                )
            ]
        if len(input_ids) > max_seq_length:
            return None, "too_long"
        if not any(label != -100 for label in labels):
            return None, "missing_action_tokens"
        token_record = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": [label != -100 for label in labels],
            "n_prompt_tokens": int(tokenized["n_prompt_tokens"]),
            "n_completion_tokens": int(tokenized["n_completion_tokens"]),
            "n_format_tokens": int(tokenized["n_format_tokens"]),
            "n_thought_tokens": int(tokenized["n_thought_tokens"]),
            "n_action_tokens": int(tokenized["n_action_tokens"]),
            "n_supervised_format_tokens": (
                0
                if action_token_only
                else int(tokenized["n_supervised_format_tokens"])
            ),
            "n_supervised_thought_tokens": (
                0
                if action_token_only
                else int(tokenized["n_supervised_thought_tokens"])
            ),
            "n_supervised_action_tokens": int(
                tokenized["n_supervised_action_tokens"]
            ),
            "n_supervised_tokens": sum(label != -100 for label in labels),
            "n_total_tokens": len(input_ids),
            "world_seed": record.get("world_seed"),
            "decision_index": record.get("decision_index"),
            "stem": record.get("stem"),
            "task_window_id": record.get("task_window_id"),
            "schedule_step": record.get("schedule_step"),
            "source_identity": record.get("source_identity"),
        }
        if action_token_weight != 1.0:
            loss_weights = [
                (
                    action_token_weight
                    if label != -100 and is_action
                    else 1.0 if label != -100 else 0.0
                )
                for label, is_action in zip(labels, tokenized["action_mask"])
            ]
            token_record.update(
                {
                    "loss_weights": loss_weights,
                    "supervised_format_weight_mass": sum(
                        weight
                        for weight, is_format in zip(
                            loss_weights,
                            tokenized["format_mask"],
                        )
                        if is_format
                    ),
                    "supervised_action_weight_mass": sum(
                        weight
                        for weight, is_action in zip(
                            loss_weights,
                            tokenized["action_mask"],
                        )
                        if is_action
                    ),
                    "supervised_total_weight_mass": sum(loss_weights),
                }
            )
        return token_record, None

    prompt_ids = _encode(tokenizer, prompt)
    target_completion = (
        completion
        if completion.endswith(_GEMMA_ASSISTANT_TURN_END)
        else completion + _GEMMA_ASSISTANT_TURN_END
    )
    completion_ids = _encode(tokenizer, target_completion, add_special_tokens=False)
    if not completion_ids:
        return None, "missing_completion"

    thought_close = _thinking_close_token_index(
        prompt_len=len(prompt_ids),
        completion=completion,
        tokenizer=tokenizer,
    )
    if thought_close == -1:
        return None, "source_thinking_truncated"

    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_seq_length:
        if thought_close is not None and thought_close > max_seq_length:
            return None, "would_truncate_thinking"
        return None, "too_long"

    return (
        {
            "input_ids": input_ids,
            "offset": len(prompt_ids),
            "n_prompt_tokens": len(prompt_ids),
            "n_completion_tokens": len(completion_ids),
            "n_total_tokens": len(input_ids),
            "world_seed": record.get("world_seed"),
            "decision_index": record.get("decision_index"),
            "stem": record.get("stem"),
            "task_window_id": record.get("task_window_id"),
            "schedule_step": record.get("schedule_step"),
            "source_identity": record.get("source_identity"),
        },
        None,
    )


def prepare_native_mlx_data(
    dataset_path: Path,
    out_dir: Path,
    *,
    tokenizer: Any,
    max_seq_length: int = 8192,
    valid_fraction: float = 0.1,
    shuffle_seed: int = 0,
    loss_mask_mode: str = "completion",
    report_path: Path | None = None,
    expected_kept_records: int | None = None,
    action_token_only: bool = False,
    action_token_weight: float = 1.0,
    preserve_row_order: bool = False,
    expected_schedule_sha256: str | None = None,
) -> dict[str, Any]:
    """Pre-tokenize prompt+completion SFT data for native-thinking MLX LoRA.

    This bypasses `mlx_lm`'s stock ChatDataset, which strips Gemma-4 native
    thought-channel text from assistant messages. Records are written as token
    ids plus the completion offset expected by mlx-lm's prompt-masked loss.
    """
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if action_token_only and loss_mask_mode != "action":
        raise ValueError("action_token_only requires loss_mask_mode='action'")
    if (
        isinstance(action_token_weight, bool)
        or not isinstance(action_token_weight, (int, float))
        or not math.isfinite(action_token_weight)
        or action_token_weight <= 0
    ):
        raise ValueError("action_token_weight must be finite and positive")
    if action_token_weight != 1.0 and loss_mask_mode != "action":
        raise ValueError("action_token_weight requires loss_mask_mode='action'")
    if action_token_only and action_token_weight != 1.0:
        raise ValueError("action_token_weight must be 1 with action_token_only")

    records = _load_jsonl(dataset_path)
    skipped: Counter[str] = Counter()
    tokenized: list[dict[str, Any]] = []
    tokenization_errors: Counter[str] = Counter()
    first_tokenization_error: Exception | None = None
    for record in records:
        try:
            token_record, skip_reason = _native_token_record(
                record,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                loss_mask_mode=loss_mask_mode,
                action_token_only=action_token_only,
                action_token_weight=action_token_weight,
            )
        except Exception as exc:  # noqa: BLE001 - persist evidence, then fail below.
            tokenization_errors[exc.__class__.__name__] += 1
            if first_tokenization_error is None:
                first_tokenization_error = exc
            continue
        if skip_reason is not None:
            skipped[skip_reason] += 1
            continue
        assert token_record is not None
        tokenized.append(token_record)

    if not preserve_row_order:
        random.Random(shuffle_seed).shuffle(tokenized)
    train_records, valid_records = _split_records(
        tokenized,
        valid_fraction=valid_fraction,
    )
    _write_raw_jsonl(out_dir / "train.jsonl", train_records)
    _write_raw_jsonl(out_dir / "valid.jsonl", valid_records)

    lengths = [int(record["n_total_tokens"]) for record in tokenized]
    completion_lengths = [int(record["n_completion_tokens"]) for record in tokenized]
    report = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": file_sha256(dataset_path),
        "format": (
            (
                "pretokenized_action_value_mask"
                if action_token_only
                else "pretokenized_action_mask"
            )
            if loss_mask_mode == "action"
            else "pretokenized_prompt_completion"
        ),
        "loss_mask_mode": loss_mask_mode,
        "action_token_only": action_token_only,
        "action_token_weight": action_token_weight,
        "format_token_weight": 1.0,
        "loss_normalization": "sum_of_token_weights",
        "loss_weight_representation": (
            "boolean_mask" if action_token_weight == 1.0 else "numeric_weights"
        ),
        "preserve_row_order": preserve_row_order,
        "max_seq_length": max_seq_length,
        "shuffle_seed": shuffle_seed,
        "valid_fraction": valid_fraction,
        "n_input_records": len(records),
        "n_kept_records": len(tokenized),
        "n_train_records": len(train_records),
        "n_valid_records": len(valid_records),
        "skipped_record_counts": dict(skipped),
        "tokenization_error_counts": dict(tokenization_errors),
        "max_total_tokens": max(lengths, default=0),
        "max_completion_tokens": max(completion_lengths, default=0),
    }
    if loss_mask_mode == "action":
        report["token_count_totals"] = {
            key: sum(int(record.get(key, 0)) for record in tokenized)
            for key in (
                "n_prompt_tokens",
                "n_completion_tokens",
                "n_format_tokens",
                "n_thought_tokens",
                "n_action_tokens",
                "n_supervised_format_tokens",
                "n_supervised_thought_tokens",
                "n_supervised_action_tokens",
                "n_supervised_tokens",
            )
        }
        if action_token_weight == 1.0:
            report["supervised_weight_mass_totals"] = {
                "supervised_format_weight_mass": float(
                    report["token_count_totals"]["n_supervised_format_tokens"]
                ),
                "supervised_action_weight_mass": float(
                    report["token_count_totals"]["n_supervised_action_tokens"]
                ),
                "supervised_total_weight_mass": float(
                    report["token_count_totals"]["n_supervised_tokens"]
                ),
            }
        else:
            report["supervised_weight_mass_totals"] = {
                key: sum(float(record.get(key, 0.0)) for record in tokenized)
                for key in (
                    "supervised_format_weight_mass",
                    "supervised_action_weight_mass",
                    "supervised_total_weight_mass",
                )
            }
    scheduled_records = [
        {
            "schedule_step": record.get("schedule_step"),
            "source_identity": record.get("source_identity"),
        }
        for record in tokenized
        if record.get("schedule_step") is not None
        or record.get("source_identity") is not None
    ]
    if scheduled_records:
        if len(scheduled_records) != len(tokenized) or any(
            isinstance(record["schedule_step"], bool)
            or not isinstance(record["schedule_step"], int)
            or record["schedule_step"] < 0
            or not isinstance(record["source_identity"], str)
            or not record["source_identity"]
            for record in scheduled_records
        ):
            raise ValueError(
                "scheduled native MLX data requires non-negative schedule_step "
                "and non-empty source_identity on every retained row"
            )
        steps_contiguous = [
            record["schedule_step"] for record in scheduled_records
        ] == list(range(len(scheduled_records)))
        ordered_step_source_sha256 = hashlib.sha256(
            json.dumps(
                scheduled_records,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        report["schedule"] = {
            "n_records": len(scheduled_records),
            "steps_are_zero_based_contiguous_in_prepared_order": steps_contiguous,
            "n_unique_source_identities": len(
                {
                    str(record["source_identity"])
                    for record in scheduled_records
                }
            ),
            "ordered_step_source_sha256": ordered_step_source_sha256,
        }
        if preserve_row_order and not steps_contiguous:
            raise ValueError(
                "preserve_row_order prepared schedule steps are not contiguous"
            )
        if (
            expected_schedule_sha256 is not None
            and ordered_step_source_sha256 != expected_schedule_sha256
        ):
            raise ValueError(
                "preserve_row_order prepared schedule hash disagrees with manifest"
            )
    elif preserve_row_order:
        raise ValueError(
            "preserve_row_order requires schedule_step and source_identity on "
            "every retained row"
        )
    if loss_mask_mode == "action":
        total_lengths = [
            int(record["n_prompt_tokens"])
            + int(record["n_completion_tokens"])
            for record in tokenized
        ]
        report["token_count_totals"]["n_total_tokens"] = sum(total_lengths)
        report["token_count_totals"]["n_batch1_padded_tokens"] = sum(
            1 + 32 * ((length + 31) // 32)
            for length in total_lengths
        )
    if expected_kept_records is not None:
        if expected_kept_records <= 0:
            raise ValueError("expected_kept_records must be positive")
        report["expected_kept_records"] = expected_kept_records
        report["retained_count_matches_expected"] = (
            len(tokenized) == expected_kept_records
        )
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    data_report_path = out_dir / "native_mlx_data_report.json"
    data_report_path.write_text(report_text, encoding="utf-8")
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.resolve() != data_report_path.resolve():
            report_path.write_text(report_text, encoding="utf-8")
    print(
        "Native MLX data: "
        f"kept {len(tokenized)}/{len(records)} examples; "
        f"skipped {sum(skipped.values())}; "
        f"skipped_record_counts={dict(skipped)}; "
        f"max_total_tokens={report['max_total_tokens']}; "
        f"max_completion_tokens={report['max_completion_tokens']}",
        file=sys.stderr,
        flush=True,
    )
    if first_tokenization_error is not None:
        raise ValueError(
            "native MLX data preparation hit tokenization errors; "
            f"tokenization_error_counts={dict(tokenization_errors)}; "
            f"report={report_path or data_report_path}"
        ) from first_tokenization_error
    if not tokenized:
        raise ValueError(
            "native MLX data preparation kept zero examples; "
            f"skipped_record_counts={dict(skipped)}; "
            f"report={report_path or data_report_path}"
        )
    if (
        expected_kept_records is not None
        and len(tokenized) != expected_kept_records
    ):
        raise ValueError(
            "native MLX data preparation retained an unexpected number of "
            f"examples: expected={expected_kept_records} actual={len(tokenized)}; "
            f"report={report_path or data_report_path}"
        )
    return report


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read dataset manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"dataset manifest {path} is not a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_model_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _model_snapshot_provenance(
    snapshot: Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    snapshot = Path(snapshot).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.name != revision:
        raise ValueError("resolved model snapshot does not match requested revision")

    required_small_files = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    files: dict[str, dict[str, Any]] = {}
    for name in required_small_files:
        path = snapshot / name
        digest = file_sha256(path)
        if not _is_sha256(digest):
            raise ValueError(f"model snapshot is missing or cannot hash {name}")
        files[name] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }

    weight_paths = sorted(snapshot.glob("*.safetensors"))
    if not weight_paths:
        raise ValueError("model snapshot has no safetensors weights")
    for path in weight_paths:
        resolved = path.resolve()
        expected_blob_sha256 = (
            resolved.name if _is_sha256(resolved.name) else None
        )
        digest = file_sha256(path)
        if not _is_sha256(digest):
            raise ValueError(f"model weight could not be identified: {path.name}")
        if expected_blob_sha256 is not None and digest != expected_blob_sha256:
            raise ValueError(f"model weight CAS digest mismatch: {path.name}")
        files[path.name] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }

    identity = {
        "model_id": model_id,
        "revision": revision,
        "files": files,
    }
    return {
        **identity,
        "snapshot_path": str(snapshot),
        "identity_sha256": _canonical_sha256(identity),
    }


def _resolve_cached_model_snapshot(
    model_id: str,
    *,
    revision: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and content-address one exact locally cached HF snapshot."""

    if not _is_model_revision(revision):
        raise ValueError(
            "model_revision must be a lowercase 40-character commit"
        )
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    ).resolve()
    provenance = _model_snapshot_provenance(
        snapshot,
        model_id=model_id,
        revision=revision,
    )
    return snapshot, provenance


def _require_model_snapshot_unchanged(
    expected_provenance: dict[str, Any] | None,
) -> None:
    if expected_provenance is None:
        return
    model_id = expected_provenance.get("model_id")
    revision = expected_provenance.get("revision")
    snapshot_path = expected_provenance.get("snapshot_path")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not _is_model_revision(revision)
        or not isinstance(snapshot_path, str)
        or not snapshot_path
    ):
        raise ValueError("base model provenance is invalid")
    observed = _model_snapshot_provenance(
        Path(snapshot_path),
        model_id=model_id,
        revision=revision,
    )
    if observed != expected_provenance:
        raise ValueError(
            "base model snapshot changed after training preflight: "
            f"expected={expected_provenance.get('identity_sha256')!r} "
            f"actual={observed.get('identity_sha256')!r}"
        )


def _ordered_step_source_sha256(rows: list[dict[str, Any]]) -> str:
    schedule = [
        {
            "schedule_step": row.get("schedule_step"),
            "source_identity": row.get("source_identity"),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            schedule,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_preserved_row_order_contract(
    *,
    dataset_path: Path,
    manifest: dict[str, Any],
    valid_fraction: float,
    iters: int,
    batch_size: int,
    grad_accumulation_steps: int,
) -> str:
    """Require a complete one-row-per-update paired schedule."""

    if valid_fraction != 0.0:
        raise ValueError("preserve_row_order requires valid_fraction=0")
    if batch_size != 1:
        raise ValueError("preserve_row_order requires batch_size=1")
    if grad_accumulation_steps != 1:
        raise ValueError(
            "preserve_row_order requires grad_accumulation_steps=1"
        )
    augmentation = manifest.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError(
            "preserve_row_order requires paired augmentation manifest metadata"
        )
    expected = {
        "kind": _PAIRED_AUGMENTATION_KIND,
        "version": _PAIRED_AUGMENTATION_VERSION,
        "n_output_rows": manifest.get("n_examples"),
    }
    mismatches = {
        key: {"stored": augmentation.get(key), "expected": value}
        for key, value in expected.items()
        if augmentation.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "preserve_row_order augmentation contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    for key in ("source_schedule_sha256", "ordered_step_source_sha256"):
        if not _is_sha256(augmentation.get(key)):
            raise ValueError(
                f"preserve_row_order augmentation {key} is invalid"
            )
    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError("preserve_row_order dataset is empty")
    for row_index, row in enumerate(rows):
        if row.get("schedule_step") != row_index:
            raise ValueError(
                "preserve_row_order schedule_step must be contiguous in dataset "
                f"order: row={row_index} stored={row.get('schedule_step')!r}"
            )
        if not _is_sha256(row.get("source_identity")):
            raise ValueError(
                f"preserve_row_order row {row_index} source_identity is invalid"
            )
    if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
        raise ValueError("preserve_row_order requires positive iters")
    if iters > len(rows):
        raise ValueError(
            "preserve_row_order iters cannot exceed the explicit schedule: "
            f"iters={iters} rows={len(rows)}"
        )
    observed = _ordered_step_source_sha256(rows)
    if augmentation["ordered_step_source_sha256"] != observed:
        raise ValueError(
            "preserve_row_order ordered_step_source_sha256 disagrees with dataset"
        )
    return observed


def _validate_search_teacher_training_contract(
    *,
    dataset_path: Path,
    manifest: dict[str, Any],
    base_model: str,
    expected_example_count: int | None,
) -> str:
    """Validate the content-addressed action-only teacher SFT boundary."""

    expected_manifest = {
        "kind": _SEARCH_TEACHER_SFT_KIND,
        "version": _SEARCH_TEACHER_SFT_VERSION,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "loss_mask_mode": "action",
        "output_contract": ACTION_ONLY_OUTPUT,
        "enable_thinking": False,
        "tokenizer_id": base_model,
    }
    mismatches = {
        key: {"stored": manifest.get(key), "expected": expected}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "search-teacher SFT manifest is incompatible with strict MLX "
            f"training: {json.dumps(mismatches, sort_keys=True)}"
        )

    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError("search-teacher SFT dataset is empty")
    n_examples = manifest.get("n_examples")
    if isinstance(n_examples, bool) or not isinstance(n_examples, int):
        raise ValueError("search-teacher SFT manifest n_examples must be an integer")
    if n_examples != len(rows):
        raise ValueError(
            "search-teacher SFT manifest n_examples disagrees with dataset: "
            f"manifest={n_examples} actual={len(rows)}"
        )
    if expected_example_count is not None and n_examples != expected_example_count:
        raise ValueError(
            "search-teacher SFT example count disagrees with the requested gate: "
            f"expected={expected_example_count} manifest={n_examples}"
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"search-teacher SFT dataset row {row_index} is not a JSON object"
            )
        expected_row = {
            "observation_version": PUBLIC_OBSERVATION_VERSION,
            "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
            "teacher_privilege": TEACHER_PRIVILEGE,
            "loss_mask_mode": "action",
            "output_contract": ACTION_ONLY_OUTPUT,
        }
        row_mismatches = {
            key: {"stored": row.get(key), "expected": expected}
            for key, expected in expected_row.items()
            if row.get(key) != expected
        }
        if row_mismatches:
            raise ValueError(
                f"search-teacher SFT dataset row {row_index} violates its "
                f"training contract: {json.dumps(row_mismatches, sort_keys=True)}"
            )
        try:
            validate_teacher_row(row)
        except ValueError as exc:
            raise ValueError(
                f"search-teacher SFT dataset row {row_index} has an invalid "
                f"action-only target: {exc}"
            ) from exc

    manifest_digest = manifest.get("dataset_sha256")
    if not _is_sha256(manifest_digest):
        raise ValueError("search-teacher SFT manifest dataset_sha256 is invalid")
    observed_digest = file_sha256(dataset_path)
    if observed_digest != manifest_digest:
        raise ValueError(
            "search-teacher SFT manifest dataset_sha256 disagrees with dataset: "
            f"manifest={manifest_digest!r} actual={observed_digest!r}"
        )
    return str(manifest_digest)


def _require_dataset_sha256(dataset_path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    observed = file_sha256(dataset_path)
    if observed != expected_sha256:
        raise ValueError(
            "search-teacher SFT dataset changed after preflight: "
            f"expected={expected_sha256!r} actual={observed!r}"
        )


def _mlx_reproducibility_config(
    *,
    seed: int,
    lora_rank: int,
    lora_scale: float,
    lora_dropout: float,
    grad_accumulation_steps: int,
    action_token_only: bool = False,
    action_token_weight: float = 1.0,
    preserve_row_order: bool = False,
    expected_schedule_sha256: str | None = None,
    base_model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "seed": seed,
        "grad_accumulation_steps": grad_accumulation_steps,
        "action_token_only": action_token_only,
        "action_token_weight": action_token_weight,
        "format_token_weight": 1.0,
        "loss_normalization": "sum_of_token_weights",
        "loss_weight_representation": (
            "boolean_mask" if action_token_weight == 1.0 else "numeric_weights"
        ),
        "preserve_row_order": preserve_row_order,
        "expected_schedule_sha256": expected_schedule_sha256,
        "lora_parameters": {
            "rank": lora_rank,
            "scale": lora_scale,
            "dropout": lora_dropout,
        },
    }
    if base_model_provenance is not None:
        config.update(
            {
                "base_model_revision": base_model_provenance["revision"],
                "base_model_identity_sha256": base_model_provenance[
                    "identity_sha256"
                ],
                "base_model_provenance": deepcopy(base_model_provenance),
            }
        )
    return config


def _warn_if_tokenizer_mismatch(manifest_path: Path, base_model: str) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer_id = manifest.get("tokenizer_id")
        if tokenizer_id and str(tokenizer_id) != base_model:
            print(
                "WARNING: dataset manifest tokenizer_id "
                f"{tokenizer_id!r} differs from base_model {base_model!r}; "
                "continuing because compatible models can share tokenizers.",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - manifest checks are advisory for MLX.
        print(
            f"WARNING: could not check dataset manifest {manifest_path}: {exc}",
            file=sys.stderr,
        )


def build_lora_cmd(
    *,
    python_exe: str,
    base_model: str,
    data_dir: Path,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    mask_prompt: bool = True,
    max_seq_length: int = 8192,
    seed: int = 0,
    grad_accumulation_steps: int = 1,
    config_path: Path | None = None,
) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        base_model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(out_adapter_dir),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(num_layers),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
        "--seed",
        str(seed),
        "--grad-accumulation-steps",
        str(grad_accumulation_steps),
    ]

    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    if mask_prompt:
        cmd.append("--mask-prompt")

    if wandb_project is not None:
        cmd.extend(["--report-to", "wandb", "--project-name", wandb_project])

    # Confirm these flag names with `python -m mlx_lm lora --help` for the
    # installed mlx-lm; pass only when set so an unknown flag cannot break a
    # default run.
    if steps_per_eval is not None:
        cmd.extend(["--steps-per-eval", str(steps_per_eval)])
    if steps_per_report is not None:
        cmd.extend(["--steps-per-report", str(steps_per_report)])
    if save_every is not None:
        cmd.extend(["--save-every", str(save_every)])
    if val_batches is not None:
        cmd.extend(["--val-batches", str(val_batches)])

    return cmd


def _run_mlx_lora(
    *,
    data_dir: Path,
    base_model: str,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    mask_prompt: bool = True,
    max_seq_length: int = 8192,
    seed: int = 0,
    grad_accumulation_steps: int = 1,
    config_path: Path | None = None,
) -> None:
    subprocess.run(
        build_lora_cmd(
            python_exe=sys.executable,
            base_model=base_model,
            data_dir=data_dir,
            out_adapter_dir=out_adapter_dir,
            num_layers=num_layers,
            iters=iters,
            batch_size=batch_size,
            learning_rate=learning_rate,
            wandb_project=wandb_project,
            steps_per_eval=steps_per_eval,
            steps_per_report=steps_per_report,
            save_every=save_every,
            val_batches=val_batches,
            mask_prompt=mask_prompt,
            max_seq_length=max_seq_length,
            seed=seed,
            grad_accumulation_steps=grad_accumulation_steps,
            config_path=config_path,
        ),
        check=True,
    )


class _PreTokenizedExample:
    def __init__(self, record: dict[str, Any]):
        self.record = record

    def __len__(self) -> int:
        return len(self.record["input_ids"])


class _PreTokenizedDataset:
    def __init__(self, records: list[dict[str, Any]], *, mask_prompt: bool = True):
        self._records = [_PreTokenizedExample(record) for record in records]
        self._mask_prompt = mask_prompt

    def process(self, example: _PreTokenizedExample) -> tuple[list[int], int]:
        record = example.record
        offset = int(record["offset"]) if self._mask_prompt else 0
        return list(record["input_ids"]), offset

    def __getitem__(self, index: int) -> _PreTokenizedExample:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)


class _MaskedPreTokenizedDataset(_PreTokenizedDataset):
    def process(
        self,
        example: _PreTokenizedExample,
    ) -> tuple[list[int], list[bool] | list[float]]:
        record = example.record
        if "loss_weights" in record:
            return list(record["input_ids"]), [
                float(weight) for weight in record["loss_weights"]
            ]
        return list(record["input_ids"]), list(record["loss_mask"])


def _load_pretokenized_dataset(
    path: Path,
    *,
    mask_prompt: bool = True,
) -> _PreTokenizedDataset:
    if not path.exists():
        return _PreTokenizedDataset([], mask_prompt=mask_prompt)
    return _PreTokenizedDataset(_load_jsonl(path), mask_prompt=mask_prompt)


def _load_masked_pretokenized_dataset(path: Path) -> _MaskedPreTokenizedDataset:
    if not path.exists():
        return _MaskedPreTokenizedDataset([])
    return _MaskedPreTokenizedDataset(_load_jsonl(path))


def _iterate_masked_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any | None = None,
    preserve_row_order: bool = False,
):
    """MLX batch iterator that preserves a non-contiguous per-token mask."""
    import mlx.core as mx
    import numpy as np

    len_fn = (
        (lambda index: dataset.itemlen(index))
        if hasattr(dataset, "itemlen")
        else (lambda index: len(dataset[index][0]))
    )
    if preserve_row_order and batch_size != 1:
        raise ValueError("preserve_row_order requires batch_size=1")
    indices = (
        list(range(len(dataset)))
        if preserve_row_order
        else sorted(range(len(dataset)), key=len_fn)
    )
    if loop and len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} examples but "
            f"only has {len(dataset)}."
        )
    offset = comm_group.rank() if comm_group is not None else 0
    step = comm_group.size() if comm_group is not None else 1
    if batch_size % step != 0:
        raise ValueError("batch size must be divisible by distributed world size")
    stop = len(indices) - batch_size + 1 if loop else len(indices)
    batches = [
        indices[index + offset : index + offset + batch_size : step]
        for index in range(0, stop, batch_size)
    ]
    batches = [batch for batch in batches if batch]
    rng = np.random.default_rng(seed)

    while True:
        batch_order = (
            range(len(batches))
            if preserve_row_order
            else rng.permutation(len(batches))
        )
        for batch_index in batch_order:
            rows = [dataset[index] for index in batches[batch_index]]
            token_rows, weight_rows = zip(*rows)
            numeric_weight_rows = [
                any(not isinstance(weight, bool) for weight in weights)
                for weights in weight_rows
            ]
            if any(numeric_weight_rows) and not all(numeric_weight_rows):
                raise ValueError(
                    "masked MLX batch cannot mix boolean masks and numeric weights"
                )
            lengths = [len(tokens) for tokens in token_rows]
            max_length = min(
                1 + 32 * ((max(lengths) + 31) // 32),
                max_seq_length,
            )
            local_batch_size = len(rows)
            tokens_array = np.zeros((local_batch_size, max_length), np.int32)
            weight_dtype = (
                np.float32 if any(numeric_weight_rows) else np.bool_
            )
            weight_array = np.zeros((local_batch_size, max_length), weight_dtype)
            for row_index, (tokens, weights) in enumerate(
                zip(token_rows, weight_rows)
            ):
                truncated_length = min(len(tokens), max_seq_length)
                tokens_array[row_index, :truncated_length] = tokens[:truncated_length]
                weight_array[row_index, :truncated_length] = weights[
                    :truncated_length
                ]
            yield mx.array(tokens_array), mx.array(weight_array)
        if not loop:
            break


def _masked_ce_loss(model: Any, batch: Any, loss_weights: Any):
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    target_weights = loss_weights[:, 1:]
    logits = model(inputs)
    total_weight = target_weights.sum()
    ce = nn.losses.cross_entropy(logits, targets) * target_weights
    return ce.astype(mx.float32).sum() / total_weight, total_weight


def _has_gemma_thought_completion(dataset_path: Path) -> bool:
    for record in _load_jsonl(dataset_path):
        completion = str(record.get("completion", ""))
        if any(marker in completion for marker in _GEMMA_THOUGHT_MARKERS):
            return True
    return False


def _run_mlx_lora_native(
    *,
    dataset_path: Path,
    data_dir: Path,
    base_model: str,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    valid_fraction: float = 0.1,
    mask_prompt: bool = True,
    preparation_report_path: Path | None = None,
    expected_example_count: int | None = None,
    expected_dataset_sha256: str | None = None,
    seed: int = 0,
    lora_rank: int = 8,
    lora_scale: float = 20.0,
    lora_dropout: float = 0.0,
    grad_accumulation_steps: int = 1,
) -> None:
    del wandb_project  # Native path uses mlx-lm's trainer directly; logging is not wired yet.
    import numpy as np
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.utils import load

    print("Loading pretrained model")
    model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})
    prepare_native_mlx_data(
        dataset_path,
        data_dir,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        valid_fraction=valid_fraction,
        report_path=preparation_report_path,
        expected_kept_records=expected_example_count,
    )
    _require_dataset_sha256(dataset_path, expected_dataset_sha256)
    train_set = _load_pretokenized_dataset(
        data_dir / "train.jsonl",
        mask_prompt=mask_prompt,
    )
    valid_set = _load_pretokenized_dataset(
        data_dir / "valid.jsonl",
        mask_prompt=mask_prompt,
    )

    cfg = deepcopy(CONFIG_DEFAULTS)
    cfg.update(
        {
            "model": base_model,
            "train": True,
            "fine_tune_type": "lora",
            "data": str(data_dir),
            "num_layers": num_layers,
            "batch_size": batch_size,
            "iters": iters,
            "learning_rate": learning_rate,
            "adapter_path": str(out_adapter_dir),
            "max_seq_length": max_seq_length,
            "mask_prompt": mask_prompt,
            "steps_per_eval": steps_per_eval
            if steps_per_eval is not None
            else CONFIG_DEFAULTS["steps_per_eval"],
            "steps_per_report": steps_per_report
            if steps_per_report is not None
            else CONFIG_DEFAULTS["steps_per_report"],
            "save_every": save_every
            if save_every is not None
            else CONFIG_DEFAULTS["save_every"],
            "val_batches": val_batches
            if val_batches is not None
            else CONFIG_DEFAULTS["val_batches"],
            "project_name": None,
            "report_to": None,
            **_mlx_reproducibility_config(
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
            ),
        }
    )
    args = types.SimpleNamespace(**cfg)

    print("Training")
    # Calling train_model directly bypasses mlx_lm.lora.run(), which normally
    # seeds NumPy before its batch iterator uses the process-global RNG.
    np.random.seed(int(args.seed))
    train_model(args, model, train_set, valid_set)


def _run_mlx_lora_action_masked(
    *,
    dataset_path: Path,
    data_dir: Path,
    base_model: str,
    model_load_path: str | None,
    base_model_provenance: dict[str, Any] | None,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    valid_fraction: float = 0.1,
    preparation_report_path: Path | None = None,
    expected_example_count: int | None = None,
    expected_dataset_sha256: str | None = None,
    seed: int = 0,
    lora_rank: int = 8,
    lora_scale: float = 20.0,
    lora_dropout: float = 0.0,
    grad_accumulation_steps: int = 1,
    action_token_only: bool = False,
    action_token_weight: float = 1.0,
    preserve_row_order: bool = False,
    expected_schedule_sha256: str | None = None,
) -> None:
    """Run mlx-lm LoRA with an arbitrary action/format token mask."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.lora import (
        CONFIG_DEFAULTS,
        linear_to_lora_layers,
        print_trainable_parameters,
        save_config,
    )
    from mlx_lm.tuner.trainer import CacheDataset, TrainingArgs, train as tuner_train
    from mlx_lm.utils import load

    print("Loading pretrained model")
    _require_model_snapshot_unchanged(base_model_provenance)
    model, tokenizer = load(
        model_load_path or base_model,
        tokenizer_config={"trust_remote_code": True},
    )
    _require_model_snapshot_unchanged(base_model_provenance)
    prepare_native_mlx_data(
        dataset_path,
        data_dir,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        valid_fraction=valid_fraction,
        loss_mask_mode="action",
        report_path=preparation_report_path,
        expected_kept_records=expected_example_count,
        action_token_only=action_token_only,
        action_token_weight=action_token_weight,
        preserve_row_order=preserve_row_order,
        expected_schedule_sha256=expected_schedule_sha256,
    )
    _require_dataset_sha256(dataset_path, expected_dataset_sha256)
    train_set = _load_masked_pretokenized_dataset(data_dir / "train.jsonl")
    valid_set = _load_masked_pretokenized_dataset(data_dir / "valid.jsonl")

    cfg = deepcopy(CONFIG_DEFAULTS)
    cfg.update(
        {
            "model": base_model,
            "train": True,
            "fine_tune_type": "lora",
            "data": str(data_dir),
            "num_layers": num_layers,
            "batch_size": batch_size,
            "iters": iters,
            "learning_rate": learning_rate,
            "adapter_path": str(out_adapter_dir),
            "max_seq_length": max_seq_length,
            "mask_prompt": True,
            "steps_per_eval": steps_per_eval
            if steps_per_eval is not None
            else CONFIG_DEFAULTS["steps_per_eval"],
            "steps_per_report": steps_per_report
            if steps_per_report is not None
            else CONFIG_DEFAULTS["steps_per_report"],
            "save_every": save_every
            if save_every is not None
            else CONFIG_DEFAULTS["save_every"],
            "val_batches": val_batches
            if val_batches is not None
            else CONFIG_DEFAULTS["val_batches"],
            "report_to": None,
            "project_name": None,
            **_mlx_reproducibility_config(
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
                action_token_only=action_token_only,
                action_token_weight=action_token_weight,
                preserve_row_order=preserve_row_order,
                expected_schedule_sha256=expected_schedule_sha256,
                base_model_provenance=base_model_provenance,
            ),
        }
    )
    args = types.SimpleNamespace(**cfg)
    training_seed = int(args.seed)
    mx.random.seed(training_seed)
    if num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {num_layers} layers but the model has "
            f"only {len(model.layers)}."
        )
    model.freeze()
    linear_to_lora_layers(model, num_layers, args.lora_parameters)
    print_trainable_parameters(model)

    out_adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = out_adapter_dir / "adapters.safetensors"
    save_config(vars(args), out_adapter_dir / "adapter_config.json")
    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    tuner_train(
        model=model,
        optimizer=optim.Adam(learning_rate=learning_rate),
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
        loss=_masked_ce_loss,
        iterate_batches=partial(
            _iterate_masked_batches,
            seed=training_seed,
            preserve_row_order=preserve_row_order,
        ),
    )
    _require_model_snapshot_unchanged(base_model_provenance)


def train(
    dataset_path: Path,
    base_model: str,
    out_adapter_dir: Path,
    *,
    num_layers: int = 8,
    iters: int = 200,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    valid_fraction: float = 0.1,
    data_dir: Path | None = None,
    manifest_path: Path | None = None,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    mask_prompt: bool = True,
    max_seq_length: int = 8192,
    loss_mask_mode: str = "auto",
    expected_example_count: int | None = None,
    seed: int = 0,
    lora_rank: int = 8,
    lora_scale: float = 20.0,
    lora_dropout: float = 0.0,
    grad_accumulation_steps: int = 1,
    action_token_only: bool = False,
    action_token_weight: float = 1.0,
    preserve_row_order: bool = False,
    model_revision: str | None = None,
) -> Path:
    dataset_path = Path(dataset_path)
    out_adapter_dir = Path(out_adapter_dir)
    if out_adapter_dir.exists() or out_adapter_dir.is_symlink():
        raise ValueError(
            "MLX adapter output must be a fresh, nonexistent path: "
            f"{out_adapter_dir}"
        )
    if expected_example_count is not None and expected_example_count <= 0:
        raise ValueError("expected_example_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank <= 0:
        raise ValueError("lora_rank must be a positive integer")
    if not lora_scale > 0:
        raise ValueError("lora_scale must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0.0, 1.0)")
    if (
        isinstance(grad_accumulation_steps, bool)
        or not isinstance(grad_accumulation_steps, int)
        or grad_accumulation_steps <= 0
    ):
        raise ValueError("grad_accumulation_steps must be a positive integer")
    if (
        isinstance(action_token_weight, bool)
        or not isinstance(action_token_weight, (int, float))
        or not math.isfinite(action_token_weight)
        or action_token_weight <= 0
    ):
        raise ValueError("action_token_weight must be finite and positive")

    manifest: dict[str, Any] | None = None
    strict_teacher_sha256: str | None = None
    strict_schedule_sha256: str | None = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest = _load_manifest(manifest_path)
    requires_teacher_contract = expected_example_count is not None
    is_teacher_manifest = (
        manifest is not None
        and manifest.get("kind") == _SEARCH_TEACHER_SFT_KIND
    )
    if requires_teacher_contract and not is_teacher_manifest:
        raise ValueError(
            "expected_example_count requires a search_teacher_sft version 3 manifest"
        )
    if action_token_only and not is_teacher_manifest:
        raise ValueError(
            "action_token_only requires a strict search_teacher_sft version 3 "
            "action manifest"
        )
    if action_token_weight != 1.0 and not is_teacher_manifest:
        raise ValueError(
            "action_token_weight requires a strict search_teacher_sft version 3 "
            "action manifest"
        )
    if action_token_only and action_token_weight != 1.0:
        raise ValueError("action_token_weight must be 1 with action_token_only")
    if preserve_row_order and not is_teacher_manifest:
        raise ValueError(
            "preserve_row_order requires a strict search_teacher_sft version 3 "
            "action manifest"
        )
    if preserve_row_order and batch_size != 1:
        raise ValueError("preserve_row_order requires batch_size=1")
    if is_teacher_manifest:
        assert manifest_path is not None
        strict_teacher_sha256 = _validate_search_teacher_training_contract(
            dataset_path=dataset_path,
            manifest=manifest,
            base_model=base_model,
            expected_example_count=expected_example_count,
        )
        if preserve_row_order:
            strict_schedule_sha256 = _validate_preserved_row_order_contract(
                dataset_path=dataset_path,
                manifest=manifest,
                valid_fraction=valid_fraction,
                iters=iters,
                batch_size=batch_size,
                grad_accumulation_steps=grad_accumulation_steps,
            )

    resolved_loss_mask_mode = resolve_loss_mask_mode(
        loss_mask_mode,
        manifest_path=manifest_path,
    )
    if action_token_only and resolved_loss_mask_mode != "action":
        raise ValueError("action_token_only requires loss_mask_mode='action'")
    if action_token_weight != 1.0 and resolved_loss_mask_mode != "action":
        raise ValueError("action_token_weight requires loss_mask_mode='action'")
    if preserve_row_order and resolved_loss_mask_mode != "action":
        raise ValueError("preserve_row_order requires loss_mask_mode='action'")
    if is_teacher_manifest:
        if not _is_model_revision(model_revision):
            raise ValueError(
                "strict search_teacher_sft MLX training requires model_revision "
                "as a lowercase 40-character cached Hugging Face commit"
            )
    elif model_revision is not None:
        raise ValueError(
            "model_revision is supported only for strict search_teacher_sft "
            "MLX training"
        )

    if manifest_path is not None and not is_teacher_manifest:
        _warn_if_tokenizer_mismatch(manifest_path, base_model)

    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("install .[train-mlx]") from exc

    model_load_path: str | None = None
    base_model_provenance: dict[str, Any] | None = None
    if is_teacher_manifest:
        assert model_revision is not None
        snapshot, base_model_provenance = _resolve_cached_model_snapshot(
            base_model,
            revision=model_revision,
        )
        model_load_path = str(snapshot)

    out_adapter_dir.mkdir(parents=True, exist_ok=False)
    preparation_report_path = out_adapter_dir / "native_mlx_data_report.json"
    mlx_config_path = out_adapter_dir / "mlx_lora_config.json"
    mlx_config_path.write_text(
        json.dumps(
            _mlx_reproducibility_config(
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
                action_token_only=action_token_only,
                action_token_weight=action_token_weight,
                preserve_row_order=preserve_row_order,
                expected_schedule_sha256=strict_schedule_sha256,
                base_model_provenance=base_model_provenance,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if data_dir is not None:
        prepared_data_dir = Path(data_dir)
        if resolved_loss_mask_mode == "action":
            if not mask_prompt:
                raise ValueError("action loss requires mask_prompt=True")
            if wandb_project is not None:
                print(
                    "WARNING: action-masked native MLX SFT does not yet emit "
                    "wandb telemetry.",
                    file=sys.stderr,
                )
            _run_mlx_lora_action_masked(
                dataset_path=dataset_path,
                data_dir=prepared_data_dir,
                base_model=base_model,
                model_load_path=model_load_path,
                base_model_provenance=base_model_provenance,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                valid_fraction=valid_fraction,
                preparation_report_path=preparation_report_path,
                expected_example_count=expected_example_count,
                expected_dataset_sha256=strict_teacher_sha256,
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
                action_token_only=action_token_only,
                action_token_weight=action_token_weight,
                preserve_row_order=preserve_row_order,
                expected_schedule_sha256=strict_schedule_sha256,
            )
        elif _has_gemma_thought_completion(dataset_path):
            _run_mlx_lora_native(
                dataset_path=dataset_path,
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                valid_fraction=valid_fraction,
                mask_prompt=mask_prompt,
                preparation_report_path=preparation_report_path,
                expected_example_count=expected_example_count,
                expected_dataset_sha256=strict_teacher_sha256,
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
            )
        else:
            prepared_data_dir = prepare_mlx_data(
                dataset_path,
                prepared_data_dir,
                valid_fraction=valid_fraction,
            )
            _run_mlx_lora(
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                mask_prompt=mask_prompt,
                max_seq_length=max_seq_length,
                seed=seed,
                grad_accumulation_steps=grad_accumulation_steps,
                config_path=mlx_config_path,
            )
        return out_adapter_dir

    with tempfile.TemporaryDirectory(prefix="mlx_data_", dir=out_adapter_dir) as tmp:
        prepared_data_dir = Path(tmp)
        if resolved_loss_mask_mode == "action":
            if not mask_prompt:
                raise ValueError("action loss requires mask_prompt=True")
            if wandb_project is not None:
                print(
                    "WARNING: action-masked native MLX SFT does not yet emit "
                    "wandb telemetry.",
                    file=sys.stderr,
                )
            _run_mlx_lora_action_masked(
                dataset_path=dataset_path,
                data_dir=prepared_data_dir,
                base_model=base_model,
                model_load_path=model_load_path,
                base_model_provenance=base_model_provenance,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                valid_fraction=valid_fraction,
                preparation_report_path=preparation_report_path,
                expected_example_count=expected_example_count,
                expected_dataset_sha256=strict_teacher_sha256,
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
                action_token_only=action_token_only,
                action_token_weight=action_token_weight,
                preserve_row_order=preserve_row_order,
                expected_schedule_sha256=strict_schedule_sha256,
            )
        elif _has_gemma_thought_completion(dataset_path):
            _run_mlx_lora_native(
                dataset_path=dataset_path,
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                valid_fraction=valid_fraction,
                mask_prompt=mask_prompt,
                preparation_report_path=preparation_report_path,
                expected_example_count=expected_example_count,
                expected_dataset_sha256=strict_teacher_sha256,
                seed=seed,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                grad_accumulation_steps=grad_accumulation_steps,
            )
        else:
            prepared_data_dir = prepare_mlx_data(
                dataset_path,
                prepared_data_dir,
                valid_fraction=valid_fraction,
            )
            _run_mlx_lora(
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                mask_prompt=mask_prompt,
                max_seq_length=max_seq_length,
                seed=seed,
                grad_accumulation_steps=grad_accumulation_steps,
                config_path=mlx_config_path,
            )
    return out_adapter_dir
