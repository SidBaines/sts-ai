#!/usr/bin/env python
"""Decompose exact-schedule COMP-015 SFT loss into format/action tokens."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Sequence

from sts_ai.action_order_diagnostic import validate_prompt_round_trip
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.sft_loss_decomposition import (
    MlxTeacherForcedLossScorer,
    build_loss_decomposition_report,
    validate_exact_schedule,
)
from sts_ai.train.sft_format import chat_template_probe_hash


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _resolve_model_snapshot(
    model_id: str,
    *,
    expected_revision: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and content-address one exact cached Hugging Face revision."""

    if not _REVISION_RE.fullmatch(expected_revision):
        raise ValueError(
            "--expected-model-revision must be a lowercase 40-character commit"
        )
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=expected_revision,
            local_files_only=True,
        )
    ).resolve()
    if not snapshot.is_dir() or snapshot.name != expected_revision:
        raise ValueError("resolved model snapshot does not match expected revision")
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
        digest = (
            resolved.name
            if _is_sha256(resolved.name)
            else file_sha256(path)
        )
        if not _is_sha256(digest):
            raise ValueError(f"model weight could not be identified: {path.name}")
        files[path.name] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    provenance = {
        "model_id": model_id,
        "revision": expected_revision,
        "snapshot_path": str(snapshot),
        "files": files,
        "identity_sha256": _canonical_sha256(
            {
                "model_id": model_id,
                "revision": expected_revision,
                "files": files,
            }
        ),
    }
    return snapshot, provenance


def _runtime_tokenizer_provenance(tokenizer: Any) -> dict[str, Any]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("runtime tokenizer backend cannot be content-addressed")
    backend_json = backend.to_str()
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(backend_json, str) or not backend_json:
        raise ValueError("runtime tokenizer backend serialization is invalid")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("runtime tokenizer chat template is invalid")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise ValueError("runtime tokenizer vocab size is invalid")
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": vocab_size,
        "backend_tokenizer_sha256": hashlib.sha256(
            backend_json.encode("utf-8")
        ).hexdigest(),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
    }


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: row is not a JSON object"
                    )
                rows.append(value)
    except OSError as exc:
        raise ValueError(f"could not read dataset {path}: {exc}") from exc
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _checkpoint_provenance(
    path: Path,
    *,
    model_id: str,
    schedule_sha256: str,
    n_rows: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step <= 0
        or checkpoint_step > n_rows
    ):
        raise ValueError("checkpoint step must be in [1, n_rows]")
    if not path.is_dir():
        raise ValueError("checkpoint must be an adapter directory")
    provenance = adapter_provenance(path)
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint provenance could not be computed")
    files = provenance.get("files")
    if (
        not isinstance(files, dict)
        or "adapters.safetensors" not in files
        or "adapter_config.json" not in files
    ):
        raise ValueError(
            "checkpoint must contain adapters.safetensors and adapter_config.json"
        )
    adapter_config_path = path / "adapter_config.json"
    mlx_config_path = path / "mlx_lora_config.json"
    if not mlx_config_path.is_file():
        raise ValueError("checkpoint has no mlx_lora_config.json")
    adapter_config = _load_json(
        adapter_config_path,
        description="adapter config",
    )
    mlx_config = _load_json(
        mlx_config_path,
        description="MLX reproducibility config",
    )
    expected_adapter = {
        "model": model_id,
        "fine_tune_type": "lora",
        "batch_size": 1,
        "iters": n_rows,
        "grad_accumulation_steps": 1,
        "mask_prompt": True,
        "action_token_only": False,
        "preserve_row_order": True,
        "expected_schedule_sha256": schedule_sha256,
    }
    adapter_mismatches = {
        key: {"stored": adapter_config.get(key), "expected": value}
        for key, value in expected_adapter.items()
        if adapter_config.get(key) != value
    }
    if adapter_mismatches:
        raise ValueError(
            "checkpoint adapter config mismatch:"
            + json.dumps(
                adapter_mismatches,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    expected_mlx = {
        "grad_accumulation_steps": 1,
        "action_token_only": False,
        "preserve_row_order": True,
        "expected_schedule_sha256": schedule_sha256,
    }
    mlx_mismatches = {
        key: {"stored": mlx_config.get(key), "expected": value}
        for key, value in expected_mlx.items()
        if mlx_config.get(key) != value
    }
    if mlx_mismatches:
        raise ValueError(
            "checkpoint MLX config mismatch:"
            + json.dumps(
                mlx_mismatches,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    save_every = adapter_config.get("save_every")
    if (
        isinstance(save_every, bool)
        or not isinstance(save_every, int)
        or save_every <= 0
        or checkpoint_step % save_every != 0
    ):
        raise ValueError(
            "checkpoint step is not an immutable save point from the "
            "training config"
        )
    source_adapter_value = adapter_config.get("adapter_path")
    if not isinstance(source_adapter_value, str) or not source_adapter_value:
        raise ValueError("checkpoint adapter config has no source adapter_path")
    source_adapter_path = Path(source_adapter_value).expanduser().resolve()
    source_snapshot_path = (
        source_adapter_path
        / f"{checkpoint_step:07d}_adapters.safetensors"
    )
    source_snapshot_sha256 = file_sha256(source_snapshot_path)
    if source_snapshot_sha256 is None:
        raise ValueError(
            "checkpoint source snapshot does not exist: "
            f"{source_snapshot_path}"
        )
    if source_snapshot_sha256 != files["adapters.safetensors"]:
        raise ValueError(
            "checkpoint adapters.safetensors does not match its declared "
            f"step-{checkpoint_step} source snapshot"
        )
    source_adapter_config_path = source_adapter_path / "adapter_config.json"
    source_mlx_config_path = source_adapter_path / "mlx_lora_config.json"
    if file_sha256(source_adapter_config_path) != file_sha256(adapter_config_path):
        raise ValueError(
            "checkpoint adapter_config.json differs from its source training run"
        )
    if file_sha256(source_mlx_config_path) != file_sha256(mlx_config_path):
        raise ValueError(
            "checkpoint mlx_lora_config.json differs from its source training run"
        )
    source_training_report_path = (
        source_adapter_path / "native_mlx_data_report.json"
    )
    source_training_report_sha256 = file_sha256(source_training_report_path)
    if source_training_report_sha256 is None:
        raise ValueError(
            "checkpoint source training run has no native_mlx_data_report.json"
        )
    return {
        **provenance,
        "adapter_config_sha256": file_sha256(adapter_config_path),
        "mlx_lora_config_sha256": file_sha256(mlx_config_path),
        "training_contract": {
            "model": model_id,
            "batch_size": 1,
            "iters": n_rows,
            "grad_accumulation_steps": 1,
            "action_token_only": False,
            "preserve_row_order": True,
            "expected_schedule_sha256": schedule_sha256,
        },
        "snapshot": {
            "checkpoint_step": checkpoint_step,
            "source_adapter_path": str(source_adapter_path),
            "source_snapshot_path": str(source_snapshot_path),
            "source_snapshot_sha256": source_snapshot_sha256,
            "source_training_report_path": str(source_training_report_path),
            "source_training_report_sha256": (
                source_training_report_sha256
            ),
        },
    }


def _training_report_provenance(
    path: Path,
    *,
    dataset_sha256: str,
    schedule: dict[str, Any],
    expected_source_sha256: str,
) -> dict[str, Any]:
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_source_sha256:
        raise ValueError(
            "native MLX data report does not match the checkpoint's source "
            "training run"
        )
    report = _load_json(path, description="native MLX data report")
    n_rows = int(schedule["n_rows"])
    expected = {
        "dataset_sha256": dataset_sha256,
        "format": "pretokenized_action_mask",
        "loss_mask_mode": "action",
        "action_token_only": False,
        "preserve_row_order": True,
        "valid_fraction": 0.0,
        "n_input_records": n_rows,
        "n_kept_records": n_rows,
        "n_train_records": n_rows,
        "n_valid_records": 0,
        "retained_count_matches_expected": True,
        "skipped_record_counts": {},
        "tokenization_error_counts": {},
    }
    mismatches = {
        key: {"stored": report.get(key), "expected": value}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "native MLX data report mismatch:"
            + json.dumps(mismatches, separators=(",", ":"), sort_keys=True)
        )
    report_schedule = report.get("schedule")
    if not isinstance(report_schedule, dict):
        raise ValueError("native MLX data report has no schedule")
    expected_schedule = {
        "n_records": n_rows,
        "n_unique_source_identities": schedule["n_source_rows"],
        "steps_are_zero_based_contiguous_in_prepared_order": True,
        "ordered_step_source_sha256": schedule[
            "ordered_step_source_sha256"
        ],
    }
    schedule_mismatches = {
        key: {"stored": report_schedule.get(key), "expected": value}
        for key, value in expected_schedule.items()
        if report_schedule.get(key) != value
    }
    if schedule_mismatches:
        raise ValueError(
            "native MLX data report schedule mismatch:"
            + json.dumps(
                schedule_mismatches,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    report_counts = report.get("token_count_totals")
    if not isinstance(report_counts, dict):
        raise ValueError("native MLX data report has no token_count_totals")
    for key, value in schedule["token_count_totals"].items():
        if report_counts.get(key) != value:
            raise ValueError(
                f"native MLX data report token count mismatch: {key}"
            )
    return {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "dataset_sha256": dataset_sha256,
        "format": report["format"],
        "n_kept_records": report["n_kept_records"],
        "ordered_step_source_sha256": report_schedule[
            "ordered_step_source_sha256"
        ],
    }


def _runtime_prompt_renderer(tokenizer: Any):
    def render(messages: Sequence[dict[str, str]]) -> str:
        try:
            value = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            value = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(value, str):
            raise ValueError("runtime tokenizer returned a non-string prompt")
        return value

    return render


def _write_text_atomically(path: Path, contents: str) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite diagnostic output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link both publishes the fully-fsynced inode
            # atomically and fails if another process won the output-name race.
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError(
                f"refusing to overwrite diagnostic output: {path}"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-bf16",
    )
    parser.add_argument(
        "--expected-model-revision",
        required=True,
        help="Exact cached 40-character Hugging Face model revision.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
        help=(
            "Immutable saved update represented by adapters.safetensors "
            "(for example 1500). The training config's iters remains the full "
            "3,000-row schedule."
        ),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        required=True,
        help="native_mlx_data_report.json from the checkpoint's full training run.",
    )
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-training-report-sha256",
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists():
        raise ValueError(f"refusing to overwrite diagnostic output: {args.out}")
    for name, value in (
        ("--expected-dataset-sha256", args.expected_dataset_sha256),
        ("--expected-manifest-sha256", args.expected_manifest_sha256),
        (
            "--expected-training-report-sha256",
            args.expected_training_report_sha256,
        ),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    dataset_sha256 = file_sha256(args.dataset)
    manifest_sha256 = file_sha256(args.manifest)
    if dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError("dataset does not match --expected-dataset-sha256")
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("manifest does not match --expected-manifest-sha256")
    if (
        file_sha256(args.training_report)
        != args.expected_training_report_sha256
    ):
        raise ValueError(
            "training report does not match "
            "--expected-training-report-sha256"
        )
    rows = _load_jsonl(args.dataset)
    manifest = _load_json(args.manifest, description="dataset manifest")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("manifest dataset_sha256 disagrees with dataset")
    schedule = validate_exact_schedule(rows, manifest)
    if manifest.get("tokenizer_id") != args.model:
        raise ValueError("manifest tokenizer_id disagrees with --model")
    model_snapshot, model_provenance = _resolve_model_snapshot(
        args.model,
        expected_revision=args.expected_model_revision,
    )
    checkpoint = _checkpoint_provenance(
        args.checkpoint,
        model_id=args.model,
        schedule_sha256=schedule["ordered_step_source_sha256"],
        n_rows=len(rows),
        checkpoint_step=args.checkpoint_step,
    )
    training_report = _training_report_provenance(
        args.training_report,
        dataset_sha256=str(dataset_sha256),
        schedule=schedule,
        expected_source_sha256=checkpoint["snapshot"][
            "source_training_report_sha256"
        ],
    )

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    source_paths = {
        "cli": script_path,
        "diagnostic_core": (
            repo_root / "src/sts_ai/sft_loss_decomposition.py"
        ),
        "mask_core": repo_root / "src/sts_ai/train/sft_format.py",
        "candidate_model_loader": (
            repo_root / "src/sts_ai/teacher_action_eval.py"
        ),
        "paired_schedule_core": repo_root / "src/sts_ai/permutation_sft.py",
        "action_order_core": (
            repo_root / "src/sts_ai/action_order_diagnostic.py"
        ),
    }
    source_hashes = {
        name: file_sha256(path)
        for name, path in source_paths.items()
    }
    if any(digest is None for digest in source_hashes.values()):
        raise ValueError("one or more diagnostic source files could not be hashed")
    runtime_versions = _runtime_versions()

    scorer = MlxTeacherForcedLossScorer(
        str(model_snapshot),
        adapter_path=str(args.checkpoint.resolve()),
    )
    try:
        tokenizer = scorer.runtime_tokenizer()
        runtime_tokenizer_provenance = _runtime_tokenizer_provenance(tokenizer)
        runtime_chat_template_hash = chat_template_probe_hash(
            tokenizer,
            enable_thinking=False,
        )
        if runtime_chat_template_hash != manifest["chat_template_hash"]:
            raise ValueError(
                "runtime tokenizer chat-template hash disagrees with manifest"
            )
        render_prompt = _runtime_prompt_renderer(tokenizer)
        for row in rows:
            validate_prompt_round_trip(row, render_prompt)
        report = build_loss_decomposition_report(
            rows,
            manifest,
            scorer,
            provenance={
                "dataset": {
                    "path": str(args.dataset.resolve()),
                    "sha256": dataset_sha256,
                },
                "manifest": {
                    "path": str(args.manifest.resolve()),
                    "sha256": manifest_sha256,
                    "dataset_sha256": manifest["dataset_sha256"],
                    "tokenizer_id": manifest["tokenizer_id"],
                    "chat_template_hash": manifest["chat_template_hash"],
                    "augmentation": manifest["augmentation"],
                },
                "model_id": args.model,
                "model": model_provenance,
                "checkpoint": checkpoint,
                "native_mlx_data_report": training_report,
                "backend": "mlx",
                "runtime_chat_template_hash": runtime_chat_template_hash,
                "runtime_tokenizer": runtime_tokenizer_provenance,
                "runtime_versions": runtime_versions,
                "git_head": _git_head(),
                "source_sha256": source_hashes,
                "numeric_contract": {
                    "selected_token_log_softmax_dtype": "float32",
                    "per_token_transfer": "Python float",
                    "report_summation": "math.fsum",
                },
            },
        )
    finally:
        scorer.clear()
        del scorer
        gc.collect()

    if dataset_sha256 != file_sha256(args.dataset):
        raise RuntimeError("dataset changed while it was scored")
    if manifest_sha256 != file_sha256(args.manifest):
        raise RuntimeError("manifest changed while it was scored")
    if training_report["sha256"] != file_sha256(args.training_report):
        raise RuntimeError("training report changed while it was scored")
    if checkpoint != _checkpoint_provenance(
        args.checkpoint,
        model_id=args.model,
        schedule_sha256=schedule["ordered_step_source_sha256"],
        n_rows=len(rows),
        checkpoint_step=args.checkpoint_step,
    ):
        raise RuntimeError("checkpoint changed while it was scored")
    if source_hashes != {
        name: file_sha256(path)
        for name, path in source_paths.items()
    }:
        raise RuntimeError("diagnostic source changed while it was scored")
    _, current_model_provenance = _resolve_model_snapshot(
        args.model,
        expected_revision=args.expected_model_revision,
    )
    if current_model_provenance != model_provenance:
        raise RuntimeError("base model snapshot changed while it was scored")
    if _runtime_versions() != runtime_versions:
        raise RuntimeError("runtime package versions changed while it was scored")

    rendered = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_text_atomically(args.out, rendered)
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "output_sha256": file_sha256(args.out),
                "arm": report["schedule"]["arm"],
                "n_scored_rows": report["n_scored_rows"],
                "mean_format_token_nll": report["overall"]["format"][
                    "mean_token_nll"
                ],
                "mean_action_token_nll": report["overall"]["action"][
                    "mean_token_nll"
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
