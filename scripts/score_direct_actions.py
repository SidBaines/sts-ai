#!/usr/bin/env python
"""Score compact teacher actions at the direct policy token under MLX."""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Sequence

from sts_ai.action_order_diagnostic import (
    parse_teacher_menu,
    validate_prompt_round_trip,
)
from sts_ai.direct_action_eval import (
    MlxDirectActionScorer,
    build_direct_action_report,
)
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import ACTION_ONLY_OUTPUT_CONTRACT
from sts_ai.train.sft_format import chat_template_probe_hash


REPORT_KIND = "comp015_direct_action_token_diagnostic"
REPORT_VERSION = 1
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _runtime_package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"required runtime package is missing: {package}") from exc
    return result


def _resolve_model_snapshot(
    model_id: str,
    *,
    expected_revision: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one exact cached HF revision and record its content identities."""

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
        expected_blob_sha256 = (
            resolved.name if _is_sha256(resolved.name) else None
        )
        blob_sha256 = file_sha256(path)
        if not _is_sha256(blob_sha256):
            raise ValueError(f"model weight could not be identified: {path.name}")
        if (
            expected_blob_sha256 is not None
            and blob_sha256 != expected_blob_sha256
        ):
            raise ValueError(f"model weight CAS digest mismatch: {path.name}")
        files[path.name] = {
            "sha256": blob_sha256,
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
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size")),
        "backend_tokenizer_sha256": hashlib.sha256(
            backend_json.encode("utf-8")
        ).hexdigest(),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError("teacher dataset is empty")
    return rows


def _manifest_provenance(
    path: Path,
    *,
    dataset_sha256: str,
    n_rows: int,
) -> dict[str, Any]:
    value = _load_json(path, description="teacher manifest")
    expected = {
        "kind": "search_teacher_sft",
        "version": 3,
        "loss_mask_mode": "action",
        "output_contract": ACTION_ONLY_OUTPUT_CONTRACT,
        "enable_thinking": False,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "dataset_sha256": dataset_sha256,
        "n_examples": n_rows,
    }
    mismatches = {
        key: {"stored": value.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            "teacher manifest disagrees with the frozen dataset contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    source_labels = value.get("source_labels")
    if not isinstance(source_labels, dict):
        raise ValueError("teacher manifest has no source_labels provenance")
    for key in ("sha256", "manifest_sha256"):
        if not _is_sha256(source_labels.get(key)):
            raise ValueError(f"teacher manifest source_labels.{key} is invalid")
    tokenizer_id = value.get("tokenizer_id")
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("teacher manifest tokenizer_id is invalid")
    chat_template_hash = value.get("chat_template_hash")
    if (
        not isinstance(chat_template_hash, str)
        or len(chat_template_hash) != 16
        or any(
            character not in "0123456789abcdef"
            for character in chat_template_hash
        )
    ):
        raise ValueError("teacher manifest chat_template_hash is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "contents": value,
    }


def _load_teacher_inputs(
    dataset_path: Path,
    manifest_path: Path,
    *,
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
    expected_row_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    dataset_sha256 = file_sha256(dataset_path)
    if dataset_sha256 is None:
        raise ValueError(f"teacher dataset does not exist: {dataset_path}")
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError("teacher dataset does not match --expected-dataset-sha256")
    rows = _load_jsonl(dataset_path)
    if len(rows) != expected_row_count:
        raise ValueError("teacher dataset does not match --expected-row-count")
    if dataset_sha256 != file_sha256(dataset_path):
        raise RuntimeError("teacher dataset changed while it was loaded")
    manifest = _manifest_provenance(
        manifest_path,
        dataset_sha256=dataset_sha256,
        n_rows=len(rows),
    )
    if manifest["sha256"] != expected_manifest_sha256:
        raise ValueError("teacher manifest does not match --expected-manifest-sha256")

    public_hashes: set[str] = set()
    for row_index, row in enumerate(rows):
        if row.get("teacher_selection_rule") != AGGREGATED_ROOT_VISITS:
            raise ValueError(
                f"teacher dataset row {row_index} has invalid teacher_selection_rule"
            )
        if row.get("teacher_privilege") != TEACHER_PRIVILEGE:
            raise ValueError(
                f"teacher dataset row {row_index} has invalid teacher_privilege"
            )
        parsed = parse_teacher_menu(row)
        if parsed.source_public_state_hash in public_hashes:
            raise ValueError("teacher dataset public_state_hash is not unique")
        public_hashes.add(parsed.source_public_state_hash)
    return rows, manifest, dataset_sha256


def _parse_label_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        label, separator, path_text = value.partition("=")
        if (
            separator != "="
            or not _LABEL_RE.fullmatch(label)
            or not path_text
            or label in parsed
        ):
            raise ValueError(
                f"{option} values must be unique LABEL=PATH pairs; got {value!r}"
            )
        parsed[label] = Path(path_text)
    if not parsed:
        raise ValueError(f"{option} must be supplied at least once")
    return parsed


def _parse_label_hashes(values: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        label, separator, digest = value.partition("=")
        if (
            separator != "="
            or not _LABEL_RE.fullmatch(label)
            or not _is_sha256(digest)
            or label in parsed
        ):
            raise ValueError(
                f"{option} values must be unique LABEL=SHA256 pairs; got {value!r}"
            )
        parsed[label] = digest
    if not parsed:
        raise ValueError(f"{option} must be supplied at least once")
    return parsed


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(
            "checkpoint paths must be adapter directories containing "
            "adapters.safetensors and adapter_config.json"
        )
    provenance = adapter_provenance(path)
    if provenance is None:
        raise ValueError("checkpoint provenance could not be computed")
    files = provenance.get("files")
    if not isinstance(files, dict) or "adapter_config.json" not in files:
        raise ValueError("checkpoint directory has no adapter_config.json")
    adapter_config = _load_json(
        path / "adapter_config.json",
        description="checkpoint adapter config",
    )
    declared_model_id = adapter_config.get("model")
    if not isinstance(declared_model_id, str) or not declared_model_id:
        raise ValueError("checkpoint adapter_config.json has no model identity")
    mlx_config = path / "mlx_lora_config.json"
    if not mlx_config.is_file():
        raise ValueError("checkpoint directory has no mlx_lora_config.json")
    mlx_config_sha256 = file_sha256(mlx_config)
    if not _is_sha256(mlx_config_sha256):
        raise ValueError("checkpoint mlx_lora_config.json could not be hashed")
    return {
        **provenance,
        "declared_model_id": declared_model_id,
        "mlx_lora_config_sha256": mlx_config_sha256,
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


def _runtime_tokenizer(scorer: MlxDirectActionScorer) -> Any:
    """Return the tokenizer without retaining a second live model reference."""

    return scorer._load()[1]


def _drop_scorer_references(scorer: MlxDirectActionScorer) -> None:
    scorer._model = None
    scorer._tokenizer = None
    scorer._output_weight_f32 = None


def _clear_scorer(scorer: MlxDirectActionScorer) -> None:
    _drop_scorer_references(scorer)
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_atomically(path: Path, contents: str) -> None:
    """Create ``path`` from a same-directory temporary file, without overwrite."""

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
            # A same-filesystem hard link publishes the already-fsynced complete
            # temporary inode atomically and, unlike os.replace, cannot overwrite
            # a destination created by another process after our initial check.
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError(
                f"refusing to overwrite diagnostic output: {path}"
            ) from exc
        _fsync_directory(path.parent)
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _dtype_name(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def _validate_float32_report(report: dict[str, Any]) -> None:
    """Require evidence that projection, soft-cap, and normalization were fp32."""

    for key in ("model_logits_dtypes", "scoring_dtypes"):
        values = report.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or _dtype_name(value) != "float32"
                for value in values
            )
        ):
            raise ValueError(f"direct action report {key} is not float32")
    modes = report.get("output_projection_modes")
    if (
        not isinstance(modes, list)
        or not modes
        or any(
            not isinstance(mode, str)
            or not mode.startswith(
                "tied_embedding_full_vocabulary_float32_projection"
            )
            for mode in modes
        )
    ):
        raise ValueError(
            "direct action report did not use the strict float32 output projection"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-bf16")
    parser.add_argument(
        "--expected-model-revision",
        required=True,
        help="Exact cached 40-character Hugging Face model revision.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Checkpoint adapter directory. Repeat for each checkpoint.",
    )
    parser.add_argument(
        "--expected-checkpoint-identity",
        action="append",
        default=[],
        metavar="LABEL=SHA256",
        help="Expected adapter identity_sha256. Repeat for each checkpoint.",
    )
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-row-count", type=int, required=True)
    parser.add_argument(
        "--cohort-role",
        choices=("train", "development"),
        required=True,
        help="The embargoed final cohort is deliberately not an accepted role.",
    )
    parser.add_argument(
        "--cyclic-rotations",
        action="store_true",
        help="Score every nonzero cyclic rotation (development only).",
    )
    parser.add_argument(
        "--greedy-max-tokens",
        type=int,
        default=32,
        help="Maximum tokens for unpermuted deterministic JSON generation.",
    )
    parser.add_argument(
        "--no-greedy-json",
        action="store_true",
        help="Disable the separate deterministic canonical-JSON check.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite diagnostic output: {args.out}")
    if args.expected_row_count <= 0:
        raise ValueError("--expected-row-count must be positive")
    if args.greedy_max_tokens <= 0:
        raise ValueError("--greedy-max-tokens must be positive")
    if args.cyclic_rotations and args.cohort_role != "development":
        raise ValueError("--cyclic-rotations is restricted to the development cohort")
    for name, value in (
        ("--expected-dataset-sha256", args.expected_dataset_sha256),
        ("--expected-manifest-sha256", args.expected_manifest_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    checkpoints = _parse_label_paths(args.checkpoint, option="--checkpoint")
    expected_checkpoint_identities = _parse_label_hashes(
        args.expected_checkpoint_identity,
        option="--expected-checkpoint-identity",
    )
    if checkpoints.keys() != expected_checkpoint_identities.keys():
        raise ValueError(
            "checkpoint and expected-checkpoint-identity labels must match exactly"
        )
    resolved_checkpoints = [path.resolve() for path in checkpoints.values()]
    if len(set(resolved_checkpoints)) != len(resolved_checkpoints):
        raise ValueError("checkpoint paths must be unique across labels")
    if (
        len(set(expected_checkpoint_identities.values()))
        != len(expected_checkpoint_identities)
    ):
        raise ValueError("expected checkpoint identities must be unique")

    rows, manifest, dataset_sha256 = _load_teacher_inputs(
        args.dataset,
        args.manifest,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_row_count=args.expected_row_count,
    )
    manifest_contents = manifest["contents"]
    if manifest_contents["tokenizer_id"] != args.model:
        raise ValueError("teacher manifest tokenizer_id disagrees with --model")
    model_snapshot, model_provenance = _resolve_model_snapshot(
        args.model,
        expected_revision=args.expected_model_revision,
    )
    runtime_package_versions = _runtime_package_versions()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    source_paths = {
        "cli": script_path,
        "direct_action_core": repo_root / "src/sts_ai/direct_action_eval.py",
        "rotation_core": repo_root / "src/sts_ai/action_order_diagnostic.py",
        "teacher_evaluator": repo_root / "src/sts_ai/teacher_action_eval.py",
        "teacher_contract": repo_root / "src/sts_ai/teacher.py",
        "provenance_core": repo_root / "src/sts_ai/provenance.py",
        "sft_format": repo_root / "src/sts_ai/train/sft_format.py",
    }
    source_hashes = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    if any(digest is None for digest in source_hashes.values()):
        raise ValueError("one or more diagnostic source files could not be hashed")

    checkpoint_reports: dict[str, dict[str, Any]] = {}
    durable_checkpoint_provenance: dict[str, dict[str, Any]] = {}
    durable_tokenizer_provenance: dict[str, Any] | None = None
    for label in sorted(checkpoints):
        checkpoint = _checkpoint_provenance(checkpoints[label])
        if checkpoint["declared_model_id"] != args.model:
            raise ValueError(
                f"{label} checkpoint base model disagrees with --model"
            )
        if (
            checkpoint.get("identity_sha256")
            != expected_checkpoint_identities[label]
        ):
            raise ValueError(
                f"{label} checkpoint does not match its expected identity"
            )
        scorer = MlxDirectActionScorer(
            str(model_snapshot),
            adapter_path=str(checkpoints[label].resolve()),
        )
        try:
            tokenizer = _runtime_tokenizer(scorer)
            tokenizer_provenance = _runtime_tokenizer_provenance(tokenizer)
            if durable_tokenizer_provenance is None:
                durable_tokenizer_provenance = tokenizer_provenance
            elif tokenizer_provenance != durable_tokenizer_provenance:
                raise ValueError("runtime tokenizer changed across checkpoints")
            runtime_chat_template_hash = chat_template_probe_hash(
                tokenizer,
                enable_thinking=False,
            )
            if runtime_chat_template_hash != manifest_contents["chat_template_hash"]:
                raise ValueError(
                    "runtime tokenizer chat-template hash disagrees with manifest"
                )
            prompt_renderer = _runtime_prompt_renderer(tokenizer)
            for row_index, row in enumerate(rows):
                try:
                    validate_prompt_round_trip(row, prompt_renderer)
                except ValueError as exc:
                    raise ValueError(
                        f"teacher row {row_index} runtime prompt mismatch: {exc}"
                    ) from exc
            checkpoint_report = build_direct_action_report(
                rows,
                scorer,
                greedy_max_tokens=(
                    None if args.no_greedy_json else args.greedy_max_tokens
                ),
                include_cyclic_rotations=args.cyclic_rotations,
                prompt_renderer=(
                    prompt_renderer if args.cyclic_rotations else None
                ),
                provenance={
                    "checkpoint": checkpoint,
                    "runtime_chat_template_hash": runtime_chat_template_hash,
                    "runtime_tokenizer": tokenizer_provenance,
                },
            )
            _validate_float32_report(checkpoint_report)
            checkpoint_reports[label] = checkpoint_report
        finally:
            _clear_scorer(scorer)
        durable_checkpoint_provenance[label] = checkpoint

        if checkpoint != _checkpoint_provenance(checkpoints[label]):
            raise RuntimeError(f"{label} checkpoint changed while it was scored")
        if dataset_sha256 != file_sha256(args.dataset):
            raise RuntimeError("teacher dataset changed while it was scored")
        if manifest["sha256"] != file_sha256(args.manifest):
            raise RuntimeError("teacher manifest changed while it was scored")
        if source_hashes != {
            name: file_sha256(path) for name, path in source_paths.items()
        }:
            raise RuntimeError("diagnostic source changed while it was scored")
        _, current_model_provenance = _resolve_model_snapshot(
            args.model,
            expected_revision=args.expected_model_revision,
        )
        if current_model_provenance != model_provenance:
            raise RuntimeError("base model snapshot changed while it was scored")
        if _runtime_package_versions() != runtime_package_versions:
            raise RuntimeError("runtime package versions changed while scoring")

    if durable_tokenizer_provenance is None:
        raise RuntimeError("runtime tokenizer provenance was not collected")
    report = {
        "kind": REPORT_KIND,
        "version": REPORT_VERSION,
        "cohort_role": args.cohort_role,
        "cyclic_rotations_included": args.cyclic_rotations,
        "checkpoint_labels": sorted(checkpoint_reports),
        "n_source_rows": len(rows),
        "evaluation": {
            "greedy_json_enabled": not args.no_greedy_json,
            "greedy_max_tokens": (
                None if args.no_greedy_json else args.greedy_max_tokens
            ),
            "direct_output_projection": (
                "strict tied-embedding full-vocabulary float32"
            ),
            "cyclic_rotations": (
                "all nonzero right rotations"
                if args.cyclic_rotations
                else "disabled"
            ),
        },
        "checkpoints": checkpoint_reports,
        "provenance": {
            "dataset": {
                "path": str(args.dataset.resolve()),
                "sha256": dataset_sha256,
            },
            "manifest": {
                "path": manifest["path"],
                "sha256": manifest["sha256"],
                "kind": manifest_contents["kind"],
                "version": manifest_contents["version"],
                "tokenizer_id": manifest_contents["tokenizer_id"],
                "chat_template_hash": manifest_contents["chat_template_hash"],
                "n_examples": manifest_contents["n_examples"],
                "dataset_sha256": manifest_contents["dataset_sha256"],
                "source_labels": manifest_contents["source_labels"],
            },
            "model_id": args.model,
            "model": model_provenance,
            "backend": "mlx",
            "runtime_package_versions": runtime_package_versions,
            "runtime_tokenizer": durable_tokenizer_provenance,
            "git_head": _git_head(),
            "source_sha256": source_hashes,
            "checkpoints": durable_checkpoint_provenance,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_text_atomically(args.out, rendered)
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "output_sha256": file_sha256(args.out),
                "cohort_role": report["cohort_role"],
                "checkpoint_labels": report["checkpoint_labels"],
                "n_source_rows": report["n_source_rows"],
                "cyclic_rotations_included": report[
                    "cyclic_rotations_included"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
