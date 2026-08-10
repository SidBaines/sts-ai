#!/usr/bin/env python
"""Compare answer-only with visible-reasoning JSON on frozen teacher states.

Native model thinking is disabled in both arms.  The script is deliberately
fail-closed: it requires content-addressed teacher inputs, verifies their source
artifacts, rejects the embargoed final cohort, loads an exact local model
snapshot, releases each model before loading the next label, and atomically
refuses to overwrite a report.
"""
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

from sts_ai.action_order_diagnostic import parse_teacher_menu
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import ACTION_ONLY_OUTPUT_CONTRACT
from sts_ai.visible_reasoning_eval import (
    MlxStaticGenerator,
    build_model_comparison,
    build_report,
)


_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DEFAULT_EMBARGO = Path(
    "configs/competence/nob_fresh_final_cohort_v1.json"
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    result: dict[str, str | None] = {}
    for name in ("mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


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
        raise ValueError(f"could not read teacher dataset {path}: {exc}") from exc
    if not rows:
        raise ValueError("teacher dataset is empty")
    return rows


def _resolve_recorded_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _verified_file(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{description} has an invalid expected SHA-256")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{description} hash mismatch: stored={expected_sha256} "
            f"observed={observed} path={path}"
        )
    return {
        "path": str(path),
        "sha256": observed,
    }


def _source_labels_provenance(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    source = manifest.get("source_labels")
    if not isinstance(source, dict):
        raise ValueError("teacher manifest has no source_labels object")
    required = {
        "path": "sha256",
        "manifest_path": "manifest_sha256",
        "source_manifest": "source_manifest_sha256",
    }
    verified: dict[str, Any] = {}
    for path_key, hash_key in required.items():
        path_value = source.get(path_key)
        digest = source.get(hash_key)
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"teacher manifest source_labels.{path_key} is invalid")
        if not _is_sha256(digest):
            raise ValueError(f"teacher manifest source_labels.{hash_key} is invalid")
        path = _resolve_recorded_path(path_value, repo_root=repo_root)
        verified[path_key] = _verified_file(
            path,
            expected_sha256=digest,
            description=f"source_labels.{path_key}",
        )
    return {
        "declared": source,
        "verified_files": verified,
    }


def _manifest_provenance(
    path: Path,
    *,
    dataset_sha256: str,
    n_rows: int,
    repo_root: Path,
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
            "teacher manifest disagrees with strict v3 contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    tokenizer_id = value.get("tokenizer_id")
    chat_template_hash = value.get("chat_template_hash")
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("teacher manifest tokenizer_id is invalid")
    if (
        not isinstance(chat_template_hash, str)
        or len(chat_template_hash) != 16
        or any(character not in "0123456789abcdef" for character in chat_template_hash)
    ):
        raise ValueError("teacher manifest chat_template_hash is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "contents": value,
        "source_labels": _source_labels_provenance(
            value,
            repo_root=repo_root,
        ),
    }


def _embargo_provenance(
    path: Path,
    *,
    rows: Sequence[dict[str, Any]],
    dataset_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Reject any row or source artifact tied to the final holdout cohort."""

    config = _load_json(path, description="fresh-final embargo config")
    embargo = config.get("embargo")
    split = config.get("split")
    manifests = config.get("manifests")
    if (
        config.get("cohort_id") != "nob_fresh_final_cohort_v1"
        or not isinstance(embargo, dict)
        or embargo.get("teacher_action_scoring_on_holdout") is not False
        or not isinstance(split, dict)
        or not isinstance(manifests, dict)
    ):
        raise ValueError("fresh-final embargo config contract is invalid")
    holdout_ids = split.get("holdout_window_ids")
    if (
        not isinstance(holdout_ids, list)
        or not holdout_ids
        or any(not isinstance(value, str) or not value for value in holdout_ids)
        or len(set(holdout_ids)) != len(holdout_ids)
    ):
        raise ValueError("fresh-final embargo holdout_window_ids are invalid")
    holdout_set = set(holdout_ids)
    row_overlap = sorted(
        {
            str(row.get("window_id"))
            for row in rows
            if row.get("window_id") in holdout_set
        }
    )
    if row_overlap:
        raise ValueError(
            "teacher dataset overlaps embargoed final holdout windows: "
            + ", ".join(row_overlap)
        )

    blocked_paths: set[Path] = set()
    blocked_hashes: set[str] = set()
    for name in ("source", "validated_primary", "validated_repeat"):
        item = manifests.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"fresh-final embargo manifests.{name} is invalid")
        path_value = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path_value, str) or not _is_sha256(digest):
            raise ValueError(f"fresh-final embargo manifests.{name} is invalid")
        blocked_paths.add(
            _resolve_recorded_path(path_value, repo_root=repo_root)
        )
        blocked_hashes.add(digest)

    source_labels = manifest["contents"]["source_labels"]
    candidate_paths = {
        dataset_path.resolve(),
        manifest_path.resolve(),
        *(
            _resolve_recorded_path(source_labels[key], repo_root=repo_root)
            for key in ("path", "manifest_path", "source_manifest")
        ),
    }
    candidate_hashes = {
        manifest["contents"]["dataset_sha256"],
        manifest["sha256"],
        *(
            source_labels[key]
            for key in ("sha256", "manifest_sha256", "source_manifest_sha256")
        ),
    }
    if candidate_paths & blocked_paths or candidate_hashes & blocked_hashes:
        raise ValueError(
            "teacher inputs reference an embargoed fresh-final cohort artifact"
        )
    # This textual guard catches a newly-derived final artifact whose hash/path
    # has not yet been added to the frozen config.  Current development inputs
    # have no such component.
    if any(
        "fresh_final" in str(candidate).lower()
        or "fresh-final" in str(candidate).lower()
        for candidate in candidate_paths
    ):
        raise ValueError("teacher input path appears to be a fresh-final artifact")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "cohort_id": config["cohort_id"],
        "n_embargoed_holdout_windows": len(holdout_set),
        "overlap_count": 0,
        "final_cohort_queried": False,
    }


def _load_teacher_inputs(
    dataset_path: Path,
    manifest_path: Path,
    *,
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
    expected_row_count: int,
    embargo_path: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_dataset_sha256 = file_sha256(dataset_path)
    if observed_dataset_sha256 != expected_dataset_sha256:
        raise ValueError("teacher dataset does not match --expected-dataset-sha256")
    rows = _load_jsonl(dataset_path)
    if len(rows) != expected_row_count:
        raise ValueError("teacher dataset does not match --expected-row-count")
    if file_sha256(dataset_path) != observed_dataset_sha256:
        raise RuntimeError("teacher dataset changed while it was loaded")
    manifest = _manifest_provenance(
        manifest_path,
        dataset_sha256=observed_dataset_sha256,
        n_rows=len(rows),
        repo_root=repo_root,
    )
    if manifest["sha256"] != expected_manifest_sha256:
        raise ValueError("teacher manifest does not match --expected-manifest-sha256")

    public_hashes: set[str] = set()
    for row_index, row in enumerate(rows):
        if row.get("teacher_selection_rule") != AGGREGATED_ROOT_VISITS:
            raise ValueError(
                f"teacher row {row_index} has invalid teacher_selection_rule"
            )
        if row.get("teacher_privilege") != TEACHER_PRIVILEGE:
            raise ValueError(
                f"teacher row {row_index} has invalid teacher_privilege"
            )
        parsed = parse_teacher_menu(row)
        if parsed.source_public_state_hash in public_hashes:
            raise ValueError("teacher public_state_hash is not unique")
        public_hashes.add(parsed.source_public_state_hash)

    embargo = _embargo_provenance(
        embargo_path,
        rows=rows,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        manifest=manifest,
        repo_root=repo_root,
    )
    return rows, {
        "path": str(dataset_path.resolve()),
        "sha256": observed_dataset_sha256,
        "n_rows": len(rows),
        "manifest": manifest,
        "embargo_check": embargo,
    }


def _parse_label_paths(
    values: Sequence[str],
    *,
    option: str,
) -> dict[str, Path]:
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
        parsed[label] = Path(path_text).expanduser().resolve()
    return parsed


def _parse_label_hashes(
    values: Sequence[str],
    *,
    option: str,
) -> dict[str, str]:
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
    return parsed


def _directory_provenance(path: Path, *, description: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{description} is not a directory: {resolved}")
    files: dict[str, dict[str, Any]] = {}
    for candidate in sorted(resolved.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(resolved).as_posix()
        digest = file_sha256(candidate)
        if digest is None:
            raise ValueError(f"could not hash {description} file: {candidate}")
        files[relative] = {
            "sha256": digest,
            "size_bytes": candidate.stat().st_size,
        }
    if not files:
        raise ValueError(f"{description} contains no files")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(resolved),
        "n_files": len(files),
        "files": files,
        "identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _resolve_model_source(model_id: str) -> dict[str, Any]:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        resolution = "local_path"
    else:
        try:
            from huggingface_hub import snapshot_download

            resolved = Path(
                snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,
                )
            ).resolve()
        except Exception as exc:  # noqa: BLE001 - normalize offline-cache failure
            raise ValueError(
                f"model {model_id!r} is not available in the local Hugging Face cache"
            ) from exc
        resolution = "huggingface_local_snapshot"
    provenance = _directory_provenance(
        resolved,
        description="model source snapshot",
    )
    if not any(name.endswith(".safetensors") for name in provenance["files"]):
        raise ValueError("model source snapshot contains no safetensors weights")
    return {
        "model_id": model_id,
        "resolution": resolution,
        **provenance,
    }


def _adapter_source(path: Path) -> dict[str, Any]:
    value = adapter_provenance(path)
    if not isinstance(value, dict):
        raise ValueError(f"adapter provenance could not be computed: {path}")
    files = value.get("files")
    if not isinstance(files, dict) or {
        "adapters.safetensors",
        "adapter_config.json",
    } - set(files):
        raise ValueError(
            "adapter must contain adapters.safetensors and adapter_config.json"
        )
    for optional in ("mlx_lora_config.json", "native_mlx_data_report.json"):
        digest = file_sha256(path / optional)
        if digest is not None:
            files[optional] = digest
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(path),
        "files": files,
        "identity_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _assert_file_unchanged(path: str, expected_sha256: str, description: str) -> None:
    if file_sha256(Path(path)) != expected_sha256:
        raise RuntimeError(f"{description} changed during evaluation")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_atomically(path: Path, contents: str) -> None:
    """Create ``path`` from a same-directory temporary file, without overwrite."""

    if path.exists():
        raise ValueError(f"refusing to overwrite comparison output: {path}")
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
            # Publish the already-fsynced inode atomically.  A same-filesystem
            # hard link cannot overwrite a destination created after the
            # initial exists() check, unlike os.replace().
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError(
                f"refusing to overwrite comparison output: {path}"
            ) from exc
        _fsync_directory(path.parent)
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-row-count", type=int, required=True)
    parser.add_argument(
        "--embargo-config",
        type=Path,
        default=_DEFAULT_EMBARGO,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--expected-model-identity-sha256",
        required=True,
        help=(
            "Predeclared content hash of the resolved local model snapshot. "
            "The exact snapshot path is loaded and checked again before report "
            "publication."
        ),
    )
    parser.add_argument("--base-label", default="base")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--expected-adapter-identity",
        action="append",
        default=[],
        metavar="LABEL=SHA256",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="MLX sampler nucleus threshold; 0 disables top-p filtering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="MLX sampler top-k; 0 disables top-k filtering.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise ValueError(f"refusing to overwrite comparison output: {args.out}")
    if not _LABEL_RE.fullmatch(args.base_label):
        raise ValueError("--base-label is invalid")
    if not _is_sha256(args.expected_dataset_sha256):
        raise ValueError("--expected-dataset-sha256 is invalid")
    if not _is_sha256(args.expected_manifest_sha256):
        raise ValueError("--expected-manifest-sha256 is invalid")
    if not _is_sha256(args.expected_model_identity_sha256):
        raise ValueError("--expected-model-identity-sha256 is invalid")

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    embargo_path = args.embargo_config.expanduser().resolve()
    rows, dataset_provenance = _load_teacher_inputs(
        dataset_path,
        manifest_path,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_row_count=args.expected_row_count,
        embargo_path=embargo_path,
        repo_root=repo_root,
    )

    adapters = _parse_label_paths(args.adapter, option="--adapter")
    if args.base_label in adapters:
        raise ValueError("--base-label duplicates an --adapter label")
    expected_adapters = _parse_label_hashes(
        args.expected_adapter_identity,
        option="--expected-adapter-identity",
    )
    if set(expected_adapters) != set(adapters):
        raise ValueError(
            "--expected-adapter-identity labels must exactly match --adapter labels"
        )
    adapter_sources = {
        label: _adapter_source(path)
        for label, path in adapters.items()
    }
    for label, source in adapter_sources.items():
        if source["identity_sha256"] != expected_adapters[label]:
            raise ValueError(
                f"adapter {label!r} does not match its expected identity"
            )

    model_source = _resolve_model_source(args.model_id)
    if (
        model_source["identity_sha256"]
        != args.expected_model_identity_sha256
    ):
        raise ValueError("model snapshot does not match its expected identity")

    source_provenance = {
        "git_head": _git_head(),
        "runtime_versions": _runtime_versions(),
        "files": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "evaluator_module": {
                "path": str(
                    (repo_root / "src/sts_ai/visible_reasoning_eval.py").resolve()
                ),
                "sha256": file_sha256(
                    repo_root / "src/sts_ai/visible_reasoning_eval.py"
                ),
            },
            "prompting_module": {
                "path": str((repo_root / "src/sts_ai/prompting.py").resolve()),
                "sha256": file_sha256(repo_root / "src/sts_ai/prompting.py"),
            },
            "parser_module": {
                "path": str((repo_root / "src/sts_ai/agents.py").resolve()),
                "sha256": file_sha256(repo_root / "src/sts_ai/agents.py"),
            },
            "menu_parser_module": {
                "path": str(
                    (repo_root / "src/sts_ai/action_order_diagnostic.py").resolve()
                ),
                "sha256": file_sha256(
                    repo_root / "src/sts_ai/action_order_diagnostic.py"
                ),
            },
        },
    }
    generation_settings = {
        "backend": "mlx",
        "decoding": "greedy" if args.temperature == 0 else "controlled_sampled",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "master_seed": args.seed,
        "samples_per_row": args.samples_per_row,
        "paired_seed_rule": (
            "master_seed + row_index * samples_per_row + sample_index; "
            "same seed reset before action_only and reasoning_action"
        ),
        "max_retries": 0,
        "native_thinking_enabled": False,
    }

    labels: list[tuple[str, Path | None]] = [
        (args.base_label, None),
        *adapters.items(),
    ]
    comparisons: list[dict[str, Any]] = []
    for label, adapter_path in labels:
        generator: MlxStaticGenerator | None = None
        try:
            generator = MlxStaticGenerator(
                model_source["path"],
                adapter_path=(
                    str(adapter_path) if adapter_path is not None else None
                ),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            tokenizer = generator.tokenizer_provenance()
            expected_template_hash = dataset_provenance["manifest"]["contents"][
                "chat_template_hash"
            ]
            if (
                tokenizer["chat_template_probe_hash_enable_thinking_false"]
                != expected_template_hash
            ):
                raise ValueError(
                    f"runtime tokenizer mismatch for model label {label!r}"
                )
            comparisons.append(
                build_model_comparison(
                    rows,
                    generator,
                    model_label=label,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    samples_per_row=args.samples_per_row,
                    provenance={
                        "model_source": model_source,
                        "adapter": (
                            adapter_sources[label] if adapter_path is not None else None
                        ),
                        "tokenizer": tokenizer,
                    },
                )
            )
        finally:
            if generator is not None:
                generator.close()
            generator = None
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:
                pass

    # Recompute all immutable input identities before publishing.  Cached file
    # hashes make this cheap when metadata is unchanged.
    if _directory_provenance(
        Path(model_source["path"]),
        description="model source snapshot",
    )["identity_sha256"] != model_source["identity_sha256"]:
        raise RuntimeError("model source snapshot changed during evaluation")
    for label, path in adapters.items():
        if _adapter_source(path) != adapter_sources[label]:
            raise RuntimeError(f"adapter {label!r} changed during evaluation")
    _assert_file_unchanged(
        dataset_provenance["path"],
        dataset_provenance["sha256"],
        "teacher dataset",
    )
    _assert_file_unchanged(
        dataset_provenance["manifest"]["path"],
        dataset_provenance["manifest"]["sha256"],
        "teacher manifest",
    )
    for key, value in dataset_provenance["manifest"]["source_labels"][
        "verified_files"
    ].items():
        _assert_file_unchanged(
            value["path"],
            value["sha256"],
            f"source_labels.{key}",
        )
    _assert_file_unchanged(
        dataset_provenance["embargo_check"]["path"],
        dataset_provenance["embargo_check"]["sha256"],
        "fresh-final embargo config",
    )

    report = build_report(
        comparisons,
        dataset_provenance=dataset_provenance,
        generation_settings=generation_settings,
        source_provenance=source_provenance,
    )
    _write_text_atomically(
        args.out.expanduser().resolve(),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "out": str(args.out.expanduser().resolve()),
                "n_rows": len(rows),
                "model_labels": [label for label, _ in labels],
                "summaries": {
                    comparison["model_label"]: comparison["contract_summaries"]
                    for comparison in comparisons
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
