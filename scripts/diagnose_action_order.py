#!/usr/bin/env python
"""Score deterministic legal-action rotations for frozen teacher checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Sequence

from sts_ai.action_order_diagnostic import (
    build_checkpoint_diagnostic,
    build_diagnostic_report,
    parse_teacher_menu,
)
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import (
    ACTION_ONLY_OUTPUT_CONTRACT,
    MlxCandidateScorer,
)
from sts_ai.train.sft_format import chat_template_probe_hash


_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
    chat_template_hash = value.get("chat_template_hash")
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("teacher manifest tokenizer_id is invalid")
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
    expected_dataset_sha256: str | None,
    expected_manifest_sha256: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    dataset_sha256 = file_sha256(dataset_path)
    if dataset_sha256 is None:
        raise ValueError(f"teacher dataset does not exist: {dataset_path}")
    if (
        expected_dataset_sha256 is not None
        and dataset_sha256 != expected_dataset_sha256
    ):
        raise ValueError("teacher dataset does not match --expected-dataset-sha256")
    rows = _load_jsonl(dataset_path)
    if dataset_sha256 != file_sha256(dataset_path):
        raise RuntimeError("teacher dataset changed while it was loaded")
    manifest = _manifest_provenance(
        manifest_path,
        dataset_sha256=dataset_sha256,
        n_rows=len(rows),
    )
    if (
        expected_manifest_sha256 is not None
        and manifest["sha256"] != expected_manifest_sha256
    ):
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
        parsed[label] = Path(path_text)
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
    mlx_config = path / "mlx_lora_config.json"
    if not mlx_config.is_file():
        raise ValueError("checkpoint directory has no mlx_lora_config.json")
    return {
        **provenance,
        "mlx_lora_config_sha256": file_sha256(mlx_config),
    }


def _reference_report_provenance(
    path: Path,
    *,
    report: dict[str, Any],
    dataset_sha256: str,
    manifest_sha256: str,
    model_id: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("reference report has no provenance")
    dataset = provenance.get("dataset")
    manifest = provenance.get("manifest")
    adapter = provenance.get("adapter")
    if not isinstance(dataset, dict) or dataset.get("sha256") != dataset_sha256:
        raise ValueError("reference report dataset provenance mismatch")
    if not isinstance(manifest, dict) or manifest.get("sha256") != manifest_sha256:
        raise ValueError("reference report manifest provenance mismatch")
    if provenance.get("model_id") != model_id:
        raise ValueError("reference report model provenance mismatch")
    if (
        not isinstance(adapter, dict)
        or adapter.get("identity_sha256") != checkpoint.get("identity_sha256")
        or adapter.get("files") != checkpoint.get("files")
    ):
        raise ValueError("reference report adapter provenance mismatch")
    evaluator_source_sha256 = provenance.get("evaluator_source_sha256")
    if not _is_sha256(evaluator_source_sha256):
        raise ValueError("reference report evaluator provenance is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "evaluator_source_sha256": evaluator_source_sha256,
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


def _runtime_tokenizer(scorer: MlxCandidateScorer) -> Any:
    """Return the tokenizer without retaining a second live model reference."""

    return scorer._load()[1]


def _drop_scorer_references(scorer: MlxCandidateScorer) -> None:
    scorer._model = None
    scorer._tokenizer = None


def _clear_scorer(scorer: MlxCandidateScorer) -> None:
    _drop_scorer_references(scorer)
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


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
        if path.exists():
            raise ValueError(f"refusing to overwrite diagnostic output: {path}")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-bf16")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Checkpoint adapter directory. Repeat for each checkpoint.",
    )
    parser.add_argument(
        "--reference-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Matching unpermuted score report. Repeat for each checkpoint.",
    )
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite diagnostic output: {args.out}")
    for name, value in (
        ("--expected-dataset-sha256", args.expected_dataset_sha256),
        ("--expected-manifest-sha256", args.expected_manifest_sha256),
    ):
        if value is not None and not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    checkpoints = _parse_label_paths(args.checkpoint, option="--checkpoint")
    references = _parse_label_paths(
        args.reference_report,
        option="--reference-report",
    )
    if checkpoints.keys() != references.keys():
        raise ValueError("checkpoint and reference-report labels must match exactly")

    rows, manifest, dataset_sha256 = _load_teacher_inputs(
        args.dataset,
        args.manifest,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    manifest_contents = manifest["contents"]
    if manifest_contents["tokenizer_id"] != args.model:
        raise ValueError("teacher manifest tokenizer_id disagrees with --model")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    source_paths = {
        "cli": script_path,
        "diagnostic_core": repo_root / "src/sts_ai/action_order_diagnostic.py",
        "candidate_evaluator": repo_root / "src/sts_ai/teacher_action_eval.py",
    }
    source_hashes = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    if any(digest is None for digest in source_hashes.values()):
        raise ValueError("one or more diagnostic source files could not be hashed")
    checkpoint_reports: dict[str, dict[str, Any]] = {}
    durable_checkpoint_provenance: dict[str, dict[str, Any]] = {}
    durable_reference_provenance: dict[str, dict[str, Any]] = {}

    for label in sorted(checkpoints):
        checkpoint = _checkpoint_provenance(checkpoints[label])
        reference_report = _load_json(
            references[label],
            description=f"{label} reference report",
        )
        reference = _reference_report_provenance(
            references[label],
            report=reference_report,
            dataset_sha256=dataset_sha256,
            manifest_sha256=str(manifest["sha256"]),
            model_id=args.model,
            checkpoint=checkpoint,
        )
        if (
            reference["evaluator_source_sha256"]
            != source_hashes["candidate_evaluator"]
        ):
            raise ValueError(
                "reference report evaluator hash disagrees with the current "
                "candidate evaluator"
            )
        scorer = MlxCandidateScorer(
            args.model,
            adapter_path=str(checkpoints[label].resolve()),
        )
        try:
            tokenizer = _runtime_tokenizer(scorer)
            runtime_chat_template_hash = chat_template_probe_hash(
                tokenizer,
                enable_thinking=False,
            )
            if (
                runtime_chat_template_hash
                != manifest_contents["chat_template_hash"]
            ):
                raise ValueError(
                    "runtime tokenizer chat-template hash disagrees with manifest"
                )
            checkpoint_reports[label] = build_checkpoint_diagnostic(
                rows,
                scorer,
                reference_report,
                checkpoint_label=label,
                prompt_renderer=_runtime_prompt_renderer(tokenizer),
                provenance={
                    "checkpoint": checkpoint,
                    "reference_report": reference,
                    "runtime_chat_template_hash": runtime_chat_template_hash,
                },
            )
        finally:
            _clear_scorer(scorer)
        durable_checkpoint_provenance[label] = checkpoint
        durable_reference_provenance[label] = reference

        if checkpoint != _checkpoint_provenance(checkpoints[label]):
            raise RuntimeError(f"{label} checkpoint changed while it was scored")
        if reference["sha256"] != file_sha256(references[label]):
            raise RuntimeError(f"{label} reference report changed while it was scored")
        if dataset_sha256 != file_sha256(args.dataset):
            raise RuntimeError("teacher dataset changed while it was scored")
        if manifest["sha256"] != file_sha256(args.manifest):
            raise RuntimeError("teacher manifest changed while it was scored")
        if source_hashes != {
            name: file_sha256(path) for name, path in source_paths.items()
        }:
            raise RuntimeError("diagnostic source changed while it was scored")

    report = build_diagnostic_report(
        checkpoint_reports,
        provenance={
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
            "backend": "mlx",
            "git_head": _git_head(),
            "source_sha256": source_hashes,
            "checkpoints": durable_checkpoint_provenance,
            "reference_reports": durable_reference_provenance,
        },
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_text_atomically(args.out, rendered)
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "output_sha256": file_sha256(args.out),
                "checkpoint_labels": report["checkpoint_labels"],
                "n_source_rows": report["n_source_rows"],
                "transformation_set_sha256": report[
                    "transformation_set_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
