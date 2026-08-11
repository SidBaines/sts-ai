#!/usr/bin/env python
"""Evaluate compact teacher targets under a local MLX base model or adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

from sts_ai.provenance import file_sha256
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import (
    ACTION_ONLY_OUTPUT_CONTRACT,
    DEFAULT_GENERATION_MAX_TOKENS,
    EVALUATED_OUTPUT_CONTRACTS,
    MlxGreedyGenerator,
    MlxCandidateScorer,
    build_teacher_action_report,
    build_teacher_generation_report,
)


EXPECTED_OBSERVATION_VERSION = PUBLIC_OBSERVATION_VERSION
EXPECTED_TEACHER_SELECTION_RULE = AGGREGATED_ROOT_VISITS


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
    return rows


def _publish_fresh(path: Path, contents: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output_must_be_fresh:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


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


def _adapter_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"adapter path does not exist: {resolved}")
    if resolved.is_file():
        return {"path": str(resolved), "sha256": file_sha256(resolved)}
    files: dict[str, str | None] = {}
    for name in ("adapters.safetensors", "adapter_config.json"):
        candidate = resolved / name
        if candidate.is_file():
            files[name] = file_sha256(candidate)
    if "adapters.safetensors" not in files:
        raise ValueError(f"adapter directory has no adapters.safetensors: {resolved}")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True)
    return {
        "path": str(resolved),
        "files": files,
        "identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _manifest_provenance(
    path: Path | None,
    *,
    output_contract: str = ACTION_ONLY_OUTPUT_CONTRACT,
) -> dict[str, Any]:
    if path is None:
        raise ValueError("teacher scoring requires a dataset manifest")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "search_teacher_sft" or value.get("version") != 3:
        raise ValueError("teacher manifest must be search_teacher_sft version 3")
    if value.get("loss_mask_mode") != "action":
        raise ValueError("teacher manifest loss_mask_mode must be 'action'")
    if value.get("output_contract") != output_contract:
        if output_contract == ACTION_ONLY_OUTPUT_CONTRACT:
            raise ValueError("teacher manifest output_contract must be 'action_only'")
        raise ValueError(
            f"teacher manifest output_contract must be {output_contract!r}"
        )
    if value.get("enable_thinking") is not False:
        raise ValueError("action-only teacher manifest must set enable_thinking=false")
    if value.get("observation_version") != EXPECTED_OBSERVATION_VERSION:
        raise ValueError(
            "teacher manifest observation_version must be combat_public_v2"
        )
    if value.get("teacher_selection_rule") != EXPECTED_TEACHER_SELECTION_RULE:
        raise ValueError(
            "teacher manifest teacher_selection_rule must be aggregated_root_visits"
        )
    if value.get("teacher_privilege") != TEACHER_PRIVILEGE:
        raise ValueError(
            "teacher manifest teacher_privilege must be simulator_full_state"
        )
    source_labels = value.get("source_labels")
    if not isinstance(source_labels, dict):
        raise ValueError("teacher manifest must contain source_labels provenance")
    for name in ("sha256", "manifest_sha256"):
        digest = source_labels.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"teacher manifest source_labels.{name} is invalid")
    dataset_sha256 = value.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(char not in "0123456789abcdef" for char in dataset_sha256)
    ):
        raise ValueError("teacher manifest dataset_sha256 is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "kind": value.get("kind"),
        "version": value.get("version"),
        "tokenizer_id": value.get("tokenizer_id"),
        "chat_template_hash": value.get("chat_template_hash"),
        "n_examples": value.get("n_examples"),
        "enable_thinking": value.get("enable_thinking"),
        "loss_mask_mode": value.get("loss_mask_mode"),
        "output_contract": value.get("output_contract"),
        "observation_version": value.get("observation_version"),
        "teacher_selection_rule": value.get("teacher_selection_rule"),
        "teacher_privilege": value.get("teacher_privilege"),
        "dataset_sha256": dataset_sha256,
        "source_labels": source_labels,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-bf16")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Teacher manifest. If omitted, <dataset stem>.manifest.json is used when present.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--contract",
        choices=EVALUATED_OUTPUT_CONTRACTS,
        default=ACTION_ONLY_OUTPUT_CONTRACT,
    )
    parser.add_argument(
        "--mode",
        choices=("candidates", "generate"),
        default="candidates",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_GENERATION_MAX_TOKENS,
        help=(
            "Maximum new tokens in --mode generate "
            f"(default: {DEFAULT_GENERATION_MAX_TOKENS})."
        ),
    )
    parser.add_argument(
        "--per-row-out",
        type=Path,
        default=None,
        help="Optional fresh JSONL sidecar containing each scored row.",
    )
    args = parser.parse_args(argv)
    if (
        args.per_row_out is not None
        and args.per_row_out.resolve(strict=False) == args.out.resolve(strict=False)
    ):
        parser.error("--per-row-out and --out must be distinct")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError(f"output_must_be_fresh:{args.out}")
    if args.per_row_out is not None and (
        args.per_row_out.exists() or args.per_row_out.is_symlink()
    ):
        raise ValueError(f"output_must_be_fresh:{args.per_row_out}")
    manifest_path = args.manifest
    if manifest_path is None:
        candidate = args.dataset.with_suffix(".manifest.json")
        manifest_path = candidate if candidate.exists() else None
    dataset_sha256 = file_sha256(args.dataset)
    rows = _load_jsonl(args.dataset)
    if not rows:
        raise ValueError("teacher dataset is empty")
    manifest = _manifest_provenance(
        manifest_path,
        output_contract=args.contract,
    )
    if dataset_sha256 != file_sha256(args.dataset):
        raise RuntimeError("teacher dataset changed while it was loaded")
    if manifest["n_examples"] != len(rows):
        raise ValueError(
            "teacher manifest n_examples disagrees with the dataset row count"
        )
    if manifest["dataset_sha256"] != dataset_sha256:
        raise ValueError("teacher manifest dataset_sha256 disagrees with the dataset")
    for row_index, row in enumerate(rows):
        if row.get("observation_version") != EXPECTED_OBSERVATION_VERSION:
            raise ValueError(
                f"teacher dataset row {row_index} has invalid observation_version"
            )
        if row.get("teacher_selection_rule") != EXPECTED_TEACHER_SELECTION_RULE:
            raise ValueError(
                f"teacher dataset row {row_index} has invalid teacher_selection_rule"
            )
    adapter = _adapter_provenance(args.adapter_path)
    provenance = {
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": dataset_sha256,
        },
        "manifest": manifest,
        "model_id": args.model,
        "adapter": adapter,
        "backend": "mlx",
        "git_head": _git_head(),
        "evaluator_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1]
            / "src/sts_ai/teacher_action_eval.py"
        ),
    }
    adapter_path = (
        str(args.adapter_path.resolve()) if args.adapter_path else None
    )
    if args.mode == "candidates":
        scorer = MlxCandidateScorer(
            args.model,
            adapter_path=adapter_path,
        )
        report = build_teacher_action_report(
            rows,
            scorer,
            provenance=provenance,
            output_contract=args.contract,
        )
    else:
        generator = MlxGreedyGenerator(
            args.model,
            adapter_path=adapter_path,
        )
        report = build_teacher_generation_report(
            rows,
            generator,
            output_contract=args.contract,
            max_tokens=args.max_tokens,
            provenance=provenance,
        )
    if dataset_sha256 != file_sha256(args.dataset):
        raise RuntimeError("teacher dataset changed while it was scored")
    if manifest["sha256"] != file_sha256(manifest["path"]):
        raise RuntimeError("teacher manifest changed while the dataset was scored")
    if not report["n_scored_rows"]:
        raise ValueError(
            "no teacher rows survived validation: "
            + json.dumps(report["skipped_record_counts"], sort_keys=True)
        )
    if args.per_row_out is not None:
        per_row_payload = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for row in report["rows"]
        ).encode("utf-8")
        _publish_fresh(args.per_row_out, per_row_payload)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _publish_fresh(args.out, rendered.encode("utf-8"))
    print(rendered, end="")


if __name__ == "__main__":
    main()
