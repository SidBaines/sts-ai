#!/usr/bin/env python
"""Build paired identity/cyclic-rotation strict teacher SFT datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from sts_ai.permutation_sft import (
    build_paired_datasets,
    finalize_manifest,
    render_jsonl,
)
from sts_ai.provenance import file_sha256


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could_not_read_{description}:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description}_not_object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could_not_read_source_dataset:{path}:{exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"source_dataset_invalid_json_line:{line_number}:{exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"source_dataset_row_not_object:{line_number}"
                )
            rows.append(value)
    return rows


def _load_tokenizer(model_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to build permutation SFT data"
        ) from exc
    return AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )


def _render_manifest(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".manifest.json")


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_fresh_distinct_outputs(paths: Sequence[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("paired_output_paths_must_be_distinct")
    existing = [str(path) for path in paths if _path_exists(path)]
    if existing:
        raise ValueError(
            "paired_outputs_must_be_fresh:"
            + json.dumps(existing, separators=(",", ":"))
        )


def _publish_fresh_files(payloads: Sequence[tuple[Path, bytes]]) -> None:
    """Publish all files without overwriting an existing path.

    A same-directory temporary file plus ``link`` provides an atomic
    fail-if-exists create for each target.  If a later target fails, files
    created by this call are rolled back.
    """

    paths = [path for path, _ in payloads]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    _require_fresh_distinct_outputs(paths)
    temporary_paths: list[Path] = []
    published: list[Path] = []
    try:
        for target, contents in payloads:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_paths.append(Path(handle.name))
        _require_fresh_distinct_outputs(paths)
        for (target, _), temporary in zip(payloads, temporary_paths):
            os.link(temporary, target)
            published.append(target)
            temporary.unlink()
        temporary_paths.clear()
    except Exception:
        for path in reversed(published):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink()
            except OSError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Default: <source_dataset stem>.manifest.json.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control-out", type=Path, required=True)
    parser.add_argument("--augmented-out", type=Path, required=True)
    parser.add_argument("--expected-source-dataset-sha256", default=None)
    parser.add_argument("--expected-source-manifest-sha256", default=None)
    parser.add_argument(
        "--require-paired-token-counts",
        action="store_true",
        help="Promise exact per-pair prompt/completion/supervised token-count "
        "matching and abort before publication if the promise is false.",
    )
    args = parser.parse_args(argv)
    if args.passes <= 0:
        parser.error("--passes must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    for name in ("control_out", "augmented_out"):
        if getattr(args, name).suffix != ".jsonl":
            parser.error(f"--{name.replace('_', '-')} must end in .jsonl")
    if args.source_manifest is None:
        args.source_manifest = _manifest_path(args.source_dataset)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    control_manifest_path = _manifest_path(args.control_out)
    augmented_manifest_path = _manifest_path(args.augmented_out)
    output_paths = (
        args.control_out,
        control_manifest_path,
        args.augmented_out,
        augmented_manifest_path,
    )
    _require_fresh_distinct_outputs(output_paths)

    source_dataset_sha256 = file_sha256(args.source_dataset)
    source_manifest_sha256 = file_sha256(args.source_manifest)
    if source_dataset_sha256 is None:
        raise ValueError("source_dataset_missing")
    if source_manifest_sha256 is None:
        raise ValueError("source_manifest_missing")
    if (
        args.expected_source_dataset_sha256 is not None
        and source_dataset_sha256
        != args.expected_source_dataset_sha256
    ):
        raise ValueError("source_dataset_expected_sha256_mismatch")
    if (
        args.expected_source_manifest_sha256 is not None
        and source_manifest_sha256
        != args.expected_source_manifest_sha256
    ):
        raise ValueError("source_manifest_expected_sha256_mismatch")
    rows = _load_jsonl(args.source_dataset)
    source_manifest = _load_json(
        args.source_manifest,
        description="source_manifest",
    )
    if (
        source_dataset_sha256 != file_sha256(args.source_dataset)
        or source_manifest_sha256 != file_sha256(args.source_manifest)
    ):
        raise RuntimeError("source_inputs_changed_while_loading")

    tokenizer = _load_tokenizer(args.model)
    built = build_paired_datasets(
        rows,
        source_manifest,
        tokenizer=tokenizer,
        model_id=args.model,
        repetitions=args.passes,
        seed=args.seed,
        source_dataset_sha256=source_dataset_sha256,
        source_manifest_sha256=source_manifest_sha256,
        require_paired_token_counts=args.require_paired_token_counts,
    )
    control_bytes = render_jsonl(built.control_rows)
    augmented_bytes = render_jsonl(built.augmented_rows)
    builder_root = Path(__file__).resolve().parents[1]
    builder_provenance = {
        "module": "sts_ai.permutation_sft",
        "module_sha256": file_sha256(
            builder_root / "src/sts_ai/permutation_sft.py"
        ),
        "cli": "scripts/build_paired_permutation_sft.py",
        "cli_sha256": file_sha256(Path(__file__).resolve()),
    }
    control_manifest = finalize_manifest(
        built.control_manifest,
        control_bytes,
    )
    augmented_manifest = finalize_manifest(
        built.augmented_manifest,
        augmented_bytes,
    )
    control_manifest["builder"] = builder_provenance
    augmented_manifest["builder"] = builder_provenance
    control_manifest_bytes = _render_manifest(control_manifest)
    augmented_manifest_bytes = _render_manifest(augmented_manifest)

    if (
        source_dataset_sha256 != file_sha256(args.source_dataset)
        or source_manifest_sha256 != file_sha256(args.source_manifest)
    ):
        raise RuntimeError("source_inputs_changed_while_building")
    _publish_fresh_files(
        (
            (args.control_out, control_bytes),
            (control_manifest_path, control_manifest_bytes),
            (args.augmented_out, augmented_bytes),
            (augmented_manifest_path, augmented_manifest_bytes),
        )
    )
    report = {
        "control": {
            "path": str(args.control_out.resolve()),
            "sha256": hashlib.sha256(control_bytes).hexdigest(),
            "manifest_path": str(control_manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(
                control_manifest_bytes
            ).hexdigest(),
        },
        "augmented": {
            "path": str(args.augmented_out.resolve()),
            "sha256": hashlib.sha256(augmented_bytes).hexdigest(),
            "manifest_path": str(augmented_manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(
                augmented_manifest_bytes
            ).hexdigest(),
        },
        "n_examples_per_arm": len(built.control_rows),
        "passes": args.passes,
        "paired_source_schedule_sha256": control_manifest["augmentation"][
            "source_schedule_sha256"
        ],
        "paired_token_counts": control_manifest["augmentation"][
            "paired_token_counts"
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
