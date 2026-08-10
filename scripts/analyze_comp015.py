#!/usr/bin/env python
"""Analyze the frozen COMP-015 paired reports without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from sts_ai.comp015_analysis import (
    LoadedArtifact,
    build_comp015_analysis,
    declared_artifact_specs,
)


def _read_json_bytes(path: Path, reason: str) -> tuple[bytes, dict[str, Any]]:
    try:
        contents = path.read_bytes()
        value = json.loads(contents)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{reason}:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{reason}_not_object:{path}")
    return contents, value


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root used to resolve report paths in the config (default: cwd).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError(f"output_must_be_fresh:{args.out}")
    config_bytes, config = _read_json_bytes(
        args.config,
        "could_not_read_analysis_config",
    )
    artifacts: dict[str, LoadedArtifact] = {}
    for spec in declared_artifact_specs(config):
        path = args.root / spec["path"]
        contents, value = _read_json_bytes(path, "could_not_read_report")
        digest = hashlib.sha256(contents).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"declared_report_sha256_mismatch:{spec['path']}")
        artifacts[spec["path"]] = LoadedArtifact(
            sha256=digest,
            value=value,
        )
    report = build_comp015_analysis(
        config,
        artifacts,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_fresh(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "primary_pass": report[
                    "fixed_step1500_primary_decision"
                ]["augmentation_primary_claim_pass"],
                "behavior_gate": report["behavior_gate"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
