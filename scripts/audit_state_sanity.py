#!/usr/bin/env python
"""Audit serialized combat states and publish a deterministic quarantine sidecar."""
from __future__ import annotations

import argparse
from dataclasses import asdict, astuple
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from sts_ai.state_sanity import phantom_power_findings, validate_task_id


def _load_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        contents = path.read_bytes()
        text = contents.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could_not_read_labels:{path}:{exc}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid_json:{path}:{line_number}:{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"row_not_object:{path}:{line_number}")
        rows.append(value)
    return contents, rows


def _required_str(
    row: dict[str, Any],
    field: str,
    *,
    path: Path,
    row_number: int,
) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"invalid_{field}:{path}:{row_number}")
    return value


def _world_seed(row: dict[str, Any], *, path: Path, row_number: int) -> int:
    value = row.get("world_seed")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"invalid_world_seed:{path}:{row_number}")
    return value


def build_quarantine(
    labels: Sequence[Path],
    *,
    task_id: str,
) -> dict[str, Any]:
    validate_task_id(task_id)
    inputs: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    n_rows = 0

    for path in labels:
        contents, rows = _load_jsonl(path)
        inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "n_rows": len(rows),
            }
        )
        n_rows += len(rows)
        for row_number, row in enumerate(rows, start=1):
            state_text = _required_str(
                row, "state_text", path=path, row_number=row_number
            )
            window_id = _required_str(
                row, "window_id", path=path, row_number=row_number
            )
            public_state_hash = _required_str(
                row, "public_state_hash", path=path, row_number=row_number
            )
            world_seed = _world_seed(row, path=path, row_number=row_number)
            findings = sorted(
                phantom_power_findings(state_text, task_id=task_id),
                key=astuple,
            )
            identity = {
                "window_id": window_id,
                "world_seed": world_seed,
                "findings": findings,
            }
            prior = by_hash.get(public_state_hash)
            if prior is not None:
                if prior != identity:
                    raise ValueError(
                        "conflicting_duplicate_public_state_hash:"
                        f"{public_state_hash}"
                    )
                continue
            by_hash[public_state_hash] = identity

    flagged: list[dict[str, Any]] = []
    for public_state_hash, state in by_hash.items():
        findings = state["findings"]
        if not findings:
            continue
        flagged.append(
            {
                "public_state_hash": public_state_hash,
                "window_id": state["window_id"],
                "world_seed": state["world_seed"],
                "findings": [asdict(finding) for finding in findings],
            }
        )
    flagged.sort(key=lambda item: (item["window_id"], item["public_state_hash"]))
    quarantined_windows = sorted({item["window_id"] for item in flagged})
    quarantined_state_hashes = sorted(
        item["public_state_hash"] for item in flagged
    )
    return {
        "kind": "state_sanity_quarantine",
        "version": 1,
        "task_id": task_id,
        "inputs": inputs,
        "findings": flagged,
        "quarantined_windows": quarantined_windows,
        "quarantined_state_hashes": quarantined_state_hashes,
        "summary": {
            "n_rows": n_rows,
            "n_states_flagged": len(flagged),
            "n_windows_flagged": len(quarantined_windows),
        },
    }


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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"output_must_be_fresh:{path}") from exc
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
    parser.add_argument(
        "--labels",
        type=Path,
        action="append",
        required=True,
        help="Input labels JSONL (repeat for multiple files).",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError(f"output_must_be_fresh:{args.out}")
    report = build_quarantine(args.labels, task_id=args.task)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_fresh(args.out, payload)


if __name__ == "__main__":
    main()
