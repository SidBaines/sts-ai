#!/usr/bin/env python
"""Rescore per-row teacher choices with visit-share regret metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from sts_ai.teacher_metrics import build_metrics_report


def _read_bytes(path: Path, *, reason: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{reason}:{path}:{exc}") from exc


def _load_jsonl(contents: bytes, *, path: Path) -> list[dict[str, Any]]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid_utf8:{path}:{exc}") from exc
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
    if not rows:
        raise ValueError(f"jsonl_empty:{path}")
    return rows


def _load_quarantined_windows(contents: bytes, *, path: Path) -> frozenset[str]:
    try:
        value = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_quarantine_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("quarantine_not_object")
    if value.get("kind") != "state_sanity_quarantine" or value.get("version") != 1:
        raise ValueError("quarantine_contract_invalid")
    windows = value.get("quarantined_windows")
    if not isinstance(windows, list) or any(
        not isinstance(window, str) or not window for window in windows
    ):
        raise ValueError("quarantined_windows_invalid")
    if len(set(windows)) != len(windows):
        raise ValueError("quarantined_windows_duplicate")
    return frozenset(windows)


def _audit_rows_by_hash(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        public_state_hash = row.get("public_state_hash")
        if not isinstance(public_state_hash, str) or not public_state_hash:
            raise ValueError("audit_public_state_hash_invalid")
        if public_state_hash in by_hash:
            raise ValueError(f"duplicate_audit_public_state_hash:{public_state_hash}")
        by_hash[public_state_hash] = row
    return by_hash


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
    parser.add_argument("--per-row", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, default=None)
    parser.add_argument("--tie-ratio", type=float, default=0.8)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError(f"output_must_be_fresh:{args.out}")

    input_paths = [args.per_row, args.audit]
    if args.quarantine is not None:
        input_paths.append(args.quarantine)
    resolved_inputs = [path.resolve(strict=False) for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("input_paths_must_be_distinct")
    if args.out.resolve(strict=False) in set(resolved_inputs):
        raise ValueError("output_must_be_distinct_from_inputs")

    per_row_bytes = _read_bytes(args.per_row, reason="could_not_read_per_row")
    audit_bytes = _read_bytes(args.audit, reason="could_not_read_audit")
    per_row_rows = _load_jsonl(per_row_bytes, path=args.per_row)
    audit_rows = _load_jsonl(audit_bytes, path=args.audit)
    choices = []
    for row in per_row_rows:
        choices.append(
            {
                "public_state_hash": row.get("public_state_hash"),
                "chosen_index": row.get("top1_action_index"),
            }
        )

    quarantined_windows: frozenset[str] = frozenset()
    quarantine_bytes: bytes | None = None
    if args.quarantine is not None:
        quarantine_bytes = _read_bytes(
            args.quarantine,
            reason="could_not_read_quarantine",
        )
        quarantined_windows = _load_quarantined_windows(
            quarantine_bytes,
            path=args.quarantine,
        )
    metrics = build_metrics_report(
        choices,
        _audit_rows_by_hash(audit_rows),
        quarantined_windows=quarantined_windows,
        tie_ratio=args.tie_ratio,
    )

    input_contents = [per_row_bytes, audit_bytes]
    if quarantine_bytes is not None:
        input_contents.append(quarantine_bytes)
    inputs = [
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        for path, contents in zip(input_paths, input_contents)
    ]
    for path, contents in zip(input_paths, input_contents):
        if hashlib.sha256(_read_bytes(path, reason="input_changed")).digest() != (
            hashlib.sha256(contents).digest()
        ):
            raise RuntimeError(f"input_changed_while_rescoring:{path}")

    report = {
        "kind": "teacher_regret_report",
        "version": 1,
        "inputs": inputs,
        **metrics,
    }
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_fresh(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "n": report["overall"]["all"]["n"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
