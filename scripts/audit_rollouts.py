#!/usr/bin/env python
"""Audit rollout JSONL/meta files for trace-contract and parser-quality issues."""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from sts_ai.schemas import SCHEMA_VERSION


@dataclass
class AuditIssue:
    severity: str
    code: str
    path: str
    message: str


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_records(path: Path, issues: list[AuditIssue]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                issues.append(
                    AuditIssue(
                        "error",
                        "jsonl_parse_error",
                        f"{path}:{lineno}",
                        str(exc),
                    )
                )
    return records


def audit_file(path: Path, *, require_meta: bool = True) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    records = _iter_records(path, issues)
    meta_path = path.with_suffix(".meta.json")
    meta: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            meta = _load_json(meta_path)
        except json.JSONDecodeError as exc:
            issues.append(AuditIssue("error", "meta_parse_error", str(meta_path), str(exc)))
    elif require_meta:
        issues.append(AuditIssue("error", "missing_meta", str(path), "missing .meta.json sidecar"))

    invalid = 0
    no_json = 0
    truncated = 0
    unclosed_think = 0
    json_in_thinking = 0
    missing_thinking = 0
    unexecuted = 0
    legacy_fallback = 0
    retry_records = 0

    for idx, record in enumerate(records):
        agent = record.get("agent", {}) or {}
        raw = str(agent.get("raw_response", "") or "")
        raw_lower = raw.lower()
        thinking = agent.get("thinking")
        metadata = agent.get("metadata", {}) or {}
        valid = bool(agent.get("valid", True))
        action_executed = bool(record.get("action_executed", True))

        if not valid:
            invalid += 1
            if metadata.get("error") == "no json object":
                no_json += 1
            if metadata.get("error") == "truncated_before_json":
                truncated += 1
            if action_executed:
                legacy_fallback += 1
                issues.append(
                    AuditIssue(
                        "error",
                        "invalid_action_executed",
                        f"{path}:{idx + 1}",
                        "invalid agent decision executed a fallback action",
                    )
                )
        if not action_executed:
            unexecuted += 1
        if _as_int(agent.get("retries"), 0) > 0:
            retry_records += 1
        if "<think>" in raw_lower and "</think>" not in raw_lower:
            unclosed_think += 1
            issues.append(
                AuditIssue(
                    "warning",
                    "unclosed_think",
                    f"{path}:{idx + 1}",
                    "raw response opens <think> without closing </think>",
                )
            )
        if thinking and ('"reasoning"' in str(thinking) or "```json" in str(thinking).lower()):
            json_in_thinking += 1
            issues.append(
                AuditIssue(
                    "warning",
                    "json_in_thinking",
                    f"{path}:{idx + 1}",
                    "agent.thinking appears to contain final JSON",
                )
            )
        if "<think>" in raw_lower and thinking in (None, ""):
            missing_thinking += 1
            issues.append(
                AuditIssue(
                    "warning",
                    "missing_thinking",
                    f"{path}:{idx + 1}",
                    "raw response has <think> but agent.thinking is empty/missing",
                )
            )

    if meta is not None:
        if _as_int(meta.get("schema_version"), 0) != SCHEMA_VERSION:
            issues.append(
                AuditIssue(
                    "error",
                    "schema_version_mismatch",
                    str(meta_path),
                    f"expected schema_version {SCHEMA_VERSION}, found {meta.get('schema_version')}",
                )
            )
        if _as_int(meta.get("n_decisions")) != len(records):
            issues.append(
                AuditIssue(
                    "error",
                    "meta_decision_count_mismatch",
                    str(meta_path),
                    f"meta n_decisions={meta.get('n_decisions')} but JSONL has {len(records)}",
                )
            )
        if _as_int(meta.get("n_invalid")) != invalid:
            issues.append(
                AuditIssue(
                    "error",
                    "meta_invalid_count_mismatch",
                    str(meta_path),
                    f"meta n_invalid={meta.get('n_invalid')} but JSONL has {invalid}",
                )
            )
        if unexecuted and meta.get("stopped_reason") != "agent_invalid":
            issues.append(
                AuditIssue(
                    "error",
                    "unexecuted_without_agent_invalid",
                    str(meta_path),
                    "unexecuted decision present but stopped_reason is not agent_invalid",
                )
            )

    row = {
        "path": str(path),
        "records": len(records),
        "invalid": invalid,
        "no_json": no_json,
        "truncated_before_json": truncated,
        "unclosed_think": unclosed_think,
        "json_in_thinking": json_in_thinking,
        "missing_thinking": missing_thinking,
        "unexecuted": unexecuted,
        "legacy_fallback": legacy_fallback,
        "retry_records": retry_records,
        "meta": meta is not None,
    }
    return row, issues


def expand_paths(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.extend(Path(match) for match in matches)
        else:
            files.append(Path(pattern))
    return sorted({path for path in files if path.suffix == ".jsonl"})


def audit_paths(paths: list[Path], *, require_meta: bool = True) -> tuple[list[dict[str, Any]], list[AuditIssue]]:
    rows: list[dict[str, Any]] = []
    issues: list[AuditIssue] = []
    for path in sorted(paths):
        if not path.exists():
            issues.append(AuditIssue("error", "missing_file", str(path), "file does not exist"))
            continue
        row, file_issues = audit_file(path, require_meta=require_meta)
        rows.append(row)
        issues.extend(file_issues)
    return rows, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rollout JSONL files and .meta.json sidecars.")
    parser.add_argument("paths", nargs="*", help="JSONL files or glob patterns.")
    parser.add_argument("--glob", default="data/rollouts/**/*.jsonl")
    parser.add_argument("--no-require-meta", action="store_true")
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    patterns = args.paths or [args.glob]
    rows, issues = audit_paths(expand_paths(patterns), require_meta=not args.no_require_meta)

    if args.json:
        print(json.dumps({"rows": rows, "issues": [asdict(issue) for issue in issues]}, indent=2))
    else:
        print(f"files: {len(rows)}")
        print(f"records: {sum(row['records'] for row in rows)}")
        print(f"invalid: {sum(row['invalid'] for row in rows)}")
        print(f"unexecuted: {sum(row['unexecuted'] for row in rows)}")
        print(f"issues: {len(issues)}")
        for issue in issues:
            print(f"{issue.severity}\t{issue.code}\t{issue.path}\t{issue.message}")

    has_errors = any(issue.severity == "error" for issue in issues)
    if has_errors and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
