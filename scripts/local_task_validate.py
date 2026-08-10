#!/usr/bin/env python
"""Replay-validate local-task windows in isolated subprocesses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from sts_ai.local_tasks import base, get_task
from sts_ai.local_tasks.validation import (
    RESULT_STATUSES,
    build_validated_manifest,
    validate_manifest_isolated,
    validate_one_window,
)


def _add_replay_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--combat-observation",
        choices=("legacy", "combat_public_v1", "combat_public_v2"),
        default="legacy",
    )
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--max-act", type=int, default=3)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay-validate every local-task manifest window in an "
        "isolated subprocess."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Optionally write a passing-only manifest with explicit exclusions.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Wall-clock timeout for each isolated window replay (default: 30).",
    )
    _add_replay_settings(parser)
    return parser.parse_args(argv)


def parse_internal_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_validate-window", action="store_true", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    _add_replay_settings(parser)
    return parser.parse_args(argv)


def _validate_internal(argv: Sequence[str]) -> int:
    args = parse_internal_args(argv)
    manifest = base.load_manifest(args.manifest)
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(
            f"manifest task_id={manifest['task_id']!r} != --task {args.task!r}"
        )
    if args.window_index < 0 or args.window_index >= len(manifest["windows"]):
        raise IndexError(f"window index out of range: {args.window_index}")
    result = validate_one_window(
        manifest,
        window_index=args.window_index,
        task=task,
        combat_observation=args.combat_observation,
        battle_simulations=args.battle_simulations,
        max_act=args.max_act,
    )
    base.write_json(args.result, result)
    return int(result["returncode"])


def _progress(completed: int, total: int, result: dict) -> None:
    print(
        f"[{completed}/{total}] {result['window_id']}: {result['status']}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--_validate-window" in raw_argv:
        return _validate_internal(raw_argv)

    args = parse_args(raw_argv)
    manifest = base.load_manifest(args.manifest)
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(
            f"manifest task_id={manifest['task_id']!r} != --task {args.task!r}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    report = validate_manifest_isolated(
        manifest,
        source_manifest_path=args.manifest,
        task_id=task.task_id,
        timeout_seconds=args.timeout_seconds,
        combat_observation=args.combat_observation,
        battle_simulations=args.battle_simulations,
        max_act=args.max_act,
        script_path=Path(__file__).resolve(),
        repo_root=repo_root,
        progress=_progress,
    )
    base.write_json(args.report, report)
    print(f"wrote validation report: {args.report}")

    if args.out_manifest is not None:
        validated = build_validated_manifest(
            manifest,
            report,
            source_manifest_path=args.manifest,
            report_path=args.report,
        )
        base.write_json(args.out_manifest, validated)
        print(f"wrote validated manifest: {args.out_manifest}")

    counts = report["status_counts"]
    summary = ", ".join(f"{status}={counts[status]}" for status in RESULT_STATUSES)
    print(f"validated {report['n_windows']} windows: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
