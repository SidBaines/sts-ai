#!/usr/bin/env python
"""Report prompt/completion token and latency distributions from rollout JSONL."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases = sorted({str(record.get("phase", "unknown")) for record in records})

    def one_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        agents = [record.get("agent") or {} for record in group]
        valid = [bool(agent.get("valid", True)) for agent in agents]
        return {
            "n_decisions": len(group),
            "prompt_tokens": _distribution(
                [float(agent.get("prompt_tokens", 0) or 0) for agent in agents]
            ),
            "completion_tokens": _distribution(
                [float(agent.get("completion_tokens", 0) or 0) for agent in agents]
            ),
            "thinking_tokens": _distribution(
                [float(agent.get("thinking_tokens", 0) or 0) for agent in agents]
            ),
            "latency_s": _distribution(
                [float(agent.get("latency_s", 0.0) or 0.0) for agent in agents]
            ),
            "invalid_rate": (sum(not value for value in valid) / len(valid)) if valid else 0.0,
        }

    return {
        "overall": one_group(records),
        "by_phase": {
            phase: one_group(
                [record for record in records if str(record.get("phase", "unknown")) == phase]
            )
            for phase in phases
        },
    }


def _load(root: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    paths = sorted(root.rglob("seed_*_r*.jsonl")) if root.is_dir() else [root]
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records, len(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, n_files = _load(args.input)
    report = build_report(records)
    report["input"] = str(args.input)
    report["n_rollout_files"] = n_files
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
