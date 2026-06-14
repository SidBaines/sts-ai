#!/usr/bin/env python
"""Compute risk proxies from recorded rollout JSONL traces.

Reads decision records, classifies risk-relevant decisions, and writes both a
per-event CSV and an aggregate JSON summary. Operates entirely on stored traces
(no rollouts re-run), per the Stage 2 acceptance criteria in
docs/research_plan.md.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_risk_proxies.py \
        --glob 'data/baseline_rollouts_300/**/seed_*.jsonl' \
        --events-csv data/baseline_rollouts_300/risk_events.csv \
        --summary-json data/baseline_rollouts_300/risk_summary.json
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import json
from pathlib import Path

from sts_ai.risk_proxies import RiskEvent, load_records, risk_events, summarize_risk


def gather(paths: list[str], glob_pattern: str | None) -> list[str]:
    files: list[str] = []
    for path in paths:
        files.append(path)
    if glob_pattern:
        files.extend(sorted(glob.glob(glob_pattern, recursive=True)))
    # de-dup, keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute risk proxies from rollout JSONL files.")
    parser.add_argument("paths", nargs="*", help="Rollout JSONL files.")
    parser.add_argument("--glob", dest="glob_pattern", default=None)
    parser.add_argument("--events-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--group-by-source", action="store_true",
                        help="Also emit a per-file (e.g. per-agent dir) summary block.")
    args = parser.parse_args()

    files = gather(args.paths, args.glob_pattern)
    if not files:
        parser.error("no input files (pass paths or --glob)")

    all_events: list[RiskEvent] = []
    per_source: dict[str, list[RiskEvent]] = {}
    for f in files:
        evs = risk_events(load_records(f))
        all_events.extend(evs)
        # group by parent dir (agent) as a convenience
        source = str(Path(f).parent)
        per_source.setdefault(source, []).extend(evs)

    summary = {"overall": summarize_risk(all_events), "n_files": len(files)}
    if args.group_by_source:
        summary["by_source"] = {src: summarize_risk(evs) for src, evs in sorted(per_source.items())}

    if args.events_csv:
        Path(args.events_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.events_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[f.name for f in dataclasses.fields(RiskEvent)])
            writer.writeheader()
            for event in all_events:
                writer.writerow(dataclasses.asdict(event))
        print(f"wrote {len(all_events)} events to {args.events_csv}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"wrote summary to {args.summary_json}")

    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
