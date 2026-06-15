from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_error(path: Path) -> dict[str, Any] | None:
    error_path = path.with_suffix(".error.json")
    if not error_path.exists():
        return None
    with error_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_file(path: Path) -> dict[str, Any]:
    records = load_records(path)
    error = load_error(path)
    stopped_reason = error["stopped_reason"] if error is not None else "ok"
    error_type = ""
    error_message = ""
    if error is not None:
        error_body = error.get("error", {})
        error_type = str(error_body.get("type", ""))
        error_message = str(error_body.get("message", ""))

    if not records:
        return {
            "path": str(path),
            "world_seed": "",
            "decisions": 0,
            "valid_rate": 0.0,
            "invalid": 0,
            "total_retries": 0,
            "avg_retries": 0.0,
            "final_act": "",
            "final_floor": "",
            "final_outcome": "",
            "final_hp": "",
            "final_max_hp": "",
            "final_gold": "",
            "final_done": "",
            "stopped_reason": stopped_reason,
            "error_type": error_type,
            "error_message": error_message,
            "screen_counts": "{}",
        }

    valid_count = sum(1 for record in records if record["agent"]["valid"])
    retries = [int(record["agent"]["retries"]) for record in records]
    screens = Counter(record["state"]["screen_state"] for record in records)
    final = records[-1]["after_state"]

    return {
        "path": str(path),
        "world_seed": records[0]["world_seed"],
        "decisions": len(records),
        "valid_rate": valid_count / len(records),
        "invalid": len(records) - valid_count,
        "total_retries": sum(retries),
        "avg_retries": mean(retries) if retries else 0.0,
        "final_act": final["act"],
        "final_floor": final["floor"],
        "final_outcome": final["outcome"],
        "final_hp": final["cur_hp"],
        "final_max_hp": final["max_hp"],
        "final_gold": final["gold"],
        "final_done": final["done"],
        "stopped_reason": stopped_reason,
        "error_type": error_type,
        "error_message": error_message,
        "screen_counts": json.dumps(dict(sorted(screens.items())), sort_keys=True),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rollout files found.")
        return

    columns = [
        "world_seed",
        "decisions",
        "valid_rate",
        "invalid",
        "total_retries",
        "stopped_reason",
        "error_type",
        "final_floor",
        "final_outcome",
        "final_hp",
        "final_gold",
        "path",
    ]
    print("\t".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        print("\t".join(values))

    print()
    print(f"files: {len(rows)}")
    print(f"total_decisions: {sum(int(row['decisions']) for row in rows)}")
    if rows:
        print(f"mean_valid_rate: {mean(float(row['valid_rate']) for row in rows):.3f}")
        print(f"mean_final_floor: {mean(float(row['final_floor']) for row in rows if row['final_floor'] != ''):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rollout JSONL files.")
    parser.add_argument("paths", nargs="*", help="Rollout JSONL files or glob patterns.")
    parser.add_argument("--glob", default="data/rollouts/**/*.jsonl")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    patterns = args.paths or [args.glob]
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        files.extend(Path(match) for match in matches)
    files = sorted(set(files))

    rows = [summarize_file(path) for path in files]
    print_table(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"wrote: {args.csv}")


if __name__ == "__main__":
    main()
