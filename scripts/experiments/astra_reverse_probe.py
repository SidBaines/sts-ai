"""Reverse/reindex legal menus while preserving state text and action identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts_ai.provenance import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/experiments/astra_5sep/policy/cohort.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/astra_5sep/policy/cohort_reversed.jsonl"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    with args.out.open("x") as handle:
        for row in rows:
            actions = row["legal_actions"]
            if [a["index"] for a in actions] != list(range(len(actions))):
                raise ValueError("source menu is not contiguous")
            row["probe_original_indices"] = [a["index"] for a in reversed(actions)]
            row["legal_actions"] = [{**a, "index": i} for i, a in enumerate(reversed(actions))]
            handle.write(json.dumps(row) + "\n")
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "source_sha256": file_sha256(args.source), "sha256": file_sha256(args.out),
        "script_sha256": file_sha256(__file__), "n": len(rows),
        "intervention": "reverse legal action display order and renumber; state_text unchanged; invert indices before value comparison",
    }, indent=2))


if __name__ == "__main__":
    main()
