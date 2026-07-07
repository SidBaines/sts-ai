#!/usr/bin/env python
"""Prepare a detachable local-curriculum task manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts_ai.local_tasks import base, get_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local-task manifest.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-rollout-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Manifest path. Defaults to data/local_curricula/<task>/manifests/source.json",
    )
    parser.add_argument("--holdout-mod", type=int, default=4)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = get_task(args.task)
    manifest = task.build_manifest(
        args.source_rollout_dir,
        holdout_mod=args.holdout_mod,
        holdout_remainder=args.holdout_remainder,
    )
    out = args.out or Path("data") / "local_curricula" / args.task / "manifests" / "source.json"
    base.write_json(out, manifest)
    print(f"wrote manifest: {out}")
    print(
        json.dumps(
            {
                "task_id": manifest["task_id"],
                "n_windows": manifest["n_windows"],
                "split_counts": manifest["split_counts"],
                "label_counts": manifest["label_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
