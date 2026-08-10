#!/usr/bin/env python
"""Build an SFT dataset for a detachable local-curriculum task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from sts_ai.local_tasks import base
from sts_ai.local_tasks.sft_dataset import build_local_sft_dataset
from sts_ai.prompting import NEUTRAL_FRAME


def _framing_arg(value: str) -> str:
    return NEUTRAL_FRAME if value == "neutral" else value


def _load_tokenizer(tokenizer_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load the tokenizer. Install optional "
            "dependencies with `pip install -e '.[llm]'` or the train extras."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_id)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local-task SFT dataset.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--label-mode",
        choices=("won", "convincing", "all"),
        default="won",
    )
    parser.add_argument(
        "--weighting-mode",
        choices=("filter", "rwr"),
        default="rwr",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--loss-mask",
        choices=("action", "completion"),
        default="action",
        help="Competence default is action-only; 'completion' reproduces the "
        "historical full-response objective.",
    )
    parser.add_argument("--allow-thinking", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = base.load_manifest(args.manifest)
    if manifest["task_id"] != args.task:
        raise ValueError(f"manifest task_id={manifest['task_id']!r} != --task {args.task!r}")
    tokenizer = _load_tokenizer(args.tokenizer)
    examples, dataset_manifest = build_local_sft_dataset(
        manifest,
        framing=_framing_arg(args.framing),
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer,
        split=args.split,
        label_mode=args.label_mode,
        weighting_mode=args.weighting_mode,
        require_no_thinking=not args.allow_thinking,
        loss_mask_mode=args.loss_mask,
    )
    base.write_jsonl(args.out, examples)
    manifest_path = args.out.with_suffix(".manifest.json")
    base.write_json(manifest_path, dataset_manifest)
    print(f"wrote examples: {args.out}")
    print(f"wrote manifest: {manifest_path}")
    print(
        json.dumps(
            {
                "n_examples": dataset_manifest["n_examples"],
                "n_unique_examples": dataset_manifest["n_unique_examples"],
                "n_included_windows": dataset_manifest["n_included_windows"],
                "loss_mask_mode": dataset_manifest.get(
                    "loss_mask_mode", "completion"
                ),
                "label_counts": dataset_manifest["label_counts"],
                "multiplicity_histogram": dataset_manifest["multiplicity_histogram"],
                "skipped_record_counts": dataset_manifest["skipped_record_counts"],
                "token_accounting": dataset_manifest.get("token_accounting"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
