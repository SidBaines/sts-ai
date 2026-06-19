#!/usr/bin/env python
"""Build a policy-gradient JSONL dataset from rollout traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts_ai.prompting import NEUTRAL_FRAME
from sts_ai.train.pg_dataset import build_pg_dataset


def _framing_arg(value: str) -> str:
    if value == "neutral":
        return NEUTRAL_FRAME
    return value


def _load_tokenizer(tokenizer_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load the tokenizer. Install training "
            "dependencies with `pip install -e '.[train-cuda]'` or "
            "`pip install -e '.[llm]'`."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_id)


def _print_report(manifest: dict[str, Any]) -> None:
    report = manifest["advantage_report"]
    lines = [
        f"mode: {manifest['mode']}",
        f"n_examples: {manifest['n_examples']}",
        f"n_trajectories_with_advantage: {manifest['n_trajectories_with_advantage']}",
        f"advantage_min: {report['advantage_min']}",
        f"advantage_max: {report['advantage_max']}",
        f"advantage_mean: {report['advantage_mean']}",
        "skipped_record_counts: "
        + json.dumps(manifest["skipped_record_counts"], sort_keys=True),
    ]
    print("\n".join(lines))


def _write_outputs(out: Path, examples: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote examples: {out}")
    print(f"wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a policy-gradient dataset from rollouts."
    )
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer id.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--framing", default="neutral")
    parser.add_argument(
        "--mode",
        choices=("offline", "group"),
        default="offline",
    )
    parser.add_argument(
        "--baseline",
        choices=("median", "mean"),
        default="median",
    )
    parser.add_argument("--std-norm", dest="std_norm", action="store_true", default=True)
    parser.add_argument("--no-std-norm", dest="std_norm", action="store_false")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--min-act", type=int, default=1)
    parser.add_argument(
        "--drop-phase",
        action="append",
        choices=("out_of_combat", "combat"),
        default=[],
    )
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="Allow native/prompted reasoning modes in rollout metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer)

    examples, manifest = build_pg_dataset(
        args.rollout_dir,
        framing=_framing_arg(args.framing),
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer,
        mode=args.mode,
        baseline=args.baseline,
        std_norm=args.std_norm,
        eps=args.eps,
        min_act=args.min_act,
        require_no_thinking=not args.allow_thinking,
        drop_phases=tuple(args.drop_phase),
    )

    _print_report(manifest)
    _write_outputs(args.out, examples, manifest)


if __name__ == "__main__":
    main()
