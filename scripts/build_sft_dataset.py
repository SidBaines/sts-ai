#!/usr/bin/env python
"""Build an offline SFT JSONL dataset from rollout traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sts_ai.prompting import NEUTRAL_FRAME
from sts_ai.train.dataset_builder import build_dataset


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
    report = manifest["filter_report"]
    lines = [
        f"n_rollouts_discovered: {manifest['n_rollouts_discovered']}",
        f"n_missing_meta: {manifest['n_missing_meta']}",
        f"n_kept_trajectories: {manifest['n_kept_trajectories']}",
        f"n_examples: {manifest['n_examples']}",
        f"n_positives: {report['n_positives']}",
        f"fallback_engaged: {report['fallback_engaged']}",
        f"threshold_floor: {report['threshold_floor']}",
        "stopped_reason_counts: "
        + json.dumps(report["stopped_reason_counts"], sort_keys=True),
        "kept_stopped_reason_counts: "
        + json.dumps(report["kept_stopped_reason_counts"], sort_keys=True),
        f"agent_invalid_rate: {report['agent_invalid_rate']:.6f}",
        "skipped_record_counts: "
        + json.dumps(manifest["skipped_record_counts"], sort_keys=True),
    ]
    if manifest.get("weighting_mode") == "rwr":
        rwr_report = manifest["rwr_report"]
        lines.extend(
            [
                "weighting_mode: rwr",
                f"rwr_beta: {rwr_report['beta']}",
                f"rwr_baseline_value: {rwr_report['baseline_value']}",
                "rwr_multiplicity_histogram: "
                + json.dumps(rwr_report["multiplicity_histogram"], sort_keys=True),
                f"n_unique_examples: {manifest['n_unique_examples']}",
            ]
        )
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
    parser = argparse.ArgumentParser(description="Build an offline SFT dataset from rollouts.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer id.")
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--min-act", type=int, default=1)
    parser.add_argument("--min-positives", type=int, default=20)
    parser.add_argument("--fallback-floor-quantile", type=float, default=0.8)
    parser.add_argument(
        "--weighting-mode",
        choices=("filter", "rwr"),
        default="filter",
    )
    parser.add_argument("--rwr-beta", type=float, default=5.0)
    parser.add_argument(
        "--rwr-baseline",
        choices=("median", "mean"),
        default="median",
    )
    parser.add_argument("--rwr-max-multiplier", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="Allow native/prompted reasoning modes in rollout metadata.",
    )
    parser.add_argument(
        "--drop-phase",
        action="append",
        choices=("out_of_combat", "combat"),
        default=[],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer)

    examples, manifest = build_dataset(
        args.rollout_dir,
        framing=_framing_arg(args.framing),
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer,
        min_act=args.min_act,
        fallback_floor_quantile=args.fallback_floor_quantile,
        min_positives=args.min_positives,
        weighting_mode=args.weighting_mode,
        rwr_beta=args.rwr_beta,
        rwr_baseline=args.rwr_baseline,
        rwr_max_multiplier=args.rwr_max_multiplier,
        require_no_thinking=not args.allow_thinking,
        drop_phases=tuple(args.drop_phase),
    )

    _print_report(manifest)
    if args.weighting_mode == "rwr":
        print("RWR mode: sparsity guardrail not applicable")
    elif manifest["filter_report"]["fallback_engaged"] and not args.allow_fallback:
        print(
            "Refusing to write fallback dataset: positives were below "
            "the threshold. Re-run with --allow-fallback to emit the "
            "floor-quantile fallback dataset."
        )
        sys.exit(2)

    _write_outputs(args.out, examples, manifest)


if __name__ == "__main__":
    main()
