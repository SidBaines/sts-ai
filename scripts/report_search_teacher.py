#!/usr/bin/env python
"""Summarize a search-teacher stability/privilege JSONL collection."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from typing import Any

from sts_ai import provenance
from sts_ai.local_tasks import base
from sts_ai.teacher import (
    PUBLIC_OBSERVATION_VERSION,
    validate_search_teacher_manifest,
    validate_search_teacher_rows,
)


def _budget_summary(row: dict[str, Any], budget: int) -> dict[str, Any]:
    return (row["reference"].get("budget_consensus") or {})[str(budget)]


def build_report(
    rows: list[dict[str, Any]],
    *,
    source_labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    teacher_selection_rule = validate_search_teacher_rows(rows)
    budgets = sorted(
        int(key)
        for key in (rows[0]["reference"].get("budget_consensus") or {})
    )
    if not budgets:
        raise ValueError("teacher rows have no budget consensus")
    reference_budget = max(budgets)
    collection_budget = 5000 if 5000 in budgets else reference_budget

    per_state: list[dict[str, Any]] = []
    native_selection_methods: Counter[str] = Counter()
    predicted_hp: list[float] = []
    for row in rows:
        low = _budget_summary(row, collection_budget)
        high = _budget_summary(row, reference_budget)
        five_k = (
            _budget_summary(row, 5000)
            if 5000 in budgets
            else None
        )
        fifty_k = (
            _budget_summary(row, 50000)
            if 50000 in budgets
            else None
        )
        hidden = row["reference"].get("hidden_order_consensus")
        high_action = high.get("consensus_action_index")
        low_action = low.get("consensus_action_index")
        five_k_action = (
            five_k.get("consensus_action_index")
            if five_k is not None
            else None
        )
        fifty_k_action = (
            fifty_k.get("consensus_action_index")
            if fifty_k is not None
            else None
        )
        five_k_fraction = (
            float(five_k.get("consensus_fraction", 0.0))
            if five_k is not None
            else None
        )
        fifty_k_fraction = (
            float(fifty_k.get("consensus_fraction", 0.0))
            if fifty_k is not None
            else None
        )
        hidden_fraction = (
            float(hidden["consensus_fraction"])
            if isinstance(hidden, dict)
            else None
        )
        hidden_action = (
            hidden.get("consensus_action_index")
            if isinstance(hidden, dict)
            else None
        )
        hidden_matches_reference = (
            hidden_action == high_action
            if isinstance(hidden, dict) and high_action is not None
            else None
        )
        hidden_matches_fifty_k = (
            hidden_action == fifty_k_action
            if isinstance(hidden, dict) and fifty_k_action is not None
            else None
        )
        fifty_k_reference_is_usable = (
            fifty_k_action is not None
            and not bool((fifty_k or {}).get("tied", False))
            and fifty_k_fraction is not None
            and fifty_k_fraction >= 2 / 3
        )
        hidden_order_is_stable = (
            isinstance(hidden, dict)
            and hidden_action == fifty_k_action
            and hidden_fraction is not None
            and hidden_fraction >= 2 / 3
        )
        eligible_for_direct_50k_collection = (
            fifty_k_reference_is_usable and hidden_order_is_stable
        )
        eligible_for_5k_collection = (
            eligible_for_direct_50k_collection
            and five_k_action == fifty_k_action
            and not bool((five_k or {}).get("tied", False))
            and five_k_fraction is not None
            and five_k_fraction >= 2 / 3
        )
        base_action = int(row["base_action"]["display_index"])
        per_state.append(
            {
                "window_id": row["window_id"],
                "turn": row["turn"],
                "public_state_hash": row["public_state_hash"],
                "base_action_index": base_action,
                "collection_action_index": low_action,
                "reference_action_index": high_action,
                "five_k_action_index": five_k_action,
                "five_k_consensus_fraction": five_k_fraction,
                "fifty_k_action_index": fifty_k_action,
                "fifty_k_consensus_fraction": fifty_k_fraction,
                "budget_agreement": low_action is not None and low_action == high_action,
                "reference_unanimous": bool(high.get("unanimous", False)),
                "collection_unanimous": bool(low.get("unanimous", False)),
                "hidden_consensus_fraction": hidden_fraction,
                "hidden_consensus_action_index": hidden_action,
                "hidden_action_matches_reference": hidden_matches_reference,
                "hidden_action_matches_50k": hidden_matches_fifty_k,
                "teacher_differs_from_base": high_action is not None and high_action != base_action,
                "eligible_for_5k_collection": eligible_for_5k_collection,
                "eligible_for_direct_50k_collection": (
                    eligible_for_direct_50k_collection
                ),
            }
        )
        for query in row.get("teacher_queries") or []:
            search = query.get("search") or {}
            native_selection_methods[
                str(search.get("selection_method", "unknown"))
            ] += 1
            if search.get("predicted_player_hp") is not None:
                predicted_hp.append(float(search["predicted_player_hp"]))

    n = len(per_state)
    agreement_rows = [state for state in per_state if state["budget_agreement"]]
    hidden_values = [
        state["hidden_consensus_fraction"]
        for state in per_state
        if state["hidden_consensus_fraction"] is not None
    ]
    hidden_agreements = [
        state["hidden_action_matches_reference"]
        for state in per_state
        if state["hidden_action_matches_reference"] is not None
    ]
    unanimous_by_budget = {
        str(budget): sum(
            bool(_budget_summary(row, budget).get("unanimous", False))
            for row in rows
        ) / n
        for budget in budgets
    }
    return {
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": teacher_selection_rule,
        "source_labels": source_labels,
        "n_states": n,
        "n_windows": len({str(row["window_id"]) for row in rows}),
        "budgets": budgets,
        "collection_budget": collection_budget,
        "reference_budget": reference_budget,
        "collection_vs_reference_agreement": len(agreement_rows) / n,
        "unanimous_rate_by_budget": unanimous_by_budget,
        "hidden_order": {
            "n_audited": len(hidden_values),
            "mean_consensus_fraction": statistics.mean(hidden_values) if hidden_values else None,
            "fraction_at_least_two_thirds": (
                sum(value >= 2 / 3 for value in hidden_values) / len(hidden_values)
                if hidden_values else None
            ),
            "fraction_matching_reference_action": (
                sum(hidden_agreements) / len(hidden_agreements)
                if hidden_agreements else None
            ),
        },
        "teacher_differs_from_base_rate": sum(
            state["teacher_differs_from_base"] for state in per_state
        ) / n,
        "eligible_for_5k_collection": {
            "n": sum(state["eligible_for_5k_collection"] for state in per_state),
            "fraction": sum(
                state["eligible_for_5k_collection"] for state in per_state
            ) / n,
        },
        "eligible_for_direct_50k_collection": {
            "n": sum(
                state["eligible_for_direct_50k_collection"] for state in per_state
            ),
            "fraction": sum(
                state["eligible_for_direct_50k_collection"] for state in per_state
            ) / n,
        },
        "native_search_selection_method_counts": dict(native_selection_methods),
        "predicted_player_hp_mean": statistics.mean(predicted_hp) if predicted_hp else None,
        "decision": {
            "use_5k_default": (
                collection_budget == 5000
                and len(agreement_rows) / n >= 0.90
                and unanimous_by_budget.get("5000", 0.0) >= 0.80
            ),
            "thresholds": {
                "budget_agreement": 0.90,
                "search_seed_unanimity_at_5k": 0.80,
                "row_hidden_consensus": 2 / 3,
                "row_hidden_action_must_match_reference": True,
            },
        },
        "per_state": per_state,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = base.load_jsonl(args.labels)
    selection_rule = validate_search_teacher_rows(rows)
    manifest_path = args.labels.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(
            f"search-teacher labels require adjacent manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("search-teacher label manifest is not a JSON object")
    validate_search_teacher_manifest(
        manifest,
        n_rows=len(rows),
        selection_rule=selection_rule,
    )
    report = build_report(
        rows,
        source_labels={
            "path": str(args.labels.resolve()),
            "sha256": provenance.file_sha256(args.labels),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": provenance.file_sha256(manifest_path),
            "manifest_version": manifest["version"],
            "source_manifest": manifest.get("source_manifest"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        },
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
