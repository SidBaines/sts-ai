#!/usr/bin/env python
"""Replay base-visited local-task states and query the native search teacher.

The live episode always follows the recorded source action, so collection stays
on the base policy's state distribution.  Every search runs on an internal clone;
teacher actions and root statistics are labels/provenance, never observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
from typing import Any

from sts_ai import provenance
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.lightspeed_import import import_lightspeed
from sts_ai.local_tasks import base, get_task
from sts_ai.local_tasks.runner import (
    replay_task_start,
    resolve_task_replay_action,
    windows_for_split,
)
from sts_ai.rollout import current_git_sha, prepare_decision
from sts_ai.search_policy_eval import root_visit_vote
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    NATIVE_SELECTED_ACTION,
    PUBLIC_OBSERVATION_VERSION,
    SEARCH_TEACHER_LABEL_VERSION,
    TEACHER_PRIVILEGE,
    consensus_summary,
    teacher_label,
)


def _csv_ints(raw: str, *, allow_empty: bool = False) -> list[int]:
    if not raw.strip():
        if allow_empty:
            return []
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    try:
        values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values and not allow_empty:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _validate_independent_seeds(
    values: list[int],
    *,
    name: str,
    default_random_engine: bool = False,
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate seeds")
    # libc++'s std::default_random_engine normalizes seed 0 to seed 1. The
    # searcher therefore produces byte-identical streams for 0 and 1, which can
    # create a false 2/3 consensus. Single-seed diagnostics may still use either.
    if default_random_engine and len(values) > 1 and 0 in values and 1 in values:
        raise ValueError(
            f"{name} seeds 0 and 1 alias under std::default_random_engine; "
            "use empirically distinct seeds such as 1,2,3"
        )


def _source_action(
    env: LightspeedHybridEnv,
    record: dict[str, Any],
    legal_actions: list[Any],
) -> tuple[int, dict[str, Any]]:
    selected = record.get("selected_action")
    if record.get("action_executed", True) is False or not isinstance(selected, dict):
        raise ValueError("source record has no executed selected_action")
    index = resolve_task_replay_action(env, selected)
    live_matches = [action for action in legal_actions if int(action.index) == index]
    if len(live_matches) != 1:
        raise ValueError(
            f"resolved source action index {index} is absent or ambiguous in the "
            "prepared legal-action view"
        )
    live = live_matches[0]
    return index, {
        "display_index": index,
        "bits": int(live.bits),
        "description": str(live.description),
        "recorded_bits": (
            int(selected["bits"]) if selected.get("bits") is not None else None
        ),
        "recorded_description": str(selected.get("description", "")),
    }


def _query_state(
    env: LightspeedHybridEnv,
    view: dict[str, Any],
    *,
    budgets: list[int],
    search_seeds: list[int],
    draw_order_seeds: list[int],
    selection_rule: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    max_budget = max(budgets)
    for simulations in budgets:
        # Hidden-order sensitivity is judged at the strongest budget. Repeating
        # every perturbation at weak budgets multiplies cost while confounding
        # search-budget noise with privileged-state sensitivity.
        draw_variants: list[int | None] = (
            [None, *draw_order_seeds] if simulations == max_budget else [None]
        )
        for search_seed in search_seeds:
            for draw_order_seed in draw_variants:
                try:
                    result = env.search_best_combat_action(
                        simulations,
                        search_seed=search_seed,
                        draw_order_seed=draw_order_seed,
                    )
                except ValueError as exc:
                    if (
                        selection_rule != "aggregated_root_visits"
                        or draw_order_seed is None
                        or "draw_order_seed is unsupported for draw-index card selection"
                        not in str(exc)
                    ):
                        raise
                    # Keep the planned hidden intervention in the denominator.
                    # Omitting it would inflate consensus on exactly the state
                    # where action bits cannot safely cross the shuffled clone.
                    result = {
                        "error": "unsupported_draw_index_hidden_order_intervention",
                        "error_detail": str(exc),
                        "selection_method": "abstain",
                        "simulations_requested": simulations,
                        "provenance": {
                            "search_seed": search_seed,
                            "draw_order_seed": draw_order_seed,
                            "live_state_mutated": False,
                        },
                    }
                    vote = {
                        "action_index": None,
                        "abstained": True,
                        "reason": "unsupported_draw_index_hidden_order_intervention",
                    }
                else:
                    vote = (
                        root_visit_vote(result, view["legal_actions"])
                    if selection_rule == AGGREGATED_ROOT_VISITS
                        else None
                    )
                label = teacher_label(
                    state_text=view["state_text"],
                    legal_actions=view["legal_actions"],
                    search_result=result,
                    teacher_vote=vote,
                    selection_rule=selection_rule,
                )
                label["query"] = {
                    "requested_simulations": simulations,
                    "search_seed": search_seed,
                    "draw_order_seed": draw_order_seed,
                }
                labels.append(label)

    reference_rows = [
        row for row in labels
        if int(row["query"]["requested_simulations"]) == max_budget
        and row["query"]["draw_order_seed"] is None
    ]
    reference = consensus_summary(reference_rows)
    reference["simulations"] = max_budget
    reference["eligible"] = (
        reference["consensus_action_index"] is not None
        and float(reference["consensus_fraction"]) >= 2 / 3
    )

    budget_consensus: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        rows = [
            row for row in labels
            if int(row["query"]["requested_simulations"]) == budget
            and row["query"]["draw_order_seed"] is None
        ]
        budget_consensus[str(budget)] = consensus_summary(rows)
    reference["budget_consensus"] = budget_consensus

    if draw_order_seeds:
        hidden_rows = [
            row for row in labels
            if int(row["query"]["requested_simulations"]) == max_budget
        ]
        reference["hidden_order_consensus"] = consensus_summary(hidden_rows)
    return labels, reference


def collect_window(
    manifest: dict[str, Any],
    window: dict[str, Any],
    *,
    task: Any,
    budgets: list[int],
    search_seeds: list[int],
    draw_order_seeds: list[int],
    state_selection: str,
    max_states: int,
    selection_rule: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    env = LightspeedHybridEnv(
        world_seed=int(window["world_seed"]),
        combat_control="llm",
        combat_observation="combat_public_v2",
        battle_simulations=50,
    )
    if env.combat_observation != "combat_public_v2":
        raise RuntimeError("search teacher collection requires combat_public_v2")
    replay_task_start(env, window, task)
    source_records = base.source_records_for_window(manifest, window)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    last_queried_turn: int | None = None

    for source_offset, source_record in enumerate(source_records):
        status, view = prepare_decision(env)
        if status != "ok" or view is None:
            raise RuntimeError(
                f"{window['window_id']} source offset {source_offset}: replay reached {status}"
            )
        if view["phase"] != "combat":
            raise RuntimeError(
                f"{window['window_id']} source offset {source_offset}: expected combat, "
                f"got {view['phase']}"
            )
        source_index, source_action = _source_action(
            env,
            source_record,
            view["legal_actions"],
        )
        counters["source_decisions"] += 1
        combat = (view["state"].get("combat") or {})
        turn = int(combat.get("turn", 0))
        should_query = state_selection == "every" or turn != last_queried_turn
        if max_states > 0 and len(rows) >= max_states:
            should_query = False

        if should_query:
            legal_dicts = view["legal_action_dicts"]
            labels, reference = _query_state(
                env,
                view,
                budgets=budgets,
                search_seeds=search_seeds,
                draw_order_seeds=draw_order_seeds,
                selection_rule=selection_rule,
            )
            last_queried_turn = turn
            counters["queried_states"] += 1
            counters["teacher_queries"] += len(labels)
            if not reference["eligible"]:
                counters["ambiguous_reference"] += 1
            rows.append(
                {
                    "task_id": task.task_id,
                    "window_id": window["window_id"],
                    "split": window.get("split"),
                    "world_seed": int(window["world_seed"]),
                    "source_stem": window["source_stem"],
                    "source_decision_index": int(window["start_index"]) + source_offset,
                    "turn": turn,
                    "player_hp": int(combat.get("player_cur_hp", 0)),
                    "observation_version": PUBLIC_OBSERVATION_VERSION,
                    "teacher_privilege": TEACHER_PRIVILEGE,
                    "teacher_selection_rule": selection_rule,
                    "state_text": view["state_text"],
                    "legal_actions": legal_dicts,
                    "public_state_hash": labels[0]["public_state_hash"],
                    "base_action": source_action,
                    "teacher_queries": labels,
                    "reference": reference,
                    "target": (
                        json.dumps(
                            {"action_index": reference["consensus_action_index"]},
                            separators=(",", ":"),
                        )
                        if reference["eligible"]
                        else None
                    ),
                }
            )

        # Follow the recorded base action after all non-mutating teacher queries.
        env.step(source_index)
        if task.completion_reason(env.summary()) is not None:
            break

    return rows, dict(counters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "holdout"), default="train")
    parser.add_argument("--simulations", default="1000,5000,50000")
    parser.add_argument(
        "--search-seeds",
        default="1,2,3",
        help="Independent search RNG seeds (0 and 1 must not be used together).",
    )
    parser.add_argument(
        "--draw-order-seeds",
        default="",
        help="Optional cloned draw-pile permutations in addition to the unmodified state.",
    )
    parser.add_argument(
        "--state-selection",
        choices=("every", "first_per_turn"),
        default="every",
    )
    parser.add_argument(
        "--max-states-per-window",
        type=int,
        default=0,
        help="0 means unlimited; useful for a cheap audit/smoke pass.",
    )
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument(
        "--selection-rule",
        choices=(AGGREGATED_ROOT_VISITS, NATIVE_SELECTED_ACTION),
        default=AGGREGATED_ROOT_VISITS,
        help="Per-search action vote. Aggregated root visits is the COMP-004A winner; "
        "native selection is retained only for historical replication.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = _csv_ints(args.simulations)
    if any(value < 1 for value in budgets):
        raise ValueError("simulation budgets must be >= 1")
    _validate_independent_seeds(budgets, name="--simulations")
    search_seeds = _csv_ints(args.search_seeds)
    draw_order_seeds = _csv_ints(args.draw_order_seeds, allow_empty=True)
    _validate_independent_seeds(
        search_seeds,
        name="--search-seeds",
        default_random_engine=True,
    )
    _validate_independent_seeds(draw_order_seeds, name="--draw-order-seeds")
    manifest = base.load_manifest(args.manifest)
    simulator_module = import_lightspeed()
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(f"manifest task_id={manifest['task_id']!r} != --task {task.task_id!r}")
    windows = windows_for_split(manifest, args.split)
    if args.limit_windows > 0:
        windows = windows[: args.limit_windows]

    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for position, window in enumerate(windows, start=1):
        window_rows, counters = collect_window(
            manifest,
            window,
            task=task,
            budgets=budgets,
            search_seeds=search_seeds,
            draw_order_seeds=draw_order_seeds,
            state_selection=args.state_selection,
            max_states=args.max_states_per_window,
            selection_rule=args.selection_rule,
        )
        rows.extend(window_rows)
        totals.update(counters)
        print(
            f"[{position}/{len(windows)}] {window['window_id']}: "
            f"{len(window_rows)} states",
            flush=True,
        )

    base.write_jsonl(args.out, rows)
    reference_actions = Counter(
        str(row["reference"]["consensus_action_index"])
        for row in rows if row["reference"]["eligible"]
    )
    manifest_out = {
        "kind": "search_teacher_labels",
        "version": SEARCH_TEACHER_LABEL_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "task_id": task.task_id,
        "split": args.split,
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": provenance.file_sha256(args.manifest),
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "teacher_selection_rule": args.selection_rule,
        "simulations": budgets,
        "search_seeds": search_seeds,
        "draw_order_seeds": draw_order_seeds,
        "state_selection": args.state_selection,
        "max_states_per_window": args.max_states_per_window,
        "n_windows": len(windows),
        "n_rows": len(rows),
        "counts": dict(totals),
        "reference_action_counts": dict(reference_actions),
        "interface_provenance": {
            "collector_sha256": provenance.file_sha256(Path(__file__)),
            "teacher_boundary_sha256": provenance.file_sha256(
                Path(__file__).resolve().parents[1] / "src/sts_ai/teacher.py"
            ),
            "python_serializer_sha256": provenance.file_sha256(
                Path(__file__).resolve().parents[1] / "src/sts_ai/lightspeed.py"
            ),
            "simulator_patch_sha256": provenance.file_sha256(
                Path(__file__).resolve().parents[1]
                / "patches/sts_lightspeed_python_api.patch"
            ),
            "simulator_binary_sha256": provenance.file_sha256(
                simulator_module.__file__
            ),
        },
    }
    base.write_json(args.out.with_suffix(".manifest.json"), manifest_out)
    print(json.dumps(manifest_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
