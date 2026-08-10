#!/usr/bin/env python
"""Compare winning-sequence and most-visited-root native search policies."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import statistics
from typing import Any

from sts_ai import provenance
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.lightspeed_import import import_lightspeed
from sts_ai.local_tasks import base, get_task
from sts_ai.local_tasks.runner import replay_task_start, windows_for_split
from sts_ai.rollout import current_git_sha, prepare_decision
from sts_ai.search_policy_eval import (
    action_vote_consensus,
    root_visit_vote,
    winning_sequence_vote,
)
from sts_ai.teacher import public_observation_hash


POLICIES = ("winning_sequence", "most_visited_root")


def _csv_ints(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _json_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop native action objects while retaining all scalar search evidence."""
    return {
        str(key): _json_search_result(value)
        for key, value in result.items()
        if str(key) != "action"
    } if isinstance(result, dict) else (
        [_json_search_result(value) for value in result]
        if isinstance(result, (list, tuple))
        else result
    )


def _query_decision(
    env: LightspeedHybridEnv,
    view: dict[str, Any],
    *,
    simulations: int,
    search_seeds: list[int],
) -> dict[str, Any]:
    seed_queries: list[dict[str, Any]] = []
    winning_votes: list[int | None] = []
    root_votes: list[int | None] = []
    for search_seed in search_seeds:
        result = env.search_best_combat_action(
            simulations,
            search_seed=search_seed,
            draw_order_seed=None,
        )
        winning = winning_sequence_vote(result, view["legal_actions"])
        root = root_visit_vote(result, view["legal_actions"])
        winning_votes.append(winning["action_index"])
        root_votes.append(root["action_index"])
        seed_queries.append(
            {
                "search_seed": search_seed,
                "winning_sequence_vote": winning,
                "most_visited_root_vote": root,
                "search": _json_search_result(result),
            }
        )

    return {
        "simulations": simulations,
        "search_seeds": search_seeds,
        "seed_queries": seed_queries,
        "consensus": {
            "winning_sequence": action_vote_consensus(winning_votes),
            "most_visited_root": action_vote_consensus(root_votes),
        },
    }


def _failed_metrics(
    task: Any,
    decisions: list[dict[str, Any]],
    terminal_state: dict[str, Any],
    stopped_reason: str,
    window: dict[str, Any],
) -> dict[str, Any]:
    partial = task.metrics_from_episode(
        decisions,
        terminal_state,
        stopped_reason,
        window,
    )
    return {
        "entry_hp": partial["entry_hp"],
        "exit_hp": partial["exit_hp"],
        "hp_loss": None,
        "partial_hp_loss_at_failure": partial["hp_loss"],
        "survived": False,
        "reward": -1.0,
        "n_decisions": len(decisions),
        "n_turns": partial["n_turns"],
    }


def evaluate_window(
    manifest: dict[str, Any],
    window: dict[str, Any],
    *,
    task: Any,
    policy: str,
    simulations: int,
    search_seeds: list[int],
    max_decisions: int,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy!r}")
    env = LightspeedHybridEnv(
        world_seed=int(window["world_seed"]),
        combat_control="llm",
        combat_observation="combat_public_v2",
        battle_simulations=50,
    )
    if env.combat_observation != "combat_public_v2":
        raise RuntimeError("search root-policy evaluation requires combat_public_v2")
    replay_task_start(env, window, task)
    decisions: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    stopped_reason = "task_complete"
    error: dict[str, str] | None = None

    for decision_index in range(max_decisions):
        try:
            status, view = prepare_decision(env)
            if status != "ok" or view is None:
                stopped_reason = status
                break
            if view["phase"] != "combat":
                raise RuntimeError(f"expected combat decision, got {view['phase']!r}")
            evidence = _query_decision(
                env,
                view,
                simulations=simulations,
                search_seeds=search_seeds,
            )
            consensus = evidence["consensus"][policy]
            query_record = {
                "decision_index": decision_index,
                "turn": int((view["state"].get("combat") or {}).get("turn", 0)),
                "public_state_hash": public_observation_hash(
                    view["state_text"],
                    view["legal_actions"],
                ),
                "legal_actions": view["legal_action_dicts"],
                "query": evidence,
                "selected_action_index": consensus["action_index"],
            }
            query_records.append(query_record)
            if not consensus["eligible"]:
                stopped_reason = "ambiguous_consensus"
                break

            action_index = int(consensus["action_index"])
            selected = env.step(action_index)
            after_state = env.summary()
            selected_action = env.action_dict(selected)
            query_record["selected_action"] = selected_action
            decisions.append(
                {
                    "state": view["state"],
                    "selected_action": selected_action,
                    "after_state": after_state,
                }
            )
            completion = task.completion_reason(after_state)
            if completion is not None:
                stopped_reason = completion
                break
        except Exception as exc:  # noqa: BLE001 - retain failure in diagnostic artifact.
            stopped_reason = "simulator_error"
            error = {"type": type(exc).__name__, "message": str(exc)}
            break
    else:
        stopped_reason = "max_decisions"

    terminal_state = env.summary()
    if stopped_reason in ("task_complete", "player_loss"):
        metrics = task.metrics_from_episode(
            decisions,
            terminal_state,
            stopped_reason,
            window,
        )
        metrics["partial_hp_loss_at_failure"] = None
    else:
        metrics = _failed_metrics(
            task,
            decisions,
            terminal_state,
            stopped_reason,
            window,
        )
    return {
        "task_id": task.task_id,
        "window_id": window["window_id"],
        "world_seed": int(window["world_seed"]),
        "split": window.get("split"),
        "policy": policy,
        "stopped_reason": stopped_reason,
        "error": error,
        "metrics": metrics,
        "n_search_queries": sum(
            len(record["query"]["seed_queries"])
            for record in query_records
        ),
        "decisions": query_records,
    }


def _policy_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(episode["metrics"]["reward"]) for episode in episodes]
    completed = [
        episode
        for episode in episodes
        if episode["stopped_reason"] in ("task_complete", "player_loss")
    ]
    hp_losses = [
        int(episode["metrics"]["hp_loss"])
        for episode in completed
        if episode["metrics"]["hp_loss"] is not None
    ]
    stop_counts = Counter(str(episode["stopped_reason"]) for episode in episodes)
    n_wins = sum(bool(episode["metrics"]["survived"]) for episode in episodes)
    decision_consensus = [
        decision["query"]["consensus"][episode["policy"]]
        for episode in episodes
        for decision in episode["decisions"]
    ]
    selected_descriptions = Counter(
        str(decision["selected_action"]["description"])
        for episode in episodes
        for decision in episode["decisions"]
        if isinstance(decision.get("selected_action"), dict)
    )
    return {
        "n_windows": len(episodes),
        "n_completed": len(completed),
        "n_wins": n_wins,
        "win_rate_all_starts": n_wins / len(episodes) if episodes else None,
        "mean_reward_all_starts": statistics.mean(rewards) if rewards else None,
        "mean_hp_loss_completed": statistics.mean(hp_losses) if hp_losses else None,
        "stopped_reason_counts": dict(stop_counts),
        "n_decisions_queried": len(decision_consensus),
        "n_consensus_eligible": sum(
            bool(consensus["eligible"]) for consensus in decision_consensus
        ),
        "consensus_eligible_fraction": (
            sum(bool(consensus["eligible"]) for consensus in decision_consensus)
            / len(decision_consensus)
            if decision_consensus
            else None
        ),
        "n_unanimous_decisions": sum(
            bool(consensus["unanimous"]) for consensus in decision_consensus
        ),
        "unanimous_decision_fraction": (
            sum(bool(consensus["unanimous"]) for consensus in decision_consensus)
            / len(decision_consensus)
            if decision_consensus
            else None
        ),
        "selected_action_description_counts": dict(selected_descriptions),
        "n_native_search_queries": sum(
            int(episode["n_search_queries"]) for episode in episodes
        ),
    }


def _paired_decision_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (str(episode["window_id"]), str(episode["policy"])): episode
        for episode in episodes
    }
    n_aligned_positions = 0
    n_same_public_state = 0
    n_same_selected_action = 0
    for window_id in sorted({str(episode["window_id"]) for episode in episodes}):
        winning = by_key.get((window_id, "winning_sequence"))
        root = by_key.get((window_id, "most_visited_root"))
        if winning is None or root is None:
            continue
        for winning_decision, root_decision in zip(
            winning["decisions"],
            root["decisions"],
        ):
            n_aligned_positions += 1
            if winning_decision["public_state_hash"] != root_decision["public_state_hash"]:
                continue
            n_same_public_state += 1
            winning_action = winning_decision.get("selected_action_index")
            root_action = root_decision.get("selected_action_index")
            if winning_action is not None and winning_action == root_action:
                n_same_selected_action += 1
    return {
        "definition": "same decision position and identical public-state hash",
        "n_aligned_decision_positions": n_aligned_positions,
        "n_same_public_state": n_same_public_state,
        "n_same_selected_action_on_shared_state": n_same_selected_action,
        "selected_action_agreement_on_shared_states": (
            n_same_selected_action / n_same_public_state
            if n_same_public_state
            else None
        ),
    }


def _build_report(
    *,
    status: str,
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    simulator_module: Any,
) -> dict[str, Any]:
    source_manifest = Path(args.manifest)
    by_policy = {
        policy: [episode for episode in episodes if episode["policy"] == policy]
        for policy in POLICIES
    }
    return {
        "kind": "search_root_policy_comparison",
        "version": 1,
        "status": status,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "task_id": args.task,
        "split": args.split,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": provenance.file_sha256(source_manifest),
        "simulations_per_query": args.simulations,
        "search_seeds": args.search_seeds,
        "consensus_rule": {
            "min_fraction": 2 / 3,
            "denominator": "all_three_search_seeds_including_abstentions",
            "tie_handling": "fail_closed_policy_window_reward_minus_one",
            "gameplay_fallback": None,
        },
        "policies": {
            "winning_sequence": "first action of best observed winning sequence",
            "most_visited_root": (
                "unique maximum visits after aggregating valid native root edges "
                "by displayed action within each search seed"
            ),
        },
        "n_expected_windows_per_policy": args.n_expected_windows,
        "summary": {
            policy: _policy_summary(policy_episodes)
            for policy, policy_episodes in by_policy.items()
        },
        "paired_decision_summary": _paired_decision_summary(episodes),
        "interface_provenance": {
            "evaluator_sha256": provenance.file_sha256(Path(__file__)),
            "policy_helpers_sha256": provenance.file_sha256(
                Path(__file__).resolve().parents[1]
                / "src/sts_ai/search_policy_eval.py"
            ),
            "python_serializer_sha256": provenance.file_sha256(
                Path(__file__).resolve().parents[1] / "src/sts_ai/lightspeed.py"
            ),
            "simulator_patch_sha256": provenance.file_sha256(
                Path(__file__).resolve().parents[1]
                / "patches/sts_lightspeed_python_api.patch"
            ),
            "simulator_binary_path": str(simulator_module.__file__),
            "simulator_binary_sha256": provenance.file_sha256(
                simulator_module.__file__
            ),
        },
        "episode_records": [
            {
                "window_id": episode["window_id"],
                "world_seed": episode["world_seed"],
                "policy": episode["policy"],
                "stopped_reason": episode["stopped_reason"],
                "metrics": episode["metrics"],
                "n_search_queries": episode["n_search_queries"],
                "path": str(_episode_path(args.out, episode)),
                "sha256": provenance.file_sha256(_episode_path(args.out, episode)),
            }
            for episode in episodes
        ],
    }


def _episode_path(report_path: Path, episode: dict[str, Any]) -> Path:
    return (
        Path(report_path).parent
        / "episodes"
        / f"{episode['window_id']}.{episode['policy']}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gremlin_nob")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "holdout"), default="holdout")
    parser.add_argument("--simulations", type=int, default=50_000)
    parser.add_argument("--search-seeds", type=_csv_ints, default=[1, 2, 3])
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.simulations < 1:
        parser.error("--simulations must be positive")
    if len(args.search_seeds) != 3 or len(set(args.search_seeds)) != 3:
        parser.error("--search-seeds must contain exactly three distinct seeds")
    if 0 in args.search_seeds and 1 in args.search_seeds:
        parser.error("search seeds 0 and 1 alias under std::default_random_engine")
    if args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --overwrite")
    manifest = base.load_manifest(args.manifest)
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(
            f"manifest task_id={manifest['task_id']!r} != --task {task.task_id!r}"
        )
    windows = windows_for_split(manifest, args.split)
    if args.limit_windows > 0:
        windows = windows[: args.limit_windows]
    args.n_expected_windows = len(windows)
    simulator_module = import_lightspeed()

    episodes: list[dict[str, Any]] = []
    for window_position, window in enumerate(windows, start=1):
        for policy in POLICIES:
            episode = evaluate_window(
                manifest,
                window,
                task=task,
                policy=policy,
                simulations=args.simulations,
                search_seeds=args.search_seeds,
                max_decisions=args.max_decisions,
            )
            episodes.append(episode)
            base.write_json(_episode_path(args.out, episode), episode)
            base.write_json(
                args.out,
                _build_report(
                    status="running",
                    args=args,
                    episodes=episodes,
                    simulator_module=simulator_module,
                ),
            )
            print(
                f"[{window_position}/{len(windows)}] {window['window_id']} "
                f"{policy}: {episode['stopped_reason']} "
                f"reward={episode['metrics']['reward']}",
                flush=True,
            )

    report = _build_report(
        status="complete",
        args=args,
        episodes=episodes,
        simulator_module=simulator_module,
    )
    base.write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
