from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from sts_ai.interactive.replay import replay_actions
from sts_ai.local_tasks import base
from sts_ai.rollout import (
    append_jsonl,
    build_decision_record,
    build_rollout_meta,
    clamp_action_index,
    error_payload,
    prepare_decision,
    write_rollout_meta,
)
from sts_ai.schemas import DecisionRecord, RolloutResult
from sts_ai.seeding import derive_policy_seed


class LocalTaskEnv:
    """Thin env wrapper that makes a local task look terminal to rollout runners.

    The underlying simulator is still a normal ``LightspeedHybridEnv``. The
    wrapper only changes ``is_terminal`` so generic orchestrators stop when the
    task episode is over (e.g. Gremlin Nob dies) rather than continuing the full
    run.
    """

    def __init__(self, env: Any, task: Any):
        self._env = env
        self._task = task

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def is_terminal(self) -> bool:
        return self._env.is_terminal() or self._task.completion_reason(
            self._env.summary()
        ) is not None


def replay_task_start(env: Any, window: dict[str, Any]) -> int:
    return replay_actions(env, window.get("pre_actions") or [])


def run_local_task_episode(
    *,
    task: Any,
    env: Any,
    agent: Any,
    window: dict[str, Any],
    max_decisions: int,
    output_path: Path | None = None,
    run_meta: dict[str, Any] | None = None,
    rollout_index: int = 0,
    policy_seed: int | None = None,
) -> RolloutResult:
    world_seed = int(env.world_seed)
    if policy_seed is None:
        policy_seed = derive_policy_seed(world_seed, rollout_index)
    agent.reseed(policy_seed)

    decisions: list[DecisionRecord] = []
    stopped_reason = "task_complete"
    error: dict[str, Any] | None = None

    for decision_index in range(max_decisions):
        try:
            status, view = prepare_decision(env)
        except Exception as exc:  # noqa: BLE001
            stopped_reason = "simulator_error"
            error = error_payload(exc, "advance_to_decision", decision_index)
            break
        if status != "ok":
            stopped_reason = status
            break

        agent_decision = agent.choose_action(view["state_text"], view["legal_actions"])
        action_index = clamp_action_index(agent_decision, len(view["legal_actions"]))
        if not agent_decision.valid:
            record = build_decision_record(
                world_seed=world_seed,
                decision_index=decision_index,
                state=view["state"],
                state_text=view["state_text"],
                legal_action_dicts=view["legal_action_dicts"],
                selected_action_dict={},
                agent_decision=agent_decision,
                after_state=env.summary(),
                phase=view["phase"],
                policy_seed=policy_seed,
                rollout_index=rollout_index,
                action_executed=False,
            )
            decisions.append(record)
            if output_path is not None:
                append_jsonl(output_path, asdict(record))
            stopped_reason = "agent_invalid"
            break

        try:
            selected = env.step(action_index)
        except Exception as exc:  # noqa: BLE001
            stopped_reason = "simulator_error"
            error = error_payload(exc, "step", decision_index)
            break

        record = build_decision_record(
            world_seed=world_seed,
            decision_index=decision_index,
            state=view["state"],
            state_text=view["state_text"],
            legal_action_dicts=view["legal_action_dicts"],
            selected_action_dict=env.action_dict(selected),
            agent_decision=agent_decision,
            after_state=env.summary(),
            phase=view["phase"],
            policy_seed=policy_seed,
            rollout_index=rollout_index,
        )
        decisions.append(record)
        if output_path is not None:
            append_jsonl(output_path, asdict(record))

        completion = task.completion_reason(record.after_state)
        if completion is not None:
            stopped_reason = completion
            break
    else:
        stopped_reason = "max_decisions"

    result = RolloutResult(
        world_seed=world_seed,
        decisions=decisions,
        terminal_state=env.summary(),
        stopped_reason=stopped_reason,
        error=error,
        policy_seed=policy_seed,
        rollout_index=rollout_index,
    )
    if output_path is not None:
        task_metrics = task.metrics_from_episode(
            decisions,
            result.terminal_state,
            stopped_reason,
            window,
        )
        meta_extra = dict((run_meta or {}).get("extra", {}))
        meta_extra["local_task"] = {
            "task_id": task.task_id,
            "window_id": window["window_id"],
            "source_stem": window["source_stem"],
            "split": window.get("split"),
            "label": task_metrics["label"],
            "reward": task_metrics["reward"],
            "metrics": task_metrics,
        }
        merged_meta = dict(run_meta or {})
        merged_meta["extra"] = meta_extra
        write_rollout_meta(output_path, build_rollout_meta(result, env, agent, merged_meta))
    return result


def run_local_task_episodes(
    specs: list[tuple[int, int]],
    make_env: Callable[[int], Any],
    agent: Any,
    *,
    output_for: Callable[[int, int], Path | None],
    concurrency: int,
    max_decisions: int,
    run_meta: dict[str, Any] | None = None,
    hint_cfg: Any | None = None,
    max_retries: int | None = None,
    task: Any,
    windows_by_seed: dict[int, dict[str, Any]],
) -> list[RolloutResult]:
    if hint_cfg is not None:
        raise ValueError("hinting is not supported for local-task MLX episodes")
    _ = concurrency, max_retries
    results: list[RolloutResult] = []
    for world_seed, rollout_index in specs:
        window = windows_by_seed[int(world_seed)]
        env = make_env(int(world_seed))
        results.append(
            run_local_task_episode(
                task=task,
                env=env,
                agent=agent,
                window=window,
                max_decisions=max_decisions,
                output_path=output_for(world_seed, rollout_index),
                run_meta=run_meta,
                rollout_index=rollout_index,
            )
        )
    return results


def windows_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [window for window in manifest["windows"] if window.get("split") == split]


def one_window_per_seed(windows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for window in windows:
        seed = int(window["world_seed"])
        if seed in by_seed:
            raise ValueError(f"multiple local-task windows for world_seed={seed}")
        by_seed[seed] = window
    return by_seed
