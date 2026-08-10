from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Callable

from sts_ai.interactive.replay import ReplayError, resolve_action_index
from sts_ai.local_tasks import base
from sts_ai.local_tasks.start_state import validate_stored_start_state_signature
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


_PUBLIC_CARD_UPGRADE_RE = re.compile(r"\+\d*(?= \(cost|$)")
_PUBLIC_CARD_VALUE_RE = re.compile(r"=-?\d+(?= \(cost|$)")
_PUBLIC_CARD_TYPE_RE = re.compile(
    r" \[(?:Attack|Skill|Power|Curse|Status)\](?= \(cost|$)"
)
_GENERATED_OPTION_RE = re.compile(
    r"^select card for (?P<task>DISCOVERY|CODEX) \(option \d+\)$"
)
_POTION_TARGET_SLOT_RE = re.compile(
    r"^(?P<label>drink potion .+ -> .+) \[enemy \d+\]$"
)


def _without_public_card_adornments(description: str) -> str:
    """Normalize combat card details absent from historical source traces."""
    description = _PUBLIC_CARD_TYPE_RE.sub("", description)
    if description.startswith("select card for "):
        # v2 selection labels use the full public instance label so same-name
        # cards with different current cost/free/retain state remain distinct.
        # Historical traces carried the name only; exact action bits still guard
        # this migration path before the normalized comparison is accepted.
        description = re.sub(r" \(cost [^)]*\)$", "", description)
    description = _PUBLIC_CARD_UPGRADE_RE.sub("", description)
    description = _PUBLIC_CARD_VALUE_RE.sub("", description)
    return description.replace("(cost X)", "(cost -1)").replace(
        "(cost unplayable)", "(cost -2)"
    )


def _generated_option_label_matches(recorded: str, current: str) -> bool:
    match = _GENERATED_OPTION_RE.fullmatch(recorded)
    return bool(match and current.startswith(f"select card for {match['task']}:"))


def _historical_potion_target_matches(recorded: str, current: str) -> bool:
    """Match the pre-disambiguation label only when action bits also match."""
    current_match = _POTION_TARGET_SLOT_RE.fullmatch(current)
    return bool(current_match and current_match["label"] == recorded)


def resolve_task_replay_action(env: Any, action: dict[str, Any]) -> int:
    """Resolve a historical local-task action without accepting semantic drift."""
    try:
        return resolve_action_index(
            env,
            action.get("bits"),
            str(action.get("description", "")),
            action.get("index"),
        )
    except ReplayError:
        # Historical combat traces used CardInstance::getName(), which omitted
        # the upgrade marker. Public-state serialization corrected the live
        # label to e.g. ``play Bash+ (cost 2)`` and now includes public mutable
        # values on a few cards. Keep old local-task source windows replayable,
        # but only through an exact-bits match whose text is otherwise identical
        # after removing those card-label adornments.
        recorded = _without_public_card_adornments(
            str(action.get("description", ""))
        )
        bits = action.get("bits")
        candidates = [
            legal
            for legal in env.legal_actions()
            if bits is not None
            and int(legal.bits) == int(bits)
            and (
                _without_public_card_adornments(legal.description) == recorded
                or _generated_option_label_matches(recorded, legal.description)
                or _historical_potion_target_matches(recorded, legal.description)
            )
        ]
        if len(candidates) == 1:
            return candidates[0].index
        raise


# Compatibility for existing callers/tests while the public resolver above is
# adopted by teacher collection as well as start replay.
_resolve_task_replay_action = resolve_task_replay_action


def replay_task_start(env: Any, window: dict[str, Any], task: Any | None = None) -> int:
    actions = window.get("pre_actions") or []
    applied = 0
    for action in actions:
        env.advance_to_decision()
        if env.is_terminal():
            raise ReplayError(
                f"env reached a terminal state after {applied} of {len(actions)} "
                f"replayed local-task actions; cannot apply {action.get('description')!r}"
            )
        env.step(resolve_task_replay_action(env, action))
        applied += 1
    env.advance_to_decision()
    if task is not None:
        validate_start = getattr(task, "validate_start", None)
        if validate_start is not None:
            validate_start(env.summary(), window)
    # Source manifests intentionally remain regeneration-compatible and have no
    # public signature. Replay-validated manifests do, and every consumer that
    # enters through this shared helper (eval, teacher collection, GRPO) fails
    # closed if the complete public start choice has drifted.
    validate_stored_start_state_signature(env, window)
    return applied


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


def inject_local_task_meta(
    meta_path: Path,
    jsonl_path: Path,
    task: Any,
    window: dict[str, Any],
) -> bool:
    """Inject task label/reward/metrics into an episode's meta sidecar.

    Reads the episode trace from disk rather than an in-memory RolloutResult so
    a resumed eval can also repair metas written by an earlier crashed process
    (those episodes are skipped on resume and never appear in the new run's
    results). Idempotent. Returns True if the meta was written.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return False
    meta = base.load_json(meta_path)
    decisions = base.load_jsonl(jsonl_path) if Path(jsonl_path).exists() else []
    terminal_state = (decisions[-1].get("after_state") or {}) if decisions else {}
    completion = task.completion_reason(terminal_state)
    effective_stopped_reason = completion or str(meta.get("stopped_reason"))
    metrics = task.metrics_from_episode(
        decisions,
        terminal_state,
        effective_stopped_reason,
        window,
    )
    extra = dict(meta.get("extra") or {})
    extra["local_task"] = {
        "task_id": task.task_id,
        "window_id": window["window_id"],
        "source_stem": window["source_stem"],
        "split": window.get("split"),
        "label": metrics["label"],
        "reward": metrics["reward"],
        "metrics": metrics,
    }
    meta["extra"] = extra
    if completion is not None and meta.get("stopped_reason") == "terminal":
        meta["stopped_reason"] = completion
    base.write_json(meta_path, meta)
    return True


def windows_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [window for window in manifest["windows"] if window.get("split") == split]


def window_rollout_indices(window: dict[str, Any], rollouts_per_window: int) -> list[int]:
    """Collision-free rollout indices for K samples of one task window.

    ``ordinal * K + k`` keeps distinct windows of the same world seed apart and
    reduces to the historical ``rollout_index = ordinal`` when K == 1, so K=1
    output stems stay byte-compatible with pre-K eval dirs.
    """
    if rollouts_per_window < 1:
        raise ValueError("rollouts_per_window must be >= 1")
    ordinal = int(window.get("ordinal", 0))
    return [ordinal * rollouts_per_window + k for k in range(rollouts_per_window)]


def one_window_per_seed(windows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for window in windows:
        seed = int(window["world_seed"])
        if seed in by_seed:
            raise ValueError(f"multiple local-task windows for world_seed={seed}")
        by_seed[seed] = window
    return by_seed
