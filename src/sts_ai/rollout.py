from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sts_ai import affordances, glossary
from sts_ai.agents import ActionAgent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.schemas import DecisionRecord, RolloutMeta, RolloutResult
from sts_ai.seeding import derive_policy_seed


def current_git_sha() -> str | None:
    """Best-effort short HEAD sha for run provenance (None if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def build_decision_record(
    *,
    world_seed: int,
    decision_index: int,
    state: dict[str, Any],
    state_text: str,
    legal_action_dicts: list[dict[str, Any]],
    selected_action_dict: dict[str, Any],
    agent_decision: Any,
    after_state: dict[str, Any],
    phase: str,
    policy_seed: int | None,
    rollout_index: int,
    action_executed: bool = True,
) -> DecisionRecord:
    """Build one DecisionRecord (incl. computed affordances). Shared by the serial
    loop here and the parallel orchestrator so both emit identical records."""
    return DecisionRecord(
        world_seed=world_seed,
        decision_index=decision_index,
        state=state,
        state_text=state_text,
        legal_actions=legal_action_dicts,
        selected_action=selected_action_dict,
        agent=asdict(agent_decision),
        after_state=after_state,
        phase=phase,
        affordances=affordances.compute(state, state_text, legal_action_dicts, phase),
        policy_seed=policy_seed,
        rollout_index=rollout_index,
        action_executed=action_executed,
    )


def prepare_decision(env: LightspeedHybridEnv) -> tuple[str, dict[str, Any] | None]:
    """Advance to the next pending decision and assemble the agent-facing view
    (`state_text` already glossary-augmented). Returns (status, view) where status
    is "ok" | "terminal" | "no_legal_actions"; `view` is a dict when ok. May raise
    on simulator failure (the caller records it). Shared by serial + parallel."""
    env.advance_to_decision()
    if env.is_terminal():
        return "terminal", None
    legal_actions = env.legal_actions()
    if not legal_actions:
        return "no_legal_actions", None
    phase = env.phase()
    legal_action_dicts = [env.action_dict(action) for action in legal_actions]
    # Fold the static effect/status reference into what the agent sees + records.
    # On a MAP_SCREEN, env.map_graph() supplies the act DAG so glossary renders a
    # neutral per-choice path summary (None elsewhere -> no-op).
    state_text = glossary.augment(
        env.describe_state(), legal_action_dicts, phase, map_graph=env.map_graph()
    )
    return "ok", {
        "phase": phase,
        "state": env.summary(),
        "state_text": state_text,
        "legal_actions": legal_actions,
        "legal_action_dicts": legal_action_dicts,
    }


def clamp_action_index(agent_decision: Any, n_actions: int) -> int:
    """Mark out-of-range actions invalid; returned index is never executed if invalid."""
    idx = agent_decision.action_index
    if idx < 0 or idx >= n_actions:
        agent_decision.valid = False
        agent_decision.metadata["invalid_reason"] = "agent returned out-of-range action"
        return 0
    return idx


def run_rollout(
    env: LightspeedHybridEnv,
    agent: ActionAgent,
    max_decisions: int = 200,
    output_path: str | Path | None = None,
    run_meta: dict[str, Any] | None = None,
    rollout_index: int = 0,
    policy_seed: int | None = None,
) -> RolloutResult:
    world_seed = env.world_seed
    if policy_seed is None:
        policy_seed = derive_policy_seed(world_seed, rollout_index)
    agent.reseed(policy_seed)

    decisions: list[DecisionRecord] = []
    stopped_reason = "terminal"
    error: dict[str, Any] | None = None

    for decision_index in range(max_decisions):
        try:
            status, view = prepare_decision(env)
        except Exception as exc:  # noqa: BLE001 - preserve simulator failures in rollout metadata.
            stopped_reason = "simulator_error"
            error = error_payload(exc, "advance_to_decision", decision_index)
            break

        if status != "ok":
            stopped_reason = status  # "terminal" or "no_legal_actions"
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
        except Exception as exc:  # noqa: BLE001 - preserve simulator failures in rollout metadata.
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
        meta = build_rollout_meta(result, env, agent, run_meta)
        write_rollout_meta(output_path, meta)
    return result


def build_rollout_meta(
    result: RolloutResult,
    env: LightspeedHybridEnv,
    agent: ActionAgent,
    run_meta: dict[str, Any] | None = None,
) -> RolloutMeta:
    """Summarize a finished rollout + capture run provenance (model/config/framing/
    git). Provenance comes from the agent's `config` (if any) and the caller's
    `run_meta`; outcome/aggregates from the result + terminal state."""
    run_meta = run_meta or {}
    cfg = getattr(agent, "config", {}) or {}
    term = result.terminal_state or {}
    extra = dict(run_meta.get("extra", {}))
    budget_truncated = result.stopped_reason == "max_decisions"
    extra["budget_truncated"] = budget_truncated
    if budget_truncated:
        warnings.warn(
            f"rollout world_seed={result.world_seed} hit the decision budget "
            "(max_decisions) before the game ended; outcome is truncated, not final.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg:
        extra["agent_config"] = dict(cfg)

    hp_trajectory: list[int] = []
    n_combat = n_out_of_combat = n_invalid = 0
    for d in result.decisions:
        after = d.after_state or {}
        combat = after.get("combat")
        if after.get("phase") == "combat" and isinstance(combat, dict):
            hp_trajectory.append(int(combat.get("player_cur_hp", after.get("cur_hp", 0))))
        else:
            hp_trajectory.append(int(after.get("cur_hp", 0)))
        if d.phase == "combat":
            n_combat += 1
        else:
            n_out_of_combat += 1
        if not d.agent.get("valid", True):
            n_invalid += 1

    return RolloutMeta(
        world_seed=result.world_seed,
        agent=getattr(agent, "name", str(run_meta.get("agent", ""))),
        model_id=cfg.get("model_id", run_meta.get("model_id")),
        framing=cfg.get("framing", run_meta.get("framing")),
        temperature=cfg.get("temperature", run_meta.get("temperature")),
        max_tokens=cfg.get("max_tokens", run_meta.get("max_tokens")),
        thinking=cfg.get("thinking", run_meta.get("thinking")),
        max_retries=cfg.get("max_retries", run_meta.get("max_retries")),
        ascension=int(getattr(env, "ascension", 0)),
        combat_control=str(getattr(env, "combat_control", "")),
        battle_simulations=run_meta.get("battle_simulations"),
        git_sha=run_meta.get("git_sha"),
        timestamp=run_meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        outcome=str(term.get("outcome", "")),
        stopped_reason=result.stopped_reason,
        error=result.error,
        undefined_behavior_evoked=bool(term.get("undefined_behavior_evoked", False)),
        final_act=int(term.get("act", 0)),
        final_floor=int(term.get("floor", 0)),
        final_hp=int(term.get("cur_hp", 0)),
        max_hp=int(term.get("max_hp", 0)),
        n_decisions=len(result.decisions),
        n_combat=n_combat,
        n_out_of_combat=n_out_of_combat,
        n_invalid=n_invalid,
        hp_trajectory=hp_trajectory,
        extra=extra,
        policy_seed=result.policy_seed,
        rollout_index=result.rollout_index,
    )


def write_rollout_meta(output_path: str | Path, meta: RolloutMeta) -> Path:
    """Write the per-rollout meta as a `<output>.meta.json` sidecar."""
    meta_path = Path(output_path).with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta_path


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def error_payload(exc: Exception, phase: str, decision_index: int) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "phase": phase,
        "decision_index": decision_index,
    }
