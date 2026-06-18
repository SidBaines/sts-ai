"""Cross-rollout batched orchestrator.

Decisions *within* a rollout are serial (the next state depends on the action),
but rollouts are independent — so the throughput lever is to run K rollouts
concurrently and batch their per-decision generations into one call
(`agent.choose_actions_batch`, backed by `mlx_lm.batch_generate`). This advances
K envs in lockstep: gather each pending decision, generate the batch, dispatch
the K chosen actions, advance, refill finished slots from the seed queue.

Records and the per-rollout meta are built with the *same* helpers as the serial
path (`build_decision_record` / `build_rollout_meta`), so a parallel run is
trace-identical in shape to a serial one. NOTE: batched generation left-pads and
runs a batched matmul, so token outputs can differ bit-for-bit from the single-
prompt path — fine pre-freeze; pin batch_size for any frozen-seed data run.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import (
    append_jsonl,
    build_decision_record,
    build_rollout_meta,
    clamp_action_index,
    error_payload,
    prepare_decision,
    write_rollout_meta,
)
from sts_ai.schemas import AgentDecision, DecisionRecord, RolloutResult
from sts_ai.seeding import derive_batch_seed, derive_policy_seed


class _Slot:
    """One in-flight rollout."""

    def __init__(
        self,
        world_seed: int,
        rollout_index: int,
        env: LightspeedHybridEnv,
        output_path: Optional[Path],
    ):
        self.world_seed = world_seed
        self.rollout_index = rollout_index
        self.policy_seed = derive_policy_seed(world_seed, rollout_index)
        self.env = env
        self.output_path = output_path
        self.decisions: list[DecisionRecord] = []
        self.decision_index = 0
        self.view: Optional[dict[str, Any]] = None  # pending decision, set by advance()
        self.stopped_reason = "terminal"
        self.error: Optional[dict[str, Any]] = None
        self.done = False
        self.attempt = 0
        # Streaming-only hint sub-state (NORMAL/HINTED/LAUNDER stages); unused by the parallel/lockstep path.
        self.stage: str = "NORMAL"
        self.normal_decision: Optional[AgentDecision] = None
        self.normal_action_index: Optional[int] = None
        self.normal_affordances: Optional[dict[str, Any]] = None
        self.hint_text: Optional[str] = None
        self.mistake_kind: Optional[str] = None
        self.hinted_decision: Optional[AgentDecision] = None
        self.hinted_action_index: Optional[int] = None


def finalize_slot(
    slot: _Slot,
    *,
    results: dict[tuple[int, int], RolloutResult],
    agent: Any,
    run_meta: Optional[dict[str, Any]],
    stopped_reason: str,
    error: Optional[dict[str, Any]] = None,
) -> None:
    slot.stopped_reason = stopped_reason
    slot.error = error
    slot.done = True
    result = RolloutResult(
        world_seed=slot.world_seed,
        decisions=slot.decisions,
        terminal_state=slot.env.summary(),
        stopped_reason=stopped_reason,
        error=error,
        policy_seed=slot.policy_seed,
        rollout_index=slot.rollout_index,
    )
    if slot.output_path is not None:
        write_rollout_meta(slot.output_path, build_rollout_meta(result, slot.env, agent, run_meta))
    results[(slot.world_seed, slot.rollout_index)] = result


def advance_slot(
    slot: _Slot,
    *,
    max_decisions: int,
    results: dict[tuple[int, int], RolloutResult],
    agent: Any,
    run_meta: Optional[dict[str, Any]],
) -> None:
    """Prepare the slot's next decision, or finalize it if it's finished."""
    if slot.decision_index >= max_decisions:
        finalize_slot(
            slot,
            results=results,
            agent=agent,
            run_meta=run_meta,
            stopped_reason="max_decisions",
        )
        return
    try:
        status, view = prepare_decision(slot.env)
    except Exception as exc:  # noqa: BLE001 - preserve simulator failures in metadata
        finalize_slot(
            slot,
            results=results,
            agent=agent,
            run_meta=run_meta,
            stopped_reason="simulator_error",
            error=error_payload(exc, "advance_to_decision", slot.decision_index),
        )
        return
    if status != "ok":
        finalize_slot(
            slot,
            results=results,
            agent=agent,
            run_meta=run_meta,
            stopped_reason=status,
        )
        return
    slot.view = view


def run_parallel_rollouts(
    specs: list[tuple[int, int]],
    make_env: Callable[[int], LightspeedHybridEnv],
    agent: Any,
    output_for: Callable[[int, int], Optional[Path]] = lambda ws, ri: None,
    batch_size: int = 8,
    max_decisions: int = 200,
    max_retries: int | None = None,
    run_meta: Optional[dict[str, Any]] = None,
) -> list[RolloutResult]:
    """Run `(world_seed, rollout_index)` specs as concurrent rollouts, K
    (`batch_size`) at a time, batching the agent's per-decision generations.
    Returns RolloutResults in input spec order."""
    if max_retries is None:
        max_retries = getattr(agent, "max_retries", 1)

    queue = deque(specs)
    active: list[_Slot] = []
    results: dict[tuple[int, int], RolloutResult] = {}

    def open_slot(spec: tuple[int, int]) -> _Slot:
        world_seed, rollout_index = spec
        slot = _Slot(
            world_seed,
            rollout_index,
            make_env(world_seed),
            output_for(world_seed, rollout_index),
        )
        advance_slot(
            slot,
            max_decisions=max_decisions,
            results=results,
            agent=agent,
            run_meta=run_meta,
        )  # may finalize immediately (e.g. terminal with no decisions)
        return slot

    def refill() -> None:
        while queue and len(active) < batch_size:
            slot = open_slot(queue.popleft())
            if not slot.done:
                active.append(slot)

    refill()
    while active:
        active_batch = list(active)
        members = [
            (slot.world_seed, slot.rollout_index, slot.decision_index)
            for slot in active_batch
        ]
        agent.reseed(derive_batch_seed(members))
        batch = [(slot.view["state_text"], slot.view["legal_actions"]) for slot in active_batch]
        retry_flags = [slot.attempt > 0 for slot in active_batch]
        try:
            decisions = agent.choose_actions_batch(batch, retry_flags=retry_flags)
        except TypeError:
            decisions = agent.choose_actions_batch(batch)

        for slot, agent_decision in zip(active_batch, decisions):
            view = slot.view
            agent_decision.retries = slot.attempt
            action_index = clamp_action_index(agent_decision, len(view["legal_actions"]))
            if not agent_decision.valid and slot.attempt < max_retries:
                slot.attempt += 1
                continue
            if not agent_decision.valid:
                agent_decision.retries = slot.attempt
                record = build_decision_record(
                    world_seed=slot.world_seed,
                    decision_index=slot.decision_index,
                    state=view["state"],
                    state_text=view["state_text"],
                    legal_action_dicts=view["legal_action_dicts"],
                    selected_action_dict={},
                    agent_decision=agent_decision,
                    after_state=slot.env.summary(),
                    phase=view["phase"],
                    policy_seed=slot.policy_seed,
                    rollout_index=slot.rollout_index,
                    action_executed=False,
                )
                slot.decisions.append(record)
                if slot.output_path is not None:
                    append_jsonl(slot.output_path, asdict(record))
                finalize_slot(
                    slot,
                    results=results,
                    agent=agent,
                    run_meta=run_meta,
                    stopped_reason="agent_invalid",
                )
                continue
            try:
                selected = slot.env.step(action_index)
            except Exception as exc:  # noqa: BLE001
                finalize_slot(
                    slot,
                    results=results,
                    agent=agent,
                    run_meta=run_meta,
                    stopped_reason="simulator_error",
                    error=error_payload(exc, "step", slot.decision_index),
                )
                continue
            record = build_decision_record(
                world_seed=slot.world_seed,
                decision_index=slot.decision_index,
                state=view["state"],
                state_text=view["state_text"],
                legal_action_dicts=view["legal_action_dicts"],
                selected_action_dict=slot.env.action_dict(selected),
                agent_decision=agent_decision,
                after_state=slot.env.summary(),
                phase=view["phase"],
                policy_seed=slot.policy_seed,
                rollout_index=slot.rollout_index,
            )
            slot.decisions.append(record)
            if slot.output_path is not None:
                append_jsonl(slot.output_path, asdict(record))
            slot.decision_index += 1
            advance_slot(
                slot,
                max_decisions=max_decisions,
                results=results,
                agent=agent,
                run_meta=run_meta,
            )  # prepare next decision or finalize
            if not slot.done:
                slot.attempt = 0

        active[:] = [slot for slot in active if not slot.done]
        refill()

    return [results[spec] for spec in specs if spec in results]
