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
from sts_ai.schemas import DecisionRecord, RolloutResult


class _Slot:
    """One in-flight rollout."""

    def __init__(self, seed: int, env: LightspeedHybridEnv, output_path: Optional[Path]):
        self.seed = seed
        self.env = env
        self.output_path = output_path
        self.decisions: list[DecisionRecord] = []
        self.decision_index = 0
        self.view: Optional[dict[str, Any]] = None  # pending decision, set by advance()
        self.stopped_reason = "terminal"
        self.error: Optional[dict[str, Any]] = None
        self.done = False


def run_parallel_rollouts(
    seeds: list[int],
    make_env: Callable[[int], LightspeedHybridEnv],
    agent: Any,
    output_for: Callable[[int], Optional[Path]] = lambda s: None,
    batch_size: int = 8,
    max_decisions: int = 200,
    run_meta: Optional[dict[str, Any]] = None,
) -> list[RolloutResult]:
    """Run `seeds` as concurrent rollouts, K (`batch_size`) at a time, batching
    the agent's per-decision generations. Returns RolloutResults in seed order."""
    queue = deque(seeds)
    active: list[_Slot] = []
    results: dict[int, RolloutResult] = {}

    def finalize(slot: _Slot, stopped_reason: str, error: Optional[dict[str, Any]] = None) -> None:
        slot.stopped_reason = stopped_reason
        slot.error = error
        slot.done = True
        result = RolloutResult(
            seed=slot.seed,
            decisions=slot.decisions,
            terminal_state=slot.env.summary(),
            stopped_reason=stopped_reason,
            error=error,
        )
        if slot.output_path is not None:
            write_rollout_meta(slot.output_path, build_rollout_meta(result, slot.env, agent, run_meta))
        results[slot.seed] = result

    def advance(slot: _Slot) -> None:
        """Prepare the slot's next decision, or finalize it if it's finished."""
        if slot.decision_index >= max_decisions:
            finalize(slot, "max_decisions")
            return
        try:
            status, view = prepare_decision(slot.env)
        except Exception as exc:  # noqa: BLE001 - preserve simulator failures in metadata
            finalize(slot, "simulator_error", error_payload(exc, "advance_to_decision", slot.decision_index))
            return
        if status != "ok":
            finalize(slot, status)
            return
        slot.view = view

    def open_slot(seed: int) -> _Slot:
        slot = _Slot(seed, make_env(seed), output_for(seed))
        advance(slot)  # may finalize immediately (e.g. terminal with no decisions)
        return slot

    def refill() -> None:
        while queue and len(active) < batch_size:
            slot = open_slot(queue.popleft())
            if not slot.done:
                active.append(slot)

    refill()
    while active:
        batch = [(slot.view["state_text"], slot.view["legal_actions"]) for slot in active]
        decisions = agent.choose_actions_batch(batch)

        for slot, agent_decision in zip(active, decisions):
            view = slot.view
            action_index = clamp_action_index(agent_decision, len(view["legal_actions"]))
            try:
                selected = slot.env.step(action_index)
            except Exception as exc:  # noqa: BLE001
                finalize(slot, "simulator_error", error_payload(exc, "step", slot.decision_index))
                continue
            record = build_decision_record(
                seed=slot.seed,
                decision_index=slot.decision_index,
                state=view["state"],
                state_text=view["state_text"],
                legal_action_dicts=view["legal_action_dicts"],
                selected_action_dict=slot.env.action_dict(selected),
                agent_decision=agent_decision,
                after_state=slot.env.summary(),
                phase=view["phase"],
            )
            slot.decisions.append(record)
            if slot.output_path is not None:
                append_jsonl(slot.output_path, asdict(record))
            slot.decision_index += 1
            advance(slot)  # prepare next decision or finalize

        active[:] = [slot for slot in active if not slot.done]
        refill()

    return [results[seed] for seed in seeds if seed in results]
