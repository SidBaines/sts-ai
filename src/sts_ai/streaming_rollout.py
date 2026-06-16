"""Continuous-batching rollout orchestrator for vLLM.

This path drives `LLMEngine.step()` synchronously, keeping up to `concurrency`
rollouts in flight so the GPU stays saturated. Unlike the lockstep
`parallel_rollout` path, it does not block each round on the slowest generation:
finished requests are dispatched back to their rollout slots immediately, then
resubmitted for the next decision.

Each request is seeded independently from its own
`(world_seed, rollout_index, decision_index)`, so sampling is independent of
batch composition and completion order. Records and per-rollout metadata reuse
the same helpers as the serial and parallel paths, making trace shape identical
across orchestrators.

This module is vLLM-only. MLX has no continuous-batching engine and continues to
use `parallel_rollout`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.parallel_rollout import _Slot, advance_slot, finalize_slot
from sts_ai.rollout import (
    append_jsonl,
    build_decision_record,
    clamp_action_index,
    error_payload,
)
from sts_ai.schemas import RolloutResult
from sts_ai.seeding import derive_batch_seed


def run_streaming_rollouts(
    specs: list[tuple[int, int]],
    make_env: Callable[[int], LightspeedHybridEnv],
    agent: Any,
    output_for: Callable[[int, int], Optional[Path]] = lambda ws, ri: None,
    concurrency: int = 48,
    max_decisions: int = 200,
    run_meta: Optional[dict[str, Any]] = None,
) -> list[RolloutResult]:
    """Run rollout specs through a continuous-batching generation backend."""
    queue = deque(specs)
    results: dict[tuple[int, int], RolloutResult] = {}
    # in_flight owns termination/slot mapping; stream_has_unfinished() is for async backends.
    in_flight: dict[str, _Slot] = {}

    def request_id(slot: _Slot) -> str:
        return f"{slot.world_seed}:{slot.rollout_index}:{slot.decision_index}"

    def submit(slot: _Slot) -> None:
        rid = request_id(slot)
        seed = derive_batch_seed(
            [(slot.world_seed, slot.rollout_index, slot.decision_index)]
        )
        agent.stream_submit(
            rid,
            slot.view["state_text"],
            slot.view["legal_actions"],
            seed,
        )
        in_flight[rid] = slot

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
        )
        return slot

    def fill() -> None:
        while queue and len(in_flight) < concurrency:
            slot = open_slot(queue.popleft())
            if not slot.done:
                submit(slot)

    fill()
    # Each poll advances one engine step (agent.stream_poll() wraps LLMEngine.step()),
    # so [] is expected while sequences decode; parallel_rollout blocks per round.
    while in_flight:
        for rid, output in agent.stream_poll():
            slot = in_flight.pop(rid)
            view = slot.view
            decision = agent.build_decision_from_text(
                output["text"],
                output["prompt_tokens"],
                output["completion_tokens"],
                view["legal_actions"],
            )
            action_index = clamp_action_index(decision, len(view["legal_actions"]))
            try:
                selected = slot.env.step(action_index)
            except Exception as exc:  # noqa: BLE001 - preserve simulator failures in metadata
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
                agent_decision=decision,
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
            )
            if not slot.done:
                submit(slot)
        fill()

    return [results[spec] for spec in specs if spec in results]
