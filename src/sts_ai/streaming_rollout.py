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

from sts_ai import affordances, hinting
from sts_ai.hinting import HintConfig
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.parallel_rollout import _Slot, advance_slot, finalize_slot
from sts_ai.rollout import (
    append_jsonl,
    build_decision_record,
    clamp_action_index,
    error_payload,
)
from sts_ai.schemas import RolloutResult
from sts_ai.seeding import derive_batch_seed, derive_stage_seed


def run_streaming_rollouts(
    specs: list[tuple[int, int]],
    make_env: Callable[[int], LightspeedHybridEnv],
    agent: Any,
    output_for: Callable[[int, int], Optional[Path]] = lambda ws, ri: None,
    concurrency: int = 48,
    max_decisions: int = 200,
    max_retries: int | None = None,
    run_meta: Optional[dict[str, Any]] = None,
    hint_cfg: HintConfig | None = None,
) -> list[RolloutResult]:
    """Run rollout specs through a continuous-batching generation backend.

    Invalid responses get per-rollout retries as new non-blocking requests;
    after retry exhaustion, the invalid response is recorded and the rollout
    stops with `agent_invalid` before any fallback action is executed.
    """
    if max_retries is None:
        max_retries = getattr(agent, "max_retries", 1)

    queue = deque(specs)
    results: dict[tuple[int, int], RolloutResult] = {}
    # in_flight owns termination/slot mapping; stream_has_unfinished() is for async backends.
    in_flight: dict[str, _Slot] = {}
    hints_enabled = hint_cfg is not None and hint_cfg.enabled

    def request_id(slot: _Slot) -> str:
        base = f"{slot.world_seed}:{slot.rollout_index}:{slot.decision_index}:a{slot.attempt}"
        if not hints_enabled or slot.stage == "NORMAL":
            return base
        if slot.stage == "HINTED":
            return base + ":h"
        if slot.stage == "LAUNDER":
            return base + ":l"
        raise RuntimeError(f"unknown streaming hint stage: {slot.stage}")

    def submit(slot: _Slot) -> None:
        rid = request_id(slot)
        if slot.stage == "NORMAL":
            state_text = slot.view["state_text"]
            seed = derive_batch_seed(
                [(slot.world_seed, slot.rollout_index, slot.decision_index)]
            )
            retry = slot.attempt > 0
        elif slot.stage == "HINTED":
            if slot.hint_text is None:
                raise RuntimeError("HINTED stage missing hint_text")
            state_text = slot.view["state_text"] + hinting.build_hinted_prompt_suffix(
                slot.hint_text
            )
            seed = derive_stage_seed(
                slot.world_seed,
                slot.rollout_index,
                slot.decision_index,
                "HINTED",
            )
            retry = False
        elif slot.stage == "LAUNDER":
            if slot.hinted_action_index is None:
                raise RuntimeError("LAUNDER stage missing hinted_action_index")
            state_text = hinting.build_launder_state_text(
                slot.view["state_text"],
                slot.view["legal_action_dicts"][slot.hinted_action_index],
            )
            seed = derive_stage_seed(
                slot.world_seed,
                slot.rollout_index,
                slot.decision_index,
                "LAUNDER",
            )
            retry = False
        else:
            raise RuntimeError(f"unknown streaming hint stage: {slot.stage}")
        agent.stream_submit(
            rid,
            state_text,
            slot.view["legal_actions"],
            seed,
            retry=retry,
        )
        in_flight[rid] = slot

    def reset_hint_state(slot: _Slot) -> None:
        slot.stage = "NORMAL"
        slot.attempt = 0
        slot.normal_decision = None
        slot.normal_action_index = None
        slot.normal_affordances = None
        slot.hint_text = None
        slot.mistake_kind = None
        slot.hinted_decision = None
        slot.hinted_action_index = None

    def commit(
        slot: _Slot,
        decision: Any,
        action_index: int,
        hint_applied: bool,
        affordances_override: dict[str, Any] | None,
    ) -> None:
        view = slot.view
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
            return
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
            hint_applied=hint_applied,
            affordances_override=affordances_override,
        )
        slot.decisions.append(record)
        if slot.output_path is not None:
            append_jsonl(slot.output_path, asdict(record))
        slot.decision_index += 1
        reset_hint_state(slot)
        advance_slot(
            slot,
            max_decisions=max_decisions,
            results=results,
            agent=agent,
            run_meta=run_meta,
        )
        if not slot.done:
            submit(slot)

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
            # Per-decision wall-time from the backend (submit->finish). Set here, like
            # decision.retries below, so build_decision_from_text's signature is unchanged.
            decision.latency_s = output.get("latency_s", 0.0)

            if slot.stage == "HINTED":
                if (
                    slot.normal_decision is None
                    or slot.normal_action_index is None
                    or slot.normal_affordances is None
                    or slot.hint_text is None
                    or slot.mistake_kind is None
                ):
                    raise RuntimeError("HINTED stage missing normal decision state")
                hinted_action_index = clamp_action_index(
                    decision,
                    len(view["legal_actions"]),
                )
                if not (
                    decision.valid
                    and hinted_action_index != slot.normal_action_index
                ):
                    reason = "hinted_invalid" if not decision.valid else "no_change"
                    final = hinting.no_change_provenance(
                        slot.normal_decision,
                        slot.hint_text,
                        slot.mistake_kind,
                        reason,
                    )
                    commit(
                        slot,
                        final,
                        slot.normal_action_index,
                        hint_applied=False,
                        affordances_override=slot.normal_affordances,
                    )
                    continue

                slot.hinted_decision = decision
                slot.hinted_action_index = hinted_action_index
                slot.stage = "LAUNDER"
                submit(slot)
                continue

            if slot.stage == "LAUNDER":
                if (
                    slot.normal_decision is None
                    or slot.hinted_decision is None
                    or slot.normal_affordances is None
                    or slot.hint_text is None
                    or slot.mistake_kind is None
                ):
                    raise RuntimeError("LAUNDER stage missing hinted decision state")
                final = hinting.finalize_hinted_decision(
                    normal_decision=slot.normal_decision,
                    hinted_decision=slot.hinted_decision,
                    laundered_decision=decision,
                    hint_text=slot.hint_text,
                    mistake_kind=slot.mistake_kind,
                    cfg=hint_cfg,
                )
                action_index = clamp_action_index(final, len(view["legal_actions"]))
                hint_applied = final.metadata.get("hint", {}).get(
                    "launder_outcome"
                ) in ("laundered", "fallback_action_only")
                commit(
                    slot,
                    final,
                    action_index,
                    hint_applied=hint_applied,
                    affordances_override=slot.normal_affordances,
                )
                continue

            if slot.stage != "NORMAL":
                raise RuntimeError(f"unknown streaming hint stage: {slot.stage}")

            if not decision.valid and slot.attempt < max_retries:
                slot.attempt += 1
                submit(slot)
                continue
            decision.retries = slot.attempt
            action_index = clamp_action_index(decision, len(view["legal_actions"]))
            if not decision.valid:
                record = build_decision_record(
                    world_seed=slot.world_seed,
                    decision_index=slot.decision_index,
                    state=view["state"],
                    state_text=view["state_text"],
                    legal_action_dicts=view["legal_action_dicts"],
                    selected_action_dict={},
                    agent_decision=decision,
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

            affordances_override = None
            if hints_enabled and view["phase"] == "combat" and decision.valid:
                aff = affordances.compute(
                    view["state"],
                    view["state_text"],
                    view["legal_action_dicts"],
                    view["phase"],
                )
                hint = hinting.detect_mistake(
                    aff,
                    view["legal_action_dicts"][action_index],
                    view["state"]["combat"],
                    view["state_text"],
                    hint_cfg,
                )
                if hint is not None:
                    slot.normal_decision = decision
                    slot.normal_action_index = action_index
                    slot.normal_affordances = aff
                    slot.hint_text = hint
                    slot.mistake_kind = hinting.mistake_kind_for(hint)
                    slot.stage = "HINTED"
                    submit(slot)
                    continue
                affordances_override = aff

            commit(
                slot,
                decision,
                action_index,
                hint_applied=False,
                affordances_override=affordances_override,
            )
        fill()

    return [results[spec] for spec in specs if spec in results]
