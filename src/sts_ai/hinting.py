"""Pure hinting helpers for hinted rollout generation.

This module only decides whether a tactical-truth hint is warranted and how to
record the final decision after laundering. It does not call a model, simulator,
or environment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from .affordances import action_contributes_block, action_is_single_target_lethal
from .schemas import AgentDecision

__all__ = [
    "BLOCK_HINT",
    "LETHAL_HINT",
    "HintConfig",
    "action_only_raw_response",
    "build_hinted_prompt_suffix",
    "build_launder_state_text",
    "detect_mistake",
    "finalize_hinted_decision",
    "launder_guardrail_ok",
    "mistake_kind_for",
    "no_change_provenance",
]


@dataclass(frozen=True)
class HintConfig:
    enabled: bool = False
    full_block_hp_fraction: float = 1.0
    full_block_min_incoming: int = 1
    on_launder_fail: str = "action_only"


LETHAL_HINT = "You can defeat an enemy this turn."
BLOCK_HINT = "You can fully block the incoming damage this turn."

_MISTAKE_KIND_BY_HINT = {
    LETHAL_HINT: "lethal",
    BLOCK_HINT: "block",
}


def mistake_kind_for(hint: str) -> str:
    try:
        return _MISTAKE_KIND_BY_HINT[hint]
    except KeyError as exc:
        raise ValueError(f"unknown hint: {hint!r}") from exc


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def detect_mistake(
    affordances: dict[str, Any],
    chosen_action: dict[str, Any],
    combat: dict[str, Any],
    state_text: str,
    cfg: HintConfig,
) -> str | None:
    """Return a single factual hint if the chosen action misses a known affordance."""
    if not affordances:
        return None

    lethal_available = affordances.get("single_target_lethal_available")
    lethal_taken = action_is_single_target_lethal(chosen_action, combat)
    if lethal_available and not lethal_taken:
        return LETHAL_HINT

    incoming = _as_int(affordances.get("incoming_damage_total", 0))
    player_hp = _as_int(combat.get("player_cur_hp", 0))
    block_threshold = max(
        cfg.full_block_min_incoming,
        round(cfg.full_block_hp_fraction * player_hp),
    )
    if (
        affordances.get("full_block_possible")
        and incoming >= block_threshold
        and not action_contributes_block(chosen_action, state_text)
    ):
        return BLOCK_HINT

    return None


def build_hinted_prompt_suffix(hint: str) -> str:
    """Suffix appended to state text for the hinted re-decision."""
    return f"\n\nFactual hint: {hint}\n"


def build_launder_state_text(
    original_state_text: str,
    chosen_action: dict[str, Any],
) -> str:
    """Un-hinted state text plus an instruction to justify and emit one action."""
    action_index = chosen_action.get("index")
    description = str(chosen_action.get("description", ""))
    return (
        original_state_text.rstrip()
        + "\n\nTarget action: "
        + f"{action_index}: {description}\n"
        + "Explain your reasoning for that action, then return exactly one JSON object "
        + f'selecting {{"reasoning": "...", "action_index": {action_index}}}.\n'
    )


def launder_guardrail_ok(
    laundered_decision: AgentDecision,
    target_action_index: int,
) -> bool:
    return bool(
        laundered_decision.valid
        and laundered_decision.action_index == target_action_index
    )


def action_only_raw_response(target_action_index: int) -> str:
    return json.dumps(
        {"reasoning": "", "action_index": int(target_action_index)},
        separators=(",", ":"),
    )


def _hint_provenance(
    *,
    normal_decision: AgentDecision,
    hint_text: str,
    mistake_kind: str,
    final_action_index: int,
    hinted_raw_response: str | None,
    launder_outcome: str,
    triggered: bool,
) -> dict[str, Any]:
    return {
        "triggered": triggered,
        "hint_text": hint_text,
        "mistake_kind": mistake_kind,
        "original_action_index": normal_decision.action_index,
        "final_action_index": final_action_index,
        "original_raw_response": normal_decision.raw_response,
        "original_reasoning": normal_decision.reasoning,
        "original_thinking": normal_decision.thinking,
        "hinted_raw_response": hinted_raw_response,
        "launder_outcome": launder_outcome,
    }


def _metadata_with_hint(
    metadata: dict[str, Any],
    hint_provenance: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(metadata)
    updated["hint"] = hint_provenance
    return updated


def no_change_provenance(
    normal_decision: AgentDecision,
    hint_text: str,
    mistake_kind: str,
    reason: str,
) -> AgentDecision:
    """Return the original decision with consistent hint provenance attached."""
    provenance = _hint_provenance(
        normal_decision=normal_decision,
        hint_text=hint_text,
        mistake_kind=mistake_kind,
        final_action_index=normal_decision.action_index,
        hinted_raw_response=None,
        launder_outcome=reason,
        triggered=False,
    )
    return replace(
        normal_decision,
        metadata=_metadata_with_hint(normal_decision.metadata, provenance),
    )


def finalize_hinted_decision(
    *,
    normal_decision: AgentDecision,
    hinted_decision: AgentDecision,
    laundered_decision: AgentDecision | None,
    hint_text: str,
    mistake_kind: str,
    cfg: HintConfig,
) -> AgentDecision:
    """Produce the recorded decision after a hinted correction attempt."""
    if laundered_decision is not None and launder_guardrail_ok(
        laundered_decision,
        hinted_decision.action_index,
    ):
        provenance = _hint_provenance(
            normal_decision=normal_decision,
            hint_text=hint_text,
            mistake_kind=mistake_kind,
            final_action_index=hinted_decision.action_index,
            hinted_raw_response=hinted_decision.raw_response,
            launder_outcome="laundered",
            triggered=True,
        )
        return replace(
            hinted_decision,
            raw_response=laundered_decision.raw_response,
            reasoning=laundered_decision.reasoning,
            thinking=laundered_decision.thinking,
            valid=True,
            retries=0,
            metadata=_metadata_with_hint(hinted_decision.metadata, provenance),
        )

    if cfg.on_launder_fail == "action_only":
        provenance = _hint_provenance(
            normal_decision=normal_decision,
            hint_text=hint_text,
            mistake_kind=mistake_kind,
            final_action_index=hinted_decision.action_index,
            hinted_raw_response=hinted_decision.raw_response,
            launder_outcome="fallback_action_only",
            triggered=True,
        )
        # This synthetic fallback carries no thought-channel content even if the
        # copied metadata says gemma_thought. Future SFT normalization must not
        # infer marker-bearing content from that tag for fallback records.
        return replace(
            hinted_decision,
            raw_response=action_only_raw_response(hinted_decision.action_index),
            reasoning="",
            thinking="",
            valid=True,
            retries=0,
            metadata=_metadata_with_hint(hinted_decision.metadata, provenance),
        )

    provenance = _hint_provenance(
        normal_decision=normal_decision,
        hint_text=hint_text,
        mistake_kind=mistake_kind,
        final_action_index=normal_decision.action_index,
        hinted_raw_response=hinted_decision.raw_response,
        launder_outcome="drop",
        triggered=False,
    )
    return replace(
        normal_decision,
        metadata=_metadata_with_hint(normal_decision.metadata, provenance),
    )
