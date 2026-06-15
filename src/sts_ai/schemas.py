from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegalAction:
    index: int
    bits: int
    description: str


@dataclass
class AgentDecision:
    action_index: int
    raw_response: str = ""
    # `reasoning`: the brief justification from the JSON action object.
    # `thinking`: the model's chain-of-thought from a <think>...</think> block
    #   (populated in thinking mode; empty otherwise). Captured separately so
    #   Stage 5 training data can place reasoning in the forward context and so it
    #   can be audited for framing leakage. Added 2026-06-14 (additive, default "").
    reasoning: str = ""
    thinking: str = ""
    valid: bool = True
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    seed: int
    decision_index: int
    state: dict[str, Any]
    state_text: str
    legal_actions: list[dict[str, Any]]
    selected_action: dict[str, Any]
    agent: dict[str, Any]
    after_state: dict[str, Any]
    # `phase`: "out_of_combat" (Python-controlled Neow/path/reward/shop/event/campfire
    #   decisions) or "combat" (in-combat micro-decisions under full LLM control).
    #   Distinguishes the two decision kinds for training/eval. Additive change with a
    #   default so pre-combat traces still load (added 2026-06-14). Combat-specific
    #   state (turn, enemy HP/intents, player block/energy) rides in `state["combat"]`,
    #   which needs no schema change since `state` is free-form.
    phase: str = "out_of_combat"


@dataclass
class RolloutResult:
    seed: int
    decisions: list[DecisionRecord]
    terminal_state: dict[str, Any]
    stopped_reason: str
    error: dict[str, Any] | None = None
