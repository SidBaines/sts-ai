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
    reasoning: str = ""
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


@dataclass
class RolloutResult:
    seed: int
    decisions: list[DecisionRecord]
    terminal_state: dict[str, Any]
    stopped_reason: str
