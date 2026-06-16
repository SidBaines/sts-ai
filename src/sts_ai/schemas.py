from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# On-disk format version. Bumped when the kept-data contract changes. v1 was the
# first kept-data version; v2 is a breaking rename from `seed` to `world_seed`
# plus explicit `policy_seed` / `rollout_index` identity fields.
SCHEMA_VERSION = 3


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
    # Per-decision generation telemetry (additive; defaults keep old traces valid).
    # `latency_s` is wall-time for this decision's generation (batch-amortized in
    # parallel mode — see parallel_rollout); token counts are the model-independent
    # signal for "how much did it reason" (thinking_tokens) and total cost.
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    world_seed: int
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
    # Eval-support: structured, sim-grounded summary of what the agent *could* have
    # done this decision (combat only; {} out of combat). Computed in pure Python by
    # sts_ai.affordances; lets evals ask "could it have full-blocked / taken lethal"
    # without re-parsing state_text. Additive; does not change what the model sees.
    affordances: dict[str, Any] = field(default_factory=dict)
    policy_seed: int | None = None
    rollout_index: int = 0
    # False only for terminal agent-invalid records: the model response is kept
    # for audit, but no simulator action was executed from it.
    action_executed: bool = True


@dataclass
class RolloutResult:
    world_seed: int
    decisions: list[DecisionRecord]
    terminal_state: dict[str, Any]
    stopped_reason: str
    error: dict[str, Any] | None = None
    policy_seed: int | None = None
    rollout_index: int = 0


@dataclass
class RolloutMeta:
    """One-per-rollout summary + provenance, written as a `<output>.meta.json`
    sidecar (not in the decision JSONL, so existing loaders are untouched).

    Captures what the decision stream cannot: the run's identity (model, config,
    and especially the **framing** — the study's independent variable, recorded
    nowhere else), the final outcome, and rollout-level aggregates for eval/RL.
    v2 is a breaking kept-data version (SCHEMA_VERSION)."""
    world_seed: int
    schema_version: int = SCHEMA_VERSION
    # Provenance / run identity
    agent: str = ""
    model_id: str | None = None
    framing: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: bool | None = None
    max_retries: int | None = None
    ascension: int = 0
    combat_control: str = ""
    battle_simulations: int | None = None
    git_sha: str | None = None
    timestamp: str = ""
    # Outcome / aggregates
    outcome: str = ""
    stopped_reason: str = ""
    error: dict[str, Any] | None = None
    undefined_behavior_evoked: bool = False
    final_act: int = 0
    final_floor: int = 0
    final_hp: int = 0
    max_hp: int = 0
    n_decisions: int = 0
    n_combat: int = 0
    n_out_of_combat: int = 0
    n_invalid: int = 0
    # Per-decision player HP after each action (combat: player_cur_hp; else cur_hp).
    hp_trajectory: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    policy_seed: int | None = None
    rollout_index: int = 0
