from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sts_ai.agents import ActionAgent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.schemas import DecisionRecord, RolloutResult


def run_rollout(
    env: LightspeedHybridEnv,
    agent: ActionAgent,
    max_decisions: int = 200,
    output_path: str | Path | None = None,
) -> RolloutResult:
    decisions: list[DecisionRecord] = []
    stopped_reason = "terminal"

    for decision_index in range(max_decisions):
        env.advance_to_decision()
        if env.is_terminal():
            stopped_reason = "terminal"
            break

        legal_actions = env.legal_actions()
        if not legal_actions:
            stopped_reason = "no_legal_actions"
            break

        state = env.summary()
        state_text = env.describe_state()
        agent_decision = agent.choose_action(state_text, legal_actions)
        action_index = agent_decision.action_index

        if action_index < 0 or action_index >= len(legal_actions):
            agent_decision.valid = False
            agent_decision.metadata["fallback_reason"] = "agent returned out-of-range action"
            action_index = 0

        selected = env.step(action_index)
        record = DecisionRecord(
            seed=env.seed,
            decision_index=decision_index,
            state=state,
            state_text=state_text,
            legal_actions=[env.action_dict(action) for action in legal_actions],
            selected_action=env.action_dict(selected),
            agent=asdict(agent_decision),
            after_state=env.summary(),
        )
        decisions.append(record)

        if output_path is not None:
            append_jsonl(output_path, asdict(record))
    else:
        stopped_reason = "max_decisions"

    return RolloutResult(
        seed=env.seed,
        decisions=decisions,
        terminal_state=env.summary(),
        stopped_reason=stopped_reason,
    )


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
