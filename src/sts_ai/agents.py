from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from typing import Protocol

from sts_ai.prompting import NEUTRAL_FRAME, render_action_prompt
from sts_ai.schemas import AgentDecision, LegalAction


class ActionAgent(Protocol):
    name: str

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        ...


class FirstLegalAgent:
    name = "first"

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        return AgentDecision(action_index=0, raw_response="first legal action")


class RandomLegalAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        return AgentDecision(
            action_index=self.rng.randrange(len(legal_actions)),
            raw_response="random legal action",
        )


class SimpleHeuristicAgent:
    name = "heuristic"

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        descriptions = [a.description.lower() for a in legal_actions]

        for preferred in ("take gold", "take relic", "take potion"):
            for action, description in zip(legal_actions, descriptions):
                if preferred in description:
                    return AgentDecision(action_index=action.index, raw_response=f"matched {preferred}")

        hp_match = re.search(r"HP: (\d+)/(\d+)", state_text)
        hp_ratio = 1.0
        if hp_match:
            cur_hp, max_hp = int(hp_match.group(1)), int(hp_match.group(2))
            hp_ratio = cur_hp / max(max_hp, 1)

        if "rest" in " ".join(descriptions) or "smith" in " ".join(descriptions):
            target = "smith" if hp_ratio >= 0.55 else "rest"
            for action, description in zip(legal_actions, descriptions):
                if target in description:
                    return AgentDecision(action_index=action.index, raw_response=f"campfire {target}")

        for action, description in zip(legal_actions, descriptions):
            if "take card" in description:
                return AgentDecision(action_index=action.index, raw_response="first card reward")

        return AgentDecision(action_index=0, raw_response="heuristic fallback")


class MlxQwenJsonAgent:
    name = "mlx"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        framing: str = NEUTRAL_FRAME,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install with `.venv/bin/python -m pip install -e '.[llm]'`."
            ) from exc

        self.model_id = model_id
        self.framing = framing
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model, self.tokenizer = load(model_id)
        self._generate = generate

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        prompt = render_action_prompt(state_text, legal_actions, self.framing)
        response = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )
        return parse_json_action(response, legal_actions)


def parse_json_action(response: str, legal_actions: list[LegalAction]) -> AgentDecision:
    parsed = _extract_json(response)
    if parsed is None:
        return AgentDecision(
            action_index=0,
            raw_response=response,
            valid=False,
            metadata={"error": "no json object"},
        )

    action_index = parsed.get("action_index")
    reasoning = str(parsed.get("reasoning", ""))
    if not isinstance(action_index, int) or action_index < 0 or action_index >= len(legal_actions):
        return AgentDecision(
            action_index=0,
            raw_response=response,
            reasoning=reasoning,
            valid=False,
            metadata={"error": "invalid action_index", "parsed": parsed},
        )

    return AgentDecision(
        action_index=action_index,
        raw_response=response,
        reasoning=reasoning,
        valid=True,
        metadata={"parsed": parsed, "legal_action": asdict(legal_actions[action_index])},
    )


def _extract_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
