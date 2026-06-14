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
        model_id: str = "mlx-community/Qwen3-4B-4bit",
        framing: str = NEUTRAL_FRAME,
        max_tokens: int = 256,
        temperature: float = 0.2,
        max_retries: int = 1,
        enable_thinking: bool = False,
    ) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install with `.venv/bin/python -m pip install -e '.[llm]'`."
            ) from exc

        try:
            from mlx_lm.sample_utils import make_sampler
        except (ImportError, ModuleNotFoundError):
            make_sampler = None

        self.model_id = model_id
        self.framing = framing
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.model, self.tokenizer = load(model_id)
        self._generate = generate
        self._sampler = make_sampler(temp=temperature) if make_sampler is not None else None

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        base_prompt = render_action_prompt(state_text, legal_actions, self.framing)
        last_decision: AgentDecision | None = None

        for attempt in range(self.max_retries + 1):
            prompt = base_prompt
            if attempt > 0:
                prompt += (
                    "\n\nYour previous response was invalid. Return only one JSON object "
                    "with a legal integer action_index from the listed actions."
                )

            response = self._generate_text(prompt)
            decision = parse_json_action(response, legal_actions)
            decision.retries = attempt
            last_decision = decision
            if decision.valid:
                return decision

        assert last_decision is not None
        return last_decision

    def _generate_text(self, prompt: str) -> str:
        chat_prompt = self._apply_chat_template(prompt)
        kwargs = {
            "prompt": chat_prompt,
            "max_tokens": self.max_tokens,
        }
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler

        try:
            return self._generate(self.model, self.tokenizer, **kwargs)
        except TypeError:
            # Older mlx-lm releases accepted temp directly; newer releases use sampler.
            if self._sampler is not None:
                raise
            kwargs["temp"] = self.temperature
            return self._generate(self.model, self.tokenizer, **kwargs)

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


def parse_json_action(response: str, legal_actions: list[LegalAction]) -> AgentDecision:
    thinking = _extract_thinking(response)
    parsed = _extract_json(response)
    if parsed is None:
        return AgentDecision(
            action_index=0,
            raw_response=response,
            thinking=thinking,
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
            thinking=thinking,
            valid=False,
            metadata={"error": "invalid action_index", "parsed": parsed},
        )

    return AgentDecision(
        action_index=action_index,
        raw_response=response,
        reasoning=reasoning,
        thinking=thinking,
        valid=True,
        metadata={"parsed": parsed, "legal_action": asdict(legal_actions[action_index])},
    )


def _extract_thinking(text: str) -> str:
    """Return the chain-of-thought inside a <think>...</think> block.

    Captures the content of the first think block. If the block is opened but
    never closed (a truncated thinking-mode generation), returns everything after
    the opening tag so the partial reasoning is not lost. Returns "" when there is
    no think block (e.g. no-thinking mode)."""
    lower = text.lower()
    start = lower.find("<think>")
    if start == -1:
        return ""
    inner_start = start + len("<think>")
    end = lower.find("</think>", inner_start)
    if end == -1:
        return text[inner_start:].strip()
    return text[inner_start:end].strip()


def _extract_json(text: str) -> dict | None:
    for candidate_text in (text, _strip_thinking(text)):
        parsed = _parse_json_dict(candidate_text)
        if parsed is not None:
            return parsed

        for candidate in reversed(_balanced_json_candidates(candidate_text)):
            parsed = _parse_json_dict(candidate)
            if parsed is not None:
                return parsed

    return None


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _parse_json_dict(text: str) -> dict | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1])
                start = None

    return candidates
