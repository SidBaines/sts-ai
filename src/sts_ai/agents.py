from __future__ import annotations

import json
import random
import re
import time
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
    """Local MLX agent that emits a JSON action.

    IMPORTANT — `max_tokens` must be large (default 4096). Reasoning models need
    room to finish thinking AND still emit the closing JSON: a small cap (e.g.
    256) truncates mid-thought, so `parse_json_action` finds no JSON, the decision
    is marked invalid, and the loop silently falls back to action 0 — i.e. the
    "policy" degenerates to "always the first legal action". This is harmless to
    raise for no-thinking models (generation stops at EOS ~60-90 tokens, well
    under the cap), so a high default costs nothing there and saves the reasoning
    runs. See scripts/CLAUDE.md and the invalid-rate/token telemetry in evals."""

    name = "mlx"

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-4B-4bit",
        framing: str = NEUTRAL_FRAME,
        max_tokens: int = 4096,
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
        try:
            from mlx_lm import batch_generate
        except (ImportError, ModuleNotFoundError):
            batch_generate = None

        self.model, self.tokenizer = load(model_id)
        self._generate = generate
        self._batch_generate = batch_generate
        self._sampler = make_sampler(temp=temperature) if make_sampler is not None else None

    @property
    def config(self) -> dict:
        """Run provenance for the per-rollout meta record (see RolloutMeta)."""
        return {
            "model_id": self.model_id,
            "framing": self.framing,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": self.enable_thinking,
            "max_retries": self.max_retries,
        }

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:  # noqa: BLE001 - token counting must never break a rollout
            return 0

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]) -> AgentDecision:
        base_prompt = render_action_prompt(state_text, legal_actions, self.framing)
        last_decision: AgentDecision | None = None
        start = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            prompt = base_prompt
            if attempt > 0:
                prompt += (
                    "\n\nYour previous response was invalid. Return only one JSON object "
                    "with a legal integer action_index from the listed actions."
                )

            chat_prompt = self._apply_chat_template(prompt)
            response = self._generate_chat(chat_prompt)
            decision = parse_json_action(response, legal_actions)
            decision.retries = attempt
            decision.prompt_tokens = self._count_tokens(chat_prompt)
            decision.completion_tokens = self._count_tokens(response)
            decision.thinking_tokens = self._count_tokens(decision.thinking)
            last_decision = decision
            if decision.valid:
                break

        assert last_decision is not None
        last_decision.latency_s = round(time.perf_counter() - start, 4)
        return last_decision

    def _generate_chat(self, chat_prompt: str) -> str:
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

    def choose_actions_batch(self, items: list[tuple[str, list[LegalAction]]]) -> list[AgentDecision]:
        """Decide for K independent rollouts in one batched generation call (the
        cross-rollout throughput lever; see parallel_rollout). v1 does no per-item
        retry — invalid JSON falls back to index 0 like the single path. `latency_s`
        is the batch wall-time amortized across items (token counts are exact)."""
        if not items:
            return []
        if self._batch_generate is None:  # older mlx-lm: degrade to serial
            return [self.choose_action(st, la) for st, la in items]

        prompts = [self._apply_chat_template(render_action_prompt(st, la, self.framing)) for st, la in items]
        prompt_ids = [self.tokenizer.encode(p) for p in prompts]
        kwargs: dict = {"max_tokens": self.max_tokens}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler  # batch_generate defaults to greedy otherwise

        start = time.perf_counter()
        response = self._batch_generate(self.model, self.tokenizer, prompt_ids, **kwargs)
        per_item_latency = round((time.perf_counter() - start) / len(items), 4)

        decisions: list[AgentDecision] = []
        for (_, legal_actions), prompt, text in zip(items, prompts, response.texts):
            decision = parse_json_action(text, legal_actions)
            decision.retries = 0
            decision.prompt_tokens = self._count_tokens(prompt)
            decision.completion_tokens = self._count_tokens(text)
            decision.thinking_tokens = self._count_tokens(decision.thinking)
            decision.latency_s = per_item_latency
            decisions.append(decision)
        return decisions

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
