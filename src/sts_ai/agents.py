from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict
from typing import Protocol

from sts_ai.prompting import (
    ACTION_TEXT_OUTPUT,
    NEUTRAL_FRAME,
    REASONING_ACTION_OUTPUT,
    TURN_PLAN_OUTPUT,
    render_action_prompt,
    retry_instruction,
    validate_output_contract,
)
from sts_ai.schemas import AgentDecision, LegalAction


def resolve_output_contract(agent: object, phase: str | None) -> str:
    """The output contract an agent should use for a decision in ``phase``.

    Combat decisions (and phase-less calls, the historical path) use the
    agent's ``output_contract``; out-of-combat decisions switch to
    ``ooc_output_contract`` when one is set. This exists because semantic
    combat adapters speak exact action text, while out-of-combat menus (long
    event/shop strings) are only reliably answered via the index contract —
    measured 2026-08-14: action_text OOC drove both base and adapter to ~100%
    invalid stops at the first event screen."""
    ooc = getattr(agent, "ooc_output_contract", None)
    if phase is not None and phase != "combat" and ooc is not None:
        return ooc
    return getattr(agent, "output_contract", REASONING_ACTION_OUTPUT)


class ActionAgent(Protocol):
    name: str

    def reseed(self, policy_seed: int) -> None:
        ...

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        phase: str | None = None,
    ) -> AgentDecision:
        ...


class CompositeAgent:
    """Route combat decisions to one agent and out-of-combat decisions to
    another (e.g. a combat-SFT adapter that fights + the base model that
    navigates). Exists because semantic combat adapters catastrophically forget
    the index contract (measured 2026-08-14: `{"action_index":"play_card"}` on
    event screens), so no single contract lets them play whole games. As an
    experiment design bonus, arms built with the same ``ooc_agent`` share their
    out-of-combat policy exactly, isolating combat skill.

    Phase-less calls (``phase=None``) route to the combat agent, matching the
    historical single-agent default."""

    def __init__(self, combat_agent, ooc_agent, name: str | None = None) -> None:
        self.combat_agent = combat_agent
        self.ooc_agent = ooc_agent
        self.name = name or f"composite({combat_agent.name}+{ooc_agent.name})"

    def _agent_for(self, phase: str | None):
        return self.ooc_agent if phase is not None and phase != "combat" else self.combat_agent

    def reseed(self, policy_seed: int) -> None:
        self.combat_agent.reseed(policy_seed)
        self.ooc_agent.reseed(policy_seed)

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        phase: str | None = None,
    ) -> AgentDecision:
        return self._agent_for(phase).choose_action(state_text, legal_actions, phase=phase)

    def choose_actions_batch(
        self,
        items: list[tuple[str, list[LegalAction]]],
        retry_flags: list[bool] | None = None,
        phases: list[str | None] | None = None,
    ) -> list[AgentDecision]:
        if retry_flags is None:
            retry_flags = [False] * len(items)
        if phases is None:
            phases = [None] * len(items)
        combat_idx = [i for i, p in enumerate(phases) if p is None or p == "combat"]
        ooc_idx = [i for i, p in enumerate(phases) if not (p is None or p == "combat")]
        decisions: list[AgentDecision | None] = [None] * len(items)
        for agent, indices in ((self.combat_agent, combat_idx), (self.ooc_agent, ooc_idx)):
            if not indices:
                continue
            sub = agent.choose_actions_batch(
                [items[i] for i in indices],
                retry_flags=[retry_flags[i] for i in indices],
                phases=[phases[i] for i in indices],
            )
            for i, decision in zip(indices, sub):
                decisions[i] = decision
        assert all(d is not None for d in decisions)
        return decisions  # type: ignore[return-value]

    @property
    def config(self) -> dict:
        return {
            "agent": self.name,
            "composite": True,
            "combat_agent": dict(self.combat_agent.config),
            "ooc_agent": dict(self.ooc_agent.config),
            # Top-level identity mirrors the combat agent (the intervention arm).
            **{
                key: value
                for key, value in dict(self.combat_agent.config).items()
                if key in ("model_id", "framing", "temperature", "max_tokens", "thinking",
                           "max_retries", "output_contract", "adapter_path", "reasoning_mode")
            },
        }


class GenerationBackend(Protocol):
    def stream_submit(
        self,
        request_id: str,
        state_text: str,
        legal_actions: list[LegalAction],
        seed: int,
        retry: bool = False,
    ) -> None:
        ...

    def stream_poll(self) -> list[tuple[str, dict]]:
        ...

    def stream_has_unfinished(self) -> bool:
        ...

    def build_decision_from_text(
        self,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        legal_actions: list[LegalAction],
    ) -> AgentDecision:
        ...


class FirstLegalAgent:
    name = "first"

    def reseed(self, policy_seed: int) -> None:
        return None

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        phase: str | None = None,
    ) -> AgentDecision:
        return AgentDecision(action_index=0, raw_response="first legal action")


class RandomLegalAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reseed(self, policy_seed: int) -> None:
        self.rng = random.Random(policy_seed)

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        phase: str | None = None,
    ) -> AgentDecision:
        return AgentDecision(
            action_index=self.rng.randrange(len(legal_actions)),
            raw_response="random legal action",
        )


class SimpleHeuristicAgent:
    name = "heuristic"

    def reseed(self, policy_seed: int) -> None:
        return None

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        phase: str | None = None,
    ) -> AgentDecision:
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
    is marked invalid, and the rollout stops with `agent_invalid` after retries.
    This is harmless to
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
        adapter_path: str | None = None,
        output_contract: str = REASONING_ACTION_OUTPUT,
        ooc_output_contract: str | None = None,
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
        self.adapter_path = adapter_path
        validate_output_contract(output_contract)
        self.output_contract = output_contract
        if ooc_output_contract is not None:
            validate_output_contract(ooc_output_contract)
        self.ooc_output_contract = ooc_output_contract
        try:
            from mlx_lm import batch_generate
        except (ImportError, ModuleNotFoundError):
            batch_generate = None
        try:
            from mlx_lm import stream_generate
        except (ImportError, ModuleNotFoundError):
            stream_generate = None

        self.model, self.tokenizer = (
            load(model_id, adapter_path=adapter_path) if adapter_path else load(model_id)
        )
        self._generate = generate
        self._batch_generate = batch_generate
        self._stream_generate = stream_generate
        self._sampler = make_sampler(temp=temperature) if make_sampler is not None else None

    def reseed(self, policy_seed: int) -> None:
        import mlx.core as mx

        mx.random.seed(policy_seed)

    @property
    def config(self) -> dict:
        """Run provenance for the per-rollout meta record (see RolloutMeta)."""
        return {
            "model_id": self.model_id,
            "framing": self.framing,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": self.enable_thinking,
            "reasoning_mode": "native" if self.enable_thinking else "none",
            "max_retries": self.max_retries,
            "adapter_path": self.adapter_path,
            "output_contract": getattr(
                self, "output_contract", REASONING_ACTION_OUTPUT
            ),
            "ooc_output_contract": getattr(self, "ooc_output_contract", None),
        }

    def sleep(self, level: int = 1) -> None:
        """Release the policy model so the co-resident trainer can use memory."""
        self.model = None
        import mlx.core as mx

        mx.clear_cache()

    def wake(self) -> None:
        """Reload the model after sleep; no-op when already resident."""
        if self.model is not None:
            return

        from mlx_lm import load

        self.model, self.tokenizer = (
            load(self.model_id, adapter_path=self.adapter_path)
            if self.adapter_path
            else load(self.model_id)
        )

    def set_adapter(self, adapter_path: str) -> None:
        """Use a newly trained adapter on next wake (sleep -> train -> set -> wake)."""
        self.adapter_path = adapter_path
        self.model = None

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:  # noqa: BLE001 - token counting must never break a rollout
            return 0

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        prompt_override: str | None = None,
        phase: str | None = None,
    ) -> AgentDecision:
        # `prompt_override` (Interactive Studio advanced-template editor) bypasses
        # render_action_prompt with a fully-rendered user prompt. Default None keeps
        # the harness path byte-identical.
        contract = resolve_output_contract(self, phase)
        base_prompt = (
            prompt_override
            if prompt_override is not None
            else render_action_prompt(
                state_text,
                legal_actions,
                self.framing,
                output_contract=contract,
            )
        )
        last_decision: AgentDecision | None = None
        start = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            prompt = base_prompt
            if attempt > 0:
                prompt += retry_instruction(contract)

            chat_prompt = self._apply_chat_template(prompt)
            response = self._generate_chat(chat_prompt)
            completion_tokens = self._count_tokens(response)
            decision = parse_json_action(
                response,
                legal_actions,
                completion_tokens=completion_tokens,
                max_tokens=getattr(self, "max_tokens", None),
                output_contract=contract,
            )
            decision.retries = attempt
            decision.prompt_tokens = self._count_tokens(chat_prompt)
            decision.completion_tokens = completion_tokens
            decision.thinking_tokens = self._count_tokens(decision.thinking)
            last_decision = decision
            if decision.valid:
                break

        assert last_decision is not None
        last_decision.latency_s = round(time.perf_counter() - start, 4)
        return last_decision

    def stream_choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        prompt_override: str | None = None,
        phase: str | None = None,
    ):
        """Generator for the Interactive Studio's live token view. Yields each
        incremental text segment (str) as the model decodes, then RETURNS the
        parsed `AgentDecision` (read it from ``StopIteration.value``). No retries
        — streaming is single-shot/interactive. Raises if mlx-lm lacks
        stream_generate (the caller should fall back to choose_action)."""
        if self._stream_generate is None:
            raise RuntimeError("this mlx-lm build has no stream_generate; use choose_action")
        contract = resolve_output_contract(self, phase)
        base_prompt = (
            prompt_override
            if prompt_override is not None
            else render_action_prompt(
                state_text,
                legal_actions,
                self.framing,
                output_contract=contract,
            )
        )
        chat_prompt = self._apply_chat_template(base_prompt)
        kwargs: dict = {"max_tokens": self.max_tokens}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        pieces: list[str] = []
        start = time.perf_counter()
        for response in self._stream_generate(self.model, self.tokenizer, chat_prompt, **kwargs):
            segment = response.text
            pieces.append(segment)
            yield segment
        full = "".join(pieces)
        completion_tokens = self._count_tokens(full)
        decision = parse_json_action(
            full,
            legal_actions,
            completion_tokens=completion_tokens,
            max_tokens=self.max_tokens,
            output_contract=contract,
        )
        decision.retries = 0
        decision.prompt_tokens = self._count_tokens(chat_prompt)
        decision.completion_tokens = completion_tokens
        decision.thinking_tokens = self._count_tokens(decision.thinking)
        decision.latency_s = round(time.perf_counter() - start, 4)
        return decision

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

    def choose_actions_batch(
        self,
        items: list[tuple[str, list[LegalAction]]],
        retry_flags: list[bool] | None = None,
        phases: list[str | None] | None = None,
    ) -> list[AgentDecision]:
        """Decide for K independent rollouts in one batched generation call (the
        cross-rollout throughput lever; see parallel_rollout). `retry_flags`
        selects the JSON-only repair prompt per item. `latency_s` is the batch
        wall-time amortized across items (token counts are exact)."""
        if not items:
            return []
        if self._batch_generate is None:  # older mlx-lm: degrade to serial
            return [self.choose_action(st, la) for st, la in items]

        if retry_flags is None:
            retry_flags = [False] * len(items)
        if phases is None:
            phases = [None] * len(items)
        contracts = [resolve_output_contract(self, item_phase) for item_phase in phases]
        prompts = []
        for (state_text, legal_actions), retry, contract in zip(items, retry_flags, contracts):
            prompt = render_action_prompt(
                state_text,
                legal_actions,
                self.framing,
                output_contract=contract,
            )
            if retry:
                prompt += retry_instruction(contract)
            prompts.append(self._apply_chat_template(prompt))
        prompt_ids = [self.tokenizer.encode(p) for p in prompts]
        kwargs: dict = {"max_tokens": self.max_tokens}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler  # batch_generate defaults to greedy otherwise

        start = time.perf_counter()
        response = self._batch_generate(self.model, self.tokenizer, prompt_ids, **kwargs)
        per_item_latency = round((time.perf_counter() - start) / len(items), 4)

        decisions: list[AgentDecision] = []
        for (_, legal_actions), prompt, text, contract in zip(
            items, prompts, response.texts, contracts
        ):
            completion_tokens = self._count_tokens(text)
            decision = parse_json_action(
                text,
                legal_actions,
                completion_tokens=completion_tokens,
                max_tokens=getattr(self, "max_tokens", None),
                output_contract=contract,
            )
            decision.retries = 0
            decision.prompt_tokens = self._count_tokens(prompt)
            decision.completion_tokens = completion_tokens
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


class VllmJsonAgent:
    """vLLM-backed CUDA agent that emits a JSON action.

    IMPORTANT: keep `max_tokens` large (default 4096). Reasoning models can be
    truncated mid-thought by a small cap, leaving no closing JSON for
    `parse_json_action`; after retry exhaustion the rollout stops with
    `agent_invalid`. This is safe for no-thinking models because generation stops
    at EOS before the cap.

    Streaming primitives below drive vLLM continuous batching for
    `streaming_rollout.run_streaming_rollouts`; per-request seeds keep sampling
    independent of batch composition.
    """

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        framing: str = NEUTRAL_FRAME,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        # Nucleus / top-k sampling. Defaults are vLLM's "disabled" sentinels
        # (top_p=1.0 keeps all mass; top_k=-1 considers every token), so omitting
        # them reproduces the prior temperature-only behaviour. Gemma is typically
        # run at temperature~1.0, top_p=0.95, top_k=64.
        top_p: float = 1.0,
        top_k: int = -1,
        max_retries: int = 1,
        enable_thinking: bool = False,
        preserve_special_tokens: bool | None = None,
        enable_prefix_caching: bool = True,
        adapter_path: str | None = None,
        max_lora_rank: int = 16,
        enable_lora: bool = False,
        enable_sleep_mode: bool = False,
        # "auto" lets vLLM use each model's native dtype (bf16 for Qwen3/Gemma3/
        # Llama-3). Do NOT hardcode float16: Gemma3 rejects it ("does not support
        # float16, numerical instability — use bfloat16 or float32").
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        seed: int = 0,
        output_contract: str = REASONING_ACTION_OUTPUT,
        ooc_output_contract: str | None = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install with `.venv/bin/python -m pip install -e '.[vllm]'`."
            ) from exc

        self.model_id = model_id
        self.framing = framing
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.enable_prefix_caching = enable_prefix_caching
        self.adapter_path = adapter_path
        self.max_lora_rank = max_lora_rank
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self._seed = seed
        validate_output_contract(output_contract)
        self.output_contract = output_contract
        if ooc_output_contract is not None:
            validate_output_contract(ooc_output_contract)
        self.ooc_output_contract = ooc_output_contract

        llm_kwargs = {
            "model": model_id,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "trust_remote_code": True,
        }
        LoRARequest = None
        if enable_lora or adapter_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank
        if enable_sleep_mode:
            llm_kwargs["enable_sleep_mode"] = True
        if adapter_path:
            from vllm.lora.request import LoRARequest

        self.llm = LLM(**llm_kwargs)
        self._lora_counter = 1 if adapter_path else 0
        self._lora_request = LoRARequest("sts_lora", 1, adapter_path) if adapter_path else None
        self.tokenizer = self.llm.get_tokenizer()
        self._SamplingParams = SamplingParams
        self._native_thinking = self._probe_native_thinking()
        self.preserve_special_tokens = (
            self.reasoning_mode == "native"
            if preserve_special_tokens is None
            else preserve_special_tokens
        )

    def reseed(self, policy_seed: int) -> None:
        self._seed = policy_seed

    @property
    def config(self) -> dict:
        """Run provenance for the per-rollout meta record (see RolloutMeta)."""
        return {
            "model_id": self.model_id,
            "framing": self.framing,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "thinking": self.enable_thinking,
            "enable_prefix_caching": self.enable_prefix_caching,
            "reasoning_mode": self.reasoning_mode,
            "preserve_special_tokens": self.preserve_special_tokens,
            "max_retries": self.max_retries,
            "backend": "vllm",
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "adapter_path": self.adapter_path,
            "output_contract": getattr(
                self, "output_contract", REASONING_ACTION_OUTPUT
            ),
            "ooc_output_contract": getattr(self, "ooc_output_contract", None),
        }

    @property
    def reasoning_mode(self) -> str:
        if not self.enable_thinking:
            return "none"
        if self._native_thinking:
            return "native"
        return "prompted"

    def sleep(self, level: int = 1) -> None:
        """Call vLLM offline `LLM.sleep(level)` to free cache/GPU memory."""
        self.llm.sleep(level)

    def wake(self) -> None:
        """Call vLLM offline `LLM.wake_up()` after sleep-mode training."""
        self.llm.wake_up()

    def set_adapter(self, adapter_path: str) -> None:
        """Hot-swap a LoRA via vLLM `LoRARequest` with a fresh adapter id."""
        from vllm.lora.request import LoRARequest

        self._lora_counter += 1
        self._lora_request = LoRARequest("sts_lora", self._lora_counter, adapter_path)
        self.adapter_path = adapter_path

    def _probe_native_thinking(self) -> bool:
        messages = [{"role": "user", "content": "ping"}]
        try:
            with_thinking = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            without_thinking = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return False
        return with_thinking != without_thinking

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(self.reasoning_mode == "native"),
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _base_prompt(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        output_contract: str | None = None,
    ) -> str:
        return render_action_prompt(
            state_text,
            legal_actions,
            self.framing,
            induce_reasoning=(self.reasoning_mode == "prompted"),
            output_contract=(
                output_contract
                if output_contract is not None
                else getattr(self, "output_contract", REASONING_ACTION_OUTPUT)
            ),
        )

    def _render_prompt(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        output_contract: str | None = None,
    ) -> str:
        prompt = self._base_prompt(state_text, legal_actions, output_contract)
        return self._apply_chat_template(prompt)

    def _generate(self, prompts: list[str]) -> list[dict] | None:
        try:
            params = self._SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                seed=self._seed,
                skip_special_tokens=not getattr(self, "preserve_special_tokens", False),
            )
            if getattr(self, "_lora_request", None) is not None:
                outputs = self.llm.generate(prompts, params, lora_request=self._lora_request)
            else:
                outputs = self.llm.generate(prompts, params)
            return [
                {
                    "text": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                }
                for output in outputs
            ]
        except Exception:  # noqa: BLE001 - generation failures should not kill a rollout sweep
            return None

    def stream_submit(
        self,
        request_id: str,
        state_text: str,
        legal_actions: list[LegalAction],
        seed: int,
        retry: bool = False,
        phase: str | None = None,
    ) -> None:
        contract = resolve_output_contract(self, phase)
        base = self._base_prompt(state_text, legal_actions, contract)
        if retry:
            base += retry_instruction(contract)
        # Parsing happens later (stream_poll -> build_decision_from_text), so the
        # per-request contract must survive until then. hasattr guard keeps
        # object.__new__ test instances working.
        if not hasattr(self, "_request_contracts"):
            self._request_contracts = {}
        self._request_contracts[request_id] = contract
        prompt = self._apply_chat_template(base)
        params = self._SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            seed=seed,
            skip_special_tokens=not getattr(self, "preserve_special_tokens", False),
        )
        # Record submit time so stream_poll can report this request's submit->finish
        # wall-time. hasattr guard keeps object.__new__ test instances working.
        if not hasattr(self, "_submit_ts"):
            self._submit_ts = {}
        self._submit_ts[request_id] = time.perf_counter()
        # NOTE: vLLM's low-level LLMEngine.add_request signature is version-sensitive.
        # Verified target API: add_request(request_id, prompt, params). Keep this call
        # isolated here so a vLLM version bump is a one-line change.
        if getattr(self, "_lora_request", None) is not None:
            self.llm.llm_engine.add_request(request_id, prompt, params, lora_request=self._lora_request)
        else:
            self.llm.llm_engine.add_request(request_id, prompt, params)

    def stream_poll(self) -> list[tuple[str, dict]]:
        finished: list[tuple[str, dict]] = []
        submit_ts = getattr(self, "_submit_ts", {})
        now = time.perf_counter()
        for output in self.llm.llm_engine.step():
            if output.finished:
                started = submit_ts.pop(output.request_id, None)
                # submit->finish wall-time for this request: in continuous batching this
                # is the per-decision latency (queue wait + decode). 0.0 if unrecorded.
                latency_s = round(now - started, 4) if started is not None else 0.0
                finished.append(
                    (
                        output.request_id,
                        {
                            "text": output.outputs[0].text,
                            "prompt_tokens": len(output.prompt_token_ids),
                            "completion_tokens": len(output.outputs[0].token_ids),
                            "latency_s": latency_s,
                        },
                    )
                )
        return finished

    def stream_has_unfinished(self) -> bool:
        return self.llm.llm_engine.has_unfinished_requests()

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:  # noqa: BLE001 - token counting must never break a rollout
            return 0

    def build_decision_from_text(
        self,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        legal_actions: list[LegalAction],
        request_id: str | None = None,
        output_contract: str | None = None,
    ) -> AgentDecision:
        contract = output_contract
        if contract is None and request_id is not None:
            contract = getattr(self, "_request_contracts", {}).pop(request_id, None)
        if contract is None:
            contract = resolve_output_contract(self, None)
        decision = parse_json_action(
            text,
            legal_actions,
            completion_tokens=completion_tokens,
            max_tokens=getattr(self, "max_tokens", None),
            output_contract=contract,
        )
        decision.retries = 0
        decision.prompt_tokens = prompt_tokens
        decision.completion_tokens = completion_tokens
        decision.thinking_tokens = self._count_tokens(decision.thinking)
        return decision

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
        prompt_override: str | None = None,
        phase: str | None = None,
    ) -> AgentDecision:
        # `prompt_override` (Interactive Studio advanced-template editor) bypasses
        # _base_prompt with a fully-rendered user prompt. Default None keeps the
        # harness path byte-identical.
        contract = resolve_output_contract(self, phase)
        base_prompt = (
            prompt_override
            if prompt_override is not None
            else self._base_prompt(state_text, legal_actions, contract)
        )
        last_decision: AgentDecision | None = None
        start = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            prompt = base_prompt
            if attempt > 0:
                prompt += retry_instruction(contract)

            chat_prompt = self._apply_chat_template(prompt)
            results = self._generate([chat_prompt])

            if results is None:
                decision = AgentDecision(
                    action_index=0,
                    raw_response="",
                    valid=False,
                    metadata={"error": "vllm generation failed"},
                )
            else:
                result = results[0]
                decision = parse_json_action(
                    result["text"],
                    legal_actions,
                    completion_tokens=result["completion_tokens"],
                    max_tokens=getattr(self, "max_tokens", None),
                    output_contract=contract,
                )
                decision.prompt_tokens = result["prompt_tokens"]
                decision.completion_tokens = result["completion_tokens"]
                decision.thinking_tokens = self._count_tokens(decision.thinking)

            decision.retries = attempt
            last_decision = decision
            if decision.valid:
                break

        assert last_decision is not None
        last_decision.latency_s = round(time.perf_counter() - start, 4)
        return last_decision

    def choose_actions_batch(
        self,
        items: list[tuple[str, list[LegalAction]]],
        retry_flags: list[bool] | None = None,
        phases: list[str | None] | None = None,
    ) -> list[AgentDecision]:
        """Decide for K independent rollouts in one vLLM generation call."""
        if not items:
            return []

        if retry_flags is None:
            retry_flags = [False] * len(items)
        if phases is None:
            phases = [None] * len(items)
        contracts = [resolve_output_contract(self, item_phase) for item_phase in phases]
        prompts = []
        for (state_text, legal_actions), retry, contract in zip(items, retry_flags, contracts):
            if retry:
                base = render_action_prompt(
                    state_text,
                    legal_actions,
                    self.framing,
                    output_contract=contract,
                )
                base += retry_instruction(contract)
                prompts.append(self._apply_chat_template(base))
            else:
                prompts.append(self._render_prompt(state_text, legal_actions, contract))
        start = time.perf_counter()
        results = self._generate(prompts)
        per_item_latency = round((time.perf_counter() - start) / len(items), 4)

        if results is None:
            return [
                AgentDecision(
                    action_index=0,
                    raw_response="",
                    valid=False,
                    latency_s=per_item_latency,
                    metadata={"error": "vllm generation failed"},
                )
                for _ in items
            ]

        # vLLM returns outputs in input order, so positional zip preserves item alignment.
        assert len(results) == len(prompts)

        decisions: list[AgentDecision] = []
        for (_, legal_actions), result, contract in zip(items, results, contracts):
            decision = self.build_decision_from_text(
                result["text"],
                result["prompt_tokens"],
                result["completion_tokens"],
                legal_actions,
                output_contract=contract,
            )
            decision.latency_s = per_item_latency
            decisions.append(decision)
        return decisions


def _resolve_semantic_action(
    parsed: dict,
    legal_actions: list[LegalAction],
) -> tuple[int | None, dict[str, object]]:
    """Map a semantic-contract JSON object to a legal-action list position.

    Mirrors ``teacher_action_eval.evaluate_generated_action``'s exact-text
    matching (byte-exact membership, first matching position) but is
    deliberately laxer about the surrounding schema: live play executes the
    model's clearly-stated intent, so extra keys are tolerated and a valid
    integer ``action_index`` is accepted as a fallback when no ``action`` text
    matches (the menu still displays indices). The static scorer remains the
    strict measurement surface.
    """
    metadata: dict[str, object] = {}
    plan = parsed.get("plan")
    if isinstance(plan, list):
        metadata["plan"] = plan
    action = parsed.get("action")
    if isinstance(action, str):
        # Models often copy the rendered menu line verbatim, which includes the
        # displayed index ("0: play Strike ..."). Strip that prefix before
        # matching — the stated intent is unambiguous.
        menu_prefix = re.match(r"^(\d+):\s+(.*)$", action)
        if menu_prefix is not None and menu_prefix.group(2):
            action = menu_prefix.group(2)
            metadata["semantic_match"] = "stripped_menu_index"
        descriptions = [legal.description for legal in legal_actions]
        if action in descriptions:
            return descriptions.index(action), metadata
        # A truncated-but-unambiguous copy (e.g. dropping the appended
        # "-> TARGET (deal N)" annotation) resolves iff it prefixes exactly
        # one legal description; any ambiguity stays invalid.
        if action.strip():
            prefix_hits = [
                position
                for position, description in enumerate(descriptions)
                if description.startswith(action)
            ]
            if len(prefix_hits) == 1:
                metadata["semantic_match"] = "unique_prefix"
                return prefix_hits[0], metadata
    fallback = parsed.get("action_index")
    if (
        isinstance(fallback, int)
        and not isinstance(fallback, bool)
        and 0 <= fallback < len(legal_actions)
    ):
        metadata["semantic_fallback"] = "action_index"
        return fallback, metadata
    return None, metadata


def parse_json_action(
    response: str,
    legal_actions: list[LegalAction],
    *,
    completion_tokens: int | None = None,
    max_tokens: int | None = None,
    output_contract: str = REASONING_ACTION_OUTPUT,
) -> AgentDecision:
    extracted = _extract_json_with_span(response)
    parsed = extracted[0] if extracted is not None else None
    json_span = extracted[1] if extracted is not None else None
    thinking, thinking_meta = _extract_thinking_with_metadata(response, json_span)
    base_metadata: dict[str, object] = dict(thinking_meta)
    if json_span is not None:
        base_metadata["json_span"] = [json_span[0], json_span[1]]

    if parsed is None:
        error = "no json object"
        if completion_tokens is not None and max_tokens is not None and completion_tokens >= max_tokens:
            error = "truncated_before_json"
        base_metadata["error"] = error
        base_metadata["parse_error"] = error
        return AgentDecision(
            action_index=0,
            raw_response=response,
            thinking=thinking,
            valid=False,
            metadata=base_metadata,
        )

    reasoning = str(parsed.get("reasoning", ""))
    if output_contract in (ACTION_TEXT_OUTPUT, TURN_PLAN_OUTPUT):
        resolved, semantic_metadata = _resolve_semantic_action(parsed, legal_actions)
        base_metadata.update(semantic_metadata)
        base_metadata["parsed"] = parsed
        if resolved is None:
            base_metadata["error"] = "unmatched action text"
            base_metadata["parse_error"] = "unmatched action text"
            return AgentDecision(
                action_index=0,
                raw_response=response,
                reasoning=reasoning,
                thinking=thinking,
                valid=False,
                metadata=base_metadata,
            )
        base_metadata["legal_action"] = asdict(legal_actions[resolved])
        return AgentDecision(
            action_index=resolved,
            raw_response=response,
            reasoning=reasoning,
            thinking=thinking,
            valid=True,
            metadata=base_metadata,
        )

    action_index = parsed.get("action_index")
    if not isinstance(action_index, int) or action_index < 0 or action_index >= len(legal_actions):
        base_metadata["error"] = "invalid action_index"
        base_metadata["parse_error"] = "invalid action_index"
        base_metadata["parsed"] = parsed
        return AgentDecision(
            action_index=0,
            raw_response=response,
            reasoning=reasoning,
            thinking=thinking,
            valid=False,
            metadata=base_metadata,
        )

    base_metadata["parsed"] = parsed
    base_metadata["legal_action"] = asdict(legal_actions[action_index])
    return AgentDecision(
        action_index=action_index,
        raw_response=response,
        reasoning=reasoning,
        thinking=thinking,
        valid=True,
        metadata=base_metadata,
    )


def _extract_thinking_with_metadata(
    text: str,
    json_span: tuple[int, int] | None,
) -> tuple[str, dict[str, object]]:
    """Return the model's chain-of-thought plus parser-quality metadata.

    Handles two reasoning formats: the ``<think>…</think>`` block (Qwen3 native and
    the prompted-reasoning path), and **Gemma-4's native ``thought`` channel** (see
    ``_extract_gemma_thought``). If a block is opened but never closed before the
    final JSON, the thinking text is bounded at the JSON start so the action payload
    is not stored as CoT; with no JSON, the partial text is kept for truncation audits.
    """
    lower = text.lower()
    json_start = json_span[0] if json_span is not None else None
    metadata: dict[str, object] = {
        "thinking_closed": False,
        "thinking_truncated": False,
        "stray_think_close": False,
        "json_inside_unclosed_think": False,
    }

    # --- <think>…</think> (Qwen3 native + prompted reasoning) ---
    start = lower.find("<think>")
    metadata["stray_think_close"] = start == -1 and lower.find("</think>") != -1
    if start != -1:
        inner_start = start + len("<think>")
        end = lower.find("</think>", inner_start)
        if end != -1 and (json_start is None or end <= json_start):
            metadata["thinking_closed"] = True
            return text[inner_start:end].strip(), metadata
        if json_start is not None and json_start > inner_start:
            metadata["json_inside_unclosed_think"] = True
            return text[inner_start:json_start].strip(), metadata
        metadata["thinking_truncated"] = True
        return text[inner_start:].strip(), metadata

    # --- Gemma-4 native `thought` channel ---
    thinking, gemma_meta = _extract_gemma_thought(text, json_start)
    if thinking is not None:
        metadata.update(gemma_meta)
        return thinking, metadata

    return "", metadata


def _strip_answer_fence(text: str) -> str:
    """Trim a trailing markdown code-fence opener (```` ```json ````) that precedes
    the JSON answer, so it is not stored as part of the captured reasoning."""
    return re.sub(r"`{3,}(?:json)?\s*$", "", text.strip(), flags=re.IGNORECASE).strip()


def _extract_gemma_thought(
    text: str,
    json_start: int | None,
) -> tuple[str | None, dict[str, object]]:
    """Capture Gemma-4 native reasoning.

    Gemma-4 wraps chain-of-thought in a ``<|channel>thought\\n … <channel|>`` block.
    vLLM's default ``skip_special_tokens=True`` drops the ``<|channel>`` markers, so
    the completion we receive simply *starts with* the ``thought`` role label
    followed by the reasoning and then the JSON answer (the stripped form we observe
    on the H100). This handles both: bound the reasoning at the channel close if it
    survived, else at the JSON answer start. Returns ``(None, {})`` when the text is
    not a Gemma-thought completion, so callers fall through to "no thinking".
    """
    lower = text.lower()
    open_idx = lower.find("<|channel>")
    if open_idx != -1:  # special tokens preserved (skip_special_tokens=False)
        label = lower.find("thought", open_idx)
        body_start = (label + len("thought")) if (label != -1 and label - open_idx <= 12) else open_idx + len("<|channel>")
    elif lower.lstrip().startswith("thought"):  # stripped form (the default)
        body_start = (len(text) - len(text.lstrip())) + len("thought")
    else:
        return None, {}

    meta: dict[str, object] = {"reasoning_format": "gemma_thought", "thinking_closed": False, "thinking_truncated": False}
    for close in ("<channel|>", "</channel>", "<|channel|>"):
        c = lower.find(close, body_start)
        if c != -1 and (json_start is None or c <= json_start):
            meta["thinking_closed"] = True
            return _strip_answer_fence(text[body_start:c]), meta
    if json_start is not None and json_start > body_start:
        return _strip_answer_fence(text[body_start:json_start]), meta
    meta["thinking_truncated"] = True
    return _strip_answer_fence(text[body_start:]), meta


def _extract_json_with_span(text: str) -> tuple[dict, tuple[int, int]] | None:
    stripped = text.strip()
    offset = len(text) - len(text.lstrip())
    parsed = _parse_json_dict(stripped)
    if parsed is not None:
        return parsed, (offset, offset + len(stripped))

    for candidate, span in reversed(_balanced_json_candidates(text)):
        parsed = _parse_json_dict(candidate)
        if parsed is not None:
            return parsed, span

    return None


def _parse_json_dict(text: str) -> dict | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _balanced_json_candidates(text: str) -> list[tuple[str, tuple[int, int]]]:
    candidates: list[tuple[str, tuple[int, int]]] = []
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
                candidates.append((text[start : idx + 1], (start, idx + 1)))
                start = None

    return candidates
