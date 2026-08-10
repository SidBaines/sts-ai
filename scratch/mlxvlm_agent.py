"""Minimal mlx-vlm-backed JSON-action agent for gemma-4-12B (gemma4_unified).

Same choose_action(state_text, legal_actions) -> AgentDecision interface as
MlxQwenJsonAgent, so the Nob probe can use it as a drop-in. mlx_lm cannot load
gemma4_unified; mlx-vlm can. Reuses render_action_prompt + parse_json_action
(incl. the gemma_thought extractor) so the reasoning/JSON handling matches the
mlx_lm path. Text-only generation (image=None, num_images=0).
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
from sts_ai.agents import parse_json_action
from sts_ai.prompting import NEUTRAL_FRAME, render_action_prompt
from sts_ai.schemas import LegalAction

_RETRY_SUFFIX = (
    "\n\nYour previous response was invalid. Return only one JSON object with a "
    "legal integer action_index from the listed actions."
)


class MlxVlmJsonAgent:
    name = "mlxvlm"

    def __init__(self, model_id, framing=NEUTRAL_FRAME, max_tokens=4096,
                 temperature=0.7, max_retries=1, enable_thinking=True):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        try:
            from mlx_vlm.utils import load_config
            self.config = load_config(model_id)
        except Exception:
            self.config = getattr(load, "__self__", {})
        self.model, self.processor = load(model_id)
        self._generate = generate
        self._apply_ct = apply_chat_template
        self.model_id = model_id
        self.framing = framing
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking

    def reseed(self, policy_seed: int) -> None:
        return None

    def _format(self, prompt: str) -> str:
        try:
            return self._apply_ct(self.processor, self.config, prompt,
                                  add_generation_prompt=True, num_images=0,
                                  enable_thinking=self.enable_thinking)
        except TypeError:
            return self._apply_ct(self.processor, self.config, prompt,
                                  add_generation_prompt=True, num_images=0)

    def choose_action(self, state_text: str, legal_actions: list[LegalAction]):
        base = render_action_prompt(state_text, legal_actions, self.framing)
        last = None
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            prompt = base + (_RETRY_SUFFIX if attempt > 0 else "")
            formatted = self._format(prompt)
            res = self._generate(self.model, self.processor, formatted, image=None,
                                 max_tokens=self.max_tokens, temperature=self.temperature,
                                 verbose=False)
            text = res.text if hasattr(res, "text") else str(res)
            ntok = int(getattr(res, "generation_tokens", 0) or 0)
            decision = parse_json_action(text, legal_actions,
                                         completion_tokens=ntok, max_tokens=self.max_tokens)
            decision.retries = attempt
            decision.completion_tokens = ntok
            last = decision
            if decision.valid:
                break
        last.latency_s = round(time.perf_counter() - start, 2)
        return last
