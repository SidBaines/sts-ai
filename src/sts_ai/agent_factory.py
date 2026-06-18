from __future__ import annotations

from sts_ai.agents import FirstLegalAgent, MlxQwenJsonAgent, RandomLegalAgent, SimpleHeuristicAgent, VllmJsonAgent


def build_agent(
    agent_name: str,
    *,
    model: str = "mlx-community/Qwen3-4B-4bit",
    max_tokens: int = 4096,  # reasoning needs room to finish + emit JSON; see MlxQwenJsonAgent
    temperature: float = 0.2,
    top_p: float = 1.0,  # vLLM-only sampling controls (ignored by the MLX agent); the
    top_k: int = -1,     # defaults are vLLM's "disabled" sentinels (keep prior behaviour).
    max_retries: int = 1,
    thinking: bool = False,
    preserve_special_tokens: bool | None = None,
    enable_prefix_caching: bool = True,
    adapter_path: str | None = None,
    max_lora_rank: int = 16,
):
    if agent_name == "first":
        return FirstLegalAgent()
    if agent_name == "random":
        return RandomLegalAgent()
    if agent_name == "heuristic":
        return SimpleHeuristicAgent()
    if agent_name == "mlx":
        return MlxQwenJsonAgent(
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            enable_thinking=thinking,
            adapter_path=adapter_path,
        )
    if agent_name == "vllm":
        return VllmJsonAgent(
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_retries=max_retries,
            enable_thinking=thinking,
            preserve_special_tokens=preserve_special_tokens,
            enable_prefix_caching=enable_prefix_caching,
            adapter_path=adapter_path,
            max_lora_rank=max_lora_rank,
        )
    raise ValueError(f"unknown agent: {agent_name}")


def agent_label(
    agent_name: str,
    *,
    model: str = "mlx-community/Qwen3-4B-4bit",
    max_tokens: int = 4096,  # reasoning needs room to finish + emit JSON; see MlxQwenJsonAgent
    thinking: bool = False,
) -> str:
    if agent_name not in {"mlx", "vllm"}:
        return agent_name

    model_slug = model.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")
    mode = "thinking" if thinking else "nothinking"
    return f"{agent_name}_{model_slug}_{mode}_{max_tokens}"
