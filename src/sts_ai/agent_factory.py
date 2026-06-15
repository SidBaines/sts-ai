from __future__ import annotations

from sts_ai.agents import FirstLegalAgent, MlxQwenJsonAgent, RandomLegalAgent, SimpleHeuristicAgent, VllmJsonAgent


def build_agent(
    agent_name: str,
    *,
    model: str = "mlx-community/Qwen3-4B-4bit",
    max_tokens: int = 4096,  # reasoning needs room to finish + emit JSON; see MlxQwenJsonAgent
    temperature: float = 0.2,
    max_retries: int = 1,
    thinking: bool = False,
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
        )
    if agent_name == "vllm":
        return VllmJsonAgent(
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            enable_thinking=thinking,
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
