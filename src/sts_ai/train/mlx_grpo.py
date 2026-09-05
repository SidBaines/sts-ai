"""MLX backend wiring for the GRPO outer loop (grpo_loop.run_grpo).

Lets the same loop run locally: MLX agent (MlxQwenJsonAgent) + lockstep parallel
rollouts + the in-process MLX LoRA PG trainer. No vLLM, no CUDA.
"""
from __future__ import annotations

import functools
from functools import partial
from typing import Any, Callable, NamedTuple

from sts_ai.parallel_rollout import run_parallel_rollouts


def run_parallel_as_streaming(
    specs: list[tuple[int, int]],
    make_env: Callable[[int], Any],
    agent: Any,
    *,
    output_for: Callable[[int, int], Any],
    concurrency: int,
    max_decisions: int,
    run_meta: dict[str, Any] | None = None,
    hint_cfg: Any | None = None,
    max_retries: int | None = None,
    policy_seed_salt: int = 0,
) -> Any:
    """Adapt GRPO's streaming rollout call to MLX lockstep parallel rollouts."""
    if hint_cfg is not None:
        raise ValueError("hinted rollouts are not supported on the lockstep MLX path")
    return run_parallel_rollouts(
        specs,
        make_env,
        agent,
        output_for=output_for,
        batch_size=concurrency,
        max_decisions=max_decisions,
        max_retries=max_retries,
        run_meta=run_meta,
        policy_seed_salt=policy_seed_salt,
    )


class MlxBackend(NamedTuple):
    agent: Any
    run_fn: Callable[..., Any]
    train_fn: Callable[..., Any]
    build_dataset_fn: Callable[..., Any]


def build_mlx_backend(
    *,
    base_model: str,
    framing: str,
    thinking: bool,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_seq_len: int = 4096,
    max_retries: int = 1,
    resume_adapter: str | None = None,
    output_contract: str = "reasoning_action",
    ooc_output_contract: str | None = None,
) -> MlxBackend:
    """Build the MLX agent, rollout shim, trainer, and PG dataset builder."""
    from sts_ai.agents import MlxQwenJsonAgent
    from sts_ai.train import train_pg_mlx
    from sts_ai.train.pg_dataset import build_pg_dataset

    agent = MlxQwenJsonAgent(
        model_id=base_model,
        framing=framing,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        enable_thinking=thinking,
        adapter_path=resume_adapter,
        output_contract=output_contract,
        ooc_output_contract=ooc_output_contract,
    )
    return MlxBackend(
        agent=agent,
        run_fn=run_parallel_as_streaming,
        train_fn=functools.partial(train_pg_mlx.train, max_seq_len=max_seq_len),
        build_dataset_fn=partial(build_pg_dataset, require_no_thinking=False),
    )
