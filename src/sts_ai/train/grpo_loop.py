"""In-process GRPO outer loop over streaming rollouts and TRL adapter updates.

Periodic eval is intentionally out of v1: run eval separately with
``run_until.py --split eval`` and compare adapters with ``compare_paired.py``.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable

from sts_ai.seeding import expand_specs, rollout_stem
from sts_ai.streaming_rollout import run_streaming_rollouts
from sts_ai.rollout import current_git_sha
from sts_ai.train import train_pg_trl
from sts_ai.train.pg_dataset import build_pg_dataset

train_pg_trl_train = train_pg_trl.train

log = logging.getLogger(__name__)


def select_iteration_seeds(
    train_seeds: list[int],
    iteration: int,
    seeds_per_iter: int,
) -> list[int]:
    """Return a deterministic rotating seed window for one GRPO iteration."""
    if iteration < 0:
        raise ValueError("iteration must be >= 0")
    if seeds_per_iter < 1:
        raise ValueError("seeds_per_iter must be >= 1")
    if not train_seeds:
        return []
    if seeds_per_iter >= len(train_seeds):
        return list(train_seeds)

    start = (iteration * seeds_per_iter) % len(train_seeds)
    return [
        train_seeds[(start + offset) % len(train_seeds)]
        for offset in range(seeds_per_iter)
    ]


def _manifest_path(dataset_path: Path) -> Path:
    return Path(str(dataset_path) + ".manifest.json")


def _write_dataset(
    dataset_path: Path,
    examples: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    manifest_path = _manifest_path(dataset_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _run_meta(
    *,
    iteration: int,
    num_iterations: int,
    group_size: int,
    seeds: list[int],
    concurrency: int,
    max_decisions: int,
) -> dict[str, Any]:
    return {
        "git_sha": current_git_sha(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": {
            "orchestrator": "grpo_loop",
            "iteration": iteration,
            "num_iterations": num_iterations,
            "group_size": group_size,
            "seeds": list(seeds),
            "concurrency": concurrency,
            "max_decisions": max_decisions,
        },
    }


def run_grpo(
    *,
    agent: Any,
    make_env: Callable[[int], Any],
    base_model: str,
    tokenizer: Any,
    tokenizer_id: str,
    framing: str,
    train_seeds: list[int],
    out_dir: Path,
    num_iterations: int,
    group_size: int = 8,
    seeds_per_iter: int = 8,
    concurrency: int = 48,
    max_decisions: int = 1500,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
    learning_rate: float = 1e-5,
    std_norm: bool = True,
    eps: float = 1e-6,
    build_dataset_fn: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = build_pg_dataset,
    train_fn: Callable[..., Path] = train_pg_trl_train,
    run_streaming_fn: Callable[..., Any] = run_streaming_rollouts,
) -> dict[str, Any]:
    """Run in-process GRPO with vLLM sleep/wake and LoRA hot-swap.

    Eval is intentionally not interleaved in v1; run it separately with
    ``run_until.py --split eval`` and ``compare_paired.py``.
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_decisions < 1:
        raise ValueError("max_decisions must be >= 1")
    if not train_seeds:
        raise ValueError("train_seeds must not be empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iterations: list[dict[str, Any]] = []
    current_adapter: str | None = None

    for iteration in range(num_iterations):
        agent.wake()
        seeds = select_iteration_seeds(train_seeds, iteration, seeds_per_iter)
        specs = expand_specs(seeds, group_size)
        iter_dir = out_dir / f"iter_{iteration}"
        rollouts_dir = iter_dir / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_streaming_fn(
                specs,
                make_env,
                agent,
                output_for=lambda ws, ri, d=rollouts_dir: (
                    d / f"{rollout_stem(ws, ri)}.jsonl"
                ),
                concurrency=concurrency,
                max_decisions=max_decisions,
                run_meta=_run_meta(
                    iteration=iteration,
                    num_iterations=num_iterations,
                    group_size=group_size,
                    seeds=seeds,
                    concurrency=concurrency,
                    max_decisions=max_decisions,
                ),
                hint_cfg=None,
            )
        finally:
            agent.sleep()

        examples, manifest = build_dataset_fn(
            rollouts_dir,
            framing=framing,
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            mode="group",
            std_norm=std_norm,
            eps=eps,
        )
        dataset_path = iter_dir / "pg.jsonl"
        manifest_path = _write_dataset(dataset_path, examples, manifest)

        new_adapter = iter_dir / "adapter"
        train_fn(
            dataset_path=dataset_path,
            base_model=base_model,
            out_adapter_dir=new_adapter,
            init_adapter_path=current_adapter,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
            learning_rate=learning_rate,
            manifest_path=manifest_path,
        )
        current_adapter = str(new_adapter)
        agent.set_adapter(current_adapter)

        stats = {
            "iteration": iteration,
            "seeds": seeds,
            "n_specs": len(specs),
            "n_examples": len(examples),
            "advantage_report": manifest.get("advantage_report", {}),
            "current_adapter": current_adapter,
        }
        iterations.append(stats)
        log.info(
            "grpo iter=%s specs=%s examples=%s adapter=%s advantage_report=%s",
            iteration,
            len(specs),
            len(examples),
            current_adapter,
            stats["advantage_report"],
        )

    return {"iterations": iterations, "final_adapter": current_adapter}
