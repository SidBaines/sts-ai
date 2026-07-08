#!/usr/bin/env python
"""Evaluate a model or adapter on a detachable local-curriculum task."""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from sts_ai.agent_factory import build_agent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.local_tasks import base, get_task
from sts_ai.local_tasks.runner import (
    LocalTaskEnv,
    inject_local_task_meta,
    replay_task_start,
    run_local_task_episode,
    window_rollout_indices,
    windows_for_split,
    one_window_per_seed,
)
from sts_ai.rollout import current_git_sha
from sts_ai.seeding import rollout_stem
from sts_ai.streaming_rollout import run_streaming_rollouts


def _run_vllm_eval(
    *,
    args: argparse.Namespace,
    task,
    windows: list[dict],
    run_meta: dict,
    agent,
) -> int:
    by_seed = one_window_per_seed(windows)
    specs = []
    for window in windows:
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            out = args.output_dir / f"{rollout_stem(int(window['world_seed']), rollout_index)}.jsonl"
            if out.with_suffix(".meta.json").exists():
                continue  # completed episode from a prior (crashed/resumed) run
            if out.exists():
                out.unlink()  # partial episode: meta is written last, so no meta = incomplete
            specs.append((int(window["world_seed"]), rollout_index))

    def make_env(seed: int) -> LocalTaskEnv:
        env = LightspeedHybridEnv(
            world_seed=seed,
            combat_control="llm",
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )
        replay_task_start(env, by_seed[int(seed)])
        return LocalTaskEnv(env, task)

    results = run_streaming_rollouts(
        specs,
        make_env,
        agent,
        output_for=lambda ws, ri: args.output_dir / f"{rollout_stem(ws, ri)}.jsonl",
        concurrency=args.concurrency,
        max_decisions=args.max_decisions,
        max_retries=args.max_retries,
        run_meta=run_meta,
    )
    # Post-process every expected episode (not just this run's results) so
    # episodes completed by an earlier crashed process also get task metrics.
    for window in windows:
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            out = args.output_dir / f"{rollout_stem(int(window['world_seed']), rollout_index)}.jsonl"
            inject_local_task_meta(out.with_suffix(".meta.json"), out, task, window)
    return len(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-task eval.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("mlx", "vllm"), default="mlx")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--preserve-special-tokens",
        choices=("auto", "on", "off"),
        default="auto",
        help="vLLM-only: preserve native special tokens in completions.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument(
        "--rollouts-per-window",
        type=int,
        default=1,
        help="Sample K episodes per task window (policy seed varies with the "
        "rollout index; use temperature > 0 or all K repeats are identical).",
    )
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = base.load_manifest(args.manifest)
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(f"manifest task_id={manifest['task_id']!r} != --task {args.task!r}")
    windows = windows_for_split(manifest, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(
        args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_retries=args.max_retries,
        thinking=args.thinking,
        preserve_special_tokens={"auto": None, "on": True, "off": False}[
            args.preserve_special_tokens
        ],
        enable_prefix_caching=args.enable_prefix_caching,
        adapter_path=args.adapter_path,
    )
    run_meta = {
        "git_sha": current_git_sha(),
        "battle_simulations": args.battle_simulations,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": {
            "local_task_eval": True,
            "task_id": task.task_id,
            "split": args.split,
            "source_manifest": str(args.manifest),
            "adapter_path": args.adapter_path,
            "orchestrator": "streaming" if args.backend == "vllm" else "serial",
            "concurrency": args.concurrency if args.backend == "vllm" else 1,
        },
    }
    if args.backend == "vllm":
        if args.overwrite:
            for window in windows:
                for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
                    out = args.output_dir / f"{rollout_stem(int(window['world_seed']), rollout_index)}.jsonl"
                    for path in (out, out.with_suffix(".meta.json"), out.with_suffix(".error.json")):
                        if path.exists():
                            path.unlink()
        completed = _run_vllm_eval(
            args=args,
            task=task,
            windows=windows,
            run_meta=run_meta,
            agent=agent,
        )
        print(f"completed local-task eval episodes: {completed}")
        return

    completed = 0
    for window in windows:
        world_seed = int(window["world_seed"])
        # Local eval outputs identify task windows (and the sample index within
        # a window), not source rollout files: rollout_index = ordinal * K + k.
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            out = args.output_dir / f"{rollout_stem(world_seed, rollout_index)}.jsonl"
            if out.with_suffix(".meta.json").exists() and not args.overwrite:
                continue
            for path in (out, out.with_suffix(".meta.json"), out.with_suffix(".error.json")):
                if path.exists():
                    path.unlink()
            env = LightspeedHybridEnv(
                world_seed=world_seed,
                combat_control="llm",
                battle_simulations=args.battle_simulations,
                max_act=args.max_act,
            )
            replay_task_start(env, window)
            run_local_task_episode(
                task=task,
                env=env,
                agent=agent,
                window=window,
                max_decisions=args.max_decisions,
                output_path=out,
                run_meta=run_meta,
                rollout_index=rollout_index,
            )
            completed += 1
    print(f"completed local-task eval episodes: {completed}")


if __name__ == "__main__":
    main()
