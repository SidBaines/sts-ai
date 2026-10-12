#!/usr/bin/env python3
"""Bounded BC, PPO and isolated evaluation for public-state candidate policies."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from sts_policy.curriculum import TaskSpec, world_split, TASK_VERSION, REWARD_VERSION
from sts_policy.encoding import ENCODER_VERSION
from sts_policy.losses import PPOConfig
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import (collate_public, forward_policy, cached_public_collator,
                                    CompositeActorCritic, policy_from_payload)
from sts_policy.training import (train_bc, evaluate_bc, save_checkpoint, load_checkpoint,
                                 file_identity, seed_training)
from sts_policy.training import warm_optimizer
from sts_policy.rollout_training import collect_rollouts, update_ppo
from sts_policy.workers import CurriculumWorker


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w") as output:
        json.dump(metrics, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def source_model(args) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    seed_training(args.seed)
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint)
        if payload["metadata"].get("encoder_version") != ENCODER_VERSION:
            raise ValueError("checkpoint encoder version mismatch")
        model = policy_from_payload(payload)
        return model.to(args.device), payload
    return CandidateActorCritic(args.size, args.context).to(args.device), None


def teacher_rows(path: Path, phase: str, split: str) -> list[dict[str, Any]]:
    rows = read_rows(path)
    for row in rows:
        if row.get("split", world_split(row["world_seed"])) != world_split(row["world_seed"]):
            raise ValueError("teacher split differs from the fixed world split")
    return [row for row in rows if world_split(row["world_seed"]) == split and
            (phase == "all" or row["observation"].get("phase") ==
             ("combat" if phase == "combat" else "out_of_combat"))]


def tasks_for(args) -> list[TaskSpec]:
    if args.tasks:
        tasks = [TaskSpec.from_dict(row) for row in read_rows(args.tasks)]
        if args.task_kind != "battle":
            tasks = list({task.seed: task for task in tasks}.values())
            tasks = [TaskSpec(seed=task.seed, kind=args.task_kind, horizon=args.horizon,
                              max_act=args.max_act, battle_simulations=args.simulations) for task in tasks]
        else:
            tasks = [replace(task, horizon=args.horizon, max_act=args.max_act,
                             battle_simulations=args.simulations) for task in tasks]
    else:
        tasks = []
        seed = args.world_start
        while len(tasks) < args.episodes:
            if world_split(seed) == args.split:
                tasks.append(TaskSpec(seed=seed, kind=args.task_kind, horizon=args.horizon,
                                      max_act=args.max_act, battle_simulations=args.simulations))
            seed += 1
    tasks = [task for task in tasks if world_split(task.seed) == args.split]
    random.Random(args.seed).shuffle(tasks)
    if not tasks:
        raise ValueError("no tasks remain after world split filtering")
    return tasks[:args.episodes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bc", "ppo", "evaluate"))
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--dev-teacher", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--combat-checkpoint", type=Path)
    parser.add_argument("--ooc-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="checkpoint path; evaluate uses metrics only")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--resume", action="store_true", help="restore optimizer/RNG; require identical training data")
    parser.add_argument("--warm-optimizer", action="store_true", help="retain Adam moments for compatible PPO chunks; stage changes reset")
    parser.add_argument("--phase", choices=("all", "combat", "ooc"), default="all")
    parser.add_argument("--task-kind", choices=("battle", "ooc", "run"), default="battle")
    parser.add_argument("--split", choices=("train", "dev", "final"), default="train")
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--context", choices=("none", "mean"), default="mean")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--inference-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--fast-observations", action="store_true")
    parser.add_argument("--fast-replay", action="store_true")
    parser.add_argument("--deferred-bootstrap", action="store_true")
    parser.add_argument("--reward-mode", choices=("potential", "sparse", "progress"), default="potential")
    parser.add_argument("--terminal-hp-weight", type=float, default=0.)
    parser.add_argument("--progress-floor-weight", type=float, default=.05)
    parser.add_argument("--goal-act-clear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--metric-interval", type=int, default=32)
    parser.add_argument("--bucket-actions", action="store_true")
    parser.add_argument("--tensor-cache", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--world-start", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--max-act", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--teacher-simulations", type=int, help="collect DAgger labels during PPO")
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    parser.add_argument("--clip-ratio", type=float, default=.2)
    parser.add_argument("--timeout", type=float, default=10.)
    args = parser.parse_args()
    if min(args.seconds, args.updates, args.batch_size, args.workers, args.episodes, args.threads) <= 0:
        parser.error("time, updates, batch size, workers, episodes and threads must be positive")
    if args.command == "bc" and args.teacher is None:
        parser.error("BC requires --teacher")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    if args.warm_optimizer and (args.checkpoint is None or args.resume or args.command != "ppo"):
        parser.error("--warm-optimizer requires PPO --checkpoint and cannot combine with --resume")
    if args.command != "evaluate" and args.split != "train":
        parser.error("training is restricted to train worlds")
    if args.command == "evaluate" and not (args.checkpoint or args.combat_checkpoint):
        parser.error("evaluate requires a checkpoint")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if not all(math.isfinite(value) and value >= 0 for value in (args.terminal_hp_weight, args.progress_floor_weight)):
        parser.error("reward weights must be finite and nonnegative")
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, payload = source_model(args)
    if args.combat_checkpoint or args.ooc_checkpoint:
        if args.command != "evaluate" or args.checkpoint or not (args.combat_checkpoint and args.ooc_checkpoint):
            parser.error("both specialized checkpoints are supported for evaluate only")
        policies = []
        for path in (args.combat_checkpoint, args.ooc_checkpoint):
            state = load_checkpoint(path)
            if state["metadata"].get("encoder_version") != ENCODER_VERSION:
                raise ValueError("composite policy encoder mismatch")
            policy = policy_from_payload(state)
            if not isinstance(policy, CandidateActorCritic):
                raise ValueError("two-source evaluation requires candidate policy heads")
            policies.append(policy.to(args.device))
        model = CompositeActorCritic(*policies).to(args.device)
    model_config = model.metadata() if hasattr(model, "metadata") else {}
    metrics: dict[str, Any] = {"command": args.command, "config": {k: str(v) if isinstance(v, Path) else v
                                                                           for k, v in vars(args).items()},
                               "encoder_version": ENCODER_VERSION, "task_version": TASK_VERSION,
                               "model_config": model_config}
    lineage = {}
    if payload is not None:
        lineage["parent_policy_checkpoint"] = {"path": str(args.checkpoint),
                                                "sha256": file_identity(args.checkpoint),
                                                "model_config": payload["model_config"]}
        if "composition_provenance" in payload["metadata"]:
            lineage["composition_provenance"] = payload["metadata"]["composition_provenance"]
    from sts_policy.composition import training_environment_record
    lineage["training_environment"] = training_environment_record()
    metrics["training_environment"] = lineage["training_environment"]
    lineage["model_kind"] = model_config.get("kind", "candidate")
    if lineage["model_kind"] == "composite":
        lineage["model_config_version"] = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    step = payload["step"] if payload and args.resume else 0
    identity = file_identity(args.teacher) if args.command == "bc" else (
        file_identity(args.tasks) if args.tasks else f"generated:{args.world_start}:{args.episodes}:{args.task_kind}:{args.split}")
    reward_config = {"gamma": args.gamma, "reward_mode": args.reward_mode,
                     "goal_act_clear": args.goal_act_clear,
                     "reward_version": REWARD_VERSION,
                     "progress_floor_weight": args.progress_floor_weight,
                     "terminal_hp_weight": args.terminal_hp_weight}
    metrics["reward_config"] = reward_config
    optimizer_carry = {"restored": False, "reason": "fresh_optimizer"}
    if args.warm_optimizer:
        optimizer_carry = warm_optimizer(optimizer, payload, method=args.command, task_kind=args.task_kind,
                                        reward_config=reward_config, learning_rate=args.learning_rate)
        if optimizer_carry["restored"]:
            step = payload["step"]
        optimizer_carry["source_checkpoint_identity"] = file_identity(args.checkpoint)
        optimizer_carry["target_data_identity"] = identity
    metrics["optimizer_carry"] = optimizer_carry
    if args.resume:
        load_checkpoint(args.checkpoint, model, optimizer=optimizer, restore_rng=True, expected_data_identity=identity)
        if payload["metadata"].get("reward_config") != reward_config:
            raise ValueError("resumed checkpoint reward configuration differs")
        previous_config = payload["metadata"].get("config", {})
        for key in ("phase", "task_kind", "horizon", "max_act", "simulations"):
            if key in previous_config and previous_config[key] != getattr(args, key):
                raise ValueError(f"resumed checkpoint {key} differs from the requested configuration")
    if args.command == "bc":
        records = teacher_rows(args.teacher, args.phase, "train")
        if not records:
            raise ValueError("no training teacher records")
        collate = cached_public_collator([r["observation"] for r in records], tensor_cache=args.tensor_cache)
        result = train_bc(model, records, collate=collate, forward=forward_policy,
                          device=args.device, learning_rate=args.learning_rate, batch_size=args.batch_size,
                          epochs=max(1, math.ceil(args.updates * args.batch_size / len(records))),
                          max_steps=args.updates, max_seconds=max(0., args.seconds - (time.monotonic() - started)),
                          seed=args.seed, optimizer=optimizer, reseed=not args.resume,
                          metric_interval=args.metric_interval, bucket_actions=args.bucket_actions)
        optimizer = result.pop("optimizer")
        step += result["steps"]
        metrics["training"] = result
        development = teacher_rows(args.dev_teacher or args.teacher, args.phase, "dev")
        metrics["development"] = evaluate_bc(model, development, collate=collate_public,
                                              forward=forward_policy, device=args.device)
    else:
        tasks = tasks_for(args)
        ppo = PPOConfig(gamma=args.gamma, learning_rate=args.learning_rate,
                        minibatch_size=args.batch_size, epochs=args.ppo_epochs,
                        entropy_weight=args.entropy_coef, clip_ratio=args.clip_ratio)
        workers = [CurriculumWorker(config={"gamma": args.gamma, "reward_mode": args.reward_mode,
                                            "terminal_hp_weight": args.terminal_hp_weight,
                                            "progress_floor_weight": args.progress_floor_weight,
                                            "goal_act_clear": args.goal_act_clear,
                                            "fast_observations": args.fast_observations or args.fast_replay,
                                            "fast_replay": args.fast_replay}, timeout=args.timeout)
                   for _ in range(args.workers)]
        metrics["rounds"] = []
        try:
            for iteration in range(1 if args.command == "evaluate" else args.updates):
                remaining = args.seconds - (time.monotonic() - started)
                if remaining <= 0:
                    break
                random.Random(args.seed + iteration).shuffle(tasks)
                model.to(args.inference_device)
                result = collect_rollouts(model, workers, tasks, collate=collate_public,
                                           forward=forward_policy, config=ppo, device=args.inference_device,
                                           max_seconds=remaining if args.command == "evaluate" else min(remaining, 120.),
                                           deterministic=args.command == "evaluate",
                                           teacher_simulations=args.teacher_simulations,
                                           deferred_bootstrap=args.deferred_bootstrap)
                transitions, labels = result.pop("transitions"), result.pop("teacher_records")
                result["transitions"] = len(transitions)
                result["teacher_records"] = len(labels)
                if labels:
                    label_path = args.output.with_suffix(".dagger.jsonl")
                    with label_path.open("a") as output:
                        for record in labels:
                            output.write(json.dumps(record, sort_keys=True) + "\n")
                if args.command == "ppo":
                    update = update_ppo(model, transitions, collate=collate_public, forward=forward_policy,
                                         config=ppo, device=args.device, seed=args.seed + iteration,
                                         optimizer=optimizer,
                                         max_seconds=max(0., args.seconds - (time.monotonic() - started)))
                    optimizer = update.pop("optimizer", optimizer)
                    step += update["steps"]
                    result["update"] = update
                    save_checkpoint(args.output, model, model_config=model.metadata(), optimizer=optimizer, step=step,
                                    metadata={**lineage, "encoder_version": ENCODER_VERSION, "data_identity": identity,
                                              "reward_config": reward_config, "stage": args.command,
                                              "optimizer_carry": optimizer_carry,
                                              "config": metrics["config"]})
                metrics["rounds"].append(result)
                write_metrics(args.metrics or args.output.with_suffix(".metrics.json"), metrics)
        finally:
            for worker in workers:
                worker.close()
    if args.command == "bc":
        save_checkpoint(args.output, model, model_config=model.metadata(), optimizer=optimizer, step=step,
                        metadata={**lineage, "encoder_version": ENCODER_VERSION, "data_identity": identity,
                                  "reward_config": reward_config, "stage": args.command, "config": metrics["config"]})
    metrics["elapsed_seconds"], metrics["optimizer_step"] = time.monotonic() - started, step
    write_metrics(args.metrics or args.output.with_suffix(".metrics.json"), metrics)
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
