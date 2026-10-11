#!/usr/bin/env python3
"""Bounded, matched public-random/heuristic and privileged-native baselines."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.encoding import encode_observation, ENCODER_VERSION
from sts_policy.workers import CurriculumWorker
try:
    from scripts.report_nonllm import summarize_live
except ModuleNotFoundError:
    from report_nonllm import summarize_live


def task_identity(task: TaskSpec) -> str:
    return hashlib.sha256(json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matched_tasks(*, bank: Path | None, kind: str, split: str, episodes: int,
                  seed: int = 0, world_start: int = 1000, horizon: int = 256,
                  simulations: int = 50, max_act: int | None = None) -> list[TaskSpec]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if bank is not None:
        rows = [json.loads(line) for line in bank.read_text().splitlines() if line.strip()]
        tasks = [TaskSpec.from_dict(row) for row in rows]
        tasks = [task for task in tasks if world_split(task.seed) == split]
        if kind == "battle":
            tasks = [replace(task, kind=kind, horizon=horizon, battle_simulations=simulations,
                             max_act=task.max_act if max_act is None else max_act) for task in tasks]
        else:
            unique_worlds = {task.seed: task for task in tasks}
            tasks = [TaskSpec(seed=world, kind=kind, horizon=horizon,
                              max_act=task.max_act if max_act is None else max_act,
                              battle_simulations=simulations)
                     for world, task in unique_worlds.items()]
    else:
        tasks = []
        world = world_start
        while len(tasks) < episodes:
            if world_split(world) == split:
                tasks.append(TaskSpec(seed=world, kind=kind, horizon=horizon,
                                      max_act=max_act or 1, battle_simulations=simulations))
            world += 1
    random.Random(seed).shuffle(tasks)
    if not tasks:
        raise ValueError("no tasks remain after world split filtering")
    return tasks[:episodes]


def heuristic_action(observation: dict[str, Any]) -> int:
    """Deliberately simple public tactical/reference policy, no hidden queries."""
    encoded = encode_observation(observation)
    menu = observation["legal_actions"]
    if not menu:
        raise ValueError("cannot choose from a terminal menu")
    scores = []
    combat = observation.get("phase") == "combat"
    hp_fraction = float(encoded.state_features[2])
    needed_block = max(0., float(encoded.state_features[14] - encoded.state_features[11]) * 30)
    for action, features in zip(menu, encoded.action_features):
        description = str(action["description"]).lower()
        if combat:
            damage, block = float(features[33]) * 30, float(features[34]) * 30
            score = damage + 1.1 * min(block, needed_block) + float(features[53]) * 25
            score += float(features[46]) * 5 * 2 + float(features[47]) * 3 * 4
            score -= float(features[32]) * .3
            if "end turn" in description:
                score = -20
            elif description.startswith("play"):
                score += 1
            if description.startswith("drink"):
                score -= 10 if hp_fraction > .3 else 0
        else:
            score = 0.
            if description.startswith(("take gold", "take relic", "take boss relic", "take potion", "open chest")):
                score = 10
            elif description.startswith("skip card"):
                score = 1
            elif description.startswith("take card"):
                score = 3 if not any(name in description for name in ("strike", "defend", "clash", "curse", "status")) else 0
            elif description.startswith("rest"):
                score = 8 if hp_fraction < .55 else -2
            elif description.startswith("smith"):
                score = 4
            elif description.startswith("choose map"):
                score = (3 * float(features[59]) if hp_fraction < .6 else 0) - 3 * float(features[58])
            elif description.startswith("select card"):
                purpose = str(observation.get("text", "")).lower()
                if "choose a card to remove" in purpose:
                    score = 8 if "curse" in description else 4 if "strike" in description else 0
                elif "choose a card to upgrade" in purpose:
                    score = 0 if any(name in description for name in ("strike", "defend")) else 3
            elif description.startswith("buy card") and "remove" not in description:
                score = 2 if float(features[35]) * 300 < 70 else -1
        scores.append(score)
    position = max(range(len(menu)), key=lambda i: scores[i])
    return int(menu[position]["index"])


def evaluate_task(worker: CurriculumWorker, task: TaskSpec, baseline: str, *,
                  policy_seed: int, deadline: float, simulations: int) -> tuple[str, dict[str, Any]]:
    identity = task_identity(task)
    rng = random.Random(f"{policy_seed}:{identity}")
    reward_sum, steps = 0., 0
    try:
        observation, info = worker.reset(task)
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        while not (terminated or truncated) and observation.get("legal_actions"):
            if time.monotonic() >= deadline:
                truncated = True
                info = {**info, "termination_reason": "collector_budget"}
                break
            if baseline == "native":
                action = worker.teacher_action(simulations)
            elif baseline == "random":
                action = int(rng.choice(observation["legal_actions"])["index"])
            elif baseline == "heuristic":
                action = heuristic_action(observation)
            else:
                raise ValueError("unknown baseline")
            observation, reward, terminated, truncated, info = worker.step(action)
            reward_sum += float(reward)
            steps += 1
        state = observation.get("state", {})
        return "episode", {"world_seed": task.seed, "task_kind": task.kind, "task_id": identity,
                           "baseline": baseline, "reward": reward_sum, "steps": steps,
                           "termination_reason": info.get("termination_reason", ""),
                           "outcome": info.get("task_outcome") or info.get("outcome", ""),
                           "task_outcome": info.get("task_outcome", ""), "hp_cost": info.get("hp_cost"),
                           "floor": state.get("floor"), "act": state.get("act"),
                           "split": world_split(task.seed), "native_steps": info.get("native_steps"),
                           "fault": False, "terminated": bool(terminated), "truncated": bool(truncated)}
    except SimulatorError as error:
        return "fault", {"world_seed": task.seed, "task_kind": task.kind, "task_id": identity,
                         "baseline": baseline, "error": type(error).__name__, "message": str(error),
                         "excluded_transitions": steps, "split": world_split(task.seed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--kind", choices=("battle", "ooc", "run"), default="battle")
    parser.add_argument("--split", choices=("train", "dev", "final"), default="dev")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--baseline", choices=("random", "native", "heuristic"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-start", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--max-act", type=int)
    parser.add_argument("--timeout", type=float, default=10.)
    parser.add_argument("--fast-observations", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.workers, args.seconds, args.timeout) <= 0:
        parser.error("episode/worker counts and time bounds must be positive")
    tasks = matched_tasks(bank=args.tasks, kind=args.kind, split=args.split,
                          episodes=args.episodes, seed=args.seed, world_start=args.world_start,
                          horizon=args.horizon, simulations=args.simulations, max_act=args.max_act)
    started = time.monotonic()
    deadline = started + args.seconds
    chunks = [list(enumerate(tasks))[lane::args.workers] for lane in range(args.workers)]
    results = []
    def lane(chunk):
        output = []
        with CurriculumWorker({"fast_observations": args.fast_observations}, timeout=args.timeout) as worker:
            for index, task in chunk:
                if time.monotonic() >= deadline:
                    break
                tag, row = evaluate_task(worker, task, args.baseline, policy_seed=args.seed,
                                         deadline=deadline, simulations=args.simulations)
                output.append((index, tag, row))
        return output
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(lane, chunk) for chunk in chunks]):
            results.extend(future.result())
    results.sort(key=lambda row: row[0])
    episodes = [row for _, tag, row in results if tag == "episode"]
    faults = [row for _, tag, row in results if tag == "fault"]
    metrics = {"command": "baseline", "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "encoder_version": ENCODER_VERSION, "episodes": episodes, "faults": faults,
               "requested_tasks": len(tasks), "unstarted_tasks": len(tasks) - len(results),
               "task_ids": [task_identity(task) for task in tasks],
               "elapsed_seconds": time.monotonic() - started,
               "native_baseline_note": "privileged search for combat, scripted native OOC; not public-only or optimal"}
    metrics["summary"] = summarize_live(episodes, faults)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "faults": len(faults), "summary": metrics["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
