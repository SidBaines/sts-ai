"""Initialize, audit and evaluate the repaired strategic learner explicitly."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from sts_policy.composition import training_environment_record
from sts_policy.curriculum import world_split
from sts_policy import entities
from sts_policy.entity_model import initialize_from_legacy, save_policy, load_policy
from sts_policy.strategic_collection import collect
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.evaluate_nonllm_baselines import matched_tasks
except ModuleNotFoundError:
    from evaluate_nonllm_baselines import matched_tasks


def atomic_json(path: Path, content: dict) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w") as output:
        output.write(json.dumps(content, sort_keys=True, indent=2, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0., 1.]
    z, p = 1.959963984540054, successes / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0., center-radius), min(1., center+radius)]


def summarize(result: dict) -> dict:
    valid = [e for e in result["episodes"] if e["terminated"] and not e["truncated"]]
    wins = sum(e["task_outcome"] == "victory" for e in valid)
    fresh = [e for e in valid if e["start_family"] == "fresh"]
    continuations = [e for e in valid if e["start_family"] != "fresh"]
    fresh_wins = sum(e["task_outcome"] == "victory" for e in fresh)
    unique_fresh = len({e["world_seed"] for e in fresh}) == len(fresh)
    return {"summary_version": "strategic_summary_v2", "requested": result["requested_tasks"], "finished": len(result["episodes"]),
        "valid_completed": len(valid), "faults": len(result["faults"]),
        "truncated": sum(e["truncated"] for e in result["episodes"]), "unstarted": result["unstarted_tasks"],
        "wins": wins, "win_rate": wins / max(1, len(valid)),
        "win_wilson_95": wilson(wins, len(valid)) if not continuations and unique_fresh else None,
        "pooled_rate_note": "Mixed/repeated continuation starts are descriptive counts; no independent-binomial interval.",
        "wins_with_actor_choices": sum(e["task_outcome"] == "victory" and e["actor_choice_rows"] > 0 for e in valid),
        "wins_without_actor_choices": sum(e["task_outcome"] == "victory" and e["actor_choice_rows"] == 0 for e in valid),
        "fresh_completed": len(fresh), "fresh_wins": fresh_wins,
        "fresh_win_wilson_95": wilson(fresh_wins, len(fresh)) if unique_fresh else None,
        "fresh_mean_floor": float(np.mean([e["floor"] for e in fresh])) if fresh else None,
        "continuation_completed": len(continuations), "continuation_wins": sum(e["task_outcome"] == "victory" for e in continuations),
        "continuation_mean_floor": float(np.mean([e["floor"] for e in continuations])) if continuations else None,
        "act1_clears_fresh": sum(e["max_act_reached"] >= 2 for e in fresh),
        "act2_clears_fresh": sum(e["max_act_reached"] >= 3 for e in fresh),
        "mean_floor": float(np.mean([e["floor"] for e in valid])) if valid else None,
        "primitive_steps": result["primitive_steps"], "elapsed_seconds": result["elapsed_seconds"],
        "primitive_steps_per_second": result["primitive_steps_per_second"]}


def worker_settings() -> dict:
    return {"fast_observations": True, "fast_replay": True, "reward_mode": "progress",
            "gamma": 1., "progress_floor_weight": .05, "terminal_hp_weight": 0., "goal_act_clear": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--legacy", required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--seed", type=int, default=0)
    audit = sub.add_parser("audit")
    audit.add_argument("--teacher", type=Path, required=True)
    audit.add_argument("--rows", type=int, default=4000)
    audit.add_argument("--output", type=Path, required=True)
    for command in ("evaluate", "equivalence"):
        p = sub.add_parser(command)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--episodes", type=int, default=192 if command == "evaluate" else 8)
        p.add_argument("--world-start", type=int, default=330000000 if command == "evaluate" else 310000000)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seconds", type=float, default=180)
        p.add_argument("--split", choices=("train", "dev"), default="dev" if command == "evaluate" else "train")
        p.add_argument("--sampling-seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "initialize":
        torch.manual_seed(args.seed)
        model, transfer = initialize_from_legacy(args.legacy)
        save_policy(str(args.output), model, metadata={"stage": "common_initialization", "seed": args.seed,
            "training_environment": training_environment_record(), "transfer": transfer})
        print(json.dumps({"checkpoint": str(args.output), "sha256": file_identity(args.output),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "actor_trainable": sum(p.numel() for p in model.actor_parameters()),
                          "critic_trainable": sum(p.numel() for p in model.critic_parameters())}))
        return
    if args.command == "audit":
        started = time.monotonic()
        counters, phases, unknown = Counter(), Counter(), Counter()
        encoded_bytes, rows = 0, 0
        for line in args.teacher.open():
            record = json.loads(line)
            if world_split(int(record["world_seed"])) != "train":
                raise ValueError("encoding audit is TRAIN-only")
            obs = record["observation"]
            encoded = entities.encode_observation(obs)
            counters.update(encoded.coverage)
            phases[obs["phase"]] += 1
            rows += 1
            encoded_bytes += sum(value.nbytes for value in vars(encoded).values() if isinstance(value, np.ndarray))
            # Names of unknown public entities, without serializing arbitrary keys.
            for relic in obs["state"]["inventory"].get("relics", []):
                if entities.token("relic", str(relic["name"])) == 1:
                    unknown["relic:" + str(relic["name"])] += 1
            for enemy in obs["state"].get("combat", {}).get("enemies", []):
                for name in enemy.get("powers", {}):
                    if entities.canonical(entities._TIMER.sub("", name)) not in entities.STATUS_IDS:
                        unknown["status:" + name] += 1
            if rows >= args.rows:
                break
        result = {"rows": rows, "phase_counts": dict(phases), "coverage": dict(counters),
                  "unknown_public_names": dict(unknown), "encoded_bytes": encoded_bytes,
                  "elapsed_seconds": time.monotonic() - started, "encoder_version": entities.ENCODER_VERSION,
                  "vocabulary_sha256": entities.VOCABULARY_SHA256, "teacher_sha256": file_identity(args.teacher)}
        atomic_json(args.output, result)
        print(json.dumps(result))
        return
    model, payload = load_policy(args.checkpoint)
    tasks = matched_tasks(bank=None, kind="run", split=args.split, episodes=args.episodes,
        seed=0, world_start=args.world_start, horizon=4096, simulations=50, max_act=3)
    definitions = [task.to_dict() for task in tasks]
    with ExitStack() as stack:
        workers = [stack.enter_context(CurriculumWorker(worker_settings(), timeout=10.)) for _ in range(args.workers)]
        results = []
        for unit in (["micro", "macro"] if args.command == "equivalence" else ["macro"]):
            result = collect(model, workers, tasks, credit_unit=unit, sampling_seed=args.sampling_seed,
                             deterministic=args.command == "evaluate", max_seconds=args.seconds,
                             retain_transitions=args.command == "equivalence")
            result.pop("transitions")
            result["summary"] = summarize(result)
            results.append(result)
    output = {"checkpoint": args.checkpoint, "checkpoint_sha256": file_identity(args.checkpoint),
              "policy_metadata": payload["metadata"], "environment": training_environment_record(),
              "tasks": definitions, "worker_settings": worker_settings(), "results": results}
    if args.command == "equivalence":
        first, second = [{e["task_id"]: e for e in result["episodes"]} for result in results]
        common = set(first) & set(second)
        output["equivalence"] = {"common": len(common), "matched_traces": sum(first[k]["action_trace_sha256"] == second[k]["action_trace_sha256"] for k in common),
            "matched_rewards": sum(abs(first[k]["reward"]-second[k]["reward"]) < 1e-8 for k in common),
            "completed_both": all(r["cohort_completed"] and not r["faults"] for r in results)}
        if output["equivalence"]["matched_traces"] != len(tasks) or not output["equivalence"]["completed_both"]:
            atomic_json(args.output, output)
            raise RuntimeError("native micro/macro trajectory equivalence failed; evidence preserved")
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "summaries": [r["summary"] for r in results],
                      "equivalence": output.get("equivalence")}))


if __name__ == "__main__":
    main()
