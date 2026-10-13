"""Matched fresh-game evaluation of a learned public combat residual."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import torch

from sts_policy.combat_residual import initialize, load_policy, save_policy
from sts_policy.composition import training_environment_record
from sts_policy.strategic_collection import collect
from sts_policy.strategic_control import paired_retention
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, summarize, worker_settings, matched_tasks
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, summarize, worker_settings, matched_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--initialize-base")
    parser.add_argument("--initial-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--world-start", type=int, default=330000000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.initialize_base:
        if args.checkpoint or args.initial_output is None:
            parser.error("initialization needs --initial-output and no checkpoint")
        model, transfer = initialize(args.initialize_base)
        save_policy(str(args.initial_output), model, metadata={"stage":"zero_combat_initialization", "transfer":transfer})
        checkpoint = args.initial_output
    elif args.checkpoint:
        model, _ = load_policy(args.checkpoint)
        checkpoint = Path(args.checkpoint)
    else:
        parser.error("supply checkpoint or initialize-base")
    tasks = matched_tasks(bank=None,kind="run",split="dev",episodes=args.episodes,
                         world_start=args.world_start,seed=0,horizon=4096,max_act=3)
    with ExitStack() as stack:
        workers = [stack.enter_context(CurriculumWorker(worker_settings(),timeout=10.)) for _ in range(args.workers)]
        result = collect(model,workers,tasks,credit_unit="macro",deterministic=True,max_seconds=args.seconds,
                         retain_transitions=False,frozen_combat=False)
    result.pop("transitions")
    result["summary"] = summarize(result)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())["results"][0]
        result["paired"] = paired_retention(baseline,result)
    atomic_json(args.output,{"checkpoint":str(checkpoint),"checkpoint_sha256":file_identity(checkpoint),
                            "environment":training_environment_record(),"tasks":[t.to_dict() for t in tasks],
                            "result":result})
    print(json.dumps({"output":str(args.output),"summary":result["summary"],"paired":result.get("paired")}),flush=True)


if __name__ == "__main__":
    main()
