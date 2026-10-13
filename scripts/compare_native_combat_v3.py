"""Privileged combat competence controls with one frozen learned OOC policy."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import torch

from sts_policy.composition import training_environment_record
from sts_policy.curriculum import TaskSpec,world_split
from sts_policy.entity_model import load_policy
from sts_policy.strategic_collection import collect
from sts_policy.strategic_control import paired_retention
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json,summarize,worker_settings
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json,summarize,worker_settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--task-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--budgets",type=int,nargs="+",default=[200,1000])
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seconds",type=float,default=300)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("use fresh output")
    torch.set_num_threads(1)
    tasks=[TaskSpec.from_dict(t) for t in json.loads(args.task_manifest.read_text())["tasks"]]
    if any(world_split(t.seed)!="train" for t in tasks):
        raise ValueError("teacher competence comparisons are TRAIN-only")
    model,_=load_policy(args.checkpoint)
    args.output.mkdir(parents=True)
    atomic_json(args.output/"launch.json",{"checkpoint":args.checkpoint,"checkpoint_sha256":file_identity(args.checkpoint),
        "task_manifest_sha256":file_identity(args.task_manifest),"tasks":[t.to_dict() for t in tasks],
        "environment":training_environment_record(),"budgets":args.budgets,"worker_settings":worker_settings(),
        "note":"Privileged native combat controls; OOC parameters frozen, subsequent OOC choices adapt to reached public states."})
    with ExitStack() as stack:
        workers=[stack.enter_context(CurriculumWorker(worker_settings(),timeout=10.)) for _ in range(args.workers)]
        baseline=collect(model,workers,tasks,deterministic=True,retain_transitions=False,max_seconds=args.seconds)
        baseline.pop("transitions"); baseline["summary"]=summarize(baseline)
        atomic_json(args.output/"learned_baseline.json",baseline)
        for budget in args.budgets:
            result=collect(model,workers,tasks,deterministic=True,retain_transitions=False,
                           max_seconds=args.seconds,native_combat_simulations=budget)
            result.pop("transitions"); result["summary"]=summarize(result)
            result["paired"]=paired_retention(baseline,result)
            atomic_json(args.output/f"native_{budget}.json",result)
            print(json.dumps({"budget":budget,"summary":result["summary"],"paired":result["paired"]}),flush=True)


if __name__=="__main__":
    main()
