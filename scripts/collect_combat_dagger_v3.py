"""Bounded TRAIN-only search labels on unchanged learned trajectories."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path

import torch

from sts_policy.combat_dagger import LabelingWorker, DAGGER_VERSION
from sts_policy.combat_residual import load_policy as load_combat
from sts_policy.entity_model import load_policy as load_base
from sts_policy.composition import training_environment_record
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.data import atomic_jsonl
from sts_policy.strategic_collection import collect
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
from sts_policy.teacher_panel import PanelWorker, PANEL_VERSION
try:
    from scripts.run_strategic_v3 import atomic_json, matched_tasks, summarize, worker_settings
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, matched_tasks, summarize, worker_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--combat-residual",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=32)
    parser.add_argument("--world-start",type=int,default=340000000)
    parser.add_argument("--starts",type=Path)
    parser.add_argument("--act2-starts",type=int,default=24)
    parser.add_argument("--act3-starts",type=int,default=12)
    parser.add_argument("--budgets",type=int,nargs="+",default=[200,1000])
    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--teacher-panel",action="store_true",help="Compare search RNG and cloned hidden draw orders")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seconds",type=float,default=600)
    args = parser.parse_args()
    if args.output.exists() or min(args.episodes,args.stride,args.workers,args.seconds) <= 0:
        parser.error("fresh output and positive bounds required")
    torch.set_num_threads(1)
    model,_ = (load_combat if args.combat_residual else load_base)(args.checkpoint)
    tasks = matched_tasks(bank=None,kind="run",split="train",episodes=args.episodes,
                         world_start=args.world_start,seed=0,horizon=4096,max_act=3)
    if args.starts:
        counts = {2:0,3:0}; limits = {2:args.act2_starts,3:args.act3_starts}
        for line in args.starts.open():
            row = json.loads(line); act = int(row["start_act"])
            if counts[act] < limits[act]:
                task = replace(TaskSpec.from_dict(row),kind="run",max_act=3,horizon=4096)
                if world_split(task.seed) != "train":
                    raise ValueError("continuation supervision requires TRAIN worlds")
                tasks.append(task); counts[act] += 1
    args.output.mkdir(parents=True)
    environment = training_environment_record()
    metadata = {"data_version":DAGGER_VERSION,"checkpoint":args.checkpoint,
        "checkpoint_sha256":file_identity(args.checkpoint),"environment":environment,
        "source_identity":{str(p):file_identity(p) for p in
            (Path(__file__),Path("src/sts_policy/combat_dagger.py"),Path("src/sts_policy/strategic_collection.py"),
             Path("src/sts_policy/teacher_panel.py"))},
        "budgets":args.budgets,"stride":args.stride,"teacher_panel":PANEL_VERSION if args.teacher_panel else None,
        "tasks":[t.to_dict() for t in tasks],
        "worker_settings":worker_settings(),"note":"Privileged labels only; executed actions remain public learned policy."}
    atomic_json(args.output/"launch.json",metadata)
    with ExitStack() as stack:
        worker_class = PanelWorker if args.teacher_panel else CurriculumWorker
        raw = [stack.enter_context(worker_class(worker_settings(),timeout=10.)) for _ in range(args.workers)]
        reference = collect(model,raw,tasks,deterministic=True,retain_transitions=False,
            frozen_combat=not args.combat_residual,max_seconds=args.seconds)
        reference.pop("transitions")
        atomic_json(args.output/"no_query.json",reference)
        def sink(row):
            atomic_json(args.output/"worlds"/(row["task_id"]+".json"),row)
            print(json.dumps({"world":row["world_seed"],"status":row["status"],"labels":len(row["records"])}),flush=True)
        wrappers = [stack.enter_context(LabelingWorker(w,tuple(args.budgets),stride=args.stride,sink=sink,
                                                       panel=args.teacher_panel)) for w in raw]
        queried = collect(model,wrappers,tasks,deterministic=True,retain_transitions=False,
            frozen_combat=not args.combat_residual,max_seconds=args.seconds)
        queried.pop("transitions")
    first = {e["task_id"]:e for e in reference["episodes"]}
    second = {e["task_id"]:e for e in queried["episodes"]}
    common = set(first)&set(second)
    purity = {"common_tasks":len(common),"expected_tasks":len(tasks),
        "matched_traces":sum(first[k]["action_trace_sha256"]==second[k]["action_trace_sha256"] for k in common),
        "matched_rewards":sum(first[k]["reward"]==second[k]["reward"] for k in common),
        "cohorts_complete":reference["cohort_completed"] and queried["cohort_completed"],
        "faults":len(reference["faults"])+len(queried["faults"])}
    journals = [json.loads(p.read_text()) for p in sorted((args.output/"worlds").glob("*.json"))]
    valid = [r for r in journals if r["status"] == "completed"]
    metadata.update(purity=purity,reference_summary=summarize(reference),queried_summary=summarize(queried),
        valid_tasks=len(valid),labels=sum(len(r["records"]) for r in valid),
        query_counts={str(b):sum(r["query_counts"].get(str(b),0) for r in journals) for b in args.budgets},
        query_seconds={str(b):sum(r["query_seconds"].get(str(b),0.) for r in journals) for b in args.budgets})
    atomic_json(args.output/"queried.json",queried)
    atomic_json(args.output/"manifest.json",metadata)
    if not purity["cohorts_complete"] or purity["faults"] or purity["matched_traces"] != len(tasks):
        raise RuntimeError("Query-purity gate failed; world evidence preserved, consolidated labels withheld")
    atomic_jsonl(args.output/"teacher.jsonl",(r for w in valid for r in w["records"]))
    metadata["teacher_sha256"] = file_identity(args.output/"teacher.jsonl")
    atomic_json(args.output/"manifest.json",metadata)
    print(json.dumps({"done":True,"labels":metadata["labels"],"purity":purity}),flush=True)


if __name__ == "__main__":
    main()
