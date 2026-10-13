"""Finish a bounded replay window into a new bank without replacing its parent."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec, public_hash
from sts_policy.data import atomic_jsonl
from sts_policy.early_frontiers import coverage
from sts_policy.source_snapshot import snapshot_experiment_sources
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, worker_settings
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, worker_settings


def read_parent(bank: Path):
    manifest=json.loads((bank/"manifest.json").read_text())
    if file_identity(bank/"starts.jsonl")!=manifest["starts_sha256"]:
        raise ValueError("parent accepted starts differ from their manifest")
    failures=json.loads((bank/"replay_failures.json").read_text()).get("failures",[]) if (bank/"replay_failures.json").exists() else manifest.get("replay_failures",[])
    blocked={row["seed"] for row in failures}
    with (bank/"starts.jsonl").open() as stream:
        previous=[json.loads(line) for line in stream]
    with (bank/"unverified_starts.jsonl").open() as stream:
        pending=[json.loads(line) for line in stream]
    return ([row for row in previous if row["seed"] not in blocked],
            [row for row in pending if row["seed"] not in blocked],blocked)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=30.)
    args=parser.parse_args()
    if args.output.exists() or not math.isfinite(args.seconds) or args.seconds<=0:
        parser.error("fresh output and positive time bound required")
    previous,pending,parent_blocked=read_parent(args.bank)
    args.output.mkdir(parents=True)
    snapshot=snapshot_experiment_sources(args.output/"executed_sources")
    accepted,unverified,failures=list(previous),[],[]
    deadline=time.monotonic()+args.seconds
    with CurriculumWorker(worker_settings(),timeout=10.) as worker:
        for row in pending:
            if time.monotonic()>=deadline:
                unverified.append(row);continue
            try:
                observation,info=worker.reset(TaskSpec.from_dict(row))
                if public_hash(observation)!=row["expected_public_hash"] or observation!=row["source_observation"]:
                    raise SimulatorError("resumed frontier replay changed public observation")
                accepted.append(row)
            except SimulatorError as error:
                failures.append({"seed":row["seed"],"start_act":row["start_act"],"error":str(error)})
    failed_worlds={row["seed"] for row in failures}
    accepted=[row for row in accepted if row["seed"] not in failed_worlds]
    atomic_jsonl(args.output/"starts.jsonl",accepted)
    atomic_jsonl(args.output/"unverified_starts.jsonl",unverified)
    manifest={"parent_bank":str(args.bank),"parent_manifest_sha256":file_identity(args.bank/"manifest.json"),
        "parent_starts_sha256":file_identity(args.bank/"starts.jsonl"),"source_snapshot":snapshot,
        "prior_verified":len(previous),"pending_requested":len(pending),"unverified":len(unverified),
        "parent_quarantined_worlds":sorted(parent_blocked),
        "replay_failures":failures,"accepted_coverage":coverage(accepted),
        "starts_sha256":file_identity(args.output/"starts.jsonl"),"status":"saved"}
    atomic_json(args.output/"manifest.json",manifest)
    print(json.dumps({key:value for key,value in manifest.items() if key!="source_snapshot"}))


if __name__ == "__main__":
    main()
