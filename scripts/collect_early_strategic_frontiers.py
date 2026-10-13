"""Collect and replay-check earlier TRAIN choices with frozen learned OOC play."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from sts_env.env import SimulatorError
from sts_policy.composition import training_environment_record
from sts_policy.curriculum import TaskSpec, public_hash
from sts_policy.data import atomic_jsonl
from sts_policy.early_frontiers import FRONTIER_VERSION, EarlyFrontierWorker, completed_frontiers, coverage
from sts_policy.entity_model import load_policy
from sts_policy.strategic_collection import collect
from sts_policy.source_snapshot import snapshot_experiment_sources
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, matched_tasks, summarize, worker_settings
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, matched_tasks, summarize, worker_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--world-start", type=int, default=430000000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--collect-seconds", type=float, default=60.)
    parser.add_argument("--replay-seconds", type=float, default=30.)
    parser.add_argument("--native-combat-simulations", type=int, default=1000)
    parser.add_argument("--legacy-native-scoring", action="store_true",
        help="Timing control: compute and discard the frozen combat scorer before native search")
    parser.add_argument("--native-plan-cache",action="store_true",
        help="Separate privileged controller: reuse guarded winning plans instead of per-action replanning")
    args = parser.parse_args()
    if args.output.exists() or min(args.episodes, args.workers, args.native_combat_simulations) < 1:
        parser.error("fresh output and positive counts required")
    if any(not math.isfinite(value) or value <= 0 for value in (args.collect_seconds, args.replay_seconds)):
        parser.error("finite positive time bounds required")
    torch.set_num_threads(1)
    model, payload = load_policy(str(args.checkpoint))
    tasks = matched_tasks(bank=None, kind="run", split="train", episodes=args.episodes,
        world_start=args.world_start, seed=0, horizon=4096, max_act=3)
    args.output.mkdir(parents=True)
    snapshot = snapshot_experiment_sources(args.output/"executed_sources")
    settings = {**worker_settings(),"native_plan_cache":args.native_plan_cache}
    manifest = {"frontier_version":FRONTIER_VERSION, "status":"collecting", "config":{
        key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        "checkpoint_sha256":file_identity(args.checkpoint), "checkpoint_metadata":payload["metadata"],
        "source_snapshot":snapshot,
        "environment":training_environment_record(), "tasks":[task.to_dict() for task in tasks],
        "worker_settings":settings, "sources":{name:file_identity(Path(name)) for name in
            ("scripts/collect_early_strategic_frontiers.py", "src/sts_policy/early_frontiers.py",
             "src/sts_policy/strategic_collection.py", "src/sts_policy/workers.py", "src/sts_policy/native_plan.py")},
        "source_policy":"frozen original public OOC / privileged winning-plan reuse" if args.native_plan_cache
            else "frozen original public OOC / privileged per-action native replanning",
        "started_utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(args.output/"manifest.json", manifest)
    with ExitStack() as stack:
        workers = [stack.enter_context(EarlyFrontierWorker(settings, timeout=10.)) for _ in range(args.workers)]
        result = collect(model, workers, tasks, deterministic=True, retain_transitions=False,
            native_combat_simulations=args.native_combat_simulations, max_seconds=args.collect_seconds,
            legacy_native_scoring=args.legacy_native_scoring)
        result.pop("transitions")
        records = [row for worker in workers for row in worker.frontiers]
    usable, quarantine = completed_frontiers(records, result)
    # Prioritize independent Act 3 coverage if the bounded replay window ends.
    usable.sort(key=lambda row:(-row["start_act"],row["seed"],row["frontier_step"]))
    atomic_json(args.output/"source_collection.json",result)
    atomic_jsonl(args.output/"candidates.jsonl",usable)
    atomic_jsonl(args.output/"quarantined_starts.jsonl",quarantine)
    manifest.update(status="replaying", source_summary=summarize(result), candidate_coverage=coverage(usable),
                    quarantined_source_frontiers=len(quarantine), replay_priority="Act 3, then seed and step")
    atomic_json(args.output/"manifest.json",manifest)
    print(json.dumps({"source_summary":manifest["source_summary"],"candidate_coverage":manifest["candidate_coverage"]}),flush=True)
    verified, failures, unverified = [], [], []
    deadline = time.monotonic()+args.replay_seconds
    with CurriculumWorker(worker_settings(),timeout=10.) as worker:
        for row in usable:
            if time.monotonic() >= deadline:
                unverified.append(row)
                continue
            try:
                observation, info = worker.reset(TaskSpec.from_dict(row))
                if observation["phase"] != "out_of_combat" or len(observation["legal_actions"]) < 2:
                    raise SimulatorError("frontier replay lost its strategic choice")
                if public_hash(observation) != row["expected_public_hash"] or observation != row["source_observation"]:
                    raise SimulatorError("frontier replay changed public observation")
                verified.append(row)
            except SimulatorError as error:
                failures.append({"seed":row["seed"],"start_act":row["start_act"],
                    "frontier_step":row["frontier_step"],"type":type(error).__name__,"message":str(error)})
    failed_worlds = {row["seed"] for row in failures}
    accepted = [row for row in verified if row["seed"] not in failed_worlds]
    atomic_jsonl(args.output/"starts.jsonl",accepted)
    atomic_jsonl(args.output/"unverified_starts.jsonl",unverified)
    atomic_json(args.output/"replay_failures.json",{"failures":failures})
    manifest.update(status="saved", accepted_coverage=coverage(accepted), verified_candidates=len(verified),
        replay_failed_worlds=len(failed_worlds), unverified_candidates=len(unverified),
        starts_sha256=file_identity(args.output/"starts.jsonl"),
        finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output/"manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"accepted_coverage":manifest["accepted_coverage"],
        "replay_failed_worlds":len(failed_worlds),"unverified_candidates":len(unverified)}),flush=True)


if __name__ == "__main__":
    main()
