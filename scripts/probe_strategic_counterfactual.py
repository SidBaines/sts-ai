"""Bounded TRAIN-only complete-menu OOC continuation diagnostics; no fitting."""
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
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.entity_model import load_policy
from sts_policy.frontier_sampling import SAMPLER_VERSION, sample_frontiers
from sts_policy.strategic_counterfactual import COUNTERFACTUAL_VERSION, evaluate_menu
from sts_policy.source_snapshot import snapshot_experiment_sources
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, worker_settings
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, worker_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-act", type=int, default=2)
    parser.add_argument("--acts",type=int,nargs="+",default=[2,3])
    parser.add_argument("--selection",choices=("first","world-balanced"),default="first")
    parser.add_argument("--selection-seed",type=int,default=0)
    parser.add_argument("--decision-family",help="Filter source menu stages, e.g. card_only_rewards")
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--panel-seconds", type=float, default=30.)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--native-combat-simulations", type=int,
        help="Explicitly privileged diagnostic continuation; never a public-agent benchmark")
    parser.add_argument("--native-plan-cache",action="store_true",
        help="Separate privileged winning-plan continuation; requires native combat")
    args = parser.parse_args()
    if args.per_act < 1 or any(not math.isfinite(value) or value<=0 for value in (args.seconds,args.panel_seconds)):
        parser.error("positive source counts and time bounds required")
    if len(set(args.acts))!=len(args.acts) or any(act not in (1,2,3) for act in args.acts):
        parser.error("distinct acts 1,2,3 required")
    if args.native_plan_cache and args.native_combat_simulations is None:
        parser.error("plan cache requires explicitly privileged native combat")
    if args.output.exists():
        parser.error("output directory must be new; preserve existing diagnostics")
    with args.starts.open() as stream:
        definitions = [json.loads(line) for line in stream]
    if any(world_split(int(row["seed"])) != "train" for row in definitions):
        parser.error("source bank must be entirely TRAIN")
    selected = []
    for act in args.acts:
        worlds = set()
        for row in definitions:
            if (row.get("start_act") != act or row["seed"] in worlds
                    or (args.decision_family is not None and row.get("decision_family")!=args.decision_family)):
                continue
            selected.append(row)
            worlds.add(row["seed"])
            if len(worlds) == args.per_act:
                break
    if args.selection=="world-balanced":
        selected=sample_frontiers(definitions,acts=args.acts,per_act=args.per_act,
                                  seed=args.selection_seed,decision_family=args.decision_family)
    if not selected:
        parser.error("no eligible strategic choices remain")
    torch.set_num_threads(1)
    model, payload = load_policy(str(args.checkpoint))
    args.output.mkdir(parents=True)
    snapshot = snapshot_experiment_sources(args.output/"executed_sources")
    settings={**worker_settings(),"native_plan_cache":args.native_plan_cache}
    manifest = {"version": COUNTERFACTUAL_VERSION, "status": "running", "config": {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_sha256": file_identity(args.checkpoint), "checkpoint_metadata": payload["metadata"],
        "source_snapshot":snapshot,
        "starts_sha256": file_identity(args.starts), "environment": training_environment_record(),
        "worker_settings": settings, "source_tasks": selected,
        "source_sampler":SAMPLER_VERSION if args.selection=="world-balanced" else "historical first per world",
        "sources": {name: file_identity(Path(name)) for name in
            ("scripts/probe_strategic_counterfactual.py", "src/sts_policy/strategic_counterfactual.py",
             "src/sts_policy/strategic_collection.py", "src/sts_policy/workers.py")},
        "started_utc": datetime.now(timezone.utc).isoformat(), "panels": [], "faults": []}
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    with ExitStack() as stack:
        worker = stack.enter_context(CurriculumWorker(settings, timeout=10.))
        for number, row in enumerate(selected):
            remaining = args.seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            task = TaskSpec.from_dict(row)
            try:
                panel = evaluate_menu(model, worker, task, max_seconds=min(args.panel_seconds, remaining),
                    sampling_seeds=args.sampling_seeds, deterministic=not args.stochastic,
                    native_combat_simulations=args.native_combat_simulations)
            except SimulatorError as error:
                manifest["faults"].append({"seed": task.seed, "start_act": row["start_act"],
                    "type": type(error).__name__, "message": str(error)})
                atomic_json(args.output / "manifest.json", manifest)
                continue
            filename = f"panel_{number:03d}.json"
            atomic_json(args.output / filename, panel)
            manifest["panels"].append({"path": filename, "sha256": file_identity(args.output / filename),
                "seed": task.seed, "start_act": row["start_act"], "source_screen": row["start_screen"],
                "usable_complete_menu": panel["usable_complete_menu"], "elapsed_seconds": panel["elapsed_seconds"]})
            atomic_json(args.output / "manifest.json", manifest)
            print(json.dumps(manifest["panels"][-1]), flush=True)
    manifest.update(status="saved", elapsed_seconds=time.monotonic() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        unstarted=len(selected) - len(manifest["panels"]) - len(manifest["faults"]))
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "completed_panels": len(manifest["panels"]),
        "usable_panels": sum(panel["usable_complete_menu"] for panel in manifest["panels"]),
        "faults": len(manifest["faults"]), "unstarted": manifest["unstarted"],
        "elapsed_seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
