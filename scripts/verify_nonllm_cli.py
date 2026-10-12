#!/usr/bin/env python3
"""Verify a frozen composite through the public CLI on previously checked DEV worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from sts_env.cli import observation_hash, read_trace
from sts_policy.curriculum import world_split
from sts_policy.final_freeze import verify_final_freeze


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_dev_seeds(seeds: list[int], evidence: dict[str, Any],
                       snapshot: dict[str, Any]) -> None:
    """The wrapper has no fallback to a fresh or withheld smoke-test world."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct previously verified development seeds")
    config = evidence.get("config", {})
    if config.get("split") != "dev" or config.get("task_kind") != "run":
        raise ValueError("Evidence must be a completed development full-run evaluation")
    allowed = {row["world_seed"] for round in evidence.get("rounds", [])
               for row in round.get("episodes", []) if not row.get("fault")}
    embargo = set(snapshot.get("generated_definitions", {}).get("worlds", []))
    for seed in seeds:
        if seed == 3 or world_split(seed) != "dev" or seed in embargo or seed not in allowed:
            raise ValueError(f"World {seed} is not an explicitly verified, non-embargoed DEV world")
    expected = snapshot["backend"]["module"]["sha256"]
    if evidence.get("training_environment", {}).get("native_module_sha256") != expected:
        raise ValueError("Development evidence uses a different native backend")


def verify_trace(path: Path, *, seed: int, checkpoint: Path,
                 provenance: dict[str, Any]) -> dict[str, Any]:
    """Check both decision phases and zero automatic native decisions, every step."""
    records = read_trace(path)
    header = records[0]
    if header.get("seed") != seed or header.get("env_config", {}).get("mode") != "full":
        raise ValueError("CLI trace world/control mode differs from the command")
    if header.get("agent") != "sts_policy.model_agent:make_agent":
        raise ValueError("CLI did not use the trained policy plugin")
    if header.get("agent_config") != {"checkpoint": str(checkpoint), "device": "cpu", "deterministic": True}:
        raise ValueError("CLI agent configuration differs from the frozen policy")
    previous = None
    phases = {"combat": 0, "out_of_combat": 0}
    for row in records[1:]:
        if row["type"] in ("reset", "step", "end"):
            info = row.get("info", {})
            if info.get("native_steps") != 0 or info.get("native_provenance") != provenance:
                raise ValueError("CLI performed automatic native decisions or changed backend identity")
        if row["type"] in ("reset", "step"):
            previous = row["observation"]
        elif row["type"] == "action":
            if previous is None or row.get("observation_hash") != observation_hash(previous):
                raise ValueError("CLI action is not bound to its public observation")
            action = row["action"]
            if isinstance(action, bool) or action not in {a["index"] for a in previous["legal_actions"]}:
                raise ValueError("CLI policy action is outside the public legal menu")
            phases[previous["phase"]] += 1
    end = records[-1]
    if end.get("type") != "end" or end.get("error") is not None or end.get("status") not in {"terminated", "truncated"}:
        raise ValueError("CLI rollout did not complete without a simulator error")
    completed_steps = sum(row["type"] == "step" for row in records)
    if sum(phases.values()) != completed_steps or end.get("steps") != completed_steps:
        raise ValueError("Successful CLI trace contains an uncompleted policy action")
    if any(count == 0 for count in phases.values()):
        raise ValueError("CLI verification must exercise both combat and OOC policy heads")
    if end.get("info", {}).get("termination_reason") == "max_steps":
        raise ValueError("CLI verification exhausted its decision horizon")
    # Time is the only field expected to differ across identical fresh processes.
    normalized = [{k: value for k, value in row.items()
                   if not (row["type"] == "end" and k == "elapsed_seconds")} for row in records]
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"seed": seed, "steps": end["steps"], "decision_phases": phases,
            "native_steps": 0, "status": end["status"],
            "termination_reason": end["info"].get("termination_reason"),
            "normalized_trace_sha256": digest, "raw_sha256": file_sha(path)}


def source_identity(root: Path) -> dict[str, str]:
    paths = [*sorted((root / "src/sts_env").rglob("*.py")),
             *sorted((root / "src/sts_policy").glob("*.py")), Path(__file__).resolve()]
    return {str(path): file_sha(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default="shared")
    parser.add_argument("--dev-evidence", type=Path, required=True)
    parser.add_argument("--seeds", default="10121,10011")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-act", type=int, choices=(1, 3), default=1)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--process-timeout", type=float, default=180.)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.process_timeout <= 0:
        parser.error("Decision and process bounds must be positive")
    args.freeze = args.freeze.resolve()
    args.dev_evidence = args.dev_evidence.resolve()
    args.output_directory = args.output_directory.resolve()
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads(args.freeze.read_text())
    build = Path(snapshot["backend"]["build_dir"])
    verify_final_freeze(snapshot, backend_build_dir=build)
    checkpoint = Path(snapshot["checkpoints"][args.checkpoint_key]["path"])
    seeds = [int(seed) for seed in args.seeds.split(",")]
    evidence = json.loads(args.dev_evidence.read_text())
    verified_dev_seeds(seeds, evidence, snapshot)
    env = {**os.environ, "STS_LIGHTSPEED_BUILD": str(build), "PYTHONPATH": str(root / "src"),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    recorded_env = {key: env[key] for key in ("STS_LIGHTSPEED_BUILD", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    bindings = {"freeze": file_sha(args.freeze), "dev_evidence": file_sha(args.dev_evidence),
                "source": source_identity(root), "checkpoint": file_sha(checkpoint)}
    plan = {"checkpoint": str(checkpoint), "checkpoint_key": args.checkpoint_key,
            "bindings": bindings, "environment": recorded_env, "seeds": seeds, "commands": []}
    for seed in seeds:
        for repeat in (1, 2):
            trace = args.output_directory / f"seed{seed}_repeat{repeat}.jsonl"
            command = [sys.executable, "-m", "sts_env", "run", "--agent", "sts_policy.model_agent:make_agent",
                       "--agent-config", json.dumps({"checkpoint": str(checkpoint), "device": "cpu", "deterministic": True}),
                       "--seed", str(seed), "--policy-seed", "777", "--mode", "full", "--max-act", str(args.max_act),
                       "--max-steps", str(args.max_steps), "--battle-simulations", "200", "--step-timeout", "10",
                       "--output", str(trace)]
            plan["commands"].append({"kind": "run", "seed": seed, "trace": str(trace), "argv": command})
        plan["commands"].append({"kind": "replay", "seed": seed,
                                 "argv": [sys.executable, "-m", "sts_env", "replay",
                                          str(args.output_directory / f"seed{seed}_repeat1.jsonl")]})
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True, indent=2))
        return
    if args.output_directory.exists():
        raise ValueError("Use a new verification directory to preserve all previous traces")
    # The planned source and checkpoint are frozen before any model is loaded.
    from sts_policy.training import load_checkpoint
    if load_checkpoint(checkpoint).get("model_config", {}).get("kind") != "composite":
        raise ValueError("Frozen checkpoint is not the requested composite policy")
    args.output_directory.mkdir(parents=True)
    result = {"plan": plan, "operations": [], "verified_seeds": []}
    record_path = args.output_directory / "verification.json"
    record_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    traces = {}
    for number, operation in enumerate(plan["commands"]):
        verify_final_freeze(snapshot, backend_build_dir=build)
        if source_identity(root) != bindings["source"] or file_sha(args.freeze) != bindings["freeze"] or file_sha(args.dev_evidence) != bindings["dev_evidence"]:
            raise ValueError("Verification source or evidence changed after plan creation")
        started = time.time()
        with (args.output_directory / f"command{number}.log").open("w") as log:
            completed = subprocess.run(operation["argv"], env=env, cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=args.process_timeout)
        status = {**operation, "started": started, "finished": time.time(), "exit_code": completed.returncode}
        result["operations"].append(status)
        record_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        if completed.returncode:
            raise ValueError(f"CLI command failed; preserved evidence at {record_path}")
        verify_final_freeze(snapshot, backend_build_dir=build)
        if operation["kind"] == "run":
            checked = verify_trace(Path(operation["trace"]), seed=operation["seed"], checkpoint=checkpoint,
                                   provenance=snapshot["backend"]["provenance"])
            traces.setdefault(operation["seed"], []).append(checked)
        else:
            pair = traces[operation["seed"]]
            if pair[0]["normalized_trace_sha256"] != pair[1]["normalized_trace_sha256"]:
                raise ValueError("Fresh-process complete public traces differ")
            result["verified_seeds"].append(pair[0])
        record_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if source_identity(root) != bindings["source"]:
        raise ValueError("Source changed during verification")
    print(json.dumps({"verified": True, "record": str(record_path), "seeds": seeds}, sort_keys=True))


if __name__ == "__main__":
    main()
