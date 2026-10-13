"""Finite single-condition protected PPO pilot with permanent evidence and recovery."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import torch

from sts_policy import entities
from sts_policy.entity_model import load_policy, save_policy
from sts_policy.strategic_collection import collect
from sts_policy.strategic_control import resources, resource_gate, paired_retention, retention_gate, drift_gate, unchanged_sources, CONTROL_VERSION
from sts_policy.strategic_data import read_buffer, continuation_tasks, stratified_rows, write_buffer
from sts_policy.strategic_learning import StrategicConfig, optimizers, update, OptimizerBundle
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, summarize, worker_settings, matched_tasks
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, summarize, worker_settings, matched_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credit-unit", choices=("micro", "macro"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--fresh", type=int, default=32)
    parser.add_argument("--per-act", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--lambda", dest="gae_lambda", type=float, default=.95)
    parser.add_argument("--actor-act-balanced",action="store_true",help="Give each present act equal actor-loss mass")
    parser.add_argument("--deadline-utc", default="2026-10-13T07:30:00+00:00")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if min(args.chunks, args.eval_every, args.fresh, args.workers) < 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("positive counts and bounded GAE lambda required")
    if args.output.exists():
        parser.error("a pilot must use a fresh output directory; resume support is a separate explicit operation")
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        parser.error("deadline requires UTC timezone offset")
    torch.set_num_threads(1)
    model, payload = load_policy(args.initial)
    rehearsal_rows, buffer_metadata = read_buffer(args.rehearsal)
    if buffer_metadata.get("reference_checkpoint_sha256", buffer_metadata.get("source_checkpoint_sha256")) != file_identity(args.initial):
        raise ValueError("rehearsal reference differs from the common initialization")
    if buffer_metadata.get("reference_policy_code_identity", buffer_metadata.get("policy_code_identity")) != payload["metadata"]["policy_code_identity"]:
        raise ValueError("rehearsal policy encoding identity mismatch")
    rehearsal = [entities.encode_observation(r.observation) for r in rehearsal_rows if r.actor_choice]
    baseline_document = json.loads(args.baseline.read_text())
    if baseline_document["checkpoint_sha256"] != file_identity(args.initial):
        raise ValueError("DEV baseline differs from the common initialization")
    baseline = baseline_document["results"][0]
    from sts_policy.composition import training_environment_record
    actual_environment = training_environment_record()
    expected_environment = payload["metadata"]["training_environment"]
    for key in ("python_interface_sha256", "native_module_sha256", "environment_version", "observation_version"):
        if actual_environment.get(key) != expected_environment.get(key):
            raise ValueError(f"pilot environment differs from common initialization: {key}")
    from sts_policy.curriculum import TaskSpec
    evaluation_tasks = [TaskSpec.from_dict(row) for row in baseline_document["tasks"]]
    config = StrategicConfig(actor_lr=args.actor_lr, critic_lr=args.critic_lr,actor_act_balanced=args.actor_act_balanced)
    actor, critic = optimizers(model, config)
    state = {"control_version": CONTROL_VERSION, "run_id": str(uuid.uuid4()), "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "starting", "phase": "preflight",
        "config": {**vars(args), "output": str(args.output), "rehearsal": str(args.rehearsal),
                   "bank": str(args.bank), "baseline": str(args.baseline)}, "trainer_config": asdict(config),
        "initial_checkpoint_sha256": file_identity(args.initial), "learner_checkpoint": args.initial,
        "validated_champion": args.initial, "candidate_leader": None, "completed_chunks": 0,
        "optimizer_start": "fresh PPO moments after common initialization", "updates": [], "evaluations": []}
    tracked_files = [Path(__file__), Path("scripts/run_strategic_v3.py"),
                     *Path("src/sts_policy").glob("*.py"),
                     Path("src/sts_policy/public_vocabulary_v3.json"),
                     *Path("src/sts_env").rglob("*.py"),
                     Path("src/sts_env/_backend/native_manifest.json"),
                     Path("patches/sts_lightspeed_python_api.patch")]
    native_build = Path(os.environ["STS_LIGHTSPEED_BUILD"])
    native_manifest = json.loads((native_build / "build_manifest.json").read_text())
    tracked_files.extend((native_build / "build_manifest.json", native_build / native_manifest["module_file"]))
    source_identity = {str(path): file_identity(path) for path in tracked_files}
    state["source_identity"] = source_identity
    state["bank_sha256"] = file_identity(args.bank)
    state["rehearsal_sha256"] = file_identity(args.rehearsal)
    args.output.mkdir(parents=True)
    journal = args.output / "status.json"
    def publish(status=None, phase=None):
        if status is not None:
            state["status"] = status
        if phase is not None:
            state["phase"] = phase
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(journal, state)
    publish("running", "preflight")
    print(json.dumps({"pid": os.getpid(), "run_id": state["run_id"], "status": str(journal)}), flush=True)
    caffeine = None
    try:
        try:
            caffeine = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            state["owned_caffeinate_pid"] = caffeine.pid
        except FileNotFoundError:
            pass
        with ExitStack() as stack:
            workers = [stack.enter_context(CurriculumWorker(worker_settings(), timeout=10.)) for _ in range(args.workers)]
            total_steps = 0
            for chunk in range(1, args.chunks+1):
                state["resources"] = resources(args.output)
                reason = resource_gate(state["resources"])
                remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
                if reason or remaining < 20:
                    state["stop_reason"] = reason or "allocation_checkpoint"
                    publish("resource_pause" if reason else "allocation_checkpoint", "saved")
                    break
                if not unchanged_sources(source_identity):
                    state["stop_reason"] = "source_changed"
                    publish("attention", "saved")
                    break
                if file_identity(args.bank) != state["bank_sha256"]:
                    raise ValueError("continuation bank changed during the pilot")
                tasks = matched_tasks(bank=None, kind="run", split="train", episodes=args.fresh,
                    world_start=360000000+10000*chunk, seed=args.sampling_seed+chunk, horizon=4096, max_act=3)
                for act in (2, 3):
                    tasks += continuation_tasks(args.bank, act=act, count=args.per_act,
                                                seed=1000*chunk+act)
                publish(phase=f"collect_{chunk:04d}")
                result = collect(model, workers, tasks, credit_unit=args.credit_unit, sampling_seed=100000*args.sampling_seed+chunk,
                    gae_lambda=args.gae_lambda, max_seconds=min(180., remaining-5))
                rows = result.pop("transitions")
                result["summary"] = summarize(result)
                result["tasks"] = [task.to_dict() for task in tasks]
                # Persist outcomes/settings and a bounded public reservoir before
                # an optimizer step; interrupted conditions remain diagnosable.
                atomic_json(args.output / f"collection_{chunk:04d}.json", result)
                write_buffer(args.output / f"samples_{chunk:04d}.jsonl", stratified_rows(rows, per_group=4, seed=chunk),
                    {"source_checkpoint": state["learner_checkpoint"], "credit_unit": args.credit_unit})
                if not rows or not result["cohort_completed"] or result["faults"]:
                    state["stop_reason"] = "collection_incomplete_or_faulted"
                    publish("attention", "saved")
                    break
                if (deadline-datetime.now(timezone.utc)).total_seconds() < 20:
                    state["stop_reason"] = "allocation_checkpoint"
                    publish("allocation_checkpoint", "saved")
                    break
                publish(phase=f"update_{chunk:04d}")
                if not unchanged_sources(source_identity):
                    raise RuntimeError("sources changed between collection and update")
                metrics = update(model, rows, rehearsal=rehearsal, config=config, actor_optimizer=actor,
                    critic_optimizer=critic, seed=100000*args.seed+chunk, device=args.device)
                checkpoint = args.output / f"learner_{chunk:04d}.pt"
                if not unchanged_sources(source_identity):
                    raise RuntimeError("sources changed during update; current weights are quarantined")
                total_steps += metrics["actor_steps"]+metrics["critic_steps"]
                save_policy(str(checkpoint), model, optimizer=OptimizerBundle(actor, critic), step=total_steps,
                    metadata={**payload["metadata"], "stage": "protected_strategic_ppo", "credit_unit": args.credit_unit,
                        "optimizer_seed": args.seed, "sampling_seed": 100000*args.sampling_seed+chunk,
                        "chunk": chunk, "trainer_config": asdict(config), "gae_lambda": args.gae_lambda,
                        "rehearsal_sha256": file_identity(args.rehearsal), "source_checkpoint": state["learner_checkpoint"],
                        "runtime_loaded_sources": source_identity})
                metrics.update(checkpoint=str(checkpoint), checkpoint_sha256=file_identity(checkpoint),
                    chunk=chunk, collection_summary=result["summary"])
                atomic_json(args.output / f"update_{chunk:04d}.json", metrics)
                state.update(learner_checkpoint=str(checkpoint), learner_sha256=metrics["checkpoint_sha256"], completed_chunks=chunk)
                state["updates"].append({"chunk": chunk, "path": f"update_{chunk:04d}.json", "checkpoint_sha256": metrics["checkpoint_sha256"],
                    "original_kl": metrics["rehearsal_drift"]["original_kl"], "seconds": metrics["elapsed_seconds"]})
                reason = drift_gate(metrics)
                print(json.dumps({"chunk": chunk, "collect": result["summary"], "update_seconds": metrics["elapsed_seconds"],
                    "original_kl": metrics["rehearsal_drift"]["original_kl"], "greedy_drift": metrics["rehearsal_drift"]["changed_greedy_fraction"]}), flush=True)
                publish(phase="checkpoint_saved")
                if chunk % args.eval_every == 0 or chunk == args.chunks or reason:
                    if not unchanged_sources(source_identity):
                        raise RuntimeError("sources changed before evaluation")
                    remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
                    if remaining < 20:
                        state["stop_reason"] = "allocation_checkpoint"
                        publish("allocation_checkpoint", "saved")
                        break
                    publish(phase=f"evaluate_{chunk:04d}")
                    evaluation = collect(model, workers, evaluation_tasks, credit_unit="macro", deterministic=True,
                        max_seconds=min(180., remaining-5), retain_transitions=False)
                    evaluation.pop("transitions")
                    paired = paired_retention(baseline, evaluation)
                    evaluation.update(summary=summarize(evaluation), paired=paired, checkpoint=str(checkpoint),
                                      checkpoint_sha256=metrics["checkpoint_sha256"])
                    atomic_json(args.output / f"evaluation_{chunk:04d}.json", evaluation)
                    state["evaluations"].append({"chunk": chunk, "path": f"evaluation_{chunk:04d}.json", "paired": paired,
                                                "checkpoint_sha256": metrics["checkpoint_sha256"]})
                    if paired["victory"]["net"] > 0:
                        state["candidate_leader"] = str(checkpoint)
                    reason = reason or retention_gate(paired, len(evaluation_tasks))
                    print(json.dumps({"evaluation_chunk": chunk, "summary": evaluation["summary"], "paired": paired}), flush=True)
                if reason:
                    state["stop_reason"] = reason
                    publish("attention", "saved")
                    break
            else:
                publish("pilot_complete", "saved")
    except BaseException as error:
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        # Preserve incomplete in-memory updates for diagnosis without presenting
        # them as a validated learner/champion or a completed scientific chunk.
        try:
            partial = args.output / "failed_partial.pt"
            save_policy(str(partial), model, optimizer=OptimizerBundle(actor, critic),
                metadata={**payload["metadata"], "stage": "failed_partial_update",
                          "error": state["error"], "runtime_loaded_sources": source_identity,
                          "validated_previous_checkpoint": state["learner_checkpoint"]})
            state["quarantined_partial_checkpoint"] = str(partial)
        except Exception as save_error:
            state["partial_save_error"] = str(save_error)
        publish("failed", "recovery_preserved")
        raise
    finally:
        if caffeine is not None and caffeine.poll() is None:
            caffeine.terminate()
            caffeine.wait(timeout=5)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        publish()
    print(json.dumps({"status": state["status"], "completed_chunks": state["completed_chunks"],
                      "stop_reason": state.get("stop_reason"), "learner_checkpoint": state["learner_checkpoint"]}), flush=True)


if __name__ == "__main__":
    main()
