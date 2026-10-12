"""Bounded matched-buffer LR measurements for the new zero-output residual actor."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path

import torch

from sts_policy import entities
from sts_policy.entity_model import load_policy, save_policy
from sts_policy.strategic_collection import collect
from sts_policy.strategic_data import read_buffer, write_buffer, continuation_tasks
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[3e-5, 3e-4, .001, .003])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh measurement directory")
    torch.set_num_threads(1)
    initial, payload = load_policy(args.initial)
    tasks = matched_tasks(bank=None, kind="run", split="train", episodes=32, world_start=360010000,
                          seed=1, horizon=4096, max_act=3)
    for act in (2, 3):
        tasks += continuation_tasks(args.bank, act=act, count=16, seed=1000+act)
    with ExitStack() as stack:
        workers = [stack.enter_context(CurriculumWorker(worker_settings(), timeout=10.)) for _ in range(args.workers)]
        result = collect(initial, workers, tasks, credit_unit="macro", sampling_seed=1, max_seconds=180.)
    rows = result.pop("transitions")
    if result["faults"] or not result["cohort_completed"]:
        atomic_json(args.output / "collection.json", result)
        raise RuntimeError("calibration collection incomplete")
    rehearsal_rows, _ = read_buffer(args.rehearsal)
    rehearsal = [entities.encode_observation(row.observation) for row in rehearsal_rows if row.actor_choice]
    identity = write_buffer(args.output / "matched_buffer.jsonl", rows,
                            {"source_checkpoint_sha256": file_identity(args.initial),
                             "policy_code_identity": payload["metadata"]["policy_code_identity"]})
    atomic_json(args.output / "collection.json", {**result, "summary": summarize(result),
                                                "tasks": [t.to_dict() for t in tasks], "buffer_sha256": identity})
    summary = []
    for rate in args.rates:
        model, _ = load_policy(args.initial)
        config = StrategicConfig(actor_lr=rate)
        actor, critic = optimizers(model, config)
        metrics = update(model, rows, rehearsal=rehearsal, config=config, actor_optimizer=actor,
            critic_optimizer=critic, seed=1, device="mps")
        name = f"lr_{rate:g}"
        save_policy(str(args.output / f"{name}.pt"), model, optimizer=OptimizerBundle(actor, critic),
            metadata={**payload["metadata"], "stage": "matched_buffer_lr_calibration", "config": asdict(config),
                      "buffer_sha256": identity})
        atomic_json(args.output / f"{name}.json", metrics)
        row = {"actor_lr": rate, "actor_steps": metrics["actor_steps"], "seconds": metrics["elapsed_seconds"],
               "buffer_drift": metrics["buffer_drift"], "rehearsal_drift": metrics["rehearsal_drift"]}
        summary.append(row)
        print(json.dumps(row), flush=True)
    atomic_json(args.output / "summary.json", {"buffer_sha256": identity, "conditions": summary,
        "note": "Plasticity/safety calibration on the same stochastic TRAIN buffer; not game improvement."})


if __name__ == "__main__":
    main()
