"""Collect original-policy TRAIN rehearsal/MC targets and a common critic warm start."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path

import torch

from sts_policy.entity_model import load_policy, save_policy
from sts_policy.strategic_collection import collect
from sts_policy.strategic_data import continuation_tasks, stratified_rows, write_buffer, read_buffer
from sts_policy.strategic_learning import StrategicConfig, optimizers, update, OptimizerBundle, calibration
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, summarize, worker_settings, matched_tasks
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, summarize, worker_settings, matched_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh", type=int, default=32)
    parser.add_argument("--per-act", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--critic-epochs", type=int, default=5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if (args.output / "collection.json").exists():
        parser.error("output already contains a collection; use a fresh condition directory")
    torch.set_num_threads(1)
    model, payload = load_policy(args.initial)
    tasks = matched_tasks(bank=None, kind="run", split="train", episodes=args.fresh,
                          world_start=340000000, seed=0, horizon=4096, max_act=3)
    continuations = []
    for act in (2, 3):
        continuations.extend(continuation_tasks(args.bank, act=act, count=args.per_act, seed=act))
    tasks += continuations
    holdout_tasks = matched_tasks(bank=None, kind="run", split="train", episodes=24,
                          world_start=350000000, seed=0, horizon=4096, max_act=3)
    with ExitStack() as stack:
        workers = [stack.enter_context(CurriculumWorker(worker_settings(), timeout=10.)) for _ in range(args.workers)]
        result = collect(model, workers, tasks, credit_unit="micro", deterministic=True, max_seconds=args.seconds)
        holdout = collect(model, workers, holdout_tasks, credit_unit="micro", deterministic=True, max_seconds=args.seconds)
    rows, holdout_rows = result.pop("transitions"), holdout.pop("transitions")
    if result["faults"] or holdout["faults"] or not result["cohort_completed"] or not holdout["cohort_completed"]:
        atomic_json(args.output / "collection.json", {"train": result, "holdout": holdout})
        raise RuntimeError("common calibration collection incomplete/faulted; evidence preserved")
    selected = stratified_rows(rows, per_group=96, seed=0)
    selected_holdout = stratified_rows(holdout_rows, per_group=96, seed=1)
    metadata = {"source_checkpoint_sha256": file_identity(args.initial),
                "policy_code_identity": payload["metadata"]["policy_code_identity"],
                "training_environment": payload["metadata"]["training_environment"],
                "purpose": "TRAIN original-policy MC calibration and choice rehearsal; no teacher imitation"}
    dataset_sha = write_buffer(args.output / "calibration_train.jsonl", selected, metadata)
    holdout_sha = write_buffer(args.output / "calibration_holdout.jsonl", selected_holdout, metadata)
    atomic_json(args.output / "collection.json", {"train": result, "holdout": holdout,
        "train_summary": summarize(result), "holdout_summary": summarize(holdout),
        "tasks": [t.to_dict() for t in tasks], "holdout_tasks": [t.to_dict() for t in holdout_tasks],
        "selected_rows": len(selected), "selected_holdout_rows": len(selected_holdout),
        "calibration_train_sha256": dataset_sha, "calibration_holdout_sha256": holdout_sha})
    print(json.dumps({"train": summarize(result), "holdout": summarize(holdout),
                      "rows": len(selected), "holdout_rows": len(selected_holdout)}), flush=True)
    if args.collect_only:
        return
    config = StrategicConfig(epochs=args.critic_epochs, critic_lr=args.critic_lr)
    actor, critic = optimizers(model, config)
    before = calibration(model, selected_holdout, device=args.device)
    metrics = update(model, selected, rehearsal=[], config=config, actor_optimizer=actor,
        critic_optimizer=critic, seed=0, device=args.device, critic_only=True, completed_targets=True)
    after = calibration(model, selected_holdout, device=args.device)
    # Actor identity is checked directly, not inferred from unchanged greedy play.
    original, _ = load_policy(args.initial)
    unchanged = all(torch.equal(a.cpu(), b.cpu()) for a,b in zip(model.actor_parameters(), original.actor_parameters()))
    if not unchanged:
        raise RuntimeError("critic calibration moved actor parameters")
    save_policy(str(args.output / "common_calibrated.pt"), model,
        optimizer=OptimizerBundle(actor, critic), step=metrics["critic_steps"],
        metadata={**payload["metadata"], "stage": "common_mc_critic_calibration", "calibration_config": asdict(config),
                  "calibration_train_sha256": dataset_sha, "calibration_holdout_sha256": holdout_sha,
                  "actor_parameters_unchanged": unchanged})
    atomic_json(args.output / "critic_calibration.json", {"metrics": metrics, "holdout_before": before,
        "holdout_after": after, "actor_parameters_unchanged": unchanged,
        "checkpoint_sha256": file_identity(args.output / "common_calibrated.pt")})
    print(json.dumps({"holdout_before": before["overall"], "holdout_after": after["overall"],
                      "actor_parameters_unchanged": unchanged, "seconds": metrics["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
