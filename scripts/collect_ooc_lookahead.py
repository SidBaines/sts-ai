#!/usr/bin/env python3
"""Selective OOC relabeling from a fresh bank using bounded native continuations."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec, public_hash
from sts_policy.data import atomic_json, atomic_jsonl
from sts_policy.lookahead import CounterfactualOOCWorker, LOOKAHEAD_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--split", default="train")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--deadline-seconds", type=float, default=60)
    args = parser.parse_args()
    sources = []
    for path in sorted((args.bank / "worlds").glob("*.json")):
        world = json.loads(path.read_text())
        if world["status"] != "ok" or world["split"] not in args.split.split(","):
            continue
        prefix, previews = [], []
        for row in world["records"]:
            if row["phase"] == "out_of_combat" and len(row["observation"]["legal_actions"]) >= 2:
                source_id = f"{world['seed']}-{row['step']}"
                sources.append((source_id, row["observation"], TaskSpec(
                    seed=world["seed"], kind="run", prefix=tuple(prefix),
                    ooc_previews=tuple(previews), expected_public_hash=public_hash(row["observation"]),
                    expected_provenance=world["provenance"], max_act=world["settings"]["max_act"],
                    battle_simulations=world["settings"]["simulations"])))
            prefix.append(row["executed_index"])
            previews.append(row["phase"] == "out_of_combat")
    # Deterministic sample across worlds/phases instead of taking the earliest map.
    random.Random(19051).shuffle(sources)
    sources = sources[:args.states]
    settings = {"branches": args.branches, "horizon": args.horizon, "simulations": args.simulations}
    args.output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256((Path(__file__).read_bytes() + (Path(__file__).parents[1] / "src/sts_policy/lookahead.py").read_bytes())).hexdigest()
    request = {"source_bank": str(args.bank.resolve()), "settings": settings,
               "source_sha256": identity, "source_ids": [source[0] for source in sources],
               "split": args.split}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(saved.get(key) != value for key, value in request.items()):
            raise ValueError("Lookahead resume source/settings/code mismatch; use a new directory")
    atomic_json(manifest_path, {**request, "status": "collecting"})
    start = time.monotonic()
    def lane(items):
        with CounterfactualOOCWorker({"fast_observations": True}, timeout=args.timeout) as worker:
            for source_id, observation, spec in items:
                path = args.output / f"state-{source_id}.json"
                if path.exists():
                    continue
                if time.monotonic() - start >= args.deadline_seconds:
                    break
                try:
                    result = worker.score_ooc(spec, branches=args.branches, horizon=args.horizon,
                                              simulations=args.simulations)
                    if result["observation"] != observation:
                        raise SimulatorError("Source observation changed on replay")
                    row = {**result, "status": "ok", "source_id": source_id,
                           "phase": "out_of_combat", "teacher_kind": "native_privileged_partial_menu_lookahead"}
                except SimulatorError as exc:
                    row = {"status": "simulator_fault", "source_id": source_id,
                           "world_seed": spec.seed, "error": str(exc), "settings": settings}
                atomic_json(path, row)
                print(json.dumps({"source_id": source_id, "status": row["status"],
                                  "teacher_index": row.get("teacher_index"),
                                  "scripted_index": row.get("scripted_index")}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lane, [sources[i::args.workers] for i in range(args.workers)]))
    rows = [json.loads((args.output / f"state-{source_id}.json").read_text()) for source_id, _, _ in sources
            if (args.output / f"state-{source_id}.json").exists()]
    valid = [row for row in rows if row["status"] == "ok"]
    atomic_jsonl(args.output / "teacher.jsonl", valid)
    manifest = {**request, "status": "complete" if len(rows) == len(sources) else "deadline", "lookahead_version": LOOKAHEAD_VERSION, "source_bank": str(args.bank.resolve()),
                "settings": settings, "seconds": time.monotonic() - start, "source_sha256": identity,
                "requested_states": len(sources), "completed_states": len(rows), "valid_states": len(valid),
                "fault_states": len(rows)-len(valid), "scripted_disagreements": sum(row["teacher_index"] != row["scripted_index"] for row in valid),
                "strict_improvements": sum((row.get("score_gain_over_scripted") or 0) > 1e-9 for row in valid),
                "mean_score_gain": sum(row.get("score_gain_over_scripted") or 0 for row in valid) / max(1, len(valid))}
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
