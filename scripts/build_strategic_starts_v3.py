"""Construct and replay-check actor-bearing later starts from fresh TRAIN journals."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path

from sts_policy.curriculum import TaskSpec
from sts_policy.strategic_starts import strategic_frontier
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
try:
    from scripts.run_strategic_v3 import atomic_json, worker_settings
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json, worker_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh frontier directory")
    records = []
    sources = {}
    for path in sorted((args.bank / "worlds").glob("*.json")):
        world = json.loads(path.read_text())
        sources[path.name] = file_identity(path)
        for act in (2, 3):
            record = strategic_frontier(world, act)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError("no actor-bearing later frontiers in source bank")
    results = []
    with ExitStack() as stack:
        workers = [stack.enter_context(CurriculumWorker(worker_settings(), timeout=10.)) for _ in range(args.workers)]
        def lane(index):
            output = []
            for record in records[index::args.workers]:
                obs, info = workers[index].reset(TaskSpec.from_dict(record))
                if obs["phase"] != "out_of_combat" or len(obs["legal_actions"]) <= 1 or obs["state"]["act"] != record["start_act"]:
                    raise RuntimeError("replayed frontier lacks the expected strategic choice")
                output.append({"seed": record["seed"], "act": record["start_act"],
                               "public_hash": info["public_hash"], "legal_count": len(obs["legal_actions"]),
                               "floor": obs["state"]["floor"]})
            return output
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for output in pool.map(lane, range(args.workers)):
                results.extend(output)
    args.output.mkdir(parents=True)
    target = args.output / "starts.jsonl"
    target.write_text("".join(json.dumps(record, sort_keys=True)+"\n" for record in records))
    summary = {"source_bank_manifest_sha256": file_identity(args.bank / "manifest.json"),
               "source_world_journal_sha256": sources, "starts_sha256": file_identity(target),
               "frontiers": len(records), "act2": sum(r["start_act"] == 2 for r in records),
               "act3": sum(r["start_act"] == 3 for r in records), "verified": len(results), "replays": results}
    atomic_json(args.output / "manifest.json", summary)
    print(json.dumps({k: summary[k] for k in ("frontiers", "act2", "act3", "verified")}))


if __name__ == "__main__":
    main()
