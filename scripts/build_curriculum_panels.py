#!/usr/bin/env python3
"""Build act-balanced task panels without executing any held-out policy."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.data import atomic_json, atomic_jsonl
from sts_policy.aggregation import file_sha256


def identity(row):
    return hashlib.sha256(json.dumps(TaskSpec.from_dict(row).to_dict(), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def select_world_balanced(rows, limit, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    rng = random.Random(seed)
    worlds = sorted(grouped)
    rng.shuffle(worlds)
    for world in worlds:
        grouped[world].sort(key=lambda row: (row["floor"], identity(row)))
        rng.shuffle(grouped[world])
    result = []
    while len(result) < limit and any(grouped.values()):
        for world in worlds:
            if grouped[world] and len(result) < limit:
                result.append(grouped[world].pop())
    return result


def build_panels(bank: Path | list[Path], output: Path, *, per_act=16, seed=777):
    if per_act < 1:
        raise ValueError("per-act must be positive")
    banks = sorted({path.resolve() for path in ([bank] if isinstance(bank, Path) else bank)}, key=str)
    acts = {1: [], 2: [], 3: []}
    for source_bank in banks:
        source = source_bank / "stage_subsets"
        if all((source / f"battles_act{act}.jsonl").exists() for act in (1, 2, 3)):
            for act in (1, 2, 3):
                path = source / f"battles_act{act}.jsonl"
                acts[act].extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        else:
            # New banks need no materialized audit copies: classify by the public
            # observation at each stored prefix, streaming one world at a time.
            for path in sorted((source_bank / "worlds").glob("*.json")):
                world = json.loads(path.read_text())
                if world["status"] != "ok":
                    continue
                for row in world["battles"]:
                    obs = world["records"][len(row["prefix"])]["observation"]
                    act = int(obs["state"]["act"])
                    acts[act].append({**row, "curriculum_metadata": {
                        "act": act, "floor": obs["state"]["floor"], "room": obs["state"]["room"],
                        "source_bank": str(source_bank)}})
    for act, rows in acts.items():
        unique = {}
        for row in rows:
            if int(row["curriculum_metadata"]["act"]) != act or row.get("split") != world_split(row["seed"]):
                raise ValueError("Task metadata act/split mismatch")
            unique[identity(row)] = row
        acts[act] = list(unique.values())
    output.mkdir(parents=True, exist_ok=True)
    panels = {}
    for act, rows in acts.items():
        train = [row for row in rows if row["split"] == "train"]
        atomic_jsonl(output / f"train_act{act}.jsonl", train)
        panels[f"train_act{act}"] = train
    # Supervisor cycles these act files: equal act rounds rather than fabricated
    # duplicate rows/new task identities. Every genuine prefix stays identifiable.
    schedules = [str((output / f"train_act{act}.jsonl").resolve()) for act in (1, 2, 3)]
    for split in ("dev", "final"):
        chosen = {act: select_world_balanced([row for row in rows if row["split"] == split],
                                           per_act, seed + act) for act, rows in acts.items()}
        mixed = [row for offset in range(max(map(len, chosen.values()), default=0))
                 for act in (1, 2, 3) if offset < len(chosen[act]) for row in [chosen[act][offset]]]
        hard = [row for row in mixed if row["curriculum_metadata"]["act"] in (2, 3)]
        panels[f"{split}_mixed"] = mixed
        panels[f"{split}_hard"] = hard
        atomic_jsonl(output / f"{split}_mixed.jsonl", mixed)
        atomic_jsonl(output / f"{split}_hard.jsonl", hard)
    summary = {}
    for name, rows in panels.items():
        by_act = {}
        for act in (1, 2, 3):
            selected = [row for row in rows if row["curriculum_metadata"]["act"] == act]
            worlds = sorted({row["seed"] for row in selected})
            by_act[str(act)] = {"tasks": len(selected), "worlds": worlds, "world_count": len(worlds),
                                "floor_counts": dict(Counter(row["floor"] for row in selected)),
                                "world_uncertainty_available": len(worlds) >= 2}
        summary[name] = {"tasks": len(rows), "task_ids": [identity(row) for row in rows],
                         "by_act": by_act, "path": str((output / f"{name}.jsonl").resolve())}
    manifest = {"source_bank": str(banks[0]), "source_banks": [str(path) for path in banks],
                "source_manifest_sha256s": {str(path): file_sha256(path / "manifest.json") for path in banks},
                "seed": seed, "per_act_requested": per_act, "panels": summary,
                "train_round_robin_paths": schedules, "no_oversampled_evaluation_rows": True,
                "assembly_only_no_policy_evaluation": True,
                "note": "Cycle train_act1/2/3 by round for equal act exposure. Consult actual per-act world counts; clustered intervals require at least two worlds."}
    atomic_json(output / "panels.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bank", type=Path)
    inputs.add_argument("--banks", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-act", type=int, default=16)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    manifest = build_panels(args.banks or args.bank, args.output, per_act=args.per_act, seed=args.seed)
    print(json.dumps({name: panel["tasks"] for name, panel in manifest["panels"].items()}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
