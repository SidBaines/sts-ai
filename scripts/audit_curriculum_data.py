#!/usr/bin/env python3
"""Audit raw native labels and derive battle curriculum subsets without rewrites."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from sts_policy.data import atomic_json, atomic_jsonl


def audit(bank: Path, output: Path) -> dict:
    phases, screens, records_by_act, worlds_max_act = Counter(), Counter(), Counter(), Counter()
    enemy_by_act = defaultdict(Counter)
    inactive_slots_by_act = Counter()
    floor_by_act = defaultdict(Counter)
    battle_by_act, outcomes, faults = Counter(), Counter(), []
    split_worlds = Counter()
    label_errors = 0
    subsets = defaultdict(list)
    for path in sorted((bank / "worlds").glob("*.json")):
        world = json.loads(path.read_text())
        if world["status"] != "ok":
            faults.append({key: world.get(key) for key in ("seed", "split", "error_type", "error", "discarded_records")})
            continue
        split_worlds[world["split"]] += 1
        maximum = 0
        for row in world["records"]:
            obs = row["observation"]
            act = int(obs["state"]["act"])
            maximum = max(maximum, act)
            phases[f"{row['split']}:{row['phase']}"] += 1
            records_by_act[act] += 1
            screens[obs["state"]["screen_state"]] += 1
            index = row["teacher_index"]
            if not 0 <= index < len(obs["legal_actions"]) or obs["legal_actions"][index]["index"] != index:
                label_errors += 1
        worlds_max_act[maximum] += 1
        result = world["result"]
        outcomes[result.get("task_outcome") or result.get("termination_reason") or "unknown"] += 1
        for row in world["battles"]:
            obs = world["records"][len(row["prefix"])]["observation"]
            act, floor = int(obs["state"]["act"]), int(obs["state"]["floor"])
            battle_by_act[act] += 1
            floor_by_act[act][floor] += 1
            enemy_by_act[act].update(enemy["name"] for enemy in obs["state"]["combat"]["enemies"] if enemy.get("alive", False))
            inactive_slots_by_act[act] += sum(not enemy.get("alive", False) for enemy in obs["state"]["combat"]["enemies"])
            annotated = {**row, "curriculum_metadata": {"act": act, "floor": floor,
                          "room": obs["state"]["room"], "source_bank": str(bank.resolve())}}
            subsets[f"act{act}"].append(annotated)
            if act == 1:
                subsets["act1_early" if floor <= 3 else "act1_later"].append(annotated)
            room = str(obs["state"]["room"]).rsplit(".", 1)[-1]
            if room in {"ELITE", "BOSS"}:
                subsets[f"act{act}_{room.lower()}"].append(annotated)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in subsets.items():
        atomic_jsonl(output / f"battles_{name}.jsonl", rows)
    report = {"source_bank": str(bank.resolve()), "valid_worlds_by_split": dict(split_worlds),
              "worlds_by_furthest_observed_act": dict(worlds_max_act), "labels_by_split_phase": dict(phases),
              "records_by_act": dict(records_by_act), "screens": dict(screens),
              "battle_counts_by_act": dict(battle_by_act), "floors_by_act": {k: dict(v) for k, v in floor_by_act.items()},
              "enemies_by_act": {k: dict(v) for k, v in enemy_by_act.items()}, "inactive_enemy_slots_by_act": dict(inactive_slots_by_act), "world_outcomes": dict(outcomes),
              "label_menu_errors": label_errors, "faults": faults,
              "subset_counts": {name: {"total": len(rows), "splits": dict(Counter(row['split'] for row in rows))} for name, rows in subsets.items()}}
    atomic_json(output / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.bank, args.output)
    print(json.dumps({key: report[key] for key in ("valid_worlds_by_split", "worlds_by_furthest_observed_act", "records_by_act", "battle_counts_by_act", "world_outcomes", "label_menu_errors", "subset_counts")}, sort_keys=True), flush=True)
    print(json.dumps({"fault_worlds": len(report['faults'])}), flush=True)

if __name__ == "__main__":
    main()
