#!/usr/bin/env python3
"""Matched, bounded HP-preservation PPO pilots; run only with the campaign paused."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.rollout_training import task_identity

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
PANELS = ROOT / "data/nonllm_curriculum/panels_expanded"
ACT1 = ROOT / "data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl"


def identity(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        import os
        os.fsync(output.fileno())
    temporary.replace(path)


def load_campaign_module():
    spec = importlib.util.spec_from_file_location("health_campaign", ROOT / "scripts/run_nonllm_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_train_rows(panel_directory: Path, *, seed: int = 11, per_act: int = 64) -> list[dict[str, Any]]:
    selected = []
    for act in (1, 2, 3):
        rows = [json.loads(line) for line in (panel_directory / f"train_act{act}.jsonl").read_text().splitlines() if line]
        rows = [row for row in rows if world_split(row["seed"]) == "train"]
        random.Random(seed + act).shuffle(rows)
        seen = set()
        chosen = []
        for row in rows:
            if row["seed"] not in seen:
                seen.add(row["seed"])
                chosen.append(row)
            if len(chosen) == per_act:
                break
        if len(chosen) != per_act:
            raise ValueError(f"Act {act} has fewer than {per_act} distinct train worlds")
        selected.extend(chosen)
    random.Random(seed).shuffle(selected)
    return selected


def training_command(source: Path, output: Path, bank: Path, weight: float) -> list[str]:
    return [str(PYTHON), str(ROOT / "scripts/train_nonllm.py"), "ppo", "--checkpoint", str(source),
            "--output", str(output), "--metrics", str(output.with_suffix(".metrics.json")),
            "--tasks", str(bank), "--task-kind", "battle", "--split", "train", "--max-act", "3",
            "--horizon", "256", "--episodes", "128", "--seed", "11", "--seconds", "180",
            "--updates", "100000", "--batch-size", "1024", "--threads", "2", "--workers", "8",
            "--device", "mps", "--inference-device", "cpu", "--gamma", "1", "--clip-ratio", ".1",
            "--learning-rate", "1e-4", "--entropy-coef", ".01", "--reward-mode", "potential",
            "--terminal-hp-weight", str(weight), "--simulations", "200", "--fast-observations",
            "--fast-replay", "--deferred-bootstrap", "--warm-optimizer"]


def evaluation_command(combat: Path, ooc: Path, output: Path, cohort: str) -> list[str]:
    command = [str(PYTHON), str(ROOT / "scripts/train_nonllm.py"), "evaluate", "--output", str(output.with_suffix(".unused.pt")),
               "--metrics", str(output), "--split", "dev", "--seed", "777", "--device", "cpu",
               "--inference-device", "cpu", "--workers", "8", "--threads", "2", "--simulations", "200",
               "--fast-observations", "--fast-replay", "--deferred-bootstrap", "--seconds", "90"]
    if cohort == "composite_act1_24":
        return command + ["--combat-checkpoint", str(combat), "--ooc-checkpoint", str(ooc),
                          "--task-kind", "run", "--world-start", "10000", "--episodes", "24",
                          "--max-act", "1", "--horizon", "2048"]
    if cohort not in {"act1_48", "expanded_hard32"}:
        raise ValueError("unknown health-ablation cohort")
    hard = cohort == "expanded_hard32"
    return command + ["--checkpoint", str(combat), "--task-kind", "battle", "--episodes", "32" if hard else "48",
                      "--max-act", "3" if hard else "1", "--horizon", "256",
                      "--tasks", str(PANELS / "dev_hard.jsonl" if hard else ACT1)]


def command_tasks(command: list[str]) -> list[TaskSpec]:
    values = {token: command[index + 1] for index, token in enumerate(command[:-1]) if token.startswith("--")}
    kind, episodes = values["--task-kind"], int(values["--episodes"])
    split, seed = values["--split"], int(values["--seed"])
    horizon, max_act = int(values["--horizon"]), int(values["--max-act"])
    simulations = int(values["--simulations"])
    if "--tasks" in values:
        tasks = [replace(TaskSpec.from_dict(json.loads(line)), horizon=horizon, max_act=max_act,
                         battle_simulations=simulations) for line in Path(values["--tasks"]).read_text().splitlines() if line]
        tasks = [task for task in tasks if world_split(task.seed) == split]
    else:
        tasks = []
        world = int(values["--world-start"])
        while len(tasks) < episodes:
            if world_split(world) == split:
                tasks.append(TaskSpec(seed=world, kind=kind, horizon=horizon, max_act=max_act,
                                      battle_simulations=simulations))
            world += 1
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:episodes]
    random.Random(seed).shuffle(tasks)  # CLI's first collection round shuffle.
    return tasks


def completed_matching(metrics: dict[str, Any] | None, expected: list[str]) -> bool:
    if metrics is None or len(metrics.get("rounds", [])) != 1:
        return False
    row = metrics["rounds"][0]
    entries = row.get("episodes", []) + row.get("faults", [])
    return (Counter(expected) == Counter(row.get("task_ids", [])) == Counter(entry.get("task_id") for entry in entries)
            and not row.get("unstarted_tasks", 0) and row.get("cohort_completed") is not False
            and all(entry.get("termination_reason") != "collector_budget" for entry in row.get("episodes", [])))


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    row = metrics["rounds"][0]
    episodes = row["episodes"]
    wins = sum(episode.get("outcome") in {"battle_victory", "act_clear", "victory"} for episode in episodes)
    hp = [episode["hp_cost"] for episode in episodes if episode.get("hp_cost") is not None]
    return {"wins": wins, "episodes": len(episodes), "faults": len(row.get("faults", [])),
            "mean_hp_cost": sum(hp) / len(hp) if hp else None,
            "mean_floor": sum(episode.get("floor") or 0 for episode in episodes) / max(1, len(episodes))}


def select_health(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = results["reference"]
    eligible = []
    reasons = {}
    tradeoffs = {}
    for name, result in results.items():
        if name == "reference":
            continue
        reason = []
        for cohort in ("act1_48", "expanded_hard32", "composite_act1_24"):
            if result[cohort]["faults"] > reference[cohort]["faults"]:
                reason.append(cohort + "_faults_increased")
        battle_cohorts = ("act1_48", "expanded_hard32")
        extra_losses = sum(reference[cohort]["wins"] - result[cohort]["wins"] for cohort in battle_cohorts)
        if extra_losses > 3:
            reason.append("more_than_three_extra_battle_losses")
        if result["composite_act1_24"]["wins"] < reference["composite_act1_24"]["wins"]:
            reason.append("composite_wins_regressed")
        def battle_hp(subject):
            count = sum(subject[cohort]["episodes"] for cohort in battle_cohorts)
            if any(subject[cohort]["mean_hp_cost"] is None for cohort in battle_cohorts):
                return None
            return sum(subject[cohort]["mean_hp_cost"] * subject[cohort]["episodes"] for cohort in battle_cohorts) / max(1, count)
        candidate_hp, reference_hp = battle_hp(result), battle_hp(reference)
        if candidate_hp is None or reference_hp is None or candidate_hp >= reference_hp:
            reason.append("mean_battle_hp_cost_not_improved")
        tradeoffs[name] = {"extra_battle_losses_of_80": extra_losses,
                          "mean_battle_hp_cost_difference": None if candidate_hp is None or reference_hp is None else candidate_hp-reference_hp,
                          "composite_clear_difference_of_24": result["composite_act1_24"]["wins"]-reference["composite_act1_24"]["wins"]}
        reasons[name] = reason
        if not reason:
            full = result["composite_act1_24"]
            rank = (full["wins"], full["mean_floor"], -(full["mean_hp_cost"] or 0.), -candidate_hp)
            eligible.append((rank, name))
    selected = max(eligible)[1] if eligible else "reference"
    return {"selected_condition": selected, "eligible": [name for _, name in eligible],
            "exclusion_reasons": reasons,
            "tradeoffs": tradeoffs,
            "criterion": "at most three additional battle losses of 80; improve mean battle HP; preserve composite clears and fault coverage; rank by composite outcomes",
            "inference_limit": "Exploratory selection on one 24-world development panel; no significance claim or final-test tuning."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="assemble/pin the plan without launching any training")
    args = parser.parse_args()
    parent_journal = args.campaign / "campaign.json"
    parent = json.loads(parent_journal.read_text())
    ooc_source = Path(parent["selected"]["shared_bc"])
    source_sha, ooc_sha = identity(args.source), identity(ooc_source)
    destination = args.campaign / ("health_ablation_" + source_sha[:12])
    destination.mkdir(parents=True, exist_ok=True)
    source, ooc = destination / "reference.pt", destination / "ooc_reference.pt"
    for original, snapshot, digest in ((args.source, source, source_sha), (ooc_source, ooc, ooc_sha)):
        if snapshot.exists() and identity(snapshot) != digest:
            raise ValueError("existing health snapshot differs from pinned source")
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".tmp")
            shutil.copyfile(original, temporary)
            temporary.replace(snapshot)
    rows = balanced_train_rows(PANELS)
    bank = destination / "balanced_train_64_per_act.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if bank.exists() and bank.read_text() != content:
        raise ValueError("health train cohort changed on resume")
    bank.write_text(content)
    module = load_campaign_module()
    study = module.Campaign(destination)
    checkpoints = {"reference": source, "hp0.25": destination / "health_hp0.25.pt", "hp1.0": destination / "health_hp1.0.pt"}
    plan = {"source_checkpoint": str(args.source), "source_sha256": source_sha,
            "ooc_checkpoint": str(ooc_source), "ooc_sha256": ooc_sha, "train_bank_sha256": identity(bank),
            "train_rows": len(rows), "per_act_worlds": 64, "optimizer_seed": 11,
            "seconds_per_condition": 180, "weights": [.25, 1.], "deferred_bootstrap": True,
            "phase_progress_preserved": True, "created": time.time()}
    plan["source_code_identity"] = {str(path.relative_to(ROOT)): identity(path)
        for path in [Path(__file__), ROOT / "scripts/train_nonllm.py", ROOT / "src/sts_policy/model_agent.py",
                     ROOT / "src/sts_policy/rollout_training.py", ROOT / "src/sts_policy/workers.py"]}
    atomic(destination / "plan.json", plan)
    if args.prepare_only:
        preview = {"plan": plan, "destination": str(destination),
                   "training_commands": [training_command(source, checkpoints[f"hp{weight}"], bank, weight) for weight in (.25, 1.)],
                   "evaluation_cohorts": {cohort: len(command_tasks(evaluation_command(source, ooc, destination/(cohort+".metrics.json"), cohort)))
                       for cohort in ("act1_48", "expanded_hard32", "composite_act1_24")}}
        print(json.dumps(preview, sort_keys=True), flush=True)
        return
    for weight in (.25, 1.):
        name = f"health_hp{weight}"
        result = study.run(name, training_command(source, checkpoints[f"hp{weight}"], bank, weight), 180)
        if result is None or not checkpoints[f"hp{weight}"].exists():
            study.state["status"] = "health_training_incomplete"
            study.save()
            raise RuntimeError("health training failed or exhausted its deadline")
    results = {}
    for condition, checkpoint in checkpoints.items():
        results[condition] = {}
        for cohort in ("act1_48", "expanded_hard32", "composite_act1_24"):
            name = f"{condition}_{cohort}"
            metrics_path = destination / (name + ".metrics.json")
            command = evaluation_command(checkpoint, ooc, metrics_path, cohort)
            expected = [task_identity(task) for task in command_tasks(command)]
            metrics = study.run(name, command, 90)
            if not completed_matching(metrics, expected):
                study.state["status"] = "health_evaluation_incomplete"
                study.save()
                raise RuntimeError(f"health panel {name} incomplete or mismatched")
            results[condition][cohort] = summarize(metrics)
    selection = select_health(results)
    selection.update(results=results, checkpoints={key: str(value) for key, value in checkpoints.items()}, plan=plan)
    atomic(destination / "selection.json", selection)
    # Read fresh and merge only the requested result. Never rewrite main progress
    # from the stale snapshot used to establish the reference OOC policy.
    latest = json.loads(parent_journal.read_text())
    latest.setdefault("selected", {})["combat_health"] = str(checkpoints[selection["selected_condition"]])
    latest["health_ablation"] = {"selection": str(destination / "selection.json"), "condition": selection["selected_condition"]}
    atomic(parent_journal, latest)
    study.state["status"] = "health_complete"
    study.save()
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
