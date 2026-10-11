"""Fresh native teacher collection with atomic world journals and replayable tasks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

from sts_env.env import SimulatorError
from sts_policy.curriculum import TASK_VERSION, TaskSpec, public_hash, validate_provenance, world_split
from sts_policy.workers import CurriculumWorker

DATA_VERSION = "sts_policy_teacher_v1"


def adapter_identity() -> str:
    digest = hashlib.sha256()
    for name in ("curriculum.py", "workers.py", "data.py", "fast_observations.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def collect_world(worker: CurriculumWorker, seed: int, *, max_act: int = 1,
                  horizon: int = 1500, simulations: int = 50,
                  epsilon_ooc: float = 0.0, epsilon_combat: float = 0.0) -> dict[str, Any]:
    """On fault, discard the entire world's labels; keep explicit fault evidence."""
    if not 0 <= epsilon_ooc <= 1 or not 0 <= epsilon_combat <= 1:
        raise ValueError("epsilon must be in [0,1]")
    started = time.monotonic()
    collection_settings = {"max_act": max_act, "horizon": horizon, "simulations": simulations,
                           "epsilon_ooc": epsilon_ooc, "epsilon_combat": epsilon_combat}
    rng = random.Random(seed ^ 0x1F513)
    spec = TaskSpec(seed=seed, kind="run", horizon=horizon, max_act=max_act,
                    battle_simulations=simulations)
    records, battles = [], []
    prefix: list[int] = []
    ooc_previews: list[bool] = []
    previous_phase = ""
    info: dict[str, Any] = {}
    try:
        observation, info = worker.reset(spec)
        provenance = {"adapter_sha256": adapter_identity(), **{k: info[k] for k in ("native_provenance", "python_interface_sha256", "runtime_python_version", "environment_version",
                                         "observation_version") if k in info}}
        for step in range(horizon):
            phase = observation["phase"]
            if phase == "combat" and previous_phase != "combat":
                battle = TaskSpec(seed=seed, kind="battle", prefix=tuple(prefix), ooc_previews=tuple(ooc_previews),
                                  expected_public_hash=public_hash(observation), max_act=max_act,
                                  battle_simulations=simulations, seek_combat=False, expected_provenance=provenance)
                battles.append({**battle.to_dict(), "public_hash": public_hash(observation),
                                "floor": observation["state"]["floor"],
                                "enemy_names": [e["name"] for e in observation["state"]["combat"]["enemies"]],
                                "provenance": provenance})
            teacher = worker.teacher_action(simulations)
            epsilon = epsilon_combat if phase == "combat" else epsilon_ooc
            action = rng.randrange(len(observation["legal_actions"])) if rng.random() < epsilon else teacher
            records.append({"data_version": DATA_VERSION, "observation": observation,
                            "teacher_index": teacher, "executed_index": action,
                            "world_seed": seed, "split": world_split(seed), "phase": phase,
                            "step": step, "teacher_kind": "native_privileged_search" if phase == "combat" else "native_scripted_ooc",
                            "simulations": simulations if phase == "combat" else None})
            previous_phase = phase
            observation, reward, terminated, truncated, info = worker.step(action)
            prefix.append(action)
            ooc_previews.append(phase == "out_of_combat")
            if terminated or truncated:
                break
        return {"data_version": DATA_VERSION, "seed": seed, "split": world_split(seed),
                "status": "ok", "seconds": time.monotonic() - started,
                "records": records, "battles": battles, "provenance": provenance,
                "settings": collection_settings,
                "result": {**{k: info.get(k) for k in ("task_outcome", "termination_reason", "hp_cost", "outcome", "task_steps")},
                           "final_act": observation["state"].get("act"),
                           "final_floor": observation["state"].get("floor"),
                           "final_hp": observation["state"].get("cur_hp")}}
    except SimulatorError as exc:
        return {"data_version": DATA_VERSION, "seed": seed, "split": world_split(seed),
                "status": "simulator_fault", "seconds": time.monotonic() - started,
                "settings": collection_settings, "adapter_sha256": adapter_identity(),
                "error_type": type(exc).__name__, "error": str(exc),
                "discarded_records": len(records), "records": [], "battles": [],
                "failed_prefix": prefix, "last_info": info}


def collect_dataset(output: Path, seeds: list[int], *, workers: int = 4,
                    timeout: float = 10.0, deadline_seconds: float | None = None,
                    fast_observations: bool = False,
                    **settings: Any) -> dict[str, Any]:
    """Resume complete worlds; assigned slices keep every worker persistent."""
    if len(seeds) != len(set(seeds)):
        raise ValueError("world seeds must be unique")
    output.mkdir(parents=True, exist_ok=True)
    journals = output / "worlds"
    journals.mkdir(exist_ok=True)
    started = time.monotonic()
    expected_settings = {"max_act": 1, "horizon": 1500, "simulations": 50,
                         "epsilon_ooc": 0.0, "epsilon_combat": 0.0, **settings}
    identity = adapter_identity()
    if not seeds:
        raise ValueError("at least one seed is required")
    with CurriculumWorker({"fast_observations": fast_observations}, timeout=timeout) as probe:
        _, live_info = probe.reset(TaskSpec(seed=seeds[0], kind="run", max_act=expected_settings["max_act"]))
    for seed in seeds:
        journal = journals / f"{seed}.json"
        if journal.exists():
            row = json.loads(journal.read_text())
            saved_identity = row.get("provenance", {}).get("adapter_sha256", row.get("adapter_sha256"))
            if row.get("status") == "ok":
                validate_provenance(row.get("provenance", {}), live_info)
            if row.get("settings") != expected_settings or saved_identity != identity:
                raise ValueError(f"World {seed} journal settings/code mismatch; use a new output directory")
    pending = [seed for seed in seeds if not (journals / f"{seed}.json").exists()]
    slices = [pending[i::workers] for i in range(workers)]

    def lane(lane_seeds: list[int]) -> None:
        with CurriculumWorker({"fast_observations": fast_observations}, timeout=timeout) as worker:
            for seed in lane_seeds:
                if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
                    break
                result = collect_world(worker, seed, **settings)
                atomic_json(journals / f"{seed}.json", result)
                print(json.dumps({"seed": seed, "status": result["status"],
                                  "steps": len(result["records"]), "battles": len(result["battles"]),
                                  "seconds": round(result["seconds"], 3)}), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(lane, lane_seeds) for lane_seeds in slices]
        for future in as_completed(futures):
            future.result()
    def all_rows() -> Iterable[dict[str, Any]]:
        for seed in seeds:
            journal = journals / f"{seed}.json"
            if journal.exists():
                yield json.loads(journal.read_text())
    atomic_jsonl(output / "teacher.jsonl", (record for row in all_rows() for record in row["records"]))
    atomic_jsonl(output / "battles.jsonl", (record for row in all_rows() for record in row["battles"]))
    # Summary holds small counts rather than duplicating all observations in RAM.
    rows = [{"seed": row["seed"], "split": row["split"], "status": row["status"],
             "records": len(row["records"]), "battles": len(row["battles"])} for row in all_rows()]
    manifest = {"data_version": DATA_VERSION, "requested_worlds": len(seeds),
                "complete_worlds": len(rows), "valid_worlds": sum(r["status"] == "ok" for r in rows),
                "fault_worlds": sum(r["status"] == "simulator_fault" for r in rows),
                "records": sum(r["records"] for r in rows),
                "battles": sum(r["battles"] for r in rows),
                "seconds": time.monotonic() - started, "workers": workers, "settings": expected_settings,
                "fast_observations": fast_observations,
                "adapter_sha256": identity,
                "provenance": {k: live_info[k] for k in ("python_interface_sha256", "runtime_python_version", "environment_version", "observation_version", "native_provenance")},
                "seeds": [r["seed"] for r in rows],
                "split_counts": {split: sum(r["split"] == split for r in rows) for split in ("train", "dev", "final")}}
    atomic_json(output / "manifest.json", manifest)
    return manifest
