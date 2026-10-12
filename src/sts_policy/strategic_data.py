"""Versioned public TRAIN buffers and explicit fresh/continuation task families."""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from .curriculum import TaskSpec, world_split
from .strategic_collection import StrategicTransition

BUFFER_VERSION = "strategic_public_buffer_v3_v1"


def continuation_tasks(bank: Path, *, act: int, count: int, seed: int) -> list[TaskSpec]:
    """One later start per source world/family; raw source decks stay distinguishable.

    The pinned simulator uses global floors 18..34 for Act 2 and 35..51 for Act 3.
    The observed actual start act is logged by collection and must be checked.
    Prefer late starts to obtain both successes and failures in the short pilot.
    """
    by_world = {}
    for line in bank.open():
        row = json.loads(line)
        if world_split(int(row["seed"])) != "train":
            raise ValueError("strategic continuation bank must contain TRAIN worlds only")
        floor = int(row["floor"])
        if not ((18 <= floor <= 34) if act == 2 else (35 <= floor <= 51) if act == 3 else False):
            continue
        if floor > int(by_world.get(row["seed"], {}).get("floor", -1)):
            by_world[row["seed"]] = row
    worlds = sorted(by_world)
    random.Random(seed).shuffle(worlds)
    return [replace(TaskSpec.from_dict(by_world[world]), kind="run", horizon=4096, max_act=3)
            for world in worlds[:count]]


def stratified_rows(rows: Sequence[StrategicTransition], *, per_group: int = 128, seed: int = 0) -> list[StrategicTransition]:
    """Reservoir by public phase/act/screen/choice; no DEV/final examples accepted."""
    rng = random.Random(seed)
    groups, seen = {}, {}
    for row in rows:
        if world_split(row.world_seed) != "train":
            raise ValueError("strategic gradient/rehearsal buffer must be TRAIN-only")
        if row.completed_return is None:
            continue
        state = row.observation["state"]
        key = (row.observation["phase"], state.get("act"), state.get("screen_state"), row.actor_choice)
        seen[key] = seen.get(key, 0)+1
        group = groups.setdefault(key, [])
        if len(group) < per_group:
            group.append(row)
        else:
            position = rng.randrange(seen[key])
            if position < per_group:
                group[position] = row
    selected = [row for key in sorted(groups, key=str) for row in groups[key]]
    rng.shuffle(selected)
    return selected


def write_buffer(path: Path, rows: Sequence[StrategicTransition], metadata: dict) -> str:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("w") as output:
        header = json.dumps({"buffer_version": BUFFER_VERSION, "metadata": metadata}, sort_keys=True)+"\n"
        output.write(header)
        digest.update(header.encode())
        for row in rows:
            if world_split(row.world_seed) != "train":
                raise ValueError("public training buffer cannot include DEV/final worlds")
            record = {f.name: getattr(row, f.name) for f in fields(row) if f.name != "encoded"}
            if record["behavior_distribution"] is not None:
                record["behavior_distribution"] = record["behavior_distribution"].tolist()
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))+"\n"
            output.write(line)
            digest.update(line.encode())
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def read_buffer(path: Path) -> tuple[list[StrategicTransition], dict]:
    rows = []
    with path.open() as source:
        header = json.loads(next(source))
        if header.get("buffer_version") != BUFFER_VERSION:
            raise ValueError("unsupported strategic public buffer")
        for line in source:
            record = json.loads(line)
            if world_split(int(record["world_seed"])) != "train":
                raise ValueError("strategic public buffer is not TRAIN-only")
            if record["behavior_distribution"] is not None:
                record["behavior_distribution"] = np.asarray(record["behavior_distribution"], np.float64)
            rows.append(StrategicTransition(encoded=None, **record))
    return rows, header["metadata"]
