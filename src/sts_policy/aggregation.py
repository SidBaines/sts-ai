"""Deterministic train-world reservoir aggregation for genuine DAgger mixing."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from sts_policy.curriculum import world_split
from sts_policy.data import atomic_json, atomic_jsonl

AGGREGATION_VERSION = "train_dagger_reservoir_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_teachers(sources: Iterable[str | Path], output: str | Path, *,
                       max_records: int = 30000, seed: int = 11,
                       phase: str = "combat") -> dict[str, Any]:
    """Stream a reproducible reservoir over initial and all prior learner labels.

    World identity, public menu label legality and fixed split are checked before
    admitting train rows. File order is canonical, duplicate sources are removed,
    and row-level source/line identities survive aggregation. A new materialized
    file records the exact source hashes; it never overwrites original labels.
    """
    if max_records < 1 or phase not in {"all", "combat", "ooc"}:
        raise ValueError("invalid reservoir size or phase")
    paths = sorted({Path(path).resolve() for path in sources}, key=str)
    if not paths:
        raise ValueError("at least one initial/learner teacher source is required")
    destination = Path(output)
    if destination.resolve() in paths:
        raise ValueError("aggregation output must not overwrite a source")
    identities = {str(path): file_sha256(path) for path in paths}
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    eligible = 0
    counters = {}
    for path in paths:
        counts = Counter()
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                counts["rows"] += 1
                row = json.loads(line)
                world = row.get("world_seed")
                if isinstance(world, bool) or not isinstance(world, int) or not 0 <= world < 2**63:
                    raise ValueError(f"Invalid world identity at {path}:{line_number}")
                split = world_split(world)
                if row.get("split", split) != split:
                    raise ValueError(f"World split mismatch at {path}:{line_number}")
                if split != "train":
                    counts[f"excluded_{split}"] += 1
                    continue
                observation = row.get("observation", {})
                wanted_phase = "combat" if phase == "combat" else "out_of_combat"
                if phase != "all" and observation.get("phase") != wanted_phase:
                    counts["excluded_phase"] += 1
                    continue
                label = row.get("teacher_index")
                menu = observation.get("legal_actions", [])
                if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(menu) or menu[label].get("index") != label:
                    raise ValueError(f"Illegal teacher menu label at {path}:{line_number}")
                counts["eligible_train_rows"] += 1
                eligible += 1
                candidate = {**row, "split": "train", "aggregation_version": AGGREGATION_VERSION,
                             "aggregation_source": {"path": str(path), "sha256": identities[str(path)],
                                                    "line": line_number}}
                if len(reservoir) < max_records:
                    reservoir.append(candidate)
                else:
                    slot = rng.randrange(eligible)
                    if slot < max_records:
                        reservoir[slot] = candidate
        counters[str(path)] = dict(counts)
    if not reservoir:
        raise ValueError("No legal train-world rows survived aggregation")
    atomic_jsonl(destination, reservoir)
    selected_sources = Counter(row["aggregation_source"]["path"] for row in reservoir)
    selected_worlds = {row["world_seed"] for row in reservoir}
    manifest = {"aggregation_version": AGGREGATION_VERSION, "sources": identities,
                "source_counts": counters, "selected_source_counts": dict(selected_sources),
                "eligible_records": eligible, "selected_records": len(reservoir),
                "max_records": max_records, "seed": seed, "phase": phase,
                "selected_worlds": sorted(selected_worlds), "selected_world_count": len(selected_worlds),
                "output_sha256": file_sha256(destination),
                "aggregation_source_sha256": file_sha256(Path(__file__))}
    atomic_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest
