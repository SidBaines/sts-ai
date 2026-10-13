"""Deterministic world-first sampling of correlated strategic starts."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Sequence

from .curriculum import world_split

SAMPLER_VERSION = "strategic_world_then_start_v1"


def sample_frontiers(rows: Sequence[dict[str, Any]], *, acts: Sequence[int]=(2,3),
                     per_act: int=2, seed: int=0, decision_family: str | None=None) -> list[dict[str, Any]]:
    if per_act<1 or not acts or len(set(acts))!=len(acts) or any(act not in (1,2,3) for act in acts):
        raise ValueError("positive counts and distinct supported acts required")
    if any(world_split(int(row["seed"]))!="train" for row in rows):
        raise ValueError("frontier sampling is TRAIN-only")
    selected=[]
    for act in acts:
        groups={}
        for row in rows:
            if row.get("start_act")!=act or (decision_family is not None and row.get("decision_family")!=decision_family):
                continue
            identity=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            groups.setdefault(int(row["seed"]),{})[identity]=row
        worlds=sorted(groups)
        random.Random(f"{SAMPLER_VERSION}:{seed}:{act}:worlds").shuffle(worlds)
        for world in worlds[:per_act]:
            candidates=[groups[world][key] for key in sorted(groups[world])]
            rng=random.Random(f"{SAMPLER_VERSION}:{seed}:{act}:{world}:start")
            selected.append(candidates[rng.randrange(len(candidates))])
    return selected
