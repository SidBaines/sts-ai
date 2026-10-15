"""Tiny receipted world fixtures; never imports or runs native code."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from sts_rl.fresh_collection import summarize
from sts_rl.fresh_study import (DEAD_FILTER_ID, FILTER_ID, INFERENCE, PROTOCOL,
                                PUBLIC_CONTEXT_ID, atomic_json)
from sts_rl.recording import file_identity


def write_cohort(directory, worlds, config=None, archive=b"frozen source fixture", policy_changes=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {"mode": "eval", "bank": True, "checkpoint": "champion", "combat": None, "ooc": None,
              "samples": 24, "simulations": 1000, "teacher_labels": True,
              "teacher_probabilities": {"BOSS2": 1.}, "self_consistency_every": 100,
              "max_labels": 0, "timeout": 30., "workers": 1, "deadline_utc": None,
              "out": str(directory), "world_start": min(r["world_seed"] for r in worlds),
              "n": len(worlds), **(config or {})}
    policy = {"protocol": PROTOCOL, "filter_ids": [FILTER_ID, DEAD_FILTER_ID],
              "inference": INFERENCE, "public_context_id": PUBLIC_CONTEXT_ID,
              "champion": {"path": "champion", "sha256": "champion", "bytes": 1}, "combat": None, "ooc": None,
              **(policy_changes or {})}
    atomic_json(directory / "archive.json", {"fixture": archive.hex()})
    atomic_json(directory / "launch.json", {"effective_config": config, "parent_checkpoint": policy["champion"],
                "source_archive": file_identity(directory / "archive.json"),
                "native": worlds[0]["expected_provenance"]["native_provenance"], "runtime": {"python": "frozen"}})
    source = {"protocol": PROTOCOL, "config": config, "policy": policy,
              "launch": file_identity(directory / "launch.json")}
    atomic_json(directory / "receipt.json", source)
    records, sampling = [], Counter()
    for original in worlds:
        row = deepcopy(original)
        for part in [row, *row.get("labels", []), *row.get("bank", [])]:
            part["source"] = source
        path = directory / "worlds" / f"{row['world_seed']}.json.gz"
        atomic_json(path, row, compressed=True)
        records.append({**{k: v for k, v in row.items() if k not in {"labels", "bank", "source"}},
                        "shard": str(path.relative_to(directory)), "shard_identity": file_identity(path)})
        sampling.update(row["sampling"])
    summary = summarize(records, config["n"], sampling, sum(v for k, v in sampling.items() if k.endswith(":labelled")))
    summary.update(event="done", status="completed", stop_reasons=[])
    atomic_json(directory / "summary.json", summary)
    fields = {k: summary[k] for k in ("event", "status", "cohort_complete", "confidence_claims_allowed",
                                     "completed", "faults", "cuts", "censored", "unstarted", "stop_reasons")}
    atomic_json(directory / "completion.json", {"protocol": PROTOCOL, **fields, "source": source,
                                                "summary": file_identity(directory / "summary.json")})
    return directory


def reseal_summary(directory, summary):
    """Keep outer hashes valid when deliberately exercising deeper rejection."""
    import json
    atomic_json(directory / "summary.json", summary)
    completion = json.loads((directory / "completion.json").read_text())
    completion["summary"] = file_identity(directory / "summary.json")
    atomic_json(directory / "completion.json", completion)
