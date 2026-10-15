"""Strict completed seedgrid validation and lossless process-shard merging.

No intersection pairing, partial banks, unreceipted JSONL, or fault-to-loss
conversion. Read one world at a time; retain only summary records and paths.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil

from sts_policy.curriculum import public_hash
from sts_rl.fresh_study import (DEAD_FILTER_ID, FILTER_ID, INFERENCE, PROTOCOL,
                                PUBLIC_CONTEXT_ID, atomic_json, validate_world_range)
from sts_rl.recording import file_identity

SHARD_FIELDS = frozenset({"out", "n", "world_start", "workers", "deadline_utc"})
INVALID = ("fault", "cut", "ub", "censored", "programming_error", "truncated", "undefined_behavior_evoked")


def read_shard(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def _json(path):
    return json.loads(Path(path).read_text())


def _verify_file(path, identity):
    if not isinstance(identity, dict) or file_identity(path) != identity:
        raise ValueError(f"File identity/hash mismatch: {path}")


def _science(source):
    if source["protocol"] != PROTOCOL:
        raise ValueError("Unknown cohort protocol")
    config, policy = source["config"], source["policy"]
    validate_world_range(config["world_start"], config["n"])
    if config["mode"] not in {"eval", "collect"}:
        raise ValueError("Not a collection/eval cohort")
    if (policy["protocol"] != PROTOCOL or policy["filter_ids"] != [FILTER_ID, DEAD_FILTER_ID] or
            policy["inference"] != INFERENCE or policy["public_context_id"] != PUBLIC_CONTEXT_ID):
        raise ValueError("Cohort policy/filter identity mismatch")
    _verify_file(source["launch"]["path"], source["launch"])
    launch = _json(source["launch"]["path"])
    if launch["effective_config"] != config or launch["parent_checkpoint"] != policy["champion"]:
        raise ValueError("Cohort launch config/policy mismatch")
    archive = launch["source_archive"]
    _verify_file(archive["path"], archive)
    return {"protocol": PROTOCOL, "config": {k: v for k, v in config.items() if k not in SHARD_FIELDS},
            "policy": policy, "source_archive": {k: archive[k] for k in ("bytes", "sha256")},
            "native": launch["native"], "runtime": launch["runtime"]}


def _complete(value, n):
    if (value["protocol"] != PROTOCOL or value["event"] != "done" or value["status"] != "completed" or
            value["cohort_complete"] is not True or value["confidence_claims_allowed"] is not True or
            value["completed"] != n or any(value[k] != 0 for k in ("faults", "cuts", "censored", "unstarted")) or
            value.get("programming_errors", 0) or value.get("programming_error") or value["stop_reasons"]):
        raise ValueError("Require an exact completed fault-free cohort receipt")


def _directory(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Require completed receipted cohort directories; unreceipted JSONL/shards rejected")
    source = _json(directory / "receipt.json")
    config = source["config"]
    validate_world_range(config["world_start"], config["n"])
    completion = _json(directory / "completion.json")
    _complete(completion, config["n"])
    if completion["source"] != source:
        raise ValueError("Completion/source receipt mismatch")
    _verify_file(directory / "summary.json", completion["summary"])
    summary = _json(directory / "summary.json")
    _complete(summary, config["n"])
    sources = source.get("cohort_sources", [source])
    if not sources:
        raise ValueError("Missing cohort sources")
    if "cohort_sources" in source:
        if not source.get("cohort_inputs"):
            raise ValueError("Missing merged input receipts")
        for identity in source["cohort_inputs"]:
            _verify_file(identity["path"], identity)
    science = _science(sources[0])
    for item in sources[1:]:
        if _science(item) != science:
            raise ValueError("Science identity differs across source/config/provenance")
    if ({k: v for k, v in config.items() if k not in SHARD_FIELDS} != science["config"] or
            source["policy"] != science["policy"] or source["protocol"] != PROTOCOL):
        raise ValueError("Merged config/policy identity mismatch")
    source_worlds = {}
    for item in sources:
        c = item["config"]
        for seed in range(c["world_start"], c["world_start"] + c["n"]):
            if seed in source_worlds:
                raise ValueError("Duplicate requested seedgrid source")
            source_worlds[seed] = item
    expected = set(range(config["world_start"], config["world_start"] + config["n"]))
    if set(source_worlds) != expected:
        raise ValueError("Source requested seedgrid differs")
    records, paths, seen, expected_provenance = [], [], set(), None
    for brief in summary["records"]:
        seed = brief["world_seed"]
        if seed in seen or seed not in expected:
            raise ValueError("Duplicate/unexpected world in requested seedgrid")
        seen.add(seed)
        path = directory / "worlds" / f"{seed}.json.gz"
        if brief["shard"] != str(path.relative_to(directory)):
            raise ValueError("Unexpected world shard path")
        _verify_file(path, brief["shard_identity"])
        shard = read_shard(path)
        stripped = {k: v for k, v in shard.items() if k not in {"labels", "bank", "source"}}
        if stripped != {k: v for k, v in brief.items() if k not in {"shard", "shard_identity"}}:
            raise ValueError("World shard/summary status or identity mismatch")
        if (shard["status"] != "completed" or shard["protocol"] != PROTOCOL or
                shard["source"] != source_worlds[seed] or shard["task"]["seed"] != seed or
                shard["task"]["expected_provenance"] != shard["expected_provenance"] or
                not isinstance(shard["win"], bool) or any(shard.get(k) for k in INVALID) or
                public_hash({"actions": shard["action_prefix"]}) != shard["action_hash"]):
            raise ValueError("Invalid completed world/source/action identity; faults cannot become losses")
        provenance = shard["expected_provenance"]
        if (provenance["environment_version"] != "0.1.5" or
                provenance["native_provenance"] != science["native"] or
                (expected_provenance is not None and provenance != expected_provenance)):
            raise ValueError("Nonuniform environment/native/provenance")
        expected_provenance = provenance
        for member in [*shard["labels"], *shard["bank"]]:
            prefix = member["prefix"]
            if (member["world_seed"] != seed or member["source"] != shard["source"] or
                    member["expected_provenance"] != provenance or
                    prefix != shard["action_prefix"][:len(prefix)] or
                    len(prefix) >= len(shard["action_prefix"]) or
                    public_hash(member["observation"]) != member["public_hash"]):
                raise ValueError("Label/bank world/source/provenance/public hash/prefix identity mismatch")
        records.append(brief)
        paths.append(path)
    if seen != expected or set((directory / "worlds").glob("*")) != set(paths):
        raise ValueError("Missing/unexpected shards in exact requested seedgrid")
    # A completed cohort cannot contain silently ignored quarantined attempts.
    if any((directory / "quarantine").glob("*")):
        raise ValueError("Completed cohort contains quarantine/fault attempts")
    science = {**science, "expected_provenance": expected_provenance}
    if source.get("science_identity", science) != science:
        raise ValueError("Merged science identity mismatch")
    return {"config": config, "science_identity": science,
            "sources": sources, "records": records, "shards": paths,
            "inputs": [directory / name for name in ("receipt.json", "completion.json", "summary.json")]}


def validate_cohorts(paths, *, world_start=None, n=None):
    """Validate exact contiguous union, optionally against a caller's seedgrid.

    Only the five SHARD_FIELDS may differ. All other config keys, source archive
    SHA, environment/native provenance, policy/filter identities and runtime agree.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    try:
        cohorts = [_directory(path) for path in paths]
        if not cohorts:
            raise ValueError("No cohort directories")
        identity = cohorts[0]["science_identity"]
        if any(c["science_identity"] != identity for c in cohorts):
            raise ValueError("Science identity differs across source/config/provenance")
        records = [r for c in cohorts for r in c["records"]]
        actual = [r["world_seed"] for r in records]
        if (world_start is None) != (n is None):
            raise ValueError("Supply both exact world_start and n")
        if world_start is None:
            world_start, n = min(actual), max(actual) - min(actual) + 1
        validate_world_range(world_start, n)
        if len(actual) != len(set(actual)) or set(actual) != set(range(world_start, world_start + n)):
            raise ValueError("Missing/duplicate/unexpected worlds in exact contiguous requested seedgrid")
        indexed = sorted(zip(actual, [p for c in cohorts for p in c["shards"]], records))
        return {"config": {**cohorts[0]["config"], "world_start": world_start, "n": n},
                "science_identity": identity, "sources": [s for c in cohorts for s in c["sources"]],
                "records": [r for _, _, r in indexed], "shards": [p for _, p, _ in indexed],
                "inputs": [p for c in cohorts for p in c["inputs"]]}
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing/invalid completed cohort receipt or shard: {exc}") from exc


def merge_cohorts(paths, out, *, world_start, n):
    """Write a fresh merged summary+receipt and byte-identical world shards.

    summary.json records can go directly to fresh_statistics/fresh_compare games.
    Original launches/source archive SHAs and policy/native provenance survive.
    """
    from sts_rl.fresh_collection import summarize
    cohort = validate_cohorts(paths, world_start=world_start, n=n)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Merged output must be empty")
    with (out / ".fresh-run-claim").open("x") as claim:
        claim.write(PROTOCOL + " merge\n")
    source = {"protocol": PROTOCOL, "config": {**cohort["config"], "out": str(out)},
              "policy": cohort["science_identity"]["policy"], "cohort_sources": cohort["sources"],
              "cohort_inputs": [file_identity(p) for p in cohort["inputs"]],
              "science_identity": cohort["science_identity"]}
    atomic_json(out / "receipt.json", source)
    records, sampling, labels = [], Counter(), 0
    for path, record in zip(cohort["shards"], cohort["records"]):
        target = out / "worlds" / path.name
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(path, temporary)
        temporary.replace(target)
        identity = file_identity(target)
        if {k: identity[k] for k in ("sha256", "bytes")} != {k: record["shard_identity"][k] for k in ("sha256", "bytes")}:
            raise ValueError("World shard changed during merge")
        records.append({**record, "shard": str(target.relative_to(out)), "shard_identity": identity})
        sampling.update(record["sampling"])
        labels += sum(v for k, v in record["sampling"].items() if k.endswith(":labelled"))
    summary = summarize(records, n, sampling, labels)
    summary.update(event="done", status="completed", stop_reasons=[], science_identity=cohort["science_identity"])
    atomic_json(out / "summary.json", summary)
    fields = {k: summary[k] for k in ("event", "status", "cohort_complete", "confidence_claims_allowed",
                                     "completed", "faults", "cuts", "censored", "unstarted", "stop_reasons")}
    atomic_json(out / "completion.json", {"protocol": PROTOCOL, **fields, "source": source,
                                         "summary": file_identity(out / "summary.json")})
    validate_cohorts(out, world_start=world_start, n=n)
    return summary
