"""Thinking-only budget control; the active thinking-off sampler stays frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

from scripts import sample_fight_bank as sample
from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


SCRIPT = Path(__file__).resolve()
LIMITS = (16384, 32768, 65536, 131072)
_base_result = sample.result
_base_validate = sample.validate


def cap_hit(record: dict[str, Any], limit: int) -> bool:
    agent = record.get("agent", {})
    tokens = int(agent.get("completion_tokens", 0))
    # Saved counts re-tokenize decoded text: allow a few stripped special tokens
    # only when the response is incomplete. A valid shorter reply is not a hit.
    from sts_ai.fight_analysis import thinking_truncated
    truncated = thinking_truncated(record)
    return truncated or tokens >= limit or (not agent.get("valid", False) and tokens >= limit - 8)


def observations(output: Path, maximum: int = 20) -> list[dict[str, Any]]:
    manifest = lab.read_json(output / "manifest.json")
    records = []
    for state_id in manifest["order"]:
        path = output / f"{state_id}_r0.jsonl"
        if not path.exists():
            continue
        text = path.read_text()
        lines = text.splitlines()
        if lines and not text.endswith("\n"):
            lines.pop()  # Writer may be in the middle of its final append.
        for line in lines:
            records.append(json.loads(line))
            if len(records) == maximum:
                return records
    return records


def assessment(records: list[dict[str, Any]], limit: int, *, finished: bool = False) -> dict[str, Any]:
    from sts_ai.fight_analysis import thinking_truncated
    records = records[:20]
    n = len(records)
    hits = sum(cap_hit(r, limit) for r in records)
    frequent = hits >= 2 and n >= 5 and hits / n >= .1
    checked = n >= 20 or finished
    action = "raise" if frequent else "accept" if checked else "observe"
    if action == "raise" and limit == LIMITS[-1]:
        action = "ceiling_reached"
    return {"max_tokens": limit, "generations": n, "cap_hits": hits,
            "cap_fraction": hits / n if n else None, "action": action,
            "thinking_truncated": sum(thinking_truncated(r) for r in records),
            "max_observed_tokens": max((r.get("agent", {}).get("completion_tokens", 0)
                                        for r in records), default=0)}


def validated(output: Path) -> tuple[dict, dict, dict]:
    manifest, selected, settings = _base_validate(output)
    policy = manifest["budget_policy"]
    if file_sha256(SCRIPT) != policy["runtime_sha256"]:
        raise ValueError("thinking budget runtime changed")
    if settings["thinking"] is not True or settings["max_tokens"] not in policy["limits"]:
        raise ValueError("thinking budget settings do not match policy")
    return manifest, selected, settings


def result(output: Path, state_id: str, seconds: float) -> dict:
    saved = _base_result(output, state_id, seconds)
    trace = output / f"{state_id}_r0.jsonl"
    meta = lab.read_json(trace.with_suffix(".meta.json"))
    limit = int(meta["max_tokens"])
    saved["generation_cap_hits"] = sum(cap_hit(r, limit) for r in lab.read_records(trace))
    saved["max_tokens"] = limit
    return saved


def save_configuration(output: Path, manifest: dict, settings: dict) -> None:
    sample.atomic(output / "settings.json", settings)
    manifest["settings_sha256"] = file_sha256(output / "settings.json")
    # A larger completion must not be mistaken for a stalled model before it
    # can finish. Allow at least 10 tokens/sec plus two minutes for overhead.
    manifest["stall_seconds"] = max(300, settings["max_tokens"] // 10 + 120)
    manifest["fight_seconds"] = max(3600, manifest["stall_seconds"] * 3)
    sample.atomic(output / "manifest.json", manifest)
    queued = lab.read_json(output / "queue.json")
    queued["manifest_sha256"] = file_sha256(output / "manifest.json")
    sample.atomic(output / "queue.json", queued)


def configure(output: Path) -> None:
    """Install the explicit new policy before any thinking trajectory exists."""
    if list(output.glob("*_r0.jsonl")) or list(output.glob("*.result.json")):
        raise ValueError("initial budget configuration requires an unstarted arm")
    manifest, _, settings = _base_validate(output)
    manifest["budget_policy"] = {
        "limits": list(LIMITS), "initial_max_tokens": LIMITS[0],
        "sample_generations": 20, "minimum_generations": 5,
        "minimum_hits": 2, "threshold_fraction": .1,
        "on_raise": "archive all attempt data and restart the full cohort with its original seeds",
        "runtime_sha256": file_sha256(SCRIPT),
    }
    save_configuration(output, manifest, {**settings, "max_tokens": LIMITS[0]})
    shutil.copy2(SCRIPT, output / "code_snapshot" / SCRIPT.name)
    sample.progress(output, manifest, "queued")
    sample.atomic(output / "budget_status.json", {"action": "observe", "max_tokens": LIMITS[0],
                                                  "generations": 0, "cap_hits": 0})


def raise_budget(output: Path, report: dict[str, Any]) -> None:
    """Called only after the prior supervisor/worker have fully exited."""
    manifest, _, settings = validated(output)
    old = settings["max_tokens"]
    new = LIMITS[LIMITS.index(old) + 1]
    archive = output / "budget_calibration" / f"max_tokens_{old}_{time.time_ns()}"
    archive.mkdir(parents=True)
    for name in ("settings.json", "manifest.json", "queue.json", "budget_status.json"):
        source = output / name
        if source.exists():
            shutil.copy2(source, archive / name)
    paths = set(output.glob("*_r0.*")) | set(output.glob("*.result.json"))
    paths |= {output / name for name in ("active.json", "progress.json", "report.json",
                                       "worker.log", "supervisor.log", "partials")}
    for source in paths:
        if source.exists():
            source.rename(archive / source.name)
    history_path = output / "budget_history.json"
    history = lab.read_json(history_path) if history_path.exists() else []
    history.append({**report, "old_max_tokens": old, "new_max_tokens": new,
                    "archive": str(archive), "timestamp": time.time()})
    sample.atomic(history_path, history)
    save_configuration(output, manifest, {**settings, "max_tokens": new})
    sample.progress(output, manifest, "queued")
    sample.atomic(output / "budget_status.json", {"action": "observe", "max_tokens": new,
                                                  "generations": 0, "cap_hits": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resume", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The shared supervisor invokes its worker and result/validation hooks through
    # module attributes. Bind those hooks ONLY in this dedicated thinking process;
    # no active worker, shared file, model implementation, or prompt is modified.
    sample.SCRIPT = SCRIPT
    sample.result = result
    sample.validate = validated
    if args.command == "resume":
        sample.supervise(args.output.resolve())
    else:
        sample.worker(args.output.resolve())


if __name__ == "__main__":
    main()
