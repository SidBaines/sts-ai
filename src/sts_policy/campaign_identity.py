"""Scientific command/input identities; output artifacts are never inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

CACHE_VERSION = "campaign_condition_v1"
INPUT_FLAGS = {"--tasks", "--teacher", "--dev-teacher", "--checkpoint", "--combat-checkpoint", "--ooc-checkpoint"}
OUTPUT_FLAGS = {"--output", "--metrics"}


def semantic_arguments(command: Sequence[Any]) -> list[str]:
    """Conservatively retain all knobs, including budget/device/runtime flags."""
    tokens = list(map(str, command))
    output = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag = token.split("=", 1)[0]
        if flag in OUTPUT_FLAGS:
            index += 1 if "=" in token else 2
            continue
        output.append(token)
        index += 1
    return output


def input_paths(command: Sequence[Any]) -> dict[str, Path]:
    """Argparse's last occurrence wins for repeated input flags."""
    tokens = list(map(str, command))
    paths = {}
    for index, token in enumerate(tokens):
        flag = token.split("=", 1)[0]
        if flag not in INPUT_FLAGS:
            continue
        value = token.split("=", 1)[1] if "=" in token else tokens[index+1]
        paths[flag] = Path(value).resolve()
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_signature(command: Sequence[Any]) -> dict[str, Any]:
    inputs = {flag: {"path": str(path), "sha256": file_sha256(path)}
              for flag, path in sorted(input_paths(command).items())}
    payload = {"version": CACHE_VERSION, "arguments": semantic_arguments(command), "inputs": inputs}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "digest": digest}


def legacy_cache_valid(record: dict[str, Any], command: Sequence[Any]) -> bool:
    """Pin only matching old conditions whose effective inputs predate launch.

    Source hashes need not match: separately validated performance changes may
    preserve semantics. This check cannot recover missing historical file hashes.
    Once pinned, future comparisons must use condition_signature's input hashes.
    """
    if semantic_arguments(record.get("command", [])) != semantic_arguments(command):
        return False
    started = record.get("started")
    if not isinstance(started, (int, float)):
        return False
    try:
        return all(path.is_file() and path.stat().st_mtime <= started
                   for path in input_paths(command).values())
    except (OSError, IndexError):
        return False
