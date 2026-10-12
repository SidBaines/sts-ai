"""Scientific command/input identities; output artifacts are never inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

CACHE_VERSION = "campaign_condition_v2"
INPUT_FLAGS = {"--tasks", "--teacher", "--dev-teacher", "--checkpoint", "--combat-checkpoint", "--ooc-checkpoint"}
OUTPUT_FLAGS = {"--output", "--metrics"}


def semantic_arguments(command: Sequence[Any]) -> list[str]:
    """Canonical effective long options; positional and short-option order stays.

    Argparse consumes the last occurrence of a value option. Settings serialized
    with sorted JSON keys may reorder options without changing the experiment.
    Preserve their exact string values, including numeric formatting, and every
    option (except output destinations); only option order/redundancy normalizes.
    """
    tokens = list(map(str, command))
    positional = []
    options: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            positional.extend(tokens[index:])
            break
        if not token.startswith("--"):
            positional.append(token)
            # A short-option payload such as `python -c CODE` is positional to
            # this canonicalizer even if CODE happens to start with `--`.
            if token == "-c" and index + 1 < len(tokens):
                positional.append(tokens[index + 1])
                index += 1
            index += 1
            continue
        flag, separator, inline_value = token.partition("=")
        values = [inline_value] if separator else []
        if not separator and index + 1 < len(tokens):
            following = tokens[index + 1]
            numeric = False
            try:
                float(following)
                numeric = True
            except ValueError:
                pass
            if not following.startswith("-") or numeric:
                values = [following]
                index += 1
        if flag not in OUTPUT_FLAGS:
            # BooleanOptionalAction aliases share one argparse destination; the
            # last positive/negative choice must survive canonical sorting.
            if flag == "--goal-act-clear":
                options.pop("--no-goal-act-clear", None)
            elif flag == "--no-goal-act-clear":
                options.pop("--goal-act-clear", None)
            options[flag] = values
        index += 1
    return positional + [token for flag, values in sorted(options.items())
                         for token in [flag, *values]]


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
