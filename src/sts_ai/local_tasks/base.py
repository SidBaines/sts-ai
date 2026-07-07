from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol


class LocalTask(Protocol):
    task_id: str

    def build_manifest(
        self,
        source_rollout_dir: Path,
        *,
        holdout_mod: int = 4,
        holdout_remainder: int = 0,
    ) -> dict[str, Any]:
        ...

    def completion_reason(self, summary: dict[str, Any]) -> str | None:
        ...

    def metrics_from_episode(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
        stopped_reason: str,
        window: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SourceRollout:
    stem: str
    jsonl_path: Path
    meta_path: Path | None


def parse_rollout_stem(stem: str) -> tuple[int, int]:
    if not stem.startswith("seed_"):
        raise ValueError(f"invalid rollout stem: {stem!r}")
    raw_seed, raw_rollout = stem[len("seed_") :].rsplit("_r", 1)
    return int(raw_seed), int(raw_rollout)


def rollout_stem(world_seed: int, rollout_index: int) -> str:
    return f"seed_{world_seed}_r{rollout_index}"


def discover_source_rollouts(source_rollout_dir: Path) -> list[SourceRollout]:
    source_rollout_dir = Path(source_rollout_dir)
    rollouts: list[SourceRollout] = []
    for jsonl_path in sorted(source_rollout_dir.glob("seed_*_r*.jsonl")):
        stem = jsonl_path.stem
        meta_path = jsonl_path.with_suffix(".meta.json")
        rollouts.append(
            SourceRollout(
                stem=stem,
                jsonl_path=jsonl_path,
                meta_path=meta_path if meta_path.exists() else None,
            )
        )
    return sorted(rollouts, key=lambda r: parse_rollout_stem(r.stem))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if "task_id" not in manifest or "windows" not in manifest:
        raise ValueError(f"{path} is not a local-task manifest")
    return manifest


def source_rollout_dir(manifest: dict[str, Any]) -> Path:
    return Path(str(manifest["source_rollout_dir"]))


def source_jsonl_path(manifest: dict[str, Any], window: dict[str, Any]) -> Path:
    return source_rollout_dir(manifest) / f"{window['source_stem']}.jsonl"


def source_meta_path(manifest: dict[str, Any], window: dict[str, Any]) -> Path:
    return source_rollout_dir(manifest) / f"{window['source_stem']}.meta.json"


def source_records_for_window(
    manifest: dict[str, Any],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    records = load_jsonl(source_jsonl_path(manifest, window))
    return records[int(window["start_index"]) : int(window["end_index"]) + 1]


def source_meta_for_window(
    manifest: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    path = source_meta_path(manifest, window)
    return load_json(path) if path.exists() else {}


def reasoning_mode_from_meta(meta: dict[str, Any]) -> str:
    extra = meta.get("extra") or {}
    agent_config = extra.get("agent_config") or {}
    mode = agent_config.get("reasoning_mode")
    return "none" if mode is None else str(mode)


def split_for_seed(
    world_seed: int,
    *,
    holdout_mod: int,
    holdout_remainder: int,
) -> str:
    if holdout_mod < 2:
        raise ValueError("holdout_mod must be >= 2")
    return "holdout" if world_seed % holdout_mod == holdout_remainder else "train"


def label_counts(windows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(window["label"]) for window in windows))


def split_counts(windows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(window["split"]) for window in windows))


def _agent_dict(record: dict[str, Any]) -> dict[str, Any]:
    agent = record.get("agent")
    return agent if isinstance(agent, dict) else {}


def skip_reason(record: dict[str, Any]) -> str | None:
    agent = _agent_dict(record)
    if record.get("action_executed", True) is False:
        return "action_not_executed"
    if agent.get("valid", True) is False:
        return "agent_invalid"
    if int(agent.get("retries") or 0) > 0:
        return "agent_retried"
    return None


def selected_action_description(record: dict[str, Any]) -> str:
    selected = record.get("selected_action")
    if not isinstance(selected, dict):
        return ""
    return str(selected.get("description", ""))
