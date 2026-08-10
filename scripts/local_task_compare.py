#!/usr/bin/env python
"""Strictly compare two matched local-task evaluation arms."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from sts_ai.eval_stats import bootstrap_ci, sign_test
from sts_ai.local_tasks import base as local_task_base
from sts_ai.local_tasks.runner import window_rollout_indices, windows_for_split
from sts_ai.provenance import file_sha256
from sts_ai.seeding import derive_policy_seed


ALLOWED_INTERVENTIONS = ("adapter_path", "combat_observation")
REQUIRED_INTERFACE_DIGESTS = (
    "prompt_probe_sha256",
    "chat_template_probe_hash",
    "python_serializer_sha256",
    "glossary_sha256",
    "prompting_sha256",
    "simulator_patch_sha256",
    "simulator_binary_sha256",
)
_META_NAME_RE = re.compile(r"^seed_(?P<world_seed>\d+)_r(?P<rollout_index>\d+)\.meta\.json$")


def _load_metas(root: Path, arm: str) -> list[dict[str, Any]]:
    paths = sorted(Path(root).rglob("seed_*_r*.meta.json"))
    if not paths:
        raise ValueError(f"{arm} arm contains no rollout metadata under {root}")
    metas: list[dict[str, Any]] = []
    for path in paths:
        match = _META_NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"{arm} has malformed rollout meta filename: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{arm} rollout meta is not a JSON object: {path}")
        filename_identity = (
            int(match["world_seed"]),
            int(match["rollout_index"]),
        )
        meta_identity = (value.get("world_seed"), value.get("rollout_index"))
        if meta_identity != filename_identity:
            raise ValueError(
                f"{arm} rollout identity disagrees with filename {path.name}: "
                f"meta={meta_identity!r}, filename={filename_identity!r}"
            )
        metas.append(value)
    return metas


def _extra(meta: dict[str, Any]) -> dict[str, Any]:
    extra = meta.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("rollout meta is missing object-valued extra provenance")
    return extra


def _task(meta: dict[str, Any]) -> dict[str, Any]:
    task = _extra(meta).get("local_task")
    if not isinstance(task, dict):
        raise ValueError("rollout meta is missing extra.local_task metrics/provenance")
    return task


def _required(meta: dict[str, Any], name: str, *, location: str = "meta") -> Any:
    if name not in meta:
        raise ValueError(f"rollout {location} is missing required field {name!r}")
    return meta[name]


def _metric(meta: dict[str, Any], name: str) -> float:
    if not name or not name.strip():
        raise ValueError("metric name must be non-empty")
    task = _task(meta)
    if name == "reward":
        value = _required(task, "reward", location="extra.local_task")
    else:
        metrics = task.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("rollout extra.local_task.metrics must be an object")
        value = _required(metrics, name, location="extra.local_task.metrics")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name!r} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {name!r} must be finite, got {value!r}")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _validated_outcome(meta: dict[str, Any]) -> dict[str, Any]:
    task = _task(meta)
    metrics = task.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("rollout extra.local_task.metrics must be an object")
    label = task.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("extra.local_task.label must be a non-empty string")
    if metrics.get("label") != label:
        raise ValueError("extra.local_task.label disagrees with metrics.label")
    reward = _metric(meta, "reward")
    metric_reward = metrics.get("reward")
    if (
        isinstance(metric_reward, bool)
        or not isinstance(metric_reward, (int, float))
        or not math.isfinite(float(metric_reward))
        or float(metric_reward) != reward
    ):
        raise ValueError("extra.local_task.reward disagrees with metrics.reward")
    survived = metrics.get("survived")
    if not isinstance(survived, bool):
        raise ValueError("metrics.survived must be boolean")
    won = metrics.get("won", survived)
    if not isinstance(won, bool):
        raise ValueError("metrics.won must be boolean when present")
    if "completed" in metrics:
        completed = metrics["completed"]
        if not isinstance(completed, bool):
            raise ValueError("metrics.completed must be boolean when present")
    else:
        stopped_reason = meta.get("stopped_reason")
        if not isinstance(stopped_reason, str) or not stopped_reason:
            raise ValueError("meta.stopped_reason must be a non-empty string")
        completed = stopped_reason in {"task_complete", "player_loss"} or survived
    turns = _nonnegative_int(metrics.get("n_turns"), label="metrics.n_turns")
    task_decisions = _nonnegative_int(
        metrics.get("n_decisions"), label="metrics.n_decisions"
    )
    meta_decisions = _nonnegative_int(meta.get("n_decisions"), label="meta.n_decisions")
    if task_decisions != meta_decisions:
        raise ValueError("metrics.n_decisions disagrees with meta.n_decisions")
    invalid = _nonnegative_int(meta.get("n_invalid"), label="meta.n_invalid")
    if invalid > meta_decisions:
        raise ValueError("meta.n_invalid exceeds meta.n_decisions")
    action_counts = metrics.get("action_counts")
    if not isinstance(action_counts, dict) or not action_counts:
        raise ValueError("metrics.action_counts must be a non-empty object")
    checked_counts: dict[str, int] = {}
    for name, value in action_counts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metrics.action_counts keys must be non-empty strings")
        checked_counts[name] = _nonnegative_int(
            value, label=f"metrics.action_counts.{name}"
        )
    if sum(checked_counts.values()) != task_decisions:
        raise ValueError("metrics.action_counts does not sum to metrics.n_decisions")
    return {
        "reward": reward,
        "hp_loss": _metric(meta, "hp_loss"),
        "label": label,
        "survived": survived,
        "won": won,
        "completed": completed,
        "turns": turns,
        "n_decisions": meta_decisions,
        "n_invalid": invalid,
        "action_counts": checked_counts,
        "metrics": metrics,
    }


def _aggregate(metas: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(metas)
    outcomes = [_validated_outcome(meta) for meta in metas]
    rewards = [outcome["reward"] for outcome in outcomes]
    hp_losses = [outcome["hp_loss"] for outcome in outcomes]
    wins = [outcome["won"] for outcome in outcomes]
    completed = [outcome["completed"] for outcome in outcomes]
    turns = [outcome["turns"] for outcome in outcomes]
    encounter_values: dict[str, list[float]] = {}
    common_metrics = {
        "entry_hp",
        "exit_hp",
        "hp_loss",
        "completed",
        "won",
        "survived",
        "reward",
        "n_decisions",
        "n_turns",
        "action_counts",
    }
    for outcome in outcomes:
        metrics = outcome["metrics"]
        for name, value in metrics.items():
            if (
                name in common_metrics
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"reported encounter metric {name!r} is non-finite")
            encounter_values.setdefault(str(name), []).append(numeric)
    labels = [outcome["label"] for outcome in outcomes]
    total_invalid = sum(outcome["n_invalid"] for outcome in outcomes)
    total_decisions = sum(outcome["n_decisions"] for outcome in outcomes)
    action_counts: Counter[str] = Counter()
    for outcome in outcomes:
        action_counts.update(outcome["action_counts"])
    total_actions = sum(action_counts.values()) or 1
    return {
        "n": n,
        "mean_reward": statistics.mean(rewards),
        "mean_hp_loss": statistics.mean(hp_losses),
        "mean_turns": statistics.mean(turns),
        "completion_rate": sum(completed) / n,
        "win_rate": sum(wins) / n,
        "convincing_rate": sum(1 for label in labels if label == "convincing") / n,
        "invalid_rate": total_invalid / total_decisions if total_decisions else 0.0,
        "action_share": {
            kind: action_counts[kind] / total_actions
            for kind in sorted(action_counts)
        },
        "encounter_metric_means": {
            name: statistics.mean(values)
            for name, values in sorted(encounter_values.items())
            if values
        },
    }


def _episode_identity(meta: dict[str, Any]) -> tuple[int, int]:
    world_seed = _required(meta, "world_seed")
    rollout_index = _required(meta, "rollout_index")
    if isinstance(world_seed, bool) or not isinstance(world_seed, int):
        raise ValueError(f"world_seed must be an integer, got {world_seed!r}")
    if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
        raise ValueError(f"rollout_index must be an integer, got {rollout_index!r}")
    return world_seed, rollout_index


def _indexed_arm(
    metas: list[dict[str, Any]], arm: str
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    episodes: dict[tuple[int, int], dict[str, Any]] = {}
    windows: dict[str, list[tuple[int, int]]] = {}
    window_sources: dict[str, tuple[int, str]] = {}
    for meta in metas:
        identity = _episode_identity(meta)
        if identity in episodes:
            raise ValueError(f"{arm} contains duplicate rollout identity {identity!r}")
        task = _task(meta)
        window_id = task.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError(f"{arm} rollout {identity!r} has no non-empty window_id")
        source_stem = task.get("source_stem")
        if not isinstance(source_stem, str) or not source_stem:
            raise ValueError(f"{arm} rollout {identity!r} has no non-empty source_stem")
        policy_seed = meta.get("policy_seed")
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValueError(f"{arm} rollout {identity!r} has no integer policy_seed")
        expected_policy_seed = derive_policy_seed(*identity)
        if policy_seed != expected_policy_seed:
            raise ValueError(
                f"{arm} rollout {identity!r} has policy_seed={policy_seed}, "
                f"expected {expected_policy_seed}"
            )
        window_source = (identity[0], source_stem)
        prior_source = window_sources.setdefault(window_id, window_source)
        if prior_source != window_source:
            raise ValueError(
                f"{arm} window {window_id!r} mixes source identities: "
                f"{prior_source!r} and {window_source!r}"
            )
        episodes[identity] = meta
        windows.setdefault(window_id, []).append(identity)
    return episodes, windows


def _one_value(values: Iterable[Any], *, arm: str, label: str) -> Any:
    encoded: dict[str, Any] = {}
    for value in values:
        encoded[json.dumps(value, sort_keys=True, separators=(",", ":"))] = value
    if len(encoded) != 1:
        raise ValueError(f"{arm} mixes {label}: {list(encoded.values())!r}")
    return next(iter(encoded.values()))


def _cohort(meta: dict[str, Any]) -> dict[str, Any]:
    extra = _extra(meta)
    task = _task(meta)
    if extra.get("local_task_eval") is not True:
        raise ValueError("rollout meta is not marked as a local-task eval")
    task_id = _required(task, "task_id", location="extra.local_task")
    split = _required(task, "split", location="extra.local_task")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("extra.local_task.task_id must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ValueError("extra.local_task.split must be a non-empty string")
    if extra.get("task_id") != task_id:
        raise ValueError(
            "extra.task_id disagrees with extra.local_task.task_id: "
            f"{extra.get('task_id')!r} != {task_id!r}"
        )
    if extra.get("split") != split:
        raise ValueError(
            "extra.split disagrees with extra.local_task.split: "
            f"{extra.get('split')!r} != {split!r}"
        )
    source_hash = extra.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError("rollout meta is missing extra.source_manifest_sha256")
    return {
        "task_id": task_id,
        "split": split,
        "source_manifest_sha256": source_hash,
    }


def _generation_config(meta: dict[str, Any]) -> dict[str, Any]:
    extra = _extra(meta)
    agent_config = extra.get("agent_config")
    if not isinstance(agent_config, dict):
        raise ValueError("rollout meta is missing extra.agent_config")
    interface = extra.get("interface_provenance")
    if not isinstance(interface, dict):
        raise ValueError("rollout meta is missing extra.interface_provenance")
    for key in REQUIRED_INTERFACE_DIGESTS:
        digest = interface.get(key)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(
                f"extra.interface_provenance.{key} must be a SHA-256 digest"
            )
    for key, expected in {
        "output_contract": extra.get("output_contract"),
        "combat_observation": extra.get("combat_observation"),
        "competence_interface_version": extra.get("competence_interface_version"),
    }.items():
        if interface.get(key) != expected:
            raise ValueError(
                f"extra.interface_provenance.{key} disagrees with flat provenance: "
                f"{interface.get(key)!r} != {expected!r}"
            )
    expected_pairs = {
        "model_id": meta.get("model_id"),
        "framing": meta.get("framing"),
        "temperature": meta.get("temperature"),
        "max_tokens": meta.get("max_tokens"),
        "thinking": meta.get("thinking"),
        "max_retries": meta.get("max_retries"),
        "output_contract": extra.get("output_contract"),
    }
    for key, top_value in expected_pairs.items():
        if key not in agent_config:
            raise ValueError(f"rollout extra.agent_config is missing {key!r}")
        if agent_config[key] != top_value:
            raise ValueError(
                f"rollout generation provenance disagrees for {key}: "
                f"meta={top_value!r}, agent_config={agent_config[key]!r}"
            )
    config = {
        "agent": _required(meta, "agent"),
        "model_id": _required(meta, "model_id"),
        "framing": _required(meta, "framing"),
        "temperature": _required(meta, "temperature"),
        "max_tokens": _required(meta, "max_tokens"),
        "thinking": _required(meta, "thinking"),
        "max_retries": _required(meta, "max_retries"),
        "battle_simulations": _required(meta, "battle_simulations"),
        "combat_control": _required(meta, "combat_control"),
        "output_contract": _required(extra, "output_contract", location="extra"),
        "orchestrator": _required(extra, "orchestrator", location="extra"),
        "concurrency": _required(extra, "concurrency", location="extra"),
        "agent_config": {
            key: value for key, value in agent_config.items() if key != "adapter_path"
        },
        "interface_provenance": {
            key: value
            for key, value in interface.items()
            if key not in {"combat_observation", "competence_interface_version"}
        },
    }
    # This contract carries command fields absent from generic RolloutMeta.
    # Without it, apparently matched arms could differ in cutoff/batching/cohort.
    eval_config = extra.get("local_task_eval_config")
    if not isinstance(eval_config, dict):
        raise ValueError("rollout meta is missing extra.local_task_eval_config")
    if isinstance(eval_config, dict):
        eval_expected = {
            "task_id": extra.get("task_id"),
            "split": extra.get("split"),
            "source_manifest_sha256": extra.get("source_manifest_sha256"),
            "model_id": meta.get("model_id"),
            "framing": meta.get("framing"),
            "combat_observation": extra.get("combat_observation"),
            "max_tokens": meta.get("max_tokens"),
            "temperature": meta.get("temperature"),
            "max_retries": meta.get("max_retries"),
            "thinking": meta.get("thinking"),
            "output_contract": extra.get("output_contract"),
            "concurrency": extra.get("concurrency"),
            "battle_simulations": meta.get("battle_simulations"),
        }
        if eval_config.get("backend") == "vllm":
            eval_expected.update(
                {
                    "top_p": agent_config.get("top_p"),
                    "top_k": agent_config.get("top_k"),
                    "enable_prefix_caching": agent_config.get("enable_prefix_caching"),
                }
            )
            if agent_config.get("backend") != "vllm":
                raise ValueError("vLLM eval config disagrees with agent_config.backend")
        for key, expected in eval_expected.items():
            if eval_config.get(key) != expected:
                raise ValueError(
                    f"extra.local_task_eval_config.{key} disagrees with rollout "
                    f"provenance: {eval_config.get(key)!r} != {expected!r}"
                )
        stored_adapter = eval_config.get("adapter_path")
        flat_adapter = extra.get("adapter_path")
        if stored_adapter is not None:
            stored_adapter = str(Path(stored_adapter).expanduser().resolve())
        if flat_adapter is not None:
            flat_adapter = str(Path(flat_adapter).expanduser().resolve())
        if stored_adapter != flat_adapter:
            raise ValueError(
                "extra.local_task_eval_config.adapter_path disagrees with "
                "extra.adapter_path"
            )
        if eval_config.get("adapter_provenance") != extra.get("adapter_provenance"):
            raise ValueError(
                "extra.local_task_eval_config.adapter_provenance disagrees with "
                "extra.adapter_provenance"
            )
        config["local_task_eval_config"] = {
            key: value
            for key, value in eval_config.items()
            if key not in {"adapter_path", "adapter_provenance", "combat_observation"}
        }
    return config


def _interventions(meta: dict[str, Any]) -> dict[str, Any]:
    extra = _extra(meta)
    agent_config = extra.get("agent_config") or {}
    adapter_path = extra.get("adapter_path")
    if agent_config.get("adapter_path") != adapter_path:
        raise ValueError(
            "extra.adapter_path disagrees with extra.agent_config.adapter_path: "
            f"{adapter_path!r} != {agent_config.get('adapter_path')!r}"
        )
    adapter_identity = extra.get("adapter_provenance")
    if adapter_path is None:
        if adapter_identity is not None:
            raise ValueError("base/no-adapter rollout has unexpected adapter_provenance")
        adapter_content_identity = None
    elif not isinstance(adapter_identity, dict):
        raise ValueError("adapter rollout is missing content-addressed adapter_provenance")
    else:
        adapter_content_identity = adapter_identity.get(
            "identity_sha256", adapter_identity.get("sha256")
        )
        if (
            not isinstance(adapter_content_identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", adapter_content_identity) is None
        ):
            raise ValueError("adapter_provenance has no valid content identity digest")
    combat_observation = extra.get("combat_observation")
    if not isinstance(combat_observation, str) or not combat_observation:
        raise ValueError("extra.combat_observation must be a non-empty string")
    if extra.get("competence_interface_version") != combat_observation:
        raise ValueError(
            "extra.competence_interface_version disagrees with combat_observation"
        )
    return {
        # The intervention identity is content-only: moving/copying the same
        # adapter must not masquerade as a model intervention.
        "adapter_path": adapter_content_identity,
        "combat_observation": combat_observation,
    }


def _identity_hash(identities: Iterable[tuple[int, int]]) -> str:
    payload = json.dumps(sorted(identities), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_manifest_completeness(
    *,
    manifest_path: Path,
    rollouts_per_window: int,
    cohort: dict[str, Any],
    episodes: dict[tuple[int, int], dict[str, Any]],
    window_ids: list[str],
) -> dict[str, Any]:
    if rollouts_per_window < 1:
        raise ValueError("rollouts_per_window must be >= 1")
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash is None:
        raise ValueError(f"could not fingerprint comparison manifest: {manifest_path}")
    if manifest_hash != cohort["source_manifest_sha256"]:
        raise ValueError(
            "comparison manifest does not match rollout source cohort: "
            f"manifest={manifest_hash!r}, rollouts={cohort['source_manifest_sha256']!r}"
        )
    manifest = local_task_base.load_manifest(manifest_path)
    if manifest.get("task_id") != cohort["task_id"]:
        raise ValueError(
            "comparison manifest task differs from rollout cohort: "
            f"manifest={manifest.get('task_id')!r}, rollouts={cohort['task_id']!r}"
        )
    windows = windows_for_split(manifest, str(cohort["split"]))
    if not windows:
        raise ValueError(
            f"comparison manifest has no windows for split {cohort['split']!r}"
        )
    expected_windows: dict[str, dict[str, Any]] = {}
    expected_episodes: dict[tuple[int, int], str] = {}
    for window in windows:
        window_id = window.get("window_id")
        source_stem = window.get("source_stem")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("comparison manifest window has no non-empty window_id")
        if not isinstance(source_stem, str) or not source_stem:
            raise ValueError(f"manifest window {window_id!r} has no non-empty source_stem")
        if window_id in expected_windows:
            raise ValueError(f"comparison manifest duplicates window_id {window_id!r}")
        expected_windows[window_id] = window
        world_seed = int(window["world_seed"])
        for rollout_index in window_rollout_indices(window, rollouts_per_window):
            identity = (world_seed, rollout_index)
            if identity in expected_episodes:
                raise ValueError(
                    "comparison manifest maps multiple windows to rollout identity "
                    f"{identity!r}"
                )
            expected_episodes[identity] = window_id

    if set(window_ids) != set(expected_windows):
        raise ValueError(
            "comparison arms are not the complete manifest window cohort; "
            f"missing={sorted(set(expected_windows) - set(window_ids))!r}, "
            f"unexpected={sorted(set(window_ids) - set(expected_windows))!r}"
        )
    if set(episodes) != set(expected_episodes):
        raise ValueError(
            "comparison arms are not the complete expected rollout cohort; "
            f"missing={sorted(set(expected_episodes) - set(episodes))!r}, "
            f"unexpected={sorted(set(episodes) - set(expected_episodes))!r}"
        )
    for identity, expected_window_id in expected_episodes.items():
        task = _task(episodes[identity])
        expected_window = expected_windows[expected_window_id]
        if task.get("window_id") != expected_window_id:
            raise ValueError(
                f"rollout {identity!r} maps to {task.get('window_id')!r}; "
                f"manifest expects {expected_window_id!r}"
            )
        if task.get("source_stem") != expected_window.get("source_stem"):
            raise ValueError(
                f"rollout {identity!r} source_stem differs from comparison manifest"
            )
    eval_config = _extra(next(iter(episodes.values())))["local_task_eval_config"]
    expected_window_id_list = sorted(expected_windows)
    expected_window_hash = hashlib.sha256(
        json.dumps(expected_window_id_list, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_identity_hash = _identity_hash(expected_episodes)
    config_expected = {
        "rollouts_per_window": rollouts_per_window,
        "expected_window_count": len(expected_windows),
        "expected_window_ids_sha256": expected_window_hash,
        "expected_rollout_count": len(expected_episodes),
        "expected_rollout_identities_sha256": expected_identity_hash,
    }
    for key, expected in config_expected.items():
        if eval_config.get(key) != expected:
            raise ValueError(
                f"local_task_eval_config.{key} disagrees with manifest completeness: "
                f"{eval_config.get(key)!r} != {expected!r}"
            )
    return {
        "path": str(manifest_path),
        "sha256": manifest_hash,
        "rollouts_per_window": rollouts_per_window,
        "expected_windows": len(expected_windows),
        "expected_rollouts": len(expected_episodes),
        "expected_window_ids_sha256": expected_window_hash,
        "expected_rollout_identity_sha256": expected_identity_hash,
        "complete": True,
    }


def _validate_matched_arms(
    base: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    *,
    allowed_interventions: Iterable[str],
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    allowed = set(allowed_interventions)
    unknown = allowed - set(ALLOWED_INTERVENTIONS)
    if unknown:
        raise ValueError(f"unknown allowed interventions: {sorted(unknown)!r}")

    base_episodes, base_windows = _indexed_arm(base, "base")
    trained_episodes, trained_windows = _indexed_arm(trained, "trained")
    if set(base_windows) != set(trained_windows):
        raise ValueError(
            "comparison requires exact identical window sets; "
            f"base_only={sorted(set(base_windows) - set(trained_windows))!r}, "
            f"trained_only={sorted(set(trained_windows) - set(base_windows))!r}"
        )
    if set(base_episodes) != set(trained_episodes):
        raise ValueError(
            "comparison requires exact identical rollout identities; "
            f"base_only={sorted(set(base_episodes) - set(trained_episodes))!r}, "
            f"trained_only={sorted(set(trained_episodes) - set(base_episodes))!r}"
        )
    for identity in sorted(base_episodes):
        base_task = _task(base_episodes[identity])
        trained_task = _task(trained_episodes[identity])
        for field in ("window_id", "source_stem"):
            if base_task.get(field) != trained_task.get(field):
                raise ValueError(
                    f"rollout {identity!r} differs in local-task {field}: "
                    f"base={base_task.get(field)!r}, trained={trained_task.get(field)!r}"
                )
        if base_episodes[identity].get("policy_seed") != trained_episodes[identity].get(
            "policy_seed"
        ):
            raise ValueError(f"rollout {identity!r} differs in policy_seed")
    for window_id in sorted(base_windows):
        if sorted(base_windows[window_id]) != sorted(trained_windows[window_id]):
            raise ValueError(
                f"window {window_id!r} has different rollout identities/counts"
            )

    base_cohort = _one_value((_cohort(meta) for meta in base), arm="base", label="cohorts")
    trained_cohort = _one_value(
        (_cohort(meta) for meta in trained), arm="trained", label="cohorts"
    )
    if base_cohort != trained_cohort:
        raise ValueError(
            f"comparison arms differ in task/source cohort: "
            f"base={base_cohort!r}, trained={trained_cohort!r}"
        )

    base_config = _one_value(
        (_generation_config(meta) for meta in base), arm="base", label="generation configs"
    )
    trained_config = _one_value(
        (_generation_config(meta) for meta in trained),
        arm="trained",
        label="generation configs",
    )
    if base_config != trained_config:
        raise ValueError(
            "comparison arms differ in non-intervention generation config: "
            f"base={base_config!r}, trained={trained_config!r}"
        )

    base_interventions = _one_value(
        (_interventions(meta) for meta in base), arm="base", label="interventions"
    )
    trained_interventions = _one_value(
        (_interventions(meta) for meta in trained), arm="trained", label="interventions"
    )
    observed: dict[str, Any] = {}
    for name in ALLOWED_INTERVENTIONS:
        is_different = base_interventions[name] != trained_interventions[name]
        if is_different and name not in allowed:
            raise ValueError(
                f"comparison differs in {name}; explicitly allow it with "
                f"--allow-intervention {name}"
            )
        if not is_different and name in allowed:
            raise ValueError(
                f"declared intervention {name} is identical in both arms; "
                "refusing a no-op comparison"
            )
        observed[name] = {
            "base": base_interventions[name],
            "trained": trained_interventions[name],
            "different": is_different,
            "allowed": name in allowed,
        }

    identities = sorted(base_episodes)
    window_ids = sorted(base_windows)
    validation = {
        "mode": "strict_matched_arms",
        "status": "passed",
        "allowed_interventions": sorted(allowed),
        "observed_interventions": observed,
        "cohort": base_cohort,
        "generation_config": base_config,
        "rollout_identities": {
            "n": len(identities),
            "sha256": _identity_hash(identities),
            "exact_match": True,
        },
        "windows": {
            "n": len(window_ids),
            "ids": window_ids,
            "exact_match": True,
            "rollout_counts": {
                window_id: len(base_windows[window_id]) for window_id in window_ids
            },
        },
    }
    return base_episodes, trained_episodes, window_ids, validation


def build_report(
    base_dir: Path,
    trained_dir: Path,
    *,
    metric: str,
    allowed_interventions: Iterable[str] = (),
    manifest: Path | None = None,
    rollouts_per_window: int | None = None,
) -> dict[str, Any]:
    if not metric or not metric.strip():
        raise ValueError("metric name must be non-empty")
    base = _load_metas(base_dir, "base")
    trained = _load_metas(trained_dir, "trained")
    base_episodes, trained_episodes, window_ids, validation = _validate_matched_arms(
        base,
        trained,
        allowed_interventions=allowed_interventions,
    )
    if manifest is None or rollouts_per_window is None:
        raise ValueError(
            "strict comparison requires manifest and rollouts_per_window to "
            "verify complete cohort membership"
        )
    validation["manifest"] = _validate_manifest_completeness(
        manifest_path=manifest,
        rollouts_per_window=rollouts_per_window,
        cohort=validation["cohort"],
        episodes=base_episodes,
        window_ids=window_ids,
    )
    base_grouped: dict[str, list[float]] = {window_id: [] for window_id in window_ids}
    trained_grouped: dict[str, list[float]] = {window_id: [] for window_id in window_ids}
    for identity in sorted(base_episodes):
        window_id = str(_task(base_episodes[identity])["window_id"])
        base_grouped[window_id].append(_metric(base_episodes[identity], metric))
        trained_grouped[window_id].append(_metric(trained_episodes[identity], metric))
    deltas = [
        statistics.mean(trained_grouped[window_id]) - statistics.mean(base_grouped[window_id])
        for window_id in window_ids
    ]
    counts = list(validation["windows"]["rollout_counts"].values())
    count_range = {"min": min(counts), "max": max(counts)}
    return {
        "task_id": validation["cohort"]["task_id"],
        "metric": metric,
        "base_dir": str(base_dir),
        "trained_dir": str(trained_dir),
        "validation": validation,
        "arms": {"base": _aggregate(base), "trained": _aggregate(trained)},
        "paired": {
            "n": len(window_ids),
            "mean_delta": statistics.mean(deltas),
            "bootstrap_ci_95": list(bootstrap_ci(deltas)),
            "sign_test": sign_test(deltas),
            "rollouts_per_window": {"base": count_range, "trained": count_range},
            "window_ids": window_ids,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly compare matched local-task eval arms.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--metric", default="reward")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Exact source manifest used by both arms; its hash and full split "
        "membership are verified.",
    )
    parser.add_argument(
        "--rollouts-per-window",
        type=int,
        required=True,
        help="Expected K samples for every manifest window.",
    )
    parser.add_argument(
        "--allow-intervention",
        action="append",
        choices=ALLOWED_INTERVENTIONS,
        default=[],
        help="Explicitly allow this single intended arm difference. Repeat only "
        "for adapter_path and/or combat_observation interventions.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = build_report(
        args.base,
        args.trained,
        metric=args.metric,
        allowed_interventions=args.allow_intervention,
        manifest=args.manifest,
        rollouts_per_window=args.rollouts_per_window,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
