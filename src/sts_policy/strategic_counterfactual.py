"""Complete-menu, fixed-world OOC outcomes under frozen public continuation.

This is a diagnostic panel, not a calibrated teacher. Policy sampling seeds vary
continuation choices; they do not vary hidden futures or provide independent
worlds. No intervention trajectory may enter on-policy PPO.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time
from typing import Any, Callable, Sequence

from .curriculum import TaskSpec, public_hash, world_split
from .rollout_training import task_identity
from .strategic_collection import collect

COUNTERFACTUAL_VERSION = "public_continuation_ooc_menu_v1"


def valid_episode(result: dict[str, Any]) -> dict[str, Any] | None:
    episodes = result["episodes"]
    if result["faults"] or not result["cohort_completed"] or len(episodes) != 1:
        return None
    episode = episodes[0]
    return episode if episode["terminated"] and not episode["truncated"] else None


def paired_outcomes(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float | int]:
    return {
        "victory_delta": int(candidate["task_outcome"] == "victory") - int(reference["task_outcome"] == "victory"),
        "act1_clear_delta": int(candidate["max_act_reached"] >= 2) - int(reference["max_act_reached"] >= 2),
        "act2_clear_delta": int(candidate["max_act_reached"] >= 3) - int(reference["max_act_reached"] >= 3),
        "floor_delta": int(candidate["floor"]) - int(reference["floor"]),
        "reward_delta": float(candidate["reward"]) - float(reference["reward"]),
    }


def evaluate_menu(model: Any, worker: Any, task: TaskSpec, *, max_seconds: float = 60.,
                  sampling_seeds: Sequence[int] = (0,), deterministic: bool = True,
                  collector: Callable[..., dict[str, Any]] = collect) -> dict[str, Any]:
    """Replay every legal first choice, followed by the same frozen policy.

    One worker and one task per collection preserve inference batch shape. The
    unforced control is rerun by forcing its chosen first action; all trace and
    reward identities must match before a panel is usable. Incomplete/faulted
    branches are preserved, never scored as defeats or omitted to pick a winner.
    """
    if task.kind != "run" or task.max_act != 3 or world_split(task.seed) != "train":
        raise ValueError("counterfactual panels require TRAIN full-run tasks through Act 3")
    if not math.isfinite(max_seconds) or max_seconds <= 0 or not sampling_seeds or len(set(sampling_seeds)) != len(sampling_seeds):
        raise ValueError("positive budget and distinct continuation sampling seeds required")
    if deterministic and len(sampling_seeds) != 1:
        raise ValueError("greedy continuation has no independent policy-sampling replications")
    started = time.monotonic()
    observation, info = worker.reset(task)
    if observation["phase"] != "out_of_combat" or len(observation["legal_actions"]) < 2:
        raise ValueError("counterfactual source requires a genuine OOC choice")
    source_hash = public_hash(observation)
    task = replace(task, expected_public_hash=source_hash)
    identity = task_identity(task)
    output = {"counterfactual_version": COUNTERFACTUAL_VERSION, "task": task.to_dict(),
        "task_id": identity, "observation": observation, "public_hash": source_hash,
        "source_provenance": {key: info.get(key) for key in
            ("environment_version", "observation_version", "python_interface_sha256", "native_provenance")},
        "continuation": {"policy": "frozen public OOC / immutable legacy greedy combat",
            "deterministic_ooc": deterministic, "sampling_seeds": list(sampling_seeds),
            "hidden_futures": "one fixed native world; no hidden-state randomization"},
        "panels": [], "usable_complete_menu": False, "teacher_target": None}

    def rollout(seed: int, action: int | None) -> dict[str, Any] | None:
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return None
        chosen: list[int] = []
        def observer(observations, world_seeds, steps, actions):
            chosen.extend(index for step, index in zip(steps, actions) if step == 0)
        result = collector(model, [worker], [task], credit_unit="macro", sampling_seed=seed,
            deterministic=deterministic, retain_transitions=False, max_seconds=remaining,
            decision_observer=observer, initial_actions={} if action is None else {identity: action})
        result.pop("transitions")
        result["first_executed_index"] = chosen[0] if chosen else None
        return result

    for seed in sampling_seeds:
        reference = rollout(seed, None)
        panel = {"sampling_seed": seed, "reference": reference, "branches": [],
                 "control_trace_matches": False, "complete": False}
        output["panels"].append(panel)
        baseline = valid_episode(reference) if reference is not None else None
        if baseline is None:
            continue
        baseline_index = reference["first_executed_index"]
        for action in observation["legal_actions"]:
            index = int(action["index"])
            result = rollout(seed, index)
            candidate = valid_episode(result) if result is not None else None
            branch = {"index": index, "description": action["description"], "result": result,
                      "valid_completed": candidate is not None,
                      "paired": paired_outcomes(baseline, candidate) if candidate is not None else None}
            panel["branches"].append(branch)
            if index == baseline_index and candidate is not None:
                panel["control_trace_matches"] = (
                    candidate["action_trace_sha256"] == baseline["action_trace_sha256"]
                    and candidate["reward"] == baseline["reward"]
                    and candidate["termination_state"] == baseline["termination_state"])
        panel["complete"] = panel["control_trace_matches"] and all(
            branch["valid_completed"] for branch in panel["branches"])
    output["usable_complete_menu"] = all(panel["complete"] for panel in output["panels"])
    output["elapsed_seconds"] = time.monotonic() - started
    return output
