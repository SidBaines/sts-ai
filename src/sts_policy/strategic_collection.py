"""Matched micro/macro collection with immutable greedy combat and task RNGs.

This experimental adapter never invokes the native teacher while playing. Its
sampling stream belongs to a task and OOC decision sequence, independent of
worker order, batch shape, combat duration or unrelated torch RNG calls.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sts_env.env import SimulatorError
from . import entities
from .encoding import encode_observation as encode_legacy, collate_observations as collate_legacy
from .entity_model import ResidualOOCPolicy
from .legacy_bridge import project_legacy_menu
from .losses import generalized_advantages
from .rollout_training import task_identity, public_endpoint_state

COLLECTOR_VERSION = "frozen_combat_strategic_v3_v5"
SAMPLING_STREAM_VERSION = "frozen_combat_strategic_v3_v1"


@dataclass
class StrategicTransition:
    observation: dict[str, Any]
    encoded: entities.EncodedEntities | None
    action_position: int
    action_index: int
    behavior_log_prob: float
    behavior_value: float
    reward: float
    reward_components: dict[str, float]
    primitive_duration: int
    terminated: bool
    truncated: bool
    next_value: float
    task_id: str
    world_seed: int
    actor_choice: bool
    behavior_distribution: np.ndarray | None = None
    advantage: float = 0.
    returns: float = 0.
    completed_return: float | None = None


def sample_position(logits: np.ndarray, rng: random.Random, deterministic: bool) -> tuple[int, float]:
    if not len(logits) or not np.isfinite(logits).all():
        raise ValueError("nonempty finite legal logits required")
    shifted = logits.astype(np.float64) - float(np.max(logits))
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    if deterministic:
        position = int(logits.argmax())
    else:
        draw = rng.random()
        position = min(len(logits) - 1, int(np.searchsorted(np.cumsum(probabilities), draw, side="right")))
    return position, float(np.log(max(probabilities[position], np.finfo(np.float64).tiny)))


def finish_advantages(rows: Sequence[StrategicTransition], *, gamma: float, gae_lambda: float) -> None:
    if gamma != 1:
        raise ValueError("matched strategic aggregation currently supports undiscounted gamma=1 only")
    advantages, returns = generalized_advantages([r.reward for r in rows], [r.behavior_value for r in rows],
        [r.terminated for r in rows], [r.truncated for r in rows], [r.next_value for r in rows],
        gamma=gamma, gae_lambda=gae_lambda)
    completed = bool(rows and rows[-1].terminated and not rows[-1].truncated)
    cumulative = 0.
    for row, advantage, target in zip(reversed(rows), reversed(advantages), reversed(returns)):
        row.advantage, row.returns = advantage, target
        cumulative += row.reward
        row.completed_return = cumulative if completed else None


@torch.no_grad()
def collect(model: ResidualOOCPolicy, workers: Sequence[Any], tasks: Sequence[Any], *,
            credit_unit: str = "macro", sampling_seed: int = 0, deterministic: bool = False,
            gamma: float = 1., gae_lambda: float = .95, max_seconds: float = 120.,
            retain_transitions: bool = True, frozen_combat: bool = True,
            decision_observer=None, native_combat_simulations: int | None = None,
            initial_actions: Mapping[str, int] | None = None) -> dict[str, Any]:
    if credit_unit not in {"micro", "macro"} or gamma != 1 or not workers or max_seconds <= 0:
        raise ValueError("valid workers, budget and gamma=1 micro/macro mode required")
    if native_combat_simulations is not None and (native_combat_simulations < 1 or retain_transitions or not frozen_combat):
        raise ValueError("native combat comparator requires a positive budget, frozen OOC base and no training rows")
    initial_actions = dict(initial_actions or {})
    if initial_actions and retain_transitions:
        raise ValueError("counterfactual interventions cannot produce on-policy training rows")
    if set(initial_actions) - {task_identity(task) for task in tasks}:
        raise ValueError("initial action intervention refers to an unknown task")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in initial_actions.values()):
        raise ValueError("initial action interventions require integer menu indices")
    model.cpu().eval()
    started = time.monotonic()
    pending = iter(tasks)
    active: dict[int, dict[str, Any]] = {}
    transitions, episodes, faults = [], [], []
    resets, excluded = 0, 0
    timings = {"policy_seconds": 0., "reset_wait_seconds": 0., "step_wait_seconds": 0., "bootstrap_seconds": 0.,
               "teacher_wait_seconds":0.}
    exposures: Counter = Counter()

    def start(slot: int) -> None:
        nonlocal resets
        try:
            task = next(pending)
        except StopIteration:
            return
        identity = task_identity(task)
        active[slot] = {"task": task, "identity": identity, "future": workers[slot].submit("reset", task),
            "rows": [], "pending_row": None, "obs": None, "info": {}, "reward": 0., "steps": 0,
            "components": Counter(), "rng": random.Random(f"{SAMPLING_STREAM_VERSION}:{sampling_seed}:{identity}:ooc"),
            "trace": hashlib.sha256(), "start_state": None, "max_act": 0, "choices": Counter(),
            "fight": None, "fights": [], "hp_decreases": 0., "hp_increases": 0.}
        resets += 1

    def fault(slot: int, error: Exception, stage: str) -> None:
        nonlocal excluded
        entry = active.pop(slot)
        excluded += entry["steps"]
        faults.append({"task_id": entry["identity"], "world_seed": entry["task"].seed,
            "stage": stage, "error": type(error).__name__, "message": str(error),
            "excluded_primitive_steps": entry["steps"], "excluded_training_rows": len(entry["rows"]),
            "action_trace_sha256": entry["trace"].hexdigest(), "last_public_state": public_endpoint_state(entry["obs"])})

    def finish(slot: int, *, budget: bool = False) -> None:
        entry = active.pop(slot)
        row = entry["pending_row"]
        if row is not None:
            row.truncated = budget or bool(entry["info"].get("truncated", False))
            entry["rows"].append(row)
        if budget and entry["rows"] and not entry["rows"][-1].terminated:
            entry["rows"][-1].truncated = True
        finish_advantages(entry["rows"], gamma=gamma, gae_lambda=gae_lambda)
        if retain_transitions:
            transitions.extend(entry["rows"])
        state, info = entry["obs"].get("state", {}), entry["info"]
        final_row = entry["rows"][-1] if entry["rows"] else None
        episodes.append({"task_id": entry["identity"], "world_seed": entry["task"].seed,
            "task_kind": entry["task"].kind, "start_family": "continuation" if entry["task"].prefix else "fresh",
            "initial_action_intervention": initial_actions.get(entry["identity"]),
            "prefix_length": len(entry["task"].prefix), "start_state": entry["start_state"],
            "termination_state": public_endpoint_state(entry["obs"]), "reward": entry["reward"],
            "reward_components": dict(entry["components"]), "steps": entry["steps"],
            "training_rows": len(entry["rows"]), "actor_choice_rows": sum(r.actor_choice for r in entry["rows"]),
            "floor": state.get("floor"), "act": state.get("act"), "max_act_reached": entry["max_act"],
            "termination_reason": "collector_budget" if budget else info.get("termination_reason"),
            "task_outcome": info.get("task_outcome"), "outcome": info.get("task_outcome") or info.get("outcome"),
            "terminated": bool(info.get("terminated", False)), "truncated": budget or bool(info.get("truncated", False)),
            "native_steps": info.get("native_steps"), "action_trace_sha256": entry["trace"].hexdigest(),
            "choice_counts": dict(entry["choices"]), "fights": entry["fights"],
            "observed_hp_decreases": entry["hp_decreases"], "observed_hp_increases": entry["hp_increases"]})

    def values_at(slots: list[int]) -> dict[int, float]:
        if not slots:
            return {}
        stamp = time.monotonic()
        batch = entities.collate_public([active[slot]["obs"] for slot in slots])
        _, values = model(*batch)
        timings["bootstrap_seconds"] += time.monotonic() - stamp
        return dict(zip(slots, values.cpu().tolist()))

    for slot in range(len(workers)):
        start(slot)
    while active:
        for slot in list(active):
            entry = active[slot]
            if entry["future"] is None:
                continue
            stamp = time.monotonic()
            try:
                obs, info = entry["future"].result()
                entry.update(obs=obs, info=info, future=None, start_state=public_endpoint_state(obs))
            except SimulatorError as error:
                fault(slot, error, "reset")
            timings["reset_wait_seconds"] += time.monotonic() - stamp
        if time.monotonic() - started >= max_seconds:
            bootstrap = values_at(list(active))
            for slot in list(active):
                entry = active[slot]
                last = entry["pending_row"] or (entry["rows"][-1] if entry["rows"] else None)
                if last is not None and not last.terminated:
                    last.next_value = bootstrap[slot]
                finish(slot, budget=True)
            break
        for slot in range(len(workers)):
            if slot not in active:
                start(slot)
        if any(entry["future"] is not None for entry in active.values()):
            continue
        if not active:
            break
        stamp = time.monotonic()
        samples: dict[int, tuple[int, float, float, entities.EncodedEntities | None]] = {}
        distributions: dict[int, np.ndarray] = {}
        ooc_slots = [slot for slot, entry in active.items() if entry["obs"]["phase"] != "combat"]
        combat_slots = [slot for slot in active if slot not in ooc_slots]
        typed_groups = [ooc_slots]
        if not frozen_combat or (credit_unit == "micro" and retain_transitions):
            typed_groups.append(combat_slots)
        for typed_slots in typed_groups:
            if not typed_slots:
                continue
            encoded = [entities.encode_observation(active[slot]["obs"]) for slot in typed_slots]
            logits, values = model(*entities.collate_observations(encoded))
            logits, values = logits.cpu().numpy(), values.cpu().tolist()
            for i, slot in enumerate(typed_slots):
                obs = active[slot]["obs"]
                combat = obs["phase"] == "combat"
                position, log_prob = sample_position(logits[i, :len(obs["legal_actions"])], active[slot]["rng"], deterministic or combat)
                samples[slot] = (position, log_prob, values[i], encoded[i] if retain_transitions else None)
                if not combat:
                    values64 = logits[i, :len(obs["legal_actions"])].astype(np.float64)
                    probabilities = np.exp(values64 - np.max(values64))
                    distributions[slot] = probabilities / probabilities.sum()
        remaining_combat = combat_slots if frozen_combat else [slot for slot in combat_slots if slot not in samples]
        if remaining_combat:
            projections = [project_legacy_menu(active[slot]["obs"]) for slot in remaining_combat]
            logits, values = model.prior(*collate_legacy([encode_legacy(p.observation) for p in projections]))
            for i, (slot, projection) in enumerate(zip(remaining_combat, projections)):
                old_position = int(logits[i].argmax().item())
                index = projection.current_indices[old_position]
                position = next(j for j, a in enumerate(active[slot]["obs"]["legal_actions"]) if int(a["index"]) == index)
                # Micro critic values come from the typed critic, while the
                # frozen combat action always follows the compact legacy menu.
                typed = samples.get(slot)
                samples[slot] = (position, 0., typed[2] if typed else float(values[i]), typed[3] if typed else None)
        timings["policy_seconds"] += time.monotonic() - stamp
        if native_combat_simulations is not None:
            queries = {slot:workers[slot].submit("teacher_action",native_combat_simulations) for slot in combat_slots}
            for slot,future in queries.items():
                stamp = time.monotonic()
                try:
                    index = int(future.result())
                    position = next((i for i,a in enumerate(active[slot]["obs"]["legal_actions"]) if int(a["index"])==index),None)
                    if position is None:
                        raise SimulatorError("Native comparator selected an illegal public action")
                    previous = samples[slot]
                    samples[slot] = (position,0.,previous[2],None)
                except SimulatorError as error:
                    samples.pop(slot)
                    fault(slot,error,"native_combat_query")
                timings["teacher_wait_seconds"] += time.monotonic()-stamp
        for slot in list(samples):
            entry = active[slot]
            if entry["steps"] or entry["identity"] not in initial_actions:
                continue
            obs, index = entry["obs"], initial_actions[entry["identity"]]
            if obs["phase"] != "out_of_combat":
                raise ValueError("initial action intervention requires an OOC source")
            position = next((i for i, action in enumerate(obs["legal_actions"])
                             if int(action["index"]) == index), None)
            if position is None:
                raise ValueError("initial action intervention is outside the legal source menu")
            previous = samples[slot]
            samples[slot] = (position, 0., previous[2], None)
        if decision_observer is not None:
            slots = list(samples)
            decision_observer([active[slot]["obs"] for slot in slots],
                              [active[slot]["task"].seed for slot in slots],
                              [active[slot]["steps"] for slot in slots],
                              [int(active[slot]["obs"]["legal_actions"][samples[slot][0]]["index"]) for slot in slots])
        steps = {}
        for slot, (position, log_prob, value, encoded) in samples.items():
            entry, obs = active[slot], active[slot]["obs"]
            if credit_unit == "micro":
                if entry["rows"]:
                    entry["rows"][-1].next_value = value
            elif entry["pending_row"] is not None and obs["phase"] != "combat":
                entry["pending_row"].next_value = value
                entry["rows"].append(entry["pending_row"])
                entry["pending_row"] = None
            action = obs["legal_actions"][position]
            if credit_unit == "micro" or obs["phase"] != "combat":
                row = StrategicTransition(obs, encoded, position, int(action["index"]), log_prob, value,
                    0., {}, 0, False, False, 0., entry["identity"], entry["task"].seed,
                    obs["phase"] != "combat" and len(obs["legal_actions"]) > 1,
                    behavior_distribution=distributions.get(slot))
                if credit_unit == "micro":
                    entry["rows"].append(row)
                else:
                    entry["pending_row"] = row
            phase = obs["phase"]
            bucket = f"{phase}:act{obs['state'].get('act')}:{'choice' if len(obs['legal_actions']) > 1 else 'forced'}"
            exposures[bucket] += 1
            entry["choices"][str(action.get("public", {}).get("kind", "unknown"))] += 1
            entry["trace"].update((json.dumps({"phase": phase, "action": action}, sort_keys=True, separators=(",", ":")) + "\n").encode())
            if phase == "combat" and entry["fight"] is None:
                entry["fight"] = {"entry": public_endpoint_state(obs),
                                  "enemies": [e["name"] for e in obs["state"]["combat"].get("enemies", [])], "steps": 0}
            steps[slot] = workers[slot].submit("step", int(action["index"]))
        bootstrap_slots = []
        for slot, future in steps.items():
            stamp = time.monotonic()
            try:
                obs, reward, terminated, truncated, info = future.result()
            except SimulatorError as error:
                fault(slot, error, "step")
                timings["step_wait_seconds"] += time.monotonic() - stamp
                continue
            timings["step_wait_seconds"] += time.monotonic() - stamp
            entry = active[slot]
            old_hp = public_endpoint_state(entry["obs"])["hp"]
            new_hp = public_endpoint_state(obs)["hp"]
            if old_hp is not None and new_hp is not None:
                entry["hp_decreases"] += max(0, old_hp - new_hp)
                entry["hp_increases"] += max(0, new_hp - old_hp)
            entry["steps"] += 1
            entry["reward"] += float(reward)
            components = info.get("reward_components", {})
            entry["components"].update(components)
            entry.update(obs=obs, info={**info, "terminated": bool(terminated), "truncated": bool(truncated)})
            entry["max_act"] = max(entry["max_act"], int(obs["state"].get("act", 0)))
            row = entry["rows"][-1] if credit_unit == "micro" else entry["pending_row"]
            if row is not None:
                row.reward += float(reward)
                for key, amount in components.items():
                    row.reward_components[key] = row.reward_components.get(key, 0.) + float(amount)
                row.primitive_duration += 1
                row.terminated, row.truncated = bool(terminated), bool(truncated)
            if entry["fight"] is not None:
                entry["fight"]["steps"] += 1
                if obs["phase"] != "combat" or terminated or truncated:
                    entry["fight"]["exit"] = public_endpoint_state(obs)
                    entry["fights"].append(entry["fight"])
                    entry["fight"] = None
            if terminated:
                finish(slot)
            elif truncated:
                bootstrap_slots.append(slot)
        for slot, value in values_at(bootstrap_slots).items():
            entry = active[slot]
            row = entry["rows"][-1] if credit_unit == "micro" else entry["pending_row"]
            if row is not None:
                row.next_value = value
            finish(slot)
        for slot in range(len(workers)):
            if slot not in active and time.monotonic() - started < max_seconds:
                start(slot)
    elapsed = time.monotonic() - started
    primitive_steps = sum(e["steps"] for e in episodes)
    return {"collector_version": COLLECTOR_VERSION, "sampling_stream_version": SAMPLING_STREAM_VERSION,
        "credit_unit": credit_unit, "sampling_seed": sampling_seed,
        "deterministic_ooc": deterministic,
        "combat_controller": f"privileged native combat search ({native_combat_simulations} simulations)" if native_combat_simulations
            else "immutable legacy bridge greedy" if frozen_combat else "learned public combat residual greedy",
        "native_combat_simulations":native_combat_simulations,
        "initial_action_interventions": initial_actions,
        "transitions": transitions, "episodes": episodes, "faults": faults, "requested_tasks": len(tasks),
        "unstarted_tasks": len(tasks) - resets, "excluded_primitive_steps": excluded,
        "cohort_completed": len(episodes) + len(faults) == len(tasks) and all(e["termination_reason"] != "collector_budget" for e in episodes),
        "primitive_steps": primitive_steps, "training_rows": len(transitions), "primitive_exposures": dict(exposures),
        "elapsed_seconds": elapsed, "primitive_steps_per_second": primitive_steps / max(.001, elapsed),
        "timings": timings, "task_ids": [task_identity(t) for t in tasks]}
