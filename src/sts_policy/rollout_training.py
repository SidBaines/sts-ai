"""Batched policy sampling, fault-aware isolated collection, and masked PPO."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
import hashlib
import json
from typing import Any, Callable, Sequence

import torch

from .losses import PPOConfig, generalized_advantages, masked_distribution, ppo_objective


def task_identity(task: Any) -> str:
    value = task.to_dict() if hasattr(task, "to_dict") else asdict(task)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Transition:
    observation: dict[str, Any]
    action_position: int
    action_index: int
    behavior_log_prob: float
    behavior_value: float
    behavior_mask: list[bool]
    reward: float
    terminated: bool
    truncated: bool
    next_value: float
    world_seed: int
    advantage: float = 0.
    returns: float = 0.


def _set_advantages(episode: list[Transition], config: PPOConfig) -> None:
    advantages, returns = generalized_advantages(
        [t.reward for t in episode], [t.behavior_value for t in episode],
        [t.terminated for t in episode], [t.truncated for t in episode],
        [t.next_value for t in episode], gamma=config.gamma, gae_lambda=config.gae_lambda)
    for transition, advantage, target in zip(episode, advantages, returns):
        transition.advantage, transition.returns = advantage, target


@torch.no_grad()
def collect_rollouts(model: torch.nn.Module, workers: Sequence[Any], tasks: Sequence[Any], *,
                     collate: Callable[..., Any], forward: Callable[..., Any],
                     config: PPOConfig = PPOConfig(), device: str = "cpu",
                     max_seconds: float = 120., deterministic: bool = False,
                     teacher_simulations: int | None = None,
                     deferred_bootstrap: bool = False) -> dict[str, Any]:
    """The actor is frozen throughout collection; failures quarantine whole episodes.

    Worker operations execute concurrently. A collector budget truncation uses
    the actual next state's value. The sampling batch is transferred to CPU only
    once, never separately for each environment/action candidate.
    """
    model.eval()
    started = time.monotonic()
    pending = iter(tasks)
    active: dict[int, dict[str, Any]] = {}
    transitions: list[Transition] = []
    labels: list[dict[str, Any]] = []
    episodes, faults = [], []
    failed_steps, resets = 0, 0
    timings = {"reset_wait_seconds": 0., "step_wait_seconds": 0., "teacher_wait_seconds": 0.,
               "policy_seconds": 0., "bootstrap_seconds": 0.}
    forward_counts = {"actor_batches": 0, "actor_observations": 0, "bootstrap_batches": 0,
                      "bootstrap_observations": 0, "reused_bootstrap_values": 0}

    def bootstrap_values(observations: list[dict[str, Any]]) -> list[float]:
        if not observations:
            return []
        bootstrap_started = time.monotonic()
        _, next_values, _ = forward(model, collate(observations, device))
        output = next_values.cpu().tolist()
        timings["bootstrap_seconds"] += time.monotonic() - bootstrap_started
        forward_counts["bootstrap_batches"] += 1
        forward_counts["bootstrap_observations"] += len(observations)
        return output

    def start(slot: int) -> None:
        nonlocal resets
        try:
            task = next(pending)
        except StopIteration:
            return
        active[slot] = {"task": task, "future": workers[slot].submit("reset", task),
                        "trajectory": [], "labels": [], "obs": None, "last_info": {},
                        "reward": 0., "steps": 0}
        resets += 1

    def fault(slot: int, error: Exception) -> None:
        nonlocal failed_steps
        entry = active.pop(slot)
        failed_steps += len(entry["trajectory"])
        faults.append({"world_seed": entry["task"].seed, "error": type(error).__name__,
                       "task_id": task_identity(entry["task"]),
                       "message": str(error), "excluded_transitions": len(entry["trajectory"]),
                       "task_kind": entry["task"].kind})

    def finish(slot: int, info: dict[str, Any], budget_truncation: bool = False) -> None:
        entry = active.pop(slot)
        info = {**entry["last_info"], **info}
        trajectory = entry["trajectory"]
        if entry.get("awaiting_bootstrap"):
            raise RuntimeError("Cannot finalize a trajectory with a missing next-state value")
        if budget_truncation and trajectory and not trajectory[-1].terminated:
            trajectory[-1].truncated = True
        _set_advantages(trajectory, config)
        transitions.extend(trajectory)
        labels.extend(entry["labels"])
        state = entry["obs"].get("state", {}) if entry["obs"] is not None else {}
        episodes.append({"world_seed": entry["task"].seed, "task_kind": entry["task"].kind,
                         "task_id": task_identity(entry["task"]),
                         "reward": entry["reward"], "steps": entry["steps"],
                         "termination_reason": "collector_budget" if budget_truncation else info.get("termination_reason", ""),
                         "outcome": info.get("task_outcome") or info.get("outcome", ""),
                         "task_outcome": info.get("task_outcome", ""),
                         "hp_cost": info.get("hp_cost"), "floor": state.get("floor", state.get("floor_num")),
                         "act": state.get("act", state.get("act_num")), "split": info.get("split"),
                         "native_steps": info.get("native_steps"), "fault": False,
                         "native_provenance": info.get("native_provenance"),
                         "reward_version": info.get("reward_version"),
                         "reward_settings": info.get("reward_settings"),
                         "terminated": bool(trajectory and trajectory[-1].terminated),
                         "truncated": budget_truncation or bool(trajectory and trajectory[-1].truncated)})

    for slot in range(len(workers)):
        start(slot)
    while active:
        for slot in list(active):
            entry = active[slot]
            if entry["future"] is None:
                continue
            try:
                wait_started = time.monotonic()
                observation, info = entry["future"].result()
                timings["reset_wait_seconds"] += time.monotonic() - wait_started
                entry["obs"], entry["future"] = observation, None
                entry["last_info"] = info
                if not observation.get("legal_actions"):
                    finish(slot, info)
            except Exception as error:
                fault(slot, error)
        if time.monotonic() - started >= max_seconds:
            if deferred_bootstrap:
                pending_slots = [slot for slot, entry in active.items() if entry.get("awaiting_bootstrap")]
                values = bootstrap_values([active[slot]["obs"] for slot in pending_slots])
                for slot, value in zip(pending_slots, values):
                    active[slot]["trajectory"][-1].next_value = float(value)
                    active[slot]["awaiting_bootstrap"] = False
            for slot in list(active):
                finish(slot, {}, budget_truncation=True)
            break
        # Refill available slots; resets are resolved together next iteration.
        for slot in range(len(workers)):
            if slot not in active:
                start(slot)
        if not active:
            break
        if any(entry["future"] is not None for entry in active.values()):
            continue
        slots = list(active)
        observations = [active[slot]["obs"] for slot in slots]
        policy_started = time.monotonic()
        logits, values, mask = forward(model, collate(observations, device))
        distribution = masked_distribution(logits, mask)
        positions = logits.masked_fill(~mask, -1e9).argmax(-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(positions)
        # Single packed device transfer for action/logp/value (action IDs < 2^24).
        packed = torch.stack((positions.float(), log_probs, values), dim=-1).cpu().tolist()
        timings["policy_seconds"] += time.monotonic() - policy_started
        forward_counts["actor_batches"] += 1
        forward_counts["actor_observations"] += len(observations)
        if deferred_bootstrap:
            for slot, (_, _, value) in zip(slots, packed):
                if active[slot].get("awaiting_bootstrap"):
                    active[slot]["trajectory"][-1].next_value = float(value)
                    active[slot]["awaiting_bootstrap"] = False
                    forward_counts["reused_bootstrap_values"] += 1
        if teacher_simulations is not None:
            futures = {slot: workers[slot].submit("teacher_action", teacher_simulations) for slot in slots}
            for slot in slots:
                try:
                    wait_started = time.monotonic()
                    label = futures[slot].result()
                    timings["teacher_wait_seconds"] += time.monotonic() - wait_started
                    active[slot]["labels"].append({"observation": active[slot]["obs"],
                                                     "teacher_index": label, "world_seed": active[slot]["task"].seed,
                                                     "task_kind": active[slot]["task"].kind,
                                                     "phase": active[slot]["obs"].get("phase", ""),
                                                     "source": "dagger"})
                except Exception as error:
                    fault(slot, error)
        steps = {}
        for slot, (position, log_prob, value) in zip(slots, packed):
            if slot not in active:
                continue
            observation = active[slot]["obs"]
            action = int(observation["legal_actions"][int(position)]["index"])
            active[slot]["sample"] = (int(position), action, float(log_prob), float(value))
            steps[slot] = workers[slot].submit("step", action)
        results = {}
        for slot, future in steps.items():
            try:
                wait_started = time.monotonic()
                results[slot] = future.result()
                timings["step_wait_seconds"] += time.monotonic() - wait_started
            except Exception as error:
                fault(slot, error)
        if results:
            next_slots = list(results)
            bootstrap_slots = ([slot for slot in next_slots if results[slot][3] and not results[slot][2]]
                               if deferred_bootstrap else next_slots)
            explicit_values = dict(zip(bootstrap_slots,
                bootstrap_values([results[slot][0] for slot in bootstrap_slots])))
            for slot in next_slots:
                observation, reward, terminated, truncated, info = results[slot]
                entry = active[slot]
                position, action, log_prob, value = entry.pop("sample")
                old_observation = entry["obs"]
                entry["trajectory"].append(Transition(old_observation, position, action, log_prob, value,
                                                       [True] * len(old_observation["legal_actions"]),
                                                       float(reward), bool(terminated), bool(truncated),
                                                       0. if terminated else float(explicit_values.get(slot, 0.)), entry["task"].seed))
                entry["awaiting_bootstrap"] = deferred_bootstrap and not (terminated or truncated)
                entry["reward"] += float(reward)
                entry["steps"] += 1
                entry["obs"] = observation
                entry["last_info"] = info
                if terminated or truncated:
                    finish(slot, info)
        # An entire batch can finish/fault together; refill before the while
        # condition so later tasks do not silently disappear.
        if time.monotonic() - started < max_seconds:
            for slot in range(len(workers)):
                if slot not in active:
                    start(slot)
    requested = len(tasks)
    cohort_completed = (resets == requested and len(episodes) + len(faults) == requested
                        and all(episode["termination_reason"] != "collector_budget" for episode in episodes))
    return {"transitions": transitions, "teacher_records": labels, "episodes": episodes,
            "requested_tasks": requested, "unstarted_tasks": requested - resets,
            "cohort_completed": cohort_completed,
            "task_ids": [task_identity(task) for task in tasks],
            "faults": faults, "excluded_transitions": failed_steps, "resets": resets,
            "elapsed_seconds": time.monotonic() - started,
            "timings": timings,
            "forward_counts": forward_counts, "deferred_bootstrap": deferred_bootstrap,
            "discount_unit": "one_external_decision; native_combat_is_one_macro_transition"}


def update_ppo(model: torch.nn.Module, transitions: Sequence[Transition], *,
               collate: Callable[..., Any], forward: Callable[..., Any],
               config: PPOConfig = PPOConfig(), device: str = "cpu", seed: int = 0,
               optimizer: torch.optim.Optimizer | None = None,
               max_seconds: float | None = None) -> dict[str, Any]:
    if not transitions:
        return {"steps": 0, "samples": 0, "optimizer": optimizer}
    started = time.monotonic()
    model.to(device).train()
    # Each PPO buffer is immutable throughout its epochs. Encode public state
    # once instead of repeating Python feature extraction every minibatch/epoch.
    from .model_agent import collate_public, cached_public_collator
    if collate is collate_public:
        collate = cached_public_collator([transition.observation for transition in transitions])
    preparation_seconds = time.monotonic() - started
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    advantages = torch.tensor([t.advantage for t in transitions], device=device)
    if len(transitions) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    rng, order = random.Random(seed), list(range(len(transitions)))
    metrics, steps = [], 0
    stopped = False
    for _ in range(config.epochs):
        rng.shuffle(order)
        for offset in range(0, len(order), config.minibatch_size):
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                stopped = True
                break
            indices = order[offset:offset + config.minibatch_size]
            chosen = [transitions[i] for i in indices]
            for transition in chosen:
                menu = transition.observation["legal_actions"]
                if len(menu) != len(transition.behavior_mask) or menu[transition.action_position]["index"] != transition.action_index:
                    raise ValueError("behavior menu identity changed after sampling")
                if not transition.behavior_mask[transition.action_position]:
                    raise ValueError("stored behavior action was masked")
            logits, values, mask = forward(model, collate([t.observation for t in chosen], device))
            actions = torch.tensor([t.action_position for t in chosen], device=device)
            old_log_probs = torch.tensor([t.behavior_log_prob for t in chosen], device=device)
            returns = torch.tensor([t.returns for t in chosen], device=device)
            loss, stats = ppo_objective(logits, mask, values, actions, old_log_probs,
                                        advantages[indices], returns, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            current = {key: float(value.cpu()) for key, value in stats.items()}
            metrics.append(current)
            steps += 1
            if current["approx_kl"] > config.target_kl:
                stopped = True
                break
        if stopped:
            break
    averages = ({key: sum(row[key] for row in metrics) / len(metrics) for key in metrics[0]}
                if metrics else {})
    return {"steps": steps, "samples": len(transitions), "elapsed_seconds": time.monotonic() - started,
            "preparation_seconds": preparation_seconds,
            "config": asdict(config), "early_stop": stopped,
            **averages,
            "optimizer": optimizer}
