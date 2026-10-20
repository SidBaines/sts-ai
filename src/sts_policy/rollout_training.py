"""Batched policy sampling, fault-aware isolated collection, and masked PPO."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
import math
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
    # In-memory public features only: not part of observation/on-disk schemas.
    encoded_observation: Any = field(default=None, repr=False, compare=False)
    encoding_observation_id: int | None = field(default=None, repr=False, compare=False)
    # OOC menu classification captured before collection slims public text.
    strategic_menu: bool | None = None


def _public_number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def public_endpoint_state(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    """Small public endpoint summary; never read hidden simulator state.

    Live combat HP supersedes the game context's potentially stale HP. Missing
    fields stay unavailable instead of using reward/encoder fallback defaults.
    """
    if not isinstance(observation, dict) or not isinstance(observation.get("state"), dict):
        return None
    state = observation["state"]
    combat = state.get("combat")
    if isinstance(combat, dict) and combat:
        hp = _public_number(combat.get("player_cur_hp"))
        max_hp = _public_number(combat.get("player_max_hp"))
    else:
        hp = _public_number(state.get("cur_hp"))
        max_hp = _public_number(state.get("max_hp"))
    result = {"hp": hp, "max_hp": max_hp,
              "act": _public_number(state.get("act", state.get("act_num"))),
              "floor": _public_number(state.get("floor", state.get("floor_num"))),
              "phase": observation.get("phase", state.get("phase"))}
    for key in ("room", "screen_state", "outcome"):
        value = state.get(key)
        result[key] = value[:160] if isinstance(value, str) else None
    result["done"] = state.get("done") if isinstance(state.get("done"), bool) else None
    return result


def _episode_public_metrics(entry: dict[str, Any], *, faulted: bool = False) -> dict[str, Any]:
    start = entry.get("start_public_state") or {}
    endpoint = public_endpoint_state(entry.get("obs"))
    end = {} if faulted else endpoint or {}
    return {"public_state_metrics_version": "endpoint_public_v1",
            "start_hp": start.get("hp"), "start_max_hp": start.get("max_hp"),
            "end_hp": end.get("hp"), "end_max_hp": end.get("max_hp"),
            "max_hp": end.get("max_hp"), "max_act_reached": entry.get("max_act_reached"),
            "start_state": entry.get("start_public_state"),
            "termination_state": None if faulted else endpoint,
            "last_observed_state": endpoint if faulted else None}


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
                     deferred_bootstrap: bool = False, reuse_encodings: bool = True,
                     capture_continuations: bool = False,
                     episode_sink_factory: Callable[[], Any] | None = None,
                     actor_only: bool = False) -> dict[str, Any]:
    """The actor is frozen throughout collection; failures quarantine whole episodes.

    Worker operations execute concurrently. A collector budget truncation uses
    the actual next state's value. The sampling batch is transferred to CPU only
    once, never separately for each environment/action candidate.

    An optional sparse-return sink consumes temporary Transitions via record(t)
    and returns episode data via finish(episode). It must retain only what its
    objective needs. No trajectory/GAE is retained in this mode; faults discard
    the whole sink. This requires deferred bootstrap, without teachers or
    continuation capture. actor_only additionally skips categorical/value work
    for deterministic evaluation and must be paired with a sink.
    """
    streaming = episode_sink_factory is not None
    if streaming and (not deferred_bootstrap or teacher_simulations is not None or capture_continuations):
        raise ValueError("Episode sinks require deferred sparse collection without teachers/continuations")
    if actor_only and (not deterministic or not streaming):
        raise ValueError("Actor-only collection requires deterministic episode sinks")
    from .model_agent import collate_public, collate_public_with_encodings
    retain_encodings = reuse_encodings and collate is collate_public
    model.eval()
    started = time.monotonic()
    pending = iter(tasks)
    active: dict[int, dict[str, Any]] = {}
    transitions: list[Transition] = []
    labels: list[dict[str, Any]] = []
    continuations: list[dict[str, Any]] = []
    episodes, faults = [], []
    episode_data = []
    retained_count, retained_bytes = 0, 0
    failed_steps, resets = 0, 0
    timings = {"reset_wait_seconds": 0., "step_wait_seconds": 0., "teacher_wait_seconds": 0.,
               "policy_seconds": 0., "bootstrap_seconds": 0., "fault_wait_seconds": 0.}
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
                        "trajectory": [], "labels": [], "continuations": [], "obs": None, "last_info": {},
                        "reward": 0., "steps": 0, "action_trace": hashlib.sha256(),
                        "start_public_state": None, "max_act_reached": None}
        if streaming:
            active[slot].update(sink=episode_sink_factory(), retained_count=0, retained_bytes=0,
                                last_terminated=False, last_truncated=False)
        resets += 1

    def fault(slot: int, error: Exception, *, stage: str, wait_seconds: float) -> None:
        from .faults import FaultRecordingError
        if isinstance(error, FaultRecordingError):
            raise error
        nonlocal failed_steps
        entry = active.pop(slot)
        worker = getattr(workers[slot], 'worker', workers[slot])
        excluded = entry["steps"] if streaming else len(entry["trajectory"])
        failed_steps += excluded
        timings["fault_wait_seconds"] += wait_seconds
        faults.append({"world_seed": entry["task"].seed, "error": type(error).__name__,
                       "fault_stage": stage, "wait_seconds": wait_seconds,
                       "action_trace_sha256": entry["action_trace"].hexdigest(),
                       "attempted_action_index": entry.get("sample", (None, None))[1],
                       "task_id": task_identity(entry["task"]),
                       "message": str(error), "excluded_transitions": excluded,
                       "task_kind": entry["task"].kind, **_episode_public_metrics(entry, faulted=True)})
        faults[-1]['fault_record_path'] = getattr(worker, 'last_fault_path', None)

    def finish(slot: int, info: dict[str, Any], budget_truncation: bool = False) -> None:
        nonlocal retained_count, retained_bytes
        entry = active.pop(slot)
        info = {**entry["last_info"], **info}
        trajectory = entry["trajectory"]
        if entry.get("awaiting_bootstrap"):
            raise RuntimeError("Cannot finalize a trajectory with a missing next-state value")
        if budget_truncation and trajectory and not trajectory[-1].terminated:
            trajectory[-1].truncated = True
        if not streaming:
            _set_advantages(trajectory, config)
        transitions.extend(trajectory)
        labels.extend(entry["labels"])
        if (capture_continuations and not budget_truncation and trajectory and
                trajectory[-1].terminated and not trajectory[-1].truncated):
            continuations.extend(entry["continuations"])
        state = entry["obs"].get("state", {}) if entry["obs"] is not None else {}
        episode = {"world_seed": entry["task"].seed, "task_kind": entry["task"].kind,
                         "task_id": task_identity(entry["task"]),
                         "action_trace_sha256": entry["action_trace"].hexdigest(),
                         "reward": entry["reward"], "steps": entry["steps"],
                         "termination_reason": "collector_budget" if budget_truncation else info.get("termination_reason", ""),
                         "outcome": info.get("task_outcome") or info.get("outcome", ""),
                         "task_outcome": info.get("task_outcome", ""),
                         'action_prefix': info.get('action_prefix'), 'ooc_previews': info.get('ooc_previews'),
                         "hp_cost": info.get("hp_cost"), "floor": state.get("floor", state.get("floor_num")),
                         "act": state.get("act", state.get("act_num")), "split": info.get("split"),
                         "native_steps": info.get("native_steps"), "fault": False,
                         "native_provenance": info.get("native_provenance"),
                         "reward_version": info.get("reward_version"),
                         "reward_settings": info.get("reward_settings"),
                         "terminated": entry["last_terminated"] if streaming else bool(trajectory and trajectory[-1].terminated),
                         "truncated": budget_truncation or (entry["last_truncated"] if streaming else bool(trajectory and trajectory[-1].truncated)),
                         **_episode_public_metrics(entry)}
        if streaming:
            episode_data.append(entry["sink"].finish(episode))
            retained_count += entry["retained_count"]
            retained_bytes += entry["retained_bytes"]
        episodes.append(episode)

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
                entry["start_public_state"] = public_endpoint_state(observation)
                entry["max_act_reached"] = (entry["start_public_state"] or {}).get("act")
                if not observation.get("legal_actions"):
                    finish(slot, info)
            except Exception as error:
                fault(slot, error, stage="reset", wait_seconds=time.monotonic() - wait_started)
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
        if retain_encodings:
            policy_batch, actor_encodings = collate_public_with_encodings(observations, device)
        else:
            policy_batch, actor_encodings = collate(observations, device), [None] * len(observations)
        logits, values, mask = forward(model, policy_batch)
        if actor_only:
            if logits.shape != mask.shape or mask.dtype != torch.bool or not bool(mask.any(-1).all()) or not bool(torch.isfinite(logits[mask]).all()):
                raise ValueError("Invalid actor-only legal logits/mask")
            positions = logits.masked_fill(~mask, -1e9).argmax(-1)
            log_probs = values = logits.new_zeros(len(logits))
        else:
            distribution = masked_distribution(logits, mask)
            positions = logits.masked_fill(~mask, -1e9).argmax(-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(positions)
        # Single packed device transfer for action/logp/value (action IDs < 2^24).
        packed = torch.stack((positions.float(), log_probs, values), dim=-1).cpu().tolist()
        timings["policy_seconds"] += time.monotonic() - policy_started
        forward_counts["actor_batches"] += 1
        forward_counts["actor_observations"] += len(observations)
        if deferred_bootstrap and not streaming:
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
                    fault(slot, error, stage="teacher_action", wait_seconds=time.monotonic() - wait_started)
        steps = {}
        for slot, (position, log_prob, value), encoded in zip(slots, packed, actor_encodings):
            if slot not in active:
                continue
            observation = active[slot]["obs"]
            action = int(observation["legal_actions"][int(position)]["index"])
            active[slot]["sample"] = (int(position), action, float(log_prob), float(value))
            active[slot]["sample_encoding"] = encoded
            steps[slot] = workers[slot].submit("step", action)
        results = {}
        for slot, future in steps.items():
            try:
                wait_started = time.monotonic()
                results[slot] = future.result()
                timings["step_wait_seconds"] += time.monotonic() - wait_started
            except Exception as error:
                fault(slot, error, stage="step", wait_seconds=time.monotonic() - wait_started)
        if results:
            next_slots = list(results)
            bootstrap_slots = ([] if streaming else
                               [slot for slot in next_slots if results[slot][3] and not results[slot][2]]
                               if deferred_bootstrap else next_slots)
            explicit_values = dict(zip(bootstrap_slots,
                bootstrap_values([results[slot][0] for slot in bootstrap_slots])))
            for slot in next_slots:
                observation, reward, terminated, truncated, info = results[slot]
                entry = active[slot]
                position, action, log_prob, value = entry.pop("sample")
                old_observation = entry["obs"]
                if capture_continuations:
                    from .long_run_tasks import capture_learner_continuation
                    frontier = capture_learner_continuation(entry["task"], old_observation,
                        observation, info, terminated=bool(terminated), truncated=bool(truncated))
                    if frontier is not None:
                        entry["continuations"].append(frontier)
                action_record = {"phase": old_observation.get("phase"),
                                 "action": old_observation["legal_actions"][position]}
                entry["action_trace"].update((json.dumps(action_record, sort_keys=True, separators=(",", ":")) + "\n").encode())
                transition = Transition(old_observation, position, action, log_prob, value,
                                                       [True] * len(old_observation["legal_actions"]),
                                                       float(reward), bool(terminated), bool(truncated),
                                                       0. if terminated else float(explicit_values.get(slot, 0.)), entry["task"].seed,
                                                       encoded_observation=entry.pop("sample_encoding"),
                                                       encoding_observation_id=id(old_observation) if retain_encodings else None)
                if streaming:
                    entry["sink"].record(transition)
                    encoded = transition.encoded_observation
                    if encoded is not None:
                        entry["retained_count"] += 1
                        entry["retained_bytes"] += (encoded.state_features.nbytes + encoded.action_features.nbytes
                                                    + encoded.action_indices.nbytes)
                    entry["last_terminated"], entry["last_truncated"] = bool(terminated), bool(truncated)
                else:
                    entry["trajectory"].append(transition)
                entry["awaiting_bootstrap"] = deferred_bootstrap and not streaming and not (terminated or truncated)
                entry["reward"] += float(reward)
                entry["steps"] += 1
                entry["obs"] = observation
                observed_state = observation.get("state", {})
                observed_act = _public_number(observed_state.get("act", observed_state.get("act_num")))
                if observed_act is not None:
                    entry["max_act_reached"] = max(entry["max_act_reached"], observed_act) if entry["max_act_reached"] is not None else observed_act
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
    return {**({"continuations": continuations} if capture_continuations else {}),
            **({"episode_data": episode_data} if streaming else {}),
            "transitions": transitions, "teacher_records": labels, "episodes": episodes,
            "requested_tasks": requested, "unstarted_tasks": requested - resets,
            "cohort_completed": cohort_completed,
            "task_ids": [task_identity(task) for task in tasks],
            "faults": faults, "excluded_transitions": failed_steps, "resets": resets,
            "elapsed_seconds": time.monotonic() - started,
            "timings": timings,
            "forward_counts": forward_counts, "deferred_bootstrap": deferred_bootstrap,
            "retained_encodings": retained_count if streaming else sum(t.encoded_observation is not None for t in transitions),
            "retained_encoding_bytes": retained_bytes if streaming else sum(t.encoded_observation.state_features.nbytes +
                t.encoded_observation.action_features.nbytes + t.encoded_observation.action_indices.nbytes
                for t in transitions if t.encoded_observation is not None),
            "discount_unit": "one_external_decision; native_combat_is_one_macro_transition"}


def update_ppo(model: torch.nn.Module, transitions: Sequence[Transition], *,
               collate: Callable[..., Any], forward: Callable[..., Any],
               config: PPOConfig = PPOConfig(), device: str = "cpu", seed: int = 0,
               optimizer: torch.optim.Optimizer | None = None,
               max_seconds: float | None = None,
               micro_batch: int | None = None,
               return_scale: float = 1.0,
               entropy_weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Masked PPO over an immutable buffer.

    ``micro_batch`` splits each minibatch's forward/backward into gradient-
    accumulated chunks (same update up to float summation order). Candidate
    menus are padded per batch, so a 4096-row minibatch of large OOC menus can
    need ~10 GB of activations; chunks of ~1024 bound that peak.

    Targets are divided by ``return_scale`` without mutating the raw buffer.
    ``entropy_weights`` optionally supplies the total bonus coefficient per row.
    """
    if not math.isfinite(return_scale) or return_scale <= 0:
        raise ValueError("return_scale must be finite and positive")
    if entropy_weights is not None and len(entropy_weights) != len(transitions):
        raise ValueError("entropy weights must align with transitions")
    if not transitions:
        return {"steps": 0, "samples": 0, "early_stop": False, "optimizer": optimizer}
    started = time.monotonic()
    model.to(device).train()
    # Each PPO buffer is immutable throughout its epochs. Encode public state
    # once instead of repeating Python feature extraction every minibatch/epoch.
    from .model_agent import collate_public, cached_public_collator
    reused_encoding_count = 0
    if collate is collate_public:
        for transition in transitions:
            if transition.encoded_observation is not None:
                if transition.encoding_observation_id != id(transition.observation):
                    raise ValueError("cached encoding observation identity changed after sampling")
                reused_encoding_count += 1
        collate = cached_public_collator([transition.observation for transition in transitions],
                                        preencoded=[transition.encoded_observation for transition in transitions])
    preparation_seconds = time.monotonic() - started
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    advantages = torch.tensor([t.advantage for t in transitions], device=device)
    entropy_coefficients = (None if entropy_weights is None else
                            torch.tensor(entropy_weights, device=device))
    if len(transitions) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    rng, order = random.Random(seed), list(range(len(transitions)))
    metrics, steps = [], 0
    stopped = False
    rejected_minibatches = 0
    rejected_kl = None
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
            if micro_batch is not None and len(chosen) > micro_batch:
                # Accumulate the minibatch gradient chunk by chunk; every objective
                # term is a minibatch mean, so chunk losses are weighted by size.
                optimizer.zero_grad(set_to_none=True)
                current = {}
                for start in range(0, len(chosen), micro_batch):
                    part, part_indices = chosen[start:start + micro_batch], indices[start:start + micro_batch]
                    logits, values, mask = forward(model, collate([t.observation for t in part], device))
                    loss, stats = ppo_objective(
                        logits, mask, values, torch.tensor([t.action_position for t in part], device=device),
                        torch.tensor([t.behavior_log_prob for t in part], device=device),
                        advantages[part_indices], torch.tensor([t.returns / return_scale for t in part], device=device),
                        config, entropy_weights=(None if entropy_coefficients is None else entropy_coefficients[part_indices]))
                    weight = len(part) / len(chosen)
                    (loss * weight).backward()
                    for key, value in stats.items():
                        current[key] = current.get(key, 0.0) + weight * float(value.cpu())
                    del logits, values, mask, loss, stats
                if config.kl_guard_before_update and current["approx_kl"] > config.target_kl:
                    optimizer.zero_grad(set_to_none=True)
                    stopped = True
                    rejected_minibatches += 1
                    rejected_kl = current["approx_kl"]
                    break
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                metrics.append(current)
                steps += 1
                if current["approx_kl"] > config.target_kl:
                    stopped = True
                    break
                continue
            logits, values, mask = forward(model, collate([t.observation for t in chosen], device))
            actions = torch.tensor([t.action_position for t in chosen], device=device)
            old_log_probs = torch.tensor([t.behavior_log_prob for t in chosen], device=device)
            returns = torch.tensor([t.returns / return_scale for t in chosen], device=device)
            loss, stats = ppo_objective(logits, mask, values, actions, old_log_probs,
                                        advantages[indices], returns, config,
                                        entropy_weights=(None if entropy_coefficients is None else entropy_coefficients[indices]))
            if config.kl_guard_before_update:
                current = {key: float(value.cpu()) for key, value in stats.items()}
                if current["approx_kl"] > config.target_kl:
                    stopped = True
                    rejected_minibatches += 1
                    rejected_kl = current["approx_kl"]
                    break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if not config.kl_guard_before_update:
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
            "reused_encoding_count": reused_encoding_count,
            "config": asdict(config), "early_stop": stopped,
            "algorithm_version": "masked_ppo_v2_before_update_kl" if config.kl_guard_before_update else "masked_ppo_v1_after_update_kl",
            "rejected_minibatches": rejected_minibatches, "rejected_kl": rejected_kl,
            **averages,
            "optimizer": optimizer}
