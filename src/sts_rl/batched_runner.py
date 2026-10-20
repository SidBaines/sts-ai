"""Greedy full games with native episodes and batched CPU inference together.

The direct runner is deliberately usable without multiprocessing. Production
evaluation uses disposable spawned processes, with progress measured by completed
steps rather than process liveness. No policy sampling or training lives here.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import multiprocessing
from multiprocessing.connection import wait
import os
import time
import traceback

from sts_env.env import SimulatorError
from sts_policy.curriculum import public_hash
from sts_rl.fresh_collection import WorldMetrics, provenance, summarize, task
from sts_rl.fresh_study import PROTOCOL, policy_observation, validate_world_range


def configure_cpu():
    # Set these before importing torch in each spawned interpreter as well.
    for name in ("PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)


class BatchedPolicy:
    """Reuse StudyPolicy's loaded heads, filters and original menu indices.

    One forward per distinct model/encoder is necessary for phase replacements.
    With no replacements the champion receives one batch, including both phases.
    """
    def __init__(self, study_policy):
        configure_cpu()
        self.study = study_policy

    def choose_actions(self, observations, *, details=None):
        import torch
        from sts_policy.entity_model import EntityActorCritic
        from sts_policy.model_agent import collate_public

        filtered = [policy_observation(obs) for obs in observations]
        actions, groups = [None] * len(filtered), {}
        if details is not None:
            details.extend([None] * len(filtered))
        for i, obs in enumerate(filtered):
            legal = obs["legal_actions"]
            if not legal:
                raise ValueError("Policy requires a legal decision")
            if len(legal) == 1:
                actions[i] = int(legal[0]["index"])
                continue
            phase = "combat" if obs["phase"] == "combat" else "ooc"
            model = self.study.replacements.get(phase, self.study.champion)
            controlled = phase in getattr(self.study, "_controlled_phases", ())
            kind = "controlled" if controlled else "entity" if isinstance(model, EntityActorCritic) else "legacy"
            groups.setdefault((model, kind), []).append(i)

        with self.study._lock, torch.inference_mode():
            for (model, kind), indices in groups.items():
                values = [filtered[i] for i in indices]
                if kind == "legacy":
                    batch = collate_public(values, "cpu")
                    logits = model.actor_logits(*batch)
                else:
                    if kind == "controlled":
                        from sts_policy.controlled_encoding import encode_observation, collate_observations
                    else:
                        from sts_policy.entity_encoding import encode_observation, collate_observations
                    encoded = [encode_observation(obs) for obs in values]
                    if any(row.action_indices.tolist() != [a["index"] for a in obs["legal_actions"]]
                           for row, obs in zip(encoded, values)):
                        raise ValueError("Encoder action/menu index mismatch")
                    batch = collate_observations(encoded, device="cpu")
                    logits = model.actor_logits(batch)
                logits = logits[0] if isinstance(logits, tuple) else logits
                width = max(len(obs["legal_actions"]) for obs in values)
                if logits.shape != (len(values), width):
                    raise ValueError("Batched logits/menu shape mismatch")
                for row, i in enumerate(indices):
                    # Ignore padding, including the entity encoder's -inf mask.
                    scores = logits[row, :len(filtered[i]["legal_actions"])]
                    if not bool(torch.isfinite(scores).all()):
                        raise ValueError("Nonfinite policy logits")
                    actions[i] = int(filtered[i]["legal_actions"][int(scores.argmax())]["index"])
                    if details is not None:
                        details[i] = scores
        return actions


@dataclass
class _World:
    seed: int
    episode: object
    observation: dict | None = None
    expected: dict | None = None
    spec: object = None
    prefix: list = field(default_factory=list)
    metrics: WorldMetrics = field(default_factory=WorldMetrics)
    divergence: dict | None = None

    def record(self, status, *, reason=None, win=None, attempted_action=None):
        obs = self.observation
        result = {"protocol": PROTOCOL, "world_seed": self.seed, "status": status,
                  "task": self.spec.to_dict() if self.spec else task(self.seed).to_dict(),
                  "expected_provenance": self.expected, "win": win,
                  "floor": obs["state"].get("floor") if obs else None,
                  "bosses": self.metrics.bosses, "decisions": dict(self.metrics.decisions),
                  "behavior": self.metrics.counters, "action_prefix": list(self.prefix),
                  "action_hash": public_hash({"actions": self.prefix}),
                  "terminal_hash": public_hash(obs) if status == "completed" else None,
                  "sampling": {}, "labels": [], "bank": []}
        if reason is not None:
            result.update(reason=reason, attempted_action=attempted_action)
        if self.divergence is not None:
            result["first_divergence"] = self.divergence
        return result


def fault_record(seed, reason):
    """A world that died before reset has no observed game outcome."""
    return _World(seed, None).record("fault", reason=reason)


def _diagnose(world, action, scores, policy, references):
    reference = references.get(world.seed)
    if reference is None or world.divergence is not None:
        return
    prefix, decision = reference["action_prefix"], len(world.prefix)
    if decision < len(prefix) and prefix[decision] == action:
        return
    obs = policy_observation(world.observation)
    indices = [int(a["index"]) for a in obs["legal_actions"]]
    single = policy.study._logits(obs) if len(indices) > 1 else None
    expected = prefix[decision] if decision < len(prefix) else None
    row = {"decision": decision, "reference_action": expected, "batched_action": action,
           "singleton_action": indices[int(single.argmax())] if single is not None else indices[0],
           "singleton_logit_gap": None, "batched_logit_gap": None}
    if expected in indices and scores is not None:
        a, b = indices.index(expected), indices.index(action)
        row.update(singleton_logit_gap=float(single[a] - single[b]),
                   batched_logit_gap=float(scores[b] - scores[a]))
    world.divergence = row


def run_in_process(seeds, policy, games_per_process=1, *, episode_factory=None,
                   on_event=None, reference_records=None):
    """Run batched greedy games directly, without native crash/timeout isolation.

    Native exception guards are scoped to this call. The process driver provides
    containment; custom episode factories are useful for tests without a build.
    """
    from contextlib import nullcontext
    if episode_factory is None:
        from sts_env._backend.lightspeed_import import import_lightspeed
        from sts_rl.fresh_native import native_guard_scope
        scope = native_guard_scope(import_lightspeed())
    else:
        scope = nullcontext()
    with scope:
        return _run_in_process(seeds, policy, games_per_process, episode_factory=episode_factory,
                               on_event=on_event, reference_records=reference_records)


def _run_in_process(seeds, policy, games_per_process=1, *, episode_factory=None,
                    on_event=None, reference_records=None):
    """Run a queue of worlds directly; caller supplies isolation if required.

    Events are start(seed), decision(active worlds), record(record). A decision
    event is emitted only after an actual native step completes. The production
    adapter throttles IPC heartbeats, not the game loop. Diagnostic comparisons
    use the *same* batch forward that selected the action, before any game steps.
    """
    from sts_rl.common import WORKER_CONFIG
    from sts_rl.fresh_panel import FreshPanelEpisode
    configure_cpu()
    _positive_integer(games_per_process, "games_per_process")
    seeds = _validate_seeds(seeds)
    pending, active, records = deque(seeds), [], []
    policy = policy if isinstance(policy, BatchedPolicy) else BatchedPolicy(policy)
    references = {r["world_seed"]: r for r in reference_records or []}
    if reference_records is not None and (len(references) != len(reference_records) or
                                         set(references) != set(seeds)):
        raise ValueError("Diagnostic references must match the unique world cohort")
    factory = episode_factory or FreshPanelEpisode

    def emit(kind, value):
        if on_event:
            on_event(kind, value)

    def finish(world, status, **kwargs):
        record = world.record(status, **kwargs)
        records.append(record)
        emit("record", record)
        active.remove(world)

    while pending or active:
        while pending and len(active) < games_per_process:
            seed = pending.popleft()
            emit("start", seed)  # Before construction/reset, which can crash or hang.
            world = _World(seed, None)
            active.append(world)
            try:
                world.episode = factory({**WORKER_CONFIG, "fast_observations": False})
                world.observation, info = world.episode.reset(task(seed))
                world.expected = provenance(info)
                if world.expected["environment_version"] != "0.1.5":
                    raise RuntimeError("Batched runner requires environment 0.1.5")
                world.spec = task(seed, world.expected, expected_hash=public_hash(world.observation))
            except SimulatorError as exc:
                finish(world, "fault", reason=str(exc))
        if not active:
            continue
        details = [] if references else None
        actions = policy.choose_actions([w.observation for w in active], details=details)
        # The active list changes on completion; preserve this batch's row order.
        for i, (world, action) in enumerate(zip(list(active), actions)):
            if references:
                _diagnose(world, action, details[i], policy, references)
            try:
                after, _, terminated, truncated, info = world.episode.step(action)
            except SimulatorError as exc:
                finish(world, "fault", reason=str(exc), attempted_action=action)
                continue
            win = info.get("task_outcome") in {"act_clear", "victory"}
            world.metrics.transition(world.observation, action, after, win=win)
            world.prefix.append(action)
            world.observation = after
            emit("decision", active)
            if terminated or truncated or len(world.prefix) >= 4096:
                if truncated or info.get("task_outcome") not in {"act_clear", "victory", "defeat"}:
                    finish(world, "censored", reason="cut:" + (info.get("termination_reason") or "task_horizon"))
                else:
                    if "action_prefix" in info and info["action_prefix"] != world.prefix:
                        raise RuntimeError("Completed episode/runner action prefixes differ")
                    finish(world, "completed", win=win)
    return sorted(records, key=lambda r: r["world_seed"])


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_seeds(seeds):
    seeds = list(seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate world seeds")
    for seed in seeds:
        validate_world_range(seed, 1)
    return seeds


class _RecycleRunner(Exception):
    """A native fault retires the process, as in FreshPanelWorker."""


def _runner_child(connection, seeds, checkpoint, combat, ooc, games_per_process, reference_records=None):
    configure_cpu()
    from sts_rl.controlled_policy import policy_factory
    last_sent = 0.0

    def event(kind, value):
        nonlocal last_sent
        if kind == "decision":
            now = time.monotonic()
            if now - last_sent < .5:
                return
            last_sent = now
            # Partial metrics are only diagnostic; they never count as outcomes.
            value = [w.record("fault", reason="last progress snapshot") for w in value]
        connection.send((kind, value))
        if kind == "record" and value["status"] == "fault":
            # Fail closed: do not trust another native episode in this process
            # after a simulator failure. The parent retries its peers.
            raise _RecycleRunner()

    try:
        policy = policy_factory(checkpoint, combat, ooc)
        run_in_process(seeds, policy, games_per_process, on_event=event, reference_records=reference_records)
        connection.send(("done", None))
    except _RecycleRunner:
        pass
    except Exception:
        # Programming/loading errors must be visible and abort, never become losses.
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


@dataclass
class _Lane:
    remaining: list
    active: dict = field(default_factory=dict)
    process: object = None
    connection: object = None
    last_progress: float = 0.0
    startup_failures: int = 0
    requeues: Counter = field(default_factory=Counter)

    def accept(self, kind, value, records, now):
        if kind == "start":
            if value not in self.remaining or value in self.active:
                raise RuntimeError("Unexpected/duplicate world start")
            self.active[value] = fault_record(value, "process exited during reset")
            self.startup_failures = 0
        elif kind == "decision":
            self.last_progress = now
            self.startup_failures = 0
            for record in value:
                seed = record["world_seed"]
                if seed not in self.active:
                    raise RuntimeError("Progress for an inactive world")
                self.active[seed] = record
        elif kind == "record":
            seed = value["world_seed"]
            if seed not in self.active:
                raise RuntimeError("Result for an inactive world")
            value["requeues"] = self.requeues[seed]
            records.append(value)
            del self.active[seed]
            self.remaining.remove(seed)
        elif kind == "error":
            raise RuntimeError("Batched runner child programming error:\n" + value)
        elif kind != "done":
            raise RuntimeError(f"Unknown runner event: {kind}")

    def quarantine(self, records, reason, *, retry=True):
        for seed, partial in self.active.items():
            if retry and self.requeues[seed] == 0:
                # Remaining already contains active worlds in their original
                # order. Clearing active makes the replacement restart them.
                self.requeues[seed] += 1
                continue
            records.append({**partial, "status": "fault", "win": None, "terminal_hash": None,
                            "reason": reason, "partial_snapshot": True,
                            "requeues": self.requeues[seed]})
            self.remaining.remove(seed)
        had_active = bool(self.active)
        self.active.clear()
        # A binary that crashes at startup cannot cause an unbounded restart loop.
        self.startup_failures = 0 if had_active else self.startup_failures + 1


def _close_lane(lane):
    if lane.process is not None:
        if lane.process.is_alive():
            lane.process.terminate()
        lane.process.join(timeout=2)
        if lane.process.is_alive():
            lane.process.kill()
            lane.process.join(timeout=2)
        lane.process.close()
    if lane.connection is not None:
        lane.connection.close()
    lane.process = lane.connection = None


def run_multiprocess(seeds, checkpoint, combat=None, ooc=None, *, processes=1,
                     games_per_process=8, timeout=120.0, reference_records=None, on_record=None,
                     clock=None):
    """Spawn N runners; retry in-flight peers once after a process failure.

    Start/record events share an ordered pipe. Drain it before handling EOF/death,
    so completed worlds survive and queued-but-unstarted worlds can be retried.
    No decision round trips or acknowledgements are required.
    """
    configure_cpu()
    clock = clock or time.monotonic
    _positive_integer(processes, "processes")
    _positive_integer(games_per_process, "games_per_process")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    seeds = _validate_seeds(seeds)
    references = {r["world_seed"]: r for r in reference_records or []}
    if reference_records is not None and (len(references) != len(reference_records) or
                                         set(references) != set(seeds)):
        raise ValueError("Diagnostic references must match the unique world cohort")
    context, records = multiprocessing.get_context("spawn"), []
    lanes = [_Lane(seeds[i::processes]) for i in range(min(processes, len(seeds)))]

    def launch(lane):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_runner_child,
            args=(child, lane.remaining, checkpoint, combat, ooc, games_per_process,
                  [references[s] for s in lane.remaining] if reference_records is not None else None),
            name="batched-games")
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        lane.connection, lane.process, lane.last_progress = parent, process, clock()

    try:
        for lane in lanes:
            launch(lane)
        previous_iteration = clock()
        while any(lane.remaining for lane in lanes):
            live = [lane for lane in lanes if lane.process is not None]
            ready = wait([lane.connection for lane in live], timeout=min(.1, timeout))
            now = clock()
            gap = now - previous_iteration
            if gap > 5.0:
                # The queue SIGSTOPs the whole job on battery. A paused parent
                # cannot monitor progress, so exclude that gap from all lanes.
                for lane in live:
                    lane.last_progress += gap
            previous_iteration = now
            for lane in live:
                ended = False
                if lane.connection in ready:
                    # Limit draining so a busy lane cannot starve timeout checks.
                    for _ in range(256):
                        try:
                            kind, value = lane.connection.recv()
                        except (EOFError, OSError):
                            ended = True
                            break
                        lane.accept(kind, value, records, now)
                        if kind == "record" and on_record:
                            on_record(value)
                        if kind == "done":
                            ended = True
                            break
                        if not lane.connection.poll():
                            break
                stalled = now - lane.last_progress > timeout
                if ended or stalled:
                    reason = (f"process made no completed decision for {timeout:g}s" if stalled else
                              f"runner process exited (exitcode {lane.process.exitcode})")
                    lane.quarantine(records, reason)
                    _close_lane(lane)
                    if lane.remaining:
                        if lane.startup_failures >= 2:
                            raise RuntimeError("Runner repeatedly died before starting a world: " + reason)
                        launch(lane)
        if len(records) != len(seeds) or {r["world_seed"] for r in records} != set(seeds):
            raise RuntimeError("Runner lost or duplicated worlds")
        return sorted(records, key=lambda r: r["world_seed"])
    except Exception as exc:
        # Let a CLI preserve completed work even when a loading/programming error
        # aborts the run. Such errors still propagate to the caller.
        for lane in lanes:
            lane.quarantine(records, "runner aborted: " + str(exc), retry=False)
        exc.records = sorted(records, key=lambda r: r["world_seed"])
        exc.requeues = sum(sum(lane.requeues.values()) for lane in lanes)
        raise
    finally:
        for lane in lanes:
            _close_lane(lane)


def evaluation_summary(records, requested=None):
    """Use the original evaluator's outcome denominators and fault/cut accounting."""
    summary = summarize(records, len(records) if requested is None else requested, Counter(), 0)
    summary["requeues"] = sum(r.get("requeues", 0) for r in records)
    return summary


def compare_records(reference, candidate):
    """Require identical cohorts and report *every* differing trajectory."""
    left = {r["world_seed"]: r for r in reference}
    right = {r["world_seed"]: r for r in candidate}
    if len(left) != len(reference) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("Comparisons require identical unique world cohorts")
    differences = []
    for seed, row in right.items():
        base = left[seed]
        if row["action_hash"] != base["action_hash"] or row["status"] != base["status"]:
            differences.append({"world_seed": seed, "reference_status": base["status"],
                                "candidate_status": row["status"], **row.get("first_divergence", {})})
    return {"compared": len(left), "matching": len(left) - len(differences), "differences": differences}
