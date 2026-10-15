"""Joint public-policy PPO with binary full-game returns and independent critics.

This adapter owns its reward/optimisation contract; sts_env is unchanged.
"""

from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import resource
import signal
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sts_policy.losses import PPOConfig, masked_distribution, masked_kl_divergence
from sts_policy.model_agent import collate_public, forward_policy
from sts_policy.rollout_training import collect_rollouts
from sts_policy.workers import CurriculumWorker

from .common import (
    WORKER_CONFIG,
    FilteringWorker,
    JsonlLog,
    load_policy,
    save_atomic,
    train_tasks,
)
from .recording import (
    begin_run,
    capture_rng,
    file_identity,
    restore_rng,
    write_collection,
)
from .packed_rows import PackedRows, concatenate_rows

PHASES = ("combat", "ooc")


@dataclass(frozen=True)
class JointConfig:
    out: str
    checkpoint: str
    hours: float = 3.0
    games: int = 768
    max_games: int = 1536
    min_wins: int = 4
    collectors: int = 2
    collection_backend: str = "legacy"
    evaluation_backend: str = "legacy"
    workers: int = 6
    chunk: int = 16
    combat_lr: float = 1e-5
    ooc_lr: float = 3e-5
    critic_lr: float = 3e-4
    critic_minibatch: int = 0  # zero preserves the historical recipe for replay
    critic_init: str | None = None
    critic_fresh_games: int = 0
    critic_fresh_start: int = 60_020_000_000
    critic_health: bool = False
    buffer_compresslevel: int = 1
    clip: float = 0.1
    entropy: float = 0.001
    anchor_beta: float = 0.1
    local_kl_limit: float = 0.02
    anchor_kl_limit: float = 0.05
    bucket_kl_p95_limit: float = 0.2
    choice_drift_limit: float = 0.2
    combat_temperature: float = 0.25
    ooc_temperature: float = 0.5
    combat_epsilon: float = 0.001
    ooc_epsilon: float = 0.002
    minibatch: int = 4096
    microbatch: int = 128
    eval_n: int = 1024
    eval_every: int = 2
    eval_collectors: int = 4
    train_start: int = 60_000_000_000
    dev_start: int = 61_000_000_000
    confirm_start: int = 61_200_000_000
    anchor_start: int = 63_000_000_000
    anchor_games: int = 128
    anchor_per_bucket: int = 128
    seed: int = 19
    device: str = "cpu"
    max_iterations: int = 10000
    collection_seconds: float = 300.0
    rpc_timeout: float = 15.0
    max_rss_gib: float = 12.0
    min_disk_gib: float = 6.0

    def validate(self):
        if self.collection_backend not in ("legacy", "compact") or self.evaluation_backend not in ("legacy", "lean"):
            raise ValueError("Unknown collection/evaluation backend")
        if (
            min(
                self.hours,
                self.games,
                self.collectors,
                self.workers,
                self.chunk,
                self.microbatch,
                self.eval_n,
                self.anchor_games,
            )
            <= 0
        ):
            raise ValueError("Positive finite budgets/counts required")
        for v in [
            self.hours,
            self.combat_lr,
            self.ooc_lr,
            self.critic_lr,
            self.collection_seconds,
        ]:
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Positive finite settings required")
        if not 0 <= self.combat_epsilon < 1 or not 0 <= self.ooc_epsilon < 1:
            raise ValueError("Invalid exploration mixture")
        if (
            any(
                not math.isfinite(x) or x <= 0
                for x in (
                    self.combat_temperature,
                    self.ooc_temperature,
                    self.rpc_timeout,
                    self.max_rss_gib,
                    self.min_disk_gib,
                )
            )
            or self.min_wins < 1
        ):
            raise ValueError("Invalid temperature")
        if (
            not self.train_start
            < self.dev_start
            < self.confirm_start
            < self.anchor_start
        ):
            raise ValueError("Disjoint ordered namespaces required")
        if self.device not in ("cpu", "mps"):
            raise ValueError("CPU/MPS only")
        if self.max_games < self.games or self.max_games >= 7000:
            raise ValueError("Batch cap would overlap seed spacing")
        if (
            self.critic_minibatch < 0
            or self.critic_fresh_games < 0
            or not 1 <= self.buffer_compresslevel <= 9
        ):
            raise ValueError("Invalid critic/storage settings")
        if self.critic_fresh_games and not self.critic_init:
            raise ValueError("Fresh critic validation requires warm initialization")
        if self.critic_minibatch and not self.critic_init:
            raise ValueError("New critic recipe requires recorded warm initialization")
        if self.minibatch < self.microbatch or self.eval_every < 1:
            raise ValueError("Invalid update schedule")


def mixture_logits(logits, mask, temperature, epsilon):
    if not bool(mask.any(-1).all()):
        raise ValueError("Empty legal menu")
    if not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError("Nonfinite actor logits")
    probabilities = torch.softmax((logits / temperature).masked_fill(~mask, -1e9), -1)
    q = (1 - epsilon) * probabilities + epsilon * mask.to(logits.dtype) / mask.sum(
        -1, keepdim=True
    )
    return q.clamp_min(1e-30).log().masked_fill(~mask, -1e9)


class WinCritics(nn.Module):
    def __init__(self, prior=0.03):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                p: nn.Sequential(
                    nn.Linear(384, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )
                for p in PHASES
            }
        )
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, math.log(prior / (1 - prior)))

    @staticmethod
    def features(batch):
        state, actions, mask = batch
        mean = (actions * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(
            1
        )
        return torch.cat([state, mean], -1)

    def phase_logits(self, phase, batch):
        return self.heads[phase](self.features(batch)).squeeze(-1)

    def forward(self, batch):
        features = self.features(batch)
        combat = batch[0][:, 7] > 0.5
        result = features.new_empty(len(features))
        for phase, selection in [("combat", combat), ("ooc", ~combat)]:
            if bool(selection.any()):
                result[selection] = (
                    self.heads[phase](features[selection]).squeeze(-1).sigmoid()
                )
        return result


class BinaryFuture:
    def __init__(self, future, method):
        self.future, self.method = future, method

    def result(self, timeout=None):
        result = self.future.result(timeout=timeout)
        if self.method != "step":
            return result
        obs, _, terminated, truncated, info = result
        win = bool(
            terminated
            and not truncated
            and info.get("task_outcome") in ("victory", "act_clear")
        )
        info = {
            **info,
            "underlying_reward_version": info.get("reward_version"),
            "reward_version": "joint_sparse_binary_v1",
            "reward_components": {
                "terminal_win": float(win),
                "progress": 0.0,
                "potential": 0.0,
                "terminal_hp": 0.0,
            },
        }
        return obs, float(win), terminated, truncated, info


class BinaryWorker:
    def __init__(self, timeout):
        config = {
            **WORKER_CONFIG,
            "reward_mode": "sparse",
            "progress_floor_weight": 0.0,
            "terminal_hp_weight": 0.0,
        }
        self.worker = FilteringWorker(
            CurriculumWorker(config=config, timeout=timeout),
            dead_actions=True,
            slim=True,
        )

    @property
    def last_fault_path(self):
        return self.worker.last_fault_path

    def submit(self, method, *args):
        return BinaryFuture(self.worker.submit(method, *args), method)

    def close(self):
        self.worker.close()


def freeze_legacy_critics(actor):
    for phase in PHASES:
        for name, p in getattr(actor, phase).named_parameters():
            p.requires_grad_(not name.startswith("critic."))


def actor_forward(actor, critics, cfg, deterministic=False):
    def forward(model, batch):
        logits, _, mask = forward_policy(model, batch)
        if not deterministic:
            phase = batch[0][:, 7] > 0.5
            out = logits.clone()
            for p, sel in [("combat", phase), ("ooc", ~phase)]:
                if bool(sel.any()):
                    out[sel] = mixture_logits(
                        logits[sel],
                        mask[sel],
                        getattr(cfg, p + "_temperature"),
                        getattr(cfg, p + "_epsilon"),
                    )
            logits = out
        return logits, critics(batch), mask

    return forward


def evaluation_forward(model, batch):
    """Exact greedy actor logits; neither value head participates in eval."""
    logits = model.actor_logits(*batch)
    return logits, logits.new_zeros(len(logits)), batch[2]


def classify_episode(episode):
    if episode.get("truncated") or not episode.get("terminated"):
        return "censored"
    if episode.get("task_outcome") in ("victory", "act_clear"):
        return "win"
    if episode.get("task_outcome") == "defeat":
        return "loss"
    return "invalid"


class SparseEpisodeSink:
    """Keep one world's numeric choices, never its full observation history.

    Until finish, outcomes are unknown and all rows remain quarantinable. Only
    genuine wins/losses contribute choices, boss entries and optimization rows,
    matching compact_chunk. Censored/fault worlds never leak partial samples.
    """
    def __init__(self, keep_rows=True):
        self.keep_rows = keep_rows
        self.rows = []
        self.counts, self.choices, self.buckets = Counter(), Counter(), Counter()
        self.boss_entries = {}

    def record(self, t):
        phase = "combat" if t.observation.get("phase") == "combat" else "ooc"
        state = t.observation.get("state", {})
        act = int(state.get("act") or 0)
        room = str(state.get("room", "")).rsplit(".", 1)[-1]
        if phase == "combat" and room == "BOSS":
            self.boss_entries.setdefault(str(act), {"hp": state.get("cur_hp"), "floor": state.get("floor")})
        if len(t.behavior_mask) <= 1:
            return
        from .ooc_trainer import _label
        label = t.observation["legal_actions"][t.action_position]["description"]
        self.choices[phase + ":" + _label(label)] += 1
        self.counts[phase] += 1
        self.buckets[f"{phase}:act{act}"] += 1
        if self.keep_rows:
            e = t.encoded_observation
            if e is None or e.action_indices[t.action_position] != t.action_index:
                raise ValueError("Missing encoding or stored action/menu mismatch")
            self.rows.append({"state": e.state_features, "actions": e.action_features,
                              "position": t.action_position, "logp": t.behavior_log_prob,
                              "value": t.behavior_value, "return": 0., "advantage": 0.,
                              "seed": t.world_seed, "phase": phase, "act": act,
                              "room": room, "floor": state.get("floor"), "action_index": t.action_index})

    def finish(self, episode):
        outcome = classify_episode(episode)
        valid = outcome in ("win", "loss")
        win = float(outcome == "win")
        if valid and abs(episode["reward"] - win) > 1e-6:
            raise ValueError("Reward contract violated")
        episode.update(dataset="train", classification=outcome,
                       binary_return=win if valid else None,
                       decision_counts=dict(self.buckets) if valid else {},
                       boss_entries=self.boss_entries if valid else {})
        if not valid:
            return {"rows": [], "counts": {}, "choices": {}, "buckets": {}}
        for row in self.rows:
            row["return"], row["advantage"] = win, win - row["value"]
        return {"rows": self.rows, "counts": dict(self.counts),
                "choices": dict(self.choices), "buckets": dict(self.buckets)}


def compact_stream_chunk(result):
    rows = PackedRows.from_rows([r for e in result["episode_data"] for r in e["rows"]])
    counters = [Counter(), Counter(), Counter()]
    for e in result["episode_data"]:
        for counter, key in zip(counters, ("counts", "choices", "buckets")):
            counter.update(e[key])
    # Reuse the legacy fail-closed programming-error check without transitions.
    compact_chunk({"episodes": [], "transitions": [], "faults": result["faults"]})
    return rows, result["episodes"], *(dict(c) for c in counters)


def compact_chunk(result, keep_rows=True):
    outcomes = {e["world_seed"]: classify_episode(e) for e in result["episodes"]}
    rows = []
    counts = Counter()
    choices = Counter()
    buckets = Counter()
    boss_entries = {}
    world_counts = defaultdict(Counter)
    for t in result["transitions"]:
        if outcomes.get(t.world_seed) not in ("win", "loss"):
            continue
        phase = "combat" if t.observation.get("phase") == "combat" else "ooc"
        state = t.observation.get("state", {})
        act = int(state.get("act") or 0)
        room = str(state.get("room", "")).rsplit(".", 1)[-1]
        if phase == "combat" and room == "BOSS":
            boss_entries.setdefault(
                (t.world_seed, act),
                {"hp": state.get("cur_hp"), "floor": state.get("floor")},
            )
        if len(t.behavior_mask) <= 1:
            continue
        label = t.observation["legal_actions"][t.action_position]["description"]
        from .ooc_trainer import _label

        choices[phase + ":" + _label(label)] += 1
        buckets[f"{phase}:act{act}"] += 1
        counts[phase] += 1
        world_counts[t.world_seed][f"{phase}:act{act}"] += 1
        if keep_rows:
            e = t.encoded_observation
            if e is None:
                raise ValueError("Collection did not retain exact public encodings")
            if e.action_indices[t.action_position] != t.action_index:
                raise ValueError("Stored action/menu mismatch")
            win = float(outcomes[t.world_seed] == "win")
            rows.append(
                {
                    "state": e.state_features,
                    "actions": e.action_features,
                    "position": t.action_position,
                    "logp": t.behavior_log_prob,
                    "value": t.behavior_value,
                    "return": win,
                    "advantage": win - t.behavior_value,
                    "seed": t.world_seed,
                    "phase": phase,
                    "act": act,
                    "room": room,
                    "floor": state.get("floor"),
                    "action_index": t.action_index,
                }
            )
    episodes = []
    for e in result["episodes"]:
        row = {
            **e,
            "dataset": "train",
            "classification": outcomes[e["world_seed"]],
            "binary_return": float(outcomes[e["world_seed"]] == "win")
            if outcomes[e["world_seed"]] in ("win", "loss")
            else None,
        }
        if (
            row["classification"] in ("win", "loss")
            and abs(e["reward"] - row["binary_return"]) > 1e-6
        ):
            raise ValueError("Reward contract violated")
        row["decision_counts"] = dict(world_counts[e["world_seed"]])
        row["boss_entries"] = {
            str(act): v
            for (seed, act), v in boss_entries.items()
            if seed == e["world_seed"]
        }
        episodes.append(row)
    for f in result["faults"]:
        if f["error"] not in (
            "SimulatorError",
            "TimeoutError",
            "EOFError",
            "BrokenPipeError",
            "ConnectionResetError",
        ):
            raise RuntimeError("Unexpected collector/programming error: " + str(f))
    return rows, episodes, dict(counts), dict(choices), dict(buckets)


def _collector(conn, checkpoint, cfg_dict, workers, index):
    cfg = JointConfig(**cfg_dict)
    torch.set_num_threads(1)
    actor, _ = load_policy(checkpoint)
    critics = WinCritics()
    pool = [BinaryWorker(cfg.rpc_timeout) for _ in range(workers)]
    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break
            if request["kind"] == "close":
                break
            try:
                actor.load_state_dict(request["actor"])
                critics.load_state_dict(request["critics"])
                actor.eval()
                critics.eval()
                torch.manual_seed(request["seed"])
                row_parts = []
                episodes = []
                faults = []
                counts = Counter()
                choices = Counter()
                buckets = Counter()
                start = time.monotonic()
                encoding_bytes = 0
                timings = Counter()
                forward_counts = Counter()
                lean = request["kind"] == "eval" and cfg.evaluation_backend == "lean"
                streaming = lean or cfg.collection_backend == "compact"
                for offset in range(0, len(request["tasks"]), cfg.chunk):
                    chunk = request["tasks"][offset : offset + cfg.chunk]
                    r = collect_rollouts(
                        actor,
                        pool,
                        chunk,
                        collate=collate_public,
                        forward=evaluation_forward if lean else actor_forward(
                            actor, critics, cfg, request["kind"] != "train"
                        ),
                        config=PPOConfig(gamma=1.0, gae_lambda=1.0),
                        max_seconds=cfg.collection_seconds,
                        deterministic=request["kind"] != "train",
                        deferred_bootstrap=True,
                        reuse_encodings=not lean,
                        episode_sink_factory=(lambda: SparseEpisodeSink(request["kind"] != "eval")) if streaming else None,
                        actor_only=lean,
                    )
                    if streaming:
                        compact, eps, c, ch, b = compact_stream_chunk(r)
                    else:
                        compact, eps, c, ch, b = compact_chunk(r, keep_rows=request["kind"] != "eval")
                    row_parts.append(compact)
                    episodes.extend(eps)
                    faults.extend(r["faults"])
                    counts.update(c)
                    choices.update(ch)
                    buckets.update(b)
                    encoding_bytes += r["retained_encoding_bytes"]
                    timings.update(r["timings"])
                    forward_counts.update(r["forward_counts"])
                    write_collection(
                        Path(cfg.out) / f"progress_receipt_{index}.json.gz",
                        {
                            "tasks": [t.to_dict() for t in request["tasks"]],
                            "episodes": episodes,
                            "faults": faults,
                            "kind": request["kind"],
                            "cohort_complete": offset + len(chunk)
                            == len(request["tasks"]),
                        },
                    )
                    conn.send(
                        (
                            "progress",
                            {
                                "collector": index,
                                "completed": len(episodes) + len(faults),
                                "requested": len(request["tasks"]),
                                "elapsed": time.monotonic() - start,
                            },
                        )
                    )
                    if not r["cohort_completed"]:
                        partial = (
                            Path(cfg.out) / f"partial_{index}_{time.time_ns()}.json.gz"
                        )
                        write_collection(
                            partial,
                            {
                                "tasks": [t.to_dict() for t in request["tasks"]],
                                "episodes": episodes,
                                "faults": faults,
                                "reason": "incomplete_chunk",
                            },
                        )
                        raise RuntimeError(
                            f"Incomplete declared chunk; receipt preserved in {partial}"
                        )
                    del r, compact
                    gc.collect()
                conn.send(
                    (
                        "ok",
                        {
                            "rows": concatenate_rows(row_parts),
                            "episodes": episodes,
                            "faults": faults,
                            "counts": dict(counts),
                            "choices": dict(choices),
                            "buckets": dict(buckets),
                            "elapsed": time.monotonic() - start,
                            "encoding_bytes": encoding_bytes,
                            "timings": dict(timings),
                            "forward_counts": dict(forward_counts),
                        },
                    )
                )
            except Exception:  # noqa: BLE001 — propagate every collector failure to the owner
                import traceback

                conn.send(("error", traceback.format_exc()))
    finally:
        for worker in pool:
            worker.close()


class SparseCollectors:
    def __init__(self, cfg, collectors=None, workers=None):
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        n = cfg.collectors if collectors is None else collectors
        if n < 1 or (workers is not None and workers < 1):
            raise ValueError("Positive collector/worker counts required")
        total = max(n, workers if workers is not None else cfg.workers)
        self.worker_counts = [total // n + int(i < total % n) for i in range(n)]
        for i in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_collector, args=(child, cfg.checkpoint, asdict(cfg), self.worker_counts[i], i)
            )
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def collect(self, actor, critics, tasks, kind, seed, progress):
        if len({t.seed for t in tasks}) != len(tasks) or any(
            t.kind != "run" or t.max_act != 3 for t in tasks
        ):
            raise ValueError("Only unique fresh full-game tasks are supported")
        write_collection(
            Path(self.cfg.out) / f"request_{time.time_ns()}.json.gz",
            {
                "kind": kind,
                "tasks": [t.to_dict() for t in tasks],
                "policy_sha256": identity_state(actor),
                "critic_sha256": identity_state(critics),
                "collector_seeds": [seed * 1009 + i for i in range(len(self.conns))],
                "assignment": "tasks[i::collectors], fixed chunks",
                "worker_counts": self.worker_counts,
                "result_order": "collector index, then local episode completion" if self.cfg.collection_backend == "compact" else "legacy collector completion order",
                "config": asdict(self.cfg),
            },
        )
        actor_state = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        critic_state = {k: v.detach().cpu() for k, v in critics.state_dict().items()}
        for i, conn in enumerate(self.conns):
            conn.send(
                {
                    "kind": kind,
                    "actor": actor_state,
                    "critics": critic_state,
                    "tasks": tasks[i :: len(self.conns)],
                    "seed": seed * 1009 + i,
                }
            )
        waiting = set(range(len(self.conns)))
        parts = {}
        last = {i: time.monotonic() for i in waiting}
        completed = {i: 0 for i in waiting}
        while waiting:
            if any(
                time.monotonic() - last[i] > self.cfg.collection_seconds
                for i in waiting
            ):
                raise TimeoutError("Collector progress deadline exceeded")
            for i in list(waiting):
                if not self.procs[i].is_alive() and not self.conns[i].poll():
                    raise RuntimeError("Collector exited unexpectedly")
                if self.conns[i].poll(0.05):
                    status, data = self.conns[i].recv()
                    last[i] = time.monotonic()
                    if status == "progress":
                        completed[i] = data["completed"]
                        data.update(
                            requested_total=len(tasks),
                            completed_total=sum(completed.values()),
                            kind=kind,
                        )
                        progress(data)
                    elif status == "error":
                        raise RuntimeError(data)
                    else:
                        parts[i] = data
                        waiting.remove(i)
        out = {
            "rows": [],
            "episodes": [],
            "faults": [],
            "counts": Counter(),
            "choices": Counter(),
            "buckets": Counter(),
            "elapsed": max(p["elapsed"] for p in parts.values()),
            "encoding_bytes": 0,
            "timings": Counter(),
            "forward_counts": Counter(),
            "collector_seconds": {i: p["elapsed"] for i, p in parts.items()},
        }
        order = sorted(parts) if self.cfg.collection_backend == "compact" else list(parts)
        out["rows"] = concatenate_rows([parts[i]["rows"] for i in order])
        for i in order:
            p = parts[i]
            for key in ["episodes", "faults"]:
                out[key].extend(p[key])
            for key in ["counts", "choices", "buckets"]:
                out[key].update(p[key])
            out["encoding_bytes"] += p["encoding_bytes"]
            out["timings"].update(p["timings"])
            out["forward_counts"].update(p["forward_counts"])
        for key in ["counts", "choices", "buckets", "timings", "forward_counts"]:
            out[key] = dict(out[key])
        if len(out["episodes"]) + len(out["faults"]) != len(tasks):
            raise ValueError("Requested/accounted cohort mismatch")
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send({"kind": "close"})
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for c in self.conns:
            c.close()


def batch_rows(rows, device="cpu"):
    width = max(len(r["actions"]) for r in rows)
    state = np.stack([r["state"] for r in rows])
    actions = np.zeros((len(rows), width, 128), np.float32)
    mask = np.zeros((len(rows), width), bool)
    for i, r in enumerate(rows):
        n = len(r["actions"])
        actions[i, :n] = r["actions"]
        mask[i, :n] = True
    return tuple(torch.from_numpy(x).to(device) for x in (state, actions, mask))


def identity_state(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def wilson(wins, n):
    if not n:
        return None
    z = 1.96
    p = wins / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - radius, mid + radius]


def summary_episodes(episodes, faults, requested):
    c = Counter(e["classification"] for e in episodes)
    valid = c["win"] + c["loss"]
    floors = [e["floor"] for e in episodes if e["classification"] in ("win", "loss")]
    return {
        "requested": requested,
        "valid": valid,
        "wins": c["win"],
        "losses": c["loss"],
        "faults": len(faults),
        "censored": c["censored"],
        "invalid": c["invalid"],
        "win_rate_requested": c["win"] / requested if requested else None,
        "win_rate_valid": c["win"] / valid if valid else None,
        "win_ci95": wilson(c["win"], valid),
        "mean_floor": float(np.mean(floors)) if floors else None,
        "act1_clears": sum(
            e["max_act_reached"] >= 2
            for e in episodes
            if e["classification"] in ("win", "loss")
        ),
        "act2_clears": sum(
            e["max_act_reached"] >= 3
            for e in episodes
            if e["classification"] in ("win", "loss")
        ),
        "steps_mean": float(np.mean([e["steps"] for e in episodes]))
        if episodes
        else None,
        "steps_p95": float(np.quantile([e["steps"] for e in episodes], 0.95))
        if episodes
        else None,
        "floor_p10": float(np.quantile(floors, 0.1)) if floors else None,
        "floor_p90": float(np.quantile(floors, 0.9)) if floors else None,
        "ending_act": dict(Counter(str(e.get("act")) for e in episodes)),
    }


def phase_distribution(head, rows, cfg, phase, device):
    batch = batch_rows(rows, device)
    logits, _ = head(*batch)
    return (
        masked_distribution(
            mixture_logits(
                logits,
                batch[2],
                getattr(cfg, phase + "_temperature"),
                getattr(cfg, phase + "_epsilon"),
            ),
            batch[2],
        ),
        logits,
        batch,
    )


@torch.no_grad()
def drift_metrics(actor, reference, anchors, device):
    out = {}
    for phase in PHASES:
        phase_rows = [r for r in anchors if r["phase"] == phase]
        if not phase_rows:
            continue
        values = []
        for start in range(0, len(phase_rows), 128):
            rows = phase_rows[start : start + 128]
            batch = batch_rows(rows, device)
            a, _ = getattr(actor, phase)(*batch)
            b, _ = getattr(reference, phase)(*batch)
            kl = masked_kl_divergence(b, a, batch[2]).cpu().tolist()
            changes = (a.argmax(-1) != b.argmax(-1)).cpu().tolist()
            for r, k, ch in zip(rows, kl, changes):
                values.append((r["act"], k, float(ch)))
        for act in [0, 1, 2, 3]:
            vals = values if act == 0 else [v for v in values if v[0] == act]
            if vals:
                out[f"{phase}:act{act}"] = {
                    "n": len(vals),
                    "kl_mean": float(np.mean([x[1] for x in vals])),
                    "kl_p95": float(np.quantile([x[1] for x in vals], 0.95)),
                    "argmax_change": float(np.mean([x[2] for x in vals])),
                }
    return out


def hard_drift(drift, cfg):
    return any(
        m["n"] >= 16
        and (
            m["kl_mean"] > cfg.anchor_kl_limit
            or m["kl_p95"] > cfg.bucket_kl_p95_limit
            or m["argmax_change"] > cfg.choice_drift_limit
        )
        for m in drift.values()
    )


def actor_update(actor, reference, optimizers, rows, anchors, cfg, rng, beta):
    device = cfg.device
    actor.to(device)
    reference.to(device)
    stats = {}
    attempted_steps = 0
    attempted_drift = None
    diagnostic = {}
    rng_snapshot = capture_rng(rng)
    snap = {p: copy.deepcopy(getattr(actor, p).state_dict()) for p in PHASES}
    opts = {p: copy.deepcopy(optimizers[p].state_dict()) for p in PHASES}
    try:
        for optimizer in optimizers.values():
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != "step":
                        state[key] = value.to(device)
        for phase in PHASES:
            head = getattr(actor, phase)
            ref = getattr(reference, phase)
            data = [r for r in rows if r["phase"] == phase]
            anchor = [r for r in anchors if r["phase"] == phase]
            if not data:
                raise ValueError("Missing phase learning data")
            raw = np.array([r["advantage"] for r in data], np.float64)
            mean, std = float(raw.mean()), max(float(raw.std()), 1e-8)
            order = list(range(len(data)))
            rng.shuffle(order)
            metrics = []
            steps = 0
            before = identity_state(head)
            for begin in range(0, len(order), cfg.minibatch):
                indices = order[begin : begin + cfg.minibatch]
                optimizer = optimizers[phase]
                optimizer.zero_grad(set_to_none=True)
                total = defaultdict(float)
                for offset in range(0, len(indices), cfg.microbatch):
                    part = [data[i] for i in indices[offset : offset + cfg.microbatch]]
                    dist, logits, batch = phase_distribution(
                        head, part, cfg, phase, device
                    )
                    actions = torch.tensor([r["position"] for r in part], device=device)
                    old = torch.tensor([r["logp"] for r in part], device=device)
                    adv = torch.tensor(
                        [(r["advantage"] - mean) / std for r in part],
                        device=device,
                        dtype=torch.float32,
                    )
                    logratio = dist.log_prob(actions) - old
                    ratio = logratio.exp()
                    if not bool(torch.isfinite(ratio).all()):
                        raise FloatingPointError("Nonfinite importance ratio")
                    policy = -torch.minimum(
                        ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                    ).mean()
                    entropy = dist.entropy().mean()
                    sampled = rng.sample(anchor, min(32, len(anchor)))
                    ab = batch_rows(sampled, device)
                    al, _ = head(*ab)
                    with torch.no_grad():
                        rl, _ = ref(*ab)
                    kl = masked_kl_divergence(rl, al, ab[2]).mean()
                    loss = policy + beta * kl - cfg.entropy * entropy
                    if not bool(torch.isfinite(loss)):
                        diagnostic = {
                            "phase": phase,
                            "minibatch": begin,
                            "microbatch": offset,
                            "loss_components": {
                                name: float(value.detach())
                                if bool(torch.isfinite(value))
                                else str(float(value.detach()))
                                for name, value in {
                                    "policy": policy,
                                    "anchor_kl": kl,
                                    "entropy": entropy,
                                }.items()
                            },
                            "anchor_seeds": [r["seed"] for r in sampled],
                        }
                        raise FloatingPointError("Nonfinite actor loss")
                    weight = len(part) / len(indices)
                    (loss * weight).backward()
                    for k, v in {
                        "policy_loss": policy,
                        "anchor_loss": kl,
                        "entropy": entropy,
                        "approx_kl": ((ratio - 1) - logratio).mean(),
                        "clip_fraction": ((ratio - 1).abs() > cfg.clip).float().mean(),
                        "ratio_mean": ratio.mean(),
                        "ratio_max": ratio.max(),
                        "ratio_second": ratio.square().mean(),
                        "effective_sample_fraction": ratio.mean().square()
                        / ratio.square().mean().clamp_min(1e-20),
                        "top_probability": dist.probs.max(-1).values.mean(),
                    }.items():
                        total[k] += float(v.detach()) * weight
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in head.parameters() if p.requires_grad], 0.5
                    )
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite actor gradients")
                optimizer.step()
                steps += 1
                attempted_steps += 1
                total["gradient_norm"] = norm
                metrics.append(dict(total))
                diagnostic = {
                    "phase": phase,
                    "minibatch": begin,
                    "approx_kl": total["approx_kl"],
                    "gradient_norm": norm,
                }
                if total["approx_kl"] > cfg.local_kl_limit:
                    raise ValueError("Local KL budget exceeded")
            # Reconstruct actual generating q across the whole buffer before first update
            # and evaluate post-update trust on the same immutable samples.
            local = []
            with torch.no_grad():
                for start in range(0, len(data), cfg.microbatch):
                    part = data[start : start + cfg.microbatch]
                    dist, _, _ = phase_distribution(head, part, cfg, phase, device)
                    old = torch.tensor([r["logp"] for r in part], device=device)
                    positions = torch.tensor(
                        [r["position"] for r in part], device=device
                    )
                    lr = dist.log_prob(positions) - old
                    local.extend(((lr.exp() - 1) - lr).cpu().tolist())
            diagnostic.update(phase=phase, post_kl=float(np.mean(local)))
            if float(np.mean(local)) > cfg.local_kl_limit:
                raise ValueError("Post-update local KL budget exceeded")
            stats[phase] = {
                "samples": len(data),
                "steps": steps,
                "advantage_mean": mean,
                "advantage_std": std,
                "return_mean": float(np.mean([r["return"] for r in data])),
                "return_p95": float(np.quantile([r["return"] for r in data], 0.95)),
                "advantage_p05": float(np.quantile(raw, 0.05)),
                "advantage_p95": float(np.quantile(raw, 0.95)),
                "worlds": len(set(r["seed"] for r in data)),
                "winning_decision_fraction": float(
                    np.mean([r["return"] for r in data])
                ),
                "world_choice_p95": float(
                    np.quantile(list(Counter(r["seed"] for r in data).values()), 0.95)
                ),
                "post_kl": float(np.mean(local)),
                "actor_changed": before != identity_state(head),
                "weight_l2_change": math.sqrt(
                    sum(
                        float((value.detach() - snap[phase][key]).square().sum())
                        for key, value in head.state_dict().items()
                    )
                ),
                **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
            }
        drift = drift_metrics(actor, reference, anchors, device)
        attempted_drift = drift
        if hard_drift(drift, cfg):
            raise ValueError("Original-policy hard drift budget exceeded")
        return {
            "accepted": True,
            "phases": stats,
            "drift": drift,
            "attempted_steps": attempted_steps,
            "retained_steps": sum(v["steps"] for v in stats.values()),
        }
    except (ValueError, FloatingPointError) as error:
        for phase in PHASES:
            getattr(actor, phase).load_state_dict(snap[phase])
            optimizers[phase].load_state_dict(opts[phase])
        restore_rng({"rng_state": rng_snapshot}, rng)
        return {
            "accepted": False,
            "attempted_steps": attempted_steps,
            "reason": str(error),
            "attempted_phases": stats,
            "attempted_drift": attempted_drift,
            "failure_diagnostic": diagnostic,
            "retained_steps": 0,
            "drift": drift_metrics(actor, reference, anchors, device),
        }
    finally:
        actor.cpu()
        reference.cpu()


def likelihood_error(actor, rows, cfg):
    error = 0.0
    with torch.no_grad():
        for phase in PHASES:
            data = [r for r in rows if r["phase"] == phase]
            for start in range(0, len(data), cfg.microbatch):
                part = data[start : start + cfg.microbatch]
                dist, _, _ = phase_distribution(
                    getattr(actor, phase), part, cfg, phase, "cpu"
                )
                logp = dist.log_prob(torch.tensor([r["position"] for r in part]))
                old = torch.tensor([r["logp"] for r in part])
                error = max(error, float((logp - old).abs().max()))
    return error


def critic_update(critics, optimizer, rows, cfg, history=None, iteration=0):
    if cfg.critic_minibatch:
        from . import critic_training as ct

        data = ct.pack(critics, rows, batch_rows)
        before = ct.metrics(critics, data)
        update = ct.fit(
            critics, optimizer, data, cfg.critic_minibatch, cfg.seed + iteration
        )
        fitted = ct.metrics(critics, data, "fit")
        if history is not None:
            history.append(ct.subset(data, data["held"]))
            del history[:-4]
            held = ct.metrics(critics, ct.combine(history), "all")
        else:
            held = ct.metrics(critics, data)
        return {
            **update,
            "behavior_held_calibration": before,
            "fit_calibration": fitted,
            "held_calibration": held,
            "health": ct.health(held, fitted),
        }

    selected = []
    for phase in PHASES:
        byworld = defaultdict(list)
        for r in rows:
            if r["phase"] == phase:
                byworld[r["seed"]].append(r)
        for seed, rs in byworld.items():
            selected.extend(rs[:: max(1, len(rs) // 8)][:8])
    counts = Counter((r["seed"], r["phase"]) for r in selected)
    total = 0.0
    fit = 0
    optimizer.zero_grad(set_to_none=True)
    for phase in PHASES:
        training = [
            r
            for r in selected
            if r["phase"] == phase
            and int(hashlib.sha256(str(r["seed"]).encode()).hexdigest()[:8], 16) % 5
        ]
        worlds = len(set(r["seed"] for r in training))
        for start in range(0, len(training), 256):
            part = training[start : start + 256]
            target = torch.tensor([r["return"] for r in part])
            prediction = critics.phase_logits(phase, batch_rows(part))
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction, target, reduction="none"
            )
            weights = torch.tensor(
                [1 / counts[r["seed"], phase] / max(1, worlds) for r in part]
            )
            loss = (losses * weights).sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite critic loss")
            loss.backward()
            total += float(loss.detach())
            fit += len(part)
    norm = float(torch.nn.utils.clip_grad_norm_(critics.parameters(), 1.0))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite critic gradient")
    optimizer.step()
    diagnostics = {}
    with torch.no_grad():
        for phase in PHASES:
            data = [
                r
                for r in selected
                if r["phase"] == phase
                and not int(hashlib.sha256(str(r["seed"]).encode()).hexdigest()[:8], 16)
                % 5
            ]
            predictions = []
            for start in range(0, len(data), 256):
                predictions.extend(
                    critics.phase_logits(phase, batch_rows(data[start : start + 256]))
                    .sigmoid()
                    .tolist()
                )
            for act in [0, 1, 2, 3]:
                pairs = [
                    (r, p)
                    for r, p in zip(data, predictions)
                    if not act or r["act"] == act
                ]
                if not pairs:
                    continue
                y = np.array([r["return"] for r, p in pairs])
                v = np.array([p for r, p in pairs])
                variance = float(y.var())
                diagnostics[f"{phase}:act{act}"] = {
                    "n": len(pairs),
                    "worlds": len(set(r["seed"] for r, p in pairs)),
                    "predicted_mean": float(v.mean()),
                    "return_mean": float(y.mean()),
                    "brier": float(np.mean((v - y) ** 2)),
                    "explained_variance": 1 - float(np.var(y - v)) / variance
                    if variance > 0
                    else None,
                }
    return {
        "loss": total,
        "fit_states": fit,
        "gradient_norm": norm,
        "held_calibration": diagnostics,
    }


class ActiveClock:
    def __init__(self, path, limit, used=0.0):
        self.path = Path(path)
        self.limit = limit
        self.used = used
        self.gaps = 0
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        if self.path.exists():
            self.used = max(
                self.used, json.loads(self.path.read_text()).get("used_seconds", 0.0)
            )
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        last = time.monotonic()
        while not self.stop_event.wait(1):
            now = time.monotonic()
            delta = now - last
            last = now
            with self.lock:
                if delta < 5:
                    self.used += delta
                else:
                    self.gaps += 1
                self.write()

    def write(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "used_seconds": self.used,
                    "limit_seconds": self.limit,
                    "excluded_heartbeat_gaps": self.gaps,
                    "updated": time.time(),
                }
            )
        )
        temp.replace(self.path)

    def remaining(self):
        with self.lock:
            return max(0.0, self.limit - self.used)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        self.write()


def save_run(
    actor,
    template,
    critics,
    optimizers,
    critic_optimizer,
    cfg,
    record,
    iteration,
    actor_steps,
    rng,
    clock,
    path,
    adaptive=None,
):
    payload = dict(template)
    payload["model_state"] = {
        k: v.detach().cpu().clone() for k, v in actor.state_dict().items()
    }
    payload["optimizer_state"] = {p: o.state_dict() for p, o in optimizers.items()}
    payload["critic_state"] = critics.state_dict()
    payload["critic_optimizer_state"] = critic_optimizer.state_dict()
    payload["iteration"] = iteration
    payload["step"] = actor_steps
    payload["steps_total"] = actor_steps
    payload["rng_state"] = capture_rng(rng)
    payload["used_seconds"] = clock.used
    payload["adaptive_state"] = adaptive or {}
    payload["metadata"] = {
        "encoder_version": "public_candidate_v2",
        "stage": "joint_sparse_win_v2"
        if cfg.critic_minibatch
        else "joint_sparse_win_v1",
        "config": asdict(cfg),
        "run_record": record,
        "anchors": file_identity(Path(cfg.out) / "anchors.pt")
        if (Path(cfg.out) / "anchors.pt").exists()
        else None,
        "parent_policy_checkpoint": file_identity(cfg.checkpoint),
        "historical_parent_metadata": template.get("metadata", {}),
        "reward_contract": "Complete game win1/loss0; gamma/lambda1, invalid worlds quarantined; no shaping/native search.",
        "critic_contract": "Independent public win critics; legacy actor critic parameters frozen.",
        "critic_initialization": file_identity(cfg.critic_init)
        if cfg.critic_init
        else None,
    }
    save_atomic(payload, path)


def save_buffer(payload, path, compresslevel=1):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=compresslevel) as f:
        torch.save(payload, f)
    os.replace(temporary, path)


def build_anchors(rows, limit, seed):
    rng = random.Random(seed)
    packed = isinstance(rows, PackedRows)
    rows = list(rows)
    rng.shuffle(rows)
    counts = Counter()
    out = []
    for r in rows:
        key = (r["phase"], r["act"])
        if counts[key] < limit:
            # Selected views must not keep an entire packed collection alive.
            out.append({**r, "state": r["state"].copy(), "actions": r["actions"].copy()} if packed else r)
            counts[key] += 1
    if any(counts[p, a] < 16 for p in PHASES for a in [1, 2, 3]):
        raise ValueError(
            "Insufficient phase/act anchor coverage; increase declared anchor collection"
        )
    return out


def evaluation_records(result):
    return [
        {
            "seed": e["world_seed"],
            "floor": e["floor"],
            "win": e["classification"] == "win",
            "steps": e["steps"],
            "boss16_hp": e["boss_entries"].get("1", {}).get("hp"),
            "boss33_hp": e["boss_entries"].get("2", {}).get("hp"),
            "boss50_hp": e["boss_entries"].get("3", {}).get("hp"),
        }
        for e in result["episodes"]
        if e["classification"] in ("win", "loss")
    ]


def validate_resume_identity(payload, cfg, current):
    if payload["metadata"]["config"] != asdict(cfg):
        raise ValueError("Resume config differs; create a new condition")
    prior = json.loads(Path(payload["metadata"]["run_record"]["path"]).read_text())
    for name in ("runtime",):
        if prior[name] != current[name]:
            raise ValueError(f"Resume {name} changed")
    for key in ("source_archive", "parent_checkpoint"):
        if prior[key]["sha256"] != current[key]["sha256"]:
            raise ValueError(f"Resume {key} changed")
    if prior["native"]["module_sha256"] != current["native"]["module_sha256"]:
        raise ValueError("Resume native binary changed")
    anchor = payload["metadata"].get("anchors")
    if anchor and file_identity(anchor["path"])["sha256"] != anchor["sha256"]:
        raise ValueError("Resume rehearsal corpus changed")


def run(cfg, resume=False):
    import fcntl
    import shutil

    cfg.validate()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "owner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "latest.pt").exists() and not resume:
        raise ValueError("Existing run requires explicit resume")
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    record = begin_run(
        out,
        asdict(cfg),
        cfg.checkpoint,
        inputs=[cfg.critic_init] if cfg.critic_init else [],
    )
    attempt = record["sha256"][:12]
    if not (out / "config.json").exists():
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    actor, template = load_policy(cfg.checkpoint)
    reference, _ = load_policy(cfg.checkpoint)
    freeze_legacy_critics(actor)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    critics = WinCritics()
    assert not (
        {id(p) for p in actor.parameters()} & {id(p) for p in critics.parameters()}
    )
    optimizers = {
        p: torch.optim.AdamW(
            [v for v in getattr(actor, p).parameters() if v.requires_grad],
            lr=getattr(cfg, p + "_lr"),
            weight_decay=0.0,
        )
        for p in PHASES
    }
    critic_optimizer = torch.optim.AdamW(
        critics.parameters(), lr=cfg.critic_lr, weight_decay=0.01
    )
    if cfg.critic_init and not resume:
        warm = torch.load(cfg.critic_init, map_location="cpu", weights_only=False)
        if (
            warm["warm_record"]["actor_checkpoint"]["sha256"]
            != file_identity(cfg.checkpoint)["sha256"]
        ):
            raise ValueError("Warm critics belong to a different actor parent")
        critics.load_state_dict(warm["critic_state"])
        critic_optimizer.load_state_dict(warm["critic_optimizer_state"])
        if any(group["lr"] != cfg.critic_lr for group in critic_optimizer.param_groups):
            raise ValueError("Warm critic learning rate mismatch")
    critic_history = []
    critic_bad_windows = 0
    iteration = actor_steps = 0
    used = 0.0
    beta = cfg.anchor_beta
    insufficient = rejections = 0
    last_best_used = 0.0
    if resume:
        payload = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        validate_resume_identity(
            payload, cfg, json.loads(Path(record["path"]).read_text())
        )
        adaptive = payload.get("adaptive_state", {})
        beta = adaptive.get("beta", beta)
        critic_bad_windows = adaptive.get("critic_bad_windows", 0)
        insufficient = adaptive.get("insufficient", 0)
        rejections = adaptive.get("rejections", 0)
        last_best_used = adaptive.get("last_best_used", 0.0)
        actor.load_state_dict(payload["model_state"])
        critics.load_state_dict(payload["critic_state"])
        for p in PHASES:
            optimizers[p].load_state_dict(payload["optimizer_state"][p])
        critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
        restore_rng(payload, rng)
        iteration = payload["iteration"]
        actor_steps = payload["step"]
        used = payload["used_seconds"]
    clock = ActiveClock(out / "budget.json", cfg.hours * 3600, used)
    log = JsonlLog(out / "log.jsonl")
    log.emit(
        {
            "event": "start",
            "config": asdict(cfg),
            "run_record": record,
            "resume": resume,
            "iteration": iteration,
        }
    )
    control = out / "control.json"
    if not control.exists():
        control.write_text(json.dumps({"action": "run"}))

    def stop_requested():
        return json.loads(control.read_text()).get("action") != "run"

    def status(stage, **extra):
        data = {
            "stage": stage,
            "iteration": iteration,
            "actor_steps": actor_steps,
            "used_seconds": clock.used,
            "remaining_seconds": clock.remaining(),
            "updated": time.time(),
            "pid": os.getpid(),
            **extra,
        }
        temporary = out / "status.tmp"
        temporary.write_text(json.dumps(data))
        temporary.replace(out / "status.json")

    def progress(data):
        status("collecting", **data)
        log.emit({"event": "progress", "iteration": iteration, **data})

    def request_stop(*_):
        control.write_text(json.dumps({"action": "stop", "reason": "operator_signal"}))

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save(path):
        save_run(
            actor,
            template,
            critics,
            optimizers,
            critic_optimizer,
            cfg,
            record,
            iteration,
            actor_steps,
            rng,
            clock,
            path,
            {
                "beta": beta,
                "insufficient": insufficient,
                "rejections": rejections,
                "last_best_used": last_best_used,
                "critic_bad_windows": critic_bad_windows,
            },
        )
        log.emit(
            {
                "event": "checkpoint",
                "iteration": iteration,
                "actor_steps": actor_steps,
                "identity": file_identity(path),
                "actor_sha256": identity_state(actor),
                "attempt": attempt,
            }
        )

    pool = None
    reason = "budget_complete"
    baseline = None
    try:
        evalpool = SparseCollectors(cfg, cfg.eval_collectors, cfg.eval_collectors)
        try:
            if not (out / "baseline.json").exists():
                status("baseline")
                tasks = train_tasks(cfg.dev_start, cfg.eval_n)
                b = evalpool.collect(actor, critics, tasks, "eval", cfg.seed, progress)
                if b["faults"] or any(
                    e["classification"] not in ("win", "loss") for e in b["episodes"]
                ):
                    raise ValueError("Baseline invalid; no training")
                baseline = {
                    "summary": summary_episodes(b["episodes"], b["faults"], len(tasks)),
                    "records": evaluation_records(b),
                    "checkpoint": file_identity(cfg.checkpoint),
                    "cohort": [t.to_dict() for t in tasks],
                }
                write_collection(
                    out / "baseline_receipt.json.gz",
                    {
                        "episodes": [
                            {**x, "dataset": "baseline", "iteration": 0}
                            for x in b["episodes"]
                        ],
                        "faults": b["faults"],
                    },
                )
                (out / "baseline.json").write_text(json.dumps(baseline))
                log.emit(
                    {
                        "event": "evaluation",
                        "iteration": 0,
                        "dataset": "baseline",
                        "summary": baseline["summary"],
                        "paired_wins": None,
                    }
                )
            else:
                baseline = json.loads((out / "baseline.json").read_text())
        finally:
            evalpool.close()
        pool = SparseCollectors(cfg)
        if not (out / "anchors.pt").exists():
            status("anchor_collection")
            tasks = train_tasks(cfg.anchor_start, cfg.anchor_games)
            a = pool.collect(actor, critics, tasks, "anchor", cfg.seed, progress)
            if a["faults"] or any(
                e["classification"] not in ("win", "loss") for e in a["episodes"]
            ):
                raise ValueError("Anchor collection invalid")
            anchors = build_anchors(a["rows"], cfg.anchor_per_bucket, cfg.seed)
            save_atomic(anchors, out / "anchors.pt")
            write_collection(
                out / "anchor_receipt.json.gz",
                {
                    "tasks": [t.to_dict() for t in tasks],
                    "episodes": a["episodes"],
                    "faults": a["faults"],
                },
            )
            log.emit(
                {
                    "event": "anchors",
                    "buckets": dict(
                        Counter(f"{r['phase']}:act{r['act']}" for r in anchors)
                    ),
                    "identity": file_identity(out / "anchors.pt"),
                }
            )
        else:
            anchors = torch.load(out / "anchors.pt", weights_only=False)
        if (
            cfg.critic_fresh_games
            and not (out / "critic_fresh_validation.json").exists()
        ):
            from . import critic_training as ct

            status("critic_fresh_validation")
            tasks = train_tasks(cfg.critic_fresh_start, cfg.critic_fresh_games)
            validation = pool.collect(
                actor, critics, tasks, "train", cfg.seed + 90000, progress
            )
            if validation["faults"] or any(
                e["classification"] not in ("win", "loss")
                for e in validation["episodes"]
            ):
                raise ValueError("Invalid fresh critic validation")
            data = ct.pack(critics, validation["rows"], batch_rows)
            held = ct.metrics(critics, data, "all")
            assessment = ct.health(held)
            report = {
                "metrics": held,
                "health": assessment,
                "summary": summary_episodes(
                    validation["episodes"], validation["faults"], len(tasks)
                ),
                "actor_sha256": identity_state(actor),
                "critic_sha256": identity_state(critics),
                "source_record": record,
            }
            (out / "critic_fresh_validation.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            write_collection(
                out / "critic_fresh_receipt.json.gz",
                {
                    "tasks": [t.to_dict() for t in tasks],
                    "episodes": validation["episodes"],
                    "faults": validation["faults"],
                },
            )
            save_buffer(
                data, out / "critic_fresh_features.pt.gz", cfg.buffer_compresslevel
            )
            log.emit({"event": "critic_validation", "iteration": 0, **report})
            if assessment["urgent"]:
                raise ValueError(
                    "Fresh critic predictions failed health checks before actor training"
                )
            del validation, data
            gc.collect()
        save(out / "latest.pt")
        if not (out / "champion.pt").exists():
            save(out / "champion.pt")
        if not (out / "champion_eval.json").exists():
            (out / "champion_eval.json").write_text(json.dumps(baseline))
        while (
            iteration < cfg.max_iterations
            and clock.remaining() > 300
            and not stop_requested()
        ):
            if shutil.disk_usage(out).free < cfg.min_disk_gib * 2**30:
                reason = "disk_guard"
                break
            if (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                > cfg.max_rss_gib * 2**30
            ):
                reason = "rss_guard"
                break
            iteration += 1
            tasks = train_tasks(cfg.train_start + (iteration - 1) * 10000, cfg.games)
            status("collecting")
            start = time.monotonic()
            result = pool.collect(
                actor, critics, tasks, "train", cfg.seed + iteration, progress
            )
            summary = summary_episodes(result["episodes"], result["faults"], len(tasks))
            rows = result["rows"]
            # Increase signal support only within the prespecified batch cap; no adaptive reward changes.
            if (
                summary["wins"] < cfg.min_wins
                and cfg.max_games > cfg.games
                and clock.remaining() > 600
            ):
                extra_tasks = train_tasks(tasks[-1].seed + 1, cfg.max_games - cfg.games)
                extra = pool.collect(
                    actor,
                    critics,
                    extra_tasks,
                    "train",
                    cfg.seed + iteration + 50000,
                    progress,
                )
                tasks += extra_tasks
                rows = concatenate_rows([rows, extra["rows"]])
                result["episodes"] += extra["episodes"]
                result["faults"] += extra["faults"]
                for key in ("counts", "choices", "buckets"):
                    result[key] = dict(Counter(result[key]) + Counter(extra[key]))
                result["encoding_bytes"] += extra["encoding_bytes"]
                for key in ("timings", "forward_counts"):
                    result[key] = dict(Counter(result[key]) + Counter(extra[key]))
                summary = summary_episodes(
                    result["episodes"], result["faults"], len(tasks)
                )
            for e in result["episodes"]:
                e["iteration"] = iteration
                e["attempt"] = attempt
            write_collection(
                out / f"collection_{iteration:04d}_{attempt}.json.gz",
                {
                    "schema": "joint_sparse_collection_v1",
                    "tasks": [t.to_dict() for t in tasks],
                    "episodes": result["episodes"],
                    "faults": result["faults"],
                    "source_record": record,
                    "reward_contract": "binary win1/loss0",
                },
            )
            log.emit(
                {
                    "event": "collection",
                    "iteration": iteration,
                    "summary": summary,
                    "counts": result["counts"],
                    "choices": result["choices"],
                    "buckets": result["buckets"],
                    "seconds": time.monotonic() - start,
                    "games_per_second": len(tasks)
                    / max(1e-9, time.monotonic() - start),
                    "rows": len(rows),
                    "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 2**30,
                    "disk_free_gib": shutil.disk_usage(out).free / 2**30,
                    "encoded_bytes": result["encoding_bytes"],
                    "collector_timings_sum": result["timings"],
                    "forward_counts": result["forward_counts"],
                    "row_storage": PackedRows.format if isinstance(rows, PackedRows) else "legacy_rows",
                    "row_numeric_bytes": rows.nbytes if isinstance(rows, PackedRows) else None,
                }
            )
            if (
                summary["faults"] > max(2, 0.02 * len(tasks))
                or summary["censored"]
                or summary["invalid"]
            ):
                reason = "invalid_collection_guard"
                break
            if summary["wins"] < cfg.min_wins or summary["losses"] < cfg.min_wins:
                insufficient += 1
                log.emit(
                    {
                        "event": "alert",
                        "iteration": iteration,
                        "reason": "insufficient_sparse_success_support",
                        "summary": summary,
                    }
                )
                if insufficient >= 2:
                    reason = "insufficient_sparse_support"
                    break
                save(out / "latest.pt")
                continue
            insufficient = 0
            error = likelihood_error(actor, rows, cfg)
            if error > 1e-3:
                raise ValueError(
                    f"Behaviour likelihood reconstruction mismatch {error}"
                )
            before = {p: identity_state(getattr(actor, p)) for p in PHASES}
            oldcritics = {p: identity_state(getattr(actor, p).critic) for p in PHASES}
            save(out / f"before_{iteration:04d}_{attempt}.pt")
            save_buffer(
                {
                    "rows": rows,
                    "anchors_identity": file_identity(out / "anchors.pt"),
                    "behavior_actor_sha256": identity_state(actor),
                    "rng_state": capture_rng(rng),
                    "config": asdict(cfg),
                    "beta": beta,
                    "source_record": record,
                },
                out / f"buffer_{iteration:04d}_{attempt}.pt.gz",
                cfg.buffer_compresslevel,
            )
            status("updating")
            start = time.monotonic()
            update = actor_update(
                actor, reference, optimizers, rows, anchors, cfg, rng, beta
            )
            for p in PHASES:
                assert oldcritics[p] == identity_state(getattr(actor, p).critic)
            critic = critic_update(
                critics, critic_optimizer, rows, cfg, critic_history, iteration
            )
            if cfg.critic_health:
                assessment = critic["health"]
                enough = any(
                    m["worlds"] >= 64 and m["positive_worlds"] >= 8
                    for k, m in critic["held_calibration"].items()
                    if k.endswith("act0")
                )
                if enough:
                    critic_bad_windows = (
                        critic_bad_windows + 1 if assessment["urgent"] else 0
                    )
                for warning in assessment["warnings"]:
                    log.emit({"event": "alert", "iteration": iteration, **warning})
                health_state = {
                    "iteration": iteration,
                    "updated": time.time(),
                    "bad_windows": critic_bad_windows,
                    "health": assessment,
                    "held_calibration": critic["held_calibration"],
                    "fit_calibration": critic["fit_calibration"],
                }
                temporary = out / "health.tmp"
                temporary.write_text(json.dumps(health_state))
                temporary.replace(out / "health.json")
                log.emit(
                    {"event": "critic_health", "iteration": iteration, **health_state}
                )
            actor_steps += update["retained_steps"]
            rejections = 0 if update["accepted"] else rejections + 1
            if any(m["kl_mean"] > 0.015 for m in update["drift"].values()):
                beta = min(1.0, beta * 2)
            log.emit(
                {
                    "event": "update",
                    "iteration": iteration,
                    "accepted": update["accepted"],
                    "phases": update.get("phases", {}),
                    "attempted_phases": update.get("attempted_phases"),
                    "attempted_drift": update.get("attempted_drift"),
                    "failure_diagnostic": update.get("failure_diagnostic"),
                    "drift": update["drift"],
                    "reason": update.get("reason"),
                    "actor_steps": actor_steps,
                    "attempted_steps": update["attempted_steps"],
                    "retained_steps": update["retained_steps"],
                    "critic_steps": max(
                        (
                            int(state.get("step", 0))
                            for state in critic_optimizer.state.values()
                        ),
                        default=0,
                    ),
                    "critic": critic,
                    "likelihood_max_error": error,
                    "anchor_beta_next": beta,
                    "seconds": time.monotonic() - start,
                    "disk_free_gib": shutil.disk_usage(out).free / 2**30,
                    "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 2**30,
                    "weights_changed": {
                        p: before[p] != identity_state(getattr(actor, p))
                        for p in PHASES
                    },
                }
            )
            save(out / "latest.pt")
            status(
                "checkpoint_saved",
                checkpoint_sha256=file_identity(out / "latest.pt")["sha256"],
            )
            del rows, result
            gc.collect()
            if cfg.critic_health and critic_bad_windows >= 3:
                reason = "critic_learning_health_guard"
                break
            if rejections >= 2:
                reason = "persistent_update_rejection"
                break
            if iteration % cfg.eval_every == 0 or clock.remaining() < 600:
                pool.close()
                pool = None
                evalpool = SparseCollectors(
                    cfg, cfg.eval_collectors, cfg.eval_collectors
                )
                try:
                    status("evaluation")
                    tasks = train_tasks(cfg.dev_start, cfg.eval_n)
                    e = evalpool.collect(
                        actor, critics, tasks, "eval", cfg.seed, progress
                    )
                    if e["faults"] or any(
                        x["classification"] not in ("win", "loss")
                        for x in e["episodes"]
                    ):
                        raise ValueError("Invalid evaluation; no promotion")
                    from .evaluate import paired, paired_wins

                    records = evaluation_records(e)
                    wins = paired_wins(baseline["records"], records)
                    floors = paired(baseline["records"], records)
                    quality = summary_episodes(e["episodes"], e["faults"], len(tasks))
                    for row in e["episodes"]:
                        row.update(iteration=iteration, dataset="eval", attempt=attempt)
                    write_collection(
                        out / f"evaluation_{iteration:04d}_{attempt}.json.gz",
                        {
                            "episodes": e["episodes"],
                            "summary": quality,
                            "records": records,
                            "paired_wins": wins,
                            "paired_floor": floors,
                        },
                    )
                    save(out / f"evaluated_{iteration:04d}_{attempt}.pt")
                    log.emit(
                        {
                            "event": "evaluation",
                            "iteration": iteration,
                            "dataset": "eval",
                            "summary": quality,
                            "paired_wins": wins,
                            "paired_floor": floors,
                            "snapshot": file_identity(
                                out / f"evaluated_{iteration:04d}_{attempt}.pt"
                            ),
                        }
                    )
                    if (
                        quality["wins"] < baseline["summary"]["wins"] - 16
                        and wins["p_two_sided"] < 0.05
                        or quality["act1_clears"]
                        < baseline["summary"]["act1_clears"] - 32
                    ):
                        reason = "deployment_retention_guard"
                        break
                    if quality["wins"] > baseline["summary"]["wins"]:
                        last_best_used = clock.used
                    incumbent = json.loads((out / "champion_eval.json").read_text())
                    screen = paired_wins(incumbent["records"], records)
                    if (
                        quality["wins"] >= incumbent["summary"]["wins"] + 8
                        and screen["p_two_sided"] <= 0.05
                    ):
                        confirm_tasks = train_tasks(cfg.confirm_start, cfg.eval_n)
                        c = evalpool.collect(
                            actor, critics, confirm_tasks, "eval", cfg.seed, progress
                        )
                        # Confirmation reference is a separate frozen original evaluation, never a training input.
                        champion, _ = load_policy(out / "champion.pt")
                        r = evalpool.collect(
                            champion, critics, confirm_tasks, "eval", cfg.seed, progress
                        )
                        del champion
                        if (
                            c["faults"]
                            or r["faults"]
                            or any(
                                x["classification"] not in ("win", "loss")
                                for x in c["episodes"] + r["episodes"]
                            )
                        ):
                            raise ValueError("Invalid confirmation; no promotion")
                        cw = paired_wins(evaluation_records(r), evaluation_records(c))
                        write_collection(
                            out / f"confirmation_{iteration:04d}_{attempt}.json.gz",
                            {
                                "candidate": c["episodes"],
                                "reference": r["episodes"],
                                "paired_wins": cw,
                            },
                        )
                        if cw["wins"][1] > cw["wins"][0] and cw["p_two_sided"] <= 0.05:
                            save(out / "champion.pt")
                            (out / "champion_eval.json").write_text(
                                json.dumps(
                                    {
                                        "summary": quality,
                                        "records": records,
                                        "iteration": iteration,
                                    }
                                )
                            )
                            log.emit(
                                {
                                    "event": "promotion",
                                    "iteration": iteration,
                                    "confirmation": cw,
                                }
                            )
                finally:
                    evalpool.close()
                if reason == "deployment_retention_guard":
                    break
                pool = SparseCollectors(cfg)
            if clock.used - last_best_used > 2700:
                log.emit(
                    {
                        "event": "alert",
                        "iteration": iteration,
                        "reason": "stagnation_no_greedy_win_gain_45min",
                    }
                )
        if stop_requested():
            reason = "operator_stop"
        save(out / "latest.pt")
        status("stopped", reason=reason)
        log.emit(
            {
                "event": "done",
                "reason": reason,
                "iteration": iteration,
                "actor_steps": actor_steps,
                "used_seconds": clock.used,
            }
        )
    except Exception as error:
        status("failed", reason=f"{type(error).__name__}: {error}")
        log.emit({"event": "fatal", "reason": str(error), "iteration": iteration})
        raise
    finally:
        if pool:
            pool.close()
        clock.close()
        lock.close()
