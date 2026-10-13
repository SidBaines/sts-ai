"""Focused combat RL: PPO on single fights replayed from a learner-reached fight bank.

Each episode is one fight (boss/elite), reward +1 win / -1 loss (+ optional HP
retention on wins). Episodes are short, so Monte-Carlo advantages (lambda=1) are
low-variance and the combat critic is only a baseline. The OOC head is frozen and
asserted bit-identical. Held-out *worlds* give a fast fight-panel metric; periodic
paired full-game evaluations check that fight gains survive in whole runs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from sts_policy.curriculum import TaskSpec
from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public, forward_policy
from sts_policy.rollout_training import collect_rollouts, update_ppo
from sts_policy.workers import CurriculumWorker

from .common import save_atomic, JsonlLog, close_workers, load_policy, make_workers, save_policy
from .evaluate import evaluate, paired


@dataclass(frozen=True)
class CombatTrainConfig:
    bank: str
    out: str
    checkpoint: str
    iterations: int = 40
    fights: int = 1024
    epochs: int = 3
    minibatch: int = 4096
    lr: float = 1e-4
    clip: float = 0.2
    entropy: float = 0.0
    value_weight: float = 0.25
    gae_lambda: float = 1.0
    target_kl: float = 0.05
    micro_batch: int = 1024
    hp_weight: float = 0.3          # terminal HP retention bonus on wins
    mix: str = "BOSS2:0.35,ELITE2:0.35,BOSS1:0.1,ELITE1:0.05,BOSS3:0.075,ELITE3:0.075"
    holdout_fraction: float = 0.2
    panel_every: int = 2
    panel_per_group: int = 120
    game_eval_every: int = 10
    game_eval_n: int = 1024
    game_baseline: str | None = None
    workers: int = 8
    seed: int = 0
    threads: int = 2
    notes: str = ""


def _group(row: dict[str, Any]) -> str:
    return f"{row['room']}{row['act']}"


def _battle_workers(count: int, hp_weight: float) -> list[Any]:
    from .common import WORKER_CONFIG, FilteringWorker
    config = {**WORKER_CONFIG, "terminal_hp_weight": hp_weight}
    return [FilteringWorker(CurriculumWorker(config=config, timeout=60.0), dead_actions=False, slim=True)
            for _ in range(count)]


def fight_panel(model: torch.nn.Module, workers, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedy win rate per group on fixed held-out fight starts."""
    tasks = [TaskSpec.from_dict(r["task"]) for r in rows]
    result = collect_rollouts(model, workers, tasks, collate=collate_public, forward=forward_policy,
                              config=PPOConfig(), max_seconds=3600, deterministic=True,
                              deferred_bootstrap=True, reuse_encodings=False)
    from sts_policy.rollout_training import task_identity
    outcome = {e["task_id"]: e.get("task_outcome") == "battle_victory" for e in result["episodes"]}
    by_group = defaultdict(list)
    for row, task in zip(rows, tasks):
        won = outcome.get(task_identity(task))
        if won is not None:
            by_group[_group(row)].append(won)
    summary = {g: float(np.mean(v)) for g, v in by_group.items()}
    summary["n"] = sum(len(v) for v in by_group.values())
    summary["faults"] = len(result["faults"])
    return summary


def train(cfg: CombatTrainConfig, *, resume: bool = False) -> None:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "log.jsonl")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    rows = [json.loads(line) for line in Path(cfg.bank).read_text().splitlines() if line.strip()]
    seeds = sorted({r["task"]["seed"] for r in rows})
    random.Random(1234).shuffle(seeds)
    held = set(seeds[: int(len(seeds) * cfg.holdout_fraction)])
    train_rows = [r for r in rows if r["task"]["seed"] not in held]
    held_rows = [r for r in rows if r["task"]["seed"] in held]
    groups = defaultdict(list)
    for r in train_rows:
        groups[_group(r)].append(r)
    mix = {k: float(v) for k, v in (part.split(":") for part in cfg.mix.split(","))}
    panel_rows = []
    for g in mix:
        candidates = [r for r in held_rows if _group(r) == g]
        random.Random(7).shuffle(candidates)
        panel_rows += candidates[: cfg.panel_per_group]
    model, template = load_policy(cfg.checkpoint)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("combat."))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    first, steps_total = 1, 0
    latest = out / "latest.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        # load_state_dict restores the saved lr; the config is authoritative (branches change it).
        saved_lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr
        first, steps_total = state["iteration"] + 1, state["steps_total"]
        log.emit({"event": "resume", "iteration": state["iteration"], "saved_lr": saved_lr, "lr": cfg.lr})
    else:
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        log.emit({"event": "start", "config": asdict(cfg), "bank_train": len(train_rows), "bank_held": len(held_rows),
                  "panel": len(panel_rows), "groups": {g: len(v) for g, v in groups.items()}})
    ppo = PPOConfig(gamma=1.0, gae_lambda=cfg.gae_lambda, clip_ratio=cfg.clip, entropy_weight=cfg.entropy,
                    value_weight=cfg.value_weight, epochs=cfg.epochs, minibatch_size=cfg.minibatch,
                    learning_rate=cfg.lr, kl_guard_before_update=True, target_kl=cfg.target_kl)
    workers = _battle_workers(cfg.workers, cfg.hp_weight)
    game_workers = None
    baseline = json.loads(Path(cfg.game_baseline).read_text())["records"] if cfg.game_baseline else None
    ooc_keys = [k for k in start if k.startswith("ooc.")]
    try:
        if first == 1:
            log.emit({"event": "panel", "iteration": 0, **fight_panel(model.eval(), workers, panel_rows)})
        for iteration in range(first, cfg.iterations + 1):
            started = time.time()
            names, weights = zip(*[(g, w) for g, w in mix.items() if groups.get(g)])
            picks = rng.choices(names, weights=weights, k=cfg.fights)
            tasks = [TaskSpec.from_dict(rng.choice(groups[g])["task"]) for g in picks]
            model.eval()
            result = collect_rollouts(model, workers, tasks, collate=collate_public, forward=forward_policy,
                                      config=ppo, max_seconds=3600, deterministic=False, deferred_bootstrap=True)
            transitions = [t for t in result["transitions"] if len(t.observation["legal_actions"]) > 1]
            wins = [e.get("task_outcome") == "battle_victory" for e in result["episodes"]]
            update = update_ppo(model, transitions, collate=collate_public, forward=forward_policy, config=ppo,
                                device="cpu", seed=cfg.seed * 1000 + iteration, optimizer=optimizer,
                                micro_batch=cfg.micro_batch)
            optimizer = update.pop("optimizer")
            steps_total += update["steps"]
            state = model.state_dict()
            if not all(torch.equal(start[k], state[k]) for k in ooc_keys):
                raise RuntimeError("frozen OOC head changed during combat training")
            log.emit({"event": "train", "iteration": iteration, "collect_win": float(np.mean(wins)) if wins else None,
                      "fights": len(result["episodes"]), "faults": len(result["faults"]),
                      "transitions": len(transitions), "steps_total": steps_total,
                      **{k: update.get(k) for k in ("steps", "approx_kl", "clip_fraction", "entropy", "value_loss",
                                                     "policy_loss", "early_stop")},
                      "seconds": time.time() - started})
            save_atomic({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "iteration": iteration, "steps_total": steps_total}, latest)
            if cfg.panel_every and iteration % cfg.panel_every == 0:
                log.emit({"event": "panel", "iteration": iteration, **fight_panel(model.eval(), workers, panel_rows)})
            if cfg.game_eval_every and (iteration % cfg.game_eval_every == 0 or iteration == cfg.iterations):
                path = out / f"iter_{iteration:04d}.pt"
                save_policy(path, model, template, metadata={"stage": "sts_rl_combat_ppo", "iteration": iteration,
                                                             "config": asdict(cfg)})
                if game_workers is None:
                    game_workers = make_workers(cfg.workers, dead_action_filter=True)
                result = evaluate(model, game_workers, n=cfg.game_eval_n)
                (out / f"eval_{iteration:04d}.json").write_text(json.dumps(result))
                record = {"event": "eval", "iteration": iteration, **result["summary"]}
                if baseline is not None:
                    record["paired"] = paired(baseline, result["records"])
                log.emit(record)
    finally:
        close_workers(workers)
        if game_workers is not None:
            close_workers(game_workers)
    log.emit({"event": "done"})
