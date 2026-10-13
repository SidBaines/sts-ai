"""Out-of-combat PPO with a frozen combat head (the E8 recipe, generalised).

Every knob that the 2026-10-13 investigation identified as load-bearing is a
config field so it can be ablated: combat sampling, credit assignment, batch
size, entropy, temperature, dead-action filtering. Logs learning-health
signals (half-batch gradient cosine, displacement vs the Adam random-walk line)
alongside outcomes, so a noise-dominated run is visible within an iteration.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence

import numpy as np
import torch

from sts_policy.losses import PPOConfig, generalized_advantages, masked_distribution
from sts_policy.model_agent import cached_public_collator, collate_public
from sts_policy.rollout_training import collect_rollouts, update_ppo

from .common import (EVAL_WORLD_START, ORIGINAL_CHECKPOINT, TRAIN_WORLD_START, JsonlLog, close_workers,
                     load_policy, make_workers, phase_forward, save_policy, train_tasks)
from .evaluate import evaluate, paired


@dataclass(frozen=True)
class OOCTrainConfig:
    out: str
    checkpoint: str = ORIGINAL_CHECKPOINT
    iterations: int = 24
    games: int = 768
    epochs: int = 3
    minibatch: int = 4096
    lr: float = 3e-4
    clip: float = 0.2
    entropy: float = 0.0
    gae_lambda: float = 0.95
    target_kl: float = 0.05
    credit: str = "macro"          # macro: OOC-only bootstrap; micro: GAE through combat critic
    combat: str = "greedy"         # greedy or sample during collection
    ooc_temperature: float = 1.0
    dead_action_filter: bool = True
    train_combat: bool = False     # keep False: combat head frozen
    eval_every: int = 4
    eval_n: int = 1024
    eval_world_start: int = EVAL_WORLD_START
    baseline: str | None = None    # eval records JSON for paired comparison
    train_world_start: int = TRAIN_WORLD_START
    workers: int = 14
    seed: int = 0
    health_every: int = 1
    threads: int = 2
    notes: str = ""


def split_episodes(transitions: Sequence[Any]) -> list[list[Any]]:
    out, cur = [], []
    for t in transitions:
        cur.append(t)
        if t.terminated or t.truncated:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def macro_ooc_transitions(episodes: Sequence[Sequence[Any]], gae_lambda: float) -> list[list[Any]]:
    """Per episode, OOC choice transitions with OOC-only macro GAE (rewards summed
    to the next OOC decision; bootstrap with the OOC critic's value there)."""
    out = []
    for ep in episodes:
        idx = [i for i, t in enumerate(ep) if t.observation.get("phase") != "combat"]
        if not idx:
            continue
        rewards, values, term, trunc, nxt = [], [], [], [], []
        for k, i in enumerate(idx):
            j = idx[k + 1] if k + 1 < len(idx) else None
            seg = ep[i:j] if j is not None else ep[i:]
            rewards.append(sum(t.reward for t in seg))
            values.append(ep[i].behavior_value)
            if j is not None:
                term.append(False); trunc.append(False); nxt.append(ep[j].behavior_value)
            else:
                last = ep[-1]
                term.append(bool(last.terminated)); trunc.append(not last.terminated)
                nxt.append(0.0 if last.terminated else float(last.next_value))
        adv, ret = generalized_advantages(rewards, values, term, trunc, nxt, gamma=1.0, gae_lambda=gae_lambda)
        kept = []
        for k, i in enumerate(idx):
            t = ep[i]
            t.advantage, t.returns = adv[k], ret[k]
            if len(t.observation["legal_actions"]) > 1:
                kept.append(t)
        out.append(kept)
    return out


def micro_ooc_transitions(episodes: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Keep collect_rollouts' per-microdecision GAE (crosses the combat critic)."""
    return [[t for t in ep if t.observation.get("phase") != "combat" and len(t.observation["legal_actions"]) > 1]
            for ep in episodes]


def _label(description: str) -> str:
    m = re.search(r"room=(\w+)", description)
    if m:
        return "map:" + m.group(1)
    for prefix in ("take card", "skip card", "buy card remove", "buy card", "buy potion", "buy relic", "leave shop",
                   "rest", "smith", "recall", "take key", "take gold", "take relic", "take potion",
                   "skip rewards", "open chest", "skip chest", "take boss relic", "skip boss relic"):
        if description.startswith(prefix):
            return prefix
    return "other"


def behaviour_stats(episodes: Sequence[Sequence[Any]]) -> dict[str, float]:
    counts: dict[str, Counter] = {"rest_site": Counter(), "card_reward": Counter(), "map": Counter(), "shop": Counter()}
    fights = []
    for ep in episodes:
        fights.append(sum(1 for i, t in enumerate(ep) if t.observation.get("phase") == "combat"
                          and (i == 0 or ep[i - 1].observation.get("phase") != "combat")))
        for t in ep:
            legal = t.observation["legal_actions"]
            if t.observation.get("phase") == "combat" or len(legal) < 2:
                continue
            descs = [a.get("description", "") for a in legal]
            label = _label(descs[t.action_position])
            if any(d.startswith("rest") or d.startswith("smith") for d in descs):
                counts["rest_site"][label] += 1
            elif any(d.startswith("skip card") for d in descs):
                counts["card_reward"][label] += 1
            elif any("map node" in d for d in descs):
                counts["map"][label] += 1
            elif any(d.startswith("leave shop") for d in descs):
                counts["shop"][label] += 1
    def frac(group: str, key: str) -> float:
        total = sum(counts[group].values())
        return counts[group][key] / total if total else 0.0
    return {"smith_rate": frac("rest_site", "smith"), "card_skip_rate": frac("card_reward", "skip card"),
            "map_monster": frac("map", "map:MONSTER"), "map_elite": frac("map", "map:ELITE"),
            "map_event": frac("map", "map:EVENT"), "map_rest": frac("map", "map:REST"),
            "shop_remove": frac("shop", "buy card remove"), "shop_potion": frac("shop", "buy potion"),
            "fights_per_run": float(np.mean(fights)) if fights else 0.0}


def ooc_parameter_blocks(model: torch.nn.Module) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for name, _ in model.named_parameters():
        if name.startswith("ooc."):
            blocks.setdefault(name.split(".")[1], []).append(name)
    return blocks


def gradient_cosine(model: torch.nn.Module, forward, kept: Sequence[Sequence[Any]], *,
                    rng: random.Random, chunk: int = 4096) -> float:
    """Cosine between OOC-actor policy gradients of two disjoint random episode halves."""
    episodes = [ep for ep in kept if ep]
    if len(episodes) < 4:
        return float("nan")
    flat = [t for ep in episodes for t in ep]
    raw = np.array([t.advantage for t in flat])
    norm = {id(t): float(v) for t, v in zip(flat, (raw - raw.mean()) / max(raw.std(), 1e-8))}
    collate = cached_public_collator([t.observation for t in flat], preencoded=[t.encoded_observation for t in flat])
    names = [n for n, p in model.named_parameters() if n.startswith("ooc.") and ".critic." not in n and p.requires_grad]
    params = dict(model.named_parameters())
    order = list(range(len(episodes)))
    rng.shuffle(order)
    halves = [order[: len(order) // 2], order[len(order) // 2:]]
    grads = []
    model.train()
    for half in halves:
        trans = [t for i in half for t in episodes[i]]
        total = [torch.zeros_like(params[n]) for n in names]
        for s in range(0, len(trans), chunk):
            part = trans[s:s + chunk]
            model.zero_grad(set_to_none=True)
            logits, _, mask = forward(model, collate([t.observation for t in part], "cpu"))
            dist = masked_distribution(logits, mask)
            adv = torch.tensor([norm[id(t)] for t in part])
            loss = -(adv * dist.log_prob(torch.tensor([t.action_position for t in part]))).sum() / len(trans)
            loss.backward()
            for g, n in zip(total, names):
                if params[n].grad is not None:
                    g += params[n].grad
        grads.append(torch.cat([g.reshape(-1) for g in total]))
    model.zero_grad(set_to_none=True)
    a, b = grads
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def train(cfg: OOCTrainConfig, *, resume: bool = False) -> None:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "log.jsonl")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model, template = load_policy(cfg.checkpoint)
    start_state = {k: v.clone() for k, v in model.state_dict().items()}
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("ooc.") or (cfg.train_combat and name.startswith("combat.")))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)
    start_iteration, steps_total = 1, 0
    latest = out / "latest.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_iteration, steps_total = state["iteration"] + 1, state["steps_total"]
        log.emit({"event": "resume", "iteration": state["iteration"]})
    else:
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        log.emit({"event": "start", "config": asdict(cfg)})
    forward = phase_forward(combat=cfg.combat, temperature=cfg.ooc_temperature)
    ppo = PPOConfig(gamma=1.0, gae_lambda=cfg.gae_lambda, clip_ratio=cfg.clip, entropy_weight=cfg.entropy,
                    epochs=cfg.epochs, minibatch_size=cfg.minibatch, learning_rate=cfg.lr,
                    kl_guard_before_update=True, target_kl=cfg.target_kl)
    baseline_records = json.loads(Path(cfg.baseline).read_text())["records"] if cfg.baseline else None
    combat_keys = [k for k in start_state if k.startswith("combat.")]
    blocks = ooc_parameter_blocks(model)
    workers = make_workers(cfg.workers, dead_action_filter=cfg.dead_action_filter)

    def run_eval(iteration: int) -> None:
        result = evaluate(model, workers, n=cfg.eval_n, world_start=cfg.eval_world_start)
        (out / f"eval_{iteration:04d}.json").write_text(json.dumps(result))
        record = {"event": "eval", "iteration": iteration, **result["summary"]}
        if baseline_records is not None:
            record["paired"] = paired(baseline_records, result["records"])
        log.emit(record)

    try:
        if start_iteration == 1 and baseline_records is None:
            run_eval(0)
        for iteration in range(start_iteration, cfg.iterations + 1):
            started = time.time()
            model.eval()
            world_start = cfg.train_world_start + cfg.seed * 500_000_000 + iteration * 10_000_000
            result = collect_rollouts(model, workers, train_tasks(world_start, cfg.games), collate=collate_public,
                                      forward=forward, config=ppo, max_seconds=3600, deterministic=False,
                                      deferred_bootstrap=True)
            collect_seconds = time.time() - started
            episodes = split_episodes(result["transitions"])
            kept = (macro_ooc_transitions(episodes, cfg.gae_lambda) if cfg.credit == "macro"
                    else micro_ooc_transitions(episodes))
            flat = [t for ep in kept for t in ep]
            floors = np.array([e["floor"] for e in result["episodes"]])
            record: dict[str, Any] = {
                "event": "train", "iteration": iteration, "collect_floor": float(floors.mean()),
                "collect_floor_se": float(floors.std() / math.sqrt(max(1, len(floors)))),
                "collect_act1_clear": float((floors > 16).mean()), "collect_act2_clear": float((floors > 33).mean()),
                "collect_wins": int(sum(e["task_outcome"] in ("act_clear", "victory") for e in result["episodes"])),
                "games": len(result["episodes"]), "faults": len(result["faults"]),
                "ooc_choice_transitions": len(flat), "collect_seconds": collect_seconds,
                "adv_raw_std": float(np.std([t.advantage for t in flat])) if flat else 0.0,
                **behaviour_stats(episodes)}
            if cfg.health_every and iteration % cfg.health_every == 0:
                record["grad_cosine_half"] = gradient_cosine(model, forward, kept, rng=rng)
            update = update_ppo(model, flat, collate=collate_public, forward=forward, config=ppo, device="cpu",
                                seed=cfg.seed * 1000 + iteration, optimizer=optimizer)
            optimizer = update.pop("optimizer")
            steps_total += update["steps"]
            record.update({k: update.get(k) for k in ("steps", "approx_kl", "clip_fraction", "entropy", "value_loss",
                                                       "policy_loss", "early_stop")})
            record["steps_total"] = steps_total
            state = model.state_dict()
            if not cfg.train_combat and not all(torch.equal(start_state[k], state[k]) for k in combat_keys):
                raise RuntimeError("frozen combat head changed")
            for block, names in blocks.items():
                numel = sum(state[n].numel() for n in names)
                moved = math.sqrt(sum(float((state[n] - start_state[n]).norm() ** 2) for n in names))
                rw = cfg.lr * math.sqrt(max(1, steps_total) * numel)
                record[f"disp_ratio_{block}"] = moved / rw
            record["iteration_seconds"] = time.time() - started
            log.emit(record)
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "iteration": iteration, "steps_total": steps_total}, latest)
            if cfg.eval_every and (iteration % cfg.eval_every == 0 or iteration == cfg.iterations):
                save_policy(out / f"iter_{iteration:04d}.pt", model, template,
                            metadata={"stage": "sts_rl_ooc_ppo", "iteration": iteration, "config": asdict(cfg)})
                run_eval(iteration)
    finally:
        close_workers(workers)
    log.emit({"event": "done"})
