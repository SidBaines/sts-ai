"""Out-of-combat PPO with a frozen combat head (the E8 recipe, generalised).

Every knob that the 2026-10-13 investigation identified as load-bearing is a
config field so it can be ablated: combat sampling, credit assignment, batch
size, entropy, temperature, dead-action filtering. Logs learning-health
signals (half-batch gradient cosine, displacement vs the Adam random-walk line)
alongside outcomes, so a noise-dominated run is visible within an iteration.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gc
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
from sts_policy.encoding import encode_observation
from sts_policy.model_agent import cached_public_collator, collate_public
from sts_policy.rollout_training import collect_rollouts, update_ppo

from .common import (save_atomic, EVAL_WORLD_START, ORIGINAL_CHECKPOINT, TRAIN_WORLD_START, JsonlLog, close_workers,
                     load_policy, make_workers, phase_forward, save_policy, strategic_rows, train_tasks)
from .evaluate import evaluate, paired
from .recording import begin_run, resume_payload, restore_rng, write_collection, require_fresh_or_resume


@dataclass(frozen=True)
class OOCTrainConfig:
    out: str
    checkpoint: str = ORIGINAL_CHECKPOINT
    iterations: int = 24
    games: int = 768
    epochs: int = 3
    minibatch: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.0
    clip: float = 0.2
    entropy: float = 0.0
    strategic_entropy: float = 0.0
    value_weight: float = 0.5
    critic_gradient: str = "shared"  # shared or detached: critic head alone receives value gradients
    normalize_returns: bool = False
    gae_lambda: float = 1.0
    target_kl: float = 0.05
    credit: str = "macro"          # macro: OOC-only bootstrap; micro: GAE through combat critic
    combat: str = "greedy"         # greedy or sample during collection
    ooc_temperature: float = 1.0
    strategic_temperature: float = 1.0  # extra OOC temperature on rest/map/shop menus (exploration)
    strategic_cards: bool = False      # also apply it to card-reward menus
    card_temperature: float = 1.0      # independently soften card-reward choices only
    dead_action_filter: bool = True
    dominated_filter: bool = False  # drop strictly dominated OOC actions (rest at full HP)
    smith_above: float | None = None  # rule overlay: no 'rest' at HP fraction >= this (train + eval)
    prefer_remove: bool = False     # rule overlay: no potion buys while a removal is affordable
    elite_above: float | None = None  # rule overlay: take an offered elite path at HP fraction >= this
    train_combat: bool = False     # keep False: combat head frozen
    eval_every: int = 4
    eval_n: int = 1024
    eval_world_start: int = EVAL_WORLD_START
    baseline: str | None = None    # eval records JSON for paired comparison
    train_world_start: int = TRAIN_WORLD_START
    workers: int = 14
    in_process_games: bool = False
    seed: int = 0
    health_every: int = 1
    collect_chunk: int = 128
    micro_batch: int = 1024         # PPO forward/backward chunk (same update, bounded activation memory)
    collectors: int = 1             # >1: data-parallel collector processes (workers split between them)
    update_threads: int = 0         # torch threads for the PPO update (0: same as threads)
    threads: int = 2
    notes: str = ""


@dataclass
class ReturnNormalizer:
    """Running population standard deviation of raw OOC choice returns.

    No mean subtraction: zero still means zero reward. Statistics are updated
    once per collected buffer, never per PPO epoch or minibatch.
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def scale(self) -> float:
        return max(math.sqrt(self.m2 / self.count), 1e-8) if self.count > 1 else 1.0

    def update(self, returns: Sequence[float]) -> None:
        if len(returns) == 0:
            return
        values = np.asarray(returns, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("return statistics require finite targets")
        count, mean = len(values), float(values.mean())
        m2 = float(((values - mean) ** 2).sum())
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def trainer_state(self) -> dict[str, Any]:
        return {"return_normalizer": {"version": 1, **asdict(self)}}

    @classmethod
    def restore(cls, payload: dict[str, Any], *, enabled: bool) -> ReturnNormalizer:
        state = payload.get("trainer_state", {}).get("return_normalizer")
        if state is None:
            return cls()  # Legacy unnormalised checkpoint.
        if not enabled:
            raise ValueError("checkpoint has normalised critic units; enable normalize_returns to resume")
        if state.get("version") != 1:
            raise ValueError("unsupported return normalizer version")
        result = cls(count=state["count"], mean=state["mean"], m2=state["m2"])
        if (result.count < 0 or not math.isfinite(result.mean) or
                not math.isfinite(result.m2) or result.m2 < 0):
            raise ValueError("invalid return normalizer statistics")
        return result


def ooc_forward(forward, *, critic_gradient: str = "shared", value_scale: float = 1.0):
    """Training-only critic units/gradient adapter; actor arithmetic is unchanged."""
    if critic_gradient not in {"shared", "detached"}:
        raise ValueError("critic_gradient must be shared or detached")
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be finite and positive")

    def wrapped(model, batch):
        handle = None
        if critic_gradient == "detached":
            # Detach the entire input, including mean action context, so neither
            # encoder used by the actor receives value-loss gradients. The hook
            # is scoped to this call and changes no model/checkpoint structure.
            handle = model.ooc.critic.register_forward_pre_hook(
                lambda module, inputs: (inputs[0].detach(),))
        try:
            logits, values, mask = forward(model, batch)
        finally:
            if handle is not None:
                handle.remove()
        if value_scale != 1.0:
            is_combat = batch[0][:, 7] > .5
            values = torch.where(is_combat, values, values * value_scale)
        return logits, values, mask

    return wrapped


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
                   "rest", "smith", "recall", "lift", "toke", "dig", "skip campfire",
                   "take key", "take gold", "take relic", "take potion",
                   "skip rewards", "open chest", "skip chest", "take boss relic", "skip boss relic"):
        if description.startswith(prefix):
            return prefix
    return "other"


def is_strategic_menu(observation: dict[str, Any], encoded: Any = None) -> bool:
    """Existing strategic action flags plus explicit rest/removal menu labels."""
    if observation.get("phase") == "combat":
        return False
    labels = {_label(a.get("description", "")) for a in observation["legal_actions"]}
    if labels & {"rest", "smith", "recall", "lift", "toke", "dig", "skip campfire",
                 "take card", "skip card", "leave shop", "buy card remove"}:
        return True
    # Removal selection actions are just 'select card index ...'; the public
    # purpose line distinguishes removal from upgrade/transform selections.
    if re.search(r"^Choose a card to REMOVE\b", observation.get("text", ""), re.MULTILINE):
        return True
    encoded = encoded if encoded is not None else encode_observation(observation)
    actions = torch.as_tensor(encoded.action_features).unsqueeze(0)
    mask = torch.ones(actions.shape[:2], dtype=torch.bool)
    return bool(strategic_rows(actions, mask, cards=True)[0])


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


def collect_ooc_batch(model: torch.nn.Module, workers: Sequence[Any], tasks: Sequence[Any], *, forward,
                      ppo: PPOConfig, credit: str, gae_lambda: float, chunk: int,
                      value_scale: float = 1.0) -> dict[str, Any]:
    """Sample ``tasks`` and keep only OOC choice transitions with PPO targets.

    Chunked: keep only OOC choice transitions (with their encodings) and small
    per-episode stats; drop combat transitions and full observation dicts per
    chunk. Holding all ~270k transitions of a 768-game batch plus an evaluation
    pushed the process past 20 GB.
    """
    kept: list[list[Any]] = []
    game_episodes: list[dict[str, Any]] = []
    faults = 0
    fault_records = []
    stats_acc: dict[str, list[tuple[float, int]]] = {}
    forward = ooc_forward(forward, value_scale=value_scale)
    for chunk_start in range(0, len(tasks), chunk):
        result = collect_rollouts(model, workers, tasks[chunk_start:chunk_start + chunk],
                                  collate=collate_public, forward=forward, config=ppo, max_seconds=3600,
                                  deterministic=False, deferred_bootstrap=True)
        episodes = split_episodes(result["transitions"])
        new = macro_ooc_transitions(episodes, gae_lambda) if credit == "macro" else micro_ooc_transitions(episodes)
        # Kept transitions already carry their encodings; PPO only needs the
        # menu and phase from the observation. Slim the dict in place (its
        # identity is checked against the cached encoding).
        for ep in new:
            for t in ep:
                obs = t.observation
                t.strategic_menu = is_strategic_menu(obs, t.encoded_observation)
                if "text" in obs:
                    obs.pop("text")
                    obs["state"] = {}
        kept += new
        for key, value in behaviour_stats(episodes).items():
            stats_acc.setdefault(key, []).append((value, len(episodes)))
        game_episodes += result['episodes']
        fault_records += result['faults']
        faults += len(result["faults"])
        del result, episodes
        gc.collect()
    return {"kept": kept, "game_episodes": game_episodes, "faults": faults, 'fault_records': fault_records, "stats": stats_acc}


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
    if cfg.critic_gradient not in {"shared", "detached"}:
        raise ValueError("critic_gradient must be shared or detached")
    out = Path(cfg.out)
    require_fresh_or_resume(out, resume)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "log.jsonl")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model, template = load_policy(cfg.checkpoint)
    normalizer = ReturnNormalizer()
    start_state = {k: v.clone() for k, v in model.state_dict().items()}
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("ooc.") or (cfg.train_combat and name.startswith("combat.")))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_iteration, steps_total = 1, 0
    latest = out / "latest.pt"
    run_record = begin_run(out, {**asdict(cfg), 'resume': resume},
                           latest if resume and latest.exists() else cfg.checkpoint,
                           inputs=[cfg.baseline] if cfg.baseline else [])
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        normalizer = ReturnNormalizer.restore(state, enabled=cfg.normalize_returns)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        # load_state_dict restores the saved lr; the config is authoritative (branches change it).
        saved_lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr
            group["weight_decay"] = cfg.weight_decay
        start_iteration, steps_total = state["iteration"] + 1, state["steps_total"]
        rng_restored = restore_rng(state, rng)
        log.emit({"event": "resume", "iteration": state["iteration"], "saved_lr": saved_lr, "lr": cfg.lr,
                  "config": asdict(cfg), 'run_record': run_record, 'rng_restored': rng_restored})
        if not (out / "config.json").exists():  # a branch seeded from another run's latest.pt
            (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    else:
        normalizer = ReturnNormalizer.restore(template, enabled=cfg.normalize_returns)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        log.emit({"event": "start", "config": asdict(cfg), 'run_record': run_record})
    forward = phase_forward(combat=cfg.combat, temperature=cfg.ooc_temperature,
                            strategic_temperature=cfg.strategic_temperature, strategic_cards=cfg.strategic_cards,
                            card_temperature=cfg.card_temperature)
    update_forward = ooc_forward(forward, critic_gradient=cfg.critic_gradient)
    ppo = PPOConfig(gamma=1.0, gae_lambda=cfg.gae_lambda, clip_ratio=cfg.clip, entropy_weight=cfg.entropy,
                    value_weight=cfg.value_weight,
                    epochs=cfg.epochs, minibatch_size=cfg.minibatch, learning_rate=cfg.lr,
                    kl_guard_before_update=True, target_kl=cfg.target_kl)
    baseline_records = json.loads(Path(cfg.baseline).read_text())["records"] if cfg.baseline else None
    combat_keys = [k for k in start_state if k.startswith("combat.")]
    blocks = ooc_parameter_blocks(model)
    pool = None
    if cfg.collectors > 1:
        from .parallel_collect import CollectorPool
        pool = CollectorPool(cfg.collectors, workers=cfg.workers, checkpoint=cfg.checkpoint,
                             in_process=cfg.in_process_games,
                             dead_action_filter=cfg.dead_action_filter, dominated=cfg.dominated_filter,
                             smith_above=cfg.smith_above, prefer_remove=cfg.prefer_remove,
                             elite_above=cfg.elite_above)
        workers: list[Any] = []
    else:
        workers = make_workers(cfg.workers, dead_action_filter=cfg.dead_action_filter,
                               in_process=cfg.in_process_games,
                               dominated=cfg.dominated_filter, smith_above=cfg.smith_above,
                               prefer_remove=cfg.prefer_remove, elite_above=cfg.elite_above)

    def run_eval(iteration: int) -> None:
        restarts_before = pool.restarts if pool is not None else 0
        result = evaluate(model, pool if pool is not None else workers, n=cfg.eval_n,
                          world_start=cfg.eval_world_start)
        (out / f"eval_{iteration:04d}.json").write_text(json.dumps(result))
        record = {"event": "eval", "iteration": iteration, **result["summary"],
                  "collector_restarts": pool.restarts - restarts_before if pool is not None else 0}
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
            tasks = train_tasks(world_start, cfg.games)
            if pool is not None:
                batch = pool.collect_ooc(model, tasks, combat=cfg.combat, temperature=cfg.ooc_temperature,
                                         strategic_temperature=cfg.strategic_temperature,
                                         strategic_cards=cfg.strategic_cards,
                                         card_temperature=cfg.card_temperature,
                                         ppo=ppo, credit=cfg.credit, gae_lambda=cfg.gae_lambda,
                                         chunk=cfg.collect_chunk, seed=cfg.seed * 100_000 + iteration,
                                         value_scale=normalizer.scale if cfg.normalize_returns else 1.0)
            else:
                batch = collect_ooc_batch(model, workers, tasks, forward=forward, ppo=ppo, credit=cfg.credit,
                                          gae_lambda=cfg.gae_lambda, chunk=cfg.collect_chunk,
                                          value_scale=normalizer.scale if cfg.normalize_returns else 1.0)
            kept, game_episodes, faults, stats_acc = batch["kept"], batch["game_episodes"], batch["faults"], batch["stats"]
            collector_restarts = batch["collector_restarts"] if pool is not None else 0
            collection = out / f'collection_{iteration:04d}.json.gz'
            write_collection(collection, {'schema':'sts_training_collection_v1', 'run_record':run_record,
                'iteration':iteration, 'tasks':[task.to_dict() for task in tasks], 'episodes':game_episodes,
                'faults':batch['fault_records']})
            del batch
            collect_seconds = time.time() - started
            flat = [t for ep in kept for t in ep]
            # Kept observations were slimmed above, so a missing sampling-time encoding
            # would be silently re-encoded from an empty state. Fail loudly instead.
            if any(t.encoded_observation is None for t in flat):
                raise RuntimeError("kept OOC transition lacks its sampling-time encoding")
            if cfg.normalize_returns:
                normalizer.update([t.returns for t in flat])
            return_scale = normalizer.scale if cfg.normalize_returns else 1.0
            entropy_weights = ([cfg.entropy + cfg.strategic_entropy * bool(t.strategic_menu) for t in flat]
                               if cfg.strategic_entropy else None)
            floors = np.array([e["floor"] for e in game_episodes])
            behaviour = {k: sum(v * n for v, n in vals) / max(1, sum(n for _, n in vals)) for k, vals in stats_acc.items()}
            record: dict[str, Any] = {
                "event": "train", "iteration": iteration, "collect_floor": float(floors.mean()),
                "collect_floor_se": float(floors.std() / math.sqrt(max(1, len(floors)))),
                "collect_act1_clear": float((floors > 16).mean()), "collect_act2_clear": float((floors > 33).mean()),
                "collect_wins": int(sum(e["task_outcome"] in ("act_clear", "victory") for e in game_episodes)),
                "games": len(game_episodes), "faults": faults,
                "collector_restarts": collector_restarts,
                "ooc_choice_transitions": len(flat), "collect_seconds": collect_seconds,
                "adv_raw_std": float(np.std([t.advantage for t in flat])) if flat else 0.0,
                "return_scale": return_scale,
                **behaviour}
            update_started = time.time()
            if cfg.update_threads:
                torch.set_num_threads(cfg.update_threads)
            if cfg.health_every and iteration % cfg.health_every == 0:
                record["grad_cosine_half"] = gradient_cosine(model, update_forward, kept, rng=rng, chunk=cfg.micro_batch)
            update = update_ppo(model, flat, collate=collate_public, forward=update_forward, config=ppo, device="cpu",
                                seed=cfg.seed * 1000 + iteration, optimizer=optimizer,
                                micro_batch=cfg.micro_batch, return_scale=return_scale, entropy_weights=entropy_weights)
            torch.set_num_threads(cfg.threads)
            record["update_seconds"] = time.time() - update_started
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
            del kept, flat
            gc.collect()
            trainer_state = normalizer.trainer_state() if cfg.normalize_returns else {}
            payload = resume_payload(model, optimizer, iteration, steps_total, asdict(cfg), run_record, rng)
            payload["trainer_state"] = trainer_state
            save_atomic(payload, latest)
            if cfg.eval_every and (iteration % cfg.eval_every == 0 or iteration == cfg.iterations):
                save_policy(out / f"iter_{iteration:04d}.pt", model, template,
                            metadata={"stage": "sts_rl_ooc_ppo", "iteration": iteration, "config": asdict(cfg),
                                      'run_record': run_record}, optimizer=optimizer, step=steps_total, rng=rng,
                            trainer_state=trainer_state)
                run_eval(iteration)
    finally:
        close_workers(workers)
        if pool is not None:
            pool.close()
    log.emit({"event": "done"})
