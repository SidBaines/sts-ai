"""Collect and label a bounded sample of meaningful combat menus from complete champion games."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count, islice
import json
import multiprocessing
from pathlib import Path
import random

import numpy as np
import torch
from sts_policy.curriculum import world_split
from sts_policy.encoding import collate_observations
from sts_policy.model_agent import collate_public, is_act3_boss_combat
from sts_policy.rollout_training import collect_rollouts
from sts_rl.common import FilteringWorker, WORKER_CONFIG, close_workers, load_policy
from sts_rl.fresh_collection import task
from sts_rl.fresh_study import atomic_json
from sts_rl.in_process import InProcessWorker
from study import CFG, CHAMP, OUT, ROLES, check_parent, note


class SampleGame:
    def __init__(self):
        self.rows = [[], []]
        self.seen = [0, 0]
        self.rng = None

    def record(self, t):
        obs = t.observation
        if obs["phase"] != "combat" or len(obs["legal_actions"]) < 2:
            return
        if self.rng is None:
            self.rng = random.Random(t.world_seed)
        role = int(is_act3_boss_combat(obs))
        rows = self.rows[role]
        limit = CFG["boss_samples_per_world" if role else "ordinary_samples_per_world"]
        self.seen[role] += 1
        slot = len(rows) if len(rows) < limit else self.rng.randrange(self.seen[role])
        if slot < limit:
            row = (t.encoded_observation, t.action_position, t.world_seed)
            if slot == len(rows):
                rows.append(row)
            else:
                rows[slot] = row

    def finish(self, ep):
        return self.rows if ep["terminated"] and not ep["truncated"] else ([], [])


def init_worker():
    global teacher
    torch.set_num_threads(1)
    teacher, _ = load_policy(CHAMP)
    teacher.eval()


def actor(model, batch):
    return model.actor_logits(*batch), None, batch[2]


def pack(rows, head):
    encodings = [r[0] for r in rows]
    lengths = np.asarray([len(e.action_indices) for e in encodings], dtype=np.int32)
    values = []
    with torch.inference_mode():
        for i in range(0, len(rows), 128):
            block = encodings[i:i+128]
            batch = collate_observations(block)
            scores = head.actor_logits(*batch)
            values.append(scores.masked_select(batch[2]).numpy())
    scores = np.concatenate(values)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite teacher scores")
    return dict(states=np.stack([e.state_features for e in encodings]),
                actions=np.concatenate([e.action_features for e in encodings]),
                lengths=lengths, scores=scores, seeds=np.asarray([r[2] for r in rows]),
                chosen=np.asarray([r[1] for r in rows], dtype=np.int32))


def shard(split, index, seeds, out):
    workers = [FilteringWorker(InProcessWorker(config={**WORKER_CONFIG, "fast_observations": False}),
                               hide_ooc_potions=True, dead_actions=False, slim=False) for _ in range(24)]
    try:
        result = collect_rollouts(teacher, workers, [task(seed) for seed in seeds],
                                  collate=collate_public, forward=actor, deterministic=True,
                                  deferred_bootstrap=True, episode_sink_factory=SampleGame,
                                  actor_only=True, max_seconds=1800)
    finally:
        close_workers(workers)
    if not result["cohort_completed"]:
        raise RuntimeError("Collection did not exhaust its assigned worlds")
    episodes, faults = result["episodes"], result["faults"]
    statuses = {e["world_seed"]: "completed" if e["terminated"] and not e["truncated"] else "censored"
                for e in episodes}
    statuses.update({f["world_seed"]: "fault" for f in faults})
    if set(statuses) != set(seeds) or len(episodes) + len(faults) != len(seeds):
        raise ValueError("Shard does not cover its assigned worlds")
    arrays, sizes = {}, {}
    for i, role in enumerate(ROLES):
        rows = [sample for game in result["episode_data"] for sample in game[i]]
        items = pack(rows, teacher.combat if i == 0 else teacher.boss_specialist)
        arrays.update({f"{role}_{key}": value for key, value in items.items()})
        sizes[role] = len(rows)
    dest = out / f"{split}-{index:03d}.npz"
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(dest)
    item = dict(worlds=statuses, counts=sizes, faults=faults)
    atomic_json(dest.with_suffix(".json"), item)
    return split, len(seeds)


def main(out, jobs):
    check_parent()
    out.mkdir(parents=True, exist_ok=True)
    units = []
    for split, tag in (("train", "train"), ("validation", "dev")):
        start, n = CFG[split]
        seeds = list(islice((i for i in count(start) if world_split(i) == tag), n))
        for index, i in enumerate(range(0, n, 128)):
            units.append((split, index, seeds[i:i+128], out))
    roster = {name: [seed for split, _, part, _ in units if split == name for seed in part]
              for name in ("train", "validation")}
    if set(roster["train"]) & set(roster["validation"]):
        raise ValueError("Training and validation worlds overlap")
    atomic_json(out / "roster.json", roster)
    total = sum(len(u[2]) for u in units)
    done = 0
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker) as pool:
        futures = []
        for split, index, seeds, folder in units:
            dest = folder / f"{split}-{index:03d}.npz"
            if dest.is_file() and dest.with_suffix(".json").is_file():
                prior = json.loads(dest.with_suffix(".json").read_bytes())
                if set(map(int, prior["worlds"])) != set(seeds):
                    raise ValueError("Existing shard has a different roster")
                done += len(seeds)
            else:
                futures.append(pool.submit(shard, split, index, seeds, folder))
        for future in as_completed(futures):
            split, n = future.result()
            done += n
            note("demonstrations", f"Collected champion combat choices on {done:,} of {total:,} assigned worlds.",
                 log=done == total)
    summary = {}
    for split in ("train", "validation"):
        records = [json.loads(p.read_bytes()) for p in sorted(out.glob(f"{split}-*.json"))]
        summary[split] = {role: sum(r["counts"][role] for r in records) for role in ROLES}
        for status in ("completed", "fault", "censored"):
            summary[split][status] = sum(v == status for r in records for v in r["worlds"].values())
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT / "demonstrations")
    p.add_argument("--processes", type=int, default=6)
    a = p.parse_args()
    main(a.out, a.processes)
