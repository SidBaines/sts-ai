"""Bounded native combat cloning with public inputs and frozen strategic behavior."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from sts_policy import entities
from sts_policy.combat_residual import initialize, equivalent_positions, equivalence_loss, save_policy
from sts_policy.curriculum import world_split
from sts_policy.training import file_identity
try:
    from scripts.run_strategic_v3 import atomic_json
except ModuleNotFoundError:
    from run_strategic_v3 import atomic_json


def reservoir(group, item, seen, cap, rng):
    if len(group) < cap:
        group.append(item)
    else:
        position = rng.randrange(seen)
        if position < cap:
            group[position] = item


def prepare(path, seed, per_act):
    rng = random.Random(seed)
    groups = {"train": {}, "holdout": {}}
    counts = Counter()
    for line in path.open():
        record = json.loads(line)
        if world_split(int(record["world_seed"])) != "train":
            raise ValueError("combat supervision/validation must be TRAIN-only")
        obs = record["observation"]
        if obs["phase"] != "combat" or len(obs["legal_actions"]) < 2:
            continue
        act = int(obs["state"]["act"])
        # Independent worlds within TRAIN, never DEV/final gradient examples.
        split = "holdout" if int(hashlib.sha256(f"combat-holdout-v1:{record['world_seed']}".encode()).hexdigest()[:8], 16)%5 == 0 else "train"
        counts[(split, act)] += 1
        group = groups[split].setdefault(act, [])
        reservoir(group, record, counts[(split, act)], per_act if split == "train" else 512, rng)
    encoded = {"train": {}, "holdout": {}}
    worlds = {"train": set(), "holdout": set()}
    for split, phases in groups.items():
        for act, records in phases.items():
            converted = []
            for record in records:
                row = entities.encode_observation(record["observation"])
                teacher = row.action_indices.tolist().index(int(record["teacher_index"]))
                converted.append((row, equivalent_positions(row, teacher), teacher))
                worlds[split].add(int(record["world_seed"]))
            encoded[split][act] = converted
    if worlds["train"] & worlds["holdout"]:
        raise ValueError("combat validation world leakage")
    return encoded, {"source_counts": {f"{s}:act{a}":n for (s,a),n in counts.items()},
        "pool_counts": {s:{str(a):len(r) for a,r in phases.items()} for s,phases in encoded.items()},
        "world_counts": {s:len(w) for s,w in worlds.items()},
        "teacher_sha256": file_identity(path)}


def collate(rows, device):
    batch = entities.collate_observations([row[0] for row in rows], device)
    target = np.zeros(tuple(batch[2].shape), np.bool_)
    for i, row in enumerate(rows):
        target[i, :len(row[1])] = row[1]
    return batch, torch.as_tensor(target, device=device)


@torch.no_grad()
def validation(model, phases, device):
    model.to(device).eval()
    result = {}
    for act, rows in phases.items():
        loss, exact, equivalent, count = 0., 0, 0, 0
        for start in range(0, len(rows), 128):
            chosen = rows[start:start+128]
            batch, targets = collate(chosen, device)
            logits, _ = model(*batch)
            loss += float(equivalence_loss(logits, batch[2], targets).cpu())*len(chosen)
            picks = logits.argmax(-1).cpu().tolist()
            for row, pick in zip(chosen, picks):
                exact += pick == row[2]
                equivalent += bool(row[1][pick])
            count += len(chosen)
        result[str(act)] = {"rows":count,"equivalence_loss":loss/max(1,count),
                           "exact_route_accuracy":exact/max(1,count),"model_equivalent_accuracy":equivalent/max(1,count)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--per-act", type=int, default=6000)
    parser.add_argument("--anchor", type=float, default=.1)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--device", choices=("cpu","mps"), default="mps")
    args = parser.parse_args()
    if args.output.exists() or min(args.steps,args.batch_size,args.per_act,args.lr,args.checkpoint_every) <= 0:
        parser.error("use fresh output and positive training settings")
    torch.set_num_threads(1)
    started = time.monotonic()
    model, transfer = initialize(args.base, seed=args.seed)
    args.output.mkdir(parents=True)
    data, data_record = prepare(args.teacher, args.data_seed, args.per_act)
    if not data["train"]:
        raise ValueError("no choice combat examples")
    before = validation(model, data["holdout"], args.device)
    model.to(args.device).train()
    optimizer = torch.optim.Adam(model.combat_parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    acts = sorted(data["train"])
    weights = [{1:.5,2:.3,3:.2}[act] for act in acts]
    metrics, exposures, unique = [], Counter(), set()
    for step in range(1,args.steps+1):
        selected = []
        for act in rng.choices(acts, weights=weights, k=args.batch_size):
            position = rng.randrange(len(data["train"][act]))
            selected.append((act,position))
        chosen = [data["train"][act][position] for act,position in selected]
        batch, targets = collate(chosen,args.device)
        logits, _ = model(*batch)
        loss = equivalence_loss(logits,batch[2],targets)
        with torch.no_grad():
            reference, _ = model.base(*batch)
            old_log = torch.log_softmax(reference,-1)
        new_log = torch.log_softmax(logits,-1)
        anchor = (old_log.exp()*(old_log-new_log)).sum(-1).mean()
        objective = loss+args.anchor*anchor
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.combat_parameters(),1.,error_if_nonfinite=True)
        after_norm = torch.nn.utils.get_total_norm([p.grad for p in model.combat_parameters() if p.grad is not None],error_if_nonfinite=True)
        optimizer.step()
        metrics.append({"step":step,"bc_loss":float(loss.detach().cpu()),"anchor_kl":float(anchor.detach().cpu()),
                        "grad_before":float(norm.cpu()),"grad_after":float(after_norm.cpu())})
        exposures.update(act for act,_ in selected)
        unique.update(selected)
        if step%args.checkpoint_every == 0 or step == args.steps:
            path = args.output/f"combat_{step:04d}.pt"
            save_policy(str(path),model,optimizer=optimizer,step=step,
                metadata={"stage":"native_combat_equivalence_bc","transfer":transfer,"data":data_record,
                    "config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    "actual_exposures_by_act":dict(exposures),"unique_rows_used":len(unique)})
            scores = validation(model,data["holdout"],args.device)
            atomic_json(args.output/f"metrics_{step:04d}.json",{"data":data_record,"before":before,"after":scores,
                "optimizer_metrics":metrics,"actual_exposures_by_act":dict(exposures),"unique_rows_used":len(unique),
                "checkpoint_sha256":file_identity(path),"elapsed_seconds":time.monotonic()-started,
                "note":"Native private-state teacher labels; held TRAIN worlds, not live-game improvement."})
            print(json.dumps({"step":step,"validation":scores,"seconds":time.monotonic()-started}),flush=True)
            model.train()
    # Freeze integrity is verified directly, including actor and critic weights.
    original, _ = initialize(args.base, seed=args.seed)
    if any(not torch.equal(a.detach().cpu(),b.detach().cpu()) for a,b in zip(model.base.parameters(),original.base.parameters())):
        raise RuntimeError("combat training moved frozen strategic/prior parameters")
    print(json.dumps({"done":True,"exposures":sum(exposures.values()),"unique":len(unique),"base_unchanged":True}),flush=True)


if __name__ == "__main__":
    main()
