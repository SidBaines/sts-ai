"""Bounded native combat cloning with public inputs and frozen strategic behavior."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from sts_policy import entities
from sts_policy.combat_residual import initialize, equivalent_positions, equivalence_loss, save_policy
from sts_policy.curriculum import world_split
from sts_policy.composition import training_environment_record
from sts_policy.curriculum import validate_provenance
from sts_env import __version__ as ENV_VERSION
from sts_env.env import OBSERVATION_VERSION
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


def validate_teacher_manifest(path, current):
    manifest_path = path.parent/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "provenance" in manifest:
        actual = {"environment_version":ENV_VERSION,"observation_version":OBSERVATION_VERSION,
                  "native_provenance":current.get("sourced_env_manifest"),
                  **{k:current.get(k) for k in ("python_interface_sha256","runtime_python_version")}}
        validate_provenance(manifest["provenance"],actual)
    elif "environment" in manifest:
        for key in ("python_interface_sha256","runtime_python_version","native_module_sha256"):
            if current.get(key) is None or manifest["environment"].get(key) != current[key]:
                raise ValueError(f"teacher native/interface provenance mismatch: {key}")
        purity = manifest.get("purity",{})
        if (not purity.get("cohorts_complete") or purity.get("faults") or
                purity.get("matched_traces") != purity.get("expected_tasks")):
            raise ValueError("learner-state teacher query-purity gate is unavailable or failed")
        if manifest.get("teacher_sha256") != file_identity(path):
            raise ValueError("teacher data does not match its manifest digest")
    else:
        raise ValueError("teacher bank lacks native/interface provenance")
    return {"manifest_path":str(manifest_path),"manifest_sha256":file_identity(manifest_path),
            "native_module_sha256":current["native_module_sha256"],"data_version":manifest.get("data_version")}


def prepare(path, seed, per_act, teacher_budget=None, panel_consensus=False):
    rng = random.Random(seed)
    groups = {"train": {}, "holdout": {}}
    counts = Counter()
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            if world_split(int(record["world_seed"])) != "train":
                raise ValueError("combat supervision/validation must be TRAIN-only")
            obs = record["observation"]
            if obs["phase"] != "combat" or len(obs["legal_actions"]) < 2:
                continue
            if teacher_budget is not None:
                key = str(teacher_budget)
                if key not in record.get("teacher_by_budget",{}):
                    raise ValueError("requested same-state teacher budget is unavailable")
                record["teacher_index"] = int(record["teacher_by_budget"][key])
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
                item = (row, equivalent_positions(row, teacher), teacher)
                if panel_consensus:
                    panel = record.get("teacher_panels",{}).get("1000")
                    if panel is None:
                        raise ValueError("consensus targets require an explicitly recorded 1000-simulation panel")
                    masks = []
                    for result in panel["results"]:
                        if result["status"]=="ok":
                            position = row.action_indices.tolist().index(int(result["index"]))
                            masks.append(equivalent_positions(row,position))
                    if not masks:
                        raise ValueError("teacher panel has no valid labels")
                    item = (*item,np.stack(masks))
                converted.append(item)
                worlds[split].add(int(record["world_seed"]))
            encoded[split][act] = converted
    if worlds["train"] & worlds["holdout"]:
        raise ValueError("combat validation world leakage")
    return encoded, {"source_counts": {f"{s}:act{a}":n for (s,a),n in counts.items()},
        "pool_counts": {s:{str(a):len(r) for a,r in phases.items()} for s,phases in encoded.items()},
        "world_counts": {s:len(w) for s,w in worlds.items()},
        "teacher_sha256": file_identity(path),"teacher_budget_override":teacher_budget,
        "target_version":"native_search_draw_consensus_v1" if panel_consensus else "native_model_equivalent_onehot_v1"}


def collate(rows, device):
    batch = entities.collate_observations([row[0] for row in rows], device)
    if len(rows[0])==4:
        groups=max(len(row[3]) for row in rows)
        targets=np.zeros((len(rows),groups,batch[2].shape[1]),np.bool_)
        weights=np.zeros((len(rows),groups),np.float32)
        for i,row in enumerate(rows):
            targets[i,:len(row[3]),:row[3].shape[1]]=row[3]
            weights[i,:len(row[3])]=1/len(row[3])
        return batch,(torch.as_tensor(targets,device=device),torch.as_tensor(weights,device=device))
    target = np.zeros(tuple(batch[2].shape), np.bool_)
    for i, row in enumerate(rows):
        target[i, :len(row[1])] = row[1]
    return batch, torch.as_tensor(target, device=device)


def target_loss(logits,mask,targets):
    if not isinstance(targets,tuple):
        return equivalence_loss(logits,mask,targets)
    groups,weights=targets
    if (groups.shape[:2]!=weights.shape or groups.shape[2]!=mask.shape[1] or
            not bool(torch.isfinite(weights).all()) or not bool((weights>=0).all()) or
            not bool(torch.isclose(weights.sum(-1),torch.ones_like(weights[:,0])).all()) or
            not bool((((groups & mask[:,None,:]).any(-1)) | (weights==0)).all())):
        raise ValueError("valid normalized consensus group weights and legal masks required")
    log_prob=torch.log_softmax(logits.masked_fill(~mask,-1e9),-1)
    group_log=torch.logsumexp(log_prob[:,None,:].masked_fill(~groups,-1e9),-1)
    return -(group_log*weights).sum(-1).mean()


@torch.no_grad()
def validation(model, phases, device):
    model.to(device).eval()
    result = {}
    for act, rows in phases.items():
        loss, exact, equivalent, count, panel_agreement = 0., 0, 0, 0, 0.
        for start in range(0, len(rows), 128):
            chosen = rows[start:start+128]
            batch, targets = collate(chosen, device)
            logits, _ = model(*batch)
            loss += float(target_loss(logits, batch[2], targets).cpu())*len(chosen)
            picks = logits.argmax(-1).cpu().tolist()
            for row, pick in zip(chosen, picks):
                exact += pick == row[2]
                equivalent += bool(row[1][pick])
                if len(row)==4:
                    panel_agreement += float(row[3][:,pick].mean())
            count += len(chosen)
        result[str(act)] = {"rows":count,"equivalence_loss":loss/max(1,count),
                           "exact_route_accuracy":exact/max(1,count),"model_equivalent_accuracy":equivalent/max(1,count),
                           "expected_panel_agreement":panel_agreement/max(1,count) if rows and len(rows[0])==4 else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--teacher-budget",type=int,help="Choose an explicitly recorded same-state search budget")
    parser.add_argument("--panel-consensus",action="store_true",help="Average loss over recorded search/draw intervention label groups")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--per-act", type=int, default=6000)
    parser.add_argument("--anchor", type=float, default=.1)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--device", choices=("cpu","mps"), default="mps")
    parser.add_argument("--build-dir",type=Path,default=Path(os.environ.get("STS_LIGHTSPEED_BUILD",
        str(Path(__file__).resolve().parents[1]/"external/sts_lightspeed/build"))))
    args = parser.parse_args()
    if args.output.exists() or min(args.steps,args.batch_size,args.per_act,args.lr,args.checkpoint_every) <= 0:
        parser.error("use fresh output and positive training settings")
    torch.set_num_threads(1)
    started = time.monotonic()
    os.environ["STS_LIGHTSPEED_BUILD"] = str(args.build_dir.resolve())
    environment = training_environment_record()
    teacher_manifest = validate_teacher_manifest(args.teacher,environment)
    model, transfer = initialize(args.base, seed=args.seed)
    args.output.mkdir(parents=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    atomic_json(args.output/"launch.json",{"config":config,"transfer":transfer,"environment":environment,
        "teacher_manifest":teacher_manifest,"trainer_sha256":file_identity(__file__)})
    data, data_record = prepare(args.teacher, args.data_seed, args.per_act, args.teacher_budget,args.panel_consensus)
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
        loss = target_loss(logits,batch[2],targets)
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
                    "config":config,"environment":environment,"teacher_manifest":teacher_manifest,
                    "actual_exposures_by_act":dict(exposures),"unique_rows_used":len(unique)})
            scores = validation(model,data["holdout"],args.device)
            atomic_json(args.output/f"metrics_{step:04d}.json",{"data":data_record,"before":before,"after":scores,
                "optimizer_metrics":metrics,"actual_exposures_by_act":dict(exposures),"unique_rows_used":len(unique),
                "checkpoint_sha256":file_identity(path),"elapsed_seconds":time.monotonic()-started,
                "config":config,"teacher_manifest":teacher_manifest,
                "note":"Native private-state teacher labels; held TRAIN worlds, not live-game improvement."})
            print(json.dumps({"step":step,"validation":scores,"seconds":time.monotonic()-started}),flush=True)
            model.train()
    # Freeze integrity is verified directly, including actor and critic weights.
    original, _ = initialize(args.base, seed=args.seed)
    if any(not torch.equal(a.detach().cpu(),b.detach().cpu()) for a,b in zip(model.base.parameters(),original.base.parameters())):
        raise RuntimeError("combat training moved frozen strategic/prior parameters")
    atomic_json(args.output/"completion.json",{"steps":args.steps,"base_unchanged":True,
        "checkpoint":str(path),"checkpoint_sha256":file_identity(path),"elapsed_seconds":time.monotonic()-started})
    print(json.dumps({"done":True,"exposures":sum(exposures.values()),"unique":len(unique),"base_unchanged":True}),flush=True)


if __name__ == "__main__":
    main()
