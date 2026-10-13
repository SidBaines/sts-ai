"""Matched preference/one-hot controls on filtered current-profile TRAIN panels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from sts_policy import entities
from sts_policy.combat_preferences import preferences,preference_loss,TARGET_VERSION
from sts_policy.combat_residual import initialize,equivalent_positions,equivalence_loss,save_policy
from sts_policy.composition import training_environment_record
from sts_policy.curriculum import world_split
from sts_policy.training import file_identity
from scripts.train_combat_residual_v3 import reservoir,validate_teacher_manifest
from scripts.run_strategic_v3 import atomic_json


def prepare(path,seed,cap):
    rng=random.Random(seed);groups={'train':{},'holdout':{}};seen=Counter();discarded=Counter()
    with path.open() as source:
        for line in source:
            record=json.loads(line);world=int(record['world_seed'])
            if world_split(world)!='train':
                raise ValueError('preference supervision is TRAIN-only')
            obs=record['observation'];act=int(obs['state']['act'])
            if obs['phase']!='combat' or len(obs['legal_actions'])<2:
                continue
            row=entities.encode_observation(obs)
            pairs=preferences(row,record['teacher_panels']['1000'])
            if not pairs:
                discarded[act]+=1;continue
            split='holdout' if int(hashlib.sha256(f'combat-holdout-v1:{world}'.encode()).hexdigest()[:8],16)%5==0 else 'train'
            teacher=row.action_indices.tolist().index(record['teacher_index'])
            item=(row,pairs,equivalent_positions(row,teacher),world)
            seen[(split,act)]+=1
            reservoir(groups[split].setdefault(act,[]),item,seen[(split,act)],cap if split=='train' else 512,rng)
    worlds={s:{r[3] for rows in phases.values() for r in rows} for s,phases in groups.items()}
    if worlds['train']&worlds['holdout']:
        raise ValueError('holdout world leakage')
    return groups,{'target_version':TARGET_VERSION,'teacher_sha256':file_identity(path),
        'source_counts':{f'{s}:act{a}':n for (s,a),n in seen.items()},'discarded_without_stable_pairs':dict(discarded),
        'pool_counts':{s:{str(a):len(rows) for a,rows in phases.items()} for s,phases in groups.items()},
        'world_counts':{s:len(w) for s,w in worlds.items()},
        'thresholds':{'minimum_visits':8,'minimum_conditions':4,'minimum_gap':.15,'agreement':.8,'maximum_pairs':8}}


def loss(model,rows,device,target):
    batch=entities.collate_observations([r[0] for r in rows],device)
    logits,_=model(*batch)
    if target in {'preference','correction'}:
        objective=preference_loss(logits,[r[1] for r in rows],hinge_margin=.25 if target=='correction' else None)
    else:
        masks=torch.zeros_like(batch[2])
        for i,row in enumerate(rows):masks[i,:len(row[2])]=torch.as_tensor(row[2],device=device)
        objective=equivalence_loss(logits,batch[2],masks)
    return objective,logits,batch


@torch.no_grad()
def validation(model,data,device):
    result={}
    for act,rows in data.items():
        total=order=weight=exact=0.
        for start in range(0,len(rows),128):
            chosen=rows[start:start+128];value,logits,_=loss(model,chosen,device,'preference')
            total+=float(value.cpu())*len(chosen)
            logits=logits.cpu().numpy()
            for i,row in enumerate(chosen):
                exact+=bool(row[2][logits[i,:len(row[2])].argmax()])
                for high,low,w in row[1]:
                    order+=float(logits[i,:len(high)][high].mean()>logits[i,:len(low)][low].mean())*w;weight+=w
        result[str(act)]={'rows':len(rows),'preference_loss':total/max(1,len(rows)),
                          'weighted_order_accuracy':order/max(1e-9,weight),'primary_teacher_agreement':exact/max(1,len(rows))}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--target',choices=['preference','onehot','correction'],required=True)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--data-seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=160);parser.add_argument('--per-act',type=int,default=6000)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps');args=parser.parse_args()
    if args.output.exists() or min(args.steps,args.per_act)<=0:parser.error('fresh output and positive bounds required')
    torch.set_num_threads(1);started=time.monotonic();environment=training_environment_record()
    manifest=validate_teacher_manifest(args.teacher,environment)
    data,record=prepare(args.teacher,args.data_seed,args.per_act)
    if not data['train']:raise ValueError('no supported preference data')
    model,transfer=initialize(args.base,seed=args.seed);model.to(args.device)
    if args.target=='correction':
        correction_counts={}
        with torch.no_grad():
            for split,phases in data.items():
                correction_counts[split]={}
                for act,rows in list(phases.items()):
                    retained=[]
                    for start in range(0,len(rows),128):
                        chosen=rows[start:start+128]
                        logits,_=model.base(*entities.collate_observations([r[0] for r in chosen],args.device))
                        for row,pick in zip(chosen,logits.argmax(-1).cpu().tolist()):
                            pairs=[pair for pair in row[1] if pair[1][pick]]
                            if pairs:retained.append((row[0],pairs,row[2],row[3]))
                    correction_counts[split][str(act)]=len(retained)
                    if retained:phases[act]=retained
                    else:del phases[act]
        record.update(correction_pool_counts=correction_counts,correction_margin=.25,
                      correction_note='Only supported pairs whose low group contains the fixed original greedy choice; hinge stops at margin.')
        if not data['train']:raise ValueError('no supported original-policy corrections')
    original={k:v.clone() for k,v in model.base.state_dict().items()}
    before=validation(model,data['holdout'],args.device)
    optimizer=torch.optim.Adam(model.combat_parameters(),lr=3e-4);rng=random.Random(args.seed)
    acts=sorted(data['train']);weights=[{1:.5,2:.3,3:.2}[a] for a in acts]
    args.output.mkdir(parents=True);config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    snapshot=args.output/'source';snapshot.mkdir()
    for source in (Path(__file__),Path('src/sts_policy/combat_preferences.py')):
        (snapshot/source.name).write_bytes(source.read_bytes())
    atomic_json(args.output/'launch.json',{'config':config,'data':record,'teacher_manifest':manifest,
        'environment':environment,'trainer_sha256':file_identity(__file__),'target_code_sha256':file_identity('src/sts_policy/combat_preferences.py')})
    metrics=[];exposures=Counter();unique=set()
    for step in range(1,args.steps+1):
        selected=[(a,rng.randrange(len(data['train'][a]))) for a in rng.choices(acts,weights=weights,k=256)]
        chosen=[data['train'][a][i] for a,i in selected]
        objective,logits,batch=loss(model,chosen,args.device,args.target)
        with torch.no_grad():old,_=model.base(*batch);old_log=torch.log_softmax(old,-1)
        new_log=torch.log_softmax(logits,-1);anchor=(old_log.exp()*(old_log-new_log)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True);(objective+.1*anchor).backward()
        norm=torch.nn.utils.clip_grad_norm_(model.combat_parameters(),1.,error_if_nonfinite=True)
        clipped=torch.nn.utils.get_total_norm([p.grad for p in model.combat_parameters() if p.grad is not None],error_if_nonfinite=True)
        optimizer.step();exposures.update(a for a,_ in selected);unique.update(selected)
        metrics.append({'step':step,'target_loss':float(objective.detach().cpu()),'anchor_kl':float(anchor.detach().cpu()),
                        'grad_before':float(norm.cpu()),'grad_after':float(clipped.cpu())})
        if step%40==0 or step==args.steps:
            path=args.output/f'combat_{step:04d}.pt';save_policy(str(path),model,optimizer=optimizer,step=step,
                metadata={'stage':TARGET_VERSION,'target':args.target,'transfer':transfer,'data':record,'config':config,
                          'environment':environment,'actual_exposures_by_act':dict(exposures),'unique_rows_used':len(unique)})
            after=validation(model,data['holdout'],args.device)
            atomic_json(args.output/f'metrics_{step:04d}.json',{'data':record,'config':config,'before':before,'after':after,
                'optimizer_metrics':metrics,'actual_exposures_by_act':dict(exposures),'unique_rows_used':len(unique),
                'checkpoint_sha256':file_identity(path),'elapsed_seconds':time.monotonic()-started,
                'note':'Private search preferences are provisional; live outcomes and retention decide usefulness.'})
            print(json.dumps({'step':step,'validation':after}),flush=True)
    if any(not torch.equal(value,original[k]) for k,value in model.base.state_dict().items()):
        raise RuntimeError('frozen base changed')
    atomic_json(args.output/'completion.json',{'steps':args.steps,'base_unchanged':True,'checkpoint_sha256':file_identity(path)})


if __name__=='__main__':main()
