#!/usr/bin/env python3
"""Fixed-world-split value predictors; actor weights and deployed decisions unchanged."""
import argparse
from collections import Counter
import gzip,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import torch
    from torch import nn
    from sts_policy.encoding import encode_observation
    from sts_rl.recording import begin_run,capture_rng,file_identity
    from probe_fight_value import public_extras,EXTRA_NAMES
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=1024);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();out=Path(a.out)
    if out.exists():p.error('Fresh output required')
    directory=Path(a.bank);summary=json.loads((directory/'summary.json').read_text())
    if summary['faults'] or summary['censored'] or summary['unstarted']:raise ValueError('Only complete clean data accepted')
    record=begin_run(out,vars(a),a.checkpoint,inputs=[directory]);torch.set_num_threads(a.threads)
    old=[];extra=[];target=[];meta=[]
    for path in sorted(directory.glob('states_*.jsonl.gz')):
        for line in gzip.open(path,'rt'):
            row=json.loads(line);e=encode_observation(row['obs'])
            old.append(np.concatenate((e.state_features,e.action_features.mean(0))))
            extra.append(public_extras(row['obs']));target.append(float(row['fight_win']))
            meta.append((row['seed'],row['floor'],row['room']+str(row['act']),row['turn']))
    old=np.asarray(old,dtype=np.float32);extra=np.asarray(extra,dtype=np.float32);y=np.asarray(target,dtype=np.float32)
    held={seed for seed,_,_,_ in meta if int(hashlib.sha256(f'value-bank-held-v1:{seed}'.encode()).hexdigest()[:8],16)%5==0}
    test=np.asarray([seed in held for seed,_,_,_ in meta]);train=~test
    if not train.any() or not test.any():raise ValueError('Empty world split')
    counts=Counter((seed,floor) for seed,floor,_,_ in meta)
    first={}
    for i,(seed,floor,_,_) in enumerate(meta):first.setdefault((seed,floor),i)
    group_counts=Counter(meta[i][2] for i in first.values() if train[i])
    weights=np.asarray([1/counts[seed,floor]/max(1,group_counts[group]) for seed,floor,group,_ in meta],dtype=np.float32)
    weights/=weights[train].mean()
    train_ids=np.flatnonzero(train).tolist();test_ids=np.flatnonzero(test)
    priors={}
    for group in sorted({m[2] for m in meta}):
        rows=[i for i in first.values() if train[i] and meta[i][2]==group]
        priors[group]=float(y[rows].mean()) if rows else .5
    predictions={'group_prior':np.asarray([priors[m[2]] for m in meta])};models={};timings={}
    torch.manual_seed(17)
    base=nn.Sequential(nn.Linear(384,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    initial={k:v.clone() for k,v in base.state_dict().items()}
    for name in ['legacy','legacy_plus_public']:
        torch.manual_seed(17);rng=random.Random(17)
        model=nn.Sequential(nn.Linear(384,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1));model.load_state_dict(initial)
        branch=nn.Sequential(nn.Linear(len(EXTRA_NAMES),64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(branch[-1].weight);nn.init.zeros_(branch[-1].bias)
        center=old[train].mean(0);scale=np.maximum(old[train].std(0),1e-6)
        ec=extra[train].mean(0);es=np.maximum(extra[train].std(0),1e-6)
        x=torch.from_numpy(np.clip((old-center)/scale,-10,10));z=torch.from_numpy(np.clip((extra-ec)/es,-10,10))
        labels=torch.from_numpy(y);w=torch.from_numpy(weights)
        parameters=list(model.parameters())+(list(branch.parameters()) if name=='legacy_plus_public' else [])
        optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01);started=time.monotonic();steps=0
        for epoch in range(a.epochs):
            ids=train_ids.copy();rng.shuffle(ids)
            for start in range(0,len(ids),a.batch):
                idx=torch.tensor(ids[start:start+a.batch],dtype=torch.long);logits=model(x[idx]).squeeze(-1)
                if name=='legacy_plus_public':logits=logits+branch(z[idx]).squeeze(-1)
                loss=(nn.functional.binary_cross_entropy_with_logits(logits,labels[idx],reduction='none')*w[idx]).mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();steps+=1
        with torch.no_grad():
            logits=model(x).squeeze(-1)
            if name=='legacy_plus_public':logits=logits+branch(z).squeeze(-1)
            predictions[name]=torch.sigmoid(logits).numpy()
        timings[name]={'seconds':time.monotonic()-started,'updates':steps}
        artifact=out/(name+'.pt')
        torch.save({'schema':'sts_fight_value_predictor_v1','model_state':model.state_dict(),'extra_branch_state':branch.state_dict() if name=='legacy_plus_public' else None,
                    'optimizer_state':optimizer.state_dict(),'rng_state':capture_rng(rng),'center':center,'scale':scale,'extra_center':ec,'extra_scale':es,
                    'config':vars(a),'run_record':record,'parent_policy':file_identity(a.checkpoint),'extra_names':EXTRA_NAMES,
                    'architecture':{'base':[384,128,64,1],'extra':[len(EXTRA_NAMES),64,1] if name=='legacy_plus_public' else None}},artifact)
        models[name]=file_identity(artifact)
    # Every fight has equal weight inside its group; start metrics use one row.
    groups={}
    for group in sorted({m[2] for m in meta}):
        ids=[i for i in test_ids if meta[i][2]==group];starts=[i for i in first.values() if test[i] and meta[i][2]==group]
        if not ids:continue
        fw=np.asarray([1/counts[meta[i][0],meta[i][1]] for i in ids])
        groups[group]={'held_fights':len(starts),'held_states':len(ids),'wins':int(y[starts].sum()),
                       'all_state_brier':{name:float(np.average((pred[ids]-y[ids])**2,weights=fw)) for name,pred in predictions.items()},
                       'entry_brier':{name:float(np.mean((pred[starts]-y[starts])**2)) for name,pred in predictions.items()}}
    report={'schema':'sts_world_held_fight_value_fit_v1','run_record':record,'rows':len(meta),'fit_worlds':len({m[0] for m in meta})-len(held),
            'held_worlds':sorted(held),'groups':groups,'models':models,'timings':timings,'epochs':a.epochs,
            'scope':'Fixed 20% whole-world held split, fixed epochs/hyperparameters, matched base initialization/shuffles. Augmented model adds a zero-initialized residual branch over 35 public fields. Fit-only normalization; group-balanced completed-fight loss. Entry metrics are first decision per fight. No actor updates or deployment/competence claim.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'groups':groups,'timings':timings}),flush=True)

if __name__=='__main__':main()
