"""Read-only critic trace diagnosis. No simulator execution or optimizer steps."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from sts_rl.joint_sparse import WinCritics, batch_rows
from sts_rl.critic_training import select_rows, held_world, metrics
from sts_policy.encoding import ACTION_FLAGS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
RUN = args.run.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
os.nice(10)
START = time.time()

def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def extract():
    chunks, metadata, inputs = [], [], []
    critic = WinCritics()
    for iteration in range(1, 11):
        path, = RUN.glob(f'buffer_{iteration:04d}_*.pt.gz')
        inputs.append(identity(path))
        with gzip.open(path, 'rb') as f:
            payload = torch.load(f, weights_only=False, map_location='cpu')
        rows = [r for r in select_rows(payload['rows']) if held_world(r['seed'])]
        xs = []
        for start in range(0, len(rows), 128):
            xs.append(critic.features(batch_rows(rows[start:start+128])).numpy())
        counts = {}
        for r in rows:
            key = r['seed'], r['phase']
            counts[key] = counts.get(key, 0) + 1
        for r in rows:
            flags = [flag for flag, value in zip(ACTION_FLAGS, r['actions'][r['position']][:32]) if value > .5]
            metadata.append({'iteration': iteration, 'seed': r['seed'], 'phase': r['phase'],
                             'act': r['act'], 'room': r['room'], 'floor': r['floor'],
                             'hp_fraction': float(r['state'][2]), 'hp': float(r['state'][0]*100),
                             'menu_size': len(r['actions']), 'chosen_action_index': r['action_index'],
                             'chosen_flags': flags, 'y': r['return'], 'behavior': r['value'],
                             'w': 1/counts[r['seed'], r['phase']]})
        chunks.append(np.concatenate(xs))
        del rows, payload, xs
        gc.collect()
        print(f'extracted {iteration}: {len(metadata)} held states', flush=True)
    x = np.concatenate(chunks)
    np.savez_compressed(OUT/'held_features.npz', x=x)
    (OUT/'held_states.json').write_text(json.dumps(metadata))
    return x, metadata, inputs

def predict(x, meta, path):
    state = torch.load(path, weights_only=False, map_location='cpu')
    critic = WinCritics(); critic.load_state_dict(state['critic_state']); critic.eval()
    pred = np.empty(len(x))
    with torch.no_grad():
        for phase in ('combat', 'ooc'):
            ids = np.array([i for i,m in enumerate(meta) if m['phase']==phase])
            for start in range(0, len(ids), 1024):
                ix = ids[start:start+1024]
                pred[ix] = critic.heads[phase](torch.from_numpy(x[ix])).squeeze(-1).sigmoid().numpy()
    return pred

def stats(meta, pred, mask):
    ids = np.flatnonzero(mask)
    if not len(ids): return None
    y = np.array([meta[i]['y'] for i in ids]); w = np.array([meta[i]['w'] for i in ids]); w /= w.sum()
    p = pred[ids]; mu = np.dot(w,y); var = np.dot(w,(y-mu)**2)
    brier = np.dot(w,(p-y)**2); error = y-p
    seeds = np.array([meta[i]['seed'] for i in ids]); unique, inv = np.unique(seeds,return_inverse=True)
    posmass, negmass = w*y, w*(1-y)
    _, rank = np.unique(p,return_inverse=True)
    pos = np.bincount(rank,weights=posmass); neg = np.bincount(rank,weights=negmass)
    auc = np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum()) if pos.sum()*neg.sum() else None
    return {'states':len(ids),'worlds':len(unique),'winning_worlds':len(set(seeds[y==1])),
            'return_mean':float(mu),'prediction_mean':float(np.dot(w,p)),
            'prediction_std':float(np.sqrt(np.dot(w,(p-np.dot(w,p))**2))),
            'brier':float(brier),'bce':float(-np.dot(w,y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1)))),
            'EV':float(1-np.dot(w,(error-np.dot(w,error))**2)/var) if var else None,
            'auc_weighted':float(auc) if auc is not None else None,
            'mean_prediction_wins':float(np.dot(w*y,p)/(w*y).sum()) if (w*y).sum() else None,
            'mean_prediction_losses':float(np.dot(w*(1-y),p)/(w*(1-y)).sum()) if (w*(1-y)).sum() else None}

def paired(meta, old, new, mask):
    ids = np.flatnonzero(mask)
    y = np.array([meta[i]['y'] for i in ids]); w = np.array([meta[i]['w'] for i in ids])
    seeds = np.array([meta[i]['seed'] for i in ids]); unique,inv = np.unique(seeds,return_inverse=True)
    mass = np.bincount(inv,weights=w)
    delta = np.bincount(inv,weights=w*((new[ids]-y)**2-(old[ids]-y)**2))
    rng = np.random.default_rng(4007); samples=[]
    for _ in range(10):
        ix = rng.integers(0,len(unique),size=(200,len(unique)))
        samples.extend(delta[ix].sum(1)/mass[ix].sum(1))
    return {'brier_new_minus_old':float(delta.sum()/mass.sum()),
            'world_bootstrap_ci95':np.quantile(samples,[.025,.975]).tolist(),'worlds':len(unique)}

def main():
    x,meta,inputs=extract()
    checkpoints={'warm':ROOT/'data/rl_runs/joint_critic_warm_v2_20261015/warm.pt',
                 'update8':next(RUN.glob('evaluated_0008_*.pt')),
                 'update10':next(RUN.glob('evaluated_0010_*.pt'))}
    preds={k:predict(x,meta,p) for k,p in checkpoints.items()}
    preds['behavior']=np.array([m['behavior'] for m in meta])
    np.savez_compressed(OUT/'predictions.npz',**preds)
    phase=np.array([m['phase'] for m in meta]);act=np.array([m['act'] for m in meta]);it=np.array([m['iteration'] for m in meta])
    masks={}
    for name,cohort in {'early1_4':it<=4,'warning7_10':it>=7,'all1_10':it>0,**{f'batch{i}':it==i for i in range(1,11)}}.items():
        for ph in ('combat','ooc'):
            for a in (0,1,2,3): masks[f'{name}/{ph}/act{a}']=cohort&(phase==ph)&((act==a) if a else True)
    summary={key:{model:stats(meta,p,mask) for model,p in preds.items()} for key,mask in masks.items()}
    comparisons={key:{'update10_vs_warm':paired(meta,preds['warm'],preds['update10'],mask),
                      'update10_vs_update8':paired(meta,preds['update8'],preds['update10'],mask)}
                 for key,mask in masks.items() if key.startswith(('warning','early','all'))}
    groups={}
    warning=(it>=7)&(phase=='ooc')
    for room in sorted(set(m['room'] for m in meta)):
        mask=warning&np.array([m['room']==room for m in meta])
        if mask.sum(): groups['room/'+room]={k:stats(meta,p,mask) for k,p in preds.items()}
    for flag in ('map node','rest','smith','buy','take card','skip card','take relic','take potion','take gold'):
        mask=warning&np.array([flag in m['chosen_flags'] for m in meta])
        if mask.sum():groups['chosen_semantic_flag/'+flag]={k:stats(meta,p,mask) for k,p in preds.items()}
    reliability={}
    for ph in ('combat','ooc'):
        for a in (2,3):
            for model in ('warm','update10'):
                pp=preds[model]; bins=[]
                for lo,hi in zip((0,.01,.03,.05,.1,.2,.4),(.01,.03,.05,.1,.2,.4,1.00001)):
                    mask=warning&(phase==ph)&(act==a)&(pp>=lo)&(pp<hi) if ph=='ooc' else (it>=7)&(phase==ph)&(act==a)&(pp>=lo)&(pp<hi)
                    value=stats(meta,pp,mask)
                    if value:bins.append({'range':[lo,hi],**value})
                reliability[f'{ph}/act{a}/{model}']=bins
    receipt={}
    for i in range(7,11):
        path=next(RUN.glob(f'collection_{i:04d}_*.json.gz'))
        with gzip.open(path,'rt') as f:r=json.load(f)
        receipt.update({e['world_seed']:e for e in r['episodes']})
    scores={}
    for i,m in enumerate(meta):
        if m['iteration']<7 or m['phase']!='ooc' or m['act'] not in (2,3):continue
        delta=(preds['update10'][i]-m['y'])**2-(preds['warm'][i]-m['y'])**2
        key=m['seed'];scores.setdefault(key,[]).append((delta,i))
    worst=sorted(scores,key=lambda s:np.mean([v[0] for v in scores[s]]),reverse=True)[:12]
    examples=[]
    for seed in worst:
        rows=sorted(scores[seed],reverse=True)[:3];episode=receipt[seed]
        examples.append({'seed':seed,'mean_state_brier_increase':float(np.mean([v[0] for v in scores[seed]])),
                         'episode':{k:episode.get(k) for k in ('classification','floor','act','steps','end_hp','boss_entries','termination_state','termination_reason')},
                         'states':[{**meta[i],'warm_prediction':float(preds['warm'][i]),'update8_prediction':float(preds['update8'][i]),'update10_prediction':float(preds['update10'][i])} for _,i in rows]})
    report={'schema':'joint_critic_trace_diagnosis_v1','created_at':time.time(),'seconds':time.time()-START,
            'sampling':'Exact same up-to8 uniformly strided choice states perworld/phase as training diagnostics; held-world hash exclusion verified. Perworld/phase totalweight1. No gradients or newgames. Decisionflag groups overlap and describe selected action semantics, not causal actionquality.',
            'limitations':['Checkpoints share data history; comparison is diagnostic, not independent model selection.','EV/AUC/score point estimates use correlated states; Brier differences use paired whole-world bootstrap2000 replicates.','Rolloutreturn is one stochastic future, not the state true winprobability; no counterfactual actions observed.','Early/late cohorts differ in seeds and policy; cannot alone identify policy-induced distribution shift.','All analyses are exploratory; many subgroups and limited positiveworldsupport.'],
            'inputs':inputs,'analysis_script':identity(Path(__file__)),'checkpoints':{k:identity(v) for k,v in checkpoints.items()},
            'summary':summary,'paired_comparisons':comparisons,'warning_cohort_groups':groups,'reliability':reliability,'largest_deterioration_examples':examples}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',report['seconds'],flush=True)

if __name__=='__main__':main()
