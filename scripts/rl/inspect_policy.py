#!/usr/bin/env python3
"""Bounded reached-state strategy statistics and public-encoding contrast probes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import statistics
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def category(obs):
    if obs['phase']=='combat': return 'combat'
    descriptions=[a['description'].lower() for a in obs['legal_actions']]
    if any('neow' in d for d in descriptions): return 'neow'
    if any(d.startswith('take boss relic') for d in descriptions): return 'boss_relic'
    if any(d.startswith('smith') for d in descriptions): return 'rest'
    if any(d.startswith('leave shop') for d in descriptions): return 'shop'
    if any(d.startswith('skip card') for d in descriptions): return 'card_reward'
    if any('map node' in d for d in descriptions): return 'map'
    return str(obs.get('state',{}).get('screen_state','other'))


def main():
    import numpy as np
    import torch
    from sts_env.env import SimulatorError
    from sts_policy.encoding import encode_observation
    from sts_policy.model_agent import collate_public, forward_policy
    from sts_rl.common import JsonlLog, load_policy, make_workers, train_tasks
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--n',type=int,default=64)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--world-start',type=int,default=40_000_000_000)
    parser.add_argument('--progress-output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    if (out/'summary.json').exists():parser.error('Output already exists')
    out.mkdir(parents=True,exist_ok=True)
    identity=begin_run(out,vars(args),args.checkpoint)
    torch.set_num_threads(1)
    model,_=load_policy(args.checkpoint);model.eval()
    tasks=train_tasks(args.world_start,args.n)
    lock=threading.Lock();progress=JsonlLog(args.progress_output)

    def run(slot):
        worker=make_workers(1,dead_action_filter=True,slim=False)[0]
        counts=defaultdict(Counter);gaps=defaultdict(list);probes=defaultdict(Counter)
        examples=[];outcomes=[];combat_seen=0
        with gzip.open(out/f'samples_{slot:02d}.jsonl.gz','wt',compresslevel=1) as sink:
            try:
                for task in tasks[slot::args.threads]:
                    try:
                        obs,info=worker.submit('reset',task).result();term=trunc=False
                        while not (term or trunc):
                            with lock,torch.no_grad():logits,_,_=forward_policy(model,collate_public([obs]))
                            choices=logits[0,:len(obs['legal_actions'])];pos=int(choices.argmax())
                            kind=category(obs)
                            if len(choices)>1:
                                counts[kind][obs['legal_actions'][pos]['description']]+=1
                                gaps[kind].append(float(choices.topk(2).values.diff().abs()[0]))
                            if kind!='combat' and len(choices)>1 and kind!='map':
                                sink.write(json.dumps({'seed':task.seed,'category':kind,'obs':obs,
                                    'logits':choices.tolist(),'chosen_position':pos})+'\n')
                            if kind=='combat':
                                combat_seen+=1
                                if combat_seen%11==0:
                                    base=encode_observation(obs)
                                    enemies=obs.get('state',{}).get('combat',{}).get('enemies',[])
                                    for key,increment in [('max_hp',17),('block',7)]:
                                        if not enemies or key not in enemies[0]:continue
                                        changed=deepcopy(obs);changed['state']['combat']['enemies'][0][key]+=increment
                                        enc=encode_observation(changed)
                                        state_changed=not np.array_equal(base.state_features,enc.state_features)
                                        action_changed=not np.array_equal(base.action_features,enc.action_features)
                                        probes[key]['tested']+=1;probes[key]['state_changed']+=state_changed
                                        probes[key]['actions_changed']+=action_changed
                                        if not state_changed and not action_changed and len(examples)<12:
                                            examples.append({'kind':'constructed_public_contrast_on_reached_state',
                                                'seed':task.seed,'field':key,'before':enemies[0][key],
                                                'after':changed['state']['combat']['enemies'][0][key], 'obs':obs})
                                    cards=obs.get('state',{}).get('combat',{}).get('cards',{}).get('hand',[])
                                    name_costs=defaultdict(set)
                                    for card in cards:
                                        name_costs[card.get('name')].add(card.get('cost_for_turn',card.get('cost')))
                                    if any(len(v)>1 for v in name_costs.values()):
                                        probes['same_name_different_cost']['reached_states']+=1
                                        if len(examples)<16:examples.append({'kind':'reached_same_name_different_cost',
                                            'seed':task.seed,'obs':obs})
                            obs,_,term,trunc,info=worker.submit('step',int(obs['legal_actions'][pos]['index'])).result()
                        rec={'seed':task.seed,'floor':obs['state'].get('floor'),'outcome':info.get('task_outcome')}
                    except SimulatorError as exc:
                        rec={'seed':task.seed,'fault':str(exc),'fault_record_path':worker.worker.last_fault_path}
                    outcomes.append(rec)
                    with lock:progress.emit({'event':'world','slot':slot,**rec})
            finally:worker.close()
        return counts,gaps,probes,examples,outcomes

    with ThreadPoolExecutor(args.threads) as executor:parts=list(executor.map(run,range(args.threads)))
    counts=defaultdict(Counter);gaps=defaultdict(list);probes=defaultdict(Counter);examples=[];outcomes=[]
    for cs,gs,ps,es,os in parts:
        for k,v in cs.items():counts[k].update(v)
        for k,v in gs.items():gaps[k]+=v
        for k,v in ps.items():probes[k].update(v)
        examples+=es;outcomes+=os
    summary={'schema':'sts_reached_policy_audit_v1','run_record':identity,
        'choice_counts':{k:dict(v) for k,v in counts.items()},
        'median_top2_logit_gap':{k:statistics.median(v) for k,v in gaps.items() if v},
        'encoding_probes':{k:dict(v) for k,v in probes.items()},'outcomes':outcomes,
        'probe_scope':'Synthetic single-field contrasts on real reached observations; not proof both variants are reachable under identical history.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with gzip.open(out/'encoding_examples.json.gz','wt',compresslevel=1) as sink:json.dump(examples,sink)
    progress.emit({'event':'done','output':str(out),'worlds':len(outcomes),'faults':sum('fault' in r for r in outcomes)})


if __name__=='__main__':main()
