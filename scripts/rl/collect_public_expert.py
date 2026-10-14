#!/usr/bin/env python3
"""Paired fight outcomes and labels from fights actually played by public search."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main():
    import torch
    from sts_env.env import SimulatorError
    from sts_policy.curriculum import TaskSpec,public_hash
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_policy.workers import CurriculumWorker
    from sts_rl.common import WORKER_CONFIG,JsonlLog,load_policy
    from sts_rl.panel import PanelWorker
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--bank',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--groups',default='BOSS2:40,BOSS3:24')
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--simulations',type=int,default=1000)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seconds',type=float,default=1800.)
    args=parser.parse_args();out=Path(args.out)
    if (out/'summary.json').exists() or any(out.glob('shard_*.jsonl')):parser.error('Output already has results')
    out.mkdir(parents=True,exist_ok=True)
    run_record=begin_run(out,vars(args),args.checkpoint,inputs=[args.bank])
    torch.set_num_threads(1);model,_=load_policy(args.checkpoint);model.eval()
    rows=[json.loads(l) for l in Path(args.bank).read_text().splitlines() if l.strip()]
    random.Random(17).shuffle(rows);picked=[]
    for group,count in (part.split(':') for part in args.groups.split(',')):
        seen=set()
        for row in rows:
            if f"{row['room']}{row['act']}"!=group or row['task']['seed'] in seen:continue
            picked.append(row);seen.add(row['task']['seed'])
            if len(seen)>=int(count):break
    log=JsonlLog(out/'log.jsonl');lock=threading.Lock();started=time.monotonic()
    log.emit({'event':'start','config':vars(args),'picked':len(picked),'run_record':run_record})

    def choose(obs):
        with lock,torch.no_grad():logits,_,_=forward_policy(model,collate_public([obs]))
        return int(obs['legal_actions'][int(logits[0].argmax())]['index'])

    def run(slot):
        base=CurriculumWorker(WORKER_CONFIG,timeout=60.)
        expert=PanelWorker(WORKER_CONFIG,timeout=60.)
        results=[]
        with (out/f'shard_{slot:02d}.jsonl').open('x') as sink:
            try:
                for row in picked[slot::args.threads]:
                    if time.monotonic()-started>=args.seconds:break
                    spec=TaskSpec.from_dict(row['task']);labels=[]
                    rec={'seed':spec.seed,'group':f"{row['room']}{row['act']}",'floor':row['floor']}
                    try:
                        initial,_=base.reset(spec);obs=initial;done=False;cut=False
                        while not (done or cut):obs,_,done,cut,info=base.step(choose(obs))
                        if cut:raise RuntimeError('Baseline fight censored; no labels accepted')
                        rec['base_win']=info.get('task_outcome')=='battle_victory'
                        obs,_=expert.reset(spec)
                        if public_hash(obs)!=public_hash(initial):raise RuntimeError('Paired initial state differs')
                        done=cut=False;steps=0
                        while not (done or cut):
                            if len(obs['legal_actions'])>1:
                                votes=expert.search_votes_public(args.simulations,list(range(1,args.samples+1)),int(public_hash(obs)[:8],16))
                            else:votes={}
                            if votes:
                                action=max(votes.items(),key=lambda kv:kv[1])[0]
                                labels.append({'type':'combat','seed':spec.seed,'obs':obs,'votes':votes,
                                               'teacher_played':True,'teacher_protocol':'canonical_rng_resampled_search_played_fight_v2'})
                            else:action=choose(obs)
                            obs,_,done,cut,info=expert.step(action);steps+=1
                        if cut:raise RuntimeError('Expert fight censored; no labels accepted')
                        rec.update(expert_win=info.get('task_outcome')=='battle_victory',labels=len(labels),steps=steps)
                        for label in labels:
                            label['teacher_fight_win']=rec['expert_win']
                            sink.write(json.dumps(label)+'\n')
                        sink.flush()
                    except SimulatorError as exc:
                        rec.update(fault=str(exc),discarded_labels=len(labels),
                                   fault_record_path=expert.last_fault_path or base.last_fault_path)
                    results.append(rec)
                    with lock:log.emit({'event':'fight',**rec})
            finally:base.close();expert.close()
        return results

    with ThreadPoolExecutor(args.threads) as executor:results=[r for part in executor.map(run,range(args.threads)) for r in part]
    summary={}
    for group in sorted({r['group'] for r in results}):
        valid=[r for r in results if r['group']==group and 'fault' not in r]
        summary[group]={'n':len(valid),'base_wins':sum(r['base_win'] for r in valid),
                        'expert_wins':sum(r['expert_win'] for r in valid),
                        'base_only':sum(r['base_win'] and not r['expert_win'] for r in valid),
                        'expert_only':sum(r['expert_win'] and not r['base_win'] for r in valid)}
    report={'schema':'canonical_rng_resampled_search_played_fight_v2','run_record':run_record,'requested':len(picked),
            'completed':len(results),'unstarted':len(picked)-len(results),'groups':summary,
            'labels':sum(r.get('labels',0) for r in results),'faults':sum('fault' in r for r in results),'fights':results}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    log.emit({'event':'done','groups':summary,'labels':report['labels'],'faults':report['faults'],'unstarted':report['unstarted']})


if __name__=='__main__':main()
