#!/usr/bin/env python3
"""Bounded, TRAIN-only phase-submodel crossover; no optimizer/training changes."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.rollout_training import task_identity
from sts_policy.training import load_checkpoint
from sts_policy.model_agent import policy_from_payload
from sts_policy.long_run_control import atomic_json, sha_file, digest, panel_quality, process_birth, track_owned_members, stop_owned_group
ROOT=Path(__file__).resolve().parents[1]


def phase_swap(old, new, combat, ooc):
    payload=copy.deepcopy(old)
    models={'O':old,'N':new}
    if old['model_config']!=new['model_config'] or old['model_config'].get('kind')!='composite':
        raise ValueError('Crossover needs compatible complete composite configs')
    payload['model_state']={key:models[combat if key.startswith('combat.') else ooc]['model_state'][key].clone()
                            for key in old['model_state']}
    if any(not key.startswith(('combat.','ooc.')) for key in payload['model_state']):raise ValueError('Unexpected shared tensor; cannot perform clean swap')
    for key,tensor in payload['model_state'].items():
        origin=models[combat if key.startswith('combat.') else ooc]['model_state'][key]
        if not torch.equal(tensor,origin):raise ValueError('Incomplete phase transplant')
    payload['optimizer_state']=None
    payload['metadata']={**old['metadata'],'stage':'diagnostic_phase_crossover','diagnostic_only':True,'phases':{'combat':combat,'ooc':ooc}}
    policy_from_payload(payload).eval()  # Validate strict architecture/encoder loading on CPU.
    return payload


def verify_phase_payload(payload,old,new,combat,ooc):
    models={'O':old,'N':new}
    if payload['model_config']!=old['model_config'] or set(payload['model_state'])!=set(old['model_state']):
        raise ValueError('Cached composite architecture/keyset mismatch')
    if payload.get('optimizer_state') is not None:raise ValueError('Diagnostic checkpoint must not retain optimizer')
    for key,tensor in payload['model_state'].items():
        if not key.startswith(('combat.','ooc.')):raise ValueError('Unexpected shared routing tensor')
        donor=models[combat if key.startswith('combat.') else ooc]['model_state'][key]
        if tensor.dtype!=donor.dtype or not torch.equal(tensor,donor):raise ValueError('Cached diagnostic phase tensor differs from donor')
    policy_from_payload(payload).eval()


def prepare(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    s=json.loads(args.journal.read_text())
    if s['status'] not in ('operator_paused','attention_required','pilot_complete','allocation_complete'):raise RuntimeError('Learner must remain paused')
    old=load_checkpoint(args.original);new=load_checkpoint(args.regressed)
    manifest={'diagnostic_only':True,'created':time.time(),'source':'TRAIN causal crossover','policy':'deterministic argmax; seed777 taskordering',
              'original':{'path':str(args.original),'sha256':sha_file(args.original)},'regressed':{'path':str(args.regressed),'sha256':sha_file(args.regressed)},
              'native_backend':s['identity']['backend'],'sources':s['identity']['sources'],'conditions':{},'panels':{}}
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    for combat,ooc in (('O','O'),('N','N'),('O','N'),('N','O')):
        name=combat+ooc;path=checkpoints/(name+'.pt')
        if not path.exists():
            payload=phase_swap(old,new,combat,ooc);temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)
        verify_phase_payload(load_checkpoint(path),old,new,combat,ooc)
        manifest['conditions'][name]={'path':str(path),'sha256':sha_file(path),'combat':combat,'ooc':ooc,
            'meaning':{'OO':'original both','NN':'regressed both','ON':'original combat; regressed OOC','NO':'regressed combat; original OOC'}[name]}
    tasks=[];seed=args.world_start
    while len(tasks)<64:
        if world_split(seed)=='train':tasks.append(TaskSpec(seed=seed,kind='run',seek_combat=False,max_act=3,horizon=4096,battle_simulations=200))
        seed+=1
    for i in range(2):
        chunk=tasks[i*32:(i+1)*32];path=out/f'fresh_train_{i}.jsonl';text=''.join(json.dumps(t.to_dict(),sort_keys=True)+'\n' for t in chunk)
        if path.exists() and path.read_text()!=text:raise RuntimeError('Immutable diagnostic cohort changed')
        path.write_text(text)
        manifest['panels'][str(i)]={'path':str(path),'sha256':sha_file(path),'task_ids':[task_identity(t) for t in chunk],'worlds':[t.seed for t in chunk]}
    atomic_json(out/'manifest.json',manifest);return manifest


def evaluate(args,manifest,condition,panel):
    directory=args.output/'jobs';directory.mkdir(exist_ok=True)
    metrics=directory/f'{condition}_panel{panel}.metrics.json'
    cp=manifest['conditions'][condition]
    command=['/usr/bin/nice','-n','15',sys.executable,str(ROOT/'scripts/train_nonllm.py'),'evaluate',
        '--checkpoint',cp['path'],'--tasks',manifest['panels'][str(panel)]['path'],'--preserve-task-prefixes',
        '--task-kind','run','--split','train','--max-act','3','--horizon','4096','--episodes','32','--seed','777','--simulations','200',
        '--device','cpu','--inference-device','cpu','--workers','4','--threads','1','--seconds','120',
        '--fast-observations','--fast-replay','--deferred-bootstrap','--reward-mode','progress','--gamma','1','--gae-lambda','.95',
        '--progress-floor-weight','.05','--terminal-hp-weight','0','--goal-act-clear',
        '--output',str(metrics.with_suffix('.unused.pt')),'--metrics',str(metrics)]
    record_path=directory/f'{condition}_panel{panel}.record.json'
    if record_path.exists():
        r=json.loads(record_path.read_text())
        if r.get('status')=='complete' and sha_file(metrics)==r['metrics_sha256']:return r
        raise RuntimeError('Prior incomplete artifacts preserved; use another output directory')
    s=json.loads(args.journal.read_text())
    if s['status'] not in ('operator_paused','attention_required','pilot_complete','allocation_complete'):raise RuntimeError('Learner resumed; diagnostics stop')
    if sha_file(cp['path'])!=cp['sha256']:raise RuntimeError('Diagnostic checkpoint changed')
    backend=manifest['native_backend'];build=Path(backend['build_dir'])
    if sha_file(build/backend['provenance']['module_file'])!=backend['provenance']['module_sha256']:raise RuntimeError('Pinned binary changed')
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),STS_LIGHTSPEED_BUILD=str(build),PYTHONDONTWRITEBYTECODE='1')
    record={'condition':condition,'panel':panel,'command':command,'started':time.time(),'status':'running','checkpoint_sha256':cp['sha256'],'backend':backend}
    process=None
    try:
        with metrics.with_suffix('.log').open('xb') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(pid=process.pid,pgid=process.pid,process_birth=process_birth(process.pid),owned_members={})
            record['group_creation']={'new_session':True,'leader_pid':process.pid,'leader_birth':record['process_birth'],
                'started_at':record['started'],'command_sha256':digest(command),'max_lifetime_seconds':140}
            atomic_json(record_path,record);deadline=time.monotonic()+140
            while process.poll() is None:
                track_owned_members(record)
                if time.monotonic()>deadline:
                    record['stop_reason']='diagnostic_timeout';stop_owned_group(record);break
                time.sleep(.25)
            process.wait(timeout=10);record['exit_code']=process.returncode
    finally:
        if process:stop_owned_group(record)
        record['finished']=time.time();record['elapsed_seconds']=record['finished']-record['started']
    if record.get('exit_code')!=0 or not metrics.is_file():
        record['status']='failed';atomic_json(record_path,record);raise RuntimeError('Diagnostic condition failed: '+condition)
    m=json.loads(metrics.read_text());quality=panel_quality(m)
    if Counter(quality['cohort'])!=Counter(manifest['panels'][str(panel)]['task_ids']):raise RuntimeError('Cohort mismatch')
    from sts_env.provenance import interface_identity
    if m['training_environment'].get('native_module_sha256')!=backend['provenance']['module_sha256'] or m['training_environment'].get('python_interface_sha256')!=interface_identity()['python_interface_sha256']:raise RuntimeError('Output native/interface mismatch')
    for relative,expected in manifest['sources'].items():
        if sha_file(ROOT/relative)!=expected:raise RuntimeError('Pinned source changed during diagnostics')
    eps=[e for r in m['rounds'] for e in r['episodes']]
    record.update(status='complete',metrics_path=str(metrics),metrics_sha256=sha_file(metrics),quality=quality,
        act1_reached=sum(e.get('max_act_reached',e.get('act',0))>=2 for e in eps),
        act2_reached=sum(e.get('max_act_reached',e.get('act',0))>=3 for e in eps),
        defeats=sum(e['outcome']=='defeat' for e in eps),floor_mean=quality['floor_mean'])
    atomic_json(record_path,record);return record


def paired_crossover_statistics(episodes, fault_ids):
    conditions=('OO','NN','ON','NO')
    common=set.intersection(*(set(episodes[c]) for c in conditions))
    common={identity for identity in common if all(identity not in fault_ids.get(c,set()) and
        not episodes[c][identity].get('fault') and not episodes[c][identity].get('truncated') and
        episodes[c][identity].get('termination_reason')!='collector_budget' and
        isinstance(episodes[c][identity].get('floor'),(int,float)) and
        not isinstance(episodes[c][identity]['floor'],bool) and math.isfinite(episodes[c][identity]['floor']) for c in conditions)}
    if not common:return {'common_valid':0,'classifiable':False,'reason':'no common healthy uncensored worlds'}
    common=sorted(common)
    def passed(entry,metric):
        if metric=='victory':return entry.get('outcome')=='victory'
        return entry.get('max_act_reached',entry.get('act',0)) >= (2 if metric=='act1_clear' else 3)
    floor={c:sum(episodes[c][i]['floor'] for i in common)/len(common) for c in conditions}
    contrasts={}
    for newer,older,label in (('NO','OO','combat_given_original_ooc'),('NN','ON','combat_given_new_ooc'),
                              ('ON','OO','ooc_given_original_combat'),('NN','NO','ooc_given_new_combat')):
        outcomes={}
        for metric in ('victory','act1_clear','act2_clear'):
            gains=[i for i in common if passed(episodes[newer][i],metric) and not passed(episodes[older][i],metric)]
            losses=[i for i in common if passed(episodes[older][i],metric) and not passed(episodes[newer][i],metric)]
            outcomes[metric]={'gains':len(gains),'losses':len(losses),'net':len(gains)-len(losses),'gain_task_ids':gains,'loss_task_ids':losses}
        contrasts[label]={'mean_floor_delta':floor[newer]-floor[older],'outcomes':outcomes}
    return {'common_valid':len(common),'classifiable':True,'common_task_ids':common,'mean_floor':floor,
            'common_outcomes':{c:{metric:sum(passed(episodes[c][i],metric) for i in common) for metric in ('victory','act1_clear','act2_clear')} for c in conditions},
            'contrasts':contrasts,'interaction_floor':floor['NN']-floor['NO']-floor['ON']+floor['OO']}


def summarize(args,manifest,records):
    summaries={};episodes={};fault_ids={}
    for condition in manifest['conditions']:
        selected=[r for r in records if r['condition']==condition]
        eps={e['task_id']:e for r in selected for roll in json.loads(Path(r['metrics_path']).read_text())['rounds'] for e in roll['episodes']}
        faults={f['task_id'] for r in selected for roll in json.loads(Path(r['metrics_path']).read_text())['rounds'] for f in roll['faults']}
        episodes[condition]=eps;fault_ids[condition]=faults;summaries[condition]={'requested':sum(r['quality']['requested'] for r in selected),
            'victories':sum(r['quality']['clears'] for r in selected),'defeats':sum(r['defeats'] for r in selected),
            'faults':len(faults),'truncated':sum(r['quality']['truncated'] for r in selected),
            'complete':all(r['quality']['complete'] for r in selected),'mean_floor':sum(e['floor'] for e in eps.values())/len(eps),
            'act1_clears':sum(r['act1_reached'] for r in selected),'act2_clears':sum(r['act2_reached'] for r in selected),
            'seconds':sum(r['elapsed_seconds'] for r in selected)}
    paired=paired_crossover_statistics(episodes,fault_ids)
    result={'diagnostic_only':True,'source_checkpoint_identities':{k:manifest[k] for k in ('original','regressed')},
            'summaries':summaries,'paired':paired,'records':records,
            'uncertainty':'Small fixed TRAIN cohort; descriptive paired effects only. Composite head interactions matter; no population significance claim.'}
    atomic_json(args.output/'results.json',result);print(json.dumps({'summaries':summaries,'paired':paired},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True);parser.add_argument('--regressed',type=Path,required=True)
    parser.add_argument('--journal',type=Path,default=ROOT/'data/long_rl_v2/campaign/journal.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/long_rl_v2/diagnosis/phase_crossover')
    parser.add_argument('--world-start',type=int,default=500000000)
    parser.add_argument('--expand-fast',action='store_true',help='Prespecified runtime-only rule: add remaining32 if every first condition completes in<=45s')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:print('Plan only; --execute runs CPU diagnostics.');return
    torch.set_num_threads(1);manifest=prepare(args);records=[]
    for c in ('OO','NN','ON','NO'):records.append(evaluate(args,manifest,c,0))
    if args.expand_fast and all(r['elapsed_seconds']<=45 and r['quality']['complete'] for r in records):
        for c in ('OO','NN','ON','NO'):records.append(evaluate(args,manifest,c,1))
    summarize(args,manifest,records)


if __name__=='__main__':main()
