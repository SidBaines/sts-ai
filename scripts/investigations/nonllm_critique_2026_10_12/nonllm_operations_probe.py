#!/usr/bin/env python3
"""Read immutable campaign artifacts and profile a bounded CPU actor fixture.

No training, simulator trajectory, evaluator, GPU call, or source/artifact edit.
Run from long-curriculum-v2 with PYTHONPATH=src and PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations
import argparse
import collections
import cProfile
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('/private/tmp/nonllm_operations_probe.json'))
    args = ap.parse_args()
    roots = {name:Path('/Users/sidbaines/.codex/worktrees')/tree/'SlayTheSpireAI'
             for name,tree in [('first','nonllm-curriculum'),('v1','long-curriculum'),('v2','long-curriculum-v2')]}
    campaign=roots['v2']/'data/long_rl_v2/campaign'
    journal=json.loads((campaign/'journal.json').read_text())
    result={'scope':'read-only immutable artifacts; CPU fixture only; no environment/learning/GPU',
            'journal_status':journal['status'],'allocation':journal['allocation']}
    counts=collections.Counter();timings=collections.Counter();wall=0.;elapsed=0.
    panels=collections.defaultdict(list);families=collections.defaultdict(collections.Counter)
    metric_values=collections.defaultdict(list)
    for job in journal['jobs']:
        if job.get('status')!='complete': continue
        if job.get('kind')=='eval':
            panels[job.get('panel_id')].append(job['finished']-job['started']);continue
        if job.get('kind')!='train': continue
        counts['chunks']+=1
        metric=json.loads(Path(job['metrics']).read_text())
        taskfile=Path(job['command'][job['command'].index('--tasks')+1])
        manifest=json.loads(Path(str(taskfile)+'.manifest.json').read_text())
        metadata={m['task_id']:m for m in manifest['tasks']}
        wall+=job['finished']-job['started'];elapsed+=metric['elapsed_seconds']
        for row in metric['rounds']:
            counts.update({'transitions':row['transitions'],'excluded_transitions':row['excluded_transitions'],
                           'requested':row['requested_tasks'],'faults':len(row['faults']),
                           'unstarted':row['unstarted_tasks']})
            timings.update(row['timings'])
            update=row.get('update',{});timings['update_seconds']+=update.get('elapsed_seconds',0)
            counts['updates']+=update.get('steps',0);counts['rejected_minibatches']+=update.get('rejected_minibatches',0)
            for key in ('approx_kl','value_loss','entropy','policy_loss','clip_fraction'):
                if key in update: metric_values[key].append(update[key])
            for episode in row['episodes']:
                m=metadata[episode['task_id']];key=f"{m['start_source']}:act{m['start_act']}"
                families[key].update({'valid_episodes':1,'transitions':episode['steps'],
                                      'victories':int(episode.get('outcome')=='victory')})
    result.update(counts=dict(counts),training_wall_seconds=wall,cli_elapsed_seconds=elapsed,
                  outside_cli_elapsed_seconds=wall-elapsed,train_transitions_per_wall_second=counts['transitions']/wall,
                  timings=dict(timings),family_counts={k:dict(v)for k,v in families.items()},
                  evaluation_wall_seconds=dict(panels),
                  ppo_metrics={k:{'mean':statistics.mean(v),'min':min(v),'max':max(v)}for k,v in metric_values.items()})
    storage=collections.defaultdict(lambda:[0,0])
    for path in campaign.rglob('*'):
        if path.is_file():
            row=storage[''.join(path.suffixes)];row[0]+=1;row[1]+=path.stat().st_size
    result['storage_by_suffix']=dict(storage)
    result['deletion_bytes']=sum(d.get('bytes',0)for d in journal['deletions'])
    confirmations=[]
    for event in journal['events']:
        if event.get('type')!='champion_confirmation':continue
        sha=event['checkpoint_sha256']
        job=next(j for j in journal['jobs']if j.get('kind')=='train'and j.get('checkpoint_sha256')==sha)
        confirmations.append({'round':job['round'],'sha256':sha,'exists':Path(job['checkpoint']).is_file(),
                              'deletions':[d for d in journal['deletions']if d.get('sha256')==sha]})
    result['confirmed_checkpoint_retention']=confirmations
    result['journal_serialization']={}
    for name in ('v1','v2'):
        path=roots[name]/f'data/long_rl_{name}/campaign/journal.json'
        doc=json.loads(path.read_text());samples=[]
        for _ in range(5):
            begun=time.perf_counter();json.dumps(doc,sort_keys=True,indent=2);samples.append(time.perf_counter()-begun)
        result['journal_serialization'][name]={'bytes':path.stat().st_size,'median_seconds':statistics.median(samples)}
    live=roots['v2']/'docs/long_curriculum_v2_live.md'
    result['generated_live']={'bytes':live.stat().st_size,'lines':len(live.read_text().splitlines()),
                              'status':live.read_text().splitlines()[2]}
    cold=[]
    for _ in range(3):
        begun=time.perf_counter()
        subprocess.run([str(roots['v2']/'.venv/bin/python'),'-c',
                        'import torch; from sts_policy.model_agent import policy_from_payload; from sts_policy.workers import CurriculumWorker'],
                       cwd=roots['v2'],env={**os.environ,'PYTHONPATH':'src','PYTHONDONTWRITEBYTECODE':'1'},
                       check=True,capture_output=True)
        cold.append(time.perf_counter()-begun)
    result['cold_import_seconds']=cold
    begun=time.perf_counter();import torch
    from sts_policy.model_agent import policy_from_payload,collate_public,forward_policy,cached_public_collator
    from sts_policy.training import load_checkpoint
    from sts_policy.curriculum import world_split
    result['parent_import_seconds']=time.perf_counter()-begun
    torch.set_num_threads(1)
    checkpoint=roots['first']/'data/nonllm_native_revision/campaign/joint_act3_0030.pt'
    begun=time.perf_counter();model=policy_from_payload(load_checkpoint(checkpoint)).eval()
    result['model_load_seconds']=time.perf_counter()-begun
    teacher=roots['first']/'data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl'
    rows=[];phases=collections.Counter()
    with teacher.open() as stream:
        for line in stream:
            row=json.loads(line)
            if world_split(row['world_seed'])!='train':continue
            phase=row['observation']['phase']
            if phases[phase]>=256:continue
            rows.append(row['observation']);phases[phase]+=1
            if len(rows)==512:break
    cache=cached_public_collator(rows);fixtures={}
    for name,collate in [('fresh',collate_public),('cached',cache)]:
        samples=[]
        with torch.no_grad():
            for _ in range(5):
                begun=time.perf_counter()
                for offset in range(0,len(rows),8):forward_policy(model,collate(rows[offset:offset+8],'cpu'))
                samples.append(time.perf_counter()-begun)
        fixtures[name]={'median_seconds':statistics.median(samples),'trials':samples}
    profile=cProfile.Profile()
    with torch.no_grad():
        profile.enable()
        for offset in range(0,len(rows),8):forward_policy(model,collate_public(rows[offset:offset+8],'cpu'))
        profile.disable()
    stats=pstats.Stats(profile)
    result['actor_fixture']={'source':str(teacher),'checkpoint':str(checkpoint),'rows':len(rows),
        'phase_counts':dict(phases),'batch_size':8,'threads':1,'single_choice_states':sum(len(o['legal_actions'])==1 for o in rows),
        'timings':fixtures,'limitation':'balanced saved Act1 TRAIN fixture; cached-path difference is a ceiling, not an achieved rollout speedup',
        'profile':[{'function':str(k),'cumulative_seconds':v[3],'calls':v[1]}
                   for k,v in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:25]]}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(args.output),'chunks':counts['chunks'],'transitions':counts['transitions'],
                      'train_wall_tps':result['train_transitions_per_wall_second']}))

if __name__=='__main__':main()
