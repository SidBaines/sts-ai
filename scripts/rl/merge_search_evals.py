#!/usr/bin/env python3
"""Merge completed search shards against an exact declared cohort, retaining extras."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main():
    from sts_rl.common import train_tasks
    from sts_rl.evaluate import finish_evaluation,paired,paired_wins
    from sts_rl.recording import file_identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--world-start',type=int,required=True)
    parser.add_argument('--n',type=int,required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():parser.error('Merged output already exists')
    expected={t.seed for t in train_tasks(args.world_start,args.n)}
    records={};conditions=[];identities=[];elapsed=0.
    for name in args.sources:
        path=Path(name);data=json.loads(path.read_text());summary=data['summary']
        if not summary.get('cohort_completed') or summary.get('faults') or summary.get('unstarted'):
            raise ValueError('This exact complete merge requires complete fault-free shards; preserve/investigate incomplete ones separately')
        launch_paths=list(Path(str(path)+'.record/records').glob('*/launch.json'))
        if len(launch_paths)!=1:raise ValueError('Missing/ambiguous launch identity')
        launch=json.loads(launch_paths[0].read_text());cfg=launch['effective_config']
        condition={k:cfg.get(k) for k in ['checkpoint','samples','simulations','rooms','min_act']}
        condition['checkpoint_sha256']=launch['parent_checkpoint']['sha256']
        condition['native_sha256']=launch['native']['module_sha256']
        conditions.append(condition)
        elapsed+=summary['elapsed']
        for row in data['records']:
            if row['seed'] in records:raise ValueError('Duplicate world across shards')
            records[row['seed']]=row
        identities.append({'evaluation':file_identity(path),'launch':file_identity(launch_paths[0])})
    if any(c!=conditions[0] for c in conditions):raise ValueError('Shard conditions differ')
    missing=expected-set(records)
    if missing:raise ValueError(f'Missing {len(missing)} requested worlds; do not report an exact cohort')
    retained=[records[s] for s in sorted(expected)];extras=[records[s] for s in sorted(set(records)-expected)]
    baseline=json.loads(Path(args.baseline).read_text())
    if {r['seed'] for r in baseline['records']}!=expected:raise ValueError('Baseline cohort differs')
    if baseline['summary'].get('faults'):raise ValueError('Baseline contains faults')
    result=finish_evaluation(retained,0,elapsed,mode=conditions[0].get('mode','greedy+public_search[BOSS]'),world_start=args.world_start)
    result.update(schema='sts_exact_search_merge_v1',source_identities=identities,condition=conditions[0],
                  extra_world_records=extras,requested=args.n,cohort_completed=True,
                  paired=paired(baseline['records'],retained),paired_wins=paired_wins(baseline['records'],retained))
    result['summary'].update(checkpoint=conditions[0]['checkpoint'],samples=conditions[0]['samples'],
                             simulations=conditions[0]['simulations'],extra_worlds_excluded=len(extras))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':result['summary'],'paired':result['paired'],'paired_wins':result['paired_wins']}))


if __name__=='__main__':main()
