#!/usr/bin/env python3
"""Merge complete shards only after exact cohort, inference, source and native gates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))

def main():
    from sts_rl.common import train_tasks
    from sts_rl.evaluate import finish_evaluation,paired,paired_wins
    from sts_rl.recording import file_identity
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--n',type=int,required=True);p.add_argument('--world-start',type=int,required=True)
    p.add_argument('--kind',choices=['greedy','canonical_search'],required=True);p.add_argument('--baseline')
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Fresh merged output required')
    expected={t.seed for t in train_tasks(a.world_start,a.n)};records={};conditions=[];sources=[];elapsed=0.
    ignored={'n','world_start','workers','collectors','threads','output','baseline','progress_output','seconds'}
    for name in a.sources:
        path=Path(name);d=json.loads(path.read_text());s=d['summary']
        if s.get('faults') or s.get('unstarted') or s.get('cohort_completed') is False:
            raise ValueError('Faulted/incomplete shards require separate recovery, not this merge')
        launches=list(Path(str(path)+'.record/records').glob('*/launch.json'))
        if len(launches)!=1:raise ValueError('Ambiguous source launch identity')
        launch=json.loads(launches[0].read_text());cfg=launch['effective_config']
        if len(d['records'])!=cfg['n'] or len({r['seed'] for r in d['records']})!=cfg['n']:
            raise ValueError('Shard does not contain its requested number of unique worlds')
        if {r['seed'] for r in d['records']}!={t.seed for t in train_tasks(cfg['world_start'],cfg['n'])}:
            raise ValueError('Shard requested cohort mismatch')
        if any(r['steps']>=4096 for r in d['records']):raise ValueError('Potential horizon censoring needs explicit review')
        if a.kind=='greedy':
            if cfg.get('mode')!='greedy' or cfg['workers']!=cfg['collectors']:
                raise ValueError('Prospective greedy shards must use one worker per collector')
            script='scripts/rl/eval_policy.py'
        else:
            if s.get('search_protocol')!='canonical_rng_resampled_native_v2':
                raise ValueError('Wrong search protocol')
            script='scripts/rl/eval_search_boss.py'
        source_hashes={}
        for f in launch['source_files']:
            rel=str(Path(f['path']).relative_to(launch['cwd']))
            if rel.startswith('src/') or rel==script:source_hashes[rel]=f['sha256']
        if script not in source_hashes:raise ValueError('Missing evaluator source hash')
        condition={'config':{k:v for k,v in cfg.items() if k not in ignored},
                   'parent_sha256':launch['parent_checkpoint']['sha256'],
                   'native_sha256':launch['native']['module_sha256'],'runtime':launch['runtime'],
                   'source_hashes':source_hashes,'kind':a.kind}
        conditions.append(condition);elapsed+=s['elapsed']
        for r in d['records']:
            if r['seed'] in records:raise ValueError('Duplicate world across shards')
            records[r['seed']]=r
        sources.append({'evaluation':file_identity(path),'launch':file_identity(launches[0])})
    if any(c!=conditions[0] for c in conditions):raise ValueError('Shard scientific/source conditions differ')
    if set(records)!=expected:raise ValueError('Merged cohort has missing or extra worlds')
    mode='greedy_single_observation' if a.kind=='greedy' else 'greedy+canonical_rng_resampled_native_search'
    result=finish_evaluation([records[s] for s in sorted(expected)],0,elapsed,mode=mode,world_start=a.world_start)
    result.update(schema='sts_exact_evaluation_merge_v2',requested=a.n,cohort_completed=True,condition=conditions[0],source_identities=sources)
    if a.baseline:
        b=json.loads(Path(a.baseline).read_text())
        if {r['seed'] for r in b['records']}!=expected or b['summary']['faults']:
            raise ValueError('Baseline cohort/fault gate differs')
        if b['condition']['parent_sha256']!=conditions[0]['parent_sha256'] or b['condition']['native_sha256']!=conditions[0]['native_sha256']:
            raise ValueError('Baseline model/native identity differs')
        result.update(baseline_identity=file_identity(a.baseline),paired=paired(b['records'],result['records']),paired_wins=paired_wins(b['records'],result['records']))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['summary','paired','paired_wins'] if k in result}))

if __name__=='__main__':main()
