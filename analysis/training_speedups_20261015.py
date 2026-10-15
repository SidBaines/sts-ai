#!/usr/bin/env python3
"""Verify and summarize the bounded B/C/A speedup experiment artifacts."""
import argparse
import ast
from datetime import datetime
import gzip
import hashlib
import json
import re
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
import zipfile

CORE = ['src/sts_policy/model.py', 'src/sts_policy/model_agent.py',
        'src/sts_policy/rollout_training.py', 'src/sts_rl/joint_sparse.py', 'src/sts_rl/packed_rows.py']
NAMES = ['speed_bench_train_base', 'speed_bench_eval_lean', 'speed_tensor_eval_base',
         'speed_tensor_train_c2w6', 'speed_tensor_train_c4w8', 'speed_tensor_train_c6w12',
         'speed_tensor_train_c8w16', 'speed_tensor_train_c4w12', 'speed_tensor_train_c6w18',
         'speed_tensor_train_c8w24', 'speed_tensor_eval_c6w6', 'speed_tensor_eval_c8w8',
         'speed_tensor_numeric_base', 'speed_tensor_numeric_compact',
         'speed_tensor_full_train_base', 'speed_tensor_full_train_c2w6',
         'speed_tensor_full_train_c8w24', 'speed_tensor_full_eval_base', 'speed_tensor_full_eval_c8w8']


def identity(p):
    p=Path(p)
    return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def semantic_hash(source):
    return hashlib.sha256(ast.dump(ast.parse(source),include_attributes=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='data/rl_runs')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists(): parser.error('Fresh output required')
    runs=Path(args.runs)
    reports={name:json.loads((runs/name/'report.json').read_text()) for name in NAMES}
    summary={}
    native=set()
    parents=set()
    warm=set()
    runtime=set()
    for name,report in reports.items():
        record=json.loads(Path(report['record']['path']).read_text())
        native.add(record['native']['module_sha256']);parents.add(record['parent_checkpoint']['sha256'])
        warm.add(record['data_inputs'][0]['sha256'])
        runtime.add(json.dumps(record['runtime'],sort_keys=True))
        samples=report['samples']
        assert all(s['faults']==0 for s in samples),name
        assert all(s['fingerprints']==samples[0]['fingerprints'] for s in samples),name
        source={}
        with zipfile.ZipFile(record['source_archive']['path']) as z:
            for rel in CORE:
                if rel in z.namelist():source[rel]=semantic_hash(z.read(rel))
        # Exclude the earlier array-only prototype used for the initial lean
        # eval timing. Every final tensor-path core must match current source.
        if name.startswith('speed_tensor_') and 'baseline' not in report['arguments']['source_root']:
            assert source=={rel:semantic_hash(Path(rel).read_bytes()) for rel in CORE},name
        totals={}
        for receipt in (runs/name).glob('progress_receipt_*.json.gz'):
            episodes=json.load(gzip.open(receipt,'rt'))['episodes']
            for e in episodes:totals[e['world_seed']]=e['steps']
        assert len(totals)==report['arguments']['games'],name
        seconds=[s['seconds'] for s in samples]
        summary[name]=dict(arguments=report['arguments'],report=identity(runs/name/'report.json'),
            source_archive=record['source_archive'],core_semantic_hashes=source,
            repeat_count=len(samples),median_seconds=statistics.median(seconds),
            seconds_range=[min(seconds),max(seconds)],total_steps=sum(totals.values()),
            choices=samples[0]['rows'],wins=samples[0]['wins'],faults=0,
            decisions_per_second=sum(totals.values())/statistics.median(seconds),
            owner_peak_rss_bytes=report['owner_peak_rss_bytes'],
            children_peak_rss_bytes=report['children_peak_rss_bytes'],
            fingerprints_sha256=hashlib.sha256(json.dumps(samples[0]['fingerprints'],sort_keys=True).encode()).hexdigest())
        for field in ['persist_seconds','persist_bytes','likelihood_seconds','update_seconds']:
            if field in samples[0]:summary[name]['median_'+field]=statistics.median(s[field] for s in samples)
    assert len(native)==len(parents)==len(runtime)==len(warm)==1
    same=lambda a,b:reports[a]['samples'][0]['fingerprints']==reports[b]['samples'][0]['fingerprints']
    checks={
        'train_format_exact_256':same('speed_bench_train_base','speed_tensor_train_c2w6'),
        'train_format_exact_64':same('speed_tensor_numeric_base','speed_tensor_numeric_compact'),
        'train_format_exact_768':same('speed_tensor_full_train_base','speed_tensor_full_train_c2w6'),
        'greedy_eval_exact_all_256_layouts':all(same('speed_tensor_eval_base',n) for n in ['speed_bench_eval_lean','speed_tensor_eval_c6w6','speed_tensor_eval_c8w8']),
        'greedy_eval_exact_1024':same('speed_tensor_full_eval_base','speed_tensor_full_eval_c8w8')}
    keys=['actor_sha256','critic_sha256','optimizer_hash','critic_optimizer_hash']
    for old,new in [('speed_tensor_numeric_base','speed_tensor_numeric_compact'),('speed_tensor_full_train_base','speed_tensor_full_train_c2w6')]:
        checks['optimizer_exact_'+str(reports[old]['arguments']['games'])]=all(
            reports[old]['samples'][0][k]==reports[new]['samples'][0][k] for k in keys)
    for suffix in ('base','compact'):
        acceptance=json.loads((runs/('speed_tensor_replay_'+suffix)/'acceptance.json').read_text())
        checks['fresh_process_replay_'+suffix]=acceptance['fresh_process_replay_exact']
    assert all(checks.values()),checks
    controllers={}
    for name in ['speed_controller_base','speed_controller_fast']:
        d=runs/name
        rows=[json.loads(line) for line in (d/'log.jsonl').read_text().splitlines()]
        start=next(r for r in rows if r['event']=='start');done=next(r for r in rows if r['event']=='done')
        updates=[r for r in rows if r['event']=='update']
        collections=[r for r in rows if r['event']=='collection']
        assert len(updates)==2 and all(u['accepted'] for u in updates),name
        assert len(collections)==2 and all(c['summary']['faults']==c['summary']['censored']==c['summary']['invalid']==0 for c in collections),name
        wall=re.findall(r'^real\s+(\d+(?:\.\d+)?)$',(d/'stdout.log').read_text(),re.M)
        assert wall,name
        user=re.findall(r'^user\s+(\d+(?:\.\d+)?)$',(d/'stdout.log').read_text(),re.M)
        system=re.findall(r'^sys\s+(\d+(?:\.\d+)?)$',(d/'stdout.log').read_text(),re.M)
        assert user and system,name
        controllers[name]=dict(seconds=float(wall[-1]),reported_cpu_seconds=float(user[-1])+float(system[-1]),event_seconds=done['t']-start['t'],done=done,
            updates=len(updates),actor_steps=updates[-1]['actor_steps'],
            training_worlds=sum(c['summary']['requested'] for c in collections),
            training_decisions=sum(c['summary']['steps_mean']*c['summary']['valid'] for c in collections),
            owner_peak_rss_gib=max(u['rss_gib'] for u in updates),
            log=identity(d/'log.jsonl'),config=json.loads((d/'config.json').read_text()))
    assert controllers['speed_controller_base']['training_worlds']==controllers['speed_controller_fast']['training_worlds']==512
    controllers['speedup']=controllers['speed_controller_base']['seconds']/controllers['speed_controller_fast']['seconds']
    validation=json.loads((runs/'speed_tensor_validation/acceptance.json').read_text())
    assert all(r['returncode']==0 for r in validation['results'])
    evidence=dict(schema='joint_training_speedup_evidence_v1',created=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        baseline_commit='e9fc6a9',native_sha256=next(iter(native)),parent_sha256=next(iter(parents)),warm_critic_sha256=next(iter(warm)),
        runtime=json.loads(next(iter(runtime))),measurements=summary,correctness=checks,
        controllers=controllers,validation=identity(runs/'speed_tensor_validation/acceptance.json'),
        limitations=['Training collector layouts change seeded stochastic trajectories and batched floating-point execution. Compare decision throughput as well as games per second.',
                     'Optimizer format parity uses stable world ordering through a Sequence adapter so list/packed access costs remain in the measured update.',
                     'RSS is owner-process high-water RSS, not aggregate physical memory across the process tree.',
                     'The controller fixture uses two 256-game rounds (expandable to 512), 128-world eval, 128-world anchors and 64-world critic validation in separate TRAIN-only timing namespaces.',
                     'The controller already omits rolling critic-health history from checkpoints. Full uninterrupted-versus-resumed guard decisions are not certified.',
                     'Interactive-host timing ranges are descriptive repeated timings, not statistical confidence intervals.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(correctness=checks,controller_speedup=controllers['speedup']),indent=2))


if __name__=='__main__':main()
