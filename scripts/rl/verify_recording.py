#!/usr/bin/env python3
"""Bounded native checks: recording overhead, exact resume, and real fault replay."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import gzip
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main():
    import torch
    from sts_policy.curriculum import TaskSpec, world_split
    from sts_policy.model_agent import collate_public, forward_policy
    from sts_policy.losses import PPOConfig
    from sts_policy.rollout_training import collect_rollouts
    from sts_policy.replay_fault import replay_fault
    from sts_policy.workers import CurriculumWorker
    from sts_rl.common import WORKER_CONFIG, FilteringWorker, load_policy
    from sts_rl.ooc_trainer import OOCTrainConfig, train
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--historical-eval',required=True)
    parser.add_argument('--timing-only',action='store_true')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    model,_=load_policy(args.checkpoint)
    model.eval()
    seeds=[s for s in range(5_000_000_000,5_000_000_100) if world_split(s)=='train'][:16]
    tasks=[TaskSpec(seed=s,kind='run',max_act=1,horizon=2048,seek_combat=False) for s in seeds]
    timings={False:[],True:[]}; traces=[]
    # ABBA order limits thermal/order confounding; each timed pass has warm workers.
    for enabled in [False,True,True,False]:
        cfg={**WORKER_CONFIG,'fault_dir':str(out/'timing_faults') if enabled else None}
        workers=[FilteringWorker(CurriculumWorker(cfg),dead_actions=True,slim=True) for _ in range(4)]
        try:
            for worker in workers: worker.submit('reset',tasks[0]).result()
            started=time.perf_counter()
            result=collect_rollouts(model,workers,tasks,collate=collate_public,forward=forward_policy,
                                    config=PPOConfig(),deterministic=True,max_seconds=60,
                                    deferred_bootstrap=True,reuse_encodings=False)
            timings[enabled].append(time.perf_counter()-started)
            traces.append({e['world_seed']:e['action_trace_sha256'] for e in result['episodes']})
            assert not result['faults'] and result['cohort_completed']
        finally:
            for worker in workers: worker.close()
    assert all(t==traces[0] for t in traces)
    timing={'logging_off_seconds':timings[False], 'logging_on_seconds':timings[True],
            'median_overhead_fraction':statistics.median(timings[True])/statistics.median(timings[False])-1,
            'all_four_control_traces_match':True,'timing_games_per_pass':len(tasks)}
    (out/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    if args.timing_only:
        print(json.dumps(timing));return
    baseline=out/'empty_baseline.json';baseline.write_text('{"records": []}')
    cfg=OOCTrainConfig(out=str(out/'continuous'),checkpoint=args.checkpoint,iterations=2,games=2,
                       epochs=1,minibatch=512,micro_batch=64,collectors=2,workers=2,threads=1,
                       update_threads=1,health_every=0,eval_every=0,baseline=str(baseline),seed=17,gae_lambda=1.)
    train(cfg)
    split=replace(cfg,out=str(out/'split'),iterations=1)
    train(split)
    train(replace(split,iterations=2),resume=True)
    a=torch.load(out/'continuous/latest.pt',weights_only=False)
    b=torch.load(out/'split/latest.pt',weights_only=False)
    assert a['steps_total']==b['steps_total']
    assert all(torch.equal(v,b['model_state'][k]) for k,v in a['model_state'].items())
    for iteration in [1,2]:
        with gzip.open(out/'continuous'/f'collection_{iteration:04d}.json.gz','rt') as stream: x=json.load(stream)
        with gzip.open(out/'split'/f'collection_{iteration:04d}.json.gz','rt') as stream: y=json.load(stream)
        assert [e['action_trace_sha256'] for e in x['episodes']]==[e['action_trace_sha256'] for e in y['episodes']]
    # Missing historical valid worlds identify candidates, not known root causes.
    historical=json.loads(Path(args.historical_eval).read_text())
    valid={r['seed'] for r in historical['records']}
    start=historical['summary']['world_start']
    requested=[];seed=start
    while len(requested)<4096:
        if world_split(seed)=='train': requested.append(seed)
        seed+=1
    candidates=[s for s in requested if s not in valid][:8]
    fault_result=None
    worker=FilteringWorker(CurriculumWorker({**WORKER_CONFIG,'fault_dir':str(out/'real_faults')}),dead_actions=True,slim=True)
    try:
        candidate_tasks=[TaskSpec(seed=s,kind='run',max_act=3,horizon=4096,seek_combat=False,battle_simulations=200) for s in candidates]
        result=collect_rollouts(model,[worker],candidate_tasks,collate=collate_public,forward=forward_policy,
                               config=PPOConfig(),deterministic=True,max_seconds=60,deferred_bootstrap=True,reuse_encodings=False)
        paths=sorted((out/'real_faults').glob('fault-*.json'))
        if paths:
            record=json.loads(paths[0].read_text())
            fault_result={'record':str(paths[0]),'exception':record['exception'],
                          'seed':record['task']['seed'], 'rpc_count':len(record['successful_rpcs']),
                          'replays':[replay_fault(record,timeout=10,seconds=30,fault_dir=str(out/'replayed_faults')) for _ in range(2)]}
    finally: worker.close()
    metrics={'schema':'sts_recording_verification_v1','logging_off_seconds':timings[False],
             'logging_on_seconds':timings[True],
             'median_overhead_fraction':statistics.median(timings[True])/statistics.median(timings[False])-1,
             'timing_games_per_pass':len(tasks),'all_four_control_traces_match':True,
             'fresh_process_resume_weights_and_trajectories_exact':True,
             'collection_file_bytes':(out/'continuous/collection_0002.json.gz').stat().st_size,
             'collection_games':cfg.games,'resume_checkpoint_bytes':(out/'continuous/latest.pt').stat().st_size,
             'real_fault':fault_result}
    (out/'verification.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics))


if __name__=='__main__': main()
