#!/usr/bin/env python3
"""Independent resumable long RL controller; no child runs without --execute.

A 15-minute pilot invocation can share a persisted 10-hour active allocation with a
later main invocation: --session-seconds900 --session-kindpilot, then resume the
same output without a session cap. Pauses and offline gaps do not consume it.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

from sts_policy.long_run_control import LongRunConfig, LongRunController, STAGES

ROOT=Path(__file__).resolve().parents[1]


def utc_deadline(value: str) -> float:
    instant=datetime.fromisoformat(value.replace('Z','+00:00'))
    if instant.tzinfo is None or instant.utcoffset().total_seconds()!=0:
        raise argparse.ArgumentTypeError('deadline must include UTC Z or +00:00')
    return instant.timestamp()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/long_rl_v1/campaign')
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/long_curriculum_protocol.json')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--backend-build',type=Path)
    parser.add_argument('--pool-bank',type=Path)
    parser.add_argument('--pool-panels',type=Path)
    allocation=parser.add_mutually_exclusive_group()
    allocation.add_argument('--duration-hours',type=float)
    allocation.add_argument('--deadline-utc',type=utc_deadline)
    parser.add_argument('--session-seconds',type=float)
    parser.add_argument('--session-kind',choices=('pilot','main'),default='main')
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--nice',type=int,default=10)
    parser.add_argument('--episodes',type=int,default=128)
    parser.add_argument('--dev-episodes',type=int,default=48)
    parser.add_argument('--chunk-seconds',type=float,default=180.)
    parser.add_argument('--evaluation-seconds',type=float,default=240.)
    parser.add_argument('--dev-interval',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--gae-lambda',type=float,default=.95)
    parser.add_argument('--gpu-duty-cycle',type=float,default=1.,help='whole-job active/cooling fraction; not a GPU-utilization cap')
    parser.add_argument('--battery-pause-at',type=float,default=25.)
    parser.add_argument('--min-free-gib',type=float,default=10.)
    parser.add_argument('--floor-regression',type=float,default=1.)
    parser.add_argument('--retention-clear-drop',type=int,default=2)
    parser.add_argument('--capture-continuations',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--prevent-sleep',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--action',choices=('run','status','pause'),default='run')
    parser.add_argument('--extend-hours',type=float,default=0.,help='explicit addition to final mixed-stage active budget on resume')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.duration_hours is not None and args.duration_hours<=0:parser.error('duration must be positive')
    if args.session_seconds is not None and args.session_seconds<=0:parser.error('session must be positive')
    journal=args.output/'journal.json'
    old=json.loads(journal.read_text()) if journal.is_file() else None
    if args.action in ('status','pause'):
        if old is None:parser.error('no existing controller journal')
        if args.action=='status':
            print(json.dumps({key:old.get(key) for key in ('status','controller','allocation','current_checkpoint','best_checkpoint','last_checkpoint')},indent=2,sort_keys=True));return
        import os
        from sts_policy.long_run_control import process_birth
        owner=old.get('controller',{})
        if not owner.get('pid') or not owner.get('process_birth') or process_birth(owner['pid'])!=owner['process_birth']:
            print(json.dumps({'paused':True,'reason':'recorded controller already absent; no process signaled'}));return
        os.kill(owner['pid'],signal.SIGTERM)
        print(json.dumps({'pause_requested':True,'pid':owner['pid'],'resume':'run main_command.json argv; active remaining budget is preserved'}));return
    if not all((args.backend_build,args.pool_bank,args.pool_panels)):parser.error('run requires --backend-build/--pool-bank/--pool-panels')
    if args.extend_hours<0:parser.error('extension must be nonnegative')
    protocol=json.loads(args.protocol.read_text())
    checkpoint=(args.checkpoint or Path(protocol['checkpoint']['path'])).resolve()
    if old:
        persisted=old['identity']['config']
        duration=persisted['duration_hours']
        if args.deadline_utc is None:args.deadline_utc=persisted.get('hard_wall_deadline')
        if args.duration_hours is not None and args.duration_hours!=duration:parser.error('resume cannot reset the initial duration; use --extend-hours')
        remaining=old['allocation']['authorized_seconds']-old['allocation']['used_seconds']+args.extend_hours*3600
        deadline=time.time()+max(0.,remaining)
    else:
        if args.extend_hours:parser.error('extension requires an existing run')
        duration=args.duration_hours if args.duration_hours is not None else (None if args.deadline_utc is not None else 10.)
        deadline=args.deadline_utc if args.deadline_utc is not None else time.time()+duration*3600
    config=LongRunConfig(checkpoint=str(checkpoint),backend_build=str(args.backend_build.resolve()),
        pool_bank=str(args.pool_bank.resolve()),pool_panels=str(args.pool_panels.resolve()),deadline=deadline,duration_hours=duration,
        protocol=str(args.protocol.resolve()),device=args.device,workers=args.workers,threads=args.threads,nice=args.nice,
        episodes=args.episodes,dev_episodes=args.dev_episodes,chunk_seconds=args.chunk_seconds,evaluation_seconds=args.evaluation_seconds,
        dev_interval=args.dev_interval,batch_size=args.batch_size,learning_rate=args.learning_rate,gae_lambda=args.gae_lambda,
        hard_wall_deadline=args.deadline_utc, duty_cycle=args.gpu_duty_cycle,battery_pause_at=args.battery_pause_at,min_free_gib=args.min_free_gib,
        floor_regression=args.floor_regression,retention_clear_drop=args.retention_clear_drop,
        capture_continuations=args.capture_continuations,prevent_sleep=args.prevent_sleep)
    if not args.execute:
        print(json.dumps({'mode':'plan_only','config':asdict(config),'stages':[asdict(stage) for stage in STAGES],
            'session_kind':args.session_kind,'session_seconds':args.session_seconds,'allocation_start_not_committed':old is None},indent=2,sort_keys=True))
        return
    controller=LongRunController(ROOT,args.output,config)
    if args.extend_hours:controller.extend_hours(args.extend_hours)
    import sys
    from sts_policy.long_run_control import atomic_json
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]
    excluded={'--session-seconds','--session-kind','--extend-hours'}
    sanitized=[];index=0
    while index<len(command):
        flag=command[index].split('=',1)[0]
        if flag in excluded:
            index+=1 if '=' in command[index] else 2
        else:sanitized.append(command[index]);index+=1
    command=sanitized+['--session-kind','main']
    atomic_json(args.output/'main_command.json',{'argv':command,'cwd':str(ROOT),'environment':{'PYTHONPATH':str(ROOT/'src')}})
    def interrupt(signum,frame):raise KeyboardInterrupt('controller stop signal')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        controller.execute(session_seconds=args.session_seconds,session_kind=args.session_kind)
    except KeyboardInterrupt:
        controller.set_accounting(False);controller.state['status']='operator_paused';controller.save()
    except Exception as error:
        controller.state['status']='attention_required';controller.state['events'].append({'at':time.time(),'type':'failure','error':type(error).__name__,'message':str(error)})
        controller.save();raise
    finally:
        controller.close()


if __name__=='__main__':main()
