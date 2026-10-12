#!/usr/bin/env python3
"""Authorized v2: continuous full-game learner, separate DEV champion, finite active budget."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time
from sts_policy.long_run_control import atomic_json, process_birth, sha_file
from sts_policy.long_run_v2 import V2Config, V2Controller, V2_STAGE

ROOT = Path(__file__).resolve().parents[1]


def sanitized_main_command(argv):
    excluded = {'--session-seconds','--session-kind','--extend-hours','--acknowledge-attention'}
    result=[]; i=0
    while i<len(argv):
        token=argv[i]; flag=token.split('=',1)[0]
        if flag in excluded:
            i += 1 if '=' in token or flag=='--acknowledge-attention' else 2
        else: result.append(token); i+=1
    return result + ['--session-kind','main']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/long_rl_v2/campaign')
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/long_curriculum_v2_protocol.json')
    parser.add_argument('--backend-build',type=Path)
    parser.add_argument('--pool-bank',type=Path)
    parser.add_argument('--pool-panels',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--active-seconds',type=float,default=18424.127912292024)
    parser.add_argument('--budget-transfer',type=Path)
    parser.add_argument('--session-seconds',type=float)
    parser.add_argument('--session-kind',choices=('pilot','main'),default='main')
    parser.add_argument('--action',choices=('run','status','pause'),default='run')
    parser.add_argument('--extend-hours',type=float,default=0.)
    parser.add_argument('--acknowledge-attention',action='store_true')
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--nice',type=int,default=10)
    parser.add_argument('--chunk-seconds',type=float,default=180.)
    parser.add_argument('--evaluation-seconds',type=float,default=300.)
    parser.add_argument('--capture-continuations',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--prevent-sleep',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    journal=args.output/'journal.json'; old=json.loads(journal.read_text()) if journal.is_file() else None
    if args.action in ('status','pause'):
        if old is None: parser.error('No existing v2 journal')
        if args.action=='status':
            print(json.dumps({k:old.get(k) for k in ('status','controller','allocation','learner','champion','learning_health')},indent=2));return 0
        owner=old.get('controller',{})
        if owner.get('pid') and owner.get('process_birth') and process_birth(owner['pid'])==owner['process_birth']:
            os.kill(owner['pid'],signal.SIGTERM);print(json.dumps({'pause_requested':True,'pid':owner['pid']}))
        else: print(json.dumps({'paused':True,'reason':'Recorded controller absent; no process signaled'}))
        return 0
    if not all((args.backend_build,args.pool_bank,args.pool_panels)):parser.error('Run needs backend build and TRAIN continuation pool paths')
    if args.active_seconds<=0 or args.extend_hours<0 or (args.session_seconds is not None and args.session_seconds<=0):parser.error('Invalid active/session/extension budget')
    if args.extend_hours and old is None:parser.error('Explicit extension requires an existing v2 allocation')
    protocol=json.loads(args.protocol.read_text());checkpoint=(args.checkpoint or Path(protocol['checkpoint']['path'])).resolve()
    active=args.active_seconds
    if args.budget_transfer is None and protocol.get('budget_transfer'):args.budget_transfer=Path(protocol['budget_transfer']['path'])
    if old:
        persisted=old['identity']['config'];active=persisted['initial_active_seconds']
        explicit=argv if argv is not None else sys.argv[1:]
        if any(str(token).split('=',1)[0]=='--active-seconds' for token in explicit) and args.active_seconds!=active:
            parser.error('Resume cannot reset allocation; use explicit --extend-hours')
        if args.budget_transfer is None and persisted.get('budget_transfer_path'):args.budget_transfer=Path(persisted['budget_transfer_path'])
    config=V2Config(checkpoint=str(checkpoint),backend_build=str(args.backend_build.resolve()),
        pool_bank=str(args.pool_bank.resolve()),pool_panels=str(args.pool_panels.resolve()),protocol=str(args.protocol.resolve()),
        deadline=time.time()+active,duration_hours=active/3600,initial_active_seconds=active,train_world_start=100000000,dev_world_start=400000,
        device=args.device,workers=args.workers,threads=args.threads,nice=args.nice,chunk_seconds=args.chunk_seconds,
        evaluation_seconds=args.evaluation_seconds,capture_continuations=args.capture_continuations,prevent_sleep=args.prevent_sleep,
        budget_transfer_path=str(args.budget_transfer.resolve()) if args.budget_transfer else None,
        budget_transfer_sha256=sha_file(args.budget_transfer) if args.budget_transfer else None)
    if not args.execute:
        print(json.dumps({'mode':'plan_only','config':asdict(config),'stage':asdict(V2_STAGE),
                          'learner_parent':'continuous; no ordinary development rollback','champion':'confirmed full3 DEV rank',
                          'allocation_not_started':old is None},indent=2));return 0
    controller=V2Controller(ROOT,args.output,config)
    if args.extend_hours:controller.extend_hours(args.extend_hours)
    command=[sys.executable,str(Path(__file__).resolve()),*(argv if argv is not None else sys.argv[1:])]
    atomic_json(args.output/'main_command.json',{'argv':sanitized_main_command(command),'cwd':str(ROOT),'environment':{'PYTHONPATH':str(ROOT/'src')}})
    def interrupt(signum,frame):
        controller.pause_requested=True
        raise KeyboardInterrupt('Operator requested v2 pause')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        controller.execute(session_seconds=args.session_seconds,session_kind=args.session_kind,acknowledge_attention=args.acknowledge_attention)
    except KeyboardInterrupt:
        controller.safe_owned_cleanup();controller.set_accounting(False);controller.state['status']='operator_paused';controller.save()
    except Exception as exc:
        controller.state['status']='attention_required';controller.state['events'].append({'at':time.time(),'type':'failure','error':type(exc).__name__,'message':str(exc)});controller.save();raise
    finally:controller.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
