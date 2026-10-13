"""Canonical control traces across explicit observation/native revisions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch

from sts_policy.entity_model import load_policy
from sts_policy.composition import training_environment_record
from sts_policy.strategic_collection import collect
from sts_policy.training import file_identity
from sts_policy.workers import CurriculumWorker
from scripts.run_strategic_v3 import atomic_json,matched_tasks,worker_settings,summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=192)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('fresh output required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    model,_=load_policy(args.checkpoint)
    tasks=matched_tasks(bank=None,kind='run',split='dev',episodes=args.episodes,world_start=330000000,
                        seed=0,horizon=4096,max_act=3)
    traces={t.seed:hashlib.sha256() for t in tasks}
    with ExitStack() as stack:
        journals={t.seed:stack.enter_context((args.output/f'{t.seed}.jsonl').open('w')) for t in tasks}
        workers=[stack.enter_context(CurriculumWorker(worker_settings(),timeout=10.)) for _ in range(8)]
        def observer(observations,seeds,steps,choices):
            for obs,seed,step,index in zip(observations,seeds,steps,choices):
                action=next(a for a in obs['legal_actions'] if a['index']==index)
                line=json.dumps({'step':step,'phase':obs['phase'],'index':index,'description':action['description']},
                                sort_keys=True,separators=(',',':'))+'\n'
                journals[seed].write(line);traces[seed].update(line.encode())
        result=collect(model,workers,tasks,deterministic=True,retain_transitions=False,
                       max_seconds=300,decision_observer=observer)
    result.pop('transitions');result['summary']=summarize(result)
    atomic_json(args.output/'result.json',{'checkpoint_sha256':file_identity(args.checkpoint),
        'environment':training_environment_record(),'tasks':[t.to_dict() for t in tasks],'result':result,
        'canonical_traces':{str(seed):digest.hexdigest() for seed,digest in traces.items()},
        'canonical_note':'Phase, index and description preserve executed control; new public metadata intentionally changes full action JSON hashes.'})
    print(json.dumps(result['summary']))


if __name__=='__main__':
    main()
