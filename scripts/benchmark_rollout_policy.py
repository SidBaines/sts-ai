#!/usr/bin/env python3
"""Compare actual small-batch CPU/MPS collection using one frozen checkpoint."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import torch
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.losses import PPOConfig
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import collate_public, forward_policy
from sts_policy.rollout_training import collect_rollouts
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    payload=load_checkpoint(args.checkpoint)
    model=CandidateActorCritic(**payload['model_config'])
    model.load_state_dict(payload['model_state'])
    tasks=[replace(TaskSpec.from_dict(json.loads(line)),horizon=256) for line in args.tasks.read_text().splitlines()]
    tasks=[task for task in tasks if world_split(task.seed)=='dev']
    random.Random(777).shuffle(tasks)
    tasks=tasks[:48]
    workers=[CurriculumWorker({'fast_observations':True,'fast_replay':True},timeout=10) for _ in range(8)]
    rows=[]
    try:
        for device in ('cpu','mps','mps','cpu'):
            model.to(device)
            result=collect_rollouts(model,workers,tasks,collate=collate_public,forward=forward_policy,
                                    device=device,config=PPOConfig(),max_seconds=60,deterministic=True)
            transitions=result.pop('transitions');result.pop('teacher_records')
            result.update(device=device,transitions=len(transitions),wins=sum(x['outcome']=='battle_victory' for x in result['episodes']))
            rows.append(result)
            print(json.dumps({k:result[k] for k in ('device','transitions','wins','elapsed_seconds','timings')}),flush=True)
    finally:
        for worker in workers:worker.close()
    metrics={'checkpoint':str(args.checkpoint),'model_config':payload['model_config'],'rows':rows,
             'selection_note':'ABBA order; actor+bootstrap timings include public encoding, device transfers and distribution checks. No updates or final worlds.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
