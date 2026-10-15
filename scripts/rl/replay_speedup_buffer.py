#!/usr/bin/env python3
"""Replay a benchmark's exact saved rows in a fresh process, without games."""
import argparse
import gzip
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    import torch
    from benchmark_joint_collection import CanonicalRows, hash_tensors
    from sts_rl.common import load_policy
    from sts_rl.joint_sparse import (JointConfig, WinCritics, actor_update, critic_update,
                                    freeze_legacy_critics, identity_state)
    from sts_rl.recording import begin_run, file_identity
    source=Path(a.benchmark)
    report=json.loads((source/'report.json').read_text())
    args=report['arguments']
    expected=report['samples'][-1]
    if 'actor_sha256' not in expected: raise ValueError('Benchmark must include an optimizer step')
    buffer=source/f"buffer_{expected['repeat']}.pt.gz"
    out=Path(a.out)
    if (out/'acceptance.json').exists(): raise ValueError('Fresh output required')
    record=begin_run(out,vars(a),args['checkpoint'],inputs=[str(buffer),args['critic_init']])
    torch.set_num_threads(2)
    torch.manual_seed(19)
    actor,_=load_policy(args['checkpoint'])
    reference,_=load_policy(args['checkpoint'])
    freeze_legacy_critics(actor)
    critics=WinCritics()
    critics.load_state_dict(torch.load(args['critic_init'],weights_only=False)['critic_state'])
    with gzip.open(buffer,'rb') as f: rows=torch.load(f,weights_only=False)['rows']
    cfg=JointConfig(out=str(out),checkpoint=args['checkpoint'],critic_minibatch=256,critic_init=args['critic_init'])
    ordered=CanonicalRows(rows)
    anchors=[]
    for phase in ('combat','ooc'):
        phase_rows=[r for r in ordered if r['phase']==phase]
        anchors.extend(phase_rows[::max(1,len(phase_rows)//128)][:128])
    opts={phase:torch.optim.AdamW([v for v in getattr(actor,phase).parameters() if v.requires_grad],
                                  lr=getattr(cfg,phase+'_lr'),weight_decay=0) for phase in ('combat','ooc')}
    co=torch.optim.AdamW(critics.parameters(),lr=cfg.critic_lr,weight_decay=.01)
    update=actor_update(actor,reference,opts,ordered,anchors,cfg,random.Random(19),.1)
    critic_update(critics,co,ordered,cfg)
    got=dict(actor_sha256=identity_state(actor),critic_sha256=identity_state(critics),
             optimizer_hash=hash_tensors({p:o.state_dict() for p,o in opts.items()}),
             critic_optimizer_hash=hash_tensors(co.state_dict()))
    assert update['accepted'],update
    for key,value in got.items():
        if value!=expected[key]: raise AssertionError((key,value,expected[key]))
    (out/'acceptance.json').write_text(json.dumps(dict(record=record,buffer=file_identity(buffer),
        fresh_process_replay_exact=True,rows=len(rows),**got),indent=2)+'\n')
    print(json.dumps(dict(fresh_process_replay_exact=True,rows=len(rows),**got)),flush=True)


if __name__=='__main__': main()
