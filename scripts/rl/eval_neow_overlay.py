#!/usr/bin/env python3
"""Measure the saturated boss-relic-swap habit with a public menu restriction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main():
    import torch
    from sts_rl.common import _MappedFuture,make_workers,close_workers,load_policy
    from sts_rl.evaluate import evaluate,paired,paired_wins
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--n',type=int,default=1024)
    parser.add_argument('--world-start',type=int,default=5800000000)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():parser.error('Output already exists')
    record=begin_run(str(out)+'.record',vars(args),args.checkpoint,inputs=[args.baseline])
    torch.set_num_threads(1);model,_=load_policy(args.checkpoint);model.eval()

    class Worker:
        def __init__(self,inner):self.worker=inner;self.restricted=0
        def map(self,result):
            obs=result[0]
            if obs.get('phase')!='combat' and obs.get('state',{}).get('floor')==0:
                legal=obs['legal_actions'];kept=[a for a in legal if not ('boss relic' in a['description'].lower() and 'starter relic' in a['description'].lower())]
                if kept and len(kept)!=len(legal):
                    self.restricted+=1;obs={**obs,'legal_actions':kept};result=(obs,*result[1:])
            return result
        def submit(self,method,*args):
            future=self.worker.submit(method,*args)
            return _MappedFuture(future,self.map) if method in {'reset','step'} else future
        def close(self):self.worker.close()

    workers=[Worker(w) for w in make_workers(args.workers,dead_action_filter=True,slim=True)]
    try:result=evaluate(model,workers,n=args.n,world_start=args.world_start,chunk=64)
    finally:close_workers(workers)
    baseline=json.loads(Path(args.baseline).read_text())
    result.update(run_record=record,overlay='avoid_neow_boss_swap_v1',restricted_menus=sum(w.restricted for w in workers),
                  paired=paired(baseline['records'],result['records']),paired_wins=paired_wins(baseline['records'],result['records']))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result)+'\n')
    print(json.dumps({k:result[k] for k in ['summary','restricted_menus','paired','paired_wins']}))


if __name__=='__main__':main()
