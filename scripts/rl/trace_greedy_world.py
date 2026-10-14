#!/usr/bin/env python3
"""Bounded single-observation greedy trace for cross-condition parity diagnosis."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))

def main():
    import torch
    from sts_policy.curriculum import public_hash
    from sts_policy.encoding import encode_observation
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_rl.common import load_policy,make_workers,train_tasks
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();os.chdir(ROOT);out=Path(args.output)
    if out.exists():parser.error('Fresh output required')
    record=begin_run(str(out)+'.record',vars(args),args.checkpoint)
    torch.set_num_threads(1);model,_=load_policy(args.checkpoint);model.eval()
    task=train_tasks(args.seed,1)[0]
    if task.seed!=args.seed:raise ValueError('Diagnostic seed must be TRAIN')
    worker=make_workers(1,dead_action_filter=True,slim=False)[0];rows=[]
    try:
        obs,info=worker.submit('reset',task).result();done=cut=False
        while not (done or cut):
            encoded=encode_observation(obs)
            with torch.no_grad():logits,_,_=forward_policy(model,collate_public([obs]))
            scores=logits[0,:len(obs['legal_actions'])];pos=int(scores.argmax());choice=obs['legal_actions'][pos]
            rows.append({'public_sha256':public_hash(obs),'encoding_sha256':hashlib.sha256(encoded.state_features.tobytes()+encoded.action_features.tobytes()).hexdigest(),
                         'action_index':choice['index'],'description':choice['description'],'chosen_logit':float(scores[pos]),
                         'top2_gap':float(scores.topk(2).values.diff().abs()[0]) if len(scores)>1 else None})
            obs,_,done,cut,info=worker.submit('step',choice['index']).result()
            if len(rows)>4096:raise RuntimeError('Diagnostic decision bound exceeded')
    finally:worker.close()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'run_record':record,'seed':args.seed,'records':rows,'final_public_sha256':public_hash(obs),
                              'final_floor':obs['state']['floor'],'outcome':info.get('task_outcome'),'truncated':cut})+'\n')
    print('Single-row trace complete',args.seed,len(rows),obs['state']['floor'],flush=True)

if __name__=='__main__':main()
