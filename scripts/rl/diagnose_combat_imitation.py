#!/usr/bin/env python3
"""Held-world vote-distribution versus actual teacher plurality diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_rl.common import load_policy
    from sts_rl.distill import load_records,soft_target
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--records',required=True)
    parser.add_argument('--out',required=True)
    a=parser.parse_args();out=Path(a.out)
    if out.exists():parser.error('Fresh diagnostic directory required')
    record=begin_run(out,vars(a),a.checkpoint,inputs=[a.baseline,a.records])
    torch.set_num_threads(1)
    student,payload=load_policy(a.checkpoint);reference,_=load_policy(a.baseline)
    student.eval();reference.eval();cfg=payload['metadata']['config']
    assert cfg['records']==a.records and cfg['checkpoint']==a.baseline
    rows=load_records(a.records);seeds=sorted({r['seed'] for r in rows})
    random.Random(cfg['seed']).shuffle(seeds)
    held=set(seeds[:max(1,int(len(seeds)*cfg['val_fraction']))])
    labelled=[r for r in rows if r['type']=='combat' and r['seed'] in held]
    results=[]
    with torch.no_grad():
        for start in range(0,len(labelled),256):
            batch=labelled[start:start+256];inputs=collate_public([r['obs'] for r in batch])
            base,_,_=forward_policy(reference,inputs);new,_,_=forward_policy(student,inputs)
            for i,row in enumerate(batch):
                legal=row['obs']['legal_actions'];positions={int(x['index']):j for j,x in enumerate(legal)}
                target=torch.tensor(soft_target(row,cfg['tau']))
                votes=row['votes'];counts=list(votes.values());top=max(counts)
                actual=positions[int(max(votes.items(),key=lambda item:item[1])[0])]
                state=row['obs']['state'];group=f"{str(state['room']).rsplit('.',1)[-1]}{state['act']}"
                result={'seed':row['seed'],'group':group,'vote_max_fraction':top/sum(counts),
                        'plurality_tied':counts.count(top)>1,'soft_target_argmax_differs_from_actual_teacher':int(target.argmax())!=actual}
                for name,logits in [('base',base),('new',new)]:
                    lp=torch.log_softmax(logits[i,:len(legal)],0)
                    result[name+'_vote_ce']=float(-(target*lp).sum())
                    result[name+'_actual_plurality_ce']=float(-lp[actual])
                    result[name+'_actual_plurality_agree']=int(logits[i,:len(legal)].argmax())==actual
                    result[name+'_soft_target_argmax_agree']=int(logits[i,:len(legal)].argmax())==int(target.argmax())
                results.append(result)
    groups={}
    for group in ['all']+sorted({r['group'] for r in results}):
        chosen=results if group=='all' else [r for r in results if r['group']==group]
        groups[group]={'n':len(chosen)}
        for key in chosen[0]:
            if key not in ['seed','group']:
                groups[group][key]=sum(r[key] for r in chosen)/len(chosen)
    result={'schema':'sts_teacher_plurality_diagnostic_v1','run_record':record,'held_worlds':len(held),
            'held_labels':len(labelled),'groups':groups,
            'scope':'Fixed held-world records. Batched diagnostic logits only; proxy scores do not establish deployment performance. Actual teacher tie rule is first maximal vote entry, as in the saved teacher player.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
