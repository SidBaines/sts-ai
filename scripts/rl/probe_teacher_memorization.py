#!/usr/bin/env python3
"""Bounded training-pipeline sanity: fit 32 fixed TRAIN states, never deploy."""
import argparse
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import torch
    from sts_policy.curriculum import public_hash
    from sts_policy.encoding import encode_observation
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_rl.common import JsonlLog,load_policy,save_policy
    from sts_rl.distill import load_records
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--records',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    if out.exists():parser.error('Fresh diagnostic directory required')
    cfg={**vars(args),'seed':17,'states':32,'steps':128,'batch':32,'lr':.001,'weight_decay':.01,
         'head':'combat','threads':1,'rehearsal':False,'diagnostic_only':True}
    record=begin_run(out,cfg,args.checkpoint,inputs=[args.records])
    torch.set_num_threads(1);torch.manual_seed(17);rng=random.Random(17)
    rows=[r for r in load_records(args.records) if r['type']=='combat'];rng.shuffle(rows)
    picked=[];worlds=set();features=set()
    for row in rows:
        if row['seed'] in worlds:continue
        encoded=encode_observation(row['obs'])
        key=(encoded.state_features.tobytes(),encoded.action_features.tobytes())
        if key in features:continue
        positions={int(a['index']):i for i,a in enumerate(row['obs']['legal_actions'])}
        target=positions[row['label_index']]
        if any(i<target and np.array_equal(encoded.action_features[i],encoded.action_features[target]) for i in range(target)):
            continue
        picked.append(row);worlds.add(row['seed']);features.add(key)
        if len(picked)==32:break
    assert len(picked)==32
    selection=[{'seed':r['seed'],'floor':r['obs']['state']['floor'],'public_sha256':public_hash(r['obs']),
                'label_index':r['label_index']} for r in picked]
    (out/'selected_states.json').write_text(json.dumps(selection,indent=2)+'\n')
    model,template=load_policy(args.checkpoint)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('combat.'))
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith('ooc.')}
    inputs=collate_public([r['obs'] for r in picked])
    targets=torch.tensor([next(i for i,a in enumerate(r['obs']['legal_actions']) if a['index']==r['label_index']) for r in picked])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    log=JsonlLog(out/'log.jsonl');log.emit({'event':'start','config':cfg,'run_record':record})
    def measure(step):
        model.eval()
        with torch.no_grad():
            logits,_,_=forward_policy(model,inputs)
            result={'step':step,'accuracy':float((logits.argmax(-1)==targets).float().mean()),
                    'ce':float(torch.nn.functional.cross_entropy(logits,targets))}
        log.emit({'event':'measure',**result});return result
    initial=measure(0)
    for step in range(1,129):
        model.train();logits,_,_=forward_policy(model,inputs)
        loss=torch.nn.functional.cross_entropy(logits,targets)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%16==0:final=measure(step)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in frozen.items())
    checkpoint=out/'diagnostic_only.pt'
    save_policy(checkpoint,model,template,metadata={'stage':'sts_teacher_memorization_probe','config':cfg,
                'run_record':record,'diagnostic_only':True,'scope':'Fits the same32 stored training states. No held-out/game performance, no candidate promotion or deployment.'},optimizer=optimizer,step=128,rng=rng)
    report={'schema':'sts_teacher_memorization_probe_v1','run_record':record,'initial':initial,'final':final,
            'selected_states':selection,'ooc_bit_identical':True,'checkpoint':str(checkpoint),
            'scope':'Sanity check only: no rehearsal, no held-out worlds, no live-game evaluation. Memorization success cannot establish generalization or game competence.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    log.emit({'event':'done',**final});print(json.dumps({'initial':initial,'final':final}),flush=True)


if __name__=='__main__':main()
