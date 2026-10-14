#!/usr/bin/env python3
"""Measure exact candidate-feature aliases on actually visited teacher states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import torch
    from sts_policy.encoding import encode_observation
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_rl.common import load_policy
    from sts_rl.distill import load_records
    from sts_rl.recording import begin_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--records',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    if out.exists():parser.error('Fresh diagnostic output required')
    record=begin_run(out,vars(args),args.checkpoint,inputs=[args.records])
    torch.set_num_threads(1);model,_=load_policy(args.checkpoint);model.eval()
    results=[];examples=[];score_cases=[]
    for row in load_records(args.records):
        if row['type']!='combat':continue
        obs=row['obs'];encoded=encode_observation(obs)
        assert np.isfinite(encoded.state_features).all() and np.isfinite(encoded.action_features).all()
        legal=obs['legal_actions'];position={int(a['index']):i for i,a in enumerate(legal)}
        chosen=position[int(max(row['votes'].items(),key=lambda pair:pair[1])[0])]
        classes=defaultdict(list)
        for i,features in enumerate(encoded.action_features):classes[features.tobytes()].append(i)
        aliases=[indices for indices in classes.values() if len(indices)>1]
        teacher_class=classes[encoded.action_features[chosen].tobytes()]
        state=obs['state'];group=f"{str(state['room']).rsplit('.',1)[-1]}{state['act']}"
        result={'seed':row['seed'],'floor':state['floor'],'group':group,
                'any_candidate_alias':bool(aliases),'teacher_candidate_aliased':len(teacher_class)>1,
                'teacher_not_first_in_alias':chosen!=teacher_class[0]}
        results.append(result)
        if aliases:score_cases.append((obs,aliases))
        if result['teacher_not_first_in_alias'] and len(examples)<12:
            examples.append({'seed':row['seed'],'floor':state['floor'],'group':group,
                             'teacher_index':legal[chosen]['index'],
                             'identical_feature_actions':[legal[i] for i in teacher_class]})
    maximum_spread=0.
    with torch.no_grad():
        for start in range(0,len(score_cases),128):
            cases=score_cases[start:start+128]
            logits,_,_=forward_policy(model,collate_public([obs for obs,_ in cases]))
            for i,(_,aliases) in enumerate(cases):
                for indices in aliases:
                    scores=logits[i,indices]
                    maximum_spread=max(maximum_spread,float(scores.max()-scores.min()))
    groups={}
    for group in ['all']+sorted({r['group'] for r in results}):
        rows=results if group=='all' else [r for r in results if r['group']==group]
        groups[group]={'n':len(rows)}
        for key in ['any_candidate_alias','teacher_candidate_aliased','teacher_not_first_in_alias']:
            groups[group][key]=sum(r[key] for r in rows)
    report={'schema':'sts_teacher_candidate_alias_audit_v1','run_record':record,'groups':groups,
            'maximum_observed_parent_logit_spread_within_exact_alias':maximum_spread,'examples':examples,
            'scope':'Actually reached teacher states only. Exact float32 candidate features; shared candidate scorer has no menu-position input. Many duplicate choices may be semantically equivalent, so aliases alone do not prove game harm. No state representation change or policy update.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['groups','maximum_observed_parent_logit_spread_within_exact_alias']}),flush=True)


if __name__=='__main__':main()
