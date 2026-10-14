#!/usr/bin/env python3
"""Bounded start/state replay and independent frozen-policy outcome verification."""
import argparse,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))

def rows(directory,pattern):
    return [json.loads(line) for path in sorted(directory.glob(pattern)) for line in gzip.open(path,'rt') if line.strip()]

def main():
    import numpy as np
    import torch
    from sts_policy.curriculum import TaskSpec,public_hash
    from sts_policy.encoding import encode_observation
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_policy.workers import CurriculumWorker
    from sts_rl.common import WORKER_CONFIG,load_policy
    from sts_rl.recording import begin_run
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--all-starts',action='store_true');a=p.parse_args();directory=Path(a.bank);out=Path(a.out)
    if (out/'acceptance.json').exists():p.error('Fresh acceptance output required')
    record=begin_run(out,vars(a),a.checkpoint,inputs=[directory]);torch.set_num_threads(1);model,_=load_policy(a.checkpoint);model.eval()
    bank=rows(directory,'bank_*.jsonl.gz');states=rows(directory,'states_*.jsonl.gz');worlds=rows(directory,'worlds_*.jsonl.gz')
    summary=json.loads((directory/'summary.json').read_text())
    if summary['faults'] or summary['censored'] or summary['unstarted']:raise ValueError('Require a complete clean bank for this acceptance')
    assert len(bank)==summary['fights'] and len(states)==summary['states'] and len(worlds)==summary['completed']
    actions={r['seed']:r['actions'] for r in worlds};byfight={(r['seed'],r['floor']):r for r in bank}
    for s in states:
        fight=byfight[s['seed'],s['floor']];assert s['fight_win']==fight['won'];assert s['fight_start_public_sha256']==fight['task']['expected_public_hash']
    selected={}
    for r in bank:selected.setdefault((r['room'],r['act']),r)
    starts=bank if a.all_starts else list(selected.values());checks=[]
    worker=CurriculumWorker(WORKER_CONFIG,timeout=60.)
    try:
        for r in starts:
            spec=TaskSpec.from_dict(r['task']);obs,_=worker.reset(spec)
            assert public_hash(obs)==spec.expected_public_hash
            done=cut=False
            while not (done or cut):
                with torch.no_grad():logits,_,_=forward_policy(model,collate_public([obs]))
                pos=int(logits[0,:len(obs['legal_actions'])].argmax())
                obs,_,done,cut,info=worker.step(obs['legal_actions'][pos]['index'])
            assert not cut and (info['task_outcome']=='battle_victory')==r['won']
            checks.append({'seed':r['seed'],'floor':r['floor'],'start_hash_matches':True,'replayed_outcome_matches':True})
        sampled=[s for s in states if s['turn']>0][:12]
        for s in sampled:
            parent=TaskSpec.from_dict(byfight[s['seed'],s['floor']]['task'])
            spec=TaskSpec(seed=s['seed'],kind='battle',prefix=tuple(actions[s['seed']][:s['world_prefix_length']]),seek_combat=False,max_act=3,horizon=4096,battle_simulations=parent.battle_simulations,
                          expected_public_hash=s['full_public_sha256'],expected_provenance=parent.expected_provenance)
            obs,_=worker.reset(spec);full=encode_observation(obs);stored=encode_observation(s['obs'])
            assert public_hash(obs)==s['full_public_sha256']
            assert np.array_equal(full.state_features,stored.state_features) and np.array_equal(full.action_features,stored.action_features)
    finally:worker.close()
    result={'schema':'sts_value_bank_replay_acceptance_v1','run_record':record,'bank_fights':len(bank),'states':len(states),
            'replayed_starts_and_outcomes':checks,'midfight_state_replays':len(sampled),'slimmed_legacy_encodings_exact':True,
            'scope':'All starts when explicitly requested; otherwise one per room/act. At most twelve midfight states. Remaining rows are not independently replay-certified.'}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print('Value bank replay/outcome/encoding checks pass',len(checks),len(sampled),flush=True)

if __name__=='__main__':main()
