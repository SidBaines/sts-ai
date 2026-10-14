#!/usr/bin/env python3
"""Replay five pre-floor50 BOSS cases; public encounter/event evidence only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))


def main():
 import torch
 from sts_policy.curriculum import TaskSpec,public_hash
 from sts_policy.model_agent import collate_public,forward_policy
 from sts_rl.common import load_policy,make_workers
 from sts_rl.recording import begin_run
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
 a=p.parse_args();out=Path(a.out)
 if out.exists():p.error('Fresh diagnostic output required')
 seeds=[5800000240,5800000265,5800000578,5800000964,5800000988]
 record=begin_run(out,{**vars(a),'seeds':seeds,'purpose':'Public replay evidence; no policy training or hidden inputs.'},a.checkpoint)
 torch.set_num_threads(1);model,_=load_policy(a.checkpoint);model.eval();worker=make_workers(1,timeout=10.,dead_action_filter=True,slim=False)[0];results=[]
 try:
  for seed in seeds:
   obs,info=worker.submit('reset',TaskSpec(seed=seed,kind='run',seek_combat=False,max_act=3,horizon=4096,battle_simulations=200)).result();done=cut=False;events=[];boss=None;steps=0
   while not(done or cut):
    state=obs['state'];room=str(state.get('room','')).rsplit('.',1)[-1]
    if obs['phase']!='combat' and 'portal' in obs['text'].lower():events.append({'floor':state['floor'],'public_text':obs['text'],'public_sha256':public_hash(obs)})
    if boss is None and obs['phase']=='combat' and room=='BOSS' and state.get('act')==3:
     boss={'floor':state['floor'],'enemies':[e.get('name') for e in state['combat']['enemies']],'public_text':obs['text'],'public_sha256':public_hash(obs)}
    with torch.no_grad():logits,_,_=forward_policy(model,collate_public([obs]))
    action=obs['legal_actions'][int(logits[0,:len(obs['legal_actions'])].argmax())]['index'];obs,_,done,cut,info=worker.submit('step',action).result();steps+=1
   assert not cut and boss is not None
   results.append({'seed':seed,'portal_events':events,'boss':boss,'final_floor':obs['state']['floor'],'outcome':info['task_outcome'],'steps':steps})
 finally:worker.close()
 report={'schema':'sts_portal_boss_public_replay_v1','run_record':record,'worlds':results,'scope':'Full public observations retained at portal event and boss entry; unchanged curated model. No training, private policy input or native modification.'};(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps([{'seed':r['seed'],'portal_seen':bool(r['portal_events']),'boss_floor':r['boss']['floor'],'enemies':r['boss']['enemies'],'outcome':r['outcome']} for r in results]),flush=True)

if __name__=='__main__':main()
