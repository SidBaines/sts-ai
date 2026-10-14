#!/usr/bin/env python3
"""Replayable single-row policy fights and sparse public states with completed outcomes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
PROVENANCE_KEYS=('python_interface_sha256','runtime_python_version','environment_version','observation_version','native_provenance')

def main():
    import torch
    from sts_env.env import SimulatorError
    from sts_policy.curriculum import TaskSpec,public_hash
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_rl.common import JsonlLog,load_policy,make_workers,slim_text,train_tasks
    from sts_rl.recording import begin_run
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=512);p.add_argument('--world-start',type=int,default=44000000000)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seconds',type=float,default=1800.)
    a=p.parse_args();out=Path(a.out)
    if out.exists():p.error('Fresh output directory required')
    if min(a.n,a.threads,a.seconds)<=0:p.error('Positive bounds required')
    record=begin_run(out,vars(a),a.checkpoint);torch.set_num_threads(1);model,_=load_policy(a.checkpoint);model.eval()
    tasks=train_tasks(a.world_start,a.n);lock=threading.Lock();start=time.monotonic();log=JsonlLog(out/'log.jsonl')
    log.emit({'event':'start','config':vars(a),'run_record':record})
    def collect(slot):
        worker=make_workers(1,dead_action_filter=True,slim=False)[0];worlds=[];counts=Counter();labels=0
        with gzip.open(out/f'bank_{slot:02d}.jsonl.gz','wt',compresslevel=1) as bank,gzip.open(out/f'states_{slot:02d}.jsonl.gz','wt',compresslevel=1) as states,gzip.open(out/f'worlds_{slot:02d}.jsonl.gz','wt',compresslevel=1) as receipts:
            try:
                for task in tasks[slot::a.threads]:
                    if time.monotonic()-start>=a.seconds:break
                    actions=[];fights=[];current=None;done=cut=False
                    try:
                        obs,info=worker.submit('reset',task).result()
                        provenance={key:info[key] for key in PROVENANCE_KEYS}
                        def finish(won):
                            nonlocal current
                            if current is not None:
                                current['won']=won;fights.append(current);current=None
                        while not (done or cut):
                            if obs['phase']=='combat':
                                s=obs['state'];c=s['combat']
                                if current is None:
                                    spec=TaskSpec(seed=task.seed,kind='battle',prefix=tuple(actions),seek_combat=False,max_act=3,horizon=4096,battle_simulations=task.battle_simulations,
                                                  expected_public_hash=public_hash(obs),expected_provenance=provenance)
                                    current={'task':spec.to_dict(),'seed':task.seed,'floor':s['floor'],'act':s['act'],'room':str(s['room']).rsplit('.',1)[-1],
                                             'hp':s['cur_hp'],'max_hp':s['max_hp'],'samples':[],'sampled_turns':set()}
                                turn=int(c['turn'])
                                if (not current['samples'] or turn%2==0) and turn not in current['sampled_turns'] and len(current['samples'])<8:
                                    current['sampled_turns'].add(turn)
                                    stored={**obs,'text':slim_text(obs['text'])}
                                    current['samples'].append({'seed':task.seed,'act':s['act'],'floor':s['floor'],'room':current['room'],'turn':turn,
                                                               'world_prefix_length':len(actions),'full_public_sha256':public_hash(obs),'obs':stored})
                            elif current is not None:
                                finish(True)
                            with lock,torch.no_grad():logits,_,_=forward_policy(model,collate_public([obs]))
                            pos=int(logits[0,:len(obs['legal_actions'])].argmax());action=int(obs['legal_actions'][pos]['index'])
                            obs,_,done,cut,info=worker.submit('step',action).result();actions.append(action)
                        if cut:
                            receipt={'seed':task.seed,'censored':True,'discarded_fights':len(fights)+(current is not None),'actions':actions}
                        else:
                            finish(info.get('task_outcome') in ('victory','act_clear'))
                            for fight in fights:
                                samples=fight.pop('samples');fight.pop('sampled_turns')
                                bank.write(json.dumps(fight)+'\n');counts[f"{fight['room']}{fight['act']}"]+=1
                                for sample in samples:
                                    sample.update(fight_win=fight['won'],fight_start_public_sha256=fight['task']['expected_public_hash'],collector_protocol='single_policy_value_bank_v1')
                                    states.write(json.dumps(sample)+'\n');labels+=1
                            bank.flush();states.flush()
                            receipt={'seed':task.seed,'outcome':info.get('task_outcome'),'floor':obs['state']['floor'],'fights':len(fights),'actions':actions,'final_public_sha256':public_hash(obs)}
                    except SimulatorError as error:
                        receipt={'seed':task.seed,'fault':str(error),'fault_record_path':worker.worker.last_fault_path,'discarded_fights':len(fights)+(current is not None),'actions':actions}
                        worker.close();worker=make_workers(1,dead_action_filter=True,slim=False)[0]
                    receipts.write(json.dumps(receipt)+'\n');receipts.flush();worlds.append({k:v for k,v in receipt.items() if k!='actions'})
                    with lock:log.emit({'event':'world','slot':slot,**worlds[-1]})
            finally:worker.close()
        return worlds,counts,labels
    with ThreadPoolExecutor(a.threads) as executor:parts=list(executor.map(collect,range(a.threads)))
    worlds=[r for rows,_,_ in parts for r in rows];groups=Counter()
    for _,counts,_ in parts:groups.update(counts)
    summary={'schema':'sts_single_policy_value_bank_v1','run_record':record,'requested':a.n,'completed':len(worlds),'unstarted':a.n-len(worlds),
             'faults':sum('fault' in r for r in worlds),'censored':sum(r.get('censored',False) for r in worlds),'groups':dict(groups),
             'fights':sum(groups.values()),'states':sum(n for _,_,n in parts),'worlds':worlds,
             'scope':'Frozen policy single-observation inference; up to eight even-turn public states per fight, text slimmed using the legacy encoder allowlist. Full public hashes/provenance and world action prefixes support replay. Whole-world quarantine on a later fault/cut; ordinary completed fight outcomes only. No actor/value training.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log.emit({'event':'done',**{k:summary[k] for k in ['completed','unstarted','faults','censored','fights','states']}})
    print(json.dumps({k:summary[k] for k in ['completed','unstarted','faults','censored','fights','states','groups']}))

if __name__=='__main__':main()
