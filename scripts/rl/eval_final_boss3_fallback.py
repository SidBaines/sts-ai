#!/usr/bin/env python3
"""Public-context learned-policy ablation: original combat only at final floor50 Act3 boss."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from sts_env.env import SimulatorError
    from sts_policy.curriculum import public_hash
    from sts_policy.model_agent import collate_public,forward_policy
    from sts_rl.common import BOSS_FLOORS,WIN_OUTCOMES,JsonlLog,load_policy,make_workers,train_tasks
    from sts_rl.evaluate import finish_evaluation
    from sts_rl.recording import begin_run,file_identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--fallback-checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--progress-output',required=True)
    parser.add_argument('--n',type=int,default=1024)
    parser.add_argument('--world-start',type=int,default=5800000000)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seconds',type=float,default=3600.)
    parser.add_argument('--use-fallback',action=argparse.BooleanOptionalAction,default=True)
    a=parser.parse_args()
    if Path(a.output).exists() or min(a.n,a.threads,a.seconds)<=0 or not math.isfinite(a.seconds):
        parser.error('Require fresh output and positive finite bounds')
    record=begin_run(str(a.output)+'.record',vars(a),a.checkpoint,inputs=[a.fallback_checkpoint])
    torch.set_num_threads(1);student,_=load_policy(a.checkpoint);original,_=load_policy(a.fallback_checkpoint)
    student.eval();original.eval()
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in student.state_dict().items() if k.startswith('ooc.'))
    tasks=train_tasks(a.world_start,a.n);started=time.monotonic();lock=threading.Lock();log=JsonlLog(a.progress_output);completed=[0]
    log.emit({'event':'start','config':vars(a),'run_record':record})
    def run(slot):
        worker=make_workers(1,timeout=10.,dead_action_filter=True,slim=True)[0]
        rows=[];faults=censored=0
        try:
            for task in tasks[slot::a.threads]:
                if time.monotonic()-started>=a.seconds:break
                try:
                    obs,info=worker.submit('reset',task).result();done=cut=False;steps=switched=0;boss_hp={};boss3_hash=None
                    while not (done or cut):
                        state=obs['state'];room=str(state.get('room','')).rsplit('.',1)[-1]
                        if obs['phase']=='combat' and state.get('floor') in BOSS_FLOORS:
                            boss_hp.setdefault(state['floor'],state.get('cur_hp'))
                            if state['floor']==50 and boss3_hash is None:boss3_hash=public_hash(obs)
                        eligible=obs['phase']=='combat' and room=='BOSS' and int(state.get('act') or 0)==3 and state.get('floor')==50
                        use=a.use_fallback and eligible
                        with lock,torch.no_grad():logits,_,_=forward_policy(original if use else student,collate_public([obs]))
                        action=obs['legal_actions'][int(logits[0,:len(obs['legal_actions'])].argmax())]['index']
                        switched+=int(use);obs,_,done,cut,info=worker.submit('step',int(action)).result();steps+=1
                    if cut or steps>=4096:
                        censored+=1;progress={'seed':task.seed,'censored':True,'steps':steps}
                    else:
                        row={'seed':task.seed,'floor':obs['state'].get('floor'),'win':info.get('task_outcome') in WIN_OUTCOMES,'steps':steps,
                             'boss3_fallback_decisions':switched,'boss3_entry_public_sha256_slimmed':boss3_hash}
                        for floor in BOSS_FLOORS:row[f'boss{floor}_hp']=boss_hp.get(floor)
                        rows.append(row);progress=row
                except SimulatorError as error:
                    faults+=1;progress={'seed':task.seed,'fault':str(error),'fault_record_path':worker.last_fault_path}
                    worker.close();worker=make_workers(1,timeout=10.,dead_action_filter=True,slim=True)[0]
                with lock:
                    completed[0]+=1;log.emit({'event':'world','completed':completed[0],'requested':a.n,'elapsed_seconds':time.monotonic()-started,**progress})
        finally:worker.close()
        return rows,faults,censored
    with ThreadPoolExecutor(a.threads) as executor:parts=list(executor.map(run,range(a.threads)))
    rows=[r for records,_,_ in parts for r in records]
    if not rows:raise RuntimeError('No valid worlds; inspect progress/faults')
    result=finish_evaluation(rows,sum(f for _,f,_ in parts),time.monotonic()-started,
                             mode='public_final_boss3_learned_fallback_v2' if a.use_fallback else 'greedy_single_observation_control',world_start=a.world_start)
    result['summary'].update(requested=a.n,unstarted=a.n-completed[0],censored=sum(c for _,_,c in parts),
                             cohort_completed=completed[0]==a.n,inference_protocol='single_public_observation_argmax_v1',
                             student=file_identity(a.checkpoint),fallback=file_identity(a.fallback_checkpoint),use_fallback=a.use_fallback,
                             fallback_protocol='Public phase==combat,room==BOSS,act==3,floor==50 only; student everywhere else. No native search.')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result)+'\n')
    log.emit({'event':'done',**result['summary']});print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
