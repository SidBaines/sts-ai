#!/usr/bin/env python3
"""Bounded fixed-observation inference and optional real collector benchmarks.

Collector timing is descriptive. Free-running random policies die at different
points; only the teacher-forced trajectory mode supports matched throughput.
The old control defaults to the champion's large/none architecture with fresh
weights; it is not parameter-matched to the entity transformer.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import torch
from sts_policy.entity_model import EntityActorCritic
from sts_policy.training import save_checkpoint, seed_training
from sts_rl.fresh_study import StudyPolicy, atomic_json, policy_observation, validate_world_range
from sts_rl.fresh_training import (TrainingControl, actor_logits, collate, iter_rows, make_model,
                                    Example, soft_cross_entropy, memory_stats, training_world, FILTERS)
from sts_rl.recording import file_identity
from sts_policy.curriculum import public_hash


def timed_logits(policy, observation):
    from sts_policy import entity_encoding, encoding
    start=time.perf_counter()
    observation=policy_observation(observation)
    phase='combat' if observation['phase']=='combat' else 'ooc'
    model=policy.replacements.get(phase,policy.champion)
    entity=isinstance(model,EntityActorCritic)
    module=entity_encoding if entity else encoding
    encoded=module.encode_observation(observation)
    batch=module.collate_observations([encoded],device='cpu')
    encoded_at=time.perf_counter()
    with torch.inference_mode():
        logits=model.actor_logits(batch) if entity else model.actor_logits(*batch)
    finished=time.perf_counter()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite benchmark logits')
    return logits[0],encoded_at-start,finished-encoded_at


def fixed_workload(policy, observations, *, seconds=5, max_passes=10, control=None):
    start=time.perf_counter()
    count,encode,inference=0,0.,0.
    for _ in range(max_passes):
        for observation in observations:
            if control:
                control.check()
            if time.perf_counter()-start>=seconds:
                break
            _,e,i=timed_logits(policy,observation)
            count+=1
            encode+=e
            inference+=i
        if time.perf_counter()-start>=seconds:
            break
    elapsed=time.perf_counter()-start
    return dict(decisions=count,seconds=elapsed,decisions_per_s=count/elapsed,
                encode_collate_seconds=encode,inference_seconds=inference,**memory_stats(),
                protocol='single_observation_cpu_one_torch_thread_including_forced_menus',
                workload='identical frozen TRAIN observations',games_per_s=None)


class TimedCollectorPolicy:
    def __init__(self, policy, *, force_champion=False):
        self.policy=policy
        self.force_champion=force_champion
        self.identity=policy.identity
        self.decisions,self.choice,self.encode,self.inference=0,0,0.,0.

    def choose_action(self, observation):
        filtered=policy_observation(observation)
        logits,e,i=timed_logits(self.policy,observation)
        self.decisions+=1
        self.choice+=int(len(filtered['legal_actions'])>1)
        self.encode+=e
        self.inference+=i
        if self.force_champion:
            # Same loaded champion, same raw trajectory for both contenders.
            original=self.policy.replacements
            self.policy.replacements={}
            try:
                return self.policy.choose_action(observation)
            finally:
                self.policy.replacements=original
        return int(filtered['legal_actions'][int(logits.argmax())]['index'])


def microbenchmark(observations, *, width, layers, heads, steps=40, control=None):
    """Optional bounded supervised tiny-overfit parity, never a device speed claim."""
    from sts_policy.entity_encoding import encode_observation
    examples=[]
    for observation in observations:
        o=policy_observation(observation)
        if len(o['legal_actions'])<2:
            continue
        # A separate, explicitly artificial two-choice overfit workload.
        tiny={**o,'legal_actions':[o['legal_actions'][0],o['legal_actions'][-1]]}
        examples.append(Example(encode_observation(tiny),np.array([1.,0.],np.float32),{}))
        if len(examples)==4:
            break
    if not examples:
        return {'available':False,'reason':'No nontrivial choices in bounded workload'}
    results={}
    for device in ('cpu','mps'):
        if device=='mps' and not torch.backends.mps.is_available():
            results[device]={'available':False}
            continue
        seed_training(1919)
        model=make_model('new','combat',width=width,layers=layers,heads=heads).to(device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003)
        batch,target,mask=collate(examples,'new',device)
        if device=='mps':
            torch.mps.synchronize()
        start=time.perf_counter()
        losses=[]
        for _ in range(steps):
            if control:
                control.check()
            loss=soft_cross_entropy(model.training_logits(batch),target,mask).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite microbenchmark loss/gradient')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if device=='mps':
            torch.mps.synchronize()
        elapsed=time.perf_counter()-start
        with torch.no_grad():
            logits=model.training_logits(batch)
            final_loss=float(soft_cross_entropy(logits,target,mask).mean().cpu())
            agreement=float((logits.argmax(-1)==target.argmax(-1)).float().mean().cpu())
        results[device]={'available':True,'seconds':elapsed,'losses':losses,'final_loss':final_loss,
            'target_agreement':agreement,'tiny_overfit_target_met':final_loss<.05 and agreement==1.,
            'finite_gradients':True,'loss_decreased':final_loss<losses[0],
            'target_note':'Artificial first-action onehot on two retained choices; no study labels used'}
    if results.get('mps',{}).get('available'):
        results['tiny_overfit_parity']={'final_loss_absolute_difference':abs(results['cpu']['losses'][-1]-results['mps']['losses'][-1]),
            'both_targets_met':results['cpu']['tiny_overfit_target_met'] and results['mps']['tiny_overfit_target_met']}
    return results


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--checkpoint',required=True,help='Champion teacher for baseline, never weight initialization')
    p.add_argument('--out',required=True)
    p.add_argument('--deadline-utc',required=True)
    p.add_argument('--world-start',type=int,default=90_900_000_000)
    p.add_argument('--n',type=int,default=1)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--observations',type=int,default=32)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--old-size',choices=('small','medium','large'),default='large',
                   help='Old candidate size (default: large, matching champion architecture; fresh weights)')
    p.add_argument('--old-context',choices=('none','mean'),default='none',
                   help='Old candidate menu context (default: none, matching champion architecture)')
    p.add_argument('--seed',type=int,default=1919)
    p.add_argument('--collect',action='store_true',help='Run the real collector on bounded fresh TRAIN games')
    p.add_argument('--teacher-forced',action='store_true',help='Run shadow decisions on champion trajectories')
    p.add_argument('--microbenchmark',action='store_true')
    return p


def main(argv=None):
    p=parser()
    a=p.parse_args(argv)
    if not 0<a.seconds<=120 or not 1<=a.n<=8 or not 1<=a.observations<=256:
        p.error('Bounds: seconds <=120, n <=8, observations <=256')
    validate_world_range(a.world_start,a.n)
    if not training_world(a.world_start) or not training_world(a.world_start+a.n-1):
        p.error('Requires fresh TRAIN namespace')
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Fresh benchmark output required')
    torch.set_num_threads(1)
    seed_training(a.seed)
    control=TrainingControl(out,a.deadline_utc)
    control.start()
    try:
        observations=[]
        for row,meta in iter_rows(a.data,control=control):
            if meta['split'] != 'FIT':
                continue
            observations.append(row['observation'])
            if len(observations)>=a.observations:
                break
        if not observations:
            raise ValueError('No fixed workload observations')
        paths={architecture:{} for architecture in ('new','old')}
        for architecture in paths:
            for phase in ('combat','ooc'):
                control.check()
                model=make_model(architecture,phase,width=a.width,layers=a.layers,heads=a.heads,
                                 old_size=a.old_size,old_context=a.old_context)
                metadata={**model.metadata(),'initialization':'from_scratch','parent_checkpoint':None}
                if architecture=='old':
                    metadata.update(encoder_version='public_candidate_v2',phase=phase,
                                    old_size=a.old_size,old_context=a.old_context)
                paths[architecture][phase]=str(out/(architecture+'-'+phase+'.pt'))
                save_checkpoint(paths[architecture][phase],model,model_config=model.metadata(),metadata=metadata)
        champion=StudyPolicy(a.checkpoint)
        policies=[('champion',champion,None)]
        for name,architecture in (('random_old','old'),('random_entity','new')):
            phase_paths=paths[architecture]
            policies.append((name,StudyPolicy(a.checkpoint,phase_paths['combat'],phase_paths['ooc']),architecture))
        report={'fixed_workload':{},'collector':{},'initialization':'random from scratch',
                'filters':FILTERS,'workload_public_hashes':[public_hash(o) for o in observations],
                'workload_split':'FIT',
                'champion':file_identity(a.checkpoint),
                'fresh_models':{k:file_identity(v) for k,v in paths['new'].items()},
                'old_models':{k:file_identity(v) for k,v in paths['old'].items()},
                'old_control':{'old_size':a.old_size,'old_context':a.old_context,'initialization':'from_scratch',
                    'match':'champion architecture at large/none; not parameter-matched to transformer'},
                'action_semantics':'decisions count all exposed filtered menus; choice decisions reported separately',
                'performance_claim':'bounded measurements only; free-running games/s depends on survival'}
        for name,policy,_ in policies:
            report['fixed_workload'][name]=fixed_workload(policy,observations,seconds=a.seconds,control=control)
        if a.microbenchmark:
            report['supervised_microbenchmark']=microbenchmark(observations,width=a.width,layers=a.layers,heads=a.heads,control=control)
        if a.collect:
            from sts_rl.fresh_collection import run_collection, DEFAULT_PROBABILITIES
            for name,policy,architecture in policies:
                control.check()
                timed=TimedCollectorPolicy(policy,force_champion=a.teacher_forced)
                cfg=dict(mode='eval',checkpoint=a.checkpoint,combat=paths[architecture]['combat'] if architecture else None,
                    ooc=paths[architecture]['ooc'] if architecture else None,out=str(out/name),world_start=a.world_start,
                    n=a.n,workers=1,timeout=30.,deadline_utc=a.deadline_utc,bank=False,teacher_labels=True,
                    teacher_probabilities=DEFAULT_PROBABILITIES,max_labels=0,samples=24,simulations=1000,self_consistency_every=0)
                cfg['benchmark_teacher_forced_shadow']=a.teacher_forced
                cfg.update(old_size=a.old_size,old_context=a.old_context)
                start=time.perf_counter()
                result=run_collection(cfg,timed)
                elapsed=time.perf_counter()-start
                report['collector'][name]={'summary':result,'seconds':elapsed,
                    'games_per_s':result['completed']/elapsed,'decisions_per_s':timed.decisions/elapsed,
                    'decisions':timed.decisions,'choice_decisions':timed.choice,
                    'encode_collate_seconds':timed.encode,'inference_seconds':timed.inference,
                    **memory_stats(),'teacher_forced':a.teacher_forced}
            if a.teacher_forced:
                def actions(name):
                    import gzip
                    rows=[]
                    for path in sorted((out/name/'worlds').glob('*.json.gz')):
                        with gzip.open(path,'rt') as stream:
                            r=json.load(stream)
                        rows.append((r['world_seed'],r['action_hash'],r['terminal_hash']))
                    return rows
                report['teacher_forced_trajectories_equal']=(len(actions('champion'))==a.n and
                    all(actions(name)==actions('champion') for name,_,_ in policies))
                if not report['teacher_forced_trajectories_equal']:
                    raise ValueError('Teacher-forced real trajectories differ or are incomplete')
        atomic_json(out/'benchmark.json',report)
        print(json.dumps(report,allow_nan=False))
    finally:
        control.close()

if __name__=='__main__':
    main()
