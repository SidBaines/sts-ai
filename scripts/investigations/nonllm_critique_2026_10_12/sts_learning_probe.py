from __future__ import annotations
import json, math, random, statistics, time, os
from collections import Counter, defaultdict
from pathlib import Path
import torch
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.training import load_checkpoint, seed_training, file_identity
from sts_policy.model_agent import policy_from_payload, collate_public, forward_policy
from sts_policy.rollout_training import collect_rollouts
from sts_policy.losses import PPOConfig, masked_distribution
from sts_policy.workers import CurriculumWorker

BASE = Path('/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI')
ORIGINAL = Path(os.environ.get('STS_PROBE_CHECKPOINT','/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0030.pt'))

def describe(rows):
    if not rows: return {}
    return {'n':len(rows),'mean':sum(rows)/len(rows),'median':statistics.median(rows),'min':min(rows),'max':max(rows)}

def calibration(rows):
    if not rows: return {}
    vs=[x[0] for x in rows]; ys=[x[1] for x in rows]; errors=[v-y for v,y in rows]
    var=statistics.pvariance(ys)
    return {'n':len(rows),'value_mean':statistics.mean(vs),'MC_mean':statistics.mean(ys),'bias':statistics.mean(errors),'RMSE':math.sqrt(statistics.mean(e*e for e in errors)), 'explained_variance':1-statistics.pvariance(errors)/var if var>0 else None}

def main():
    torch.set_num_threads(1); seed_training(991)
    model=policy_from_payload(load_checkpoint(ORIGINAL)).cpu()
    seeds=[]
    for seed in range(200000000,200001000):
        if world_split(seed)=='train': seeds.append(seed)
        if len(seeds)==32: break
    tasks=[TaskSpec(seed=x, kind='run', seek_combat=False,max_act=3,horizon=4096,battle_simulations=200) for x in seeds]
    config=PPOConfig(gamma=1,gae_lambda=.95,learning_rate=.0001,minibatch_size=1024,epochs=4,clip_ratio=.1,entropy_weight=.01,kl_guard_before_update=True)
    workers=[CurriculumWorker(config={'gamma':1,'reward_mode':'progress','progress_floor_weight':.05,'terminal_hp_weight':0,'goal_act_clear':True,'fast_observations':True,'fast_replay':True},timeout=10) for _ in range(4)]
    try:
        result=collect_rollouts(model,workers,tasks,collate=collate_public,forward=forward_policy,config=config,max_seconds=30,device='cpu',deterministic=os.environ.get('STS_PROBE_GREEDY')=='1',deferred_bootstrap=True)
    finally:
        for worker in workers: worker.close()
    ts=result.pop('transitions'); result.pop('teacher_records')
    byphase=Counter(); byactphase=Counter(); forced=Counter(); calib=defaultdict(list); byworld=defaultdict(list)
    for t in ts:
        phase=t.observation['phase']; act=t.observation['state']['act']; byphase[phase]+=1; byactphase[(act,phase)]+=1
        if len(t.observation['legal_actions'])==1: forced[phase]+=1
        byworld[t.world_seed].append(t)
    credit=[]; episodes=[]
    for seed,rows in byworld.items():
        future=0.
        for t in reversed(rows):
            future=t.reward+future
            phase=t.observation['phase'];act=t.observation['state']['act']
            calib[f'act{act}/{phase}'].append((t.behavior_value,future))
            calib[phase].append((t.behavior_value,future))
        for i,t in enumerate(rows):
            if t.observation['phase']=='out_of_combat' and len(t.observation['legal_actions'])>1:
                credit.append((len(rows)-1-i,.95**(len(rows)-1-i)))
        episodes.append({'world':seed,'steps':len(rows),'phase_counts':dict(Counter(t.observation['phase'] for t in rows)),'reward':sum(t.reward for t in rows)})
    # Phase-local probes: actor/critic share encoders. Backward only, no optimizer step.
    grads={}
    for phase in sorted(byphase):
        subset=[t for t in ts if t.observation['phase']==phase]
        random.Random(991).shuffle(subset); subset=subset[:512]
        logits,values,mask=forward_policy(model,collate_public([t.observation for t in subset],'cpu'))
        dist=masked_distribution(logits,mask)
        act=torch.tensor([t.action_position for t in subset])
        adv=torch.tensor([t.advantage for t in ts]); normalized=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        index={id(t):i for i,t in enumerate(ts)}
        local_adv=torch.tensor([normalized[index[id(t)]] for t in subset])
        actor=(-dist.log_prob(act)*local_adv).mean()-.01*dist.entropy().mean()
        critic=.25*(values-torch.tensor([t.returns for t in subset])).square().mean()
        def vector(loss):
            model.zero_grad(set_to_none=True);loss.backward(retain_graph=True)
            return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for name,p in model.named_parameters() if 'encoder' in name])
        av=vector(actor);cv=vector(critic)
        grads[phase]={'rows':len(subset),'actor_loss':float(actor.detach()),'weighted_critic_loss':float(critic.detach()),'encoder_actor_norm':float(av.norm()),'encoder_critic_norm':float(cv.norm()),'encoder_actor_critic_cosine':float(torch.nn.functional.cosine_similarity(av,cv,dim=0)), 'critic_action_encoder_grad_norm':math.sqrt(sum(float(p.grad.square().sum()) for name,p in model.named_parameters() if 'action_encoder' in name and p.grad is not None)), 'normalized_advantages':describe(local_adv.tolist())}
    out={'probe':('read-only '+('greedy' if os.environ.get('STS_PROBE_GREEDY')=='1' else 'stochastic')+' policy; four CPU workers, one thread; no optimizer steps'),'model_config':load_checkpoint(ORIGINAL)['model_config'],'source':str(ORIGINAL),'source_sha256':file_identity(ORIGINAL),'training_environment':load_checkpoint(ORIGINAL)['metadata'].get('training_environment'),'worlds':seeds,'config':config.__dict__,'collection':result,'transition_phase_counts':dict(byphase),'transition_act_phase_counts':{str(k):v for k,v in byactphase.items()},'forced_phase_counts':dict(forced),'MC_calibration':{k:calibration(v) for k,v in calib.items()},'choice_OOC_decisions_to_terminal':describe([x[0] for x in credit]),'choice_OOC_direct_lambda_factor':describe([x[1] for x in credit]),'gradients':grads,'episodes':episodes}
    path=Path(os.environ.get('STS_PROBE_OUTPUT','/private/tmp/sts_learning_probe.json'));path.write_text(json.dumps(out,indent=2)+'\n')
    for k in ('transition_phase_counts','transition_act_phase_counts','forced_phase_counts','MC_calibration','choice_OOC_decisions_to_terminal','choice_OOC_direct_lambda_factor','gradients'): print(k,json.dumps(out[k]))
    print('collection',result['elapsed_seconds'],'n',len(ts),'outcomes',dict(Counter(e['outcome'] for e in result['episodes'])),'faults',len(result['faults']),'cuts',sum(e['truncated'] for e in result['episodes']))
    print('artifact',path)
if __name__=='__main__':main()
