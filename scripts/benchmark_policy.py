#!/usr/bin/env python3
"""Bounded, synchronized CPU/MPS/MLX candidate-policy throughput screen."""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from sts_policy.model import CandidateActorCritic, parameter_count


def atomic(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def inputs(batch,n,seed=31):
    rng=np.random.default_rng(seed)
    states=rng.normal(0,.25,(batch,256)).astype(np.float32)
    actions=rng.normal(0,.25,(batch,n,128)).astype(np.float32)
    mask=np.arange(n)[None,:] < rng.integers(max(1,n//2),n+1,(batch,1))
    labels=np.argmax(np.where(mask,actions[:,:,0],-1e9),axis=1)
    return states,actions,mask,labels


def sync(device):
    if device=='mps': torch.mps.synchronize()


def torch_row(size,context,device,batch,n,steps):
    torch.manual_seed(17)
    model=CandidateActorCritic(size,context).to(device)
    tensors=[torch.as_tensor(x,device=device) for x in inputs(batch,n)]
    states,actions,mask,labels=tensors
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    def update():
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(states,actions,mask)
        loss=torch.nn.functional.cross_entropy(logits,labels)
        loss.backward(); optimizer.step()
        return loss
    for _ in range(3): update()
    sync(device); began=time.perf_counter()
    for _ in range(steps): loss=update()
    sync(device); elapsed=time.perf_counter()-began
    loss_value=float(loss.detach().cpu())
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(states,actions,mask)
        sync(device); began=time.perf_counter()
        for _ in range(max(steps,10)): logits,_=model(states,actions,mask)
        sync(device); inference=time.perf_counter()-began
    assert np.isfinite(loss_value)
    valid=logits.masked_fill(~mask,-1e9).argmax(-1)
    assert bool(mask.gather(1,valid[:,None]).all().cpu())
    row=dict(backend=device,size=size,context=context,batch=batch,candidates=n,
             parameters=parameter_count(model),train_examples_per_second=batch*steps/elapsed,
             inference_examples_per_second=batch*max(steps,10)/inference,
             train_seconds=elapsed,inference_seconds=inference,loss=loss_value)
    del model,optimizer,tensors
    gc.collect()
    if device=='mps': torch.mps.empty_cache()
    return row


def mlx_class():
    import mlx.core as mx
    import mlx.nn as nn
    class Model(nn.Module):
        def __init__(self,size,context):
            super().__init__(); self.context=context
            width,depth={'small':(128,2),'medium':(256,2),'large':(512,3)}[size]
            def mlp(dim):
                layers=[nn.Linear(dim,width),nn.SiLU()]
                for _ in range(depth-1): layers.extend([nn.Linear(width,width),nn.SiLU()])
                return nn.Sequential(*layers)
            self.state_encoder=mlp(256); self.action_encoder=mlp(128)
            self.scorer=nn.Sequential(nn.Linear(width*(4 if context=='mean' else 3),width),nn.SiLU(),nn.Linear(width,1))
            self.critic=nn.Sequential(nn.Linear(width*(2 if context=='mean' else 1),width),nn.SiLU(),nn.Linear(width,1))
        def __call__(self,state,actions,mask):
            se=self.state_encoder(state); ae=self.action_encoder(actions)
            sc=mx.broadcast_to(se[:,None,:],ae.shape)
            pieces=[sc,ae,sc*ae]; vp=[se]
            if self.context=='mean':
                weights=mask[:,:,None].astype(mx.float32)
                context=mx.sum(ae*weights,axis=1)/mx.maximum(mx.sum(weights,axis=1),1)
                pieces.append(mx.broadcast_to(context[:,None,:],ae.shape)); vp.append(context)
            logits=mx.where(mask,self.scorer(mx.concatenate(pieces,axis=-1)).squeeze(-1),-1e9)
            values=self.critic(mx.concatenate(vp,axis=-1)).squeeze(-1)
            return logits,values
    return Model


def mlx_row(size,context,batch,n,steps):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    torch.manual_seed(17)
    reference=CandidateActorCritic(size,context).eval()
    model=mlx_class()(size,context)
    weights=[]
    for name,value in reference.state_dict().items():
        pieces=name.split('.')
        weights.append((pieces[0]+'.layers.'+'.'.join(pieces[1:]),mx.array(value.numpy())))
    model.load_weights(weights)
    arrays=[mx.array(x) for x in inputs(batch,n)]
    state,actions,mask,labels=arrays
    with torch.no_grad():
        ts,ta,tm,_=[torch.as_tensor(x) for x in inputs(batch,n)]
        expected=reference(ts,ta,tm)[0].numpy()
    actual=model(state,actions,mask)[0]
    mx.eval(actual)
    valid=np.asarray(mask)
    max_error=float(np.max(np.abs(np.asarray(actual)[valid]-expected[valid])))
    assert max_error < 2e-4,(size,context,max_error)
    def loss_fn(m,state,actions,mask,labels):
        return nn.losses.cross_entropy(m(state,actions,mask)[0],labels,reduction='mean')
    vg=nn.value_and_grad(model,loss_fn)
    optimizer=optim.Adam(learning_rate=3e-4)
    def update():
        loss,grads=vg(model,state,actions,mask,labels)
        optimizer.update(model,grads)
        mx.eval(model.parameters(),optimizer.state,loss)
        return loss
    for _ in range(3): update()
    began=time.perf_counter()
    for _ in range(steps): loss=update()
    elapsed=time.perf_counter()-began
    loss_value=float(loss.item()); assert np.isfinite(loss_value)
    for _ in range(3): mx.eval(model(state,actions,mask))
    began=time.perf_counter()
    for _ in range(max(steps,10)): mx.eval(model(state,actions,mask))
    inference=time.perf_counter()-began
    row=dict(backend='mlx',size=size,context=context,batch=batch,candidates=n,
             parameters=parameter_count(reference),train_examples_per_second=batch*steps/elapsed,
             inference_examples_per_second=batch*max(steps,10)/inference,
             train_seconds=elapsed,inference_seconds=inference,loss=loss_value,
             forward_parity_max_error=max_error)
    del model,optimizer,reference,arrays
    gc.collect(); mx.clear_cache()
    return row


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=150); p.add_argument('--steps',type=int,default=8)
    args=p.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    began=time.monotonic(); rows=[]; errors=[]
    # Realistic menus first, ensuring useful evidence if the wall ceiling hits.
    configs=[(size,context,batch,8) for batch in (32,128,512,2048)
             for size in ('small','medium','large') for context in ('mean',)]
    configs += [(size,'none',256,8) for size in ('small','medium','large')]
    configs += [('small','mean',batch,64) for batch in (32,128,512)]
    for size,context,batch,n in configs:
        for backend in ('cpu','mps','mlx'):
            if time.monotonic()-began>args.seconds: break
            try:
                row=(mlx_row(size,context,batch,n,args.steps) if backend=='mlx'
                     else torch_row(size,context,backend,batch,n,args.steps))
                rows.append(row); print(json.dumps(row),flush=True)
            except Exception as error:
                errors.append(dict(backend=backend,size=size,context=context,batch=batch,n=n,error=str(error)))
                print(json.dumps(errors[-1]),flush=True)
            atomic(args.output,dict(rows=rows,errors=errors,elapsed_seconds=time.monotonic()-began,
                                    torch=torch.__version__,cpu_threads=4))
        if time.monotonic()-began>args.seconds: break
    atomic(args.output,dict(rows=rows,errors=errors,elapsed_seconds=time.monotonic()-began,
                            torch=torch.__version__,cpu_threads=4))

if __name__=='__main__': main()
