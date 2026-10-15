#!/usr/bin/env python3
"""Bounded frozen-policy collection benchmark; launch through queue_runner.

Use --source-root to compare an archived source tree in a fresh process. Each
repeat resets collector RNG identically, checks per-world action/feature hashes,
and reports collection separately from fingerprinting, persistence and updates.
No training controller, old run, or existing checkpoint is modified.
"""
import argparse
from collections.abc import Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time


class CanonicalRows(Sequence):
    """Stable world ordering while preserving the original row-storage costs."""
    def __init__(self, rows):
        self.rows = rows
        self.order = sorted(range(len(rows)), key=lambda i: rows[i]['seed'])

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.rows[i] for i in self.order[index]]
        return self.rows[self.order[index]]


def fingerprint(result):
    worlds = {}
    for row in result['rows']:
        h = worlds.setdefault(row['seed'], hashlib.sha256())
        h.update(row['state'].tobytes())
        h.update(row['actions'].tobytes())
        h.update(json.dumps({k:v for k,v in row.items() if k not in ('state','actions')},
                            sort_keys=True, separators=(',',':')).encode())
    return {'episodes': {str(e['world_seed']): hashlib.sha256(json.dumps(e, sort_keys=True,
                        separators=(',',':')).encode()).hexdigest() for e in result['episodes']},
            'rows': {str(seed): h.hexdigest() for seed,h in worlds.items()},
            'counts': result['counts'], 'choices': result['choices'], 'buckets': result['buckets']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--collection-backend', choices=['legacy','compact'], default='legacy')
    p.add_argument('--evaluation-backend', choices=['legacy','lean'], default='legacy')
    p.add_argument('--kind', choices=['train','eval'], default='eval')
    p.add_argument('--collectors', type=int, default=2)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--games', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--world-start', type=int, default=64_500_000_000)
    p.add_argument('--persist', action='store_true')
    p.add_argument('--update', action='store_true')
    p.add_argument('--critic-init')
    a = p.parse_args()
    if a.repeats < 1 or (a.update and a.repeats != 1):
        p.error('Positive repeats required; --update requires --repeats 1')
    sys.path.insert(0, str(Path(a.source_root).resolve()/'src'))
    import torch
    from sts_rl.common import load_policy, train_tasks
    from sts_rl.joint_sparse import (JointConfig, SparseCollectors, WinCritics, actor_update,
        critic_update, freeze_legacy_critics, identity_state, likelihood_error, save_buffer)
    from sts_rl.recording import begin_run
    out = Path(a.out).resolve()
    if (out/'report.json').exists():
        p.error('Fresh output required')
    out.mkdir(parents=True, exist_ok=True)
    record = begin_run(out, vars(a), a.checkpoint, inputs=[a.critic_init] if a.critic_init else ())
    torch.set_num_threads(2)
    torch.manual_seed(19)
    actor, _ = load_policy(a.checkpoint)
    reference, _ = load_policy(a.checkpoint)
    freeze_legacy_critics(actor)
    critics = WinCritics()
    if a.critic_init:
        payload = torch.load(a.critic_init, weights_only=False)
        critics.load_state_dict(payload['critic_state'])
    fields = dict(out=str(out), checkpoint=a.checkpoint, collectors=a.collectors, workers=a.workers,
                  games=a.games, max_games=a.games, chunk=16,
                  critic_minibatch=256 if a.critic_init else 0, critic_init=a.critic_init)
    # The baseline predates these flags; reject unsupported requested paths.
    for key in ('collection_backend','evaluation_backend'):
        value = getattr(a,key)
        if key in JointConfig.__dataclass_fields__:
            fields[key] = value
        elif value != 'legacy':
            p.error('This source snapshot does not implement '+key)
    cfg = JointConfig(**fields)
    cfg.validate()
    launch = time.monotonic()
    pool = SparseCollectors(cfg)
    samples = []
    try:
        warm = pool.collect(actor,critics,train_tasks(a.world_start+100000, max(a.collectors*2,8)),
                            a.kind,19, lambda x: None)
        del warm
        startup_seconds = time.monotonic()-launch
        for repeat in range(a.repeats):
            if (out/'STOP').exists():
                raise RuntimeError('Benchmark STOP requested between stages')
            start = time.monotonic()
            result = pool.collect(actor,critics,train_tasks(a.world_start,a.games),a.kind,19,
                                  lambda x: print(json.dumps({'progress': x}),flush=True))
            elapsed = time.monotonic()-start
            if result['faults'] or len(result['episodes']) != a.games or any(
                e['classification'] not in ('win','loss') for e in result['episodes']):
                raise RuntimeError('Benchmark cohort invalid; preserve faults/cuts, never replace worlds')
            identity = fingerprint(result)
            if samples and identity != samples[0]['fingerprints']:
                raise RuntimeError('Fresh repeat changed per-world trajectory/features')
            sample = dict(repeat=repeat,seconds=elapsed,games_per_second=a.games/elapsed,
                          rows=len(result['rows']),wins=sum(e['classification']=='win' for e in result['episodes']),
                          faults=len(result['faults']),encoded_bytes=result['encoding_bytes'],fingerprints=identity,
                          collector_seconds=result.get('collector_seconds'),timings_sum=result.get('timings'),
                          forward_counts=result.get('forward_counts'),
                          owner_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          children_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            if a.persist:
                start=time.monotonic()
                path=out/f'buffer_{repeat}.pt.gz'
                save_buffer({'rows':result['rows']},path,compresslevel=6)
                sample.update(persist_seconds=time.monotonic()-start,persist_bytes=path.stat().st_size)
                # The fingerprint and final receipt preserve benchmark identity;
                # keep only the final buffer for numerical replay.
                if repeat: (out/f'buffer_{repeat-1}.pt.gz').unlink()
            if a.update:
                start=time.monotonic()
                sample['likelihood_error']=likelihood_error(actor,result['rows'],cfg)
                sample['likelihood_seconds']=time.monotonic()-start
                if sample['likelihood_error'] > 1e-3: raise RuntimeError('Likelihood mismatch')
                opts={phase:torch.optim.AdamW([v for v in getattr(actor,phase).parameters() if v.requires_grad],
                                              lr=getattr(cfg,phase+'_lr'),weight_decay=0) for phase in ('combat','ooc')}
                co=torch.optim.AdamW(critics.parameters(),lr=cfg.critic_lr,weight_decay=.01)
                # Collector completion order is intentionally irrelevant to this
                # numerical format test. Canonical world order supplies identical
                # immutable rows to both optimizers, without reordering a world.
                ordered=CanonicalRows(result['rows'])
                sample['update_row_storage']=getattr(result['rows'],'format','legacy_rows')
                anchors=[]
                for phase in ('combat','ooc'):
                    phase_rows=[r for r in ordered if r['phase']==phase]
                    anchors.extend(phase_rows[::max(1,len(phase_rows)//128)][:128])
                del phase_rows
                start=time.monotonic()
                update=actor_update(actor,reference,opts,ordered,anchors,cfg,random.Random(19),.1)
                critic=critic_update(critics,co,ordered,cfg)
                sample.update(update_seconds=time.monotonic()-start,accepted=update['accepted'],
                              actor_sha256=identity_state(actor),critic_sha256=identity_state(critics),
                              optimizer_hash=hash_tensors({p:o.state_dict() for p,o in opts.items()}),
                              critic_optimizer_hash=hash_tensors(co.state_dict()),
                              update=update,critic=critic)
                if not update['accepted']: raise RuntimeError('Numerical benchmark update rejected')
                # Exactly one optimizer comparison; collection repeats need frozen weights.
            samples.append(sample)
            (out/'report.partial.json').write_text(json.dumps(samples,indent=2)+'\n')
            print(json.dumps({k:v for k,v in sample.items() if k not in ('fingerprints','update','critic')}),flush=True)
            del result
            gc.collect()
    finally:
        pool.close()
    report={'schema':'joint_collection_benchmark_v1','arguments':vars(a),'record':record,
            'startup_seconds':startup_seconds,'samples':samples,
            'owner_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'children_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def hash_tensors(value):
    """Stable optimizer hash, independent of pickle's storage identifiers."""
    import torch
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.cpu().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str): h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v: visit(item)
        else: h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


if __name__=='__main__': main()
