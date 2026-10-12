#!/usr/bin/env python3
"""CPU-only DEV encoder profile against the pre-cache ordered hash arithmetic."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import time
import tracemalloc

from sts_policy import encoding
from sts_policy.curriculum import world_split


def reference_bag(target, prefix, text, weight=1.0):
    words = encoding._WORDS.findall(str(text).lower())
    for word in [*words, *(a + ':' + b for a, b in zip(words, words[1:]))]:
        index, sign = encoding._hash(prefix + ':' + word, len(target))
        target[index] += sign * weight


@contextmanager
def using_bag(function):
    previous = encoding._bag
    encoding._bag = function
    try: yield
    finally: encoding._bag = previous


def run(rows, bag, repeats=1):
    with using_bag(bag):
        start = time.perf_counter()
        for _ in range(repeats):
            for observation in rows: encoding.encode_observation(observation)
        return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl'))
    parser.add_argument('--per-phase', type=int, default=256)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, worlds, phases = [], set(), Counter()
    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    with args.source.open() as source:
        for line in source:
            row = json.loads(line)
            if world_split(row['world_seed']) != 'dev': continue
            phase = row['observation']['phase']
            if phases[phase] >= args.per_phase: continue
            rows.append(row['observation']); worlds.add(row['world_seed']); phases[phase] += 1
            if all(phases[p] >= args.per_phase for p in ('combat','out_of_combat')): break
    if not rows: raise ValueError('no DEV observations')
    candidate = encoding._bag
    encoder_digest=hashlib.sha256(Path(encoding.__file__).read_bytes()).hexdigest()
    run(rows, reference_bag); run(rows, candidate)
    old, new = [], []
    for _ in range(args.trials):
        old.append(run(rows, reference_bag, args.repeats)); new.append(run(rows, candidate, args.repeats))
    result = {'source':str(args.source),'source_sha256':source_digest,'split':'dev','rows':len(rows),
              'worlds':sorted(worlds),'phase_counts':dict(phases),'trials':args.trials,'repeats':args.repeats,
              'reference_seconds':old,'cached_seconds':new,'speedup':statistics.median(old)/statistics.median(new),
              'reference_ms_per_observation':1000*statistics.median(old)/(len(rows)*args.repeats),
              'cached_ms_per_observation':1000*statistics.median(new)/(len(rows)*args.repeats),
              'encoder_version':encoding.ENCODER_VERSION,
              'encoder_sha256':encoder_digest}
    old_encoded, new_encoded = [], []
    with using_bag(reference_bag): old_encoded = [encoding.encode_observation(row) for row in rows]
    new_encoded = [encoding.encode_observation(row) for row in rows]
    for old_row, new_row in zip(old_encoded, new_encoded):
        for key in ('state_features','action_features','action_indices'):
            assert getattr(old_row,key).tobytes() == getattr(new_row,key).tobytes()
        assert old_row.coverage == new_row.coverage
    result['feature_arrays_indices_coverage_bit_equal'] = True
    for phase in phases:
        group = [row for row in rows if row['phase']==phase]
        old_time = run(group, reference_bag, args.repeats); new_time = run(group, candidate, args.repeats)
        result[phase] = {'rows':len(group),'reference_ms':1000*old_time/(len(group)*args.repeats),
                         'cached_ms':1000*new_time/(len(group)*args.repeats),'speedup':old_time/new_time}
    profile = cProfile.Profile()
    with using_bag(reference_bag):
        profile.enable(); [encoding.encode_observation(row) for row in rows]; profile.disable()
    stats = pstats.Stats(profile)
    result['reference_top_cumulative'] = [{'function':str(key),'seconds':value[3],'calls':value[1]}
        for key,value in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:12]]
    encoding._bag_terms.cache_clear(); tracemalloc.start(); run(rows, candidate)
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    result.update(cache_info=encoding._bag_terms.cache_info()._asdict(),traced_cache_current_bytes=current,
                  traced_cache_peak_bytes=peak,cache_text_limit=encoding._BAG_CACHE_TEXT_LIMIT)
    # Inference on saved DEV arrays only: no simulator, GPU, or policy episodes.
    import torch
    from sts_policy.model import CandidateActorCritic
    from sts_policy.model_agent import CompositeActorCritic
    torch.set_num_threads(1); torch.manual_seed(1729)
    encoding._bag_terms.cache_clear()
    result['cold_cache_full_encode_seconds'] = run(rows,candidate)
    result['fresh_cpu_collation']={}
    for batch_size in (8,32):
        times=[]
        for bag in (reference_bag,candidate):
            samples=[]
            with using_bag(bag):
                for _ in range(3):
                    start=time.perf_counter()
                    for offset in range(0,len(rows),batch_size):
                        encoding.collate_observations([encoding.encode_observation(row) for row in rows[offset:offset+batch_size]])
                    samples.append(time.perf_counter()-start)
            times.append(samples)
        result['fresh_cpu_collation'][str(batch_size)]={'reference_seconds':times[0],'cached_seconds':times[1],
                                                      'speedup':statistics.median(times[0])/statistics.median(times[1])}
    result['preencoded_cpu_collation_seconds']={}
    for batch_size in (8,32):
        start=time.perf_counter()
        for offset in range(0,len(rows),batch_size): encoding.collate_observations(new_encoded[offset:offset+batch_size])
        result['preencoded_cpu_collation_seconds'][str(batch_size)]=time.perf_counter()-start
    models = [CandidateActorCritic('small','mean'), CompositeActorCritic(CandidateActorCritic('small','mean'),CandidateActorCritic('small','mean'))]
    torch_result=[]
    for model in models:
        model.eval()
        with torch.no_grad():
            for start in range(0,min(128,len(rows)),8):
                a = encoding.collate_observations(old_encoded[start:start+8]); b = encoding.collate_observations(new_encoded[start:start+8])
                for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
                old_logits,old_values=model(*a);new_logits,new_values=model(*b)
                torch.testing.assert_close(old_logits,new_logits,rtol=0,atol=0)
                torch.testing.assert_close(old_values,new_values,rtol=0,atol=0)
                assert torch.equal(old_logits.argmax(-1),new_logits.argmax(-1))
                torch.manual_seed(83);old_choices=torch.distributions.Categorical(logits=old_logits).sample()
                torch.manual_seed(83);new_choices=torch.distributions.Categorical(logits=new_logits).sample()
                assert torch.equal(old_choices,new_choices)
        torch_result.append(type(model).__name__)
    result['cpu_collation_logits_values_deterministic_and_sampled_actions_bit_equal'] = torch_result
    if encoder_digest != hashlib.sha256(Path(encoding.__file__).read_bytes()).hexdigest(): raise ValueError('encoder source changed during profile')
    if source_digest != hashlib.sha256(args.source.read_bytes()).hexdigest(): raise ValueError('source data changed during profile')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:result[key] for key in ('rows','phase_counts','speedup','reference_ms_per_observation','cached_ms_per_observation','cache_info','traced_cache_current_bytes','cpu_collation_logits_values_deterministic_and_sampled_actions_bit_equal')},sort_keys=True))


if __name__=='__main__': main()
