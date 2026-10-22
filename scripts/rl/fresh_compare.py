#!/usr/bin/env python3
"""Predeclared paired Gate C comparisons; no collection or confirmation access."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from sts_rl.fresh_statistics import (compare_games, compare_labels, compare_panel, teacher_consistency,
                                     validate_expectations)
from sts_rl.fresh_study import atomic_json, validate_world_range
from sts_rl.recording import file_identity


def read_games(path, *, allow_teacher_plays=False):
    """C4 brief summaries omit source; recover it only after cohort validation."""
    path = Path(path).resolve()
    if path.is_dir():
        directory = path
    elif path.name == 'summary.json' and path.is_file():
        directory = path.parent
    else:
        raise ValueError('Games require a completed receipted cohort directory or its summary.json')
    # D unit fixtures keep C3 dependencies. Root supplies C4 at integration;
    # importing it here keeps pure comparison helpers native/dependency-free.
    from sts_rl.fresh_cohort import validate_cohorts
    cohort = validate_cohorts([directory], allow_teacher_plays=allow_teacher_plays)
    policy = cohort['science_identity']['policy']
    records = [{**r, 'source':{'policy':policy}} for r in cohort['records']]
    for row in records:
        validate_world_range(row['world_seed'],1)
    return records


def frozen_expectations(path, mode):
    identity = file_identity(path)
    expectations = json.loads(Path(path).read_text())
    validate_expectations(expectations, mode)
    identities = ([expectations[arm+'_checkpoint'] for arm in ('student','control')] if mode=='labels'
                  else [p[k] for arm in ('student','champion') for p in [expectations[arm+'_policy']]
                        for k in ('champion','combat','ooc') if p[k] is not None])
    for item in identities:
        if file_identity(item['path']) != item:
            raise ValueError('Frozen comparison checkpoint file identity mismatch')
    if mode=='labels':
        from sts_policy.training import load_checkpoint
        from sts_rl.fresh_training import checkpoint_training_condition
        for arm in ('student','control'):
            payload = load_checkpoint(expectations[arm+'_checkpoint']['path'])
            architecture = expectations[arm+'_architecture']
            metadata = payload['metadata']
            expected_encoder = 'public_entities_fresh_v1' if architecture=='new' else 'public_candidate_v2'
            if (metadata['config']['architecture'] != architecture or metadata.get('encoder_version') != expected_encoder or
                    checkpoint_training_condition(payload) != expectations[arm+'_condition']):
                raise ValueError('Frozen label expectation differs from selected checkpoint config/architecture')
    if file_identity(path) != identity:
        raise ValueError('Comparison expectations changed while reading')
    return expectations, identity, identities


def read_rows(path, mode=None):
    if mode=='games':
        return read_games(path)
    path = Path(path)
    if path.is_dir():
        shards = sorted((path/'worlds').glob('*.json.gz'))
        if not shards:
            raise ValueError(f'No world shards in {path}/worlds; pass actual shard directories, not a sharded group root')
        return [row for p in shards for row in read_rows(p,mode)]
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:
        if path.name.endswith(('.json','.json.gz')):
            value=json.load(stream)
            rows=value if isinstance(value,list) else [value]
        else:
            rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows)==1 and isinstance(rows[0],dict) and 'records' in rows[0]:
        rows = rows[0]['records']
    for row in rows:
        if 'world_seed' in row:
            validate_world_range(row['world_seed'],1)
    if mode=='consistency':
        # Drop ordinary observations before opening the next source world.
        return [label for world in rows for label in world.get('labels',[world]) if label.get('self_consistency')]
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('labels','games','panel','consistency'))
    p.add_argument('--student',required=True)
    p.add_argument('--champion')
    p.add_argument('--cohort',help='JSON exact seed list or manifest row-ID list (required for final games/labels)')
    p.add_argument('--expectations',help='Frozen v1 policy/checkpoint/condition bindings; required for games/labels')
    p.add_argument('--grid')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.mode in ('labels','games'):
        if not a.cohort or not a.champion or not a.expectations:
            p.error('games/labels require --cohort, --champion and --expectations')
        expectations, identity, checkpoints = frozen_expectations(a.expectations,a.mode)
        rows=read_rows(a.student,a.mode)
        cohort=json.loads(Path(a.cohort).read_text())
        other=read_rows(a.champion,a.mode)
        result=(compare_games(rows,other,cohort,expectations=expectations) if a.mode=='games'
                else compare_labels(rows,other,expected_ids=cohort,expectations=expectations))
        for item in [identity,*checkpoints]:
            if file_identity(item['path']) != item:
                raise ValueError('Frozen comparison input changed during comparison')
        result['provenance'] = {'expectations_file':identity}
        print(f"Frozen expectations: {identity['path']} (SHA256 {identity['sha256']})",file=sys.stderr)
    elif a.mode=='panel':
        rows=read_rows(a.student,a.mode)
        if not a.grid:
            p.error('panel requires --grid')
        result=compare_panel(rows,json.loads(Path(a.grid).read_text()))
    else:
        rows=read_rows(a.student,a.mode)
        if not rows:
            p.error('consistency found no repeat-query rows; pass actual shard directories with repeats')
        result=teacher_consistency(rows)
    atomic_json(a.out,result)
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__':
    main()
