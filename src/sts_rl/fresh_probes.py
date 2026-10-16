"""Unlabelled fixed 93e9 movement probes. Never a supervised target source."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json

import numpy as np
import torch

from sts_policy.curriculum import public_hash
from sts_policy import entity_encoding
from sts_rl.fresh_study import PROTOCOL, INFERENCE, policy_observation
from sts_rl.fresh_training import (_integer, _validate_collection_source, digest, shard_files,
                                   validate_provenance, public_surface, coverage, movement, observation_encoder)
from sts_rl.recording import file_identity

PROBE_LABEL = 'fixed93e9 probes'


def iter_probe_rows(sources, *, expected_surface=None, control=None):
    """Only completed C3 world shards; source actions verify replay, never targets."""
    verified, worlds, seen = {}, set(), set()
    surface = expected_surface
    for path in shard_files(sources):
        if control:
            control.check()
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            shard = json.load(stream)
        if shard.get('protocol') != PROTOCOL or shard.get('status') != 'completed' or 'labels' not in shard:
            raise ValueError('Probe requires a completed C3 world shard')
        world = _integer(shard['world_seed'], 'probe world')
        if not 93_000_000_000 <= world < 94_000_000_000 or world in worlds:
            raise ValueError('Duplicate/out-of-namespace probe world')
        worlds.add(world)
        source, prov = shard['source'], shard['expected_provenance']
        start = _integer(source['config']['world_start'], 'probe source world start')
        n = _integer(source['config']['n'], 'probe source world count')
        if not 93_000_000_000 <= start <= world < start+n <= 94_000_000_000:
            raise ValueError('Probe world outside receipted source namespace/range')
        key = digest(source)
        if key not in verified:
            verified[key] = _validate_collection_source(source, labelled=False)
        identity, launch = verified[key]
        validate_provenance(prov, launch)
        current_surface = public_surface(source, prov)
        surface = surface or current_surface
        if current_surface != surface:
            raise ValueError('Probe/label public environment/native/input-context identity mismatch')
        prefix = shard['action_prefix']
        if (not isinstance(prefix, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in prefix)
                or len(shard['labels']) != len(prefix)):
            raise ValueError('Incomplete probe trajectory')
        for position, row in enumerate(shard['labels']):
            if control and position % 128 == 0:
                control.check()
            if row.get('label_kind') != 'probe_unlabelled' or 'label' in row or 'votes' in row:
                raise ValueError('Probe rows must be unlabelled')
            if (row['world_seed'] != world or row['source'] != source or row['expected_provenance'] != prov
                    or row['prefix'] != prefix[:position] or row['source_action'] != prefix[position]):
                raise ValueError('Probe source trajectory/prefix mismatch')
            _integer(row['source_action'], 'probe source action')
            obs = row['observation']
            if public_hash(obs) != row['public_hash']:
                raise ValueError('Probe public hash mismatch')
            prepared = policy_observation(obs)
            indices = [_integer(a['index'], 'probe action index') for a in prepared['legal_actions']]
            if not indices or len(set(indices)) != len(indices) or row['source_action'] not in indices:
                raise ValueError('Probe action/menu identity mismatch')
            if obs['state'].get('observation_version', prov['observation_version']) != prov['observation_version']:
                raise ValueError('Probe observation version mismatch')
            act = _integer(obs['state']['act'], 'probe act')
            group = f"{str(obs['state'].get('room', '')).rsplit('.',1)[-1]}{act}"
            if act not in (1, 2, 3) or row['group'] != group:
                raise ValueError('Probe act/room group mismatch')
            row_id = digest([world, position, row['public_hash']])
            if row_id in seen:
                raise ValueError('Duplicate probe row')
            seen.add(row_id)
            yield row, {'source_row_id': row_id, 'world_seed': world, 'group': group, 'act': act,
                'phase': 'combat' if obs['phase'] == 'combat' else 'ooc', 'choice': len(indices)>1,
                'public_hash': row['public_hash'], 'prefix_length': position,
                'action_indices': indices, 'menu_sha256': digest(prepared['legal_actions']),
                'shard': str(path.resolve()), 'source_identity': identity,
                'source_archive': launch['source_archive'], 'public_surface': current_surface}


@dataclass
class ProbeExample:
    encoded: object
    meta: dict


def select_probes(sources, phase, *, expected_surface=None, control=None):
    """Architecture/seed/subset-independent smallest 64 stable row IDs per ACT."""
    files = sorted((file_identity(p) for p in shard_files(sources)), key=lambda identity:identity['path'])
    chosen, counts, worlds, source_records = defaultdict(list), Counter(), set(), {}
    for raw, meta in iter_probe_rows(sources, expected_surface=expected_surface, control=control):
        worlds.add(meta['world_seed'])
        source_records[meta['source_identity']] = {'source': raw['source'], 'archive': meta['source_archive'],
                                                  'public_surface': meta['public_surface']}
        counts['raw_rows'] += 1
        if meta['phase'] != phase:
            continue
        counts['phase_rows'] += 1
        counts[f"act{meta['act']}_"+('choices' if meta['choice'] else 'forced')] += 1
        if meta['choice']:
            bucket = chosen[meta['act']]
            bucket.append((raw, meta))
            bucket.sort(key=lambda pair: pair[1]['source_row_id'])
            del bucket[64:]
    rows = [pair for act in sorted(chosen) for pair in chosen[act]]
    if not rows:
        raise ValueError('No nontrivial probes for configured phase')
    # Always report B3 coverage, even when the old model consumes this shared set.
    examples = [ProbeExample(entity_encoding.encode_observation(policy_observation(raw['observation'])), meta)
                for raw, meta in rows]
    if files != sorted((file_identity(p) for p in shard_files(sources)), key=lambda identity:identity['path']):
        raise ValueError('Probe source changed during selection')
    selection = {'schema': 'fixed93e9_probe_selection_v1', 'label': PROBE_LABEL, 'phase': phase,
        'rule': 'smallest stable row IDs per ACT, nontrivial filtered choices only, up to 64 per ACT',
        'source_files': files, 'sources': source_records, 'source_worlds': sorted(worlds),
        'selected_worlds': sorted({meta['world_seed'] for _, meta in rows}), 'counts': dict(counts),
        'rows': [meta for _, meta in rows], 'selected_count': len(rows),
        'by_act': {str(act): {'selected': sum(x.meta['act']==act for x in examples),
            'eligible_choices': counts[f'act{act}_choices'],
            'source_worlds': sorted({x.meta['world_seed'] for x in examples if x.meta['act']==act})}
            for act in (1, 2, 3)},
        'b3_vocabulary_sha256': entity_encoding.VOCABULARY_SHA256, 'b3_coverage': coverage(examples),
        'use': 'actor movement monitoring only; excluded from splits, targets, optimizer and selection'}
    selection['selection_identity'] = digest(selection)
    return selection, [raw for raw, _ in rows]


def encode_probes(raws, selection, architecture):
    encoder = observation_encoder(architecture)
    result = []
    for raw, meta in zip(raws, selection['rows'], strict=True):
        encoded = encoder.encode_observation(policy_observation(raw['observation']))
        if encoded.action_indices.tolist() != meta['action_indices']:
            raise ValueError('Probe encoder/menu alignment failure')
        result.append(ProbeExample(encoded, meta))
    return result


class ProbeMonitor:
    """CPU singleton actor-only inference; accelerator optimizer tensors stay put."""
    def __init__(self, model, probes, architecture, control=None):
        self.probes, self.architecture, self.control = probes, architecture, control
        self.shadow = None
        if next(model.parameters()).device.type != 'cpu':
            # deepcopy allocates detached storage without constructing/randomizing a model.
            self.shadow = deepcopy(model).to('cpu').eval()
            self.shadow.requires_grad_(False)
        self.start, self.start_positions = self.score(model)
        self.previous = self.start
        self.previous_positions = self.start_positions

    @torch.inference_mode()
    def score(self, model):
        scorer = self.shadow or model
        if self.shadow is not None:
            self.shadow.load_state_dict({k: v.detach().to('cpu') for k, v in model.state_dict().items()})
        was_training = scorer.training
        scorer.eval()
        encoder = observation_encoder(self.architecture)
        result, positions = [], []
        try:
            for probe in self.probes:
                if self.control:
                    self.control.check()
                batch = encoder.collate_observations([probe.encoded], device='cpu')
                logits = scorer.actor_logits(*batch) if self.architecture == 'old' else scorer.actor_logits(batch)
                if logits.shape != (1, len(probe.meta['action_indices'])) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite/misaligned singleton probe logits')
                result.append(logits[0].softmax(-1).numpy().copy())
                positions.append(int(logits[0].argmax()))
        finally:
            scorer.train(was_training)
        return result, positions

    def measure(self, model):
        current, positions = self.score(model)
        def summarize(indices):
            probabilities = [current[i] for i in indices]
            stats = movement(probabilities, [self.previous[i] for i in indices], [self.start[i] for i in indices])
            stats.pop('label')
            stats.update(argmax_change_previous=float(np.mean([
                positions[i] != self.previous_positions[i] for i in indices])),
                argmax_change_start=float(np.mean([positions[i] != self.start_positions[i] for i in indices])))
            stats.update(count=len(indices), entropy=float(np.mean([
                -(p*np.log(np.maximum(p, 1e-30))).sum() for p in probabilities])),
                top_probability=float(np.mean([p.max() for p in probabilities])),
                source_worlds=sorted({self.probes[i].meta['world_seed'] for i in indices}))
            return stats
        event = {'label': PROBE_LABEL, 'phase': self.probes[0].meta['phase'], 'inference': INFERENCE,
                 'device': 'cpu', 'single_observation': True, 'actor_only': True,
                 'pooled': summarize(list(range(len(self.probes)))),
                 'by_act': {str(act): summarize([i for i, p in enumerate(self.probes) if p.meta['act']==act])
                            for act in (1, 2, 3) if any(p.meta['act']==act for p in self.probes)}}
        self.previous = current
        self.previous_positions = positions
        return event
