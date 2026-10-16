"""Synthetic controlled-builder interface and tiny receipts; never fits/runs native."""
from contextlib import contextmanager
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sts_policy.entity_encoding import VOCABULARY_SHA256
from sts_policy.model import CandidateActorCritic
from sts_policy.entity_model import EntityActorCritic
from sts_policy.model_agent import CompositeActorCritic
from sts_policy.training import save_checkpoint
from sts_rl import fresh_training
from sts_rl.fresh_scoring import SELECTION_CRITERION
from sts_rl.fresh_study import atomic_json
from sts_rl.recording import file_identity


class StubControlled(torch.nn.Module):
    def __init__(self, phase='combat', legacy_action_features=False):
        super().__init__()
        self.config = SimpleNamespace(width=8, layers=1, heads=2, dropout=0., phase=phase,
                                      legacy_action_features=legacy_action_features)
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def actor_logits(self, batch):
        assert not self.training and not torch.is_grad_enabled()
        return self.weight.expand(batch['action_mask'].shape)

    def training_logits(self, batch):
        raise AssertionError('vectorized training path called during inference')

    def metadata(self):
        return dict(kind='entity_controlled_v1', model_config=vars(self.config),
                    encoder_sha256='e' * 64, vocabulary_sha256=VOCABULARY_SHA256)


@contextmanager
def builder_stub():
    """Patch only the not-yet-integrated modules/helper; old/new stay real."""
    model = ModuleType('sts_policy.controlled_entity')
    encoder = ModuleType('sts_policy.controlled_encoding')
    observed = []

    def load(payload):
        meta = payload['metadata']
        if meta.get('encoder_sha256') != 'e' * 64 or meta.get('vocabulary_sha256') != VOCABULARY_SHA256:
            raise ValueError('Controlled source/vocabulary mismatch')
        result = StubControlled(meta['model_config']['phase'], meta['model_config']['legacy_action_features'])
        result.load_state_dict(payload['model_state'], strict=True)
        return result

    def encode(obs):
        observed.append(deepcopy(obs))
        return SimpleNamespace(action_indices=np.asarray([a['index'] for a in obs['legal_actions']]))

    def collate(rows, device='cpu'):
        assert device == 'cpu' and len(rows) == 1
        n = len(rows[0].action_indices)
        return dict(action_mask=torch.ones(1, n, dtype=torch.bool), legacy_action_features=torch.zeros(1, n, 128))

    original = fresh_training.checkpoint_training_condition

    def condition(payload):
        if payload['metadata']['config']['architecture'] != 'controlled':
            return original(payload)
        copied = deepcopy(payload)
        copied['metadata']['config']['architecture'] = 'new'
        result = original(copied)
        feature = payload['metadata']['model_config']['legacy_action_features']
        if type(feature) is not bool or payload['metadata']['config'].get('legacy_action_features') != feature:
            raise ValueError('Controlled condition feature mismatch')
        return dict(result, legacy_action_features=feature)

    model.ControlledEntityActorCritic, model.policy_from_payload = StubControlled, load
    encoder.encode_observation, encoder.collate_observations = encode, collate
    with patch.dict(sys.modules, {model.__name__: model, encoder.__name__: encoder}), \
            patch.object(fresh_training, 'checkpoint_training_condition', side_effect=condition):
        yield SimpleNamespace(observed=observed, encoder=encoder, model=model, condition=condition)


def champion(path):
    model = CompositeActorCritic(CandidateActorCritic(size='small'), CandidateActorCritic(size='small'))
    save_checkpoint(path, model, model_config=model.metadata(), metadata={'encoder_version': 'public_candidate_v2'})
    return path


def checkpoint(path, architecture='controlled', phase='combat', feature=False, **extra):
    model = (StubControlled(phase, feature) if architecture == 'controlled' else
             EntityActorCritic(width=8, layers=1, heads=2, phase=phase) if architecture == 'new' else
             CandidateActorCritic(size='small', context='none'))
    config = dict(architecture=architecture, phase=phase, fit_size='full', lr=.001, seed=7, steps=12,
                  min_steps=4, eval_every=2, patience=4, device='cpu', batch=8, defer_test=True)
    if architecture in ('new', 'controlled'):
        config.update(width=8, layers=1, heads=2)
    else:
        config.update(old_size='small', old_context='none')
    if architecture == 'controlled':
        config['legacy_action_features'] = feature
    meta = model.metadata() if architecture != 'old' else dict(encoder_version='public_candidate_v2', phase=phase)
    meta.update(config=config, data_identity='d' * 64, filters=fresh_training.FILTERS,
                vocabulary_sha256=VOCABULARY_SHA256, initialization='from_scratch', parent_checkpoint=None,
                training_state={'best_step': 12}, **extra)
    save_checkpoint(path, model, model_config=model.metadata(), metadata=meta, step=12)
    return file_identity(path)


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))


def score_stats(rows):
    groups = {}
    for row in rows:
        tags = ['all_rows', 'choice', 'group:' + row['group']]
        if row['confident']:
            tags.append('high_confidence_choice')
        for tag in tags:
            groups.setdefault(tag, []).append(row)
    return dict(groups={k: dict(rows=len(v), soft_ce=float(np.mean([r['ce'] for r in v])),
                               agreement=float(np.mean([r['correct'] for r in v]))) for k, v in groups.items()})


def complete_game_fixture(directory, worlds, policy, native_bytes):
    """Augment the existing pure cohort fixture with full launch snapshots."""
    import gzip
    from tests.unit.fresh_cohort_fixture import write_cohort
    directory = Path(directory)
    snapshot = directory / 'records' / 'launch-fixture'
    snapshot.mkdir(parents=True)
    native = dict(module_file='fixture.so', module_sha256=hashlib.sha256(native_bytes).hexdigest())
    for world in worlds:
        world['expected_provenance']['native_provenance'] = native
        world['task']['expected_provenance']['native_provenance'] = native
        for row in [*world['labels'], *world['bank']]:
            row['expected_provenance']['native_provenance'] = native
    config = dict(checkpoint=policy['champion']['path'], combat=policy['combat']['path'], ooc=None)
    write_cohort(directory, worlds, config=config, policy_changes=policy)
    source = json.loads((directory / 'receipt.json').read_text())
    (snapshot / 'source.zip').write_bytes(b'frozen source fixture')
    (snapshot / 'fixture.so').write_bytes(native_bytes)
    atomic_json(snapshot / 'build_manifest.json', native)
    launch = dict(schema='sts_training_launch_v1', effective_config=source['config'], parent_checkpoint=policy['champion'],
        source_archive=file_identity(snapshot / 'source.zip'), native=native, data_inputs=[policy['combat']],
        runtime=dict(python='fixture', platform='fixture', packages=dict(torch='fixture', numpy='fixture')), environment={})
    atomic_json(snapshot / 'launch.json', launch)
    source['launch'] = file_identity(snapshot / 'launch.json')
    atomic_json(directory / 'receipt.json', source)
    summary = json.loads((directory / 'summary.json').read_text())
    for brief in summary['records']:
        path = directory / brief['shard']
        with gzip.open(path, 'rt') as stream:
            value = json.load(stream)
        for row in [value, *value['labels'], *value['bank']]:
            row['source'] = source
        atomic_json(path, value, compressed=True)
        brief['shard_identity'] = file_identity(path)
    atomic_json(directory / 'summary.json', summary)
    completion = json.loads((directory / 'completion.json').read_text())
    completion.update(source=source, summary=file_identity(directory / 'summary.json'))
    atomic_json(directory / 'completion.json', completion)


class ScoreFixture:
    def __init__(self, root, stub, left='controlled', right='old'):
        self.root, self.stub = Path(root), stub
        self.root.mkdir(parents=True, exist_ok=True)
        self.native_bytes = b'never-loaded native fixture'
        binary = self.root / 'native.fixture'
        binary.write_bytes(self.native_bytes)
        self.native = dict(module_file='fixture.so', module_sha256=file_identity(binary)['sha256'])
        self.runtime = dict(python='fixture', platform='fixture', packages=dict(torch='fixture', numpy='fixture'))
        self.label_source = self.root / 'labels.json'
        atomic_json(self.label_source, {'synthetic': True})
        self.worlds = [w for w in range(90_000_000_000, 90_000_000_100) if fresh_training.world_split(w) == 'TEST'][:3]
        self.target = np.asarray([.8, .2], dtype=np.float32).tolist()
        self.metas = [dict(source_row_id=fresh_training.digest(['row', i]), world_seed=w, group='BOSS2',
            phase='combat', public_hash=fresh_training.digest(['obs', i]), target_sha256=fresh_training.digest(self.target),
            choice=True, confident=True, has_target=True, abstention=False, shard=str(self.label_source),
            source_identity='a' * 64, split='TEST') for i, w in enumerate(self.worlds)]
        self.manifest = dict(schema='fresh_data_manifest_v2', filters=fresh_training.FILTERS,
            source_files=[file_identity(self.label_source)], rows=self.metas,
            public_surface={'provenance': {'native_provenance': self.native}})
        self.manifest['data_identity'] = fresh_training.digest(self.manifest)
        self.manifest_path = self.root / 'manifest.json'
        atomic_json(self.manifest_path, self.manifest)
        self.checkpoints, self.specs, self.rows, self.selections = {}, {}, {}, {}
        for arm, arch in (('left', left), ('right', right)):
            path = self.root / (arm + '.pt')
            checkpoint(path, arch)
            payload = torch.load(path, weights_only=False)
            payload['metadata']['data_identity'] = self.manifest['data_identity']
            training_launch = self.launch(self.root / ('train-' + arm), payload['metadata']['config'], None, [])
            payload['metadata']['source_receipt'] = dict(launch=training_launch, data_identity=self.manifest['data_identity'],
                weight_parent=None, label_source_files=self.manifest['source_files'], filters=fresh_training.FILTERS,
                vocabulary_sha256=VOCABULARY_SHA256, architecture=arch, defer_test=True)
            torch.save(payload, path)
            identity = file_identity(path)
            condition = stub.condition(payload)
            self.checkpoints[arm] = identity
            self.specs[arm] = dict(checkpoint=identity, architecture=arch, condition=condition)
            selection = dict(selected_checkpoint=identity, manifest=file_identity(self.manifest_path),
                selection_criterion=SELECTION_CRITERION, architecture=arch, training_condition=condition)
            selection_path = self.root / (arm + '-selection.json')
            atomic_json(selection_path, selection)
            self.selections[arm] = selection_path
            self.make_score(arm, payload, selection_path)
        self.cohort = self.root / 'cohort.json'
        atomic_json(self.cohort, [m['source_row_id'] for m in self.metas])
        self.expected = dict(schema='controlled_comparison_expectations_v1', version=1, mode='labels',
            left=self.specs['left'], right=self.specs['right'], cohort=file_identity(self.cohort),
            manifest=file_identity(self.manifest_path))
        self.expectations = self.root / 'expectations.json'
        atomic_json(self.expectations, self.expected)

    def launch(self, out, config, checkpoint, inputs):
        snapshot = out / 'records' / 'launch-fixture'
        snapshot.mkdir(parents=True)
        (snapshot / 'source.zip').write_bytes(b'synthetic frozen code')
        (snapshot / self.native['module_file']).write_bytes(self.native_bytes)
        atomic_json(snapshot / 'build_manifest.json', self.native)
        atomic_json(snapshot / 'launch.json', dict(schema='sts_training_launch_v1', effective_config=config,
            parent_checkpoint=checkpoint, data_inputs=inputs, native=self.native, runtime=self.runtime,
            environment={'OMP_NUM_THREADS': '1'}, source_archive=file_identity(snapshot / 'source.zip')))
        return file_identity(snapshot / 'launch.json')

    def make_score(self, arm, payload, selection_path):
        out = self.root / arm
        selection = json.loads(selection_path.read_text())
        effective = dict(stage='deferred_test_scoring', training=False, weight_updates=0,
            architecture=self.specs[arm]['architecture'], phase='combat', device='cpu', selection=file_identity(selection_path),
            selected_checkpoint=self.checkpoints[arm], manifest=file_identity(self.manifest_path),
            selection_criterion=SELECTION_CRITERION, data_identity=self.manifest['data_identity'])
        launch = self.launch(out, effective, self.checkpoints[arm],
            [file_identity(selection_path), file_identity(self.manifest_path), *self.manifest['source_files']])
        receipt = dict(effective, launch=launch, filters=fresh_training.FILTERS,
            vocabulary_sha256=VOCABULARY_SHA256, source_receipt=payload['metadata']['source_receipt'],
            label_source_files=self.manifest['source_files'])
        atomic_json(out / 'receipt.json', receipt)
        self.rows[arm] = [dict(m, correct=(i < 2 if arm == 'left' else i == 2),
            ce=(.2 if arm == 'left' else .7), prediction_index=17, target=self.target,
            data_identity=self.manifest['data_identity'], inference='single_observation_exact_argmax_actor_only_v1',
            selected_checkpoint=self.checkpoints[arm]['sha256'], architecture=self.specs[arm]['architecture'],
            training_condition=self.specs[arm]['condition']) for i, m in enumerate(self.metas)]
        self.reseal_score(arm)

    def reseal_score(self, arm):
        out = self.root / arm
        receipt = json.loads((out / 'receipt.json').read_text())
        write_rows(out / 'test_rows.jsonl', self.rows[arm])
        atomic_json(out / 'summary.json', dict(status='completed', training=False, weight_updates=0,
            selection=receipt['selection'], selected_checkpoint=self.checkpoints[arm], data_identity=self.manifest['data_identity'],
            test_evaluated=True, test_protocol='single_observation_exact_argmax_actor_only_v1', test_device='cpu',
            test_rows=len(self.rows[arm]), test_rows_file=file_identity(out / 'test_rows.jsonl'),
            test=score_stats(self.rows[arm])))
