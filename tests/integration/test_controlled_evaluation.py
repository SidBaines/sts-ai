"""Real builder interface on synthetic checkpoints only; no fitting/native jobs.

After integration this uses local modules. Before integration, root may set
CONTROLLED_MODEL_SOURCE to a read-only source directory containing sts_policy
and sts_rl. No files in that directory are copied or written.
"""
from contextlib import ExitStack
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import sts_policy
from sts_policy.training import save_checkpoint
from sts_rl import fresh_training, fresh_scoring
from sts_rl.controlled_policy import ControlledStudyPolicy, policy_factory
from sts_rl.controlled_comparison import compare_labels_runs
from sts_rl.fresh_study import atomic_json, policy_observation
from sts_rl.recording import file_identity
from tests.integration.test_fresh_policy import observation
from tests.unit.controlled_fixture import builder_stub, champion, ScoreFixture


class RealControlledInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        source = os.environ.get('CONTROLLED_MODEL_SOURCE')
        if source:
            cls.old_path = list(sts_policy.__path__)
            cls.addClassCleanup(lambda: setattr(sts_policy, '__path__', cls.old_path))
            sts_policy.__path__.append(str(Path(source) / 'sts_policy'))
        try:
            cls.model_module = importlib.import_module('sts_policy.controlled_entity')
            cls.encoding = importlib.import_module('sts_policy.controlled_encoding')
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest('Controlled builder modules not integrated') from exc
        cls.builder_training = fresh_training
        if source:
            name = 'sts_rl._controlled_evaluation_builder_training'
            spec = importlib.util.spec_from_file_location(name, Path(source) / 'sts_rl' / 'fresh_training.py')
            cls.builder_training = importlib.util.module_from_spec(spec)
            sys.modules[name] = cls.builder_training
            cls.addClassCleanup(lambda: sys.modules.pop(name, None))
            spec.loader.exec_module(cls.builder_training)

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.champion = champion(self.root / 'champion.pt')

    def save(self, phase, feature):
        model = self.model_module.ControlledEntityActorCritic(width=8, layers=1, heads=2,
                                                              phase=phase, legacy_action_features=feature)
        path = self.root / f'{phase}-{feature}.pt'
        save_checkpoint(path, model, model_config=model.metadata(), metadata=model.metadata())
        return path

    def test_real_encoder_actor_parity_off_on_both_phases_and_first_tie(self):
        for feature in (False, True):
            paths = {p: self.save(p, feature) for p in ('combat', 'ooc')}
            policy = policy_factory(self.champion, **paths)
            self.assertIsInstance(policy, ControlledStudyPolicy)
            for phase in ('combat', 'ooc'):
                raw = observation(phase)
                if phase == 'ooc':
                    raw['text'] = 'Potions: EMPTY_POTION_SLOT'
                    raw['legal_actions'].append(dict(index=99, description='discard potion Fire Potion (slot 1)'))
                filtered = policy_observation(raw)
                encoded = self.encoding.encode_observation(filtered)
                batch = self.encoding.collate_observations([encoded], device='cpu')
                model = policy.replacements[phase]
                with torch.inference_mode():
                    expected = model.actor_logits(batch)[0]
                self.assertTrue(torch.equal(policy._logits(filtered), expected))
                self.assertEqual(policy.choose_action(raw), int(encoded.action_indices[int(expected.argmax())]))
                with torch.no_grad():
                    for parameter in model.actor.parameters():
                        parameter.zero_()
                self.assertEqual(policy.choose_action(raw), filtered['legal_actions'][0]['index'])

    def test_real_source_metadata_and_config_copy_tampering_rejected(self):
        path = self.save('combat', True)
        saved = path.read_bytes()
        for field in ('encoder_sha256', 'controlled_model_sha256', 'legacy_encoder_sha256',
                      'entity_encoder_sha256', 'entity_model_sha256', 'vocabulary_sha256', 'model_config_sha256'):
            path.write_bytes(saved)
            payload = torch.load(path, weights_only=False)
            payload['metadata'][field] = 'f' * 64
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                ControlledStudyPolicy(self.champion, combat=path)
        path.write_bytes(saved)
        payload = torch.load(path, weights_only=False)
        payload['model_config']['model_config']['phase'] = 'ooc'
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            ControlledStudyPolicy(self.champion, combat=path)

    def test_real_builder_singleton_scoring_through_extended_scorer_and_comparison(self):
        with builder_stub() as stub:
            fixture = ScoreFixture(self.root / 'fixture', stub)
        model = self.model_module.ControlledEntityActorCritic(width=8, layers=1, heads=2, legacy_action_features=False)
        path = Path(fixture.checkpoints['left']['path'])
        payload = torch.load(path, weights_only=False)
        metadata = {**payload['metadata'], **model.metadata()}
        save_checkpoint(path, model, model_config=model.metadata(), metadata=metadata, step=12)
        identity = file_identity(path)
        condition = self.builder_training.checkpoint_training_condition(torch.load(path, weights_only=False))
        selected = json.loads(fixture.selections['left'].read_text())
        selected.update(selected_checkpoint=identity, training_condition=condition)
        atomic_json(fixture.selections['left'], selected)
        raw = observation('combat')
        raw['legal_actions'] = raw['legal_actions'][:2]
        encoded = self.encoding.encode_observation(policy_observation(raw))
        examples = [self.builder_training.Example(encoded, np.asarray(fixture.target, dtype=np.float32), meta)
                    for meta in fixture.metas]
        class Control:
            def __init__(self, *args): self.status = {}
            def start(self): pass
            def check(self): pass
            def close(self): pass
            def emit(self): pass
            def stats(self): return {}
        def begin(out, effective, checkpoint, inputs):
            return fixture.launch(Path(out), effective, file_identity(checkpoint), [file_identity(p) for p in inputs])
        with ExitStack() as stack:
            stack.enter_context(patch.object(fresh_training, 'checkpoint_training_condition',
                                             self.builder_training.checkpoint_training_condition))
            stack.enter_context(patch.object(fresh_scoring, 'TrainingControl', Control))
            stack.enter_context(patch.object(fresh_scoring, 'begin_run', side_effect=begin))
            stack.enter_context(patch.object(fresh_scoring, 'freeze_manifest', return_value=fixture.manifest))
            stack.enter_context(patch.object(fresh_scoring, 'preencode', return_value={'TEST': examples}))
            stack.enter_context(patch.object(fresh_scoring, 'score_test_checkpoint', self.builder_training.score_test_checkpoint))
            stack.enter_context(patch.object(self.builder_training, 'fit', side_effect=AssertionError('fit')))
            out = self.root / 'real-score'
            summary = fresh_scoring.score_deferred(fixture.selections['left'], out)
            self.assertEqual(summary['test_rows'], len(examples))
            fixture.expected['left'].update(checkpoint=identity, condition=condition)
            atomic_json(fixture.expectations, fixture.expected)
            report = compare_labels_runs(out, fixture.root / 'right', fixture.cohort, fixture.expectations)
        self.assertTrue(report['filesystem_validated'])
        self.assertEqual(report['paired_rows'], 3)


if __name__ == '__main__':
    unittest.main()
