"""Tiny synthetic updates only; no native execution, study corpus or outcomes."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from sts_policy import controlled_encoding
from sts_policy.controlled_entity import ControlledEntityActorCritic, policy_from_payload
from sts_policy.training import load_checkpoint
from sts_rl import fresh_training as ft
from sts_rl.fresh_probes import encode_probes, ProbeMonitor
from tests.unit.test_entity_encoding import combat_observation
from tests.unit.test_fresh_training import fixture, synthetic_begin_run
from tests.unit.test_fresh_probes import probe_fixture


class ControlledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self, root, data, **extra):
        return dict(data=[str(data)], out=str(root / 'run'), architecture='controlled', phase='combat',
                    fit_size='full', width=16, layers=1, heads=4, steps=2, min_steps=2, batch=2,
                    eval_every=1, patience=6, lr=.001, seed=1919, device='cpu', deadline_utc=None,
                    defer_test=True, **extra)

    def test_deferred_run_checkpoint_condition_and_synthetic_singleton_scoring(self):
        for phase, flag in (('combat', False), ('ooc', True)):
            with self.subTest(phase=phase, flag=flag), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                data, source = fixture(root)
                native = json.loads(Path(source['launch']['path']).read_text())['native']
                cfg = self.config(root, data, legacy_action_features=flag)
                cfg['phase'] = phase
                stack.enter_context(patch.object(ft, 'validate_training_config',
                    side_effect=lambda c: ft_validate(c, synthetic=True)))
                stack.enter_context(patch.object(ft, 'TrainingControl', side_effect=lambda out, deadline:
                    ft_control(out, deadline, ac=lambda: True, caffeinate=False)))
                stack.enter_context(patch.object(ft, 'begin_run', side_effect=synthetic_begin_run(native)))
                split, seen = [None], []
                original_iter, original_encode, original_evaluate = ft.iter_rows, controlled_encoding.encode_observation, ft.evaluate
                def track_rows(*args, **kwargs):
                    for raw, meta in original_iter(*args, **kwargs):
                        split[0] = meta['split']
                        yield raw, meta
                def guard_encode(obs):
                    self.assertNotEqual(split[0], 'TEST')
                    seen.append(split[0])
                    return original_encode(obs)
                def guard_evaluate(model, examples, *args, **kwargs):
                    self.assertTrue(all(x.meta['split'] != 'TEST' for x in examples))
                    return original_evaluate(model, examples, *args, **kwargs)
                with patch.object(ft, 'iter_rows', side_effect=track_rows), \
                     patch.object(controlled_encoding, 'encode_observation', side_effect=guard_encode), \
                     patch.object(ft, 'evaluate', side_effect=guard_evaluate), \
                     patch.object(ft, 'load_checkpoint', side_effect=AssertionError('deferred run loaded weights')):
                    summary = ft.run_training(cfg)
                self.assertEqual(summary['updates'], 2)
                self.assertFalse(summary['test_evaluated'])
                self.assertTrue(summary['test_deferred'])
                self.assertEqual(set(seen), {'FIT', 'VALIDATION'})
                self.assertFalse((root / 'run' / 'test_rows.jsonl').exists())
                payload = load_checkpoint(root / 'run' / 'best.pt')
                loaded = policy_from_payload(payload)
                self.assertEqual(loaded.config.phase, phase)
                self.assertIs(payload['metadata']['config']['legacy_action_features'], flag)
                self.assertEqual(payload['metadata']['trainable_parameter_count'],
                                 payload['metadata']['policy_active_parameter_count'])
                self.assertEqual(payload['metadata']['policy_dormant_parameter_count'], 0 if flag else 128 * 16)
                condition = ft.checkpoint_training_condition(payload)
                self.assertIs(condition['legacy_action_features'], flag)
                self.assertEqual(condition['width'], 16)
                self.assertIs(json.loads((root / 'run' / 'receipt.json').read_text())['legacy_action_features'], flag)
                for value in (not flag, 1 if flag else 0, 'false', None):
                    bad = deepcopy(payload)
                    bad['metadata']['config']['legacy_action_features'] = value
                    with self.assertRaisesRegex(ValueError, 'legacy_action_features'):
                        ft.checkpoint_training_condition(bad)
                    with self.assertRaisesRegex(ValueError, 'legacy_action_features'):
                        policy_from_payload(bad)
                manifest = json.loads((root / 'run' / 'manifest.json').read_text())
                examples = ft.preencode([data], manifest, phase, 'controlled', splits=('TEST',))['TEST']
                before = (root / 'run' / 'best.pt').read_bytes()
                score_out = root / 'synthetic-score'
                score_out.mkdir()
                def check_score(model, rows, architecture, **kwargs):
                    self.assertEqual(architecture, 'controlled')
                    self.assertTrue(kwargs['singleton'])
                    self.assertTrue(all(p.device.type == 'cpu' and not p.requires_grad for p in model.parameters()))
                    with patch.object(model.critic, 'forward', side_effect=AssertionError('scoring called critic')):
                        return original_evaluate(model, rows, architecture, **kwargs)
                with patch.object(ft, 'evaluate', side_effect=check_score), \
                     patch('torch.optim.AdamW', side_effect=AssertionError('scoring optimizer')), \
                     patch('torch.Tensor.backward', side_effect=AssertionError('scoring backward')), \
                     patch.object(ft, 'save_checkpoint', side_effect=AssertionError('scoring save')):
                    result = ft.score_test_checkpoint(root / 'run' / 'best.pt', examples, 'controlled', score_out)
                self.assertTrue(result['test_evaluated'])
                self.assertEqual(result['test_rows'], 3)
                self.assertEqual(before, (root / 'run' / 'best.pt').read_bytes())
                rows = [json.loads(line) for line in (score_out / 'test_rows.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['training_condition'] == condition for r in rows))
                self.assertTrue(all(r['prediction_index'] in x.encoded.action_indices for r, x in zip(rows, examples)))
                # Bucket identities/exposures are unchanged from the entity path.
                entity = ft.preencode([data], manifest, phase, 'new', splits=('TEST',))['TEST']
                self.assertEqual(ft.minibatch_order(examples, 'controlled', 2, random.Random(5)),
                                 ft.minibatch_order(entity, 'new', 2, random.Random(5)))
                self.assertEqual(ft.coverage(examples), ft.coverage(entity))

    def test_controlled_run_uses_fixed_probe_receipt_without_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, source = fixture(root)
            probes = probe_fixture(root / 'probe', count=1)
            native = json.loads(Path(source['launch']['path']).read_text())['native']
            cfg = self.config(root, data, probe=[str(probes)], legacy_action_features=True)
            with patch.object(ft, 'validate_training_config', side_effect=lambda c: ft_validate(c, synthetic=True)), \
                 patch.object(ft, 'TrainingControl', side_effect=lambda out, deadline:
                              ft_control(out, deadline, ac=lambda: True, caffeinate=False)), \
                 patch.object(ft, 'begin_run', side_effect=synthetic_begin_run(native)):
                summary = ft.run_training(cfg)
            self.assertFalse(summary['test_evaluated'])
            selection = json.loads((root / 'run' / 'probe_selection.json').read_text())
            payload = load_checkpoint(root / 'run' / 'best.pt')
            self.assertEqual(selection['selected_count'], 3)
            self.assertEqual(payload['metadata']['probe_selection']['selection_identity'], selection['selection_identity'])
            events = [json.loads(line) for line in (root / 'run' / 'metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(e['fixed93e9_probes']['pooled']['count'] == 3 for e in events))
            self.assertTrue(all(e['fixed93e9_probes']['actor_only'] for e in events))

    def test_fit_rejects_controlled_flag_model_disagreement_before_optimizer(self):
        cfg = self.config(Path('/unused'), '/synthetic', legacy_action_features=False)
        model = ft.make_model('controlled', 'combat', width=16, layers=1, legacy_action_features=True)
        with patch('torch.optim.AdamW', side_effect=AssertionError('optimizer before config validation')):
            with self.assertRaisesRegex(ValueError, 'legacy_action_features'):
                ft.fit(model, {}, cfg, Path('/unused'), {}, {}, None)
        cfg['defer_test'] = False
        with patch('torch.optim.AdamW', side_effect=AssertionError('optimizer before deferred guard')):
            with self.assertRaisesRegex(ValueError, 'defer_test=True'):
                ft.fit(model, {}, cfg, Path('/unused'), {}, {}, None)

    def test_config_guard_and_boolean_cli_preserve_existing_defaults(self):
        cfg = self.config(Path('/unused'), '/synthetic')
        cfg.update(steps=2000, min_steps=2000, probe=['synthetic'], defer_test=False)
        with self.assertRaisesRegex(ValueError, '--defer-test'):
            ft.validate_training_config(cfg)
        cfg['defer_test'] = True
        ft.validate_training_config(cfg)
        for flag in ('false', 0, 1, None):
            cfg['legacy_action_features'] = flag
            with self.assertRaisesRegex(ValueError, 'boolean'):
                ft.validate_training_config(cfg)
        for architecture in ('old', 'new'):
            cfg.update(architecture=architecture, legacy_action_features=True)
            with self.assertRaisesRegex(ValueError, 'requires controlled'):
                ft.validate_training_config(cfg)
            with self.assertRaisesRegex(ValueError, 'requires controlled'):
                ft.make_model(architecture, 'combat', legacy_action_features=True)
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'rl' / 'fresh_train.py'
        spec = importlib.util.spec_from_file_location('controlled_train_cli', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = ['--data', 'synthetic', '--out', 'out', '--architecture', 'controlled', '--phase', 'combat',
                '--deadline-utc', '2099-01-01T00:00:00Z', '--defer-test']
        self.assertNotIn('legacy_action_features', vars(module.parser().parse_args(args)))
        self.assertIs(module.parser().parse_args(args + ['--legacy-action-features']).legacy_action_features, True)
        self.assertIs(module.parser().parse_args(args + ['--no-legacy-action-features']).legacy_action_features, False)

    def probes(self):
        obs = combat_observation()
        indices = [a['index'] for a in obs['legal_actions']]
        selection = {'rows': [dict(action_indices=indices, phase='combat', act=1, world_seed=93_000_000_001)]}
        probes = encode_probes([{'observation': obs}], selection, 'controlled')
        self.assertTrue(all(not hasattr(p, 'target') for p in probes))
        return probes

    def test_controlled_probes_cpu_preserve_rng_weights_and_training_mode(self):
        model = ft.make_model('controlled', 'combat', width=16, layers=1, legacy_action_features=True).train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch.object(model.critic, 'forward', side_effect=AssertionError('probe critic')):
            monitor = ProbeMonitor(model, self.probes(), 'controlled')
            metrics = monitor.measure(model)
        self.assertTrue(model.training)
        self.assertEqual(metrics['pooled']['kl_start_to_current'], 0.)
        self.assertEqual(metrics['pooled']['count'], 1)
        self.assertTrue(metrics['actor_only'])
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_controlled_mps_probe_shadow_is_detached_and_keeps_accelerator_weights(self):
        model = ft.make_model('controlled', 'combat', width=16, layers=1, legacy_action_features=True).to('mps').train()
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
        monitor = ProbeMonitor(model, self.probes(), 'controlled')
        monitor.measure(model)
        self.assertTrue(model.training)
        self.assertTrue(all(p.device.type == 'mps' for p in model.parameters()))
        self.assertTrue(all(p.device.type == 'cpu' and not p.requires_grad for p in monitor.shadow.parameters()))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(mps_rng, torch.mps.get_rng_state()))
        with torch.no_grad():
            next(monitor.shadow.parameters()).add_(1)
        self.assertTrue(torch.equal(next(iter(before.values())), next(model.parameters()).detach().cpu()))


# Preserve actual implementations before tests replace only their operational boundaries.
ft_validate, ft_control = ft.validate_training_config, ft.TrainingControl


if __name__ == '__main__':
    unittest.main()
