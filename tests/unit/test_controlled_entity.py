"""Bounded public fixtures only: controlled encoder, actor and strict identity."""
from copy import deepcopy
from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sts_policy import controlled_encoding as ce, encoding, entity_encoding
from sts_policy.controlled_entity import (ControlledEntityActorCritic, describe_capacity,
                                          nearest_policy_width, policy_from_payload)
from sts_policy.entity_model import EntityActorCritic
from sts_policy.training import load_checkpoint, save_checkpoint
from tests.unit.test_entity_encoding import combat_observation, ooc_observation


class ControlledEncodingTests(unittest.TestCase):
    def test_exact_legacy_features_original_entity_fields_and_gapped_menu_order(self):
        for raw in (combat_observation(), ooc_observation()):
            raw['legal_actions'].reverse()
            encoded = ce.encode_observation(raw)
            legacy = encoding.encode_observation(raw)
            entity = entity_encoding.encode_observation(raw)
            self.assertEqual(encoded.action_indices.tolist(), [a['index'] for a in raw['legal_actions']])
            np.testing.assert_array_equal(encoded.legacy_action_features, legacy.action_features)
            self.assertEqual(encoded.legacy_action_features.shape, (len(raw['legal_actions']), 128))
            self.assertFalse(hasattr(encoded, 'state_features'))
            self.assertEqual(encoded.coverage, entity.coverage)
            batch = ce.collate_observations([encoded])
            ordinary = entity_encoding.collate_observations([entity])
            self.assertEqual(set(batch) - set(ordinary), {'legacy_action_features'})
            for key, value in ordinary.items():
                self.assertTrue(torch.equal(value, batch[key]), key)
            for field in fields(entity):
                if field.name != 'coverage':
                    np.testing.assert_array_equal(getattr(encoded.entity, field.name), getattr(entity, field.name))
            legacy.action_indices[:] = legacy.action_indices[::-1]
            if len(legacy.action_indices) > 1:
                with patch.object(encoding, 'encode_observation', return_value=legacy):
                    with self.assertRaisesRegex(ValueError, 'alignment'):
                        ce.encode_observation(raw)

    def test_no_mutation_private_fields_or_diagnostic_text_leaks(self):
        raw = combat_observation()
        private = deepcopy(raw)
        private.update(seed=123, info={'teacher_value': 777}, diagnostics={'secret': 'private'})
        private['text'] += '\nSEARCH_DIAGNOSTICS: secret_value_12345\nseed: 999'
        private['state']['private_rng'] = [91, 92]
        private['state']['combat']['cards']['draw'][0]['private_draw_order'] = 123
        private['state']['combat']['player']['secret'] = 345
        for action in private['legal_actions']:
            action.update(bits=2**53, teacher_score=999)
        for enemy in private['state']['combat']['enemies']:
            enemy['hidden_move_history'] = [123, 456]
        before = deepcopy(private)
        base, extra = ce.encode_observation(raw), ce.encode_observation(private)
        self.assertEqual(private, before)
        for field in fields(base.entity):
            if field.name == 'coverage':
                self.assertEqual(base.coverage, extra.coverage)
            else:
                np.testing.assert_array_equal(getattr(base.entity, field.name), getattr(extra.entity, field.name))
        np.testing.assert_array_equal(base.legacy_action_features, extra.legacy_action_features)
        with patch('sts_rl.fresh_study.policy_observation', side_effect=AssertionError('second filtering')):
            ce.encode_observation(raw)


class ControlledModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, flag=False, **kwargs):
        return ControlledEntityActorCritic(width=16, layers=1, heads=4, legacy_action_features=flag, **kwargs)

    def batch(self):
        large = combat_observation()
        large['state']['combat']['cards']['draw'] *= 2
        large['state']['combat']['recent_actions'] *= 2
        extra = deepcopy(large['state']['combat']['cards']['hand'][0])
        extra['index'] = 6
        large['state']['combat']['cards']['hand'].append(extra)
        descriptions = ['exhaust: hand [0] Strike [Attack] (cost 1); hand [1] Strike [Attack] (cost 1, free once); hand [2] Strike [Attack] (cost 0)']
        descriptions += [a['description'] for a in large['legal_actions']] * 2
        large['legal_actions'] = [{'index': i * 3, 'description': d} for i, d in enumerate(descriptions)]
        small = ooc_observation()
        small['legal_actions'] = small['legal_actions'][:1]
        return ce.collate_observations([ce.encode_observation(small), ce.encode_observation(combat_observation()),
                                        ce.encode_observation(large)])

    def test_same_shared_initialization_rng_and_off_original_equivalence(self):
        torch.manual_seed(19)
        original = EntityActorCritic(width=16, layers=1, heads=4)
        original_rng = torch.get_rng_state().clone()
        models = []
        for flag in (False, True):
            torch.manual_seed(19)
            model = self.model(flag)
            self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
            for key, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
            self.assertIsNone(model.actor.legacy_action_projection.bias)
            self.assertEqual(int(model.actor.legacy_action_projection.weight.count_nonzero()), 0)
            models.append(model)
        # Explicitly copy independently initialized shared weights as well.
        torch.manual_seed(28)
        original = EntityActorCritic(width=16, layers=1, heads=4)
        missing = models[0].load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(missing.missing_keys, ['actor.legacy_action_projection.weight'])
        self.assertEqual(missing.unexpected_keys, [])
        batch = self.batch()
        for training in (True, False):
            original.train(training)
            models[0].train(training)
            self.assertTrue(torch.equal(original.actor_logits(batch), models[0].actor_logits(batch)))
            self.assertTrue(torch.equal(original.training_logits(batch), models[0].training_logits(batch)))
        with torch.no_grad():
            models[0].actor.legacy_action_projection.weight.fill_(2.)
        # OFF still feeds zero even if a dormant checkpoint tensor is nonzero.
        self.assertTrue(torch.equal(original.actor_logits(batch), models[0].actor_logits(batch)))

    def test_on_sensitivity_gradients_and_disjoint_critic(self):
        model, batch = self.model(True), self.batch()
        actor_ids = {id(p) for p in model.actor_parameters()}
        critic_ids = {id(p) for p in model.critic_parameters()}
        self.assertFalse(actor_ids & critic_ids)
        self.assertEqual(actor_ids | critic_ids, {id(p) for p in model.parameters()})
        logits = model.training_logits(batch)
        (-logits.log_softmax(-1)[:, 0].mean()).backward()
        self.assertGreater(float(model.actor.legacy_action_projection.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.critic_parameters()))
        with torch.no_grad():
            model.actor.legacy_action_projection.weight.fill_(.02)
        changed = {**batch, 'legacy_action_features': torch.zeros_like(batch['legacy_action_features'])}
        self.assertFalse(torch.equal(model.actor_logits(batch), model.actor_logits(changed)))
        self.assertTrue(torch.equal(model.critic_values(batch), model.critic_values(changed)))
        with patch.object(model.critic, 'forward', side_effect=AssertionError('actor called critic')):
            model.actor_logits(batch)
        model.zero_grad(set_to_none=True)
        with patch.object(model.actor, 'forward', side_effect=AssertionError('critic called actor')):
            model.critic_values(batch).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.actor_parameters()))
        self.assertTrue(any(p.grad is not None for p in model.critic_parameters()))

    def test_padding_unpadded_singleton_and_vectorized_agreement_on_and_off(self):
        batch = self.batch()
        alone = ce.collate_observations([ce.encode_observation(combat_observation())])
        self.assertGreater(batch['action_source_mask'].shape[2], alone['action_source_mask'].shape[2])
        self.assertGreater(batch['action_mask'].shape[1], alone['action_mask'].shape[1])
        self.assertGreater(batch['entity_mask'].shape[1], alone['entity_mask'].shape[1])
        for i, count in enumerate(batch['action_mask'].sum(1)):
            self.assertEqual(int(batch['legacy_action_features'][i, count:].count_nonzero()), 0)
        for flag in (False, True):
            model = self.model(flag).eval()
            with torch.no_grad():
                model.actor.legacy_action_projection.weight.normal_()
            with torch.inference_mode():
                expected = model.actor_logits(alone)[0]
                logits = model.actor_logits(batch)
                self.assertTrue(torch.equal(expected, logits[1, :len(expected)]))
                vectorized = model.training_logits(batch)
                torch.testing.assert_close(vectorized, logits, atol=3e-6, rtol=3e-5)
                self.assertTrue(torch.isneginf(logits[~batch['action_mask']]).all())
                self.assertTrue(torch.isneginf(vectorized[~batch['action_mask']]).all())
        invalid = {**batch, 'legacy_action_features': batch['legacy_action_features'][..., :127]}
        with self.assertRaisesRegex(ValueError, '128'):
            model.actor_logits(invalid)

    def test_counts_and_count_only_helper_do_not_allocate_weights_or_move_rng(self):
        rng = torch.get_rng_state().clone()
        for flag in (False, True):
            desc = describe_capacity(16, legacy_action_features=flag, layers=1)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            model = self.model(flag)
            rng = torch.get_rng_state().clone()
            model.critic.requires_grad_(False)
            self.assertEqual(model.parameter_counts(), {k: v for k, v in desc.items()
                if k not in ('width', 'layers', 'heads', 'legacy_action_features')})
            self.assertEqual(desc['policy_dormant_parameter_count'], 0 if flag else 128 * 16)
            self.assertEqual(desc['trainable_parameter_count'], desc['policy_active_parameter_count'])
            target = describe_capacity(24, legacy_action_features=flag, layers=1)['policy_active_parameter_count']
            self.assertEqual(nearest_policy_width(target, legacy_action_features=flag, layers=1,
                                                  maximum_width=40)['nearest']['width'], 24)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_roundtrip_sources_config_copies_boolean_and_native_free_loader(self):
        model = self.model(True, phase='ooc').eval()
        batch = self.batch()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'controlled.pt'
            save_checkpoint(checkpoint, model, model_config=model.metadata(), metadata=model.metadata())
            payload = load_checkpoint(checkpoint)
            loaded = policy_from_payload(payload).eval()
            self.assertEqual(loaded.config.phase, 'ooc')
            self.assertIs(loaded.config.legacy_action_features, True)
            self.assertTrue(torch.equal(model.actor_logits(batch), loaded.actor_logits(batch)))
            self.assertTrue(torch.equal(model.critic_values(batch), loaded.critic_values(batch)))
            source_keys = [k for k in model.metadata() if k.endswith('_sha256') or k in
                           ('kind', 'encoder_version', 'entity_encoder_version', 'vocabulary_size')]
            for key in source_keys:
                bad = deepcopy(payload)
                bad['metadata'].pop(key)
                bad['model_state'] = None
                with self.subTest(key=key), patch.object(ControlledEntityActorCritic, 'load_state_dict') as load:
                    with self.assertRaisesRegex(ValueError, key):
                        policy_from_payload(bad)
                    load.assert_not_called()
            for where in ('metadata', 'model_config'):
                bad = deepcopy(payload)
                bad[where]['model_config']['legacy_action_features'] = False
                with self.assertRaisesRegex(ValueError, 'model_config'):
                    policy_from_payload(bad)
            code = """import sys, torch
from sts_policy.controlled_entity import policy_from_payload
policy_from_payload(torch.load(sys.argv[1], map_location='cpu', weights_only=False))
assert not any(n.startswith('sts_rl') or n == 'slaythespire' for n in sys.modules)
"""
            subprocess.run([sys.executable, '-B', '-c', code, str(checkpoint)], check=True, timeout=20,
                           env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')})
        for flag in (1, 0, 'false', None):
            with self.assertRaisesRegex(ValueError, 'boolean'):
                self.model(flag)
        with self.assertRaisesRegex(ValueError, 'phase'):
            self.model(phase='shared')


if __name__ == '__main__':
    unittest.main()
