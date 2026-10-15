from __future__ import annotations

from copy import deepcopy
import io
import time
import unittest
from unittest.mock import patch

import torch

from sts_policy.entity_encoding import ENCODER_SHA256, collate_observations, encode_observation
from sts_policy.entity_model import EntityActorCritic, policy_from_payload
from tests.unit.test_entity_encoding import combat_observation, ooc_observation


class EntityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.e = encode_observation(combat_observation())
        self.batch = collate_observations([self.e])

    def model(self, **kwargs):
        return EntityActorCritic(width=32, layers=1, heads=4, **kwargs)

    def test_shapes_masks_phase_and_metadata(self):
        other = ooc_observation()
        other['legal_actions'] = other['legal_actions'][:1]
        batch = collate_observations([self.e, encode_observation(other)])
        model = self.model()
        logits, values = model(batch)
        self.assertEqual(logits.shape, (2, 8))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.isfinite(logits[batch['action_mask']]).all())
        self.assertTrue(torch.isneginf(logits[~batch['action_mask']]).all())
        self.assertTrue(torch.isfinite(values).all())
        meta = model.metadata()
        self.assertEqual(meta['kind'], 'entity_fresh_v1')
        self.assertEqual(meta['encoder_sha256'], ENCODER_SHA256)
        self.assertEqual(meta['model_config']['phase'], 'combat')
        self.assertEqual(self.model(phase='ooc').metadata()['model_config']['phase'], 'ooc')
        self.assertEqual(len(meta['vocabulary_sha256']), 64)
        self.assertEqual(meta['parameter_count'], sum(p.numel() for p in model.parameters()))
        with self.assertRaises(ValueError):
            self.model(phase='shared')

    def test_round_trip_exact_cpu_and_identity_rejection(self):
        model = self.model().eval()
        expected = model(self.batch)
        buffer = io.BytesIO()
        torch.save({'model_state': model.state_dict(), 'metadata': model.metadata()}, buffer)
        buffer.seek(0)
        payload = torch.load(buffer, map_location='cpu', weights_only=True)
        loaded = policy_from_payload(payload).eval()
        actual = loaded(self.batch)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        for key in ('kind', 'vocabulary_sha256', 'encoder_version', 'encoder_sha256'):
            invalid = deepcopy(payload)
            invalid['metadata'][key] = 'wrong'
            with self.assertRaisesRegex(ValueError, key):
                policy_from_payload(invalid)

    def test_encoder_sha_missing_or_mismatched_rejected_before_weight_load(self):
        metadata = self.model().metadata()
        for identity in (None, '0' * 64, ''):
            invalid_metadata = deepcopy(metadata)
            if identity is None:
                del invalid_metadata['encoder_sha256']
            else:
                invalid_metadata['encoder_sha256'] = identity
            # Invalid weights must never be inspected for an incompatible identity.
            payload = {'metadata': invalid_metadata, 'model_state': None}
            with self.subTest(identity=identity), patch.object(EntityActorCritic, 'load_state_dict') as load:
                with self.assertRaisesRegex(ValueError, 'encoder_sha256'):
                    policy_from_payload(payload)
                load.assert_not_called()

    def test_disjoint_parameters_critic_update_and_no_critic_actor_call(self):
        model = self.model()
        actor = list(model.actor_parameters())
        critic = list(model.critic_parameters())
        self.assertFalse({id(p) for p in actor} & {id(p) for p in critic})
        self.assertEqual({id(p) for p in actor + critic}, {id(p) for p in model.parameters()})
        snapshot = [p.detach().clone() for p in actor]
        before = model.actor_logits(self.batch).detach()
        opt = torch.optim.Adam(critic, lr=.01)
        for _ in range(3):
            opt.zero_grad()
            (model.critic_values(self.batch) - 3).square().mean().backward()
            opt.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(snapshot, actor)))
        self.assertTrue(torch.equal(before, model.actor_logits(self.batch)))
        self.assertTrue(all(p.grad is None for p in actor))
        with patch.object(model.critic, 'forward', side_effect=AssertionError('critic called')):
            model.actor_logits(self.batch)
        with patch.object(model.actor, 'forward', side_effect=AssertionError('actor called')):
            model.critic_values(self.batch)

    def test_padding_batch_composition_exact_logits_and_greedy(self):
        model = self.model().eval()
        large = combat_observation()
        large['state']['combat']['recent_actions'] *= 3
        large['state']['combat']['cards']['draw'] *= 2
        large['legal_actions'] = [{'index': 0, 'description': 'exhaust: hand [0] Strike [Attack] (cost 1); hand [1] Strike [Attack] (cost 1, free once); hand [2] Strike [Attack] (cost 0)'}] + large['legal_actions'] * 2
        small = ooc_observation()
        small['legal_actions'] = small['legal_actions'][:1]
        alone = model.actor_logits(self.batch)[0]
        combined = collate_observations([encode_observation(small), self.e, encode_observation(large)])
        mixed = model.actor_logits(combined)[1, :len(alone)]
        self.assertTrue(torch.equal(alone, mixed))
        self.assertEqual(alone.argmax(), mixed.argmax())
        # No full-state position embedding: pile permutation is exact after
        # canonical multiset aggregation, regardless of input order.
        permuted = combat_observation()
        permuted['state']['combat']['cards']['draw'].reverse()
        self.assertTrue(torch.equal(alone, model.actor_logits(collate_observations([encode_observation(permuted)]))[0]))

    def test_menu_permutation_equivariance(self):
        model = self.model().eval()
        obs = combat_observation()
        permutation = [7, 3, 2, 0, 5, 1, 6, 4]
        obs['legal_actions'] = [obs['legal_actions'][i] for i in permutation]
        actual = model.actor_logits(collate_observations([encode_observation(obs)]))[0]
        self.assertTrue(torch.equal(model.actor_logits(self.batch)[0, permutation], actual))

    def test_default_inference_fused_padding_invariance_and_round_trip_exact_cpu(self):
        model = EntityActorCritic().eval()  # Default 128-wide, two-layer towers.
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertTrue(torch.backends.mha.get_fastpath_enabled())
        large = combat_observation()
        large['state']['combat']['recent_actions'] *= 3
        large['state']['combat']['player']['powers']['Vulnerable'] = 1
        for slot in range(6, 10):
            extra = deepcopy(large['state']['combat']['cards']['hand'][0])
            extra['index'] = slot
            large['state']['combat']['cards']['hand'].append(extra)
        labels = ['exhaust: hand [0] Strike [Attack] (cost 1); hand [1] Strike [Attack] (cost 1, free once); hand [2] Strike [Attack] (cost 0)']
        labels += [a['description'] for a in large['legal_actions']] * 2
        large['legal_actions'] = [{'index': i, 'description': label} for i, label in enumerate(labels)]
        small = ooc_observation()
        small['legal_actions'] = small['legal_actions'][:1]
        combined = collate_observations([encode_observation(small), self.e, encode_observation(large)])
        for prefix in ('entity', 'action', 'power', 'recent'):
            self.assertGreater(combined[prefix + '_mask'].shape[1], self.batch[prefix + '_mask'].shape[1])
        self.assertGreater(combined['action_source_mask'].shape[2], self.batch['action_source_mask'].shape[2])

        # Observe the actual fused entry point; eval alone with gradients enabled
        # exercises a different TransformerEncoderLayer path.
        with torch.inference_mode(), patch('torch._transformer_encoder_layer_fwd',
                                          wraps=torch._transformer_encoder_layer_fwd) as fused:
            expected_logits, expected_values = model(self.batch)
            self.assertEqual(fused.call_count, 4)  # Actor + critic, two layers each.
            mixed_logits, mixed_values = model(combined)
            self.assertEqual(fused.call_count, 16)
            self.assertTrue(torch.equal(expected_logits[0], mixed_logits[1, :expected_logits.shape[1]]))
            self.assertTrue(torch.equal(expected_values[0], mixed_values[1]))
            self.assertEqual(expected_logits[0].argmax(), mixed_logits[1].argmax())
            self.assertTrue(torch.isneginf(mixed_logits[~combined['action_mask']]).all())

            buffer = io.BytesIO()
            torch.save({'model_state': model.state_dict(), 'metadata': model.metadata()}, buffer)
            buffer.seek(0)
            loaded = policy_from_payload(torch.load(buffer, map_location='cpu', weights_only=True)).eval()
            actual_logits, actual_values = loaded(combined)
            self.assertEqual(fused.call_count, 28)
            self.assertTrue(torch.equal(mixed_logits, actual_logits))
            self.assertTrue(torch.equal(mixed_values, actual_values))

    def test_ordered_history_and_owner_specific_powers_affect_logits(self):
        model = self.model().eval()
        base = model.actor_logits(self.batch)
        obs = combat_observation()
        obs['state']['combat']['recent_actions'].reverse()
        reversed_history = model.actor_logits(collate_observations([encode_observation(obs)]))
        self.assertFalse(torch.equal(base, reversed_history))
        obs = combat_observation()
        enemies = obs['state']['combat']['enemies']
        enemies[0]['powers'], enemies[1]['powers'] = enemies[1]['powers'], enemies[0]['powers']
        different_owners = model.actor_logits(collate_observations([encode_observation(obs)]))
        self.assertFalse(torch.equal(base, different_owners))
        # Changing assignments of amounts to powers with the same total must
        # remain observable, even on one owner.
        obs = combat_observation()
        obs['state']['combat']['player']['powers'].update(Strength=1, Weak=2)
        self.assertFalse(torch.equal(base, model.actor_logits(collate_observations([encode_observation(obs)]))))

    def test_multiselection_membership_affects_score(self):
        model = self.model().eval()
        obs = combat_observation()
        obs['legal_actions'] = [
            {'index': 0, 'description': 'exhaust: hand [0] Strike [Attack] (cost 1); hand [1] Strike [Attack] (cost 1, free once)'},
            {'index': 1, 'description': 'exhaust: hand [0] Strike [Attack] (cost 1); hand [3] Strike [Attack] (cost 1, retained)'}]
        logits = model.actor_logits(collate_observations([encode_observation(obs)]))[0]
        self.assertNotEqual(logits[0], logits[1])

    def test_critic_observes_unowned_offers(self):
        model = self.model(phase='ooc').eval()
        obs = ooc_observation()
        obs['legal_actions'] = [{'index': 0, 'description': 'buy card Inflame [Power, Uncommon] for 75g'}]
        first = collate_observations([encode_observation(obs)])
        obs['legal_actions'][0]['description'] = 'buy card Bash [Attack, Basic] for 75g'
        second = collate_observations([encode_observation(obs)])
        self.assertTrue(torch.equal(first['entity_ids'], second['entity_ids']))
        self.assertFalse(torch.equal(model.critic_values(first), model.critic_values(second)))

    def test_invalid_masks_and_references_fail_clearly(self):
        model = self.model()
        invalid = {k: v.clone() for k, v in self.batch.items()}
        invalid['entity_mask'][0, 1] = False
        with self.assertRaisesRegex(ValueError, 'prefix'):
            model.actor_logits(invalid)
        invalid = {k: v.clone() for k, v in self.batch.items()}
        invalid['action_target'][0, 0] = 10000
        with self.assertRaisesRegex(ValueError, 'out-of-range'):
            model.actor_logits(invalid)

    def test_tiny_soft_target_overfit(self):
        started = time.perf_counter()
        model = self.model()
        obs = ooc_observation()
        obs['legal_actions'] = [{'index': i, 'description': s} for i, s in enumerate([
            'take card Bash [Attack, Basic]', 'take card Inflame [Power, Uncommon]', 'skip card reward'])]
        batch = collate_observations([encode_observation(obs)])
        target = torch.tensor([[.15, .7, .15]])
        optimizer = torch.optim.Adam(model.actor_parameters(), lr=.005)
        initial = None
        for _ in range(80):
            optimizer.zero_grad()
            loss = torch.nn.functional.kl_div(model.actor_logits(batch).log_softmax(-1), target, reduction='batchmean')
            if initial is None:
                initial = loss.item()
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.kl_div(model.actor_logits(batch).log_softmax(-1), target, reduction='batchmean').item()
        self.assertLess(final, .005)
        self.assertLess(final, initial * .1)
        self.assertLess(time.perf_counter() - started, 30)
        self.assertTrue(all(p.grad is None for p in model.critic_parameters()))

    def test_default_size_bounded_cpu_benchmark(self):
        model = EntityActorCritic().eval()
        with torch.no_grad():
            model(self.batch)  # bounded warmup
            started = time.perf_counter()
            for _ in range(5):
                model.actor_logits(self.batch)
            actor_ms = (time.perf_counter() - started) * 1000 / 5
            started = time.perf_counter()
            for _ in range(5):
                model(self.batch)
            full_ms = (time.perf_counter() - started) * 1000 / 5
        print(f'B1 CPU threads=1 entities={len(self.e.entity_ids)} actions={len(self.e.action_ids)} '
              f'parameters={model.metadata()["parameter_count"]} actor_ms={actor_ms:.3f} full_ms={full_ms:.3f}', flush=True)
        self.assertFalse(model.actor.tower.attention.enable_nested_tensor)
        self.assertLess(actor_ms, 5000)  # only catches a gross accidental hang


if __name__ == '__main__':
    unittest.main()
