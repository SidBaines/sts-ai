from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sts_policy.encoding import encode_observation
from sts_policy.losses import PPOConfig
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import cached_public_collator, collate_public, forward_policy
from sts_policy.rollout_training import collect_rollouts, update_ppo
from tests.unit.test_nonllm_training import FakeTask, FakeWorker, TinyModel, tiny_collate, tiny_forward


class RolloutEncodingReuseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def collect_pair(self, deterministic):
        torch.manual_seed(91)
        model = CandidateActorCritic(size='small', context='none')
        results = []
        for reuse in (False, True):
            torch.manual_seed(17)
            results.append(collect_rollouts(model, [FakeWorker(), FakeWorker()],
                [FakeTask(1), FakeTask(2), FakeTask(3)], collate=collate_public, forward=forward_policy,
                deterministic=deterministic, teacher_simulations=1, deferred_bootstrap=True,
                reuse_encodings=reuse))
        return model, results

    def test_cached_sampling_features_preserve_actions_gae_and_fault_quarantine(self):
        for deterministic in (False, True):
            _, (old, new) = self.collect_pair(deterministic)
            self.assertEqual(old['transitions'], new['transitions'])
            self.assertEqual(old['episodes'], new['episodes'])
            self.assertEqual(old['teacher_records'], new['teacher_records'])
            semantic_faults = lambda rows: [{k: v for k, v in row.items() if k != 'wait_seconds'} for row in rows]
            self.assertEqual(semantic_faults(old['faults']), semantic_faults(new['faults']))
            self.assertEqual(new['retained_encodings'], len(new['transitions']))
            self.assertEqual(new['excluded_transitions'], 1)
            self.assertTrue(all(t.world_seed != 2 for t in new['transitions']))
            self.assertEqual(old['retained_encoding_bytes'], 0)
            for transition in new['transitions']:
                fresh = encode_observation(transition.observation)
                np.testing.assert_array_equal(fresh.state_features, transition.encoded_observation.state_features)
                np.testing.assert_array_equal(fresh.action_features, transition.encoded_observation.action_features)
                self.assertEqual(transition.encoding_observation_id, id(transition.observation))
                self.assertNotIn('encoded_observation', transition.observation)

    def test_cached_ppo_losses_gradients_optimizer_and_weights_equal(self):
        model, (old, new) = self.collect_pair(False)
        models = [deepcopy(model), deepcopy(model)]
        config = PPOConfig(epochs=2, minibatch_size=3, target_kl=1.)
        reports = []
        for current, result in zip(models, (old, new)):
            reports.append(update_ppo(current, result['transitions'], collate=collate_public,
                                     forward=forward_policy, config=config, seed=13))
        self.assertGreater(reports[0]['steps'], 0)
        for key in ['steps', 'samples', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'ratio_mean', 'clip_fraction']:
            self.assertEqual(reports[0][key], reports[1][key])
        self.assertEqual(reports[0]['reused_encoding_count'], 0)
        self.assertEqual(reports[1]['reused_encoding_count'], len(new['transitions']))
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for a, b in zip(reports[0]['optimizer'].state.values(), reports[1]['optimizer'].state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_observation_identity_and_menu_alignment_fail_closed(self):
        model, (_, new) = self.collect_pair(True)
        transitions = new['transitions']
        transitions[0].observation = deepcopy(transitions[0].observation)
        with self.assertRaisesRegex(ValueError, 'observation identity'):
            update_ppo(model, transitions, collate=collate_public, forward=forward_policy)
        obs = transitions[1].observation
        with self.assertRaisesRegex(ValueError, 'align'):
            cached_public_collator([obs], preencoded=[])
        wrong = encode_observation(obs)
        wrong.action_indices[0] = 999
        with self.assertRaisesRegex(ValueError, 'legal menu identity'):
            cached_public_collator([obs], preencoded=[wrong])

    def test_cache_stays_outside_serialized_round_metrics_and_teacher_rows(self):
        _, (_, result) = self.collect_pair(True)
        encodings = [t.encoded_observation for t in result['transitions']]
        expected_bytes = sum(e.state_features.nbytes + e.action_features.nbytes + e.action_indices.nbytes
                             for e in encodings)
        self.assertEqual(result['retained_encoding_bytes'], expected_bytes)
        round_metrics = {key: value for key, value in result.items()
                         if key not in ('transitions', 'teacher_records')}
        serialized = json.dumps({'round': round_metrics, 'labels': result['teacher_records']})
        self.assertNotIn('encoded_observation', serialized)
        self.assertNotIn('encoding_observation_id', serialized)

    def test_generic_collator_never_changes_or_retains_public_encodings(self):
        with patch('sts_policy.model_agent.encode_observation', side_effect=AssertionError('unexpected public encoder')):
            result = collect_rollouts(TinyModel(), [FakeWorker()], [FakeTask(1)],
                collate=tiny_collate, forward=tiny_forward, reuse_encodings=True)
        self.assertEqual(result['retained_encodings'], 0)
        self.assertEqual(result['retained_encoding_bytes'], 0)
        self.assertTrue(all(t.encoded_observation is None for t in result['transitions']))


if __name__ == '__main__':
    unittest.main()
