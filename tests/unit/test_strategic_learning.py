from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import torch

from sts_policy import entities
from sts_policy.entity_model import ResidualOOCPolicy
from sts_policy.strategic_collection import StrategicTransition
from sts_policy.strategic_learning import StrategicConfig, optimizers, update, drift
from tests.unit.test_public_entities import observation


class ProtectedLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        torch.set_num_threads(1)
        self.model = ResidualOOCPolicy({"size": "small", "context": "none"}, width=32, embedding=4)
        self.obs = observation(False)
        self.obs["state"].pop("combat")
        self.obs["legal_actions"] = [
            {"index": 0, "description": "event option 0: Obtain 100 gold.", "public": {"kind": "out_of_combat"}},
            {"index": 1, "description": "event option 1: Obtain 250 gold.", "public": {"kind": "out_of_combat"}}]

    def rows(self):
        with torch.no_grad():
            logits, value = self.model(*entities.collate_public([self.obs]))
        probabilities = logits.softmax(-1)[0].numpy()
        return [StrategicTransition(self.obs, None, position, position, float(np.log(probabilities[position])),
            float(value[0]), 0., {}, 1, True, False, 0., "task", 5, True,
            behavior_distribution=probabilities.copy(), advantage=-1 if position == 0 else 1,
            returns=2., completed_return=2.) for position in (0, 1)]

    def test_choice_actor_learns_numeric_signal_while_prior_stays_bitwise_fixed(self):
        config = StrategicConfig(actor_lr=.005, critic_lr=.003, batch_size=2, epochs=30,
                                 clip_ratio=.9, rehearsal_weight=0., post_update_kl_stop=100.)
        actor, critic = optimizers(self.model, config)
        prior = {key: value.clone() for key, value in self.model.prior.state_dict().items()}
        rows = self.rows()
        with torch.no_grad():
            initial = self.model(*entities.collate_public([self.obs]))[0].softmax(-1)[0, 1].item()
        metrics = update(self.model, rows, rehearsal=[], config=config, actor_optimizer=actor,
                         critic_optimizer=critic, seed=3, device="cpu")
        with torch.no_grad():
            final = self.model(*entities.collate_public([self.obs]))[0].softmax(-1)[0, 1].item()
        self.assertGreater(final, initial+.1)
        for key, value in self.model.prior.state_dict().items():
            self.assertTrue(torch.equal(value, prior[key]))
        self.assertEqual(metrics["actor_exposures"], 60)
        self.assertEqual(metrics["unique_actor_rows_used"], 2)

    def test_critic_completed_return_fit_improves_without_changing_actor(self):
        config = StrategicConfig(critic_lr=.01, batch_size=2, epochs=20)
        actor, critic = optimizers(self.model, config)
        actor_before = [parameter.clone() for parameter in self.model.actor_parameters()]
        metrics = update(self.model, self.rows(), rehearsal=[], config=config,
            actor_optimizer=actor, critic_optimizer=critic, seed=8, device="cpu", critic_only=True,
            completed_targets=True)
        self.assertLess(metrics["calibration_after"]["overall"]["mse"], metrics["calibration_before"]["overall"]["mse"]*.1)
        self.assertEqual(metrics["actor_steps"], 0)
        for before, after in zip(actor_before, self.model.actor_parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_forced_rows_get_no_actor_exposure_and_bootstrap_critic_targets(self):
        rows = self.rows()
        forced = deepcopy(rows[0])
        forced.observation = deepcopy(self.obs)
        forced.observation["legal_actions"] = forced.observation["legal_actions"][:1]
        forced.actor_choice, forced.encoded, forced.advantage = False, None, 10000.
        rows.append(forced)
        config = StrategicConfig(batch_size=32)
        actor, critic = optimizers(self.model, config)
        metrics = update(self.model, rows, rehearsal=[], config=config,
            actor_optimizer=actor, critic_optimizer=critic, seed=1, device="cpu")
        self.assertEqual(metrics["actor_exposures"], 2)
        self.assertEqual(metrics["critic_exposures"], 3)
        self.assertEqual(metrics["advantage_before_normalization"], {"mean": 0., "std": 1.})
        with self.assertRaises(ValueError):
            StrategicConfig(actor_lr=float("nan"))

    def test_final_epoch_kl_guard_stops_further_epochs(self):
        config = StrategicConfig(actor_lr=.005, batch_size=2, epochs=30,
                                 rehearsal_weight=0., post_update_kl_stop=1e-7)
        actor, critic = optimizers(self.model, config)
        metrics = update(self.model, self.rows(), rehearsal=[], config=config,
            actor_optimizer=actor, critic_optimizer=critic, seed=3, device="cpu")
        self.assertEqual(metrics["stop_reason"], "post_epoch_behavior_kl")
        self.assertLess(metrics["actor_exposures"], 60)
        self.assertGreater(metrics["buffer_drift"]["behavior_kl_post_update"]["mean"], 1e-7)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS required for subnormal regression")
    def test_mps_subnormal_behavior_probabilities_have_finite_final_kl(self):
        encoded = entities.encode_observation(self.obs)
        metrics = drift(self.model, [encoded], device="mps",
                        behavior=[np.asarray([1e-44, 1.], dtype=np.float32)])
        value = metrics["behavior_kl_post_update"]["mean"]
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, .5)

    def test_nonfinite_behavior_kl_fails_closed(self):
        encoded = entities.encode_observation(self.obs)
        with self.assertRaises(FloatingPointError):
            drift(self.model, [encoded], behavior=[np.asarray([float("nan"), 1.])])


if __name__ == "__main__":
    unittest.main()
