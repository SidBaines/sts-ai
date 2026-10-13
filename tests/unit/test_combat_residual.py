from __future__ import annotations

from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import torch

from sts_policy import entities
from sts_policy.entity_model import ResidualOOCPolicy
from sts_policy.combat_residual import CombatResidualPolicy, equivalent_positions, equivalence_loss, save_policy, load_policy
from tests.unit.test_public_entities import observation


class CombatResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = CombatResidualPolicy(ResidualOOCPolicy({"size": "small", "context": "none"}, width=32, embedding=4),
                                          width=32, embedding=4)

    def test_initial_behavior_is_equal_and_only_combat_parameters_move(self):
        observations = [observation(), observation(False)]
        batch = entities.collate_public(observations)
        logits, values = self.model(*batch)
        before, baseline_values = self.model.base(*batch)
        torch.testing.assert_close(logits, before)
        torch.testing.assert_close(values, baseline_values)
        original = {k:v.clone() for k,v in self.model.base.state_dict().items()}
        optimizer = torch.optim.Adam(self.model.combat_parameters(), lr=.01)
        for _ in range(10):
            logits, _ = self.model(*batch)
            loss = -torch.log_softmax(logits, -1)[0, 1]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for key, value in self.model.base.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))
        after, _ = self.model(*batch)
        torch.testing.assert_close(after[1], before[1])
        self.assertFalse(torch.allclose(after[0], before[0]))

    def test_equivalent_copies_share_targets_but_free_play_does_not(self):
        obs = observation()
        encoded = entities.encode_observation(obs)
        self.assertEqual(equivalent_positions(encoded, 0).tolist(), [True, False, False])
        obs["state"]["combat"]["cards"]["hand"][0]["free_to_play_once"] = False
        obs["legal_actions"][0]["public"].update(energy_cost=1, free_to_play=False)
        obs["legal_actions"][0]["description"] = "play Strike [Attack] (cost 1) -> ACID_SLIME_L (deal 6) [hand 2]"
        encoded = entities.encode_observation(obs)
        self.assertEqual(equivalent_positions(encoded, 0).tolist(), [True, True, False])
        # Target probability depends on the group mass, not which identical
        # source ordinal was chosen by a privileged native trajectory.
        logits = torch.tensor([[0., 0., 0.]], requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        targets = torch.tensor([[True, True, False]])
        loss = equivalence_loss(logits, mask, targets)
        self.assertAlmostEqual(float(loss.detach()), -__import__('math').log(2/3), places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0]), float(logits.grad[0, 1]))

    def test_checkpoint_roundtrip_and_model_identity(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory)/"model.pt")
            save_policy(path, self.model, metadata={})
            restored, payload = load_policy(path)
            for key,value in self.model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            payload["metadata"]["combat_policy_code_identity"] = {}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "implementation"):
                load_policy(path)


if __name__ == "__main__":
    unittest.main()
