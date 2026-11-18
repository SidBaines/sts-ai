import unittest

import torch

from sts_policy.encoding import ACTION_DIM, STATE_DIM
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import Act3BossActorCritic, CompositeActorCritic, policy_from_config, route_batch


class BossPolicyTests(unittest.TestCase):
    def test_mixed_batch_uses_only_public_act_and_room(self):
        torch.manual_seed(9)
        heads = [CandidateActorCritic(size="small") for _ in range(3)]
        baseline = CompositeActorCritic(*heads[:2]).eval()
        policy = Act3BossActorCritic(*heads).eval()
        observations = [dict(phase=phase, state=dict(act=act, room=room, floor=floor))
                        for phase, act, room, floor in [
                            ("combat", 3, "Room.BOSS", 44), ("combat", 3, "BOSS", 50),
                            ("combat", 2, "Room.BOSS", 33), ("combat", 3, "MONSTER", 49),
                            ("combat", 3, "ELITE", 46), ("reward", 3, "BOSS", 50)]]
        state = torch.randn(6, STATE_DIM)
        state[:, 7] = torch.tensor([1, 1, 1, 1, 1, 0])
        batch = state, torch.randn(6, 4, ACTION_DIM), torch.ones(6, 4, dtype=torch.bool)
        routed = route_batch(policy, batch, observations)
        self.assertEqual(routed[-1].tolist(), [True, True, False, False, False, False])
        restored = policy_from_config(policy.metadata()).eval()
        restored.load_state_dict(policy.state_dict())
        with torch.no_grad():
            logits, values = restored(*routed)
            ordinary = baseline(*batch)
            special = heads[2](*(x[:2] for x in batch))
            self.assertTrue(torch.equal(logits[2:], ordinary[0][2:]))
            self.assertTrue(torch.equal(values[2:], ordinary[1][2:]))
            self.assertTrue(torch.equal(logits[:2], special[0]))
            self.assertTrue(torch.equal(values[:2], special[1]))
            self.assertTrue(torch.equal(logits, restored.actor_logits(*routed)))
            with self.assertRaisesRegex(ValueError, "boss-room"):
                restored(*batch)
            for first, second in zip(restored(*(x[-1:] for x in batch)), heads[1](*(x[-1:] for x in batch))):
                self.assertTrue(torch.equal(first, second))
