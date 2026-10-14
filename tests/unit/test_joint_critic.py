import hashlib
import math
import unittest

try:
    import numpy as np
    import torch

    from sts_rl.joint_sparse import JointConfig, WinCritics, critic_update
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch"}:
        raise
    torch = None


def held(seed):
    return int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16) % 5 == 0


@unittest.skipIf(torch is None, "Optional RL runtime absent")
class JointCriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.critic = WinCritics()
        self.cfg = JointConfig(out="/tmp/unused", checkpoint="/tmp/unused")
        self.fit = [seed for seed in range(100, 200) if not held(seed)]
        self.hold = next(seed for seed in range(100, 200) if held(seed))

    def rows(self, seed, label, count):
        result = []
        for phase in ("combat", "ooc"):
            for _ in range(count):
                state = np.zeros(256, np.float32)
                state[7] = phase == "combat"
                state[4] = 1 / 3
                result.append(
                    {
                        "seed": seed,
                        "return": label,
                        "phase": phase,
                        "act": 1,
                        "state": state,
                        "actions": np.zeros((2, 128), np.float32),
                    }
                )
        return result

    def test_world_weighting_prevents_long_worlds_dominating(self):
        rows = self.rows(self.fit[0], 1.0, 1) + self.rows(self.fit[1], 0.0, 64)
        opt = torch.optim.AdamW(self.critic.parameters(), lr=0.0003, weight_decay=0.01)
        result = critic_update(self.critic, opt, rows, self.cfg)
        self.assertAlmostEqual(
            result["loss"], -math.log(0.03) - math.log(0.97), places=5
        )
        self.assertEqual(
            result["fit_states"], 18
        )  # one winning state, eight losing states per head

    def test_whole_held_world_labels_cannot_enter_gradient(self):
        rows = self.rows(self.fit[0], 0.0, 8) + self.rows(self.hold, 1.0, 64)
        opt = torch.optim.AdamW(self.critic.parameters(), lr=0.0003)
        result = critic_update(self.critic, opt, rows, self.cfg)
        self.assertAlmostEqual(result["loss"], -2 * math.log(0.97), places=5)
        for phase in ("combat", "ooc"):
            self.assertGreater(self.critic.heads[phase][-1].bias.grad.item(), 0.0)
            self.assertEqual(result["held_calibration"][phase + ":act0"]["worlds"], 1)

    def test_zero_output_initialization_only_blocks_trunk_on_first_step(self):
        rows = self.rows(self.fit[0], 1.0, 8) + self.rows(self.fit[1], 0.0, 8)
        opt = torch.optim.AdamW(self.critic.parameters(), lr=0.0003, weight_decay=0.0)
        critic_update(self.critic, opt, rows, self.cfg)
        for phase in ("combat", "ooc"):
            self.assertEqual(self.critic.heads[phase][0].weight.grad.norm().item(), 0.0)
        critic_update(self.critic, opt, rows, self.cfg)
        for phase in ("combat", "ooc"):
            gradient = self.critic.heads[phase][0].weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
