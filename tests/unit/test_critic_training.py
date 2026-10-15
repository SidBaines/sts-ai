import unittest

try:
    import numpy as np
    import torch

    from sts_rl import critic_training as ct
    from sts_rl.joint_sparse import JointConfig, WinCritics, critic_update
except ModuleNotFoundError as e:
    if e.name not in {"torch", "numpy"}:
        raise
    torch = None


@unittest.skipIf(torch is None, "Optional RL runtime absent")
class CriticTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_minibatches_really_step_each_phase_and_preserve_collected_advantages(self):
        model = WinCritics()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
        rows = []
        seeds = [n for n in range(200) if not ct.held_world(n)][:16]
        for phase in ct.PHASES:
            for seed in seeds:
                for i in range(8):
                    state = np.zeros(256, np.float32)
                    state[7] = phase == "combat"
                    state[4] = 1 / 3
                    state[0] = seed % 2
                    rows.append(
                        {
                            "state": state,
                            "actions": np.zeros((2, 128), np.float32),
                            "return": float(seed % 2),
                            "phase": phase,
                            "seed": seed,
                            "act": 1,
                            "value": 0.03,
                            "advantage": float(seed % 2) - 0.03,
                        }
                    )
        old = [(r["value"], r["advantage"]) for r in rows]
        cfg = JointConfig(
            out="/tmp/unused",
            checkpoint="/tmp/unused",
            critic_minibatch=32,
            critic_init="unit-fixture",
        )
        history = []
        update = critic_update(model, opt, rows, cfg, history, 1)
        self.assertEqual(update["optimizer_steps"], 8)
        for p in ct.PHASES:
            self.assertEqual(update["phases"][p]["steps"], 4)
        self.assertEqual(old, [(r["value"], r["advantage"]) for r in rows])

    def sample(self, **changes):
        m = {
            "worlds": 128,
            "positive_worlds": 16,
            "predicted_std": 0.05,
            "predicted_mean": 0.125,
            "return_mean": 0.125,
            "brier_delta": -0.002,
            "brier_delta_se_world": 0.001,
            "bce": 0.3,
            "explained_variance": 0.05,
        }
        m.update(changes)
        return {"combat:act0": m}

    def test_flat_head_emits_actionable_warning(self):
        health = ct.health(self.sample(predicted_std=0.0))
        self.assertIn(
            "flat_critic_predictions", [x["reason"] for x in health["urgent"]]
        )

    def test_bad_calibration_and_overfitting_are_visible(self):
        health = ct.health(
            self.sample(predicted_mean=0.7, brier_delta=0.2),
            {"combat:act0": {"bce": 0.01}},
        )
        reasons = [x["reason"] for x in health["warnings"]]
        for reason in [
            "critic_calibration_gap",
            "critic_worse_than_constant",
            "critic_generalisation_gap",
        ]:
            self.assertIn(reason, reasons)

    def test_small_sample_does_not_trigger_false_failure(self):
        health = ct.health(self.sample(worlds=20, positive_worlds=1, predicted_std=0.0))
        self.assertFalse(health["urgent"])
        self.assertEqual(
            health["warnings"][0]["reason"], "insufficient_critic_validation_support"
        )

    def test_subset_and_combine_preserve_world_routing(self):
        data = {
            "x": torch.zeros(4, 384),
            "y": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "w": torch.ones(4),
            "seed": np.array([1, 1, 2, 2]),
            "phase": np.array([0, 1, 0, 1]),
            "act": np.ones(4),
            "held": np.array([False, False, True, True]),
        }
        one = ct.subset(data, data["held"])
        both = ct.combine([one, one])
        self.assertTrue(torch.equal(both["y"], torch.tensor([0.0, 1.0, 0.0, 1.0])))
        self.assertEqual(set(both["seed"]), {2})


if __name__ == "__main__":
    unittest.main()
