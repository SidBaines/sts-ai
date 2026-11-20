import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from sts_rl.combat_distillation import Menus, calibrate, losses
from sts_rl.distillation_collection import SampleGame, pack


class DistillationTests(unittest.TestCase):
    def test_extra_worlds_keep_teacher_scores_and_actor_choices_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = [Path(directory) / name for name in ("old", "extra")]
            for seed, folder in enumerate(folders):
                folder.mkdir()
                fields = dict(
                    states=np.zeros((1, 256), np.float32),
                    actions=np.zeros((2, 128), np.float32),
                    lengths=np.array([2]),
                    scores=np.array([0.0, 2.0], np.float32),
                    seeds=np.array([seed]),
                    chosen=np.array([0]),
                )
                np.savez(
                    folder / "train-000.npz",
                    **{r + "_" + k: v for r in ("ordinary", "boss") for k, v in fields.items()},
                )
            data = Menus(folders[0], "train", extra=(folders[1],))
            self.assertEqual(data.seeds.tolist(), [0, 0, 1, 1])
            self.assertEqual(data.chosen.tolist(), [0] * 4)
            batch = data.batch(np.array([0, 3]), "cpu")
            scale = calibrate(data, 1)
            head = SimpleNamespace(actor_logits=lambda s, a, m: batch[3])
            divergence, regret = losses(head, batch, torch.tensor(scale))
            self.assertTrue(torch.allclose(divergence, torch.zeros(2), atol=1e-6))
            self.assertTrue(torch.equal(regret, torch.zeros(2)))

    def test_sink_excludes_incomplete_games_and_empty_roles_are_representable(self):
        reservoir = SampleGame(2, 1)
        for i in range(7):
            obs = {"phase": "combat", "state": {"act": 3, "room": "BOSS"}, "legal_actions": [{}, {}]}
            reservoir.record(
                SimpleNamespace(world_seed=2, observation=obs, action_position=1, encoded_observation=i)
            )
        self.assertEqual(len(reservoir.finish({"terminated": True, "truncated": False})[1]), 1)
        self.assertEqual(reservoir.finish({"terminated": True, "truncated": True}), ([], []))
        self.assertEqual(pack([], None)["actions"].shape, (0, 128))
