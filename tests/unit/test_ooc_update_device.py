from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from sts_policy.encoding import encode_observation
from sts_policy.losses import PPOConfig, masked_distribution
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic, collate_public
from sts_policy.rollout_training import Transition, update_ppo
from sts_rl import ooc_trainer
from sts_rl.common import phase_forward
from sts_rl.ooc_trainer import OOCTrainConfig, ReturnNormalizer, is_strategic_menu, ooc_forward
from tests.unit.test_sts_rl_menus import menu


class OOCUpdateDeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        checkpoint = self.root / "input.pt"
        checkpoint.write_bytes(b"synthetic checkpoint; load_policy is mocked")
        self.cfg = OOCTrainConfig(out=str(self.root / "run"), checkpoint=str(checkpoint),
                                  iterations=2, games=4, epochs=1, minibatch=4, micro_batch=1,
                                  workers=1, threads=1, eval_every=1, health_every=1,
                                  target_kl=10., normalize_returns=True, critic_gradient="detached",
                                  entropy=.01, strategic_entropy=.02)

    def model(self):
        return CompositeActorCritic(CandidateActorCritic(), CandidateActorCritic())

    def rows(self, model, *, stale=False):
        observations = [menu("rest", "smith"), menu("take gold 10g", "skip rewards"),
                        menu("take card Anger", "skip card reward"), menu("open chest", "skip chest")]
        with torch.no_grad():
            logits, values, mask = phase_forward()(model, collate_public(observations))
            log_probs = masked_distribution(logits, mask).log_prob(torch.zeros(4, dtype=torch.long))
        return [Transition(obs, 0, 0, float(log_probs[i]) - float(stale), float(values[i]),
                           [True, True], 0., True, False, 0., i, advantage=float(i - 1),
                           returns=2. + 2. * i, encoded_observation=encode_observation(obs),
                           encoding_observation_id=id(obs), strategic_menu=is_strategic_menu(obs),
                           menu_kind="other") for i, obs in enumerate(observations)]

    def assert_cpu(self, value):
        if isinstance(value, torch.Tensor):
            self.assertEqual(value.device.type, "cpu")
        elif isinstance(value, dict):
            for child in value.values():
                self.assert_cpu(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self.assert_cpu(child)

    def assert_model_cpu(self, model):
        self.assert_cpu(model.state_dict())
        self.assert_cpu([p.grad for p in model.parameters()])

    def assert_exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_exact(a, e)
        else:
            self.assertEqual(actual, expected)

    def run_trainer(self, cfg, model, *, resume=False, stale=False):
        pool = Mock(restarts=0)

        def collect(current, *args, **kwargs):
            self.assert_model_cpu(current)
            return {"kept": [[t] for t in self.rows(current, stale=stale)],
                    "game_episodes": [{"floor": 5, "task_outcome": "defeat"}],
                    "faults": 0, "fault_records": [], "stats": {}, "collector_restarts": 0}

        def evaluate(current, *args, **kwargs):
            self.assert_model_cpu(current)
            # Exercise a real CPU forward after each update, too.
            self.rows(current)
            return {"summary": {}, "records": []}

        def update(current, transitions, **kwargs):
            self.assertEqual(kwargs["device"], cfg.update_device)
            for state in kwargs["optimizer"].state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, "cpu" if key == "step" else cfg.update_device)
            result = update_ppo(current, transitions, **kwargs)
            self.assertEqual(next(current.parameters()).device.type, cfg.update_device)
            return result

        pool.collect_ooc.side_effect = collect
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(ooc_trainer, "load_policy", return_value=(model, {})), \
                patch.object(ooc_trainer, "begin_run", return_value={}), \
                patch.object(ooc_trainer, "make_workers", return_value=[]), \
                patch("sts_rl.parallel_collect.CollectorPool", return_value=pool), \
                patch.object(ooc_trainer, "collect_ooc_batch", side_effect=collect), \
                patch.object(ooc_trainer, "evaluate", side_effect=evaluate), \
                patch.object(ooc_trainer, "write_collection"), \
                patch.object(ooc_trainer, "update_ppo", side_effect=update):
            ooc_trainer.train(cfg, resume=resume)
        self.assert_model_cpu(model)
        # No map_location: both checkpoint formats must contain CPU tensors on disk.
        latest = torch.load(Path(cfg.out) / "latest.pt", weights_only=False)
        self.assert_cpu(latest)
        self.assert_cpu(torch.load(Path(cfg.out) / f"iter_{cfg.iterations:04d}.pt", weights_only=False))
        records = [json.loads(line) for line in (Path(cfg.out) / "log.jsonl").read_text().splitlines()]
        return latest, [r for r in records if r["event"] == "train"]

    def test_update_device_validation_precedes_loading_or_creating_output(self):
        with patch.object(ooc_trainer, "load_policy") as load, \
                patch.object(torch.backends.mps, "is_available", return_value=False):
            for device, message in (("cuda", "update_device must be cpu or mps"),
                                    ("CPU", "update_device must be cpu or mps"),
                                    ("mps", "requires an available MPS backend")):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, message):
                    ooc_trainer.train(replace(self.cfg, update_device=device))
            load.assert_not_called()
        self.assertFalse(Path(self.cfg.out).exists())
        self.assertEqual(OOCTrainConfig(out="unused").update_device, "cpu")

    def test_default_cpu_matches_original_updates_exactly_and_resumes(self):
        initial = self.model()
        original = deepcopy(initial)
        reference = deepcopy(initial)
        for name, parameter in reference.named_parameters():
            parameter.requires_grad_(name.startswith("ooc."))
        optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad],
                                      lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        ppo = PPOConfig(gamma=1., gae_lambda=self.cfg.gae_lambda, clip_ratio=self.cfg.clip,
                        entropy_weight=self.cfg.entropy, value_weight=self.cfg.value_weight,
                        epochs=self.cfg.epochs, minibatch_size=self.cfg.minibatch,
                        learning_rate=self.cfg.lr, kl_guard_before_update=True, target_kl=self.cfg.target_kl)
        normalizer = ReturnNormalizer()
        for iteration in (1, 2):
            rows = self.rows(reference)
            normalizer.update([t.returns for t in rows])
            update_ppo(reference, rows, collate=collate_public,
                       forward=ooc_forward(phase_forward(), critic_gradient=self.cfg.critic_gradient),
                       config=ppo, device="cpu", seed=self.cfg.seed * 1000 + iteration,
                       optimizer=optimizer, micro_batch=self.cfg.micro_batch, return_scale=normalizer.scale,
                       entropy_weights=[self.cfg.entropy + self.cfg.strategic_entropy * t.strategic_menu for t in rows])
        with patch.object(ooc_trainer, "_move_optimizer_state") as move:
            actual, records = self.run_trainer(self.cfg, initial)
            move.assert_not_called()
        self.assert_exact(actual["model_state"], reference.state_dict())
        self.assert_exact(actual["optimizer_state"], optimizer.state_dict())
        self.assertEqual([r["steps"] for r in records], [1, 1])
        self.assertTrue(all(torch.isfinite(torch.tensor(r["grad_cosine_half"])) for r in records))
        # Old configs omit the new field; explicit CPU also resumes their state unchanged.
        first, _ = self.run_trainer(replace(self.cfg, out=str(self.root / "resumed"), iterations=1), original)
        first["config"].pop("update_device")
        legacy = OOCTrainConfig(**first["config"])
        torch.save(first, Path(legacy.out) / "latest.pt")
        resumed, _ = self.run_trainer(replace(legacy, iterations=2, update_device="cpu"), self.model(), resume=True)
        self.assertEqual(resumed["steps_total"], first["steps_total"] + 1)
        self.assert_exact(resumed["model_state"], actual["model_state"])
        self.assert_exact(resumed["optimizer_state"], actual["optimizer_state"])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_updates_restore_cpu_checkpoints_and_resume_on_either_device(self):
        for micro_batch, collectors in ((1, 1), (4, 2)):
            with self.subTest(micro_batch=micro_batch, collectors=collectors):
                cfg = replace(self.cfg, out=str(self.root / f"mps_{micro_batch}"),
                              update_device="mps", micro_batch=micro_batch, collectors=collectors)
                model = self.model()
                before = {k: v.clone() for k, v in model.state_dict().items()}
                latest, records = self.run_trainer(cfg, model)
                self.assertEqual(latest["steps_total"], 2)
                self.assertTrue(any(not torch.equal(before[k], v) for k, v in latest["model_state"].items()))
                self.assertTrue(all(r["steps"] == 1 and not r["early_stop"] for r in records))
                for iteration, device in ((3, "mps"), (4, "cpu"), (5, "mps")):
                    latest, _ = self.run_trainer(replace(cfg, iterations=iteration, update_device=device),
                                                  self.model(), resume=True)
                    self.assertEqual(latest["steps_total"], iteration)
                    self.assertTrue(all(float(state["step"]) == iteration
                                        for state in latest["optimizer_state"]["state"].values()))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_kl_guard_restores_cpu_when_update_is_rejected(self):
        for micro_batch in (1, 4):
            with self.subTest(micro_batch=micro_batch):
                cfg = replace(self.cfg, out=str(self.root / f"kl_{micro_batch}"), update_device="mps",
                              iterations=1, micro_batch=micro_batch, target_kl=.001)
                model = self.model()
                before = deepcopy(model.state_dict())
                latest, records = self.run_trainer(cfg, model, stale=True)
                self.assertEqual(latest["steps_total"], 0)
                self.assertTrue(records[0]["early_stop"])
                self.assert_exact(latest["model_state"], before)


if __name__ == "__main__":
    unittest.main()
