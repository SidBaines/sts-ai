from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from sts_policy.encoding import encode_observation
from sts_policy.losses import PPOConfig, masked_distribution, ppo_objective
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic, collate_public
from sts_policy.rollout_training import Transition, update_ppo
from sts_rl import combat_trainer, ooc_trainer
from sts_rl.common import phase_forward, save_atomic, save_policy
from sts_rl.ooc_trainer import OOCTrainConfig, ReturnNormalizer, is_strategic_menu, ooc_forward
from sts_rl.recording import resume_payload
from tests.unit.test_sts_rl_menus import menu


class OOCPPOHygieneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def model(self, context="mean"):
        return CompositeActorCritic(CandidateActorCritic(context=context), CandidateActorCritic(context=context))

    def rows(self, model):
        observations = [menu("rest", "smith"), menu("take gold 10g", "skip rewards"),
                        menu("take card Anger", "skip card reward"), menu("open chest", "skip chest")]
        with torch.no_grad():
            logits, values, mask = phase_forward()(model, collate_public(observations))
            log_probs = masked_distribution(logits, mask).log_prob(torch.zeros(4, dtype=torch.long))
        return [Transition(obs, 0, 0, float(log_probs[i]), float(values[i]), [True, True],
                           float(i), True, False, 0., i, advantage=0., returns=2. + 2. * i,
                           strategic_menu=is_strategic_menu(obs)) for i, obs in enumerate(observations)]

    def test_defaults_match_recipe_and_preserve_optional_behaviour(self):
        cfg = OOCTrainConfig(out="unused")
        self.assertEqual(cfg.gae_lambda, 1.)
        self.assertEqual(cfg.weight_decay, 0.)
        self.assertEqual(combat_trainer.CombatTrainConfig(bank="", out="", checkpoint="").weight_decay, 0.)
        self.assertEqual(cfg.critic_gradient, "shared")
        self.assertEqual(cfg.value_weight, .5)
        self.assertFalse(cfg.normalize_returns)
        self.assertEqual((cfg.entropy, cfg.strategic_entropy), (0., 0.))

    def test_weight_decay_reaches_both_optimizers_and_overrides_resume(self):
        adamw = torch.optim.AdamW
        for trainer in (ooc_trainer, combat_trainer):
            for resume in (False, True):
                with self.subTest(trainer=trainer.__name__, resume=resume), \
                        tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, \
                        contextlib.redirect_stdout(io.StringIO()):
                    root = Path(directory)
                    model = self.model()
                    out = root / "run"
                    out.mkdir()
                    if trainer is ooc_trainer:
                        cfg = OOCTrainConfig(out=str(out), iterations=0, weight_decay=.023, lr=.004)
                        worker_factory = "make_workers"
                    else:
                        bank = root / "bank.jsonl"
                        bank.write_text("")
                        cfg = combat_trainer.CombatTrainConfig(bank=str(bank), out=str(out), checkpoint="unused",
                                                              iterations=0, weight_decay=.023, lr=.004)
                        worker_factory = "_battle_workers"
                    if resume:
                        prefix = "ooc." if trainer is ooc_trainer else "combat."
                        saved = adamw([p for n, p in model.named_parameters() if n.startswith(prefix)],
                                      lr=.1, weight_decay=.9)
                        torch.save(resume_payload(model, saved, 0, 0, {}, {}, random.Random(1)), out / "latest.pt")
                    optimizers = []

                    def capture(parameters, **kwargs):
                        optimizer = adamw(parameters, **kwargs)
                        optimizers.append(optimizer)
                        return optimizer

                    with patch.object(trainer, "load_policy", return_value=(model, {})), \
                            patch.object(trainer, "begin_run", return_value={}), \
                            patch.object(trainer, worker_factory, return_value=[]), \
                            patch.object(trainer.torch.optim, "AdamW", side_effect=capture) as constructor:
                        if trainer is ooc_trainer:
                            with patch.object(trainer, "evaluate", return_value={"summary": {}, "records": []}):
                                trainer.train(cfg, resume=resume)
                        else:
                            with patch.object(trainer, "fight_panel", return_value={}):
                                trainer.train(cfg, resume=resume)
                    self.assertEqual(constructor.call_args.kwargs["weight_decay"], .023)
                    self.assertEqual(optimizers[0].param_groups[0]["weight_decay"], .023)
                    self.assertEqual(optimizers[0].param_groups[0]["lr"], .004)

    def test_detached_value_loss_has_exactly_zero_actor_gradients(self):
        for context in ("mean", "none"):
            model = self.model(context)
            batch = collate_public([menu("rest", "smith"), menu("take card Anger", "skip card reward")])
            reference = phase_forward()(model, batch)
            for mode in ("shared", "detached"):
                model.zero_grad(set_to_none=True)
                logits, values, mask = ooc_forward(phase_forward(), critic_gradient=mode)(model, batch)
                for actual, expected in zip((logits, values, mask), reference):
                    self.assertTrue(torch.equal(actual, expected))
                loss, _ = ppo_objective(logits, mask, values, torch.zeros(2, dtype=torch.long),
                                        torch.zeros(2), torch.zeros(2), torch.full((2,), 10.),
                                        PPOConfig(entropy_weight=0.))
                loss.backward()
                actor_gradients = {name: p.grad for name, p in model.ooc.named_parameters()
                                   if not name.startswith("critic.")}
                if mode == "detached":
                    for name, gradient in actor_gradients.items():
                        self.assertIsNotNone(gradient, name)
                        self.assertEqual(torch.count_nonzero(gradient).item(), 0, name)
                else:
                    for encoder in ("state_encoder.", "action_encoder.") if context == "mean" else ("state_encoder.",):
                        self.assertTrue(any(torch.count_nonzero(g).item() for n, g in actor_gradients.items()
                                            if n.startswith(encoder)))
                self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad).item()
                                    for p in model.ooc.critic.parameters()))
                self.assertEqual(len(model.ooc.critic._forward_pre_hooks), 0)
            # The adapter has not changed the model's public forward path.
            self.assertTrue(torch.equal(model(*batch)[1], reference[1]))

    def test_return_statistics_scale_targets_and_unscale_only_ooc_baselines(self):
        normalizer = ReturnNormalizer()
        self.assertEqual(normalizer.scale, 1.)
        normalizer.update([2., 4.])
        normalizer.update([6., 8.])
        self.assertEqual(normalizer.count, 4)
        self.assertEqual(normalizer.mean, 5.)
        self.assertAlmostEqual(normalizer.scale, np.std([2., 4., 6., 8.]))
        model = self.model()
        observations = [menu("rest", "smith"), menu("play Strike", "end turn", phase="combat")]
        batch = collate_public(observations)
        with torch.no_grad():
            plain = phase_forward()(model, batch)
            scaled = ooc_forward(phase_forward(), value_scale=normalizer.scale)(model, batch)
        self.assertTrue(torch.equal(plain[0], scaled[0]))
        self.assertEqual(float(scaled[1][0]), float(plain[1][0] * normalizer.scale))
        self.assertEqual(float(scaled[1][1]), float(plain[1][1]))
        rows = self.rows(model)
        for micro in (None, 1):
            result = update_ppo(model, rows, collate=collate_public, forward=phase_forward(),
                                optimizer=torch.optim.SGD(model.parameters(), lr=0.), return_scale=normalizer.scale,
                                micro_batch=micro, config=PPOConfig(epochs=1, minibatch_size=4, target_kl=10.))
            with torch.no_grad():
                _, values, _ = phase_forward()(model, collate_public([t.observation for t in rows]))
            expected = .5 * (values - torch.tensor([t.returns / normalizer.scale for t in rows])).square().mean()
            self.assertAlmostEqual(result["value_loss"], float(expected), places=5)
        self.assertEqual([t.returns for t in rows], [2., 4., 6., 8.])

    def test_return_statistics_survive_both_checkpoint_round_trips(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.)
        normalizer = ReturnNormalizer()
        normalizer.update([2., 4., 6.])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            payload = resume_payload(model, optimizer, 3, 5, {}, {}, random.Random(1))
            payload["trainer_state"] = normalizer.trainer_state()
            save_atomic(payload, root / "latest.pt")
            save_policy(root / "iter.pt", model, {}, metadata={}, trainer_state=normalizer.trainer_state())
            for name in ("latest.pt", "iter.pt"):
                restored = ReturnNormalizer.restore(torch.load(root / name, weights_only=False), enabled=True)
                self.assertEqual(asdict(restored), asdict(normalizer))
                restored.update([8., 10.])
                self.assertAlmostEqual(restored.scale, np.std([2., 4., 6., 8., 10.]))
                self.assertEqual(restored.count, 5)
                with self.assertRaisesRegex(ValueError, "normalised critic units"):
                    ReturnNormalizer.restore(torch.load(root / name, weights_only=False), enabled=False)
        self.assertEqual(ReturnNormalizer.restore({}, enabled=True).scale, 1.)

    def test_strategic_menus_include_removals_and_exclude_other_choices(self):
        strategic = [menu("rest", "smith"), menu("lift (Girya)", "dig (Shovel)"),
                     menu("buy potion Fire Potion for 50g", "leave shop"),
                     menu("take card Anger", "skip card reward"),
                     menu("choose map node x=1 room=MONSTER", "choose map node x=3 room=ELITE")]
        removal = menu("select card index 0: Strike", "select card index 1: Defend")
        removal["text"] = "Choose a card to REMOVE from your deck (chosen 0 of 1)."
        strategic.append(removal)
        for obs in strategic:
            self.assertTrue(is_strategic_menu(obs), obs)
        for obs in [menu("open chest", "skip chest"), menu("take gold 10g", "skip rewards"),
                    menu("event option 0: leave", "event option 1: pay 35 gold and heal 20 HP"),
                    menu("rest", "smith", phase="combat"),
                    dict(removal, text="Choose a card to UPGRADE (chosen 0 of 1).")]:
            self.assertFalse(is_strategic_menu(obs), obs)

    def test_strategic_entropy_changes_only_strategic_rows_in_both_update_paths(self):
        model = self.model()
        rows = self.rows(model)
        weights = [.07 + .3 * t.strategic_menu for t in rows]
        for micro in (None, 1):
            baseline, strategic = deepcopy(model), deepcopy(model)
            captured = []

            def objective(*args, **kwargs):
                loss, stats = ppo_objective(*args, **kwargs)
                captured.extend(kwargs["entropy_weights"].tolist())
                return loss, stats

            cfg = PPOConfig(epochs=1, minibatch_size=4, entropy_weight=.07, value_weight=0., target_kl=10.)
            with patch("sts_policy.rollout_training.ppo_objective", side_effect=objective):
                result = update_ppo(strategic, rows, collate=collate_public, forward=phase_forward(),
                                    optimizer=torch.optim.SGD(strategic.parameters(), lr=.1), config=cfg,
                                    micro_batch=micro, entropy_weights=weights)
            update_ppo(baseline, rows, collate=collate_public, forward=phase_forward(),
                       optimizer=torch.optim.SGD(baseline.parameters(), lr=.1), config=cfg, micro_batch=micro)
            self.assertEqual(result["steps"], 1)
            self.assertFalse(result["early_stop"])
            np.testing.assert_allclose(sorted(captured), sorted(weights))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(baseline.parameters(), strategic.parameters())))
        # Check row gradients directly, avoiding shared-parameter cross-talk.
        logits = torch.tensor([[2., 0.]] * 4, requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        args = (mask, torch.zeros(4), torch.zeros(4, dtype=torch.long), torch.zeros(4),
                torch.zeros(4), torch.zeros(4), PPOConfig(entropy_weight=0., value_weight=0.))
        loss, _ = ppo_objective(logits, *args, entropy_weights=torch.tensor([.3, 0., .3, 0.]))
        loss.backward()
        self.assertTrue(torch.count_nonzero(logits.grad[[0, 2]]).item())
        self.assertEqual(torch.count_nonzero(logits.grad[[1, 3]]).item(), 0)

    def test_collection_unscales_before_gae_and_preserves_removal_label_when_slimming(self):
        model = self.model()
        obs = menu("select card index 0: Strike", "select card index 1: Defend")
        obs["text"] = "Choose a card to REMOVE from your deck (chosen 0 of 1)."
        row = Transition(obs, 0, 0, -.7, 0., [True, True], 8., True, False, 0., 1,
                         encoded_observation=encode_observation(obs), encoding_observation_id=id(obs))

        def collect(model, workers, tasks, *, forward, **kwargs):
            with torch.no_grad():
                row.behavior_value = float(forward(model, collate_public([obs]))[1][0])
            return {"transitions": [row], "episodes": [], "faults": []}

        with torch.no_grad():
            raw = float(phase_forward()(model, collate_public([obs]))[1][0]) * 4.
        with patch.object(ooc_trainer, "collect_rollouts", side_effect=collect):
            result = ooc_trainer.collect_ooc_batch(model, [], [object()], forward=phase_forward(),
                ppo=PPOConfig(), credit="macro", gae_lambda=1., chunk=1, value_scale=4.)
        kept = result["kept"][0][0]
        self.assertAlmostEqual(kept.advantage, 8. - raw)
        self.assertEqual(kept.returns, 8.)
        self.assertNotIn("text", kept.observation)
        self.assertTrue(kept.strategic_menu)
        self.assertEqual(kept.menu_kind, "removal")

    def test_empty_update_logs_zero_steps_and_no_early_stop(self):
        result = update_ppo(self.model(), [], collate=collate_public, forward=phase_forward())
        self.assertEqual(result["steps"], 0)
        self.assertFalse(result["early_stop"])

    def test_synthetic_trainer_wires_options_and_restores_statistics_on_resume(self):
        model = self.model()
        rows = self.rows(model)
        for t in rows:
            t.encoded_observation = encode_observation(t.observation)
            t.encoding_observation_id = id(t.observation)
        collection_scales, updates = [], []

        def collect(*args, **kwargs):
            collection_scales.append(kwargs["value_scale"])
            return {"kept": [rows], "game_episodes": [{"floor": 5, "task_outcome": "defeat"}],
                    "faults": 0, "fault_records": [], "stats": {}}

        def update(*args, **kwargs):
            self.assertEqual(exploration.call_count, min(len(updates) + 1, 2))
            updates.append(kwargs)
            return {"optimizer": kwargs["optimizer"], "steps": 2, "early_stop": True}

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, contextlib.redirect_stdout(io.StringIO()):
            cfg = OOCTrainConfig(out=directory, iterations=1, eval_every=0, health_every=0,
                                 value_weight=.7, strategic_entropy=.4, entropy=.1,
                                 normalize_returns=True, critic_gradient="detached")
            with patch.object(ooc_trainer, "load_policy", return_value=(model, {})), \
                    patch.object(ooc_trainer, "begin_run", return_value={}), \
                    patch.object(ooc_trainer, "make_workers", return_value=[]), \
                    patch.object(ooc_trainer, "evaluate", return_value={"summary": {}, "records": []}), \
                    patch.object(ooc_trainer, "collect_ooc_batch", side_effect=collect), \
                    patch.object(ooc_trainer, "write_collection"), \
                    patch.object(ooc_trainer, "exploration_stats", wraps=ooc_trainer.exploration_stats) as exploration, \
                    patch.object(ooc_trainer, "update_ppo", side_effect=update):
                ooc_trainer.train(cfg)
                first = torch.load(Path(directory) / "latest.pt", weights_only=False)
                # Both saved configs predate exploration tracking; defaults fill the new field.
                first["config"].pop("exploration_stats")
                torch.save(first, Path(directory) / "latest.pt")
                (Path(directory) / "config.json").write_text(json.dumps(first["config"]))
                legacy_cfg = OOCTrainConfig(**first["config"])
                self.assertTrue(legacy_cfg.exploration_stats)
                ooc_trainer.train(replace(legacy_cfg, iterations=2), resume=True)
                ooc_trainer.train(replace(cfg, iterations=3, exploration_stats=False), resume=True)
                self.assertEqual(exploration.call_count, 2)
                self.assertEqual([call.kwargs["chunk"] for call in exploration.call_args_list], [cfg.micro_batch] * 2)
            latest = torch.load(Path(directory) / "latest.pt", weights_only=False)
            self.assertEqual(first["trainer_state"]["return_normalizer"]["count"], 4)
            self.assertEqual(latest["trainer_state"]["return_normalizer"]["count"], 12)
            self.assertEqual(latest["steps_total"], 6)
            self.assertEqual(collection_scales, [1., np.std([2., 4., 6., 8.]), np.std([2., 4., 6., 8.])])
            for kwargs in updates:
                self.assertEqual(kwargs["config"].value_weight, .7)
                self.assertEqual(kwargs["return_scale"], np.std([2., 4., 6., 8.]))
                self.assertEqual(kwargs["entropy_weights"], [.5, .1, .5, .1])
            records = [json.loads(line) for line in (Path(directory) / "log.jsonl").read_text().splitlines()]
            for row in [r for r in records if r["event"] == "train"]:
                self.assertEqual(row["steps"], 2)
                self.assertTrue(row["early_stop"])
                self.assertEqual(row["collector_restarts"], 0)
                if row["iteration"] < 3:
                    self.assertEqual(row["explore_n"], 4)
                    self.assertIsNotNone(row["explore_entropy"])
                    self.assertGreaterEqual(row["explore_seconds"], 0.)
                else:
                    self.assertFalse(any(key.startswith("explore_") for key in row))
            for row in [r for r in records if r["event"] == "eval"]:
                self.assertEqual(row["collector_restarts"], 0)

    def test_trainer_records_restarts_for_each_collection_and_evaluation(self):
        model = self.model()
        pool = Mock(restarts=5)

        def evaluate(*args, **kwargs):
            pool.restarts += 1
            return {"summary": {}, "records": []}

        def collect(*args, **kwargs):
            pool.restarts += 2
            return {"kept": [], "game_episodes": [{"floor": 5, "task_outcome": "defeat"}],
                    "faults": 0, "fault_records": [], "stats": {}, "collector_restarts": 2}

        pool.collect_ooc.side_effect = collect
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, \
             contextlib.redirect_stdout(io.StringIO()):
            cfg = OOCTrainConfig(out=directory, iterations=1, eval_every=1, health_every=0, collectors=2)
            with patch.object(ooc_trainer, "load_policy", return_value=(model, {})), \
                 patch.object(ooc_trainer, "begin_run", return_value={}), \
                 patch("sts_rl.parallel_collect.CollectorPool", return_value=pool), \
                 patch.object(ooc_trainer, "evaluate", side_effect=evaluate), \
                 patch.object(ooc_trainer, "write_collection"):
                ooc_trainer.train(cfg)
            records = [json.loads(line) for line in (Path(directory) / "log.jsonl").read_text().splitlines()]
        self.assertEqual([row["collector_restarts"] for row in records if row["event"] == "train"], [2])
        self.assertEqual([row["collector_restarts"] for row in records if row["event"] == "eval"], [1, 1])
        self.assertEqual(pool.restarts, 9)
        pool.close.assert_called_once()

    def test_parallel_collector_forwards_value_scale_without_starting_workers(self):
        from sts_rl.parallel_collect import CollectorPool
        pool = CollectorPool.__new__(CollectorPool)
        pool.restarts = 0
        with patch.object(pool, "_scatter", return_value=[]) as scatter:
            result = pool.collect_ooc(self.model(), [], combat="greedy", temperature=1., ppo=PPOConfig(),
                                      credit="macro", gae_lambda=1., chunk=1, seed=1, value_scale=3.5)
        self.assertEqual(scatter.call_args.kwargs["value_scale"], 3.5)
        self.assertEqual(result["collector_restarts"], 0)

    def test_kl_guard_reports_actual_optimizer_steps_for_both_update_paths(self):
        model = self.model()
        stale = [replace(t, behavior_log_prob=t.behavior_log_prob - 1.) for t in self.rows(model)]
        for micro in (None, 1):
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                result = update_ppo(model, stale, collate=collate_public, forward=phase_forward(),
                                    optimizer=optimizer, micro_batch=micro,
                                    config=PPOConfig(epochs=2, minibatch_size=4, kl_guard_before_update=True))
            step.assert_not_called()
            self.assertEqual(result["steps"], 0)
            self.assertTrue(result["early_stop"])


if __name__ == "__main__":
    unittest.main()
