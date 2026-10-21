from __future__ import annotations

import json
import math
import random
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from sts_policy.encoding import encode_observation
from sts_policy.losses import masked_distribution
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic, collate_public
from sts_policy.rollout_training import Transition
from sts_rl.common import phase_forward
from sts_rl.ooc_trainer import MENU_KINDS, exploration_stats, menu_kind, ooc_forward
from tests.unit.test_sts_rl_menus import menu


class OOCExplorationStatsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def model(self):
        return CompositeActorCritic(CandidateActorCritic(), CandidateActorCritic())

    def row(self, observation, probability=.5, *, kind=None):
        legal = observation["legal_actions"]
        return Transition(observation, 0, legal[0]["index"], math.log(probability), 0.,
                          [True] * len(legal), 0., True, False, 0., 1,
                          encoded_observation=encode_observation(observation),
                          encoding_observation_id=id(observation), menu_kind=kind)

    def test_menu_kinds_use_public_descriptions_and_removal_purpose_line(self):
        self.assertEqual(MENU_KINDS, ("rest_site", "card_reward", "map", "shop", "removal", "other"))
        removal = menu("select card index 0: Strike", "select card index 1: Defend")
        removal["text"] = "Deck selection\nChoose a card to REMOVE from your deck (chosen 0 of 1).\n"
        cases = [(menu("rest (heal 30% of max HP)", "dig (Shovel)"), "rest_site"),
                 (menu("smith (upgrade a card in your deck)", "skip campfire"), "rest_site"),
                 (menu("take card Armaments [Skill, Common]", "skip card reward"), "card_reward"),
                 (menu("choose map node x=1 room=MONSTER", "choose map node x=3 room=ELITE"), "map"),
                 (menu("buy card remove for 75g", "leave shop"), "shop"),
                 (removal, "removal"),
                 (menu("open chest", "skip chest"), "other"),
                 (dict(removal, text="Choose a card to UPGRADE (chosen 0 of 1)."), "other"),
                 (dict(removal, text="Choose a card to REMOVExtra"), "other"),
                 (dict(menu("rest", "smith"), text=removal["text"]), "removal")]
        for observation, expected in cases:
            with self.subTest(expected=expected, observation=observation):
                self.assertEqual(menu_kind(observation), expected)

    def test_statistics_match_known_masked_probabilities_across_chunks(self):
        model = self.model()
        probabilities = [(.5, .5), (.995, .005), (.985, .015), (.8, .2), (.7, .3),
                         (.6, .3, .1), (.25, .75), (1.,)]
        kinds = ["rest_site", "rest_site", "rest_site", "card_reward", "map", "removal", None, "shop"]
        taken = [.06, .04, .0501, .009, .011, .051, .005, .001]
        rows = [self.row(menu(*(f"event option {i}" for i in range(len(probs)))), p, kind=kind)
                for probs, p, kind in zip(probabilities, taken, kinds)]
        entropies = [-sum(p * math.log(p) for p in probs) for probs in probabilities]
        expected = {"explore_n": 7, "explore_entropy": sum(entropies[:7]) / 7,
                    "explore_rare05": 4 / 7, "explore_rare01": 2 / 7}
        for kind in MENU_KINDS:
            indices = [i for i in range(7) if (kinds[i] or "other") == kind]
            n = len(indices)
            expected[f"explore_n_{kind}"] = n
            expected[f"explore_entropy_{kind}"] = sum(entropies[i] for i in indices) / n if n else None
            expected[f"explore_maxprob_{kind}"] = sum(max(probabilities[i]) for i in indices) / n if n else None
            expected[f"explore_saturated_{kind}"] = sum(max(probabilities[i]) > .99 for i in indices) / n if n else None
            expected[f"explore_rare05_{kind}"] = sum(taken[i] < .05 for i in indices) / n if n else None

        for chunk in (1, 3, len(rows)):
            with self.subTest(chunk=chunk):
                offset = 0
                sizes = []

                def known_forward(model, batch):
                    nonlocal offset
                    self.assertFalse(model.training)
                    self.assertFalse(torch.is_grad_enabled())
                    mask = batch[2]
                    sizes.append(len(mask))
                    # A large padded logit catches accidentally unmasked statistics.
                    logits = torch.full(mask.shape, 1e6, dtype=torch.float64)
                    for i, probs in enumerate(probabilities[offset:offset + len(mask)]):
                        logits[i, :len(probs)] = torch.tensor([math.log(p) for p in probs], dtype=torch.float64)
                    offset += len(mask)
                    return logits, torch.zeros(len(mask), dtype=torch.float64), mask

                with patch("sts_rl.common.forward_policy", side_effect=known_forward):
                    actual = exploration_stats(model, ooc_forward(phase_forward()), rows, chunk=chunk)
                self.assertEqual(sizes, [min(chunk, len(rows) - i) for i in range(0, len(rows), chunk)])
                self.assertEqual(actual.keys(), expected.keys())
                for key, value in expected.items():
                    if value is None:
                        self.assertIsNone(actual[key], key)
                    elif key.startswith("explore_n"):
                        self.assertIs(type(actual[key]), int, key)
                        self.assertEqual(actual[key], value, key)
                    else:
                        self.assertAlmostEqual(actual[key], value, places=12, msg=key)
                json.dumps(actual, allow_nan=False)

    def test_empty_and_single_unmasked_action_batches_have_no_decisions(self):
        model = self.model()
        rows = [self.row(menu("leave shop"), kind="shop"),
                self.row(menu("rest", "smith"), kind="rest_site")]

        def one_legal_action(model, batch):
            mask = batch[2].clone()
            mask[:, 1:] = False
            return torch.zeros_like(mask, dtype=torch.float32), torch.zeros(len(mask)), mask

        for transitions in ([], rows):
            with self.subTest(empty=not transitions):
                forward = Mock(side_effect=one_legal_action)
                result = exploration_stats(model, forward, transitions, chunk=2)
                self.assertEqual(forward.call_count, int(bool(transitions)))
                for key, value in result.items():
                    if key.startswith("explore_n"):
                        self.assertIs(type(value), int)
                        self.assertEqual(value, 0)
                    else:
                        self.assertIsNone(value, key)
                json.dumps(result, allow_nan=False)

    def test_preserves_parameters_gradients_modes_and_rng_and_reuses_encodings(self):
        model = self.model()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("ooc."))
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        rows = [self.row(menu("rest", "smith"), kind="rest_site"),
                self.row(menu("take card Anger", "skip card reward"), kind="card_reward")]
        sampling_forward = ooc_forward(phase_forward(temperature=2., strategic_temperature=3., card_temperature=4.))
        with torch.no_grad():
            logits, _, mask = sampling_forward(model, collate_public([t.observation for t in rows]))
            expected = float(masked_distribution(logits, mask).entropy().mean())
        for row in rows:
            row.observation.pop("text")
            row.observation["state"] = {}
        parameters = [p.clone() for p in model.parameters()]
        gradients = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        requires_grad = [p.requires_grad for p in model.parameters()]
        torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        for training in (True, False):
            model.train(training)

            def forward(model, batch):
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                return sampling_forward(model, batch)

            with self.subTest(training=training), \
                    patch("sts_policy.model_agent.encode_observation", side_effect=AssertionError("must reuse encoding")):
                result = exploration_stats(model, forward, rows, chunk=1)
            self.assertAlmostEqual(result["explore_entropy"], expected, places=6)
            self.assertEqual(model.training, training)
            self.assertEqual([p.requires_grad for p in model.parameters()], requires_grad)
            for parameter, saved, gradient in zip(model.parameters(), parameters, gradients):
                self.assertTrue(torch.equal(parameter, saved))
                if gradient is None:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertTrue(torch.equal(parameter.grad, gradient))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            actual_numpy_rng = np.random.get_state()
            self.assertEqual(actual_numpy_rng[0], numpy_rng[0])
            np.testing.assert_array_equal(actual_numpy_rng[1], numpy_rng[1])
            self.assertEqual(actual_numpy_rng[2:], numpy_rng[2:])

    def test_forward_failure_restores_model_mode(self):
        model = self.model().train()
        forward = Mock(side_effect=RuntimeError("forward failed"))
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            exploration_stats(model, forward, [self.row(menu("rest", "smith"))], chunk=1)
        self.assertTrue(model.training)

    def test_cli_generates_both_exploration_flags_and_defaults_to_enabled(self):
        from scripts.rl import train_ooc

        for flags, expected in (([], True), (["--exploration-stats"], True), (["--no-exploration-stats"], False)):
            with self.subTest(flags=flags), \
                    patch.object(sys, "argv", ["train_ooc.py", "--out", "unused", "--resume", *flags]), \
                    patch.object(train_ooc, "train") as train:
                train_ooc.main()
            self.assertEqual(train.call_args.args[0].exploration_stats, expected)
            self.assertTrue(train.call_args.kwargs["resume"])


if __name__ == "__main__":
    unittest.main()
