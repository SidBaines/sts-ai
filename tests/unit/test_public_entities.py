from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import numpy as np
import torch

from sts_policy import entities
from sts_policy.entity_model import ResidualOOCPolicy, save_policy, load_policy
from sts_policy.encoding import collate_observations as legacy_collate, encode_observation as legacy_encode
from sts_policy.legacy_bridge import project_legacy_menu


def card(name="Strike", **kwargs):
    return {"name": name, "base_name": name, "type": "Attack", "rarity": "Basic", "cost": 1,
            "cost_for_turn": 1, "upgrade_count": 0, "special_data": None,
            "free_to_play_once": False, "retain": False, "exhaust": False, **kwargs}


def observation(combat=True):
    first, second = card(index=2, free_to_play_once=True), card(index=6)
    return {"phase": "combat" if combat else "out_of_combat", "text": "",
        "state": {"observation_version": "sts_env_public_v2", "act": 1, "floor": 2, "cur_hp": 50,
            "max_hp": 80, "gold": 100, "inventory": {"deck": [card()], "relics": [], "potions": {}, "boss": "Hexaghost"},
            "combat": {"player_cur_hp": 50, "player_max_hp": 80, "player_energy": 3,
                "player": {"powers": {"The Bomb (2 turns)": 40}},
                "cards": {"hand": [first, second], "draw": [card("Bash", count=2)]},
                "enemies": [{"index": 9, "name": "ACID_SLIME_L", "alive": True, "cur_hp": 50,
                    "max_hp": 65, "block": 0, "intent": "ACID_SLIME_L_TACKLE", "intent_damage": 16,
                    "intent_hits": 1, "powers": {"Vulnerable": 2}}]}},
        "legal_actions": [
            {"index": 7, "description": "play Strike [Attack] (cost 0) -> ACID_SLIME_L (deal 6) [hand 2; free once]",
             "public": {"kind": "play_card", "card": first, "source_kind": "hand", "source_index": 2,
                        "energy_cost": 0, "free_to_play": True, "target_index": 9}},
            {"index": 4, "description": "play Strike [Attack] (cost 1) -> ACID_SLIME_L (deal 6) [hand 6]",
             "public": {"kind": "play_card", "card": second, "source_kind": "hand", "source_index": 6,
                        "energy_cost": 1, "free_to_play": False, "target_index": 9}},
            {"index": 20, "description": "end turn", "public": {"kind": "end_turn"}}]}


class PublicEntityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_previously_aliased_mechanics_are_distinct(self):
        original = observation()
        base = entities.encode_observation(original)
        for mutation in (lambda o: o["state"]["combat"]["enemies"][0].update(block=20),
                         lambda o: o["state"]["combat"]["enemies"][0].update(max_hp=69),
                         lambda o: o["state"]["inventory"]["deck"][0].update(upgrade_count=5),
                         lambda o: o["state"]["combat"]["player"].update(powers={"The Bomb (3 turns)": 40})):
            changed = deepcopy(original)
            mutation(changed)
            result = entities.encode_observation(changed)
            self.assertFalse(np.array_equal(base.entity_values, result.entity_values))
        self.assertFalse(np.array_equal(base.action_values[0], base.action_values[1]))
        self.assertFalse(np.array_equal(base.action_links[0], base.action_links[1]))
        self.assertEqual(base.coverage["unresolved_source"], 0)
        self.assertEqual(base.coverage["unresolved_target"], 0)

    def test_offer_upgrades_and_unparsed_gold_keep_numeric_amounts(self):
        obs = observation(False)
        obs["legal_actions"] = [
            {"index": 2, "description": "take card Searing Blow+1", "public": {"kind": "take_card", "card": card("Searing Blow", upgrade_count=1)}},
            {"index": 9, "description": "take card Searing Blow+5", "public": {"kind": "take_card", "card": card("Searing Blow", upgrade_count=5)}},
            {"index": 12, "description": "event option 0: Obtain 100 gold.", "public": {"kind": "out_of_combat"}},
            {"index": 13, "description": "event option 1: Obtain 250 gold.", "public": {"kind": "out_of_combat"}}]
        encoded = entities.encode_observation(obs)
        self.assertFalse(np.array_equal(encoded.action_values[0], encoded.action_values[1]))
        self.assertNotEqual(encoded.action_values[2, 48], encoded.action_values[3, 48])

    def test_feasible_paths_distinguish_mandatory_elite(self):
        safe = {"nodes": [{"x": 0, "y": 0, "room": "MONSTER", "edges": [0, 1]},
                          {"x": 0, "y": 1, "room": "ELITE", "edges": [0]},
                          {"x": 1, "y": 1, "room": "REST", "edges": [1]},
                          {"x": 0, "y": 2, "room": "REST", "edges": []},
                          {"x": 1, "y": 2, "room": "ELITE", "edges": []}]}
        mandatory = deepcopy(safe)
        mandatory["nodes"][2]["edges"] = [1]
        safe["nodes"][4]["room"] = "REST"
        a, b = entities.route_features(safe)[(0, 0)], entities.route_features(mandatory)[(0, 0)]
        self.assertEqual(a[1], 0)
        self.assertEqual(b[1], 1)
        shifted = deepcopy(safe)
        for node in shifted["nodes"]:
            node["x"] += 7
            node["edges"] = [x + 7 for x in node["edges"]]
        np.testing.assert_array_equal(a, entities.route_features(shifted)[(7, 0)])

    def test_private_keys_and_routing_numbers_do_not_enter_features(self):
        obs = observation()
        base = entities.encode_observation(obs)
        changed = deepcopy(obs)
        changed["world_seed"] = 500
        changed["state"]["rng"] = "secret"
        changed["state"]["inventory"]["deck"][0]["uniqueId"] = 123456
        for action in changed["legal_actions"]:
            action["native_bits"] = 8191
            action["index"] += 100
        result = entities.encode_observation(changed)
        for name in ("legacy_state", "legacy_actions", "global_values", "entity_ids", "entity_values",
                     "entity_weights", "action_ids", "action_values", "action_links"):
            np.testing.assert_array_equal(getattr(base, name), getattr(result, name))
        self.assertNotEqual(base.action_indices.tolist(), result.action_indices.tolist())

    def test_initial_prior_is_exact_for_duplicates_and_mean_context(self):
        obs = observation()
        for context in ("none", "mean"):
            model = ResidualOOCPolicy({"size": "small", "context": context}, width=32, embedding=4)
            projection = project_legacy_menu(obs)
            logits, value = model(*entities.collate_public([obs]))
            old_logits, old_value = model.prior(*legacy_collate([legacy_encode(projection.observation)]))
            torch.testing.assert_close(logits[0], old_logits[0, [0, 0, 1]], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(value, old_value)
            self.assertEqual(int(obs["legal_actions"][logits.argmax(-1).item()]["index"]), projection.current_indices[old_logits.argmax(-1).item()])

    def test_permutation_padding_and_separate_gradient_contract(self):
        obs = observation(False)
        model = ResidualOOCPolicy({"size": "small", "context": "mean"}, width=32, embedding=4)
        # Activate residual output so invariance verifies typed processing too.
        torch.nn.init.normal_(model.actor_score[-1].weight, std=.05)
        torch.nn.init.normal_(model.value_score[-1].weight, std=.05)
        batch = entities.collate_public([obs])
        logits, value = model(*batch)
        permuted = deepcopy(obs)
        order = [2, 0, 1]
        permuted["legal_actions"] = [permuted["legal_actions"][i] for i in order]
        permuted["state"]["inventory"]["deck"].reverse()
        permuted["state"]["combat"]["cards"]["hand"].reverse()
        changed_logits, changed_value = model(*entities.collate_public([permuted]))
        torch.testing.assert_close(changed_logits, logits[:, order], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(changed_value, value)
        terminal = {**obs, "legal_actions": []}
        padded_logits, padded_value = model(*entities.collate_public([obs, terminal]))
        torch.testing.assert_close(padded_logits[0], logits[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(padded_value[0], value[0], atol=1e-6, rtol=1e-6)
        self.assertTrue((padded_logits[1] == -1e9).all())
        (-torch.log_softmax(logits, -1)[0, 1]).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor_parameters()))
        self.assertTrue(all(p.grad is None for p in model.critic_parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.prior.parameters()))
        model.zero_grad(set_to_none=True)
        model(*batch)[1].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.actor_parameters()))

    def test_unknown_zero_and_checkpoint_identity_fail_closed(self):
        obs = observation()
        first = deepcopy(obs)
        first["legal_actions"][0]["public"]["targets"] = [{"index": 9, "damage_known": False}]
        second = deepcopy(first)
        second["legal_actions"][0]["public"]["targets"] = [{"index": 9, "damage_known": True, "damage_total": 0, "hits": 1, "damage_per_hit": 0}]
        a, b = entities.encode_observation(first), entities.encode_observation(second)
        self.assertFalse(np.array_equal(a.action_values, b.action_values))
        model = ResidualOOCPolicy({"size": "small", "context": "none"}, width=32, embedding=4)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "policy.pt")
            save_policy(path, model, metadata={})
            restored, payload = load_policy(path)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            payload["model_config"]["vocabulary_sha256"] = "wrong"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                load_policy(path)
        obs["state"]["observation_version"] = "sts_env_public_v1"
        with self.assertRaisesRegex(ValueError, "requires typed"):
            entities.encode_observation(obs)


if __name__ == "__main__":
    unittest.main()
