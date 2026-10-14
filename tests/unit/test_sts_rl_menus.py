from __future__ import annotations

import unittest

import torch

from sts_policy.model_agent import collate_public
from sts_rl.common import apply_overlays, drop_dominated, force_play, phase_forward, strategic_rows
from tests.unit.test_policy_encoding import observation


def menu(*descriptions, phase="out_of_combat", **state):
    base = observation()
    return dict(base, phase=phase, state={**base.get("state", {}), **state},
                legal_actions=[{"index": i, "description": d} for i, d in enumerate(descriptions)])


class StrategicRowTests(unittest.TestCase):
    def test_card_only_temperature_preserves_other_strategic_menus(self):
        from unittest.mock import patch
        batch=collate_public([menu('take card Anger','skip card reward'),menu('rest','smith')])
        logits=torch.tensor([[8.,0.],[8.,0.]])
        with patch('sts_rl.common.forward_policy',return_value=(logits,torch.zeros(2),batch[2])):
            out,_,_=phase_forward(card_temperature=4.)(None,batch)
        torch.testing.assert_close(out[0],logits[0]/4.)
        torch.testing.assert_close(out[1],logits[1])

    def test_exact_frozen_combat_argmax_does_not_sample_tied_actions_or_change_ooc(self):
        from sts_policy.losses import masked_distribution
        from unittest.mock import patch
        batch=collate_public([menu('play Strike','end turn',phase='combat'),menu('rest','smith')])
        logits=torch.tensor([[2.,2.],[.2,.7]])
        values=torch.tensor([0.,0.]);mask=torch.ones(2,2,dtype=torch.bool)
        with patch('sts_rl.common.forward_policy',return_value=(logits,values,mask)):
            encoded,returned,_=phase_forward(combat='argmax')(None,batch)
        torch.testing.assert_close(encoded[1],logits[1])
        for _ in range(20):self.assertEqual(int(masked_distribution(encoded,mask).sample()[0]),0)
        torch.testing.assert_close(returned,values)

    def test_flags_identify_rest_map_shop_and_optionally_card_menus(self):
        obs = [menu("rest (heal 30% of max HP)", "smith (upgrade a card in your deck)"),
               menu("choose map node x=1 room=MONSTER", "choose map node x=3 room=ELITE"),
               menu("buy potion Fire Potion for 50g", "buy card remove for 75g", "leave shop"),
               menu("take card Armaments [Skill, Common]", "skip card reward"),
               menu("event option 0: The Cleric: leave", "event option 1: The Cleric: pay 35 gold and heal 20 HP")]
        state, actions, mask = collate_public(obs)
        self.assertEqual(strategic_rows(actions, mask).tolist(), [True, True, True, False, False])
        self.assertEqual(strategic_rows(actions, mask, cards=True).tolist(), [True, True, True, True, False])

    def test_strategic_temperature_keeps_argmax_and_softens_only_strategic_rows(self):
        torch.manual_seed(0)
        from sts_policy.model import CandidateActorCritic
        from sts_policy.model_agent import CompositeActorCritic
        model = CompositeActorCritic(CandidateActorCritic(), CandidateActorCritic())
        obs = [menu("rest (heal 30% of max HP)", "smith (upgrade a card in your deck)"),
               menu("take card Armaments [Skill, Common]", "skip card reward")]
        batch = collate_public(obs)
        plain, _, _ = phase_forward(combat="greedy")(model, batch)
        hot, _, _ = phase_forward(combat="greedy", strategic_temperature=8.0)(model, batch)
        self.assertEqual(plain.argmax(-1).tolist(), hot.argmax(-1).tolist())
        torch.testing.assert_close(hot[0, :2], plain[0, :2] / 8.0)
        torch.testing.assert_close(hot[1, :2], plain[1, :2])


class MenuFilterTests(unittest.TestCase):
    def test_rest_at_full_hp_is_dropped_unless_dream_catcher(self):
        full = menu("rest (heal 30% of max HP)", "smith (upgrade a card in your deck)", cur_hp=80, max_hp=80)
        self.assertEqual([a["description"] for a in drop_dominated(full)["legal_actions"]],
                         ["smith (upgrade a card in your deck)"])
        hurt = dict(full, state={**full["state"], "cur_hp": 79})
        self.assertEqual(len(drop_dominated(hurt)["legal_actions"]), 2)
        catcher = dict(full, text=str(full.get("text", "")) + "\nRelics: Dream Catcher")
        self.assertEqual(len(drop_dominated(catcher)["legal_actions"]), 2)

    def test_overlays_and_forced_play_never_empty_a_menu(self):
        shop = menu("buy potion Fire Potion for 50g", "buy card remove for 75g", "leave shop")
        self.assertEqual([a["description"] for a in apply_overlays(shop, prefer_remove=True)["legal_actions"]],
                         ["buy card remove for 75g", "leave shop"])
        rest = menu("rest (heal 30% of max HP)", "smith (upgrade a card in your deck)", cur_hp=56, max_hp=80)
        self.assertEqual(len(apply_overlays(rest, smith_above=.8)["legal_actions"]), 2)
        self.assertEqual(len(apply_overlays(rest, smith_above=.7)["legal_actions"]), 1)
        fight = menu("play Wild Strike [Attack] (cost 1) -> X", "end turn", phase="combat")
        fight["state"] = {**fight["state"], "cards": {"energy": 2}}
        self.assertEqual(len(force_play(fight)["legal_actions"]), 2)  # only an unsafe card: keep end turn


if __name__ == "__main__":
    unittest.main()
