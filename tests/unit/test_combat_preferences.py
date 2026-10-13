from __future__ import annotations
import unittest
import numpy as np
import torch
from sts_policy import entities
from sts_policy.combat_preferences import preferences,preference_loss
from tests.unit.test_public_entities import observation


class CombatPreferenceTests(unittest.TestCase):
    def test_stable_preferences_require_visits_and_cross_condition_agreement(self):
        row=entities.encode_observation(observation())
        def condition(reverse=False):
            return {'status':'ok','root_edges':[{'index':int(index),'visits':20,'mean_evaluation':float((2-i) if not reverse else i)}
                                               for i,index in enumerate(row.action_indices)]}
        panel={'results':[condition() for _ in range(6)]}
        self.assertTrue(preferences(row,panel))
        self.assertEqual(preferences(row,{'results':[condition(i%2==0) for i in range(6)]}),[])
        for result in panel['results']:
            for edge in result['root_edges']:edge['visits']=1
        self.assertEqual(preferences(row,panel),[])

    def test_pair_loss_does_not_reward_copy_multiplicity(self):
        values=torch.tensor([[1.,1.,0.]],requires_grad=True)
        groups=[[(np.array([True,True,False]),np.array([False,False,True]),1.)]]
        actual=preference_loss(values,groups)
        expected=torch.nn.functional.softplus(torch.tensor(-1.))
        torch.testing.assert_close(actual,expected);actual.backward()
        self.assertEqual(float(values.grad[0,0]),float(values.grad[0,1]))

    def test_correction_hinge_stops_pushing_an_already_correct_pair(self):
        values=torch.tensor([[1.,0.]],requires_grad=True)
        pairs=[[(np.array([True,False]),np.array([False,True]),1.)]]
        loss=preference_loss(values,pairs,hinge_margin=.25)
        loss.backward()
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(bool((values.grad==0).all()))


if __name__=='__main__':unittest.main()
