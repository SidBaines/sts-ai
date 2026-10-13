from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import torch

from sts_policy import entities
from sts_policy.entity_model import ResidualOOCPolicy
from sts_policy.legacy_bridge import project_legacy_menu
from tests.unit.test_public_entities import observation,card


class SemanticsV4Tests(unittest.TestCase):
    def test_deck_sources_distinguish_bottled_duplicate_candidates(self):
        obs=observation(False)
        obs['state']['inventory']['deck']=[card(index=3,bottled=True),card(index=9,bottled=False)]
        obs['legal_actions']=[{'index':i,'description':'select card: Strike','public':{'kind':'select_card',
            'card':card(),'source_kind':'deck','source_index':j}} for i,j in enumerate((3,9))]
        encoded=entities.encode_observation(obs)
        self.assertFalse(np.array_equal(encoded.action_links[0],encoded.action_links[1]))
        self.assertEqual(encoded.coverage['unresolved_source'],0)
        source_values=[encoded.entity_values[np.flatnonzero(encoded.action_links[i,0])[0]] for i in range(2)]
        self.assertFalse(np.array_equal(*source_values))

    def test_reward_groups_preserve_alternatives_without_encoding_group_ordinals(self):
        obs=observation(False)
        obs['legal_actions']=[{'index':i,'description':f'take card {name}','public':{'kind':'take_card',
            'card':card(name),'reward_group':i//2}} for i,name in enumerate(('Strike','Defend','Bash','Anger'))]
        encoded=entities.encode_observation(obs)
        self.assertEqual(encoded.action_links[0,1].sum(),2)
        np.testing.assert_array_equal(encoded.action_links[0,1],encoded.action_links[1,1])
        self.assertFalse(np.array_equal(encoded.action_links[0,1],encoded.action_links[2,1]))
        renamed=deepcopy(obs)
        for action in renamed['legal_actions']:
            action['public']['reward_group']+=100
        other=entities.encode_observation(renamed)
        for key in ('entity_ids','entity_values','action_ids','action_values','action_links','global_values'):
            np.testing.assert_array_equal(getattr(encoded,key),getattr(other,key))

    def test_live_gold_is_distinct_but_legacy_projection_preserves_entry_gold(self):
        obs=observation();obs['state'].update(gold=120,gold_at_battle_entry=100,cur_hp=35,hp_at_battle_entry=50,
                                            max_hp=85,max_hp_at_battle_entry=80)
        encoded=entities.encode_observation(obs)
        self.assertAlmostEqual(float(encoded.global_values[7]),.4,places=6)
        old=project_legacy_menu(obs).observation
        self.assertEqual(old['state']['gold'],100)
        self.assertEqual(old['state']['cur_hp'],50)
        self.assertEqual(old['state']['max_hp'],80)
        self.assertEqual(obs['state']['gold'],120)

    def test_old_observation_requires_explicit_encoder_migration(self):
        obs=observation();obs['state']['observation_version']='sts_env_public_v2'
        with self.assertRaisesRegex(ValueError,'v4 encoder'):
            entities.encode_observation(obs)


if __name__=='__main__':
    unittest.main()
