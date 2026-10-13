from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from sts_policy import entities
from sts_policy.curriculum import TaskSpec
from sts_policy.workers import CurriculumWorker
from scripts.investigations.probe_public_semantics_v3 import probe_gold,probe_rewards
from tests.support import requires_simulator


@requires_simulator
class NativeSemanticsV4Tests(unittest.TestCase):
    def test_reached_bottled_and_normal_upgrades_have_distinct_public_sources(self):
        fixture=json.loads((Path(__file__).parents[1]/'fixtures/public_semantics_native_v3.json').read_text())
        positions=fixture['candidate_positions']; before=None
        for position,deck_index in zip(positions,fixture['expected_deck_indices']):
            with CurriculumWorker({'fast_observations':True,'fast_replay':True},timeout=10.) as worker:
                obs,info=worker.reset(TaskSpec.from_dict(fixture['task']))
                self.assertEqual(info['native_provenance']['backend_api'],3)
                smith=next(a['index'] for a in obs['legal_actions'] if a['description'].startswith('smith'))
                current=worker.step(smith)[0]
                action=current['legal_actions'][position]
                self.assertEqual(action['public']['source_kind'],'deck')
                self.assertEqual(action['public']['source_index'],deck_index)
                after=worker.step(action['index'])[0]
                changed=next(c for c in after['state']['inventory']['deck'] if c['index']==deck_index)
                original=next(c for c in current['state']['inventory']['deck'] if c['index']==deck_index)
                self.assertEqual(changed['upgrade_count'],original['upgrade_count']+1)
                self.assertEqual(changed['bottled'],original['bottled'])
                before=current
        encoded=entities.encode_observation(before)
        self.assertFalse(np.array_equal(encoded.action_values[positions[0]],encoded.action_values[positions[1]]))
        self.assertFalse(np.array_equal(encoded.action_links[positions[0]],encoded.action_links[positions[1]]))
        self.assertEqual(encoded.coverage['unresolved_source'],0)

    def test_hand_of_greed_updates_live_gold_and_retains_entry_snapshot(self):
        result=probe_gold()
        self.assertEqual(result['live_player_gold'],120)
        self.assertEqual(result['top_level_gold'],120)
        self.assertEqual(result['after']['state']['gold_at_battle_entry'],100)
        self.assertAlmostEqual(result['encoded_global_gold'],.4,places=6)
        turn=result['after_enemy_turn']['state']
        self.assertEqual(turn['cur_hp'],turn['combat']['player_cur_hp'])
        self.assertLess(turn['cur_hp'],turn['hp_at_battle_entry'])
        juice=result['after_fruit_juice']['state']
        self.assertEqual(juice['max_hp'],juice['combat']['player_max_hp'])
        self.assertEqual(juice['max_hp'],turn['max_hp']+5)
        self.assertEqual(juice['max_hp_at_battle_entry'],turn['max_hp_at_battle_entry'])

    def test_prayer_wheel_offers_have_correct_mutual_exclusion_groups(self):
        result=probe_rewards()
        groups={a['public']['reward_group'] for a in result['before']['legal_actions'] if a['public']['kind']=='take_card'}
        self.assertEqual(len(groups),2)
        self.assertEqual(result['card_offers_before'],6)
        self.assertEqual(result['card_offers_after'],3)
        encoded=entities.encode_observation(result['before'])
        for i,action in enumerate(result['before']['legal_actions']):
            if action['public']['kind']=='take_card':
                self.assertEqual(encoded.action_links[i,1].sum(),3)


if __name__=='__main__':
    unittest.main()
