from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from sts_ai import fight_analysis as analysis
from sts_ai import fight_lab as lab
from tests.unit.test_fight_lab import record


class FightAnalysisTest(unittest.TestCase):
    def test_versioned_start_allows_only_declared_presentation_changes(self):
        old = record()
        old['state']['combat_observation'] = 'combat_public_v3'
        old['legal_actions'][0]['description'] = '{ GAMBLE none'
        new = copy.deepcopy(old)
        new['state']['combat_observation'] = 'combat_public_v4'
        new['state_text'] = 'New explanatory text'
        new['legal_actions'][0]['description'] = 'keep current hand (discard no cards)'
        self.assertTrue(analysis.same_start(new, old))
        new['state']['combat']['player_cur_hp'] -= 1
        self.assertFalse(analysis.same_start(new, old))
        new = copy.deepcopy(old)
        new['state_text'] += 'changed'
        self.assertFalse(analysis.same_start(new, old))

    def test_cap_detection_uses_actual_limit(self):
        r = {'agent': {'completion_tokens': 5000, 'valid': True}}
        self.assertFalse(analysis.generation_cap_hit(r, 16384))
        self.assertTrue(analysis.generation_cap_hit(r, 4096))
        r['agent'].update(completion_tokens=16382, valid=False)
        self.assertTrue(analysis.generation_cap_hit(r, 16384))
        r['agent'].update(completion_tokens=4092, valid=False)
        self.assertTrue(analysis.generation_cap_hit(r, 4096))
        r['agent']['valid'] = True
        self.assertFalse(analysis.generation_cap_hit(r, 4096))

    def test_explicit_truncation_is_not_lost_to_retokenization(self):
        r = {'agent': {'completion_tokens': 16375, 'valid': False,
                       'metadata': {'thinking_truncated': True}}}
        self.assertTrue(analysis.generation_cap_hit(r, 16384))
        from scripts.thinking_fight_budget import cap_hit
        self.assertTrue(cap_hit(r, 16384))
        r['agent']['metadata']['thinking_truncated'] = False
        self.assertFalse(analysis.generation_cap_hit(r, 16384))
        self.assertFalse(cap_hit(r, 16384))

    def test_old_parser_accepted_thought_json_still_counts_as_unclosed(self):
        r = {'agent': {'completion_tokens': 16378, 'valid': True,
                       'metadata': {'thinking_truncated': False},
                       'raw_response': '<|channel>thought\n{"action":"end turn"} still thinking'}}
        self.assertTrue(analysis.generation_cap_hit(r, 16384))
        from scripts.thinking_fight_budget import cap_hit
        self.assertTrue(cap_hit(r, 16384))
        r['agent']['raw_response'] = 'thought\n{"action":"end turn"}'
        self.assertFalse(analysis.generation_cap_hit(r, 16384))

    def test_invalid_response_has_no_resolved_hp_outcome(self):
        row=record()
        row['action_executed']=False
        row['after_state']={'cur_hp':60,'phase':'out_of_combat'}
        self.assertEqual(analysis.trace_outcome([row]),(False,None,None))

    def test_death_is_completed_even_if_combat_phase_remains(self):
        row=record();row['after_state']={'cur_hp':0,'phase':'combat'}
        self.assertEqual(analysis.trace_outcome([row]),(True,False,0))

    def test_combat_slice_stops_at_first_terminal_after_state(self):
        rows=[record(),record(1),record(2,combat=False),record(3,floor=2)]
        rows[0]['after_state']={'cur_hp':40,'phase':'combat'}
        rows[1]['after_state']={'cur_hp':46,'phase':'out_of_combat'}
        self.assertEqual(analysis.fight_records(rows,0),rows[:2])
        self.assertEqual(analysis.trace_outcome(rows[:2]),(True,True,46))

    def test_torn_tail_only_allowed_for_incomplete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'trace.jsonl';p.write_text('{"x":1}\n{"x":')
            self.assertEqual(analysis.read_trace(p,allow_torn_tail=True),([{'x':1}],1))
            with self.assertRaises(ValueError):analysis.read_trace(p,allow_torn_tail=False)
            p.write_text('bad\n{"x":1}\n')
            with self.assertRaises(ValueError):analysis.read_trace(p,allow_torn_tail=True)

    def test_snapshot_freezes_finished_set_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest={'order':['a','b']}
            lab.write_new(root/'a.result.json',{'state_id':'a','status':'ok'})
            saved={'inputs':analysis.snapshot_inputs(root,manifest)}
            lab.write_new(root/'b.result.json',{'state_id':'b','status':'ok'})
            self.assertEqual(len(analysis.snapshot_inputs(root,manifest,saved)),1)
            self.assertEqual(len(analysis.snapshot_inputs(root,manifest)),2)
            (root/'a.result.json').write_text('{"state_id":"a","status":"changed"}')
            with self.assertRaisesRegex(ValueError,'changed since snapshot'):
                analysis.snapshot_inputs(root,manifest,saved)

    def test_bootstrap_is_deterministic_and_retains_world_groups(self):
        rows=[{'world_seed':i,'value':float(i)} for i in range(5)]
        a=analysis.cluster_interval(rows,'value',repeats=100)
        self.assertEqual(a,analysis.cluster_interval(rows,'value',repeats=100))
        self.assertEqual(a,analysis.cluster_interval(rows*3,'value',repeats=100))
        self.assertIsNone(analysis.cluster_interval(rows[:2],'value'))
        self.assertLess(a[0],2);self.assertGreater(a[1],2)

    def test_percentile_interpolates(self):
        self.assertEqual(analysis.percentile([0,10],.25),2.5)
        self.assertIsNone(analysis.percentile([],.5))
