from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.plot_nonllm_development import episode_summary,collect_data,phase_label


class DevelopmentPlotTests(unittest.TestCase):
    def test_censored_and_faulted_rows_do_not_become_losses(self):
        row={'episodes':[{'split':'dev','world_seed':1,'task_id':'a','terminated':True,'outcome':'battle_victory'},
                         {'split':'dev','world_seed':2,'task_id':'b','truncated':True,'outcome':'battle_defeat'},
                         {'split':'dev','world_seed':3,'task_id':'c','fault':True,'outcome':'battle_defeat'}],
             'faults':[{'world_seed':4}]}
        result=episode_summary(row)
        self.assertEqual(result['wins'],1)
        self.assertEqual(result['decisive'],1)
        self.assertEqual(result['faults'],2)
        self.assertEqual(result['truncated'],1)
        self.assertEqual(result['win_rate_decisive'],1.)

    def test_final_rows_rejected_and_final_files_not_read(self):
        with self.assertRaises(ValueError): episode_summary({'episodes':[{'split':'final'}]})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'campaign.json').write_text(json.dumps({'started_at':1,'commands':[]}))
            (root/'combat_final_combined.metrics.json').write_text('not JSON and must never be read')
            data=collect_data(root)
            self.assertEqual(data['combat'],[])
            self.assertEqual(len(data['sources']),1)

    def test_cohort_identity_ignores_completion_order(self):
        self.assertEqual(episode_summary({'task_ids':['a','b']})['cohort_id'],
                         episode_summary({'task_ids':['b','a']})['cohort_id'])
        self.assertEqual(phase_label('combat_0017'),'combat training rounds')

    def test_unknown_terminal_outcome_is_not_an_invented_loss(self):
        result=episode_summary({'episodes':[{'split':'dev','terminated':True,'outcome':'unknown'}]})
        self.assertEqual(result['decisive'],0)
        self.assertEqual(result['unresolved'],1)
        self.assertIsNone(result['win_rate_decisive'])
        self.assertIsNone(result['mean_hp_cost'])
        self.assertIsNone(result['cohort_id'])
