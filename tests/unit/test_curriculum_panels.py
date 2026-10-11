from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from scripts.build_curriculum_panels import build_panels
from sts_policy.curriculum import TaskSpec, world_split


class PanelTests(unittest.TestCase):
    def test_eval_panels_do_not_oversample_and_train_uses_act_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bank=root/'bank'; source=bank/'stage_subsets'
            source.mkdir(parents=True); (bank/'manifest.json').write_text('{}')
            seeds={split:[s for s in range(100) if world_split(s)==split] for split in ('train','dev','final')}
            for act in (1,2,3):
                rows=[]
                for split in seeds:
                    chosen=seeds[split][:1 if act==3 else 5]
                    for seed in chosen:
                        for offset in range(2):
                            rows.append({**TaskSpec(seed=seed, prefix=(act,offset), expected_public_hash=f'{act}-{seed}-{offset}').to_dict(),
                                         'floor':(act-1)*17+offset+1,
                                         'curriculum_metadata':{'act':act}})
                (source/f'battles_act{act}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=build_panels([bank,bank],root/'panels',per_act=4)
            self.assertEqual(len(result['source_banks']),1)
            self.assertEqual(result['panels']['dev_mixed']['tasks'],10)
            self.assertEqual(result['panels']['final_mixed']['tasks'],10)
            self.assertEqual(result['panels']['dev_mixed']['by_act']['3']['world_count'],1)
            self.assertFalse(result['panels']['dev_mixed']['by_act']['3']['world_uncertainty_available'])
            self.assertEqual(len(result['panels']['dev_mixed']['task_ids']),len(set(result['panels']['dev_mixed']['task_ids'])))
            self.assertEqual(len(result['train_round_robin_paths']),3)
            for act in (1,2,3):
                rows=[json.loads(s) for s in (root/'panels'/f'train_act{act}.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['split']=='train' and row['curriculum_metadata']['act']==act for row in rows))
