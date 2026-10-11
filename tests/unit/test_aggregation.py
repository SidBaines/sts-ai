from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from sts_policy.aggregation import aggregate_teachers
from sts_policy.curriculum import world_split


def row(seed, marker):
    return {"world_seed": seed, "split": world_split(seed), "teacher_index": 0,
            "marker": marker, "observation": {"phase": "combat", "legal_actions": [{"index": 0, "description": "end turn"}]}}


class AggregationTests(unittest.TestCase):
    def test_stream_reservoir_train_only_reproducible_sources_and_legality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root/'initial.jsonl', root/'dagger.jsonl'
            seeds = list(range(30))
            a.write_text(''.join(json.dumps(row(seed, f'a{seed}'))+'\n' for seed in seeds))
            b.write_text(''.join(json.dumps(row(seed, f'b{seed}'))+'\n' for seed in seeds))
            first = aggregate_teachers([a,b,a], root/'one.jsonl', max_records=9, seed=17)
            second = aggregate_teachers([b,a], root/'two.jsonl', max_records=9, seed=17)
            self.assertEqual((root/'one.jsonl').read_text(), (root/'two.jsonl').read_text())
            self.assertEqual(first['output_sha256'], second['output_sha256'])
            self.assertEqual(first['selected_records'], 9)
            self.assertEqual(len(first['sources']), 2)
            selected = [json.loads(line) for line in (root/'one.jsonl').read_text().splitlines()]
            self.assertTrue(all(world_split(r['world_seed'])=='train' for r in selected))
            self.assertTrue(all('aggregation_source' in r for r in selected))
            self.assertEqual(set(first['selected_source_counts']), {str(a.resolve()),str(b.resolve())})

    def test_illegal_train_label_and_split_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source.jsonl'
            seed=next(s for s in range(100) if world_split(s)=='train')
            bad=row(seed,'x');bad['teacher_index']=1
            source.write_text(json.dumps(bad)+'\n')
            with self.assertRaises(ValueError): aggregate_teachers([source],root/'out.jsonl')
            bad=row(seed,'x');bad['split']='final'
            source.write_text(json.dumps(bad)+'\n')
            with self.assertRaises(ValueError): aggregate_teachers([source],root/'out.jsonl')
            with self.assertRaises(ValueError): aggregate_teachers([source],source)
