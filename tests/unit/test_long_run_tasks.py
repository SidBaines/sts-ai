from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import json
import io
import sys
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from scripts.train_nonllm import tasks_for
from scripts import train_nonllm
from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.long_run_tasks import (NATIVE_SHA, INTERFACE_SHA, ContinuationPool,
    load_continuations, sample_training_tasks, fixed_dev_panels, write_task_batch,
    task_identity, capture_learner_continuation, add_learner_continuations)


def provenance():
    return {'environment_version': '0.1.1', 'observation_version': 'sts_env_public_v1',
            'python_interface_sha256': INTERFACE_SHA, 'runtime_python_version': '3.12.13',
            'native_provenance': {'module_sha256': NATIVE_SHA, 'patch_sha256': 'patched'}}


def training_worlds(count=30):
    return [seed for seed in range(12000, 12500) if world_split(seed) == 'train'][:count]


def native_row(seed, act, floor=None):
    if floor is None:
        floor = 18 if act == 2 else 35
    task = TaskSpec(seed=seed, prefix=(0, 1) if act == 2 else (0, 1, 2),
                    ooc_previews=(True, False) if act == 2 else (True, False, False),
                    expected_public_hash='a' * 64, expected_provenance=provenance(),
                    max_act=3, seek_combat=False)
    return {**task.to_dict(), 'split': 'train', 'floor': floor, 'public_hash': 'a' * 64,
            'curriculum_metadata': {'act': act, 'floor': floor, 'source_bank': '/tmp/bank'}}


def pool():
    return ContinuationPool({act: tuple(native_row(seed, act) for seed in training_worlds()) for act in (2, 3)},
                            provenance(), {'manifest': {'path': '/tmp/bank/manifest.json', 'sha256': 'source'}})


class LongRunTasksTests(unittest.TestCase):
    def test_scratch_worlds_are_fresh_between_rounds_and_phase_streams(self):
        batches = [sample_training_tasks(pool(), stage='act2', stream=stream, round_index=round,
                    seed=11, episodes=20) for stream, round in [('act2_continuations', 0),
                    ('act2_continuations', 1), ('act2_full', 0)]]
        seen = set()
        for batch in batches:
            worlds = {task.seed for task in batch['tasks'] if not task.prefix}
            self.assertTrue(worlds)
            self.assertFalse(worlds & seen)
            self.assertTrue(all(seed >= 1000000 and world_split(seed) == 'train' for seed in worlds))
            seen |= worlds
            self.assertEqual(batch['metadata']['counts']['full3'], 4)
        self.assertEqual([task.to_dict() for task in batches[0]['tasks']],
                         [task.to_dict() for task in sample_training_tasks(pool(), stage='act2',
                         round_index=0, seed=11, episodes=20)['tasks']])

    def test_continuations_and_rehearsal_keep_separate_start_goal_families(self):
        batch = sample_training_tasks(pool(), stage='full3', round_index=2, seed=17, episodes=20)
        counts = {}
        for task, row in zip(batch['tasks'], batch['metadata']['tasks']):
            self.assertEqual(row['task_id'], task_identity(task))
            self.assertEqual(row['goal_act'], task.max_act)
            counts[row['component']] = counts.get(row['component'], 0) + 1
            if row['component'] in ('act2', 'act3'):
                self.assertTrue(task.prefix)
                self.assertEqual(task.kind, 'run')
                self.assertEqual(task.max_act, 3)
                self.assertEqual(row['start_source'], 'native_teacher')
            elif row['component'] == 'rehearsal':
                self.assertEqual(task.max_act, 1)
        self.assertEqual(counts, {'act2': 3, 'act3': 3, 'scratch': 12, 'rehearsal': 2})

    def test_bank_metadata_provenance_and_train_split_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);bank=root/'bank';panels=root/'panels';bank.mkdir();panels.mkdir()
            manifest = {'settings': {'max_act': 3}, 'provenance': provenance()}
            (bank/'manifest.json').write_text(json.dumps(manifest))
            for act in (2, 3):
                row = native_row(training_worlds()[0], act)
                row['curriculum_metadata']['source_bank'] = str(bank)
                (panels/f'train_act{act}.jsonl').write_text(json.dumps(row)+'\n')
            valid = load_continuations(bank, metadata_panels=panels)
            self.assertEqual(len(valid.by_act[3]), 1)
            path = panels/'train_act2.jsonl';row=json.loads(path.read_text())
            row['seed'] = next(seed for seed in range(12000, 12500) if world_split(seed)=='final')
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'Non-TRAIN'):
                load_continuations(bank, metadata_panels=panels)
            manifest['provenance']['python_interface_sha256'] = 'old'
            (bank/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'corrected native'):
                load_continuations(bank, metadata_panels=panels)

    def test_new_eval_panels_fixed_disjoint_and_artifacts_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);definition=fixed_dev_panels(out, episodes=8)
            self.assertEqual(definition, fixed_dev_panels(out, episodes=8))
            dev=set(definition['metadata']['worlds']['dev']);final=set(definition['metadata']['worlds']['final'])
            self.assertFalse(dev & final)
            self.assertTrue(all(200000<=seed<1000000 and world_split(seed)=='dev' for seed in dev))
            self.assertTrue(all(200000<=seed<1000000 and world_split(seed)=='final' for seed in final))
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                fixed_dev_panels(out, episodes=9)
            batch=sample_training_tasks(pool(),stage='full3',round_index=0,seed=1,episodes=20)
            artifact=write_task_batch(batch,out/'train.jsonl')
            self.assertEqual(artifact,write_task_batch(batch,out/'train.jsonl'))
            batch['tasks'][0]=replace(batch['tasks'][0],horizon=99)
            with self.assertRaisesRegex(ValueError,'Immutable|identity'):
                write_task_batch(batch,out/'train.jsonl')

    def test_cli_opt_in_retains_multiple_same_world_prefixes_and_per_task_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tasks.jsonl';world=training_worlds()[0]
            tasks=[replace(TaskSpec.from_dict(native_row(world,2)),kind='run',max_act=2,horizon=500),
                   replace(TaskSpec.from_dict(native_row(world,3)),kind='run',max_act=3,horizon=800)]
            path.write_text(''.join(json.dumps(task.to_dict())+'\n' for task in tasks))
            args=Namespace(tasks=path,task_kind='run',horizon=256,max_act=1,simulations=50,
                           split='train',seed=1,episodes=8,preserve_task_prefixes=True)
            preserved=tasks_for(args)
            self.assertEqual({task_identity(t) for t in preserved},{task_identity(t) for t in tasks})
            args.preserve_task_prefixes=False
            legacy=tasks_for(args)
            self.assertEqual(len(legacy),1)
            self.assertEqual((legacy[0].prefix,legacy[0].max_act,legacy[0].horizon),((),1,256))

    def test_capture_cli_rejects_eval_wrong_kind_and_overlapping_artifacts_before_model_load(self):
        conditions = [(['evaluate', '--task-kind', 'run'], 'capture.jsonl'),
                      (['ppo', '--task-kind', 'battle'], 'capture.jsonl'),
                      (['ppo', '--task-kind', 'run'], 'out.pt')]
        for tokens, capture in conditions:
            with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ['train_nonllm.py', *tokens, '--output', str(root / 'out.pt'),
                        '--continuation-output', str(root / capture)]
                with patch.object(sys, 'argv', argv), patch('sys.stderr', io.StringIO()), \
                     patch.object(train_nonllm, 'source_model') as model:
                    with self.assertRaises(SystemExit) as error:
                        train_nonllm.main()
                    self.assertEqual(error.exception.code, 2)
                model.assert_not_called()

    def test_invalid_mixtures_and_namespace_bounds_fail(self):
        for kwargs in [{'world_start':999999}, {'round_index':-1},
                       {'fractions':{'act2':.75,'act3':0.,'scratch':.15,'rehearsal':.10}}]:
            options={'stage':'act2','round_index':0,'seed':11,**kwargs}
            with self.assertRaises(ValueError):sample_training_tasks(pool(),**options)

    def test_live_learner_entry_capture_and_terminal_or_wrong_provenance_rejection(self):
        task=TaskSpec(seed=1000000,kind='run',max_act=3,horizon=4096)
        if world_split(task.seed)!='train':task=replace(task,seed=next(s for s in range(1000000,1000100) if world_split(s)=='train'))
        previous={'state':{'act':1}}
        observation={'state':{'act':2,'floor':17},'phase':'out_of_combat','legal_actions':[{'index':0}]}
        info={**provenance(),'outcome':'GameOutcome.UNDECIDED','action_prefix':[1,0], 'ooc_previews':[False,False]}
        row=capture_learner_continuation(task,previous,observation,info,terminated=False,truncated=False)
        self.assertEqual(row['long_metadata']['start_source'],'learner')
        self.assertEqual(row['long_metadata']['start_act'],2)
        self.assertIsNone(capture_learner_continuation(task,previous,observation,info,terminated=True,truncated=False))
        self.assertIsNone(capture_learner_continuation(replace(task,max_act=2),previous,observation,info,terminated=False,truncated=False))
        bad={**info,'python_interface_sha256':'old'}
        with self.assertRaisesRegex(ValueError,'different backend'):
            capture_learner_continuation(task,previous,observation,bad,terminated=False,truncated=False)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'learner.jsonl';path.write_text(json.dumps(row)+'\n')
            merged=add_learner_continuations(pool(),[path])
            self.assertEqual(len(merged.by_act[2]),len(pool().by_act[2])+1)
            native_world = training_worlds()[0]
            replacement = capture_learner_continuation(replace(task, seed=native_world), previous,
                observation, info, terminated=False, truncated=False)
            path.write_text(json.dumps(replacement)+'\n')
            changed = add_learner_continuations(pool(), [path])
            self.assertEqual(len(changed.by_act[2]), len(pool().by_act[2]))
            record = changed.source_identity['learner_replacements'][0]
            self.assertEqual(record['world_seed'], native_world)
            self.assertEqual(record['previous_start_source'], 'native_teacher')
            self.assertEqual(record['chosen_start_source'], 'learner')
