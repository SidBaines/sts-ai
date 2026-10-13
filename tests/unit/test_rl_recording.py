from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile
import unittest
import contextlib
import importlib.util
import io
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec
from sts_policy.faults import FaultRecordingError
from sts_policy.workers import CurriculumWorker
if torch is not None:
    from sts_rl.common import save_atomic, save_policy
from sts_rl.recording import begin_run, capture_rng, file_identity, restore_rng, resume_payload, require_fresh_or_resume


@unittest.skipIf(torch is None, 'optional torch unavailable')
class RecordingTests(unittest.TestCase):
    def test_existing_history_cannot_be_restarted_or_missing_state_resumed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            require_fresh_or_resume(root,False)
            with self.assertRaises(ValueError): require_fresh_or_resume(root,True)
            (root/'config.json').write_text('{}')
            with self.assertRaises(ValueError): require_fresh_or_resume(root,False)
            (root/'latest.pt').write_bytes(b'preserved')
            require_fresh_or_resume(root,True)
    def test_launch_preserves_sources_binary_inputs_and_deduplicates_copies(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=False):
            root = Path(td)
            build = root/'build'; build.mkdir()
            (build/'slaythespire.fake.so').write_bytes(b'fake native')
            native = {'module_file':'slaythespire.fake.so', 'module_sha256':file_identity(build/'slaythespire.fake.so')['sha256']}
            (build/'build_manifest.json').write_text(json.dumps(native))
            parent = root/'parent.pt'; parent.write_bytes(b'parent')
            data = root/'data.jsonl'; data.write_text('{}\n')
            with patch('sts_env._backend.lightspeed_import.native_metadata', return_value=native):
                a = begin_run(root/'runA', {'seed':7}, parent, [data], build_dir=build)
                b = begin_run(root/'runB', {'seed':7}, parent, [data], build_dir=build)
            launch = json.loads(Path(a['path']).read_text())
            self.assertEqual(launch['parent_checkpoint']['sha256'], file_identity(parent)['sha256'])
            self.assertEqual(launch['data_inputs'][0]['sha256'], file_identity(data)['sha256'])
            self.assertEqual(launch['effective_config'], {'seed':7})
            self.assertEqual(file_identity(a['path'])['sha256'], a['sha256'])
            self.assertTrue(Path(launch['source_archive']['path']).exists())
            pa, pb = Path(a['path']).parent, Path(b['path']).parent
            self.assertEqual((pa/native['module_file']).stat().st_ino, (pb/native['module_file']).stat().st_ino)
            self.assertEqual((pa/'source.zip').stat().st_ino, (pb/'source.zip').stat().st_ino)

    def test_complete_resume_restores_random_streams_optimizer_and_local_sampler(self):
        with tempfile.TemporaryDirectory() as td:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            model(torch.ones(1,2)).sum().backward(); optimizer.step()
            local = random.Random(7)
            torch.manual_seed(9); np.random.seed(3); random.seed(2)
            path = Path(td)/'latest.pt'
            save_atomic(resume_payload(model, optimizer, 4, 10, {'seed':7}, {'path':'launch'}, local), path)
            expected = (torch.rand(5), np.random.rand(5), random.random(), local.random())
            state = torch.load(path, weights_only=False)
            self.assertTrue(restore_rng(state, local))
            torch.testing.assert_close(torch.rand(5), expected[0], rtol=0, atol=0)
            np.testing.assert_array_equal(np.random.rand(5), expected[1])
            self.assertEqual((random.random(), local.random()), expected[2:])
            other = torch.optim.AdamW(model.parameters(), lr=.1); other.load_state_dict(state['optimizer_state'])
            self.assertEqual(other.param_groups[0]['lr'], .001)
            self.assertEqual(state['iteration'], 4)
            self.assertFalse(restore_rng({'torch_rng':torch.get_rng_state()}, local))

    def test_named_checkpoint_does_not_claim_stale_template_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)/'parent.pt'; parent.write_bytes(b'initial')
            model = torch.nn.Linear(2,1)
            old = {'metadata': {'training_environment':{'native':'OLD'}, 'data_identity':'OLD',
                               'parent_policy_checkpoint':{'path':'OLD'}, 'encoder_version':'public_candidate_v2'}}
            path = Path(td)/'iter.pt'
            local = random.Random(3)
            optimizer = torch.optim.AdamW(model.parameters())
            save_policy(path, model, old, metadata={'stage':'new', 'iteration':2, 'config':{'checkpoint':str(parent)}},
                        optimizer=optimizer, step=5, rng=local)
            state = torch.load(path, weights_only=False)
            self.assertNotIn('training_environment', state['metadata'])
            self.assertNotIn('data_identity', state['metadata'])
            self.assertEqual(state['metadata']['parent_policy_checkpoint']['sha256'], file_identity(parent)['sha256'])
            self.assertEqual(state['rng_state']['local_python'], local.getstate())
            self.assertEqual((state['iteration'], state['steps_total']), (2,5))
            self.assertIsNotNone(state['optimizer_state'])

    def test_failed_step_records_exact_successful_calls_and_attempt_before_cleanup(self):
        observation = {'phase':'combat','state':{},'legal_actions':[{'index':0}]}
        class Process:
            pid=123
            exitcode=None
        class Connection:
            def __init__(self): self.calls=iter([
                ('ok',(observation,{})), ('ok',0), ('ok',(observation,0,False,False,{})),
                ('error',{'type':'RuntimeError','message':'undefined behavior','traceback':'child trace'})])
            def send(self, value): pass
            def poll(self, timeout): return True
            def recv(self): return next(self.calls)
        with tempfile.TemporaryDirectory() as td, CurriculumWorker({'fault_dir':td}) as worker:
            worker._process=Process(); worker._connection=Connection()
            with patch.object(worker,'_kill',side_effect=lambda:setattr(worker,'_process',None)):
                worker.reset(TaskSpec(seed=3,kind='run'))
                worker.teacher_action(50,7)
                worker.step(0)
                with self.assertRaises(SimulatorError): worker.step(0)
            record=json.loads(Path(worker.last_fault_path).read_text())
            self.assertEqual([r['method'] for r in record['successful_rpcs']],['reset','teacher_action','step'])
            self.assertEqual(record['successful_rpcs'][1]['args'],[50,7])
            self.assertEqual(record['attempted_rpc'],{'method':'step','args':[0]})
            self.assertEqual(record['exception']['child']['traceback'],'child trace')
            self.assertEqual(record['task']['seed'],3)
            self.assertEqual(len(record['last_public_hash']),64)
            worker._connection=None

    def test_fault_log_failure_is_explicit(self):
        from sts_policy.faults import write_fault
        with tempfile.TemporaryDirectory() as td:
            bad=Path(td)/'not_directory';bad.write_text('occupied')
            with self.assertRaises(FaultRecordingError): write_fault(bad,{'exception':'failure'})

    def test_helper_quarantines_labels_and_fight_starts_when_world_later_faults(self):
        from concurrent.futures import Future
        combat={'phase':'combat','state':{'room':'Room.BOSS','act':1,'floor':16,'cur_hp':80,'max_hp':80},
                'legal_actions':[{'index':0,'description':'play Strike'},{'index':1,'description':'end turn'}]}
        ooc={'phase':'out_of_combat','state':{'room':'Room.BOSS','act':1,'floor':16},
             'legal_actions':[{'index':0,'description':'take gold'}]}
        for script in ['collect_combat_labels','collect_fight_bank']:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as td:
                spec=importlib.util.spec_from_file_location(script,Path(__file__).parents[2]/'scripts/rl'/f'{script}.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                class Worker:
                    last_fault_path='captured-fault.json'
                    def __init__(self,*args,**kwargs): self.steps=0
                    def close(self): pass
                    def search_votes(self,*args): return {0:1}
                    def submit(self,method,*args):
                        future=Future()
                        if method=='reset': future.set_result((combat,{}))
                        elif method=='step':
                            self.steps+=1
                            if script=='collect_fight_bank' and self.steps==1:
                                future.set_result((ooc,0,False,False,{}))
                            else: future.set_exception(SimulatorError('later native fault'))
                        return future
                output=Path(td)/('labels' if script=='collect_combat_labels' else 'bank.jsonl')
                argv=[script,'--checkpoint','fake.pt','--threads','1','--n','1',
                      '--record-dir' if script=='collect_combat_labels' else '--output',str(output)]
                logits=(torch.tensor([[1.,0.]]),torch.zeros(1),torch.ones(1,2,dtype=torch.bool))
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(module,'begin_run',return_value={}))
                    stack.enter_context(patch.object(module,'load_policy',return_value=(torch.nn.Identity(),{})))
                    stack.enter_context(patch.object(module,'train_tasks',return_value=[TaskSpec(seed=3,kind='run')]))
                    stack.enter_context(patch.object(module,'PanelWorker' if script=='collect_combat_labels' else 'CurriculumWorker',Worker))
                    stack.enter_context(patch.object(module,'forward_policy',return_value=logits))
                    stack.enter_context(patch.object(module,'collate_public',return_value=None))
                    stack.enter_context(patch('sys.argv',argv))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    module.main()
                if script=='collect_combat_labels':
                    self.assertEqual((output/'shard_00.jsonl').read_text(),'')
                    row=json.loads((output/'progress.jsonl').read_text())
                    self.assertEqual(row['discarded_rows'],1)
                    self.assertEqual(row['fault_record_path'],'captured-fault.json')
                else:
                    self.assertEqual(output.read_text(),'')
