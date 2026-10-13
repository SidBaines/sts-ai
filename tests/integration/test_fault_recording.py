from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sts_env.env import SimulatorError, SimulatorTimeout
from sts_policy.curriculum import TaskSpec
from sts_policy.replay_fault import replay_fault
from sts_policy.workers import CurriculumWorker
from tests.support import requires_simulator


@requires_simulator
class NativeFaultRecordingTests(unittest.TestCase):
    def test_captured_discovery_fault_replays_with_the_study_backend(self):
        from sts_env._backend.lightspeed_import import native_metadata
        record=json.loads((Path(__file__).parents[1]/'fixtures/discovery_fault_6000000009.json').read_text())
        if native_metadata()['module_sha256'] not in {
                record['native_provenance']['module_sha256'],
                '40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd'}:
            self.skipTest('Historical trace regression is scoped to the verified study binaries')
        with tempfile.TemporaryDirectory() as td:
            result=replay_fault(record,allow_native_change=True,timeout=10,seconds=30,fault_dir=td)
        self.assertEqual(result['status'],'reproduced')
        self.assertEqual(result['rpc_index'],195)

    def test_reset_failure_replays_in_a_fresh_process_and_worker_recovers(self):
        with tempfile.TemporaryDirectory() as td:
            with CurriculumWorker({'fault_dir':td}) as worker:
                with self.assertRaises(SimulatorError):
                    worker.reset(TaskSpec(seed=3, expected_public_hash='intentional mismatch'))
                record = json.loads(Path(worker.last_fault_path).read_text())
                self.assertIsNone(worker._process)
                observation, _ = worker.reset(TaskSpec(seed=3))
                self.assertEqual(observation['phase'],'combat')
            result = replay_fault(record, timeout=10, seconds=30, fault_dir=str(Path(td)/'replay'))
            self.assertEqual(result['status'],'reproduced')
            self.assertEqual(result['method'],'reset')
            with self.assertRaises(ValueError): replay_fault(record, seconds=float('nan'))

    def test_timeout_saved_as_timeout_without_a_game_outcome(self):
        with tempfile.TemporaryDirectory() as td, CurriculumWorker({'fault_dir':td},timeout=.000001) as worker:
            with self.assertRaises(SimulatorTimeout): worker.reset(TaskSpec(seed=3))
            record = json.loads(Path(worker.last_fault_path).read_text())
            self.assertEqual(record['exception']['type'],'SimulatorTimeout')
            self.assertIsNone(record['last_observation'])
            self.assertNotIn('win',record)
            self.assertIsNone(worker._process)
