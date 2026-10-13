from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class QueueRecordingTests(unittest.TestCase):
    def check_completion(self, *, child_code, fresh_done):
        spec=importlib.util.spec_from_file_location('rl_queue_test',Path(__file__).parents[2]/'scripts/rl/queue_runner.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';out.mkdir()
            old='{"event": "done"}\n'
            (out/'log.jsonl').write_text(old+(' {"event": "done"}\n' if fresh_done else ''))
            run={'name':'run','cmd':['python','--out',str(out)],'status':'running','pid':123,
                 'log_start_bytes':len(old)}
            child=Mock();child.poll.return_value=child_code
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(module,'ROOT',root))
                stack.enter_context(patch.object(module,'STATUS',root/'status.json'))
                stack.enter_context(patch.object(module,'Locked',return_value=contextlib.nullcontext()))
                stack.enter_context(patch.object(module,'read_queue',return_value={'runs':[run]}))
                stack.enter_context(patch.object(module,'write_queue'))
                stack.enter_context(patch.object(module,'on_battery',return_value=False))
                stack.enter_context(patch.object(module,'alive',return_value=False))
                stack.enter_context(patch.object(module.os,'waitpid',side_effect=ChildProcessError))
                stack.enter_context(patch.object(module.time,'sleep',side_effect=StopIteration))
                if child_code is not None:
                    module.LAUNCHED[123]=child
                with self.assertRaises(StopIteration): module.runner()
            return run

    def test_known_nonzero_exit_cannot_be_overruled_by_done_marker(self):
        run=self.check_completion(child_code=2,fresh_done=True)
        self.assertEqual(run['returncode'],2)
        self.assertEqual(run['status'],'failed')

    def test_lost_child_exit_does_not_reuse_done_from_prior_launch(self):
        run=self.check_completion(child_code=None,fresh_done=False)
        self.assertIsNone(run['returncode'])
        self.assertEqual(run['status'],'failed')
