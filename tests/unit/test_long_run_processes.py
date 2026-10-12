"""Real bounded owned-process cleanup; never signal an unrelated process group."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from sts_policy.long_run_control import (LongRunConfig, LongRunController, digest,
    process_birth, stop_owned_group, track_owned_members)

LEADER_SOURCE = '''import multiprocessing as mp
import os,sys,time
from pathlib import Path

def sleeper():
    time.sleep(30)

def wait_gate(path):
    limit=time.monotonic()+8
    while not path.exists():
        if time.monotonic()>limit:os._exit(24)
        time.sleep(.01)

if __name__=='__main__':
    wait_gate(Path(sys.argv[1]))
    child=mp.get_context('spawn').Process(target=sleeper,daemon=True)
    child.start()
    print(child.pid,flush=True)
    wait_gate(Path(sys.argv[2]))
    os._exit(23)
'''


def process_state(pid: int) -> str:
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='],
                            capture_output=True, text=True, timeout=2)
    return result.stdout.strip()


def running(pid: int) -> bool:
    state = process_state(pid)
    return bool(state) and not state.startswith('Z')


def exercise_cleanup(*, observed: bool, tamper: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        script = root / 'owned_leader.py'; script.write_text(LEADER_SOURCE)
        spawn_gate, crash_gate = root / 'spawn', root / 'crash'
        command = [sys.executable, str(script), str(spawn_gate), str(crash_gate)]
        leader = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE, text=True)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child = None
        try:
            # Verify session creation before permitting any child fork.
            birth = process_birth(leader.pid)
            assert birth is not None and os.getpgid(leader.pid) == leader.pid
            record = {'pid': leader.pid, 'pgid': leader.pid, 'process_birth': birth,
                      'command': command, 'owned_members': {str(leader.pid): birth},
                      'group_creation': {'new_session': True, 'leader_pid': leader.pid,
                         'leader_birth': birth, 'started_at': time.time(),
                         'command_sha256': digest(command), 'max_lifetime_seconds': 15.}}
            spawn_gate.touch()
            with selectors.DefaultSelector() as ready:
                ready.register(leader.stdout, selectors.EVENT_READ)
                assert ready.select(4), 'Owned child did not start within bound'
            child = int(leader.stdout.readline())
            assert running(child) and os.getpgid(child) == leader.pid
            if observed:
                track_owned_members(record)
                assert str(child) in record['owned_members']
            else:
                assert str(child) not in record['owned_members']
            crash_gate.touch()
            leader.wait(timeout=4)
            assert leader.returncode == 23 and running(child) and unrelated.poll() is None
            record['group_creation']['leader_exit_observed_at'] = time.time()
            if tamper:
                record['group_creation']['command_sha256'] = 'not-the-created-command'
            result = stop_owned_group(record, grace=.2)
            deadline = time.monotonic() + 3
            while result and running(child) and time.monotonic() < deadline:
                time.sleep(.02)
            return {'observed_before_exit': observed, 'tampered_creation': tamper,
                    'cleanup_returned': result, 'owned_child_stopped': not running(child),
                    'unrelated_alive': unrelated.poll() is None,
                    'leader_pid': leader.pid, 'child_pid': child, 'unrelated_pid': unrelated.pid,
                    'owned_members': record['owned_members'], 'group_creation': record['group_creation']}
        finally:
            # These are only the exact groups created above by this test.
            try: os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            if leader.poll() is None: leader.wait(timeout=3)
            leader.stdout.close()
            unrelated.terminate(); unrelated.wait(timeout=3)


@unittest.skipUnless(os.name == 'posix', 'owned Unix process-group test')
class LongRunProcessTests(unittest.TestCase):
    def test_recorded_orphan_stopped_after_leader_crash_unrelated_survives(self):
        result = exercise_cleanup(observed=True)
        self.assertTrue(result['cleanup_returned'])
        self.assertTrue(result['owned_child_stopped'])
        self.assertTrue(result['unrelated_alive'])

    def test_unrecorded_orphan_backfilled_from_verified_creation_after_leader_crash(self):
        result = exercise_cleanup(observed=False)
        self.assertTrue(result['cleanup_returned'])
        self.assertTrue(result['owned_child_stopped'])
        self.assertTrue(result['unrelated_alive'])

    def test_altered_creation_proof_cannot_signal_unrecorded_members(self):
        result = exercise_cleanup(observed=False, tamper=True)
        self.assertFalse(result['cleanup_returned'])
        self.assertFalse(result['owned_child_stopped'])
        self.assertTrue(result['unrelated_alive'])

    def test_duplicate_controller_cannot_change_active_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build = root / 'build'; build.mkdir()
            module = build / 'slaythespire.fake.so'; module.write_bytes(b'native')
            import hashlib
            (build / 'build_manifest.json').write_text(json.dumps({'module_file': module.name,
                'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest()}))
            checkpoint = root / 'initial.pt'; checkpoint.write_bytes(b'initial')
            config = LongRunConfig(checkpoint=str(checkpoint), backend_build=str(build),
                pool_bank=str(root/'bank'), pool_panels=str(root/'panels'),
                deadline=time.time()+60, device='cpu', prevent_sleep=False)
            source_root = Path(__file__).resolve().parents[2]
            first = LongRunController(source_root, root/'run', config)
            try:
                before = first.journal.read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'controller|holds'):
                    LongRunController(source_root, root/'run', config)
                self.assertEqual(first.journal.read_bytes(), before)
            finally:
                first.close()
            reopened = LongRunController(source_root, root/'run', config)
            reopened.close()
