"""Public-observation teacher process and explicitly named policy-preference rewards."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from .observations import legal_indices


class TeacherError(RuntimeError):
    pass


def validate_teacher(row: dict) -> dict:
    teacher = row['teacher']
    indices, probabilities = teacher['legal_indices'], teacher['probabilities']
    full = row['legal_indices']
    if (not indices or len(indices) != len(probabilities) or len(indices) != len(set(indices))
            or any(type(i) is not int or i not in full for i in indices)):
        raise ValueError('Teacher/menu index mismatch')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
           or p < 0 or p > 1 for p in probabilities):
        raise ValueError('Teacher probabilities must be finite and in [0,1]')
    if not math.isclose(sum(probabilities), 1., rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError('Teacher probabilities must sum to one')
    excluded = teacher.get('excluded_indices', [i for i in full if i not in indices])
    if set(excluded) != set(full) - set(indices):
        raise ValueError('Excluded teacher indices must exactly describe the menu projection')
    return teacher


def teacher_reward(row: dict, action_index: int | None, *, mode: str = 'probability',
                   invalid_reward: float | None = None, log_floor: float = 1e-8) -> float:
    """π_teacher(a|s), or log π, never a Q value or long-term-win estimate.

    A legal action omitted by the champion's deployment filter has probability0.
    Keep that case separate from malformed/illegal output in evaluation reports.
    """
    teacher = validate_teacher(row)
    if mode not in ('probability', 'log_probability'):
        raise ValueError('Reward mode must be probability or log_probability')
    if not 0 < log_floor <= 1:
        raise ValueError('Invalid log floor')
    if invalid_reward is None:
        invalid_reward = 0. if mode == 'probability' else math.log(log_floor)
    if not math.isfinite(invalid_reward):
        raise ValueError('Invalid output reward must be finite')
    if mode == 'log_probability' and invalid_reward > math.log(log_floor):
        raise ValueError('Log-probability invalid reward must not exceed log(log_floor)')
    if type(action_index) is not int or action_index not in row['legal_indices']:
        return float(invalid_reward)
    probability = dict(zip(teacher['legal_indices'], teacher['probabilities'])).get(action_index, 0.)
    return float(probability if mode == 'probability' else math.log(max(probability, log_floor)))


class NonLLMTeacher:
    """Finite-time isolated worker; parent imports only the clean environment.

    Teacher source/checkpoint are explicit dependencies, not vendored training
    code. Python must have the non-LLM runtime's torch and numpy installed.
    ``distributions`` sends at most ``batch_size`` observations per request, and
    ``timeout`` bounds startup and each request separately, so a large dataset
    needs no single long deadline.
    """
    def __init__(self, *, teacher_root: str, checkpoint: str, python: str | None = None,
                 timeout: float = 60., batch_size: int = 64,
                 expected_checkpoint_sha256: str | None = None):
        self.timeout, self._lock, self._counter, self._buffer = timeout, threading.Lock(), 0, b''
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError('Teacher timeout must be finite and positive')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Teacher batch size must be a positive integer')
        self.batch_size = batch_size
        self.teacher_root, self.checkpoint = Path(teacher_root).resolve(), Path(checkpoint).resolve()
        if expected_checkpoint_sha256 and hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != expected_checkpoint_sha256:
            raise ValueError('Teacher checkpoint differs from explicitly requested identity')
        self.stderr = tempfile.TemporaryFile(mode='w+b')
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        self.process = subprocess.Popen([python or sys.executable, '-u',
            str(Path(__file__).with_name('teacher_worker.py')), '--teacher-root', str(self.teacher_root),
            '--checkpoint', str(self.checkpoint)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, cwd=self.teacher_root, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self._read()
            if not ready.get('ready'):
                raise TeacherError('Teacher did not become ready')
            self.metadata = ready['metadata']
        except Exception:
            self.close()
            raise

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0 or not self.selector.select(left):
                raise TeacherError('Teacher RPC timed out')
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                self.stderr.seek(0)
                error = self.stderr.read().decode(errors='replace')[-4000:]
                raise TeacherError('Teacher worker exited: ' + error)
            self._buffer += block
            if len(self._buffer) > 64 * 1024**2:
                raise TeacherError('Teacher response exceeded64MiB')
        line, self._buffer = self._buffer.split(b'\n', 1)
        try:
            return json.loads(line)
        except ValueError as exc:
            raise TeacherError('Teacher emitted non-JSON output') from exc

    def _rpc(self, op, **payload):
        with self._lock:
            self._counter += 1
            request = {'id': self._counter, 'op': op, **payload}
            try:
                self.process.stdin.write(json.dumps(request, allow_nan=False).encode() + b'\n')
                self.process.stdin.flush()
                response = self._read()
                if response.get('id') != self._counter:
                    raise TeacherError('Teacher response ID mismatch')
                if 'error' in response:
                    raise TeacherError(str(response['error']))
                return response['result']
            except Exception:
                self.close()
                raise

    def distributions(self, observations: list[dict]) -> list[dict]:
        menus = [legal_indices(o) for o in observations]
        results = []
        for start in range(0, len(observations), self.batch_size):
            chunk = observations[start:start + self.batch_size]
            answer = self._rpc('distributions', observations=chunk)
            if not isinstance(answer, list) or len(answer) != len(chunk):
                raise TeacherError('Teacher returned incomplete batch')
            results.extend(answer)
        for full, teacher in zip(menus, results):
            validate_teacher({'legal_indices': full, 'teacher': teacher})
        return results

    def distribution(self, observation: dict) -> dict:
        return self.distributions([observation])[0]

    def refresh_metadata(self) -> dict:
        self.metadata = self._rpc('metadata')
        return self.metadata

    def export_runtime(self, destination: str, *, include_checkpoint: bool = False) -> dict:
        """Export only source files actually imported for inference, plus provenance.

        An exported runtime is an explicit dependency bundle. It does not enter
        sts_env and contains no native simulator binary. Verify it by querying
        the same observations before relying on it on another host.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        metadata = self.refresh_metadata()
        for record in metadata['runtime_files']:
            path = Path(record['path'])
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != record['sha256']:
                raise TeacherError('Teacher source changed during export')
            target = destination / 'src' / record['relative_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        if include_checkpoint:
            shutil.copyfile(self.checkpoint, destination / self.checkpoint.name)
        manifest = {**metadata, 'export_schema': 'sts_teacher_inference_runtime_v1',
                    'checkpoint_included': include_checkpoint}
        (destination / 'teacher_runtime_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        selector = getattr(self, 'selector', None)
        if selector is not None:
            selector.close()
        for stream in (getattr(process, 'stdin', None), getattr(process, 'stdout', None),
                       getattr(self, 'stderr', None)):
            if stream is not None and not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
