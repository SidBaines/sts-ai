#!/usr/bin/env python3
"""Independent, CPU-only DEV observer. Never writes the learner's artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

from sts_policy.campaign_identity import condition_signature
from sts_policy.long_run_control import (
    FileLock, atomic_json, digest, panel_quality, process_birth, resource_pause,
    sha_file, stop_owned_group, track_owned_members,
)

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('selected_best', 'latest_candidate')


def snapshot_checkpoint(source: Path, directory: Path, expected_sha: str) -> Path:
    """Keep one immutable copy from an open FD, including across source unlink."""
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise RuntimeError('Snapshot directory must not be a symlink')
    target = directory / (expected_sha + '.pt')
    if target.exists():
        if sha_file(target) != expected_sha:
            raise RuntimeError('Existing snapshot was modified')
        return target
    temporary = directory / (expected_sha + f'.{os.getpid()}.tmp')
    try:
        with source.open('rb') as reader, temporary.open('xb') as writer:
            before = os.fstat(reader.fileno())
            checksum = hashlib.sha256()
            for chunk in iter(lambda: reader.read(1 << 20), b''):
                checksum.update(chunk); writer.write(chunk)
            writer.flush(); os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError('Checkpoint changed during capture')
            reader.seek(0)
            verify = hashlib.sha256()
            for chunk in iter(lambda: reader.read(1 << 20), b''):
                verify.update(chunk)
            if checksum.hexdigest() != expected_sha or verify.hexdigest() != expected_sha:
                raise RuntimeError('Checkpoint bytes disagree with completed artifact SHA')
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def completed_candidates(learner: dict[str, Any]) -> list[dict[str, Any]]:
    completed = [j for j in learner.get('jobs', []) if j.get('kind') == 'train' and j.get('status') == 'complete']
    latest = max(completed, key=lambda j: j['finished']) if completed else None
    paths = {'selected_best': learner['best_checkpoint']}
    if latest: paths['latest_candidate'] = latest['checkpoint']
    captures = []
    for role, path in paths.items():
        sha = learner['identity']['initial_checkpoint_sha256'] if path == learner['initial_checkpoint'] else None
        for job in completed:
            if path == job.get('checkpoint'):
                sha = job.get('artifacts', {}).get(path)
        if not sha:
            raise RuntimeError('Requested role is not a SHA-bound completed checkpoint: ' + role)
        captures.append({'role': role, 'source_checkpoint': path, 'checkpoint_sha256': sha,
                         'captured_global_round': learner['global_round'], 'captured_at': time.time()})
    return captures


def learner_permission(learner: dict[str, Any], *, now: float | None = None, check_birth=True) -> str | None:
    now = time.time() if now is None else now
    if learner.get('status') != 'running': return 'learner_' + str(learner.get('status'))
    allocation = learner['allocation']
    if allocation['used_seconds'] >= allocation['authorized_seconds']: return 'allocation_exhausted'
    hard = learner['identity']['config'].get('hard_wall_deadline')
    if hard is not None and now >= hard: return 'hard_deadline'
    controller = learner.get('controller', {})
    if check_birth and process_birth(controller.get('pid', 0)) != controller.get('process_birth'):
        return 'controller_not_live'
    return None


def build_command(checkpoint: Path, metrics: Path, protocol: dict[str, Any], *, seconds: float = 300.) -> list[str]:
    panel = protocol['panels']['dev_act3']
    return ['/usr/bin/nice', '-n', '15', str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/train_nonllm.py'),
            'evaluate', '--checkpoint', str(checkpoint), '--output', str(metrics.with_suffix('.unused.pt')),
            '--metrics', str(metrics), '--tasks', panel['path'], '--preserve-task-prefixes',
            '--task-kind', 'run', '--split', 'dev', '--max-act', '3', '--horizon', '4096',
            '--episodes', str(len(panel['task_ids'])), '--seed', '777', '--simulations', '200',
            '--seconds', str(seconds), '--device', 'cpu', '--inference-device', 'cpu', '--workers', '4',
            '--threads', '1', '--fast-observations', '--fast-replay', '--deferred-bootstrap',
            '--gamma', '1', '--gae-lambda', '.95', '--clip-ratio', '.1', '--entropy-coef', '.01',
            '--learning-rate', '.0001', '--reward-mode', 'progress', '--progress-floor-weight', '.05',
            '--terminal-hp-weight', '0', '--goal-act-clear', '--kl-guard-before-update']


def outcome_record(metrics: dict[str, Any], expected_ids: list[str]) -> dict[str, Any]:
    quality = panel_quality(metrics)
    if Counter(quality['cohort']) != Counter(expected_ids):
        raise RuntimeError('Evaluation requested a different DEV cohort')
    rounds = metrics['rounds']
    episodes = [e for r in rounds for e in r.get('episodes', [])]
    faults = [e for r in rounds for e in r.get('faults', [])]
    quality.update(task_ids=quality.pop('cohort'),
                   task_ids_sha256=digest(sorted(expected_ids)),
                   defeats=sum(e.get('outcome') == 'defeat' for e in episodes),
                   unstarted=sum(r.get('unstarted_tasks', 0) for r in rounds),
                   fault_records=faults,
                   coverage_complete=quality['complete'])
    return quality


def observer_condition_key(signature, pins):
    # Observer presentation/source repairs do not change this scientific panel.
    scientific_pins = {k: v for k, v in pins.items() if k != 'observer_source_sha256'}
    return digest({'condition': signature, 'pins': scientific_pins})


def restore_state(previous, pins):
    old_science = {k: v for k, v in previous['pins'].items() if k != 'observer_source_sha256'}
    new_science = {k: v for k, v in pins.items() if k != 'observer_source_sha256'}
    if old_science != new_science:
        raise RuntimeError('Observer scientific inputs changed')
    if previous['pins'] != pins:
        previous.setdefault('observer_source_amendments', []).append({
            'at': time.time(), 'before': previous['pins'].get('observer_source_sha256'),
            'after': pins.get('observer_source_sha256'), 'reason': 'observer-only repair; scientific inputs unchanged'})
    previous['pins'] = pins
    for result in previous['histories'] + previous.get('jobs', []):
        if result.get('condition_signature'):
            result['condition_key'] = observer_condition_key(result['condition_signature'], pins)
    return previous


class Observer:
    def __init__(self, learner_path: Path, protocol_path: Path, output: Path, *, interval=1800., seconds=300.):
        self.learner_path, self.protocol_path, self.output = learner_path, protocol_path, output.resolve()
        self.interval, self.seconds, self.stop = interval, seconds, False
        self.output.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.output / 'journal.json'
        self.learner = json.loads(learner_path.read_text())
        if self.output == learner_path.parent.resolve() or self.output.is_relative_to(learner_path.parent.resolve()):
            raise RuntimeError('Observer must use a separate sibling output directory')
        self.protocol = json.loads(protocol_path.read_text())
        self.panel = self.protocol['panels']['dev_act3']
        self.pins = {'observer_source_sha256': sha_file(__file__), 'protocol_sha256': sha_file(protocol_path), 'task_sha256': self.panel['sha256'],
                     'learner_identity': digest(self.learner['identity']), 'backend': self.learner['identity']['backend']}
        self.state = json.loads(self.journal_path.read_text()) if self.journal_path.exists() else {
            'version': 1, 'monitoring_only': True, 'pins': self.pins, 'jobs': [], 'histories': [],
            'latest': {}, 'captures': [], 'next_due': 0, 'status': 'starting'}
        self.state = restore_state(self.state, self.pins)
        self.validate_inputs()
        self.load_baseline()

    def validate_inputs(self):
        from sts_policy.curriculum import TaskSpec, world_split
        from sts_policy.rollout_training import task_identity
        from sts_env.provenance import interface_identity
        if sha_file(self.protocol_path) != self.pins['protocol_sha256'] or sha_file(self.panel['path']) != self.pins['task_sha256']:
            raise RuntimeError('Fixed DEV protocol/tasks changed')
        rows = [TaskSpec.from_dict(json.loads(line)) for line in Path(self.panel['path']).read_text().splitlines() if line]
        if (len(rows) != 48 or self.panel['max_act'] != 3 or self.panel['horizon'] != 4096 or
                sorted(task_identity(row) for row in rows) != sorted(self.panel['task_ids']) or
                any(row.kind != 'run' or row.max_act != 3 or row.horizon != 4096 or row.battle_simulations != 200 or
                    row.seed < 200000 or world_split(row.seed) != 'dev' for row in rows)):
            raise RuntimeError('Expected the prespecified 48-world DEV Act3 panel')
        backend = self.pins['backend']; build = Path(backend['build_dir'])
        manifest = build / 'build_manifest.json'
        if not manifest.exists():
            candidates = list(build.glob('*manifest*.json'))
            if len(candidates) != 1: raise RuntimeError('Cannot identify pinned backend manifest')
            manifest = candidates[0]
        if sha_file(manifest) != backend['manifest_sha256'] or sha_file(build / backend['provenance']['module_file']) != backend['provenance']['module_sha256']:
            raise RuntimeError('Pinned backend changed')
        sources = self.learner['identity']['sources']
        for path, sha in sources.items():
            if sha_file(ROOT / path) != sha: raise RuntimeError('Pinned learner source changed')
        self.interface_sha = interface_identity()['python_interface_sha256']

    def load_baseline(self):
        anchor = self.protocol['anchors']['dev_act3']
        if sha_file(anchor['metrics']) != anchor['metrics_sha256'] or anchor['checkpoint_sha256'] != self.protocol['checkpoint']['sha256']:
            raise RuntimeError('Baseline artifacts changed')
        if condition_signature(anchor['command']) != anchor['condition_signature']:
            raise RuntimeError('Baseline condition changed')
        metrics = json.loads(Path(anchor['metrics']).read_text())
        quality = outcome_record(metrics, self.panel['task_ids'])
        self.validate_metrics(metrics, Path(self.protocol['checkpoint']['path']))
        if not quality['complete']: raise RuntimeError('Baseline panel incomplete')
        self.state['baseline'] = dict(quality, role='initial_anchor', checkpoint_sha256=anchor['checkpoint_sha256'],
                                      metrics_path=anchor['metrics'], metrics_sha256=anchor['metrics_sha256'],
                                      condition_signature=anchor['condition_signature'], captured_at=Path(anchor['metrics']).stat().st_mtime,
                                      timestamp_kind='saved_metrics_mtime', roles=['initial_anchor'])

    def validate_metrics(self, metrics, checkpoint):
        cfg, env = metrics['config'], metrics['training_environment']
        expected = {'seed': 777, 'max_act': 3, 'horizon': 4096, 'simulations': 200,
                    'device': 'cpu', 'inference_device': 'cpu', 'workers': 4, 'threads': 1,
                    'split': 'dev', 'task_kind': 'run', 'preserve_task_prefixes': True}
        if any(cfg.get(k) != v for k, v in expected.items()) or Path(cfg['checkpoint']).resolve() != checkpoint.resolve():
            raise RuntimeError('Panel command/checkpoint configuration differs')
        if env.get('native_module_sha256') != self.pins['backend']['provenance']['module_sha256'] or env.get('python_interface_sha256') != self.interface_sha:
            raise RuntimeError('Panel backend/interface differs')

    def read_learner(self):
        learner = json.loads(self.learner_path.read_text())
        if digest(learner['identity']) != self.pins['learner_identity']:
            raise RuntimeError('Learner identity changed')
        self.learner = learner
        return learner

    def allowed(self):
        learner = self.read_learner()
        reason = learner_permission(learner)
        if reason: return reason
        cfg = learner['identity']['config']
        return resource_pause(SimpleNamespace(min_free_gib=cfg['min_free_gib'], battery_pause_at=cfg['battery_pause_at']), self.output,
                              previously_battery_paused=learner.get('battery_paused', False))

    def save(self):
        self.state['heartbeat'] = time.time()
        atomic_json(self.journal_path, self.state)
        telemetry = {k: self.state[k] for k in ('version', 'monitoring_only', 'status', 'heartbeat', 'histories', 'latest', 'baseline')}
        telemetry.update(panel='dev_act3', task_ids=self.panel['task_ids'], task_ids_sha256=digest(sorted(self.panel['task_ids'])),
                         requested=48, next_due=self.state['next_due'], protocol_sha256=self.pins['protocol_sha256'])
        atomic_json(self.output / 'telemetry.json', telemetry)
        atomic_json(self.output / 'progress.json', {k: self.state.get(k) for k in ('status', 'heartbeat', 'next_due', 'active_job', 'pause_reason')})

    def recover(self):
        for job in self.state['jobs']:
            if job['status'] == 'running':
                stop_owned_group(job)
                job['status'] = 'interrupted'; job['finished'] = time.time()
        self.state.pop('active_job', None)

    def evaluate(self, capture: dict[str, Any]):
        self.validate_inputs()
        snapshot = Path(capture['snapshot'])
        if sha_file(snapshot) != capture['checkpoint_sha256']: raise RuntimeError('Snapshot changed')
        provisional = self.output / 'jobs' / 'placeholder.metrics.json'
        command = build_command(snapshot, provisional, self.protocol, seconds=self.seconds)
        signature = condition_signature(command)
        key = observer_condition_key(signature, self.pins)
        for old in self.state['histories']:
            if old['condition_key'] == key:
                if sha_file(old['metrics_path']) != old['metrics_sha256']: raise RuntimeError('Completed monitor metrics changed')
                return old, True
        attempt = sum(job.get('condition_key') == key for job in self.state['jobs'])
        failures = sum(j.get('condition_key') == key and (j['status'] == 'failed' or j.get('abort_reason') in ('child_timeout', 'child_stall')) for j in self.state['jobs'])
        if failures >= 2: raise RuntimeError('Observer condition failed twice; needs attention')
        metrics_path = self.output / 'jobs' / f'{key}_a{attempt}.metrics.json'
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(snapshot, metrics_path, self.protocol, seconds=self.seconds)
        if metrics_path.exists(): raise RuntimeError('Refusing unclassified existing monitor output')
        record = {'condition_key': key, 'condition_signature': signature, 'command': command,
                  'checkpoint_sha256': capture['checkpoint_sha256'], 'started': time.time(), 'status': 'running',
                  'metrics_path': str(metrics_path)}
        env = os.environ.copy(); env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                                           STS_LIGHTSPEED_BUILD=self.pins['backend']['build_dir'])
        record['effective_environment'] = {k: env[k] for k in ('PYTHONPATH', 'STS_LIGHTSPEED_BUILD', 'PYTHONDONTWRITEBYTECODE')}
        self.state['jobs'].append(record)
        log = metrics_path.with_suffix('.log')
        deadline = time.monotonic() + self.seconds + 60
        progress_at = time.monotonic(); progress_marker = None
        try:
            with log.open('xb') as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(pid=process.pid, pgid=process.pid, process_birth=process_birth(process.pid), owned_members={})
                record['group_creation'] = {'new_session': True, 'leader_pid': process.pid, 'leader_birth': record['process_birth'],
                    'started_at': record['started'], 'command_sha256': digest(command), 'max_lifetime_seconds': self.seconds + 65}
                self.state['active_job'] = len(self.state['jobs']) - 1; self.state['status'] = 'evaluating'; self.save()
                while True:
                    track_owned_members(record)
                    code = process.poll()
                    if code is not None: break
                    reason = self.allowed()
                    marker = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (log, metrics_path) if p.exists())
                    if marker != progress_marker: progress_marker = marker; progress_at = time.monotonic()
                    stalled = time.monotonic() - progress_at >= 150
                    if self.stop or reason or stalled or time.monotonic() >= deadline:
                        record['abort_reason'] = reason or ('observer_signal' if self.stop else 'child_stall' if stalled else 'child_timeout')
                        stop_owned_group(record); process.wait(timeout=10); break
                    time.sleep(2)
                record['group_creation']['leader_exit_observed_at'] = time.time()
                record['exit_code'] = process.poll()
        finally:
            if record.get('pgid'): stop_owned_group(record)
            record['finished'] = time.time()
            self.state.pop('active_job', None)
            record['status'] = 'aborted' if record.get('abort_reason') else 'failed'
            self.save()
        if record['exit_code'] != 0 or record.get('abort_reason'):
            return None, False
        metrics = json.loads(metrics_path.read_text())
        self.validate_metrics(metrics, snapshot)
        quality = outcome_record(metrics, self.panel['task_ids'])
        result = dict(quality, condition_key=key, condition_signature=signature,
                      checkpoint_sha256=capture['checkpoint_sha256'], checkpoint_path=str(snapshot),
                      captured_at=capture['captured_at'], captured_global_round=capture['captured_global_round'],
                      evaluated_at=record['finished'], roles=[], metrics_path=str(metrics_path), metrics_sha256=sha_file(metrics_path),
                      training_environment=metrics['training_environment'])
        record['status'] = 'complete'; record['metrics_sha256'] = result['metrics_sha256']
        self.state['histories'].append(result); self.save()
        return result, False

    def cycle(self, max_panels=None):
        reason = self.allowed()
        if reason: self.state.update(status='waiting_learner', pause_reason=reason); self.save(); return 0
        captures = completed_candidates(self.learner)
        count = 0
        capture_retry = False
        for capture in captures:
            try:
                snapshot = snapshot_checkpoint(Path(capture['source_checkpoint']), self.output / 'checkpoints', capture['checkpoint_sha256'])
            except FileNotFoundError:
                # Retention won the race before open: retry a newer completed journal next cycle.
                self.state['capture_retry_at'] = time.time(); capture_retry = True; continue
            capture['snapshot'] = str(snapshot)
            capture['captured_at'] = time.time()
            self.state['captures'].append(capture)
            result, reused = self.evaluate(capture)
            if result is None: continue
            role = capture['role']
            if role not in result['roles']: result['roles'].append(role)
            self.state['latest'][role] = dict(result, role=role, role_capture_at=capture['captured_at'],
                                             reused_evaluation=reused, role_captured_global_round=capture['captured_global_round'])
            count += not reused
            self.save()
            if max_panels is not None and count >= max_panels: return count
        self.state.update(next_due=time.time() + (30 if capture_retry else self.interval),
                          status='capture_retry' if capture_retry else 'waiting_interval')
        self.state.pop('pause_reason', None); self.save()
        return count

    def run(self, *, once=False, max_panels=None):
        with FileLock(self.output / 'observer.lock'):
            # Reload only under ownership; constructor does not write journal.
            if self.journal_path.exists(): self.state = restore_state(json.loads(self.journal_path.read_text()), self.pins)
            try:
                self.recover(); self.save()
                while not self.stop:
                    if once or time.time() >= self.state['next_due']:
                        self.cycle(max_panels=max_panels)
                        if once: break
                    else:
                        reason = self.allowed()
                        self.state.update(status='waiting_learner' if reason else 'waiting_interval', pause_reason=reason)
                        self.save()
                    time.sleep(30 if not once else 0)
                self.state['status'] = 'observer_stopped' if self.stop else 'one_shot_complete' if once else self.state['status']; self.save()
            except Exception as exc:
                self.state.update(status='attention_required', error=str(exc))
                self.save()
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner-journal', type=Path, default=ROOT / 'data/long_rl_v1/campaign/journal.json')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/long_curriculum_protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/long_rl_v1/full_run_eval')
    parser.add_argument('--interval-seconds', type=float, default=1800.)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--max-panels', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.interval_seconds < 60 or args.seconds <= 0 or (args.max_panels is not None and args.max_panels < 1):
        parser.error('Expected interval>=60, seconds>0, max-panels>=1')
    if not args.execute:
        print(json.dumps({'monitoring_only': True, 'device': 'cpu', 'workers': 4, 'threads': 1, 'interval_seconds': args.interval_seconds,
                          'output': str(args.output), 'execute': False})); return 0
    observer = Observer(args.learner_journal, args.protocol, args.output, interval=args.interval_seconds, seconds=args.seconds)
    def stop(signum, frame): observer.stop = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    observer.run(once=args.once, max_panels=args.max_panels)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
