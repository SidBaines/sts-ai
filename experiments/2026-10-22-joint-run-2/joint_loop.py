#!/usr/bin/env python3
"""Teacher labelling and gated combat refits beside an independent OOC PPO trainer.

Compared with the first joint loop, this version checks terminal artifacts as well
as queue state, bounds waits and retries once into fresh directories, pauses for
disk recovery, emits heartbeats, and stops as the trainer finishes. Comparisons
use the statistics library in process on completed paired worlds, allowing at most
1% missing; these changes prevent silent startup interruptions from hanging a gate.
The original config keys, fitting recipe, and wins-and-mean-floor gate are retained.
Optional fit/label/gate_timeout_seconds default to 3/4/2 hours, respectively.
trainer_startup_timeout_seconds and trainer_stall_seconds default to 30 minutes
and 3 hours; max_consecutive_skipped_cycles defaults to 3.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from sts_rl.fresh_statistics import exact_discordance


ACTIVE_QUEUE_STATES = {'pending', 'running', 'paused', 'starting'}
ENDED_QUEUE_STATES = {'done', 'failed', 'exited', 'cancelled'}


def queue_finished(status):
    return status is not None and status not in ACTIVE_QUEUE_STATES


def classify_fit_outcome(status, best_exists, queue_status=None, timed_out=False, summary=None):
    """Return (pending/success/failed, reason), including zero-exit interruptions."""
    state = status.get('status')
    details = '; '.join(str(status[k]) for k in ('reason', 'error') if status.get(k))
    completed = state == 'completed' or (summary or {}).get('status') == 'completed'
    if completed and best_exists:
        return 'success', 'completed'
    if completed or state in {'interrupted', 'failed', 'error', 'cancelled', 'incomplete'}:
        return 'failed', details or ('completed_without_best' if completed else state)
    if timed_out:
        return 'failed', f'timeout; {details}' if details else 'timeout'
    if queue_finished(queue_status):
        reason = f'queue_{queue_status}_without_completed_fit'
        return 'failed', f'{reason}; {details}' if details else reason
    return 'pending', None


def label_accepted(completions, requested, min_fraction=0.75):
    """Count completed worlds only, never faulted/censored/unstarted worlds."""
    return requested > 0 and sum(r.get('completed', 0) for r in completions) >= requested * min_fraction


def gate_decision(results):
    """Preserve selection by wins then floor, and accept equality on both metrics."""
    if not results:
        return None, False
    best = max(results, key=lambda s: (results[s]['wins'][1], results[s]['floor']['difference']))
    result = results[best]
    return best, result['wins'][1] >= result['wins'][0] and result['floor']['difference'] >= 0


def should_start_cycle(cycle, max_cycles, iterations, latest_iteration,
                       min_iterations_left=3, stopped=False, trainer_done=False):
    return (not stopped and not trainer_done and cycle < max_cycles and iterations is not None
            and iterations - latest_iteration >= min_iterations_left)


def read_json(path):
    try:
        value = Path(path).read_text()
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def game_records(summary, expected=None):
    """Validate all seeds, then keep completed outcomes rather than counting faults as losses."""
    if (summary.get('error') or summary.get('status') in {'failed', 'error', 'interrupted', 'cancelled'}
            or summary.get('programming_errors') or summary.get('unstarted')):
        raise ValueError('Evaluation error, programming errors or unstarted worlds')
    records = summary.get('records', [])
    indexed, seen = {}, set()
    expected = set(expected) if expected is not None else None
    for row in records:
        seed = row['world_seed']
        if (not isinstance(seed, int) or isinstance(seed, bool) or seed in seen
                or (expected is not None and seed not in expected)):
            raise ValueError('Evaluation contains duplicate, invalid or unexpected world seeds')
        seen.add(seed)
        if row.get('status') in {'programming_error', 'unstarted'} or row.get('programming_error'):
            raise ValueError('Evaluation contains programming errors or unstarted worlds')
        if (row.get('status') != 'completed'
                or any(row.get(k) for k in ('fault', 'cut', 'ub', 'censored', 'truncated'))):
            continue
        if (not isinstance(row.get('win'), bool)
                or isinstance(row.get('floor'), bool)
                or not isinstance(row.get('floor'), (int, float)) or not math.isfinite(row['floor'])):
            raise ValueError('Evaluation contains invalid completed outcomes')
        indexed[seed] = row
    return indexed


def compare_summaries(base, arm, expected=None):
    expected = set(expected) if expected is not None else {
        r['world_seed'] for s in (base, arm) for r in s.get('records', [])}
    base, arm = game_records(base, expected), game_records(arm, expected)
    common = sorted(base.keys() & arm.keys())
    n, dropped = len(common), len(expected) - len(common)
    if not n or dropped * 100 > len(expected):
        raise ValueError(f'Gate intersection missing {dropped}/{len(expected)} worlds (more than 1%)')
    base_wins, arm_wins = sum(base[s]['win'] for s in common), sum(arm[s]['win'] for s in common)
    only_arm = sum(arm[s]['win'] and not base[s]['win'] for s in common)
    only_base = sum(base[s]['win'] and not arm[s]['win'] for s in common)
    return {'worlds': n, 'dropped_worlds': dropped, 'wins': [base_wins, arm_wins],
            'win_rate': [base_wins / n, arm_wins / n],
            'only_baseline': only_base, 'only_arm': only_arm,
            'mcnemar_p': exact_discordance(only_arm, only_base),
            'floor': {'baseline': sum(base[s]['floor'] for s in common) / n,
                      'arm': sum(arm[s]['floor'] for s in common) / n,
                      'difference': sum(arm[s]['floor'] - base[s]['floor'] for s in common) / n}}


def missing_ranges(start, n, completed):
    """Contiguous ranges suitable for the collector's --world-start/--n interface."""
    ranges = []
    for seed in range(start, start + n):
        if seed in completed:
            continue
        if ranges and sum(ranges[-1]) == seed:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + 1)
        else:
            ranges.append((seed, 1))
    return ranges


class StopLoop(Exception):
    pass


class Loop:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.out = Path(cfg['out'])
        self.code = Path(cfg['code'])
        self.py, self.build = cfg['python'], cfg['build']
        self.runner = str(Path(cfg['queue_root']) / 'scripts/rl/queue_runner.py')
        self.dry_run = dry_run
        self.reserved_names = set()
        self.active_tasks = {}
        self.owned_jobs = set()  # Includes submissions whose runner start or cancellation failed.
        self.queue_states = {}
        self.gate_completed = False
        self.disk_waiting = False
        self.last_heartbeat = -math.inf
        if not dry_run:
            self.out.mkdir(parents=True, exist_ok=True)
        if not 0 < cfg.get('min_label_fraction', .75) <= 1:
            raise ValueError('min_label_fraction must be in (0, 1]')
        for key, default in [('min_free_gib', 15), ('fit_timeout_seconds', 10800),
                             ('label_timeout_seconds', 14400), ('gate_timeout_seconds', 7200),
                             ('trainer_startup_timeout_seconds', 1800), ('trainer_stall_seconds', 10800)]:
            if not math.isfinite(cfg.get(key, default)) or cfg.get(key, default) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        cap = cfg.get('max_consecutive_skipped_cycles', 3)
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError('max_consecutive_skipped_cycles must be a positive integer')

    def log(self, **event):
        event['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with (self.out / 'decisions.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)

    def trainer_progress(self):
        trainer = Path(self.cfg['trainer_out'])
        iterations = read_json(trainer / 'config.json').get('iterations')
        latest, done = 0, False
        try:
            with (trainer / 'log.jsonl').open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:  # the trainer may be appending the final line
                        continue
                    if row.get('event') == 'done':
                        done = True
                    if row.get('event') == 'train':
                        latest = max(latest, row['iteration'])
        except FileNotFoundError:
            pass
        return iterations, latest, done

    def stop_reason(self):
        if (self.out / 'STOP').exists():
            return 'STOP'
        if self.trainer_progress()[2]:
            return 'trainer_done'
        try:
            idle = time.time() - (Path(self.cfg['trainer_out']) / 'log.jsonl').stat().st_mtime
        except FileNotFoundError:
            return None
        if idle >= self.cfg.get('trainer_stall_seconds', 10800):
            self.log(event='trainer_stalled', seconds_since_log_update=idle)
            return 'trainer_stalled'
        return None

    def wait_for_trainer(self, cycle):
        started = time.monotonic()
        timeout = self.cfg.get('trainer_startup_timeout_seconds', 1800)
        while True:
            reason = self.stop_reason()
            if reason:
                raise StopLoop(reason)
            if (self.trainer_progress()[0] is not None
                    and (Path(self.cfg['trainer_out']) / 'log.jsonl').is_file()):
                return
            self.heartbeat(cycle, 'trainer_startup', self.active_tasks, started, self.free_gib())
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise StopLoop('trainer_startup_timeout')
            time.sleep(min(15, remaining))

    def free_gib(self):
        return shutil.disk_usage(self.out).free / (1 << 30)

    def heartbeat(self, cycle, step, jobs, started, free):
        now = time.monotonic()
        if now - self.last_heartbeat >= 60:
            atomic_json(self.out / 'heartbeat.json', {
                'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'cycle': cycle, 'step': step,
                'jobs': list(jobs), 'seconds_in_step': max(0., now - started), 'free_disk_gib': free})
            self.last_heartbeat = now

    def disk_ready(self, cycle, step, jobs, started):
        free, minimum = self.free_gib(), self.cfg.get('min_free_gib', 15)
        if free < minimum and not self.disk_waiting:
            self.disk_waiting = True
            self.log(event='waiting_for_disk', cycle=cycle, step=step, free_disk_gib=free)
        if self.disk_waiting and free >= minimum + 2:
            self.disk_waiting = False
            self.log(event='resumed', cycle=cycle, step=step, free_disk_gib=free)
        self.heartbeat(cycle, step, jobs, started, free)
        return not self.disk_waiting

    def before_step(self, cycle, step):
        started = time.monotonic()
        while True:
            self.poll_tasks()
            reason = self.stop_reason()
            if reason:
                raise StopLoop(reason)
            if self.disk_ready(cycle, step, self.active_tasks, started):
                return
            time.sleep(15)

    def queue_runs(self):
        try:
            runs = json.loads((Path(self.cfg['queue_root']) / 'data/rl_runs/queue.json').read_text())['runs']
            if not isinstance(runs, list) or any(not isinstance(r, dict) or 'name' not in r
                                                  or 'status' not in r for r in runs):
                return None
            return runs
        except (OSError, ValueError, KeyError, TypeError):
            return None  # An unavailable snapshot says nothing about a task's state.

    def queue(self, name, lane, cmd):
        # queue_runner also rejects collisions under its own lock; recheck on that race.
        for _ in range(10):
            used = self.reserved_names | {r['name'] for r in self.queue_runs() or []}
            unique, index = name, 0
            while unique in used:
                index += 1
                unique = f'{name}-rerun{index}'
            command = [self.py, self.runner, 'add', unique, '--lane', lane, '--',
                       'env', f'STS_LIGHTSPEED_BUILD={self.build}', *cmd]
            if self.dry_run:
                print(shlex.join(command))
                print(shlex.join([self.py, self.runner, 'start']))
            else:
                try:
                    subprocess.run(command, check=True, capture_output=True, timeout=30)
                except subprocess.CalledProcessError as exc:
                    if 'already queued' in str(exc.stderr):
                        self.reserved_names.add(unique)
                        continue
                    raise
                except subprocess.TimeoutExpired:
                    if any(r['name'] == unique and r.get('cmd') == command[command.index('--') + 1:]
                           for r in self.queue_runs() or []):
                        self.owned_jobs.add(unique)
                        self.try_cancel(unique)
                    raise
                self.owned_jobs.add(unique)
                try:
                    subprocess.run([self.py, self.runner, 'start'], check=True, capture_output=True, timeout=30)
                except Exception:
                    self.try_cancel(unique)
                    raise
            self.reserved_names.add(unique)
            return unique
        raise RuntimeError(f'Unable to reserve a unique queue name: {name}')

    def cancel(self, name):
        subprocess.run([self.py, self.runner, 'cancel', name], check=True, capture_output=True, timeout=30)

    def try_cancel(self, name):
        try:
            self.cancel(name)
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            self.log(event='cancel_failed', job=name, reason=reason)
            return reason
        self.owned_jobs.discard(name)
        return None

    def cancel_active(self):
        names = sorted(self.owned_jobs | self.active_tasks.keys())
        cancelled, failed = [], []
        for name in names:
            if self.try_cancel(name) is None:
                cancelled.append(name)
                self.active_tasks.pop(name, None)
            else:
                failed.append(name)
        return cancelled, failed

    def fresh_output(self, path):
        candidate, index = path, 0
        while candidate.exists():
            index += 1
            candidate = path.with_name(f'{path.name}_rerun{index}')
        return candidate

    def launch(self, cycle, step, name, lane, out, cmd):
        self.before_step(cycle, step)
        started = time.monotonic()
        try:
            name = self.queue(name, lane, cmd)
        except StopLoop:
            raise
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            self.log(event='queue_failed', cycle=cycle, step=step, job=name, reason=reason)
            return {'name': name, 'out': out, 'started': started, 'result': ('failed', reason)}
        kind = 'label' if step.startswith('label') else step
        task = {'name': name, 'out': out, 'started': started, 'kind': kind,
                'timeout': self.cfg.get(f'{kind}_timeout_seconds', {'fit': 10800, 'label': 14400, 'gate': 7200}[kind])}
        self.active_tasks[name] = task
        return task

    def classify_task(self, task, state, timed_out):
        if task['kind'] == 'fit':
            return classify_fit_outcome(read_json(task['out'] / 'status.json'),
                                        (task['out'] / 'best.pt').is_file(), state, timed_out,
                                        read_json(task['out'] / 'summary.json'))
        path = task['out'] / ('completion.json' if task['kind'] == 'label' else 'summary.json')
        if path.exists():
            if task['kind'] == 'label':
                return 'success', 'receipt'
            summary = read_json(path)
            if summary.get('error') or summary.get('status') in {'failed', 'error', 'interrupted', 'cancelled'}:
                return 'failed', summary.get('error') or summary['status']
            try:
                game_records(summary, range(self.cfg['gate_world_start'],
                                            self.cfg['gate_world_start'] + self.cfg['gate_worlds']))
                return 'success', 'completed'
            except (ValueError, KeyError, TypeError) as exc:
                task['retryable'] = False
                return 'failed', str(exc)
        if timed_out or queue_finished(state):
            return 'failed', 'timeout' if timed_out else f'queue_{state}_without_artifacts'
        return 'pending', None

    def poll_tasks(self):
        if not self.active_tasks and not self.owned_jobs:
            return
        try:
            runs = self.queue_runs()
        except Exception as exc:
            self.log(event='queue_poll_failed', reason=f'{type(exc).__name__}: {exc}')
            runs = None
        if runs is not None:
            self.queue_states = {r['name']: r['status'] for r in runs}
        for name in list(self.owned_jobs):
            if self.queue_states.get(name) in ENDED_QUEUE_STATES:
                self.owned_jobs.discard(name)
        for name, task in list(self.active_tasks.items()):
            state = self.queue_states.get(name, 'missing' if runs is not None else None)
            if 'result' in task:
                if state in ENDED_QUEUE_STATES:
                    del self.active_tasks[name]
                continue
            timed_out = time.monotonic() - task['started'] >= task['timeout']
            outcome, reason = self.classify_task(task, state, timed_out)
            if outcome == 'pending':
                continue
            cancelled = False
            if outcome == 'failed' and (timed_out or state not in ENDED_QUEUE_STATES):
                error = self.try_cancel(name)
                cancelled = error is None
                if error:
                    outcome, reason = 'failed', f'{reason}; cancel_failed: {error}'
            task['result'] = (outcome, reason)
            if cancelled or state in ENDED_QUEUE_STATES:
                del self.active_tasks[name]

    def wait(self, tasks, cycle, step):
        started = min(t['started'] for t in tasks)
        while True:
            reason = self.stop_reason()
            if reason:
                raise StopLoop(reason)
            self.disk_ready(cycle, step, self.active_tasks, started)
            self.poll_tasks()
            pending = [t for t in tasks if 'result' not in t]
            if not pending:
                return {t['name']: t['result'] for t in tasks}
            remaining = min(t['timeout'] - (time.monotonic() - t['started']) for t in pending)
            time.sleep(max(0., min(15, remaining)))

    def player(self):
        saved = sorted(Path(self.cfg['trainer_out']).glob('iter_*.pt'))
        return saved[-1] if saved else Path(self.cfg['start'])

    def shard_specs(self, cycle):
        n = self.cfg['worlds_per_cycle']
        start = self.cfg['label_world_start'] + cycle * self.cfg['label_world_stride']
        # Retain eight lanes, also covering the remainder if n is not divisible by eight.
        return [(i, start + i * (n // 8) + min(i, n % 8), n // 8 + (i < n % 8))
                for i in range(8) if n // 8 + (i < n % 8)]

    def label_command(self, player, out, start, n):
        return [self.py, str(self.code / 'scripts/rl/fresh_collect.py'), '--mode', 'collect',
                '--checkpoint', str(player), '--out', str(out), '--world-start', str(start), '--n', str(n),
                '--workers', '1', '--timeout', '30', '--teacher-probabilities',
                json.dumps(self.cfg['teacher_probabilities']), '--samples', '24', '--simulations', '1000',
                '--self-consistency-every', '10']

    def label_part(self, out, start, n):
        receipt = read_json(out / 'completion.json')
        rows = read_json(out / 'summary.json').get('records', [])
        completed = {r['world_seed'] for r in rows if r.get('status') == 'completed'}
        expected = set(range(start, start + n))
        # The collector atomically publishes only completed worlds in worlds/.
        # Snapshot explicit files after a timeout so late writes from the cancelled
        # process cannot introduce duplicates when missing worlds are retried.
        files = {int(p.name[:-8]): p for p in (out / 'worlds').glob('*.json.gz')
                 if p.name[:-8].isdigit() and int(p.name[:-8]) in expected}
        if not receipt:
            return {'out': out, 'start': start, 'requested': n, 'completed': len(files),
                    'worlds': set(files), 'sources': list(files.values())}
        count = receipt.get('completed', 0)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= n:
            count = 0
        if len(completed) != count or not completed <= expected:
            completed = set(files) if len(files) == count else None
        return {'out': out, 'start': start, 'requested': n, 'completed': count,
                'worlds': completed, 'sources': [out] if count else []}

    def label(self, cycle, player):
        self.before_step(cycle, 'label')
        iterations, latest, done = self.trainer_progress()
        if not should_start_cycle(cycle, self.cfg['max_cycles'], iterations, latest,
                                  self.cfg.get('min_iterations_left', 3), trainer_done=done):
            raise StopLoop('too_few_iterations_left')
        corpus = self.out / f'corpus_{cycle:02d}'
        corpus.mkdir()
        copy = self.out / 'players' / f'player_{cycle:02d}.pt'
        copy.parent.mkdir(exist_ok=True)
        shutil.copy2(player, copy)
        specs = self.shard_specs(cycle)
        tasks = []
        for i, start, n in specs:
            out = corpus / f'shard_{i:02d}'
            tasks.append(self.launch(cycle, 'label', f"{self.cfg['name']}-c{cycle:02d}-label{i}",
                                     f"{self.cfg['lane_prefix']}label{i}", out,
                                     self.label_command(copy, out, start, n)))
        by_shard = {spec[0]: task for spec, task in zip(specs, tasks)}

        outcomes = self.wait(tasks, cycle, 'label')
        requested = self.cfg['worlds_per_cycle']
        fraction = self.cfg.get('min_label_fraction', .75)
        parts = [self.label_part(corpus / f'shard_{i:02d}', start, n) for i, start, n in specs]
        if not label_accepted(parts, requested, fraction):
            retries = []
            retry_specs = []
            for (i, start, n), part in zip(specs, list(parts)):
                if part['completed'] == n:
                    continue
                if part['worlds'] is not None:
                    ranges = missing_ranges(start, n, part['worlds'])
                else:
                    ranges = [(start, n)]
                    parts.remove(part)  # whole-shard fallback must not duplicate worlds
                for j, (retry_start, retry_n) in enumerate(ranges):
                    out = corpus / f'shard_{i:02d}_retry1_{j:02d}'
                    task = self.launch(cycle, 'label_retry', f"{self.cfg['name']}-c{cycle:02d}-label{i}-{j}-retry1",
                                       f"{self.cfg['lane_prefix']}label{i}", out,
                                       self.label_command(copy, out, retry_start, retry_n))
                    retries.append(task)
                    retry_specs.append((out, retry_start, retry_n))
                    self.log(event='label_retry', cycle=cycle, job=task['name'], shard=i,
                             world_start=retry_start, worlds=retry_n, out=str(out),
                             reason=outcomes[by_shard[i]['name']][1], completed_before_retry=part['completed'])
            if retries:
                outcomes.update(self.wait(retries, cycle, 'label_retry'))
                parts += [self.label_part(out, start, n) for out, start, n in retry_specs]
        accepted = label_accepted(parts, requested, fraction)
        sources = [s for part in parts for s in part['sources']]
        atomic_json(corpus / 'corpus.json', {'accepted': accepted, 'shards': [str(s) for s in sources]})
        self.log(event='labelled' if accepted else 'cycle_skipped', cycle=cycle, step='label',
                 corpus=str(corpus), player=str(player), player_sha256=sha256(copy),
                 completed=sum(part['completed'] for part in parts),
                 worlds=requested, outcomes=outcomes, reason=None if accepted else 'insufficient_labels')
        return corpus if accepted else None

    def fit_command(self, data, out, seed):
        deadline = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() + 20 * 3600))
        return [self.py, str(self.code / 'scripts/rl/fresh_train.py'), '--mixed-sources', '--data', *data,
                '--probe', self.cfg['probe'], '--out', str(out), '--architecture', 'old', '--phase', 'combat',
                '--lr', '0.001', '--old-size', 'large', '--old-context', 'none', '--steps', '12000',
                '--min-steps', '2000', '--batch', '128', '--defer-test', '--eval-every', '100', '--patience', '6',
                '--seed', str(seed), '--device', 'mps', '--deadline-utc', deadline]

    def fit(self, cycle, corpora, seed):
        data = [s for c in corpora for s in read_json(c / 'corpus.json')['shards']]
        data += self.cfg['base_corpus_shards']
        for attempt in range(2):
            suffix = '_retry1' if attempt else ''
            out = self.fresh_output(self.out / f'fit_c{cycle:02d}_seed{seed}{suffix}')
            name = f"{self.cfg['name']}-c{cycle:02d}-fit{seed}" + ('-retry1' if attempt else '')
            task = self.launch(cycle, 'fit', name, f"{self.cfg['lane_prefix']}fit", out,
                               self.fit_command(data, out, seed))
            if attempt:
                self.log(event='fit_retry', cycle=cycle, seed=seed, job=task['name'], out=str(out))
            outcome, reason = self.wait([task], cycle, 'fit')[task['name']]
            self.log(event='fit_ok' if outcome == 'success' else 'fit_failed', cycle=cycle, seed=seed,
                     attempt=attempt, job=task['name'], out=str(out), reason=reason)
            if outcome == 'success':
                return out / 'best.pt'
        return None

    def eval_command(self, checkpoint, out):
        return [self.py, str(self.code / 'scripts/rl/fast_eval.py'), '--checkpoint', str(checkpoint),
                '--world-start', str(self.cfg['gate_world_start']), '--n', str(self.cfg['gate_worlds']),
                '--processes', '8', '--games-per-process', '32', '--out', str(out)]

    def combine_command(self, incumbent, student, candidate):
        return [self.py, str(self.code / 'scripts/rl/combine_heads.py'), '--ooc', str(incumbent),
                '--combat', str(student), '--output', str(candidate)]

    def compare(self, base, arm):
        return compare_summaries(read_json(base), read_json(arm),
                                 range(self.cfg['gate_world_start'],
                                       self.cfg['gate_world_start'] + self.cfg['gate_worlds']))

    def gate(self, cycle, students):
        self.gate_completed = False
        self.before_step(cycle, 'gate')
        player = self.player()
        pc = self.out / 'gate' / f'c{cycle:02d}'
        pc.mkdir(parents=True, exist_ok=True)
        incumbent = pc / 'incumbent.pt'
        shutil.copy2(player, incumbent)
        checkpoints = {'incumbent': incumbent}
        for seed, student in students.items():
            self.before_step(cycle, 'combine')
            candidate = pc / f'candidate_seed{seed}.pt'
            try:
                subprocess.run(self.combine_command(incumbent, student, candidate), check=True,
                               env={**os.environ, 'STS_LIGHTSPEED_BUILD': self.build,
                                    'PYTHONPATH': str(self.code / 'src')}, capture_output=True, timeout=600)
            except (subprocess.SubprocessError, OSError) as exc:
                self.log(event='combine_failed', cycle=cycle, seed=seed, reason=str(exc))
                continue
            checkpoints[seed] = candidate
        if len(checkpoints) == 1:
            self.log(event='cycle_skipped', cycle=cycle, step='gate', reason='no_combined_candidates')
            return False
        summaries, pending = {}, dict(checkpoints)
        validation_failed = False
        for attempt in range(2):
            tasks = {}
            for key, checkpoint in pending.items():
                label = 'incumbent' if key == 'incumbent' else f'seed{key}'
                suffix = '_retry1' if attempt else ''
                out = self.fresh_output(pc / f'{label}{suffix}')
                name = f"{self.cfg['name']}-c{cycle:02d}-gate-{label}" + ('-retry1' if attempt else '')
                tasks[key] = self.launch(cycle, 'gate', name, f"{self.cfg['lane_prefix']}eval", out,
                                         self.eval_command(checkpoint, out))
            outcomes = self.wait(list(tasks.values()), cycle, 'gate')
            pending = {}
            for key, task in tasks.items():
                outcome, reason = outcomes[task['name']]
                if outcome == 'success':
                    summaries[key] = task['out'] / 'summary.json'
                else:
                    retry = not attempt and task.get('retryable', True)
                    if not task.get('retryable', True):
                        validation_failed = True
                    pending[key] = checkpoints[key]
                    self.log(event='gate_eval_failed', cycle=cycle, evaluation=key, attempt=attempt,
                             job=task['name'], accepted=False, reason=reason, retry=retry)
            if not pending or validation_failed:
                break
        if pending:
            self.log(event='cycle_skipped', cycle=cycle, step='gate',
                     reason='gate_validation_failed' if validation_failed else 'evaluation_failed_twice')
            return False
        self.before_step(cycle, 'gate_decision')
        results = {}
        for seed in students:
            if seed not in summaries:
                continue
            try:
                results[seed] = self.compare(summaries['incumbent'], summaries[seed])
            except (ValueError, KeyError, TypeError) as exc:
                self.log(event='gate_eval_failed', cycle=cycle, evaluation=seed,
                         accepted=False, reason=str(exc), retry=False)
                self.log(event='cycle_skipped', cycle=cycle, step='gate', reason='gate_validation_failed')
                return False
        self.gate_completed = True
        best, accept = gate_decision(results)
        self.log(event='gate', cycle=cycle, incumbent=str(player), incumbent_sha256=sha256(incumbent),
                 results={str(s): {'wins': v['wins'], 'floor_difference': v['floor']['difference'],
                                   'mcnemar_p': v['mcnemar_p'], 'worlds': v['worlds'],
                                   'dropped_worlds': v['dropped_worlds']} for s, v in results.items()},
                 best_seed=best, accepted=accept)
        if accept:
            self.before_step(cycle, 'adopt')
            follow = Path(self.cfg['follow'])
            temp = follow.with_suffix('.tmp')
            shutil.copy2(students[best], temp)
            temp.replace(follow)
            self.log(event='adopted', cycle=cycle, student=str(students[best]), sha256=sha256(follow))
        else:
            self.log(event='rejected', cycle=cycle, best_seed=best)
        return accept

    def plan_cycle(self, cycle):
        """Print launch commands only; no output directories, copies, jobs or decisions."""
        corpus = self.out / f'corpus_{cycle:02d}'
        player = self.out / 'players' / f'player_{cycle:02d}.pt'
        print(shlex.join(['cp', str(self.player()), str(player)]))
        data = []
        for i, start, n in self.shard_specs(cycle):
            out = corpus / f'shard_{i:02d}'
            data.append(str(out))
            self.queue(f"{self.cfg['name']}-c{cycle:02d}-label{i}", f"{self.cfg['lane_prefix']}label{i}",
                       self.label_command(player, out, start, n))
        data = [s for c in sorted(self.out.glob('corpus_*'))
                if read_json(c / 'corpus.json').get('accepted')
                for s in read_json(c / 'corpus.json').get('shards', [])] + data + self.cfg['base_corpus_shards']
        pc = self.out / 'gate' / f'c{cycle:02d}'
        incumbent = pc / 'incumbent.pt'
        print(shlex.join(['cp', str(self.player()), str(incumbent)]))
        for seed in self.cfg['seeds']:
            student = self.out / f'fit_c{cycle:02d}_seed{seed}'
            self.queue(f"{self.cfg['name']}-c{cycle:02d}-fit{seed}", f"{self.cfg['lane_prefix']}fit",
                       self.fit_command(data, student, seed))
            candidate = pc / f'candidate_seed{seed}.pt'
            print(shlex.join(self.combine_command(incumbent, student / 'best.pt', candidate)))
            self.queue(f"{self.cfg['name']}-c{cycle:02d}-gate-seed{seed}", f"{self.cfg['lane_prefix']}eval",
                       self.eval_command(candidate, pc / f'seed{seed}'))
        self.queue(f"{self.cfg['name']}-c{cycle:02d}-gate-incumbent", f"{self.cfg['lane_prefix']}eval",
                   self.eval_command(incumbent, pc / 'incumbent'))

    def run(self):
        previous = sorted(p for p in self.out.glob('corpus_*') if p.name[7:].isdigit())
        cycle = max((int(p.name[7:]) for p in previous), default=-1) + 1
        if self.dry_run:
            self.plan_cycle(cycle)
            return
        corpora = [c for c in previous if read_json(c / 'corpus.json').get('accepted')]
        reason = None
        skipped_cycles = 0
        try:
            self.wait_for_trainer(cycle)
            while True:
                iterations, latest, done = self.trainer_progress()
                reason = self.stop_reason()
                if not should_start_cycle(cycle, self.cfg['max_cycles'], iterations, latest,
                                          self.cfg.get('min_iterations_left', 3), bool(reason), done):
                    reason = reason or ('max_cycles' if cycle >= self.cfg['max_cycles'] else
                                        'trainer_config_unavailable' if iterations is None else 'too_few_iterations_left')
                    break
                corpus = self.label(cycle, self.player())
                skipped = True
                if corpus is not None:
                    corpora.append(corpus)
                    students = {}
                    for seed in self.cfg['seeds']:
                        student = self.fit(cycle, corpora, seed)
                        if student is not None:
                            students[seed] = student
                    if students:
                        self.gate(cycle, students)
                        skipped = not self.gate_completed
                    else:
                        self.log(event='cycle_skipped', cycle=cycle, step='gate', reason='no_successful_fits')
                cycle += 1
                skipped_cycles = skipped_cycles + 1 if skipped else 0
                if skipped_cycles >= self.cfg.get('max_consecutive_skipped_cycles', 3):
                    self.log(event='too_many_skipped_cycles', cycles=cycle, consecutive=skipped_cycles)
                    raise StopLoop('too_many_skipped_cycles')
        except StopLoop as exc:
            reason = str(exc)
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            cancelled, cancel_failed = self.cancel_active()
            self.log(event='stopped', cycles=cycle, reason=reason,
                     cancelled_jobs=cancelled, cancel_failed_jobs=cancel_failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    Loop(json.loads(args.config.read_text()), dry_run=args.dry_run).run()


if __name__ == '__main__':
    main()
