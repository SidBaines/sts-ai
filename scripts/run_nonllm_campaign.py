#!/usr/bin/env python3
"""Finite local curriculum campaign; one GPU learner, resumable atomic journal."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
INITIAL = datetime.fromisoformat('2026-10-11T20:08:46+00:00').timestamp()
DEADLINE = datetime.fromisoformat('2026-10-12T06:30:00+00:00').timestamp()
BANK = ROOT / 'data/nonllm_curriculum/teacher_act1_200_seed10000'
HARD_BANK = ROOT / 'data/nonllm_curriculum/teacher_act3_200_seed11000'
LOOKAHEAD_BANK = ROOT / 'data/nonllm_curriculum/ooc_lookahead_v1'


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Campaign:
    def __init__(self, destination):
        self.destination = destination
        destination.mkdir(parents=True, exist_ok=True)
        self.journal = destination / 'campaign.json'
        self.state = json.loads(self.journal.read_text()) if self.journal.exists() else {
            'started_at': INITIAL, 'deadline': DEADLINE, 'commands': [], 'selected': {},
            'status': 'running', 'phase': 'screens', 'native_build': os.environ['STS_LIGHTSPEED_BUILD']}

    def save(self):
        self.state['heartbeat'] = time.time()
        atomic(self.journal, self.state)

    def run(self, name, command, seconds, phase_end=DEADLINE):
        metrics = self.destination / (name + '.metrics.json')
        previous = next((x for x in self.state['commands'] if x['name'] == name and x['status'] == 'complete'), None)
        if previous and metrics.exists():
            return json.loads(metrics.read_text())
        allowance = min(seconds + 90, phase_end - time.time(), DEADLINE - time.time())
        if allowance <= 2:
            return None
        source_identity = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'src/sts_policy').glob('*.py'))}
        record = {'source_identity': source_identity, 'name': name, 'command': [str(x) for x in command], 'started': time.time(),
                  'status': 'running', 'metrics': str(metrics)}
        self.state['commands'].append(record)
        self.save()
        log = self.destination / (name + '.log')
        with log.open('a') as output:
            process = subprocess.Popen([str(x) for x in command], cwd=ROOT, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = process.pid
            self.save()
            started = time.monotonic()
            while process.poll() is None:
                self.save()
                if time.monotonic() - started >= allowance:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    record['status'] = 'budget_stop'
                    break
                time.sleep(5)
            record['exit_code'] = process.returncode
        record['finished'] = time.time()
        if record['status'] == 'running':
            record['status'] = 'complete' if process.returncode == 0 else 'failed'
        self.save()
        print(json.dumps(record), flush=True)
        if record['status'] == 'complete' and metrics.exists() and metrics.stat().st_mtime >= record['started']:
            return json.loads(metrics.read_text())
        return None

    def train(self, name, method, checkpoint=None, *, seconds=120, end=DEADLINE, **kwargs):
        path = self.destination / (name + '.pt')
        command = [PYTHON, ROOT / 'scripts/train_nonllm.py', method, '--output', path,
                   '--metrics', path.with_suffix('.metrics.json'), '--seconds', seconds,
                   '--device', 'mps', '--threads', 2, '--batch-size', 1024, '--seed', 11, '--simulations', 200]
        if checkpoint:
            command += ['--checkpoint', checkpoint]
        if method == 'bc':
            command += ['--teacher', BANK / 'teacher.jsonl', '--updates', 100000]
        else:
            command += ['--workers', 8, '--updates', 100000, '--episodes', 128,
                        '--fast-observations', '--inference-device', 'cpu']
        for key, value in kwargs.items():
            command += ['--' + key.replace('_', '-'), str(value)]
        if '--fast-replay' in (ROOT/'scripts/train_nonllm.py').read_text():
            command += ['--fast-replay']
        if method == 'ppo':
            command += ['--warm-optimizer']
        result = self.run(name, command, seconds, end)
        return (path, result) if path.exists() and result is not None else (None, result)

    def evaluate(self, name, checkpoint, kind='battle', split='dev', episodes=48, seconds=90, end=DEADLINE, **kwargs):
        command = [PYTHON, ROOT / 'scripts/train_nonllm.py', 'evaluate', '--output',
                   self.destination / (name + '.unused.pt'), '--metrics', self.destination / (name + '.metrics.json'),
                   '--checkpoint', checkpoint, '--device', 'cpu', '--threads', 2,
                   '--workers', 8, '--episodes', episodes, '--seed', 777, '--seconds', seconds,
                   '--task-kind', kind, '--split', split, '--horizon', 256 if kind == 'battle' else 2048,
                   '--simulations', 200, '--fast-observations']
        if kind == 'battle':
            command += ['--tasks', BANK / 'battles.jsonl']
        else:
            command += ['--world-start', 10000]
        for key, value in kwargs.items():
            command += ['--' + key.replace('_', '-'), str(value)]
        if '--fast-replay' in (ROOT/'scripts/train_nonllm.py').read_text():
            command += ['--fast-replay']
        return self.run(name, command, seconds, end)

    @staticmethod
    def score(metrics):
        if not metrics:
            return -1e9
        episodes = [episode for row in metrics.get('rounds', []) for episode in row.get('episodes', [])]
        decisive = [x for x in episodes if x.get('terminated')]
        wins = sum(x.get('outcome') in ('battle_victory', 'act_clear', 'victory') for x in decisive)
        if not decisive:
            return -1000
        floor = sum((x.get('floor') or 0) for x in episodes) / max(1, len(episodes))
        hp = [x['hp_cost'] for x in episodes if x.get('hp_cost') is not None]
        # Unfinished ordinary episodes lower observed success; faults remain separate.
        faults = [fault for row in metrics.get('rounds', []) for fault in row.get('faults', [])]
        if len(faults) > max(2, .1 * (len(episodes) + len(faults))):
            return -1000
        return 100 * wins / max(1, len(episodes)) + .1 * floor - .01 * sum(hp) / max(1, len(hp))

    def screen(self):
        self.state['phase'] = 'architecture_screens'
        end = INITIAL + 7200
        candidates = []
        for size in ('small', 'medium', 'large'):
            for context in ('none', 'mean'):
                name = f'bc_{size}_{context}'
                path, result = self.train(name, 'bc', seconds=90, end=end, size=size, context=context)
                if path:
                    evaluation = self.evaluate(name + '_dev', path, end=end)
                    candidates.append((self.score(evaluation), path, result))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            raise RuntimeError('No successful architecture checkpoint; inspect command logs')
        top = candidates[:2]
        for _, source, _ in top:
            for rate in (1e-4, 3e-4, 1e-3):
                name = source.stem + f'_lr{rate}'
                path, result = self.train(name, 'bc', seconds=90, end=end,
                                         size=source.stem.split('_')[1], context=source.stem.split('_')[2],
                                         learning_rate=rate)
                if path:
                    evaluation = self.evaluate(name + '_dev', path, end=end)
                    candidates.append((self.score(evaluation), path, result))
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        for seed in (23, 37):
            name = best.stem + f'_rep{seed}'
            source_metrics = json.loads(best.with_suffix('.metrics.json').read_text())['config']
            path, result = self.train(name, 'bc', seconds=90, end=end, seed=seed,
                                     size=source_metrics['size'], context=source_metrics['context'],
                                     learning_rate=source_metrics['learning_rate'])
            if path:
                evaluation = self.evaluate(name + '_dev', path, end=end)
                candidates.append((self.score(evaluation), path, result))
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        self.state['architecture_panel'] = [{'score': score, 'checkpoint': str(path)} for score, path, _ in candidates]
        self.state['selected']['shared_bc'] = str(best)
        self.save()
        return best

    def curriculum(self, name, source, end, kind, **settings):
        self.state['phase'] = name
        current, best = Path(source), Path(source)
        reference = self.evaluate(name + '_initial_dev', best, kind, end=end)
        best_score = self.score(reference)
        saved_phase = self.state.setdefault('phase_progress', {}).setdefault(name,
                       {'next_round': 0, 'started': time.time(), 'current': str(source), 'best': str(best), 'best_score': best_score})
        round_index = saved_phase['next_round']
        phase_started = saved_phase['started']
        current, best = Path(saved_phase['current']), Path(saved_phase['best'])
        best_score = saved_phase['best_score']
        while time.time() < end - 180:
            tag = f'{name}_{round_index:04d}'
            round_settings = dict(settings)
            if kind == 'battle' and HARD_BANK.exists():
                elapsed_fraction = (time.time() - phase_started) / max(1, end - phase_started)
                if elapsed_fraction < .1:
                    round_settings['tasks'] = HARD_BANK/'stage_subsets/battles_act1_early.jsonl'
                elif elapsed_fraction < .35:
                    round_settings['tasks'] = BANK/'battles.jsonl'
                else:
                    round_settings['tasks'] = HARD_BANK/'battles.jsonl' if round_index % 3 else BANK/'battles.jsonl'
                round_settings['max_act'] = 3 if round_settings['tasks'].is_relative_to(HARD_BANK) else 1
            path, _ = self.train(tag, 'ppo', current, seconds=min(180, end-time.time()-120), end=end,
                                 task_kind=kind, seed=11 + round_index, **round_settings)
            if path is None:
                raise RuntimeError('PPO failed; inspect ' + tag)
            current = path
            if kind == 'battle' and round_index % 3 == 0:
                labels = path.with_suffix('.dagger.jsonl')
                if labels.exists() and labels.stat().st_size:
                    corrected, _ = self.train(tag + '_dagger', 'bc', current, seconds=20, end=end,
                                              teacher=labels, phase='combat', learning_rate=1e-4)
                    if corrected:
                        current = corrected
            evaluation = self.evaluate(tag + '_dev', current, kind, end=end)
            score = self.score(evaluation)
            if score > best_score:
                best, best_score = current, score
            # Restore a good policy after clear regressions; archive all trajectories.
            if score < best_score - 15:
                current = best
            self.state['selected'][name] = str(best)
            self.state[name + '_last'] = {'round': round_index, 'score': score, 'best_score': best_score}
            saved_phase.update(next_round=round_index+1, current=str(current), best=str(best), best_score=best_score)
            self.save()
            round_index += 1
        self.state['selected'][name] = str(best)
        self.save()
        return best

    def final(self, shared, combat, ooc):
        self.state['phase'] = 'frozen_final_evaluation'
        self.state['frozen_selection'] = {'shared': str(shared), 'combat': str(combat), 'ooc': str(ooc), 'at': time.time()}
        self.save()
        for kind, checkpoint, episodes in [('battle', combat, 160), ('ooc', ooc, 21), ('run', shared, 21)]:
            self.evaluate('final_' + kind + '_act1', checkpoint, kind, 'final', episodes, seconds=300)
        self.evaluate('final_run_act3', shared, 'run', 'final', 21, seconds=600, max_act=3)
        # The composed specialist routes phases using public observations.
        command = [PYTHON, ROOT / 'scripts/train_nonllm.py', 'evaluate', '--output', self.destination/'final_composite.unused.pt',
                   '--metrics', self.destination/'final_composite.metrics.json', '--combat-checkpoint', combat,
                   '--ooc-checkpoint', ooc, '--device', 'cpu', '--workers', 8, '--threads', 2,
                   '--episodes', 21, '--world-start', 10000, '--split', 'final', '--task-kind', 'run',
                   '--max-act', 3, '--horizon', 2048, '--simulations', 200, '--seconds', 600, '--fast-observations']
        self.run('final_composite', command, 600)
        for kind in ('battle', 'ooc', 'run'):
            for baseline in ('random', 'native', 'heuristic'):
                script = ROOT/'scripts/evaluate_nonllm_baselines.py'
                if not script.exists():
                    continue
                name = f'final_baseline_{baseline}_{kind}'
                command = [PYTHON, script, '--output', self.destination/(name+'.metrics.json'), '--baseline', baseline,
                           '--kind', kind, '--split', 'final', '--episodes', 160 if kind=='battle' else 21,
                           '--workers', 8, '--seconds', 200, '--simulations', 200, '--seed', 777,
                           '--world-start', 10000, '--horizon', 256 if kind=='battle' else 2048,
                           '--max-act', 1, '--fast-observations']
                if kind=='battle':
                    command += ['--tasks', BANK/'battles.jsonl']
                self.run(name, command, 200)
        for baseline in ('random', 'native', 'heuristic'):
            name = f'final_baseline_{baseline}_run_act3'
            command = [PYTHON, ROOT/'scripts/evaluate_nonllm_baselines.py', '--output', self.destination/(name+'.metrics.json'),
                       '--baseline', baseline, '--kind', 'run', '--split', 'final', '--episodes', 21,
                       '--world-start', 10000, '--seed', 777, '--horizon', 2048, '--max-act', 3,
                       '--workers', 8, '--seconds', 600, '--simulations', 200, '--fast-observations']
            self.run(name, command, 600)
        self.state['status'] = 'finished'
        self.save()

    def ppo_screen(self, name, source, kind, end, **settings):
        self.state['phase'] = name + '_hyperparameter_screen'
        panel = []
        for rate, reward in ((1e-4, 'potential'), (3e-4, 'potential'), (1e-3, 'potential'), (3e-4, 'sparse')):
            tag = f'{name}_pilot_lr{rate}_{reward}'
            path, result = self.train(tag, 'ppo', source, seconds=90, end=end,
                                     task_kind=kind, learning_rate=rate, reward_mode=reward, **settings)
            if path:
                evaluation = self.evaluate(tag + '_dev', path, kind, episodes=64, end=end)
                panel.append((self.score(evaluation), path, rate, reward))
        if not panel:
            return source, 3e-4, 'potential'
        panel.sort(reverse=True)
        self.state[name + '_pilot_panel'] = [{'score': score, 'checkpoint': str(path),
                                             'learning_rate': rate, 'reward': reward}
                                            for score, path, rate, reward in panel]
        self.save()
        return panel[0][1], panel[0][2], panel[0][3]

    def main(self):
        shared = self.screen()
        if HARD_BANK.exists():
            hardened, _ = self.train('combat_act3_bc', 'bc', shared, seconds=180,
                                     teacher=HARD_BANK/'teacher.jsonl', dev_teacher=BANK/'teacher.jsonl', phase='combat')
            if hardened:
                before = self.evaluate('combat_hardening_before_dev', shared)
                after = self.evaluate('combat_hardening_after_dev', hardened)
                if self.score(after) >= self.score(before) - 5:
                    shared = hardened
        combat_start, combat_rate, combat_reward = self.ppo_screen('combat', shared, 'battle', INITIAL+14400,
                                                   tasks=BANK/'battles.jsonl', horizon=256)

        combat = self.curriculum('combat', combat_start, INITIAL+14400, 'battle',
                                 tasks=BANK/'battles.jsonl', horizon=256, teacher_simulations=200,
                                 reward_mode=combat_reward, learning_rate=combat_rate)
        specialist, _ = self.train('ooc_specialist_bc', 'bc', shared, seconds=120, phase='ooc')
        ooc_start = specialist or shared
        if LOOKAHEAD_BANK.exists():
            lookahead, _ = self.train('ooc_lookahead_bc', 'bc', shared, seconds=90,
                                      teacher=LOOKAHEAD_BANK/'teacher.jsonl', dev_teacher=BANK/'teacher.jsonl', phase='ooc')
            if lookahead:
                a = self.evaluate('ooc_scripted_dev', ooc_start, 'ooc', episodes=32)
                b = self.evaluate('ooc_lookahead_dev', lookahead, 'ooc', episodes=32)
                self.state['ooc_teacher_panel'] = {'scripted_score': self.score(a), 'lookahead_score': self.score(b)}
                if self.score(b) > self.score(a):
                    ooc_start = lookahead
                self.save()
        ooc = self.curriculum('ooc', ooc_start, INITIAL+21600, 'ooc',
                             world_start=20000, horizon=1024, reward_mode='progress', learning_rate=3e-4)
        initial_panel = []
        for label, checkpoint in [('shared_bc', shared), ('combat', combat), ('ooc', ooc)]:
            evaluation = self.evaluate('joint_start_' + label + '_dev', checkpoint, 'run', episodes=32,
                                       end=INITIAL+28800)
            initial_panel.append((self.score(evaluation), checkpoint))
        initial_panel.sort(reverse=True)
        self.state['joint_start_panel'] = [{'score': score, 'checkpoint': str(path)} for score, path in initial_panel]
        self.save()
        full = self.curriculum('joint_act1', initial_panel[0][1], INITIAL+28800, 'run',
                              world_start=30000, horizon=2048, reward_mode='progress', learning_rate=1e-4)
        full = self.curriculum('joint_act3', full, INITIAL+32400, 'run',
                              world_start=40000, horizon=4096, max_act=3, reward_mode='progress', learning_rate=1e-4)
        self.final(full, combat, ooc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/nonllm_v1/campaign')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    campaign_lock = (args.output/'campaign.lock').open('w')
    fcntl.flock(campaign_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign_lock.write(str(os.getpid()))
    campaign_lock.flush()
    campaign = Campaign(args.output)
    try:
        campaign.main()
    except BaseException as error:
        campaign.state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        campaign.state['error'] = repr(error)
        campaign.save()
        raise
