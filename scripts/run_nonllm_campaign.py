#!/usr/bin/env python3
"""Finite local curriculum campaign; one GPU learner, resumable atomic journal."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from sts_policy.campaign_identity import condition_signature, legacy_cache_valid, semantic_arguments
from sts_policy.final_freeze import capture_final_freeze, verify_final_freeze, FreezeChangedError

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
INITIAL = datetime.fromisoformat('2026-10-11T20:08:46+00:00').timestamp()
DEADLINE = datetime.fromisoformat('2026-10-12T06:30:00+00:00').timestamp()
BANK = ROOT / 'data/nonllm_curriculum/teacher_act1_200_seed10000'
HARD_BANK = ROOT / 'data/nonllm_curriculum/teacher_act3_200_seed11000'
LOOKAHEAD_BANK = ROOT / 'data/nonllm_curriculum/ooc_lookahead_v1'
PANELS = ROOT / 'data/nonllm_curriculum/panels'
EXPANDED_PANELS = ROOT / 'data/nonllm_curriculum/panels_expanded'


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cohort_arguments(command):
    """Changing data, policies or task cohorts invalidates a cached condition."""
    keys={'--tasks','--teacher','--dev-teacher','--checkpoint','--combat-checkpoint',
          '--ooc-checkpoint','--world-start','--horizon','--max-act','--task-kind',
          '--split','--episodes','--baseline','--seed'}
    values={}
    for index,token in enumerate(command[:-1]):
        if str(token) in keys:values[str(token)]=str(command[index+1])
    return values


def panel_complete(metrics):
    """A successful process is insufficient: every requested task must be classified."""
    rounds = metrics.get('rounds',[]) if metrics.get('command') == 'evaluate' else [metrics]
    if not rounds:
        return False
    for row in rounds:
        episodes, faults = row.get('episodes',[]), row.get('faults',[])
        requested = row.get('task_ids',[])
        actual = [entry.get('task_id') for entry in episodes+faults]
        if not requested or Counter(requested) != Counter(actual):
            return False
        if row.get('unstarted_tasks',0) or any(entry.get('termination_reason')=='collector_budget' for entry in episodes):
            return False
        if row.get('cohort_completed') is False:
            return False
    return True


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

    def completed_frontier_seconds(self, name):
        """Recover a completed chunk's immutable budget before rebuilding argv."""
        metrics = self.destination / (name + '.metrics.json')
        checkpoint = self.destination / (name + '.pt')
        if not metrics.is_file() or not checkpoint.is_file():
            return None
        digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
        for record in reversed(self.state['commands']):
            if record.get('name') != name or record.get('status') != 'complete':
                continue
            if digest != record.get('metrics_sha256') or checkpoint.stat().st_mtime > record.get('finished', 0) + 5:
                continue
            command = list(map(str, record.get('command', [])))
            indices = [index for index, token in enumerate(command[:-1]) if token == '--seconds']
            if indices:
                token = command[indices[-1] + 1]
                seconds = float(token)
                if seconds > 0:
                    record['recovered_frontier_budget_seconds'] = seconds
                    return int(token) if token.isdigit() else seconds
        return None

    def run(self, name, command, seconds, phase_end=DEADLINE):
        if name.startswith('final_') and self.state.get('frozen_identities'):
            verify_final_freeze(self.state['frozen_identities'], backend_build_dir=self.state['native_build'])
        metrics = self.destination / (name + '.metrics.json')
        try:
            signature = condition_signature(command)
        except (OSError, IndexError) as error:
            self.state['commands'].append({'name': name, 'command': list(map(str, command)),
                'started': time.time(), 'finished': time.time(), 'status': 'failed',
                'metrics': str(metrics), 'exit_code': None, 'input_identity_error': repr(error)})
            self.save()
            return None
        previous_records = [x for x in self.state['commands'] if x['name'] == name and x['status'] == 'complete']
        if previous_records and metrics.exists():
            digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
            for previous in reversed(previous_records):
                expected = previous.get('metrics_sha256')
                metrics_valid = digest == expected or (expected is None and metrics.stat().st_mtime <= previous.get('finished', 0) + 5)
                pinned = previous.get('condition_signature')
                if pinned is not None:
                    compatible = pinned['digest'] == signature['digest']
                    if pinned.get('version') == 'campaign_condition_v1':
                        # v1 pinned argv order. Migrate only identical effective
                        # knobs AND the full previously recorded input hashes.
                        compatible = (semantic_arguments(pinned.get('arguments', [])) == signature['arguments']
                                      and pinned.get('inputs') == signature['inputs'])
                else:
                    # These options have separate CPU/native parity evidence.
                    # Only legacy records normalize them; new signatures pin
                    # every argument and effective input's SHA256.
                    performance = {'--fast-replay', '--fast-observations', '--deferred-bootstrap'}
                    normalized = [token for token in command if str(token) not in performance]
                    old = {**previous, 'command': [token for token in previous['command'] if str(token) not in performance]}
                    compatible = legacy_cache_valid(old, normalized)
                if metrics_valid and compatible:
                    previous['metrics_sha256'] = digest
                    previous['condition_signature'] = signature
                    self.save()
                    return json.loads(metrics.read_text())
        allowance = min(seconds + 90, phase_end - time.time(), DEADLINE - time.time())
        if allowance <= 2:
            return None
        # A new scientific condition may reuse a human job name. Preserve any
        # currently verifiable completed artifacts before a child can replace
        # them; cached matching conditions returned above never reach this path.
        if previous_records and metrics.is_file():
            digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
            previous = next((record for record in reversed(previous_records)
                             if record.get('metrics_sha256') == digest), None)
            if previous is not None:
                checkpoint = self.destination/(name+'.pt')
                references = [*self.state.get('selected', {}).values(), *self.state.get('frozen_selection', {}).values()]
                references += [phase.get(key) for phase in self.state.get('phase_progress', {}).values()
                               for key in ('current', 'best')]
                protected = checkpoint.is_file() and any(isinstance(value, (str, Path)) and
                    Path(value).resolve() == checkpoint.resolve() for value in references)
                if protected:
                    self.state['commands'].append({'name':name,'command':list(map(str,command)),
                        'started':time.time(),'finished':time.time(),'status':'blocked_protected_artifact',
                        'metrics':str(metrics),'exit_code':None,
                        'error':'Completed checkpoint is selected/frozen/current; use a unique condition output name.'})
                    self.save()
                    raise RuntimeError('Cannot overwrite a selected/frozen/current completed checkpoint; rename the condition artifact first.')
                archive = self.destination/'history'/(name + '-' + str(previous.get('started',0)).replace('.','_'))
                archive.mkdir(parents=True, exist_ok=True)
                artifacts = [metrics, self.destination/(name+'.pt'), self.destination/(name+'.log')]
                preserved = {}
                for artifact in artifacts:
                    if not artifact.is_file():
                        continue
                    target = archive/artifact.name
                    if not target.exists():
                        temporary = target.with_suffix(target.suffix+'.tmp')
                        shutil.copy2(artifact, temporary)
                        temporary.replace(target)
                    preserved[str(artifact)] = {'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
                previous['preserved_artifacts'] = preserved
        source_identity = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'src/sts_policy').glob('*.py'))}
        source_identity.update({str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT/'scripts/train_nonllm.py', Path(__file__)) if p.exists()})
        record = {'source_identity': source_identity, 'condition_signature': signature,
                  'name': name, 'command': [str(x) for x in command], 'started': time.time(),
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
                if time.monotonic() - started >= allowance or time.time() >= min(phase_end, DEADLINE):
                    os.killpg(process.pid, signal.SIGTERM)
                    record['stop_reason'] = 'utc_deadline' if time.time() >= DEADLINE else 'stage_or_runtime_budget'
                    try:
                        process.wait(timeout=max(.001, min(5, DEADLINE-time.time())))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    record['status'] = 'budget_stop'
                    break
                time.sleep(max(.001, min(5, min(phase_end, DEADLINE)-time.time())))
            record['exit_code'] = process.returncode
        record['finished'] = time.time()
        if record['status'] == 'running':
            record['status'] = 'complete' if process.returncode == 0 else 'failed'
        if record['status'] == 'complete' and metrics.exists():
            record['metrics_sha256'] = hashlib.sha256(metrics.read_bytes()).hexdigest()
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
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:command.append(flag)
                elif key=='goal_act_clear':command.append('--no-goal-act-clear')
            elif value is not None:
                command += [flag,str(value)]
        if '--fast-replay' in (ROOT/'scripts/train_nonllm.py').read_text():
            command += ['--fast-replay']
        if method == 'ppo' and checkpoint is not None:
            command += ['--warm-optimizer']
        result = self.run(name, command, seconds, end)
        return (path, result) if path.exists() and result is not None else (None, result)

    def evaluate(self, name, checkpoint, kind='battle', split='dev', episodes=48, seconds=90, end=DEADLINE, **kwargs):
        if kind == 'ooc' and split == 'dev':
            episodes = 32
            kwargs['horizon'] = 1024
            kwargs['world_start'] = 10000
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
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:command.append(flag)
                elif key=='goal_act_clear':command.append('--no-goal-act-clear')
            elif value is not None:
                command += [flag,str(value)]
        if '--fast-replay' in (ROOT/'scripts/train_nonllm.py').read_text():
            command += ['--fast-replay']
        return self.run(name, command, seconds, end)

    def evaluate_combat(self, name, checkpoint, *, split='dev', episodes=48, end=DEADLINE):
        primary = self.evaluate(name+'_act1', checkpoint, 'battle', split, episodes, end=end)
        hard_path = Path(self.state.get('active_panel_directory',str(PANELS))) / (split+'_hard.jsonl')
        hard = self.evaluate(name+'_hard', checkpoint, 'battle', split, 256, end=end,
                             tasks=hard_path, max_act=3) if hard_path.exists() else None
        if primary is None:
            return None
        if not panel_complete(primary) or (hard_path.exists() and (hard is None or not panel_complete(hard))):
            score = -1e9
        else:
            score = self.score(primary) if hard is None else .5*(self.score(primary)+self.score(hard))
        combined = {'selection_score':score, 'primary_score':self.score(primary),
                    'hard_score':self.score(hard) if hard else None,
                    'rounds':primary.get('rounds',[])+(hard.get('rounds',[]) if hard else []),
                    'note':'Equal weight Act1 and later-act panels; intervals must cluster by world. Thin Act3 flagged in panel manifest.'}
        atomic(self.destination/(name+'_combined.metrics.json'),combined)
        return combined

    def evaluate_composite(self, name, combat, ooc, *, split='dev', episodes=32, max_act=1, end=DEADLINE):
        command = [PYTHON, ROOT/'scripts/train_nonllm.py', 'evaluate', '--output', self.destination/(name+'.unused.pt'),
                   '--metrics', self.destination/(name+'.metrics.json'), '--combat-checkpoint',combat,
                   '--ooc-checkpoint',ooc, '--device','cpu','--threads',2,'--workers',8,'--episodes',episodes,
                   '--seed',777,'--world-start',10000,'--split',split,'--task-kind','run','--max-act',max_act,
                   '--horizon',2048,'--simulations',200,'--seconds',120,'--fast-observations','--fast-replay']
        return self.run(name,command,120,end)

    @staticmethod
    def score(metrics):
        if not metrics:
            return -1e9
        if metrics.get('command') == 'evaluate' and not panel_complete(metrics):
            return -1e9
        if 'selection_score' in metrics:
            return metrics['selection_score']
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
        requested = sum(len(row.get('task_ids', [])) for row in metrics.get('rounds', []))
        # A native fault remains a separately reported fault. Including its
        # requested task in selection coverage prevents disappearing hard states
        # from improving a policy's development score.
        return 100 * wins / max(1, requested or len(episodes)) + .1 * floor - .01 * sum(hp) / max(1, len(hp))

    def screen(self):
        saved = self.state.get('selected', {}).get('shared_bc')
        if self.state.get('architecture_panel') and saved is not None and Path(saved).is_file():
            return Path(saved)
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
        saved_phase=self.state.setdefault('phase_progress',{}).get(name)
        def evaluate_stage(tag, checkpoint):
            if kind == 'battle':
                return self.evaluate_combat(tag, checkpoint, end=end)
            if kind == 'run':
                max_act = settings.get('max_act', 1)
                return self.evaluate(tag, checkpoint, kind, end=end,
                                     max_act=max_act, horizon=settings.get('horizon', 2048),
                                     seconds=240 if max_act == 3 else 90)
            return self.evaluate(tag, checkpoint, kind, end=end)
        if saved_phase is None:
            reference = evaluate_stage(name + '_initial_dev', best)
            saved_phase={'next_round':0,'started':time.time(),'current':str(source),'best':str(best),'best_score':self.score(reference)}
            self.state['phase_progress'][name]=saved_phase
        round_index = saved_phase['next_round']
        phase_started = saved_phase['started']
        current, best = Path(saved_phase['current']), Path(saved_phase['best'])
        best_score = saved_phase['best_score']
        while time.time() < end - 180:
            tag = f'{name}_{round_index:04d}'
            round_settings = dict(settings)
            round_settings['deferred_bootstrap'] = True
            if kind == 'battle' and HARD_BANK.exists():
                elapsed_fraction = (time.time() - phase_started) / max(1, end - phase_started)
                if elapsed_fraction < .1:
                    round_settings['tasks'] = HARD_BANK/'stage_subsets/battles_act1_early.jsonl'
                elif elapsed_fraction < .35:
                    round_settings['tasks'] = BANK/'battles.jsonl'
                else:
                    round_settings['tasks'] = Path(self.state.get('active_panel_directory',str(PANELS)))/f'train_act{1 + round_index % 3}.jsonl'
                round_settings['max_act'] = 1 if round_settings['tasks'].is_relative_to(BANK) else 3
            if kind != 'battle':
                round_settings['world_start'] = settings.get('world_start',20000) + round_index*128
            seconds = self.completed_frontier_seconds(tag)
            if seconds is None:
                seconds = min(180, end-time.time()-120)
            path, _ = self.train(tag, 'ppo', current, seconds=seconds, end=end,
                                 task_kind=kind, seed=11 + round_index, **round_settings)
            if path is None:
                raise RuntimeError('PPO failed; inspect ' + tag)
            current = path
            if kind == 'battle' and round_index % 3 == 0:
                labels = path.with_suffix('.dagger.jsonl')
                if labels.exists() and labels.stat().st_size:
                    from sts_policy.aggregation import aggregate_teachers
                    aggregate = self.destination/(tag+'_aggregate.jsonl')
                    aggregate_teachers([Path(self.state.get('native_rehearsal',str(BANK/'teacher.jsonl'))),
                                        *sorted(self.destination.glob('combat_*.dagger.jsonl'))],aggregate,
                                       max_records=30000,seed=11,phase='combat')
                    corrected, _ = self.train(tag + '_dagger', 'bc', current, seconds=30, end=end,
                                              teacher=aggregate, phase='combat', learning_rate=1e-4)
                    if corrected:
                        current = corrected
            evaluation = evaluate_stage(tag + '_dev', current)
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

    def joint_composition(self, combat, ooc):
        """Never reuse a composed policy after its source or implementation changes."""
        from sts_policy.composition import compose_checkpoint
        sources = {'combat': Path(combat).resolve(), 'ooc': Path(ooc).resolve()}
        source_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key,path in sources.items()}
        names = {'composition_code_sha256': 'composition.py', 'encoder_code_sha256': 'encoding.py',
                 'model_code_sha256': 'model.py', 'wrapper_code_sha256': 'model_agent.py'}
        code_hashes = {key: hashlib.sha256((ROOT/'src/sts_policy'/name).read_bytes()).hexdigest()
                       for key,name in names.items()}
        identity = hashlib.sha256(json.dumps({'sources': source_hashes,'code':code_hashes},sort_keys=True).encode()).hexdigest()
        checkpoint = self.destination/('joint_initial_composite_' + identity[:16] + '.pt')
        manifest = checkpoint.with_suffix('.composition.json')
        if checkpoint.is_file() and manifest.is_file():
            report = json.loads(manifest.read_text())
            provenance = report['composition_provenance']
            valid = report['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            valid = valid and str(checkpoint.resolve()) == str(Path(report['output']).resolve())
            for role,path in sources.items():
                row = provenance['sources'][role]
                valid = valid and row['sha256'] == source_hashes[role] and str(Path(row['path']).resolve()) == str(path)
            valid = valid and all(provenance[key] == value for key,value in code_hashes.items())
            if not valid:
                raise RuntimeError('Composed checkpoint identity changed; preserve and investigate ' + str(checkpoint))
            return checkpoint, report
        if checkpoint.exists() or manifest.exists():
            raise RuntimeError('Partial composed artifact requires recovery: ' + str(checkpoint))
        report = compose_checkpoint(combat, ooc, checkpoint)
        atomic(manifest, report)
        return checkpoint, report

    def joint_screen(self, single, composite, references, end):
        """Equal-budget full-control pilots for shared versus phase specialists."""
        basic = {'gamma': 1., 'clip_ratio': .1, 'entropy_coef': .01, 'learning_rate': 1e-4,
                 'reward_mode': 'progress', 'progress_floor_weight': .05, 'terminal_hp_weight': 0.,
                 'kl_guard_before_update': self.state.get('kl_guard_before_update', False)}
        saved = self.state.get('phase_progress', {}).get('joint_act1')
        if saved is not None:
            return Path(saved['best']), self.state.get('joint_effective_settings', basic)
        self.state['phase'] = 'joint_shared_composite_screen'
        panel = [{'name': 'reference_' + label, 'checkpoint': str(path), 'score': score, 'role': 'reference'}
                 for label, score, path in references]
        for label, source in [('single', single), ('composite', composite)]:
            checkpoint, result = self.train('joint_pilot_' + label, 'ppo', source, seconds=90, end=end,
                          task_kind='run', world_start=30000, episodes=64, horizon=2048,
                          max_act=1, seed=11, deferred_bootstrap=True, **basic)
            if checkpoint is not None:
                evaluation = self.evaluate('joint_pilot_' + label + '_dev', checkpoint, 'run',
                                           episodes=32, end=end, horizon=2048, max_act=1)
                if evaluation is not None and panel_complete(evaluation):
                    panel.append({'name': label, 'checkpoint': str(checkpoint), 'score': self.score(evaluation),
                                  'role': 'pilot', 'settings': basic})
            self.state['joint_pilot_panel'] = panel
            self.save()
        eligible = [row for row in panel if row['score'] > -1e8]
        if not eligible:
            raise RuntimeError('No completed full-control joint development cohort')
        # Stable ordering retains a simpler untrained reference on an exact tie.
        best = max(eligible, key=lambda row: row['score'])
        self.state['joint_effective_settings'] = basic
        self.state['selected']['joint_pilot'] = best['checkpoint']
        self.save()
        return Path(best['checkpoint']), basic

    def freeze_final_selection(self, shared, combat, ooc):
        """Keep chosen policies, task definitions and backend immutable for finals."""
        checkpoints = {'shared': shared, 'combat': combat, 'ooc': ooc}
        existing = self.state.get('frozen_identities')
        if existing is not None:
            verify_final_freeze(existing, backend_build_dir=self.state['native_build'])
            for label, path in checkpoints.items():
                if str(Path(path).resolve()) != existing['checkpoints'][label]['path']:
                    raise FreezeChangedError('Final checkpoint selection changed: ' + label)
            return existing
        from sts_policy.curriculum import TaskSpec, world_split, TASK_VERSION
        panels = Path(self.state.get('active_panel_directory', str(PANELS)))
        files = {'act1_battles': BANK/'battles.jsonl', 'act1_manifest': BANK/'manifest.json',
                 'later_act_battles': panels/'final_hard.jsonl', 'panel_manifest': panels/'panels.json'}
        files.update({str(path.relative_to(ROOT)): path for path in sorted((ROOT/'src/sts_policy').glob('*.py'))})
        for path in (ROOT/'scripts/train_nonllm.py', ROOT/'scripts/evaluate_nonllm_baselines.py', Path(__file__)):
            files[str(path.relative_to(ROOT))] = path
        worlds, seed = [], 10000
        while len(worlds) < 21:
            if world_split(seed) == 'final':
                worlds.append(seed)
            seed += 1
        definitions = {'task_version': TASK_VERSION, 'split': 'final', 'world_start': 10000,
                       'worlds': worlds, 'episodes': 21, 'shuffle_seed': 777,
                       'horizon': 2048, 'simulations': 200, 'goal_act_clear': True,
                       'native_boss_multiplier': 2.0, 'panels': {}}
        for label, kind, max_act in [('ooc_act1','ooc',1),('run_act1','run',1),('run_act3','run',3)]:
            tasks = [TaskSpec(seed=world, kind=kind, horizon=2048, max_act=max_act,
                              battle_simulations=200).to_dict() for world in worlds]
            identities = [hashlib.sha256(json.dumps(task,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                          for task in tasks]
            definitions['panels'][label] = {'kind': kind, 'max_act': max_act, 'task_ids': identities}
        expected = json.loads((BANK/'manifest.json').read_text())['provenance']['native_provenance']
        snapshot = capture_final_freeze(checkpoints, files, self.state['native_build'],
                    generated_definitions=definitions, expected_backend_provenance=expected)
        self.state['frozen_selection'] = {**{key: str(path) for key,path in checkpoints.items()}, 'at': snapshot['frozen_at']}
        self.state['frozen_identities'] = snapshot
        atomic(self.destination/'frozen_identities.json', snapshot)
        self.save()
        return snapshot

    def final(self, shared, combat, ooc):
        self.state['phase'] = 'frozen_final_evaluation'
        self.freeze_final_selection(shared, combat, ooc)
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
        self.evaluate_combat('final_combat_extended',combat,split='final',episodes=160)
        required=['final_battle_act1','final_ooc_act1','final_run_act1','final_run_act3','final_composite',
                  'final_combat_extended_act1','final_combat_extended_hard']
        required += [f'final_baseline_{baseline}_{kind}' for kind in ('battle','ooc','run') for baseline in ('random','native','heuristic')]
        required += [f'final_baseline_{baseline}_run_act3' for baseline in ('random','native','heuristic')]
        complete={row['name'] for row in self.state['commands'] if row['status']=='complete'}
        hard_tasks=Path(self.state.get('active_panel_directory',str(PANELS)))/'final_hard.jsonl'
        for baseline in ('random','native','heuristic'):
            name=f'final_baseline_{baseline}_hard'
            self.run(name,[PYTHON,ROOT/'scripts/evaluate_nonllm_baselines.py','--output',self.destination/(name+'.metrics.json'),
                           '--baseline',baseline,'--kind','battle','--split','final','--episodes',256,'--tasks',hard_tasks,
                           '--seed',777,'--horizon',256,'--max-act',3,'--simulations',200,'--workers',8,
                           '--seconds',300,'--fast-observations'],300)
            required.append(name)
        complete={row['name'] for row in self.state['commands'] if row['status']=='complete'}
        self.state['final_required_panels']=required
        self.state['final_missing_panels']=[name for name in required if name not in complete or
            not (self.destination/(name+'.metrics.json')).exists() or
            not panel_complete(json.loads((self.destination/(name+'.metrics.json')).read_text()))]
        self.state['status']='finished' if not self.state['final_missing_panels'] else 'partial'
        self.save()

    def ppo_screen(self, name, source, kind, end, **settings):
        self.state['phase'] = name + '_hyperparameter_screen'
        if name in self.state.get('phase_progress',{}):
            recipe=self.state.get('combat_effective_settings', self.state['combat_expanded_promotion'][0]['settings'])
            return Path(self.state['phase_progress'][name]['best']),recipe
        basic = {'learning_rate':1e-4,'reward_mode':'potential','gamma':1.,
                 'clip_ratio':.2,'entropy_coef':.01,'terminal_hp_weight':0.}
        recipes = [('lr0.0001_potential',{}),('lr0.0003_potential',{'learning_rate':3e-4}),
                   ('lr0.001_potential',{'learning_rate':1e-3}),
                   ('lr0.0003_sparse',{'learning_rate':3e-4,'reward_mode':'sparse'}),
                   ('clip0.1',{'clip_ratio':.1}),('entropy0.005',{'entropy_coef':.005}),
                   ('entropy0.02',{'entropy_coef':.02}),('gamma0.99',{'gamma':.99}),
                   ('terminal_hp0.25',{'terminal_hp_weight':.25}),('scratch',{})]
        architecture = json.loads(Path(self.state['selected']['shared_bc']).with_suffix('.metrics.json').read_text())['config']
        panel=[]
        for label, changes in recipes:
            tag=f'{name}_pilot_{label}'
            recipe={**basic,**changes}
            extra={'size':architecture['size'],'context':architecture['context']} if label=='scratch' else {}
            path,result=self.train(tag,'ppo',None if label=='scratch' else source,seconds=90,end=end,
                                   task_kind=kind,**recipe,**settings,**extra)
            if path:
                evaluation=self.evaluate_combat(tag+'_dev',path,episodes=64,end=end)
                panel.append((self.score(evaluation),path,recipe))
        if not panel:
            return source,basic
        panel.sort(key=lambda x:x[0],reverse=True)
        self.state[name+'_pilot_panel']=[{'score':score,'checkpoint':str(path),'settings':recipe} for score,path,recipe in panel]
        self.save()
        return panel[0][1],panel[0][2]

    def ooc_screen(self, source, end):
        """Select PPO settings on one fixed live OOC development cohort."""
        from sts_policy.ooc_controls import ooc_screen_plan, ooc_evaluation_command
        saved_phase = self.state.get('phase_progress', {}).get('ooc')
        if saved_phase is not None:
            settings = self.state.get('ooc_effective_settings')
            if settings is None:
                # Recover actual historical settings rather than silently
                # changing an already-started phase on a software resume.
                metrics = Path(saved_phase['best']).with_suffix('.metrics.json')
                config = json.loads(metrics.read_text())['config']
                keys = ('learning_rate', 'reward_mode', 'gamma', 'clip_ratio', 'entropy_coef',
                        'progress_floor_weight', 'terminal_hp_weight')
                settings = {key: config[key] for key in keys}
                self.state['ooc_effective_settings'] = settings
                self.save()
            return Path(saved_phase['best']), settings
        self.state['phase'] = 'ooc_hyperparameter_screen'
        source = Path(source)
        basic = {'gamma': 1., 'clip_ratio': .1, 'entropy_coef': .01, 'terminal_hp_weight': 0.}
        default = {**basic, 'learning_rate': 1e-4, 'reward_mode': 'progress', 'progress_floor_weight': .05}
        def acceptable(metrics):
            if metrics is None or not panel_complete(metrics):
                return False
            episodes = sum(len(row.get('episodes', [])) for row in metrics.get('rounds', []))
            faults = sum(len(row.get('faults', [])) for row in metrics.get('rounds', []))
            return faults <= max(2, .1 * (episodes + faults))
        reference_command = ooc_evaluation_command(source, self.destination/'ooc_warmstart_reference_dev.metrics.json', root=ROOT)
        reference = self.run('ooc_warmstart_reference_dev', reference_command, 120, end)
        panel = [{'name': 'warmstart_reference', 'role': 'reference', 'recipe': default,
                  'checkpoint': str(source), 'source': str(source), 'evaluation_command': reference_command,
                  'cohort_complete': reference is not None and panel_complete(reference),
                  'eligible': acceptable(reference), 'score': self.score(reference) if acceptable(reference) else None,
                  'evaluation': reference}]
        self.state['ooc_reference'] = {'checkpoint': str(source),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None}
        for plan in ooc_screen_plan(source, self.destination, root=ROOT):
            checkpoint = Path(plan['checkpoint'])
            training = self.run(plan['name'], plan['training_command'], 90, end)
            evaluation = None
            if training is not None and checkpoint.exists():
                evaluation = self.run(plan['name'] + '_dev', plan['evaluation_command'], 120, end)
            settings = {**basic, 'learning_rate': plan['learning_rate'],
                        'reward_mode': plan['reward_mode'], 'progress_floor_weight': plan['progress_floor_weight']}
            entry = {'recipe': settings, 'checkpoint': str(checkpoint), 'source': str(source),
                     'name': plan['name'], 'role': 'pilot', 'training_command': plan['training_command'],
                     'evaluation_command': plan['evaluation_command'],
                     'cohort_complete': evaluation is not None and panel_complete(evaluation),
                     'eligible': acceptable(evaluation),
                     'evaluation': evaluation}
            entry['score'] = self.score(evaluation) if entry['eligible'] else None
            panel.append(entry)
            self.state['ooc_pilot_panel'] = panel
            self.save()
        eligible = [entry for entry in panel if entry['eligible']]
        if not eligible:
            raise RuntimeError('No completed matched OOC hyperparameter panel; inspect job records')
        best = max(eligible, key=lambda entry: entry['score'])
        self.state['ooc_effective_settings'] = best['recipe']
        self.state['selected']['ooc_pilot'] = best['checkpoint']
        self.save()
        return Path(best['checkpoint']), best['recipe']

    def main(self):
        shared = self.screen()
        retained = self.state.get('retained_combat')
        if retained is not None:
            combat = Path(retained['checkpoint'])
            if not combat.is_file():
                raise RuntimeError('Retained combat checkpoint is missing')
            self.state['selected']['combat'] = str(combat)
            self.save()
        else:
            if HARD_BANK.exists():
                hardened, _ = self.train('combat_act3_bc', 'bc', shared, seconds=180,
                                         teacher=HARD_BANK/'teacher.jsonl', dev_teacher=BANK/'teacher.jsonl', phase='combat')
                if hardened:
                    before = self.evaluate_combat('combat_hardening_before_dev', shared)
                    after = self.evaluate_combat('combat_hardening_after_dev', hardened)
                    if self.score(after) >= self.score(before) - 5:
                        shared = hardened
            combat_start, combat_settings = self.ppo_screen('combat', shared, 'battle', INITIAL+14400,
                                                       tasks=BANK/'battles.jsonl', horizon=256)

            if (EXPANDED_PANELS/'panels.json').exists() and 'combat' not in self.state.get('phase_progress',{}):
                self.state['active_panel_directory']=str(EXPANDED_PANELS)
                promoted=[]
                for candidate in self.state['combat_pilot_panel'][:3]:
                    checkpoint=Path(candidate['checkpoint'])
                    evaluation=self.evaluate_combat(checkpoint.stem+'_expanded_dev',checkpoint,episodes=64)
                    promoted.append((self.score(evaluation),checkpoint,candidate['settings']))
                promoted.sort(key=lambda x:x[0],reverse=True)
                combat_start,combat_settings=promoted[0][1],promoted[0][2]
                self.state['combat_expanded_promotion']=[{'score':score,'checkpoint':str(path),'settings':recipe} for score,path,recipe in promoted]
                self.save()
            rehearsal=ROOT/'data/nonllm_curriculum/native_rehearsal/teacher.jsonl'
            if rehearsal.exists():self.state['native_rehearsal']=str(rehearsal)
            self.state['combat_effective_settings']=combat_settings
            combat = self.curriculum('combat', combat_start, INITIAL+14400, 'battle',
                                     tasks=BANK/'battles.jsonl', horizon=256, teacher_simulations=200,
                                     **combat_settings)
        if 'ooc' not in self.state.get('phase_progress', {}):
            self.state['phase'] = 'ooc_warmstarts'
            self.save()
        specialist, _ = self.train('ooc_specialist_bc', 'bc', shared, seconds=120, phase='ooc')
        ooc_start = specialist or shared
        if LOOKAHEAD_BANK.exists() and not self.state.get('quarantine_lookahead', False):
            matched_scripted=self.destination/'ooc_matched_scripted_teacher.jsonl'
            with (LOOKAHEAD_BANK/'teacher.jsonl').open() as source_rows,matched_scripted.open('w') as output:
                for line in source_rows:
                    row=json.loads(line)
                    row['teacher_index']=row['scripted_index']
                    row['teacher_kind']='native_scripted_same_frontier_ablation'
                    output.write(json.dumps(row,sort_keys=True)+'\n')
            teacher_panel=[]
            for label,teacher in [('scripted_matched',matched_scripted),('lookahead_matched',LOOKAHEAD_BANK/'teacher.jsonl')]:
                trained,_=self.train('ooc_'+label+'_bc','bc',ooc_start,seconds=90,teacher=teacher,
                                     dev_teacher=BANK/'teacher.jsonl',phase='ooc')
                if trained:
                    evaluation=self.evaluate('ooc_'+label+'_dev',trained,'ooc',episodes=32)
                    teacher_panel.append((self.score(evaluation),trained,label))
            teacher_panel.sort(key=lambda x:x[0],reverse=True)
            self.state['ooc_teacher_panel']=[{'score':score,'checkpoint':str(path),'teacher':label} for score,path,label in teacher_panel]
            if teacher_panel:ooc_start=teacher_panel[0][1]
            self.save()
        ooc_start, ooc_settings = self.ooc_screen(ooc_start, INITIAL+21600)
        ooc = self.curriculum('ooc', ooc_start, INITIAL+21600, 'ooc',
                             world_start=20000, horizon=1024, **ooc_settings)
        initial_panel = []
        for label, checkpoint in [('shared_bc', shared), ('combat', combat), ('ooc', ooc)]:
            evaluation = self.evaluate('joint_start_' + label + '_dev', checkpoint, 'run', episodes=32,
                                       end=INITIAL+28800)
            initial_panel.append((self.score(evaluation), checkpoint))
        composite_dev=self.evaluate_composite('joint_start_composite_dev',combat,ooc,end=INITIAL+28800)
        self.state['composite_initial_dev_score']=self.score(composite_dev)
        initial_panel.sort(key=lambda row: row[0], reverse=True)
        self.state['joint_start_panel'] = [{'score': score, 'checkpoint': str(path)} for score, path in initial_panel]
        composite, report = self.joint_composition(combat, ooc)
        self.state['joint_composition'] = report
        references = [('single_' + str(index), score, path) for index, (score,path) in enumerate(initial_panel)]
        references.append(('composite', self.score(composite_dev), composite))
        self.save()
        joint_start, joint_settings = self.joint_screen(initial_panel[0][1], composite, references, INITIAL+28800)
        full = self.curriculum('joint_act1', joint_start, INITIAL+28800, 'run',
                              world_start=30000, horizon=2048, **joint_settings)
        full = self.curriculum('joint_act3', full, INITIAL+32400, 'run',
                              world_start=40000, horizon=4096, max_act=3, **joint_settings)
        self.final(full, combat, ooc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/nonllm_v1/campaign')
    parser.add_argument('--bank', type=Path, help='Fresh Act 1 teacher bank for this native revision')
    parser.add_argument('--hard-bank', type=Path, help='Fresh later-act teacher bank for this revision')
    parser.add_argument('--panels', type=Path, help='Frozen panels for this revision')
    parser.add_argument('--lookahead-bank', type=Path, help='Revision-matched optional OOC lookahead bank')
    args = parser.parse_args()
    if args.bank is not None:
        BANK = args.bank.resolve()
    if args.hard_bank is not None:
        HARD_BANK = args.hard_bank.resolve()
    if args.panels is not None:
        PANELS = EXPANDED_PANELS = args.panels.resolve()
    if args.lookahead_bank is not None:
        LOOKAHEAD_BANK = args.lookahead_bank.resolve()
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
