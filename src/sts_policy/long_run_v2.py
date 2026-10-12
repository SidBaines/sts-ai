"""V2 continuous full-game learner with a separate, confirmed DEV champion.

The inherited controller supplies owned process/GPU locks, active accounting,
resource pauses and watchdogs. Ordinary development outcomes never roll back
this learner. No old campaign, frozen input, final cohort or reward is rewritten.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import time
from typing import Any

from .campaign_identity import condition_signature
from .long_run_control import (LongRunConfig, LongRunController, Stage, actor_weight_identity,
                              atomic_json, digest, goal_signature, group_members, panel_quality,
                              retain_checkpoints, sha_file, stop_owned_group)

V2_STAGE = Stage('full3_continuous', 'full3', 3, 1., {'scratch': .70, 'act2': .20, 'act3': .10})
PANELS = ('screen_full3', 'confirm_full3', 'retention_act1')


@dataclass(frozen=True)
class V2Config(LongRunConfig):
    train_world_start: int = 100000000
    dev_world_start: int = 400000
    evaluation_seconds: float = 300.
    initial_active_seconds: float = 18424.127912292024
    screen_interval_seconds: float = 600.
    confirmation_interval_seconds: float = 1800.
    promising_min_interval_seconds: float = 900.
    severe_retention_fraction: float = .15
    severe_floor_drop: float = 8.
    severe_cycles: int = 2
    plateau_warning_seconds: float = 1800.
    plateau_attention_seconds: float = 3600.
    entrypoint_source_sha256: str | None = None
    budget_transfer_path: str | None = None
    budget_transfer_sha256: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if not 100000000 <= self.train_world_start < 100000000000 or self.dev_world_start < 400000:
            raise ValueError('V2 namespaces must be disjoint from V1')
        if not math.isfinite(self.initial_active_seconds) or self.initial_active_seconds <= 0:
            raise ValueError('V2 needs a positive finite authorized active allocation')
        if (self.gamma, self.gae_lambda, self.progress_floor_weight, self.terminal_hp_weight) != (1., .95, .05, 0.):
            raise ValueError('Initial V2 reward/credit coefficients are fixed by the authorized protocol')
        if min(self.screen_interval_seconds, self.confirmation_interval_seconds, self.promising_min_interval_seconds,
               self.plateau_warning_seconds, self.plateau_attention_seconds) <= 0:
            raise ValueError('Monitoring intervals must be positive')
        if self.promising_min_interval_seconds > self.confirmation_interval_seconds or not 0 < self.severe_retention_fraction < 1:
            raise ValueError('Invalid confirmation/attention policy')
        if self.severe_cycles < 2 or self.severe_floor_drop <= 0:
            raise ValueError('Strong regression requires repeated cycles and a positive floor threshold')


def outcome_quality(metrics):
    quality = panel_quality(metrics)
    quality['unstarted'] = sum(r.get('unstarted_tasks', 0) for r in metrics.get('rounds', []))
    quality['outcomes'] = {e['task_id']: {'outcome': e.get('outcome'), 'floor': e.get('floor')}
                           for r in metrics.get('rounds', []) for e in r.get('episodes', [])}
    return quality


def paired_evidence(candidate, reference):
    if candidate is None or reference is None:
        return {'valid': False, 'reason': 'missing_panel'}
    n = candidate.get('requested', 0)
    if (not candidate.get('complete') or not reference.get('complete') or
            candidate.get('cohort') != reference.get('cohort') or n != reference.get('requested') or
            candidate.get('unstarted', 0) or reference.get('unstarted', 0) or candidate.get('truncated') or reference.get('truncated')):
        return {'valid': False, 'reason': 'cohort_or_censor_incomplete'}
    fault_limit = math.floor(.02 * n)
    if max(candidate['faults'], reference['faults']) > fault_limit:
        return {'valid': False, 'reason': 'fault_rate_above_prespecified_two_percent'}
    common = sorted(set(candidate['cohort']) - set(candidate['fault_ids']) - set(reference['fault_ids']))
    if len(common) < math.ceil(.97 * n):
        return {'valid': False, 'reason': 'common_valid_coverage_below_97_percent'}
    left, right = candidate.get('outcomes', {}), reference.get('outcomes', {})
    if any(i not in left or i not in right for i in common):
        return {'valid': False, 'reason': 'missing_per_task_outcome_proof'}
    won = lambda row: row.get('outcome') in ('victory', 'act_clear', 'battle_victory')
    left_wins = sum(won(left[i]) for i in common); right_wins = sum(won(right[i]) for i in common)
    allowed = {'victory','act_clear','battle_victory','defeat','battle_defeat'}
    if any(left[i].get('outcome') not in allowed or right[i].get('outcome') not in allowed for i in common):
        return {'valid': False, 'reason': 'unresolved_terminal_outcome'}
    floors = [(float(left[i]['floor']), float(right[i]['floor'])) for i in common
              if isinstance(left[i].get('floor'), (int, float)) and not isinstance(left[i]['floor'],bool) and math.isfinite(left[i]['floor']) and
                 isinstance(right[i].get('floor'), (int, float)) and not isinstance(right[i]['floor'],bool) and math.isfinite(right[i]['floor'])]
    if len(floors) != len(common): return {'valid': False, 'reason': 'missing_paired_floor_proof'}
    return {'valid': True, 'requested': n, 'common_valid': len(common), 'fault_limit': fault_limit,
            'common_task_ids': common, 'candidate_wins': left_wins, 'reference_wins': right_wins,
            'paired_win_gain': left_wins - right_wins,
            'winning_world_gains': [i for i in common if won(left[i]) and not won(right[i])],
            'winning_world_losses': [i for i in common if won(right[i]) and not won(left[i])],
            'candidate_floor_mean': sum(x for x,y in floors) / len(floors),
            'reference_floor_mean': sum(y for x,y in floors) / len(floors),
            'paired_floor_gain': sum(x-y for x,y in floors) / len(floors)}


def confirmed_decision(candidate, champion, retention, initial_retention):
    paired, retained = paired_evidence(candidate, champion), paired_evidence(retention, initial_retention)
    if not paired['valid'] or not retained['valid']:
        return {'action': 'keep_champion', 'reason': 'fault_or_cohort_evidence_not_classifiable',
                'paired': paired, 'retention_paired': retained, 'classifiable': False}
    all_win_non_decrease = candidate['clears'] >= champion['clears']
    promote = all_win_non_decrease and (paired['paired_win_gain'] > 0 or
                  (paired['paired_win_gain'] == 0 and paired['paired_floor_gain'] > 0))
    return {'action': 'promote' if promote else 'keep_champion',
            'reason': 'descriptive_paired_full3_improvement' if promote else 'no_confirmed_full3_rank_improvement',
            'paired': paired, 'retention_paired': retained, 'classifiable': True,
            'uncertainty': 'reused DEV evidence; rank improvement is not a significance claim'}


def severe_regression(candidate, predecision_champion, retention, initial_retention, config):
    paired, retained = paired_evidence(candidate, predecision_champion), paired_evidence(retention, initial_retention)
    if not paired['valid'] or not retained['valid']: return None
    no_win_gain = paired['paired_win_gain'] <= 0
    if no_win_gain and -retained['paired_win_gain'] >= math.ceil(config.severe_retention_fraction * retention['requested']):
        return 'large_act1_retention_loss_without_full3_win_gain'
    if no_win_gain and paired['paired_floor_gain'] <= -config.severe_floor_drop:
        return 'large_full3_floor_regression_without_win_gain'
    return None


def evaluation_attempt_policy(records, retries):
    failures=0
    for row in records:
        reason=row.get('stop_reason','')
        expected=row.get('status')=='operator_interrupted' or reason.startswith('resource:') or reason=='active_budget_or_session_limit'
        if row.get('status')!='complete' and not expected: failures+=1
    if failures>retries: raise RuntimeError('Bounded genuine evaluation failures exhausted; expected pauses are not failures')
    return len(records)


def finite_policy_checkpoint(path):
    """CPU-only file validation; no model execution or accelerator work."""
    import torch
    from .training import load_checkpoint
    payload = load_checkpoint(path)
    if not all(torch.isfinite(value).all().item() for value in payload['model_state'].values()):
        raise RuntimeError('Nonfinite checkpoint weights; do not advance learner')
    return payload


class V2Controller(LongRunController):
    def __init__(self, root, output, config: V2Config):
        entrypoint = Path(root) / 'scripts/run_long_curriculum_v2.py'
        config = replace(config, duration_hours=config.initial_active_seconds / 3600,
                         entrypoint_source_sha256=config.entrypoint_source_sha256 or sha_file(entrypoint))
        self.pause_requested = False
        super().__init__(root, output, config)

    def stage_deadline(self, stage=None):
        return super().stage_deadline(None)

    def save(self):
        if self.state.get('version') == 'sts_long_run_v1':
            self.state['version'] = 'sts_long_run_v2'
            self.state['allocation']['stage_boundaries'] = {'full3_continuous': self.state['allocation']['authorized_seconds']}
            self.state.update(learner={'checkpoint': self.state['current_checkpoint'], 'sha256': self.state['identity']['initial_checkpoint_sha256'],
                'lineage_chunks': 0, 'updates': 0, 'transitions': 0, 'cold_optimizer_count': 0,
                'optimizer_restore_count': 0, 'rollback_count': 0, 'parent_sha256': None},
                champion={'checkpoint': self.state['best_checkpoint'], 'sha256': self.state['identity']['initial_checkpoint_sha256'],
                          'promotions': 0, 'active_at': 0.}, evaluations=[], initial_panels={},
                learning_health={'severe_streak': 0, 'last_severe_sha': None, 'plateau_warning': False, 'inspection_needed': False},
                monitoring={'last_screen_active': 0., 'last_confirmation_active': 0.})
        self._publish_live = False  # V2 has its own generated document below.
        super().save()
        progress = json.loads((self.output / 'progress.json').read_text())
        progress.update({key: self.state.get(key) for key in ('learner','champion','learning_health','monitoring','initial_panels')})
        atomic_json(self.output / 'progress.json', progress)
        if getattr(self, '_v2_live', False):
            lines = ['# Full-game RL v2 live progress', '', 'Status: ' + self.state['status'],
                     'Current learner: ' + self.state['current_checkpoint'], 'Evaluation champion: ' + self.state['best_checkpoint'],
                     'Remaining active seconds: ' + str(max(0., self.state['allocation']['authorized_seconds'] - self.state['allocation']['used_seconds'])),
                     'Reward coefficients remain +.05 floor progress, +1 victory, −1 death; all TRAIN goals are Act3.',
                     '', 'Learner lineage: ' + json.dumps(self.state['learner'], sort_keys=True),
                     'Champion: ' + json.dumps({k:v for k,v in self.state['champion'].items() if k not in ('quality','retention','screen_quality')}, sort_keys=True),
                     'Learning health: ' + json.dumps(self.state['learning_health'], sort_keys=True), '', 'Latest events first:']
            lines += [json.dumps(e, sort_keys=True) for e in reversed(self.state['events'][-20:])]
            signature = digest(lines)
            if getattr(self, '_last_live_signature', None) != signature:
                path = self.root / 'docs/long_curriculum_v2_live.md'; temporary = path.with_suffix('.tmp')
                temporary.write_text('\n'.join(lines) + '\n'); os.replace(temporary, path); self._last_live_signature = signature

    def verify_pins(self):
        super().verify_pins()
        if sha_file(self.root / 'scripts/run_long_curriculum_v2.py') != self.config.entrypoint_source_sha256:
            raise RuntimeError('Pinned V2 entrypoint changed')
        if self.config.budget_transfer_path and sha_file(self.config.budget_transfer_path) != self.config.budget_transfer_sha256:
            raise RuntimeError('Approved budget transfer evidence changed')

    def extend_hours(self, hours):
        if not math.isfinite(hours) or hours <= 0: raise ValueError('Explicit extension must be positive and finite')
        self.account_time(); allocation = self.state['allocation']; extra = hours * 3600
        allocation['authorized_seconds'] += extra; allocation['stage_boundaries']['full3_continuous'] += extra
        entry = {'at':time.time(),'hours':hours,'added_seconds':extra,'authority':'explicit_operator_extension','stage':'full3_continuous'}
        allocation['extensions'].append(entry); self.state['events'].append({'type':'allocation_extension',**entry}); self.save()

    def safe_owned_cleanup(self):
        """EPERM may race an exited group; never conceal a still-live owner."""
        for record in self.state['jobs']:
            if record.get('status') not in ('running','launching'): continue
            try: stop_owned_group(record)
            except PermissionError:
                live = group_members(record.get('pgid', 0))
                owned = dict(record.get('owned_members', {}))
                if record.get('pid') and record.get('process_birth'): owned[str(record['pid'])] = record['process_birth']
                if any(live.get(pid) == birth for pid,birth in owned.items()):
                    raise RuntimeError('Pause cleanup denied while verified owned process remains alive')
            record['status'] = 'operator_interrupted' if self.pause_requested else 'recovered_interruption'
            record['finished'] = time.time()
            files = [Path(record['metrics']), Path(record['checkpoint']) if record.get('checkpoint') else None]
            record['artifacts'] = {str(p):sha_file(p) for p in files if p and p.is_file()}
        self.state.pop('active_job', None)

    def run_job(self, *args, **kwargs):
        try: return super().run_job(*args, **kwargs)
        except BaseException as exc:
            if self.pause_requested or isinstance(exc, KeyboardInterrupt):
                self.pause_requested = True; self.safe_owned_cleanup()
                raise KeyboardInterrupt('Verified operator pause') from None
            raise

    def recover(self):
        self.safe_owned_cleanup()
        pending = self.state.get('pending_iteration')
        if pending and not pending.get('training_committed') and pending.get('candidate'):
            self.preserve_partial(pending)
            pending['attempt'] = pending.get('attempt', 0) + 1
            pending['recovery_failures'] = pending.get('recovery_failures', 0) + 1
            if pending['recovery_failures'] > self.config.retries + 1: raise RuntimeError('Bounded interruption recovery exhausted')
        self.save()

    def preserve_partial(self, pending):
        path = Path(pending.get('candidate', ''))
        if not path.is_file(): return False
        finite_policy_checkpoint(path)
        return super().preserve_partial(pending)

    def protocol_inputs(self):
        from .curriculum import TaskSpec, world_split
        from .rollout_training import task_identity
        from sts_env.provenance import interface_identity
        protocol = json.loads(Path(self.config.protocol).read_text())
        transfer=protocol.get('budget_transfer',{})
        if transfer.get('authorized_seconds')!=self.config.initial_active_seconds or (self.config.budget_transfer_path and
                (Path(transfer['path']).resolve()!=Path(self.config.budget_transfer_path).resolve() or transfer['sha256']!=self.config.budget_transfer_sha256)):
            raise RuntimeError('Authorized remaining budget transfer differs')
        if protocol['checkpoint']['sha256'] != self.state['identity']['initial_checkpoint_sha256'] or Path(protocol['checkpoint']['path']).resolve() != Path(self.config.checkpoint).resolve():
            raise RuntimeError('V2 must restart the authorized ORIGINAL checkpoint')
        panels, anchors = {}, {}
        interface = interface_identity()['python_interface_sha256']
        for label in PANELS:
            entry = protocol['panels'][label]; path = Path(entry['path'])
            tasks = [TaskSpec.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]
            ids = sorted(task_identity(t) for t in tasks)
            count = 48 if label == 'screen_full3' else 192
            goal = 1 if label == 'retention_act1' else 3
            if (sha_file(path) != entry['sha256'] or len(tasks) != count or len({t.seed for t in tasks}) != count or len(set(ids)) != count or ids != sorted(entry['task_ids']) or
                    any(world_split(t.seed) != 'dev' or t.seed < 400000 or t.kind != 'run' or t.max_act != goal or t.horizon != entry['horizon'] or t.battle_simulations != self.config.simulations or t.ascension != 0 or t.prefix for t in tasks)):
                raise RuntimeError('V2 fixed DEV cohort differs from approved protocol')
            panels[label] = path
            anchor = protocol['anchors'][label]; metrics = json.loads(Path(anchor['metrics']).read_text()); quality = outcome_quality(metrics)
            cfg, env = metrics['config'], metrics.get('training_environment', {})
            if (sha_file(anchor['metrics']) != anchor['metrics_sha256'] or anchor['checkpoint_sha256'] != protocol['checkpoint']['sha256'] or
                    condition_signature(anchor['command']) != anchor['condition_signature'] or not quality['complete'] or quality['cohort'] != ids or
                    Path(cfg['checkpoint']).resolve() != Path(self.config.checkpoint).resolve() or cfg['seed'] != 777 or cfg['max_act'] != goal or
                    Path(cfg['tasks']).resolve() != path.resolve() or cfg['goal_act_clear'] is not True or cfg['reward_mode'] != 'progress' or
                    cfg['gamma'] != 1 or cfg['progress_floor_weight'] != .05 or cfg['terminal_hp_weight'] != 0 or
                    anchor['condition_signature']['inputs']['--tasks']['sha256'] != entry['sha256'] or
                    anchor['condition_signature']['inputs']['--checkpoint']['sha256'] != protocol['checkpoint']['sha256'] or
                    cfg['split'] != 'dev' or cfg['horizon'] != entry['horizon'] or cfg['simulations'] != self.config.simulations or
                    env.get('native_module_sha256') != self.backend_identity['provenance']['module_sha256'] or env.get('python_interface_sha256') != interface):
                raise RuntimeError('V2 initial anchor proof incomplete, changed or wrong condition')
            quality.update(metrics=anchor['metrics'],metrics_sha256=anchor['metrics_sha256']); anchors[label] = quality
        confirm_worlds = {json.loads(line)['seed'] for line in panels['confirm_full3'].read_text().splitlines() if line}
        screen_worlds = {json.loads(line)['seed'] for line in panels['screen_full3'].read_text().splitlines() if line}
        retention_worlds = {json.loads(line)['seed'] for line in panels['retention_act1'].read_text().splitlines() if line}
        if screen_worlds & confirm_worlds or confirm_worlds != retention_worlds: raise RuntimeError('V2 screen/confirmation/paired-retention split changed')
        return panels, anchors

    def command(self, mode, source, tasks, destination, *, max_act, seconds, seed, warm=False, horizon=None):
        command = super().command(mode, source, tasks, destination, max_act=max_act, seconds=seconds, seed=seed, warm=warm, horizon=horizon)
        if mode == 'evaluate':
            command[command.index('--episodes') + 1] = str(sum(bool(line.strip()) for line in tasks.read_text().splitlines()))
        return list(map(str,command))

    def evaluate_panel(self, panel_id, source, panels):
        tasks = panels[panel_id]; rows = [json.loads(line) for line in tasks.read_text().splitlines() if line]
        source_sha = sha_file(source); label = f'{panel_id}_{source_sha[:20]}'
        destination=self.output/'metrics'/label
        command=self.command('evaluate',source,tasks,destination,max_act=rows[0]['max_act'],seconds=self.config.evaluation_seconds,seed=777,horizon=rows[0]['horizon'])
        destination=destination.with_name(destination.name+'_'+digest(self._signature(command))[:16])
        existing=[j for j in self.state['jobs'] if j.get('kind')=='eval' and j.get('metrics','').startswith(str(destination))]
        complete=next((j for j in reversed(existing) if j.get('status')=='complete'),None)
        if complete: command=complete['command']
        else:
            attempt=evaluation_attempt_policy(existing,self.config.retries)
            if attempt: destination=destination.with_name(destination.name+f'_attempt{attempt}')
            command=self.command('evaluate',source,tasks,destination,max_act=rows[0]['max_act'],seconds=self.config.evaluation_seconds,seed=777,horizon=rows[0]['horizon'])
        metrics,record=self.run_job(command,kind='eval',budget=self.config.evaluation_seconds,stage_end=self.stage_deadline())
        if metrics is None: return None
        enriched=outcome_quality(metrics)
        from .curriculum import TaskSpec
        from .rollout_training import task_identity
        expected=sorted(task_identity(TaskSpec.from_dict(row)) for row in rows)
        if enriched['cohort']!=expected or metrics['training_environment'].get('native_module_sha256')!=self.backend_identity['provenance']['module_sha256']:
            raise RuntimeError('V2 evaluation native/cohort differs')
        enriched.update(metrics=record['metrics'],metrics_sha256=sha_file(record['metrics']))
        cfg, env = metrics['config'], metrics['training_environment']
        from sts_env.provenance import interface_identity
        if (Path(cfg['checkpoint']).resolve()!=source.resolve() or cfg['seed']!=777 or cfg['split']!='dev' or cfg['task_kind']!='run' or cfg['tasks'] != str(tasks) or cfg['horizon'] != rows[0]['horizon'] or cfg['max_act'] != rows[0]['max_act'] or
                cfg['episodes'] != len(rows) or cfg['simulations'] != self.config.simulations or
                cfg['device'] != 'cpu' or cfg['inference_device'] != 'cpu' or cfg['workers'] != self.config.workers or cfg['threads'] != self.config.threads or
                env.get('python_interface_sha256') != interface_identity()['python_interface_sha256']):
            raise RuntimeError('V2 evaluation configuration/source binding differs')
        record.update(panel_id=panel_id, checkpoint_sha256=source_sha, round=self.state['global_round'], stage=V2_STAGE.name)
        self.state['evaluations'].append({'at':time.time(),'active_seconds':self.state['allocation']['used_seconds'],
            'panel_id':panel_id,'role':'learner','round':self.state['global_round'],'checkpoint':str(source),
            'checkpoint_sha256':source_sha,'quality':enriched})
        self.save(); return enriched

    def monitor_learner(self, panels):
        source = Path(self.state['current_checkpoint']); source_sha = self.state['learner']['sha256']
        now = self.state['allocation']['used_seconds']; monitor = self.state['monitoring']
        if now - monitor['last_screen_active'] >= self.config.screen_interval_seconds:
            screen = self.evaluate_panel('screen_full3', source, panels)
            if screen is None: return
            monitor.update(last_screen_active=self.state['allocation']['used_seconds'], last_screen_quality=screen, last_screen_sha=source_sha)
            paired = paired_evidence(screen, self.state['champion']['screen_quality'])
            monitor['screen_promising'] = paired['valid'] and (paired['paired_win_gain'] > 0 or
                            (paired['paired_win_gain'] == 0 and paired['paired_floor_gain'] >= 2.))
            self.save()
        elapsed = self.state['allocation']['used_seconds'] - monitor['last_confirmation_active']
        due = elapsed >= self.config.confirmation_interval_seconds or (monitor.get('screen_promising') and elapsed >= self.config.promising_min_interval_seconds)
        if due and not self.state.get('pending_confirmation'):
            self.state['pending_confirmation'] = {'source':str(source),'sha256':source_sha,'round':self.state['global_round'],
                'screen':monitor.get('last_screen_quality') if monitor.get('last_screen_sha') == source_sha else None}
            self.state['hourly_anchors']['pending_confirmation'] = str(source); self.save()
        self.finish_confirmation(panels)
        self.update_plateau()

    def finish_confirmation(self, panels):
        pending = self.state.get('pending_confirmation')
        if not pending: return
        source = Path(pending['source'])
        if sha_file(source) != pending['sha256']: raise RuntimeError('Pending confirmation checkpoint changed')
        for key, label in (('screen','screen_full3'), ('quality','confirm_full3'), ('retention','retention_act1')):
            if key not in pending or pending[key] is None:
                result = self.evaluate_panel(label, source, panels)
                if result is None: return
                pending[key] = result; self.save()
        before = self.state['champion']
        decision = confirmed_decision(pending['quality'], before['quality'], pending['retention'], self.state['initial_panels']['retention_act1'])
        reason = severe_regression(pending['quality'], before['quality'], pending['retention'], self.state['initial_panels']['retention_act1'], self.config)
        if reason is not None and decision['action']=='promote':
            decision=dict(decision,action='keep_champion',reason='promotion_blocked_by_classified_severe_regression',
                          otherwise_promising=True,severe_reason=reason)
        health = self.state['learning_health']; sha = pending['sha256']
        if reason is not None and health.get('last_severe_sha') != sha:
            health['severe_streak'] += 1; health['last_severe_sha'] = sha
        elif reason is None and decision['classifiable']:
            health['severe_streak'] = 0; health['last_severe_sha'] = None
        event = {'at':time.time(),'type':'champion_confirmation','stage':V2_STAGE.name,'round':pending['round'],
                 'checkpoint_sha256':sha,'decision':decision,'quality':pending['quality'],'retention':pending['retention'],
                 'initial_full3_paired':paired_evidence(pending['quality'],self.state['initial_panels']['confirm_full3']),
                 'severe_reason':reason,'severe_streak':health['severe_streak']}
        self.state['events'].append(event)
        if decision['action'] == 'promote':
            self.state['champion'] = {'checkpoint':str(source),'sha256':sha,'quality':pending['quality'],
                'screen_quality':pending['screen'],'retention':pending['retention'],'promotions':before['promotions'] + 1,
                'active_at':self.state['allocation']['used_seconds'],'last_eval_at':time.time()}
            self.state['best_checkpoint'] = str(source)
            if decision['paired']['paired_win_gain'] > 0:
                health.update(last_win_gain_active=self.state['allocation']['used_seconds'],plateau_warning=False,inspection_needed=False)
        monitor = self.state['monitoring']; monitor.update(last_confirmation_active=self.state['allocation']['used_seconds'], screen_promising=False)
        self.state.pop('pending_confirmation', None); self.state['hourly_anchors'].pop('pending_confirmation', None)
        self.save()
        if health['severe_streak'] >= self.config.severe_cycles:
            self.state['status'] = 'attention_required'; self.save()
            raise RuntimeError('Repeated strong regression on distinct confirmed learner snapshots; warm learner and champion preserved')

    def update_plateau(self):
        health = self.state['learning_health']; now = self.state['allocation']['used_seconds']
        age = now - health.get('last_win_gain_active', 0.)
        health['full3_win_plateau_seconds'] = age
        health['champion_rank_plateau_seconds'] = now - self.state['champion'].get('active_at', 0.)
        if age >= self.config.plateau_warning_seconds and not health['plateau_warning']:
            health['plateau_warning'] = True
            self.state['events'].append({'at':time.time(),'type':'plateau_warning','active_seconds_without_paired_full3_win_gain':age,
                                         'action':'diagnostic_only; reward/hyperparameters and warm parent unchanged'})
        if age >= self.config.plateau_attention_seconds and not health['inspection_needed']:
            health['inspection_needed'] = True
            self.state['events'].append({'at':time.time(),'type':'persistent_plateau_inspection_needed','active_seconds':age})
        self.save()

    def validate_training_result(self, pending, metrics, record):
        cfg = metrics.get('config', {}); env = metrics.get('training_environment', {})
        expected = {'task_kind':'run','split':'train','max_act':3,'horizon':self.config.horizon,
                    'gamma':1.,'gae_lambda':.95,'progress_floor_weight':.05,'terminal_hp_weight':0.,
                    'goal_act_clear':True,'reward_mode':'progress','learning_rate':self.config.learning_rate,
                    'clip_ratio':self.config.clip_ratio,'entropy_coef':self.config.entropy_coef,
                    'workers':self.config.workers,'threads':self.config.threads,'episodes':self.config.episodes,
                    'device':self.config.device,'inference_device':'cpu','updates':1}
        if (any(cfg.get(k)!=v for k,v in expected.items()) or cfg.get('checkpoint')!=pending['source'] or
                cfg.get('tasks')!=pending['tasks'] or record.get('checkpoint')!=pending['candidate'] or
                env.get('native_module_sha256')!=self.backend_identity['provenance']['module_sha256']):
            raise RuntimeError('Training output does not match source, tasks, fixed goal/reward/backend/config')
        rows=pending['metadata']['tasks']; ids=Counter(row['task_id'] for row in rows)
        if any(row.get('goal_act')!=3 or row.get('horizon')!=self.config.horizon for row in rows):
            raise RuntimeError('V2 training metadata contains a different objective')
        if len(metrics.get('rounds',[]))!=1: raise RuntimeError('Expected one fresh-task PPO round per chunk')
        for rollout in metrics['rounds']:
            actual=Counter(e.get('task_id') for e in rollout.get('episodes',[])+rollout.get('faults',[]))
            if Counter(rollout.get('task_ids',[]))!=ids or actual-ids or sum((ids-actual).values())!=rollout.get('unstarted_tasks',0):
                raise RuntimeError('TRAIN requested/observed TaskID coverage differs from immutable batch')
        candidate=Path(pending['candidate'])
        if record.get('artifacts',{}).get(str(candidate))!=sha_file(candidate):raise RuntimeError('Checkpoint is not bound to completed artifact SHA')

    def advance_learner(self, pending, metrics, record):
        self.validate_training_result(pending,metrics,record)
        candidate = Path(pending['candidate']); finite_policy_checkpoint(candidate)
        valid = sum(r.get('transitions', 0) for r in metrics['rounds']); updates = sum(r.get('update', {}).get('steps', 0) for r in metrics['rounds'])
        for row in metrics['rounds']:
            for key, value in row.get('update', {}).items():
                if isinstance(value, (int,float)) and not isinstance(value,bool) and not math.isfinite(value):
                    raise RuntimeError('Nonfinite logged PPO update; learner not advanced')
        before, after = actor_weight_identity(Path(pending['source'])), actor_weight_identity(candidate)
        if valid < 1 or updates < 1 or before['actor_sha256'] == after['actor_sha256']:
            raise RuntimeError('No genuine actor update; preserve evidence and pause rather than repeat an ineffective condition')
        parent_sha = sha_file(pending['source']); candidate_sha = sha_file(candidate)
        record.update(actor_evidence={'before':before,'after':after,'updates':updates,'transitions':valid},
            parent_sha256=parent_sha, parent_kind='continuous_learner' if pending['source']==self.state['current_checkpoint'] else 'verified_partial_recovery', checkpoint_sha256=candidate_sha, round=pending['index'], stage=V2_STAGE.name,
            optimizer_goal={'restored_requested':pending['warm'],'target':goal_signature(self.config,3,V2_STAGE)})
        learner = self.state['learner']
        if pending['source']!=self.state['current_checkpoint']: learner['lineage_counts_complete']=False
        restored = metrics.get('optimizer_carry', {}).get('restored', False)
        if pending['warm'] and not restored: raise RuntimeError('Unexpected optimizer reset in continuous V2 learner')
        learner.update(checkpoint=str(candidate),sha256=candidate_sha,parent_sha256=parent_sha,last_advanced_at=time.time())
        learner['lineage_chunks'] += 1; learner['updates'] += updates; learner['transitions'] += valid
        learner['optimizer_restore_count'] += int(restored); learner['cold_optimizer_count'] += int(not restored)
        self.state.update(current_checkpoint=str(candidate),last_checkpoint=str(candidate),current_goal=goal_signature(self.config,3,V2_STAGE))
        progress = self.state['stages'].setdefault(V2_STAGE.name, {'round':0,'updates':0,'transitions':0})
        progress['round'] += 1; progress['updates'] += updates; progress['transitions'] += valid
        metadata = {row['task_id']:row for row in pending['metadata']['tasks']}
        strata = {}
        def counts(identity):
            meta = metadata[identity]; key = json.dumps({k:meta.get(k) for k in ('start_source','start_act','goal_act','horizon')},sort_keys=True)
            return strata.setdefault(key,{k:0 for k in ('requested','episodes','clears','defeats','faults','truncated','unstarted')})
        for rollout in metrics['rounds']:
            actual = Counter(e['task_id'] for e in rollout.get('episodes', []) + rollout.get('faults', []))
            for identity in rollout['task_ids']: counts(identity)['requested'] += 1
            for identity in (Counter(rollout['task_ids']) - actual).elements(): counts(identity)['unstarted'] += 1
            for episode in rollout.get('episodes', []):
                c=counts(episode['task_id']);c['episodes']+=1;c['clears']+=int(episode.get('outcome')=='victory');c['defeats']+=int(episode.get('outcome')=='defeat');c['truncated']+=bool(episode.get('truncated'))
            for fault in rollout.get('faults', []): counts(fault['task_id'])['faults'] += 1
        record['family_strata'] = strata
        faults = sum(c['faults'] for c in strata.values()); requested = sum(c['requested'] for c in strata.values())
        excess = faults > max(3, .05*requested)
        self.state['consecutive_excessive_fault_chunks'] = self.state.get('consecutive_excessive_fault_chunks',0)+1 if excess else 0
        self.state['events'].append({'at':time.time(),'type':'learner_advanced','round':pending['index'],
            'parent_sha256':parent_sha,'checkpoint_sha256':candidate_sha,'updates':updates,'transitions':valid,'warm_optimizer':restored})
        pending['training_committed'] = True; self.save()
        if self.state['consecutive_excessive_fault_chunks'] >= 2: raise RuntimeError('Persistent excessive TRAIN faults; preserve current learner for attention')

    def execute(self, *, session_seconds=None, session_kind='main', acknowledge_attention=False):
        from . import long_run_v2_tasks as sampler
        if self.state['status'] == 'attention_required' and not acknowledge_attention:
            raise RuntimeError('Attention pause requires explicit --acknowledge-attention; no automatic resume')
        self._v2_live = True; self.current_stage = V2_STAGE.name; self.session_kind = session_kind
        self.session_end = time.time()+session_seconds if session_seconds else math.inf
        self.set_accounting(True); self.recover(); self.verify_pins()
        pool = sampler.load_continuations(bank=Path(self.config.pool_bank), metadata_panels=Path(self.config.pool_panels))
        if self.state.get('native_pool_identity',pool.source_identity) != pool.source_identity: raise RuntimeError('Native continuation pool changed')
        self.state['native_pool_identity'] = pool.source_identity
        if self.state.get('learner_continuations'): pool = sampler.add_learner_continuations(pool,self.state['learner_continuations'])
        panels, anchors = self.protocol_inputs()
        self.state['initial_panels'] = anchors
        self.state['dev_file_hashes'] = {str(path):sha_file(path) for path in panels.values()}
        champion = self.state['champion']
        if 'quality' not in champion:
            champion.update(quality=anchors['confirm_full3'],retention=anchors['retention_act1'],screen_quality=anchors['screen_full3'])
        self.state['status'] = 'running'; self.save()
        try:
            self.finish_confirmation(panels)
            while self.remaining_seconds()>30 and time.time()<self.session_end-30 and self.pause_until_safe():
                pending = self.state.get('pending_iteration')
                if not pending:
                    self.verify_pins(); index = self.state['global_round']; tasks = self.output/'tasks'/f'full3_continuous_{index:08d}.jsonl'
                    batch = sampler.sample_v2(pool,round_index=index,episodes=self.config.episodes,seed=self.config.seed,
                        horizon=self.config.horizon,simulations=self.config.simulations,world_start=self.config.train_world_start)
                    written = sampler.write_task_batch(batch,tasks)
                    manifest = Path(written['manifest'])
                    pending={'index':index,'stage':V2_STAGE.name,'source':self.state['current_checkpoint'],'tasks':str(tasks),
                        'task_sha256':written['sha256'],'manifest_sha256':sha_file(manifest),'metadata':batch['metadata'],
                        'attempt':0,'session_kind':session_kind,'warm':self.state['learner']['lineage_chunks']>0,
                        'seconds':min(self.config.chunk_seconds,max(1.,self.stage_deadline()-time.time()-5))}
                    self.state['pending_iteration']=pending;self.save()
                if sha_file(pending['tasks'])!=pending['task_sha256'] or sha_file(pending['tasks']+'.manifest.json')!=pending['manifest_sha256']:
                    raise RuntimeError('Pending immutable TRAIN batch changed')
                if not pending.get('training_committed'):
                    base=self.output/'checkpoints'/f'full3_continuous_{pending["index"]:08d}_{pending["session_kind"]}_a{pending["attempt"]}'
                    command=self.command('ppo',Path(pending['source']),Path(pending['tasks']),base,max_act=3,seconds=pending['seconds'],seed=self.config.seed+pending['index'],warm=pending['warm'])
                    base=base.with_name(base.name+'_'+digest(self._signature(command))[:16]);pending['candidate']=str(base.with_suffix('.pt'));self.save()
                    command=self.command('ppo',Path(pending['source']),Path(pending['tasks']),base,max_act=3,seconds=pending['seconds'],seed=self.config.seed+pending['index'],warm=pending['warm'])
                    metrics,record=self.run_job(command,kind='train',budget=pending['seconds'],stage_end=self.stage_deadline(),checkpoint=Path(pending['candidate']))
                    if metrics is None:
                        self.preserve_partial(pending)
                        if record.get('status') in ('insufficient_budget','resource_deadline') or self.remaining_seconds()<30 or time.time()>=self.session_end-30: break
                        pending['attempt']+=1
                        if pending['attempt']>self.config.retries: raise RuntimeError('Bounded training retry exhausted')
                        self.save();continue
                    self.advance_learner(pending,metrics,record)
                capture=Path(pending['candidate']).with_suffix('.continuations.jsonl')
                if self.config.capture_continuations and capture.is_file():
                    pool=sampler.add_learner_continuations(pool,[str(capture)]);self.state.setdefault('learner_continuations',[]).append(str(capture));self.state.setdefault('learner_file_hashes',{})[str(capture)]=sha_file(capture)
                self.state['global_round']+=1;self.state.pop('pending_iteration',None)
                self.state['hourly_anchors'].setdefault(time.strftime('%Y%m%dT%H',time.gmtime()),self.state['current_checkpoint'])
                self.save();self.monitor_learner(panels);retain_checkpoints(self.output,self.state)
            self.state['status']='pilot_complete' if session_seconds else 'allocation_complete'
            self.save()
        finally:
            self.set_accounting(False);self.sleep_control(False);self.save()
