"""Finite independent long-run orchestration; no model or simulator is imported here."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from .campaign_identity import condition_signature, input_paths


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Stage:
    name: str
    sampler_stage: str
    max_act: int
    share: float
    fractions: dict[str, float]


STAGES = (
    Stage('act2_continuations', 'act2', 2, .30, {'act2': .55, 'scratch': .15, 'full3': .20, 'rehearsal': .10}),
    Stage('act2_full', 'act2', 2, .25, {'act2': .25, 'scratch': .45, 'full3': .20, 'rehearsal': .10}),
    Stage('act3', 'act3', 3, .20, {'act3': .60, 'scratch': .30, 'rehearsal': .10}),
    Stage('mixed_full3', 'full3', 3, .25, {'act2': .15, 'act3': .15, 'scratch': .60, 'rehearsal': .10}),
)


@dataclass(frozen=True)
class LongRunConfig:
    checkpoint: str
    backend_build: str
    pool_bank: str
    pool_panels: str
    deadline: float
    duration_hours: float | None = None
    device: str = 'mps'
    workers: int = 8
    threads: int = 1
    nice: int = 10
    episodes: int = 128
    dev_episodes: int = 48
    train_world_start: int = 1000000
    dev_world_start: int = 200000
    horizon: int = 4096
    simulations: int = 200
    batch_size: int = 1024
    chunk_seconds: float = 180.
    evaluation_seconds: float = 240.
    dev_interval: int = 3
    learning_rate: float = 1e-4
    gae_lambda: float = .95
    gamma: float = 1.
    clip_ratio: float = .1
    entropy_coef: float = .01
    progress_floor_weight: float = .05
    terminal_hp_weight: float = 0.
    seed: int = 11
    duty_cycle: float = 1.
    hard_wall_deadline: float | None = None
    protocol: str | None = None
    battery_pause_at: float = 25.
    min_free_gib: float = 10.
    floor_regression: float = 1.
    retention_clear_drop: int = 2
    stall_seconds: float = 240.
    startup_grace_seconds: float = 150.
    retries: int = 1
    prevent_sleep: bool = True
    capture_continuations: bool = False
    gpu_lock_path: str = '/private/tmp/sts_policy_long_gpu.lock'

    def __post_init__(self):
        if self.device not in {'cpu', 'mps'} or not math.isfinite(self.deadline):
            raise ValueError('device/deadline invalid')
        if min(self.workers, self.threads, self.episodes, self.dev_episodes, self.batch_size, self.dev_interval) < 1:
            raise ValueError('counts must be positive')
        if not 0 < self.duty_cycle <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid duty cycle or GAE lambda')
        if not 0 <= self.battery_pause_at <= 100 or self.min_free_gib < 0 or self.retries < 0:
            raise ValueError('invalid resource guard/retries')
        if self.dev_world_start < 200000 or self.train_world_start < 1000000:
            raise ValueError('fresh world ranges must exclude the old benchmark')
        values=(self.learning_rate,self.gamma,self.gae_lambda,self.clip_ratio,self.entropy_coef,self.progress_floor_weight,self.terminal_hp_weight,self.duty_cycle,self.floor_regression,self.chunk_seconds,self.evaluation_seconds,self.stall_seconds,self.startup_grace_seconds)
        if not all(math.isfinite(value) for value in values):raise ValueError('numeric settings must be finite')
        if not 0<self.gamma<=1 or self.learning_rate<=0 or self.floor_regression<0 or self.retention_clear_drop<0:
            raise ValueError('invalid learning/selection settings')
        if min(self.chunk_seconds, self.evaluation_seconds, self.stall_seconds, self.startup_grace_seconds) <= 0:
            raise ValueError('time bounds must be positive')


def goal_signature(config: LongRunConfig, max_act: int, stage: Stage | None = None) -> dict[str, Any]:
    value = {'task_kind': 'run', 'max_act': max_act, 'goal_act_clear': True,
            'gamma': config.gamma, 'reward_mode': 'progress',
            'progress_floor_weight': config.progress_floor_weight,
            'terminal_hp_weight': config.terminal_hp_weight}
    if stage is not None:
        value.update(stage=stage.name, goal_mixture=stage.fractions)
    return value


def optimizer_compatible(previous: dict[str, Any] | None, config: LongRunConfig, max_act: int, stage: Stage | None = None) -> bool:
    # Explicit goal compatibility supplements the legacy trainer's narrower check.
    return previous is not None and previous == goal_signature(config, max_act, stage)


def panel_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    rounds = metrics.get('rounds', [])
    requested = [identity for row in rounds for identity in row.get('task_ids', [])]
    episodes = [episode for row in rounds for episode in row.get('episodes', [])]
    faults = [fault for row in rounds for fault in row.get('faults', [])]
    actual = [row.get('task_id') for row in episodes + faults]
    complete = bool(requested) and Counter(requested) == Counter(actual)
    complete = complete and all(not row.get('unstarted_tasks') and row.get('cohort_completed') is not False for row in rounds)
    complete = complete and not any(row.get('termination_reason') == 'collector_budget' for row in episodes)
    wins = sum(row.get('outcome') in ('act_clear', 'victory', 'battle_victory') for row in episodes)
    floors = [float(row['floor']) for row in episodes if row.get('floor') is not None]
    return {'complete': complete, 'cohort': sorted(requested), 'clears': wins, 'requested': len(requested),
            'fault_ids': sorted(row.get('task_id') for row in faults), 'faults': len(faults),
            'truncated': sum(bool(row.get('truncated')) for row in episodes),
            'floor_mean': sum(floors) / len(floors) if floors else 0.}


def selection_decision(candidate: dict[str, Any], anchor: dict[str, Any],
                       retention: dict[str, Any], anchor_retention: dict[str, Any], *,
                       floor_regression: float = 1., retention_clear_drop: int = 2) -> dict[str, Any]:
    panels = (candidate, anchor, retention, anchor_retention)
    if not all(p['complete'] for p in panels) or candidate['cohort'] != anchor['cohort'] or retention['cohort'] != anchor_retention['cohort']:
        return {'action': 'hold', 'reason': 'incomplete_or_different_cohort'}
    coherent = (candidate['fault_ids'] == anchor['fault_ids'] and retention['fault_ids'] == anchor_retention['fault_ids']
                and candidate['truncated'] == anchor['truncated'] and retention['truncated'] == anchor_retention['truncated'])
    if not coherent:
        return {'action': 'hold', 'reason': 'fault_or_censoring_sets_differ'}
    if retention['clears'] < anchor_retention['clears'] - retention_clear_drop:
        return {'action': 'rollback', 'reason': 'act1_retention_regression'}
    if candidate['clears'] < anchor['clears'] or (candidate['clears'] == anchor['clears'] and
            candidate['floor_mean'] < anchor['floor_mean'] - floor_regression):
        return {'action': 'rollback', 'reason': 'primary_clear_or_floor_regression'}
    if (candidate['clears'], candidate['floor_mean']) > (anchor['clears'], anchor['floor_mean']):
        return {'action': 'promote', 'reason': 'coherent_primary_improvement'}
    return {'action': 'continue', 'reason': 'within_predeclared_regression_bounds'}


def power_state() -> dict[str, Any]:
    if sys.platform != 'darwin':
        return {'on_ac': True, 'battery_percent': None, 'discharging':False, 'status': 'not_macos'}
    try:
        result = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, timeout=3, check=True)
        match = re.search(r'(\d+)%', result.stdout)
        return {'on_ac': 'AC Power' in result.stdout,
                'battery_percent': float(match.group(1)) if match else None,
                'discharging':'discharging' in result.stdout.lower(), 'status': 'verified'}
    except (OSError, subprocess.SubprocessError):
        return {'on_ac': None, 'battery_percent': None, 'discharging':None, 'status': 'unavailable'}


def resource_pause(config: LongRunConfig, output: Path, *, power: dict[str, Any] | None = None,
                   free_bytes: int | None = None, previously_battery_paused: bool = False) -> str | None:
    free = shutil.disk_usage(output).free if free_bytes is None else free_bytes
    if free < config.min_free_gib * 1024**3:
        return 'disk_below_minimum'
    power = power_state() if power is None else power
    if power['on_ac'] is None:
        return 'power_status_unavailable'
    discharging=power.get('discharging',not power['on_ac'])
    battery_low=power['battery_percent'] is not None and power['battery_percent'] <= config.battery_pause_at
    if battery_low and discharging:
        return 'battery_below_threshold'
    if previously_battery_paused and not ((power['on_ac'] and not discharging) or
            (power['battery_percent'] is not None and power['battery_percent'] > config.battery_pause_at)):
        return 'battery_waiting_for_charge'
    return None


class FileLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('a+')
    def __enter__(self):
        try: fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError('Another controller/owned learner holds ' + str(self.stream.name))
        return self
    def __exit__(self, *args):
        # Closing the parent FD preserves an inherited child's flock until its
        # last copy closes; explicit LOCK_UN would release it prematurely.
        self.stream.close()


def process_birth(pid: int) -> str | None:
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], text=True, capture_output=True, timeout=3)
    return ' '.join(result.stdout.split()) or None


def group_details(pgid: int) -> dict[str, dict[str, str]]:
    result=subprocess.run(['ps','-axo','pid=,pgid=,lstart=,command='],capture_output=True,text=True,timeout=3,check=True)
    members={}
    for line in result.stdout.splitlines():
        fields=line.strip().split(None,7)
        if len(fields)>=7 and fields[1]==str(pgid):
            members[fields[0]]={'birth':' '.join(fields[2:7]),'command':fields[7] if len(fields)==8 else ''}
    return members


def group_members(pgid: int) -> dict[str, str]:
    return {pid:row['birth'] for pid,row in group_details(pgid).items()}


def track_owned_members(record: dict[str, Any]) -> None:
    details=group_details(record['pgid'])
    if details.get(str(record['pid']),{}).get('birth')==record.get('process_birth'):
        record.setdefault('owned_members',{}).update({pid:row['birth'] for pid,row in details.items()})
        return
    creation=record.get('group_creation',{})
    # A new-session creator plus bounded birth times and command provenance can
    # recognize an as-yet-unobserved orphan. Never kill a bare reused PGID.
    if not (creation.get('new_session') and creation.get('leader_pid')==record['pid'] and
            creation.get('leader_birth')==record.get('process_birth') and
            creation.get('command_sha256')==digest(record.get('command',[]))):return
    lower=creation['started_at']-1
    upper=creation.get('leader_exit_observed_at',creation['started_at']+creation.get('max_lifetime_seconds',270))+1
    tokens=list(map(str,record.get('command',[])))
    markers=[token for token in tokens if token.endswith('.py')]
    if '-c' in tokens and tokens.index('-c')+1<len(tokens):markers.append(tokens[tokens.index('-c')+1])
    markers+=['multiprocessing.spawn','multiprocessing.resource_tracker','spawn_main']
    for pid,row in details.items():
        try:birth=datetime.strptime(row['birth'],'%a %b %d %H:%M:%S %Y').timestamp()
        except ValueError:continue
        text=' '.join(row['command'].split())
        if lower<=birth<=upper and any(' '.join(marker.split()) in text for marker in markers):
            record.setdefault('owned_members',{})[pid]=row['birth']


def stop_owned_group(record: dict[str, Any], *, grace: float = 5.) -> bool:
    pgid=record.get('pgid')
    if not pgid:return False
    track_owned_members(record)
    recorded=dict(record.get('owned_members',{}))
    if record.get('pid') and record.get('process_birth'):recorded[str(record['pid'])]=record['process_birth']
    current=group_members(pgid)
    if not any(current.get(pid)==birth for pid,birth in recorded.items()):return False
    try:os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError:return True
    until=time.monotonic()+grace
    while time.monotonic()<until:
        if not any(group_members(pgid).get(pid)==birth for pid,birth in recorded.items()):return True
        time.sleep(.1)
    if any(group_members(pgid).get(pid)==birth for pid,birth in recorded.items()):
        try:os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:pass
    return True


def actor_weight_identity(path: Path) -> dict[str, Any]:
    """CPU file inspection only; no model, native simulator or accelerator runs."""
    from .training import load_checkpoint
    payload = load_checkpoint(path)
    identity = hashlib.sha256()
    keys = []
    for key, value in sorted(payload['model_state'].items()):
        if not any('.'+part+'.' in '.'+key for part in ('state_encoder','action_encoder','scorer')):
            continue
        tensor = value.detach().cpu().contiguous()
        identity.update(key.encode()); identity.update(str(tuple(tensor.shape)).encode())
        identity.update(tensor.numpy().tobytes()); keys.append(key)
    if not keys:
        raise ValueError('No candidate actor tensors in checkpoint')
    return {'actor_sha256': identity.hexdigest(), 'actor_tensor_count': len(keys), 'step': payload['step']}


def retain_checkpoints(output: Path, state: dict[str, Any], *, now: float | None = None) -> list[dict[str, Any]]:
    """Delete only owned immutable .pt outputs after a durable SHA/path ledger."""
    directory=output/'checkpoints'
    if directory.is_symlink():raise RuntimeError('Refusing retention through a symlinked checkpoint directory')
    owned=directory.resolve()
    if not owned.is_relative_to(output.resolve()):raise RuntimeError('Checkpoint retention escaped the owned run directory')
    now = time.time() if now is None else now
    protected = {str(Path(v).resolve()) for v in state.get('hourly_anchors', {}).values()}
    for value in [state.get('initial_checkpoint'), state.get('best_checkpoint'), state.get('current_checkpoint'), state.get('last_checkpoint')]:
        if value: protected.add(str(Path(value).resolve()))
    for row in state.get('stages', {}).values():
        if row.get('best_checkpoint'): protected.add(str(Path(row['best_checkpoint']).resolve()))
    pending = state.get('pending_iteration', {})
    for key in ('source', 'candidate'):
        if pending.get(key): protected.add(str(Path(pending[key]).resolve()))
    for record in state.get('jobs', []):
        if record.get('status') == 'running':
            protected.update(str(path.resolve()) for path in input_paths(record['command']).values())
            if record.get('checkpoint'): protected.add(str(Path(record['checkpoint']).resolve()))
    files = sorted((p for p in owned.glob('*.pt') if p.is_file() and not p.is_symlink()), key=lambda p:p.stat().st_mtime, reverse=True)
    protected.update(str(p.resolve()) for p in files[:2])
    deletions = []
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(owned) or str(resolved) in protected:
            continue
        row = {'path': str(resolved), 'sha256': sha_file(path), 'bytes': path.stat().st_size,
               'at': now, 'reason': 'owned_checkpoint_retention', 'status': 'intent_recorded'}
        state.setdefault('deletions', []).append(row)
        atomic_json(output/'journal.json', state)
        if sha_file(path) != row['sha256']:
            raise RuntimeError('Checkpoint changed during retention; refusing deletion')
        path.unlink(); row['status'] = 'deleted'; deletions.append(row)
        atomic_json(output/'journal.json', state)
    return deletions


class LongRunController:
    def __init__(self, root: Path, output: Path, config: LongRunConfig):
        self.root, self.output, self.config = root.resolve(), output.resolve(), config
        self.output.mkdir(parents=True, exist_ok=True)
        self._run_lock=FileLock(self.output/'controller.lock')
        self._run_lock.__enter__()
        self.sleep_assertion=None
        self._publish_live=False
        self._clock_active=False
        self._clock_mark=time.monotonic()
        try:
            self.journal = self.output/'journal.json'
            self.session_end = math.inf
            self.session_kind = 'main'
            self.backend = Path(config.backend_build).resolve()
            manifest = self.backend/'build_manifest.json'
            native = json.loads(manifest.read_text()); module = native.get('module_file', '')
            if not module or Path(module).name != module or sha_file(self.backend/module) != native.get('module_sha256'):
                raise ValueError('Native binary/manifest mismatch')
            self.backend_identity = {'build_dir': str(self.backend), 'manifest_sha256': sha_file(manifest), 'provenance': native}
            paths = [*self.root.glob('src/sts_env/**/*.py'), *self.root.glob('src/sts_policy/*.py'),
                     self.root/'scripts/train_nonllm.py', self.root/'scripts/run_long_curriculum.py']
            self.sources = {str(p.relative_to(self.root)): sha_file(p) for p in sorted(paths) if p.is_file()}
            scientific=asdict(config);scientific.pop('deadline')
            identity = {'config': scientific, 'backend': self.backend_identity, 'sources': self.sources,
                        'initial_checkpoint_sha256': sha_file(config.checkpoint),
                        'protocol_sha256': sha_file(config.protocol) if config.protocol else None}
            if self.journal.exists():
                self.state = json.loads(self.journal.read_text())
                if self.state['identity'] != identity:
                    raise ValueError('Run identity changed; preserve artifacts and choose a new output directory')
            else:
                now = time.time()
                self.state = {'version': 'sts_long_run_v1', 'identity': identity, 'started': now,
                    'deadline': config.deadline, 'status': 'prepared', 'jobs': [], 'events': [], 'stages': {},
                    'initial_checkpoint': str(Path(config.checkpoint).resolve()),
                    'current_checkpoint': str(Path(config.checkpoint).resolve()),
                    'last_checkpoint': str(Path(config.checkpoint).resolve()),
                    'best_checkpoint': str(Path(config.checkpoint).resolve()), 'global_round': 0, 'hourly_anchors': {}}
                initial_seconds=config.duration_hours*3600 if config.duration_hours is not None else max(0.,config.deadline-now)
                cumulative=0.;boundaries={}
                for stage in STAGES:
                    cumulative+=stage.share;boundaries[stage.name]=initial_seconds*cumulative
                self.state['allocation']={'initial_seconds':initial_seconds,'authorized_seconds':initial_seconds,'used_seconds':0.,
                    'stage_boundaries':boundaries,'extensions':[],'accounting':'monotonic_active_only; manual/resource/offline pauses excluded'}
                self.save()
            if 'allocation' not in self.state:raise ValueError('Legacy wall-only journal cannot be silently converted to an active allocation')
            self.state['controller']={'pid':os.getpid(),'process_birth':process_birth(os.getpid())}
            self.save()

        except BaseException:
            self._run_lock.__exit__(None,None,None)
            raise

    def account_time(self) -> None:
        now=time.monotonic()
        if self._clock_active and hasattr(self,'state') and 'allocation' in self.state:
            self.state['allocation']['used_seconds']+=max(0.,now-self._clock_mark)
        self._clock_mark=now

    def set_accounting(self, active: bool) -> None:
        self.account_time();self._clock_active=active

    def remaining_seconds(self) -> float:
        self.account_time()
        allocation=self.state['allocation']
        return max(0.,allocation['authorized_seconds']-allocation['used_seconds'])

    def stage_deadline(self, stage: Stage | str | None = None) -> float:
        remaining=self.remaining_seconds()
        if stage is not None:
            name=stage.name if isinstance(stage,Stage) else stage
            remaining=min(remaining,max(0.,self.state['allocation']['stage_boundaries'][name]-self.state['allocation']['used_seconds']))
        deadline=min(time.time()+remaining,self.session_end)
        if self.config.hard_wall_deadline is not None:deadline=min(deadline,self.config.hard_wall_deadline)
        return deadline

    def extend_hours(self, hours: float) -> None:
        if not math.isfinite(hours) or hours<=0:raise ValueError('Explicit extension must be finite and positive')
        self.account_time();extra=hours*3600;allocation=self.state['allocation']
        allocation['authorized_seconds']+=extra
        allocation['stage_boundaries']['mixed_full3']+=extra
        row={'at':time.time(),'hours':hours,'added_seconds':extra,'used_seconds':allocation['used_seconds'],
             'authorized_seconds':allocation['authorized_seconds'],'stage':'mixed_full3','authority':'explicit_operator_extension'}
        allocation['extensions'].append(row);self.state['events'].append({'type':'allocation_extension',**row});self.save()

    def save(self):
        self.account_time()
        self.state['deadline']=self.stage_deadline()
        self.state['heartbeat'] = time.time()
        atomic_json(self.journal, self.state)
        atomic_json(self.output/'config.json', self.state['identity'])
        atomic_json(self.output/'events.json', self.state['events'])
        atomic_json(self.output/'progress.json', {key:self.state.get(key) for key in ('status','heartbeat','deadline','current_checkpoint','best_checkpoint','last_checkpoint','global_round','stages','active_job','pause_reason','controller','allocation')})
        rows = ['# Long curriculum progress', '', 'Status: ' + self.state['status'],
                'Current active-leg deadline UTC: ' + datetime.fromtimestamp(self.state['deadline'], timezone.utc).isoformat(),
                'Active remaining seconds: '+str(self.remaining_seconds()),
                'Current: ' + self.state['current_checkpoint'], 'Best: ' + self.state['best_checkpoint'],
                'Last: ' + self.state['last_checkpoint'], '',
                'Faults remain separate from ordinary game defeats; mixed rehearsal clears are not full-run competence.']
        for stage, progress in self.state['stages'].items():
            rows.append(f"- {stage}: rounds={progress.get('round',0)}, updates={progress.get('updates',0)}, transitions={progress.get('transitions',0)}")
        rows += ['', *[json.dumps(row,sort_keys=True) for row in self.state['events'][-12:]]]
        temporary=self.output/'summary.md.tmp'; temporary.write_text('\n'.join(rows)+'\n'); os.replace(temporary,self.output/'summary.md')
        meaningful=digest({key:self.state.get(key) for key in ('status','current_checkpoint','best_checkpoint','last_checkpoint','global_round','stages','events')})
        if self._publish_live and getattr(self,'_live_doc_identity',None)!=meaningful:
            self._live_doc_identity=meaningful
            lines=['# Long curriculum live progress','', 'Status: '+self.state['status'],
                   'Machine progress: '+str(self.output/'progress.json'), 'Chronological events: '+str(self.output/'events.json'),
                   'Current: '+self.state['current_checkpoint'], 'Best: '+self.state['best_checkpoint'], 'Last: '+self.state['last_checkpoint'],'',
                   '| Stage | Completed rounds | Accepted optimizer updates | Valid transitions | Primary clears/requested | Mean floor |',
                   '| --- | ---: | ---: | ---: | --- | ---: |']
            for name,stage in self.state['stages'].items():
                quality=stage.get('quality',{})
                lines.append(f"| {name} | {stage.get('round',0)} | {stage.get('updates',0)} | {stage.get('transitions',0)} | {quality.get('clears','pending')}/{quality.get('requested','pending')} | {quality.get('floor_mean',0):.2f} |")
            lines+=['','Latest events first:',*[json.dumps(event,sort_keys=True) for event in reversed(self.state['events'][-30:])]]
            path=self.root/'docs/long_curriculum_live.md';path.parent.mkdir(exist_ok=True)
            temporary=path.with_suffix('.md.tmp');temporary.write_text('\n'.join(lines)+'\n');os.replace(temporary,path)

    def verify_pins(self):
        for relative, expected in self.sources.items():
            if sha_file(self.root/relative) != expected:
                raise RuntimeError('Pinned source changed: ' + relative)
        if self.config.protocol and sha_file(self.config.protocol)!=self.state['identity']['protocol_sha256']:
            raise RuntimeError('Pinned protocol changed')
        for path,expected in self.state.get('dev_file_hashes',{}).items():
            if sha_file(path)!=expected:raise RuntimeError('Fixed development task file changed')
        def check_refs(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value and sha_file(value['path'])!=value['sha256']:raise RuntimeError('Pinned continuation source changed')
                for item in value.values():check_refs(item)
            elif isinstance(value,list):
                for item in value:check_refs(item)
        check_refs(self.state.get('native_pool_identity',{}))
        for path,expected in self.state.get('learner_file_hashes',{}).items():
            if sha_file(path)!=expected:raise RuntimeError('Saved learner continuation changed')
        if sha_file(self.config.checkpoint)!=self.state['identity']['initial_checkpoint_sha256']:raise RuntimeError('Initial checkpoint changed')
        recorded={path:sha for job in self.state['jobs'] for path,sha in job.get('artifacts',{}).items() if path.endswith('.pt')}
        recorded.update({event['path']:event['sha256'] for event in self.state['events'] if event.get('type')=='partial_checkpoint_preserved'})
        for path in [self.state.get('current_checkpoint'),self.state.get('best_checkpoint'),self.state.get('last_checkpoint'),*[row.get('best_checkpoint') for row in self.state['stages'].values()]]:
            if path and str(Path(path).resolve())!=self.state['initial_checkpoint']:
                if path not in recorded or sha_file(path)!=recorded[path]:raise RuntimeError('Published checkpoint identity changed')

        manifest = self.backend/'build_manifest.json'
        if sha_file(manifest) != self.backend_identity['manifest_sha256'] or sha_file(self.backend/self.backend_identity['provenance']['module_file']) != self.backend_identity['provenance']['module_sha256']:
            raise RuntimeError('Pinned backend changed')

    def sleep_control(self, enable: bool):
        if not enable:
            if self.sleep_assertion and self.sleep_assertion.poll() is None:
                self.sleep_assertion.terminate();self.sleep_assertion.wait(timeout=3)
            self.sleep_assertion=None
        elif self.config.prevent_sleep and sys.platform=='darwin' and self.sleep_assertion is None:
            duration=max(1,math.ceil(self.stage_deadline()-time.time()))
            self.sleep_assertion=subprocess.Popen(['caffeinate','-is','-t',str(duration),'-w',str(os.getpid())])

    def close(self):
        if self._run_lock is None:return
        self.set_accounting(False)
        if hasattr(self,'state'):self.save()
        self.sleep_control(False)
        if self._run_lock is not None:
            self._run_lock.__exit__(None,None,None);self._run_lock=None

    def pause_until_safe(self):
        while self.remaining_seconds()>0 and time.time()<self.session_end and (self.config.hard_wall_deadline is None or time.time()<self.config.hard_wall_deadline):
            retain_checkpoints(self.output,self.state)
            reason = resource_pause(self.config,self.output,previously_battery_paused=self.state.get('battery_paused',False))
            if reason is None:
                self.state['battery_paused']=False
                if self.state.get('pause_reason'):
                    self.state['events'].append({'at':time.time(),'type':'resource_resume','previous_reason':self.state.pop('pause_reason')})
                    self.state['status']='running';self.save()
                self.set_accounting(True)
                self.sleep_control(True)
                return True
            self.set_accounting(False)
            self.sleep_control(False)
            if reason.startswith('battery'):
                self.state['battery_paused']=True
            if self.state.get('pause_reason') != reason:
                self.state['events'].append({'at':time.time(),'type':'resource_pause','reason':reason})
            self.state.update(status='paused_resource',pause_reason=reason); self.save()
            time.sleep(min(5.,max(.01,self.session_end-time.time())))
        return False

    def _signature(self, command):
        return {'condition': condition_signature(command), 'backend': self.backend_identity, 'sources': self.sources}

    def run_job(self, command: list[Any], *, kind: str, budget: float, stage_end: float,
                checkpoint: Path | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        self.verify_pins()
        signature = self._signature(command)
        metrics = Path(command[command.index('--metrics')+1])
        old = next((r for r in reversed(self.state['jobs']) if r['signature']==signature and r['status']=='complete'),None)
        if old:
            artifacts=old['artifacts']
            if all(Path(path).is_file() and sha_file(path)==sha for path,sha in artifacts.items()):
                return json.loads(metrics.read_text()),old
            raise RuntimeError('Completed job artifacts changed; refusing overwrite/relaunch')
        if metrics.exists() or (checkpoint and checkpoint.exists()):
            raise RuntimeError('Unclassified existing artifacts; use a distinct attempt name')
        def current_deadline():
            if getattr(self,'current_stage',None):return self.stage_deadline(self.current_stage)
            return min(self.stage_deadline(),stage_end,self.session_end)
        deadline=current_deadline()
        if time.time()+2 >= deadline:
            return None,{'status':'insufficient_budget'}
        if not self.pause_until_safe():return None,{'status':'resource_deadline'}
        deadline=current_deadline()
        if time.time()+2>=deadline:return None,{'status':'insufficient_budget'}
        record={'signature':signature,'command':list(map(str,command)),'kind':kind,'metrics':str(metrics),
                'checkpoint':str(checkpoint) if checkpoint else None,'started':time.time(),'status':'launching', 'session_kind':self.session_kind}
        self.state['jobs'].append(record);self.state['active_job']=len(self.state['jobs'])-1;self.save()
        metrics.parent.mkdir(parents=True,exist_ok=True)
        if checkpoint:checkpoint.parent.mkdir(parents=True,exist_ok=True)
        log=metrics.with_suffix('.log')
        environment=os.environ.copy();environment['STS_LIGHTSPEED_BUILD']=str(self.backend)
        record['effective_environment']={'STS_LIGHTSPEED_BUILD':str(self.backend)}
        gpu = FileLock(Path(self.config.gpu_lock_path)) if kind=='train' and self.config.device=='mps' else None
        if gpu:
            try:gpu.__enter__()
            except RuntimeError:
                record.update(status='blocked_gpu_lock',finished=time.time());self.state.pop('active_job',None);self.save()
                return None,record
        process=None
        try:
            with log.open('a') as stream:
                process=subprocess.Popen(list(map(str,command)),cwd=self.root,env=environment,stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True,
                    pass_fds=(gpu.stream.fileno(),) if gpu else ())
                record.update(pid=process.pid,pgid=process.pid,process_birth=process_birth(process.pid),status='running')
                record['group_creation']={'new_session':True,'leader_pid':process.pid,'leader_birth':record['process_birth'],
                    'started_at':record['started'],'command_sha256':digest(record['command']),'max_lifetime_seconds':budget+90}
                self.save(); began=time.monotonic();last_progress=began;last_signature=None
                while True:
                    track_owned_members(record)
                    if process.poll() is not None:
                        record['group_creation']['leader_exit_observed_at']=time.time();break
                    now=time.monotonic()
                    progress=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in [metrics,checkpoint] if p and p.is_file())
                    if progress and progress!=last_signature:last_progress=now;last_signature=progress
                    reason=resource_pause(self.config,self.output)
                    if reason and reason.startswith('battery'):self.state['battery_paused']=True
                    stop= ('resource:'+reason if reason else 'active_budget_or_session_limit' if time.time()>=deadline or self.remaining_seconds()<=5 else
                          'job_budget' if now-began>=budget+90 else
                          'child_stall' if now-began>self.config.startup_grace_seconds and now-last_progress>self.config.stall_seconds else None)
                    if stop:
                        record['stop_reason']=stop
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=min(5.,max(.01,self.stage_deadline()-time.time())))
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                        break
                    self.save();time.sleep(min(2.,max(.01,deadline-time.time())))
                record.update(exit_code=process.returncode,finished=time.time())
        finally:
            if process:
                stop_owned_group(record)
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            if gpu:gpu.__exit__(None,None,None)
        self.state.pop('active_job',None)
        capture=checkpoint.with_suffix('.continuations.jsonl') if checkpoint else None
        artifacts=[p for p in (metrics,checkpoint,log,capture) if p and p.is_file()]
        record['artifacts']={str(p):sha_file(p) for p in artifacts}
        record['status']='complete' if record.get('exit_code')==0 and metrics.is_file() else 'interrupted' if record.get('stop_reason') else 'failed'
        self.save()
        return (json.loads(metrics.read_text()) if record['status']=='complete' else None),record

    def preserve_partial(self,pending):
        path=Path(pending.get('candidate',''))
        if not path.is_file():return False
        try:evidence=actor_weight_identity(path)
        except (OSError,ValueError,RuntimeError):return False
        self.state['last_checkpoint']=str(path)
        pending.update(source=str(path),warm=True)
        self.state['events'].append({'at':time.time(),'type':'partial_checkpoint_preserved','path':str(path),'sha256':sha_file(path),'actor':evidence})
        return True

    def recover(self):
        for record in self.state['jobs']:
            if record['status'] in ('running','launching'):
                stopped=stop_owned_group(record)
                record.update(status='recovered_interruption',recovered_at=time.time(),owned_process_stopped=stopped)
                record['artifacts']={str(p):sha_file(p) for p in [Path(record['metrics']),Path(record['checkpoint']) if record.get('checkpoint') else None] if p and p.is_file()}
        pending=self.state.get('pending_iteration')
        if pending and not pending.get('training_committed'):
            self.preserve_partial(pending)
            if pending.get('candidate'):
                pending['attempt']+=1
                pending['recovery_failures']=pending.get('recovery_failures',0)+1
                if pending['recovery_failures']>self.config.retries+1:raise RuntimeError('Bounded interruption recovery exhausted')
        self.save()

    def command(self, mode: str, source: Path, tasks: Path, destination: Path, *, max_act: int,
                seconds: float, seed: int, warm: bool = False, horizon: int | None = None) -> list[Any]:
        command=[sys.executable,self.root/'scripts/train_nonllm.py',mode,'--checkpoint',source,
            '--output',destination.with_suffix('.pt'),'--metrics',destination.with_suffix('.metrics.json'),
            '--tasks',tasks,'--preserve-task-prefixes','--task-kind','run','--split','train' if mode=='ppo' else 'dev',
            '--max-act',max_act,'--horizon',horizon or self.config.horizon,'--episodes',self.config.episodes if mode=='ppo' else self.config.dev_episodes,
            '--seed',seed,'--simulations',self.config.simulations,'--seconds',seconds,
            '--device',self.config.device if mode=='ppo' else 'cpu','--inference-device','cpu',
            '--workers',self.config.workers,'--threads',self.config.threads,'--fast-observations','--fast-replay','--deferred-bootstrap',
            '--gamma',self.config.gamma,'--gae-lambda',self.config.gae_lambda,'--clip-ratio',self.config.clip_ratio,
            '--entropy-coef',self.config.entropy_coef,'--learning-rate',self.config.learning_rate,
            '--reward-mode','progress','--progress-floor-weight',self.config.progress_floor_weight,
            '--terminal-hp-weight',self.config.terminal_hp_weight,'--goal-act-clear','--kl-guard-before-update']
        if mode=='ppo':
            command += ['--updates',1,'--batch-size',self.config.batch_size]
            if warm:command.append('--warm-optimizer')
            if self.config.capture_continuations:command += ['--continuation-output',destination.with_suffix('.continuations.jsonl')]
        if self.config.nice:
            command=[shutil.which('nice') or '/usr/bin/nice','-n',self.config.nice,*command]
        return command

    def evaluate(self, label: str, source: Path, tasks: Path, max_act: int, stage_end: float):
        destination=self.output/'metrics'/label
        command=self.command('evaluate',source,tasks,destination,max_act=max_act,seconds=self.config.evaluation_seconds,seed=777,horizon=json.loads(tasks.open().readline())['horizon'])
        unique=digest(self._signature(command))[:16]
        destination=destination.with_name(destination.name+'_'+unique)
        command=self.command('evaluate',source,tasks,destination,max_act=max_act,seconds=self.config.evaluation_seconds,seed=777,horizon=json.loads(tasks.open().readline())['horizon'])
        existing=[r for r in self.state['jobs'] if r.get('kind')=='eval' and r.get('metrics','').startswith(str(destination))]
        complete=next((r for r in reversed(existing) if r.get('status')=='complete'),None)
        if complete:
            command=complete['command']
        else:
            attempts=len(existing)
            if attempts>self.config.retries:return None
            if attempts:
                destination=destination.with_name(destination.name+f'_retry{attempts}')
                command=self.command('evaluate',source,tasks,destination,max_act=max_act,seconds=self.config.evaluation_seconds,seed=777,horizon=json.loads(tasks.open().readline())['horizon'])
        metrics,record=self.run_job(command,kind='eval',budget=self.config.evaluation_seconds,stage_end=stage_end)
        if metrics is None:return None
        destination=Path(record['metrics']).with_suffix('').with_suffix('')
        from .curriculum import TaskSpec
        from .rollout_training import task_identity
        expected=sorted(task_identity(TaskSpec.from_dict(json.loads(line))) for line in tasks.read_text().splitlines() if line)
        quality=panel_quality(metrics)
        environment=metrics.get('training_environment',{})
        cfg=metrics.get('config',{})
        if quality['cohort']!=expected or str(Path(cfg.get('checkpoint','')).resolve())!=str(source.resolve()) or cfg.get('split')!='dev' or cfg.get('task_kind')!='run' or cfg.get('seed')!=777 or environment.get('native_module_sha256')!=self.backend_identity['provenance']['module_sha256']:
            raise RuntimeError('Evaluation does not match fixed tasks, checkpoint or backend')
        quality.update(metrics=str(destination.with_suffix('.metrics.json')),metrics_sha256=sha_file(destination.with_suffix('.metrics.json')))
        return quality

    def protocol_inputs(self):
        from .curriculum import TaskSpec, world_split
        from .rollout_training import task_identity
        protocol=json.loads(Path(self.config.protocol).read_text())
        if protocol['checkpoint']['path'] != str(Path(self.config.checkpoint).resolve()) or protocol['checkpoint']['sha256'] != sha_file(self.config.checkpoint):
            raise ValueError('Protocol anchor differs from the initial checkpoint')
        panels,anchors={},{}
        from sts_env.provenance import interface_identity
        interface=interface_identity()['python_interface_sha256']
        for label,entry in protocol['panels'].items():
            if label not in ('dev_act1','dev_act2','dev_act3'):continue
            path=Path(entry['path']);rows=[TaskSpec.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]
            if sha_file(path)!=entry['sha256'] or len(rows)!=self.config.dev_episodes or any(world_split(t.seed)!='dev' or t.seed<200000 or t.kind!='run' or t.max_act!=entry['max_act'] or t.horizon!=entry['horizon'] for t in rows):
                raise ValueError('Protocol development tasks changed')
            identities=sorted(task_identity(t) for t in rows)
            if identities!=sorted(entry['task_ids']):raise ValueError('Protocol task IDs differ')
            panels[label]=path
            if label not in protocol.get('anchors',{}):continue
            anchor=protocol['anchors'][label];metrics_path=Path(anchor['metrics']);metrics=json.loads(metrics_path.read_text());quality=panel_quality(metrics)
            config=metrics['config'];environment=metrics.get('training_environment',{})
            valid=(sha_file(metrics_path)==anchor['metrics_sha256'] and quality['complete'] and quality['cohort']==identities and
                   anchor['checkpoint_sha256']==sha_file(self.config.checkpoint) and str(Path(config['checkpoint']).resolve())==str(Path(self.config.checkpoint).resolve()) and
                   condition_signature(anchor['command'])==anchor['condition_signature'] and config['split']=='dev' and config['task_kind']=='run' and
                   config['seed']==777 and config['max_act']==entry['max_act'] and config['horizon']==entry['horizon'] and
                   environment.get('native_module_sha256')==self.backend_identity['provenance']['module_sha256'] and
                   environment.get('python_interface_sha256')==interface)
            if not valid:raise ValueError('Protocol anchor metrics are stale, incomplete or from another condition')
            quality.update(metrics=str(metrics_path),metrics_sha256=sha_file(metrics_path));anchors[label]=quality
        if set(panels)!={'dev_act1','dev_act2','dev_act3'}:raise ValueError('Protocol needs all three development panels')
        return panels,anchors

    def execute(self, *, session_seconds: float | None = None, session_kind: str = 'main'):
        from . import long_run_tasks as sampler
        self._publish_live=True
        self.session_kind=session_kind
        self.session_end=time.time()+session_seconds if session_seconds else math.inf
        self.set_accounting(True)
        with nullcontext():
            self.recover()
            pool=sampler.load_continuations(bank=Path(self.config.pool_bank),metadata_panels=Path(self.config.pool_panels))
            if self.state.get('native_pool_identity',pool.source_identity)!=pool.source_identity:raise RuntimeError('Native continuation pool changed')
            self.state['native_pool_identity']=pool.source_identity;self.save()
            if self.state.get('learner_continuations'):pool=sampler.add_learner_continuations(pool,self.state['learner_continuations'])
            if self.config.protocol:
                panels,provided_anchors=self.protocol_inputs()
            else:
                result=sampler.fixed_dev_panels(self.output/'panels',world_start=self.config.dev_world_start,
                    episodes=self.config.dev_episodes,horizon=self.config.horizon,simulations=self.config.simulations,seed=777)
                paths=result.get('paths',result)
                panels={key:Path(value['path'] if isinstance(value,dict) else value) for key,value in paths.items() if key.startswith('dev_act')}
                provided_anchors={}
            if set(panels)!= {'dev_act1','dev_act2','dev_act3'}:raise ValueError('fixed development panel API incomplete')
            panel_hashes={key:sha_file(path) for key,path in panels.items()}
            if self.state.get('dev_panel_hashes',panel_hashes)!=panel_hashes:raise RuntimeError('Fixed development panels changed')
            self.state['dev_panel_hashes']=panel_hashes
            self.state['dev_file_hashes']={str(path):panel_hashes[key] for key,path in panels.items()}
            if provided_anchors:self.state.setdefault('retention_reference',provided_anchors['dev_act1'])
            leaders=self.state.setdefault('goal_leaders',{})
            for label,quality in provided_anchors.items():leaders.setdefault(label,{'checkpoint':self.state['initial_checkpoint'],'quality':quality,'goal':None})
            self.save()
            try:
                for stage in STAGES:
                    self.current_stage=stage.name
                    stage_end=self.stage_deadline(stage)
                    if time.time()>=self.session_end:break
                    if self.remaining_seconds()<=0:break
                    if self.state['allocation']['used_seconds']>=self.state['allocation']['stage_boundaries'][stage.name]:continue
                    self.state['status']='running';self.save()
                    progress=self.state['stages'].setdefault(stage.name,{'round':0,'updates':0,'transitions':0})
                    anchor=Path(progress.get('best_checkpoint',self.state['goal_leaders'].get(f'dev_act{stage.max_act}',{}).get('checkpoint',self.state['best_checkpoint'])))
                    if 'quality' not in progress:
                        transfer=Path(self.state['current_checkpoint'])
                        leader=self.state['goal_leaders'].get(f'dev_act{stage.max_act}')
                        if transfer!=anchor and leader:
                            transfer_quality=self.evaluate(stage.name+'_transfer',transfer,panels[f'dev_act{stage.max_act}'],stage.max_act,stage_end)
                            transfer_retention=self.evaluate(stage.name+'_transfer_retention',transfer,panels['dev_act1'],1,stage_end)
                            decision=selection_decision(transfer_quality,leader['quality'],transfer_retention,self.state['retention_reference'],floor_regression=self.config.floor_regression,retention_clear_drop=self.config.retention_clear_drop) if transfer_quality and transfer_retention else {'action':'hold','reason':'incomplete_transfer_evidence'}
                            self.state['events'].append({'at':time.time(),'type':'new_goal_transfer','stage':stage.name,'decision':decision})
                            if decision['action']=='promote':
                                leader={'checkpoint':str(transfer),'quality':transfer_quality,'goal':self.state.get('current_goal')};self.state['goal_leaders'][f'dev_act{stage.max_act}']=leader;anchor=transfer
                            elif decision['action']=='continue':progress['transfer_parent']=str(transfer)
                        self.state['current_checkpoint']=str(progress.get('transfer_parent',anchor));self.state['current_goal']=leader.get('goal') if leader and str(anchor)==leader['checkpoint'] else None

                    if 'quality' not in progress:
                        quality=provided_anchors.get(f'dev_act{stage.max_act}') if str(anchor.resolve())==self.state['initial_checkpoint'] else None
                        retention=self.state.get('retention_reference')
                        quality=quality or self.state['goal_leaders'].get(f'dev_act{stage.max_act}',{}).get('quality')
                        quality=quality or self.evaluate(stage.name+'_anchor',anchor,panels[f'dev_act{stage.max_act}'],stage.max_act,stage_end)
                        retention=retention or self.evaluate(stage.name+'_anchor_retention',anchor,panels['dev_act1'],1,stage_end)
                        if not quality or not retention or not quality['complete'] or not retention['complete']:
                            raise RuntimeError('Incomplete initial development anchor; no training is authorized against an unknown cohort')
                        progress.update(quality=quality,retention=retention,best_checkpoint=str(anchor),best_goal=self.state.get('current_goal'));self.save()
                    if progress.pop('needs_session_eval',False):
                        source=Path(self.state['current_checkpoint'])
                        quality=self.evaluate(stage.name+'_session_resume_'+str(self.state['global_round']),source,panels[f'dev_act{stage.max_act}'],stage.max_act,stage_end)
                        retention=self.evaluate(stage.name+'_session_resume_retention_'+str(self.state['global_round']),source,panels['dev_act1'],1,stage_end)
                        decision=selection_decision(quality,progress['quality'],retention,progress['retention'],floor_regression=self.config.floor_regression,retention_clear_drop=self.config.retention_clear_drop) if quality and retention else {'action':'hold','reason':'incomplete_session_evaluation'}
                        self.state['events'].append({'at':time.time(),'type':'session_resume_evaluation','decision':decision})
                        if decision['action']=='promote':progress.update(best_checkpoint=str(source),quality=quality,best_goal=self.state['current_goal'])
                        elif decision['action'] in ('rollback','hold'):self.state['current_checkpoint']=progress['best_checkpoint'];self.state['current_goal']=progress.get('best_goal')
                        self.save()
                    while time.time()<self.stage_deadline(stage)-30 and self.pause_until_safe():
                        stage_end=self.stage_deadline(stage)
                        pending=self.state.get('pending_iteration')
                        if not pending or pending['stage']!=stage.name:
                            self.verify_pins()
                            index=self.state['global_round'];source=Path(self.state['current_checkpoint'])
                            task_path=self.output/'tasks'/f'{stage.name}_{index:08d}.jsonl'
                            manifest=task_path.with_suffix(task_path.suffix+'.manifest.json')
                            if task_path.is_file() and manifest.is_file():
                                metadata=json.loads(manifest.read_text())
                                if metadata['task_file_sha256']!=sha_file(task_path):raise RuntimeError('Saved task batch changed')
                            else:
                                batch=sampler.sample_training_tasks(pool,stage=stage.sampler_stage,stream=stage.name,round_index=index,
                                    seed=self.config.seed,episodes=self.config.episodes,world_start=self.config.train_world_start,
                                    horizon=self.config.horizon,simulations=self.config.simulations,fractions=stage.fractions)
                                sampler.write_task_batch(batch,task_path);metadata=batch.get('metadata',{})
                            pending={'stage':stage.name,'index':index,'source':str(source),'tasks':str(task_path),'metadata':metadata,
                                'task_sha256':sha_file(task_path),'manifest_sha256':sha_file(manifest),'attempt':0,'session_kind':self.session_kind,'warm':optimizer_compatible(self.state.get('current_goal'),self.config,stage.max_act,stage),
                                'seconds':min(self.config.chunk_seconds,max(1.,min(stage_end,self.session_end)-time.time()-5))}
                            self.state['pending_iteration']=pending;self.save()
                        source=Path(pending['source']);task_path=Path(pending['tasks']);seconds=pending['seconds']
                        if sha_file(task_path)!=pending['task_sha256'] or sha_file(task_path.with_suffix(task_path.suffix+'.manifest.json'))!=pending['manifest_sha256']:raise RuntimeError('Pending immutable task batch changed')
                        if not pending.get('training_committed'):
                            base=self.output/'checkpoints'/f'{stage.name}_{pending["index"]:08d}_{pending["session_kind"]}_a{pending["attempt"]}'
                            command=self.command('ppo',source,task_path,base,max_act=stage.max_act,seconds=seconds,seed=self.config.seed+pending['index'],warm=pending['warm'])
                            unique=digest(self._signature(command))[:16];base=base.with_name(base.name+'_'+unique)
                            command=self.command('ppo',source,task_path,base,max_act=stage.max_act,seconds=seconds,seed=self.config.seed+pending['index'],warm=pending['warm'])
                            candidate=base.with_suffix('.pt');pending['candidate']=str(candidate);self.save()
                            metrics,record=self.run_job(command,kind='train',budget=seconds,stage_end=stage_end,checkpoint=candidate)
                            if metrics is None:
                                if record.get('status') in ('insufficient_budget','resource_deadline'):break
                                pending['attempt']+=1;self.save()
                                if record.get('stop_reason','').startswith('resource:'):
                                    self.preserve_partial(pending);self.save();continue
                                if record.get('stop_reason')=='active_budget_or_session_limit':
                                    self.preserve_partial(pending);self.save();break
                                if pending['attempt']>self.config.retries:raise RuntimeError('Bounded child retries exhausted; preserve failed artifacts')
                                continue
                            updates=sum(row.get('update',{}).get('steps',0) for row in metrics.get('rounds',[]))
                            valid=sum(row.get('transitions',0) for row in metrics.get('rounds',[]))
                            before,after=actor_weight_identity(source),actor_weight_identity(candidate)
                            if not updates or not valid or before['actor_sha256']==after['actor_sha256']:
                                record['status']='no_meaningful_actor_update';self.save()
                                raise RuntimeError('Completed child made no actor update; investigate throughput/config rather than endlessly relaunching')
                            record['actor_evidence']={'before':before,'after':after,'updates':updates,'transitions':valid}
                            record['optimizer_goal']={'restored_requested':pending['warm'],'target':goal_signature(self.config,stage.max_act,stage),
                                'source_task_file_sha256':sha_file(task_path),'critic_recalibration_required':not pending['warm']}
                            # Training clears are stratified by actual TaskSpec ID, never pooled as Act2/full3 mastery.
                            families={row['task_id']:row for row in pending['metadata'].get('tasks',[])}
                            strata={}
                            def family_counts(identity):
                                if identity not in families:raise RuntimeError('Training task ID is absent from immutable family manifest')
                                family=families[identity]
                                key=json.dumps({k:family.get(k) for k in ('start_source','start_act','goal_act','horizon')},sort_keys=True)
                                return strata.setdefault(key,{'episodes':0,'clears':0,'defeats':0,'faults':0,'truncated':0,'requested':0,'unstarted':0})
                            for row in metrics.get('rounds',[]):
                                classified={episode['task_id'] for episode in row.get('episodes',[])+row.get('faults',[])}
                                for identity in row.get('task_ids',[]):
                                    counts=family_counts(identity);counts['requested']+=1;counts['unstarted']+=int(identity not in classified)
                                for episode in row.get('episodes',[]):
                                    counts=family_counts(episode['task_id']);counts['episodes']+=1
                                    counts['clears']+=int(episode.get('outcome') in ('act_clear','victory'))
                                    counts['defeats']+=int(episode.get('outcome') in ('defeat','battle_defeat'))
                                    counts['truncated']+=int(bool(episode.get('truncated')))
                                for fault in row.get('faults',[]):family_counts(fault['task_id'])['faults']+=1
                            record['family_strata']=strata
                            total_faults=sum(count['faults'] for count in strata.values());requested=sum(count['requested'] for count in strata.values())
                            excessive=total_faults>max(3,.05*requested)
                            self.state['consecutive_excessive_fault_chunks']=self.state.get('consecutive_excessive_fault_chunks',0)+1 if excessive else 0
                            if excessive:self.state['events'].append({'at':time.time(),'type':'excessive_training_fault_warning','faults':total_faults,'requested':requested})
                            if self.state['consecutive_excessive_fault_chunks']>=2:
                                self.state['last_checkpoint']=str(candidate);self.save()
                                raise RuntimeError('Persistent excessive training faults; pause for investigation rather than waste allocation')
                            pending.update(training_committed=True,updates=updates,transitions=valid,record_index=len(self.state['jobs'])-1,
                                evaluate_due=(progress['round']+1)%self.config.dev_interval==0 or time.time()+self.config.chunk_seconds>=stage_end)
                            self.state.update(current_checkpoint=str(candidate),last_checkpoint=str(candidate),current_goal=goal_signature(self.config,stage.max_act,stage))
                            self.save()
                        candidate=Path(pending['candidate']);record=self.state['jobs'][pending['record_index']]
                        if pending['evaluate_due']:
                            quality=self.evaluate(stage.name+f'_{pending["index"]:08d}',candidate,panels[f'dev_act{stage.max_act}'],stage.max_act,stage_end)
                            retention=self.evaluate(stage.name+f'_{pending["index"]:08d}_retention',candidate,panels['dev_act1'],1,stage_end)
                            decision=selection_decision(quality,progress['quality'],retention,progress['retention'],
                                floor_regression=self.config.floor_regression,retention_clear_drop=self.config.retention_clear_drop) if quality and retention else {'action':'hold','reason':'evaluation_not_completed'}
                            self.state['events'].append({'at':time.time(),'stage':stage.name,'round':pending['index'],'decision':decision,'quality':quality,'retention':retention})
                            if decision['action']=='promote':
                                progress.update(best_checkpoint=str(candidate),quality=quality,best_goal=self.state['current_goal'])
                                self.state['goal_leaders'][f'dev_act{stage.max_act}']={'checkpoint':str(candidate),'quality':quality,'goal':self.state['current_goal']}
                            elif decision['action'] in ('rollback','hold'):
                                self.state['current_checkpoint']=progress['best_checkpoint'];self.state['current_goal']=progress.get('best_goal')
                        progress['updates']+=pending['updates'];progress['transitions']+=pending['transitions'];progress['round']+=1
                        self.state['best_checkpoint']=progress['best_checkpoint']
                        hour=datetime.fromtimestamp(time.time(),timezone.utc).strftime('%Y%m%dT%H')
                        self.state['hourly_anchors'].setdefault(hour,self.state['best_checkpoint'])
                        capture=candidate.with_suffix('.continuations.jsonl')
                        if self.config.capture_continuations and capture.is_file():
                            pool=sampler.add_learner_continuations(pool,[capture]);self.state.setdefault('learner_continuations',[]).append(str(capture));self.state.setdefault('learner_file_hashes',{})[str(capture)]=sha_file(capture)
                        self.state['global_round']+=1;self.state.pop('pending_iteration',None);self.save()
                        retain_checkpoints(self.output,self.state)
                        cooldown=(record['finished']-record['started'])*(1/self.config.duty_cycle-1)
                        until=min(self.stage_deadline(),self.session_end,time.time()+cooldown)
                        while time.time()<until:time.sleep(min(5.,until-time.time()))
                    if self.state['allocation']['used_seconds']>=self.state['allocation']['stage_boundaries'][stage.name]-30:
                        self.state['current_checkpoint']=progress['best_checkpoint'];self.state['current_goal']=progress.get('best_goal')
                    elif self.state['current_checkpoint']!=progress['best_checkpoint']:
                        progress['needs_session_eval']=True
                    self.save()
                    while self.remaining_seconds()>0 and time.time()<self.session_end and self.state['allocation']['used_seconds']<self.state['allocation']['stage_boundaries'][stage.name]:
                        if not self.pause_until_safe():break
                        time.sleep(min(1.,max(.01,self.stage_deadline(stage)-time.time())))
                        self.save()
                self.set_accounting(False)
                self.state['status']=('pilot_complete' if self.session_kind=='pilot' else 'session_complete') if self.remaining_seconds()>0 else 'allocation_complete' if all(self.state['stages'].get(s.name,{}).get('updates',0)>0 for s in STAGES) else 'partial_allocation'
                self.save()
            finally:
                self.sleep_control(False)
