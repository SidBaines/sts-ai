"""Replay recorded simulator RPCs with finite deadlines and explicit outcomes."""
from __future__ import annotations

import time
import math

from sts_env._backend.lightspeed_import import native_metadata
from sts_env.provenance import interface_identity
from .curriculum import public_hash
from .faults import digest
from .workers import CurriculumWorker
from sts_env.env import SimulatorError


def replay_fault(record, *, build_dir=None, allow_native_change=False, timeout=30.0, seconds=120.0,
                 fault_dir=None, allow_interface_change=False):
    if record.get('schema') != 'sts_rpc_fault_v1':
        raise ValueError('Unsupported fault record')
    if not math.isfinite(timeout) or not math.isfinite(seconds) or timeout <= 0 or seconds <= 0 or timeout > 60 or seconds > 600:
        raise ValueError('Replay needs finite bounds: timeout <=60s, total <=600s')
    if digest(record['successful_rpcs']) != record['rpc_history_sha256']:
        raise ValueError('Fault RPC history checksum mismatch')
    config = dict(record['worker_config'])
    if build_dir:
        config['build_dir'] = str(build_dir)
    actual = native_metadata(config.get('build_dir'))
    changed = actual != record['native_provenance']
    if changed and not allow_native_change:
        raise ValueError('Native identity differs; use --allow-native-change for a separately labelled candidate')
    if interface_identity() != {k:record[k] for k in interface_identity()} and not allow_interface_change:
        raise ValueError('Public Python interface/runtime differs from fault record')
    config.update(fault_dir=fault_dir)
    worker_type = record['worker_class']
    if worker_type == 'PanelWorker':
        from sts_rl.panel import PanelWorker
        cls = PanelWorker
    elif worker_type == 'CurriculumWorker':
        cls = CurriculumWorker
    else:
        raise ValueError('Unknown fault worker class')
    allowed = {'reset', 'step', 'teacher_action', 'panel', 'afterstates', 'encounters', 'menu',
               'search_votes', 'search_votes_public'}
    calls = record['successful_rpcs'] + [record['attempted_rpc']]
    started = time.monotonic()
    with cls(config, timeout=min(timeout, seconds)) as worker:
        for index, call in enumerate(calls):
            if call['method'] not in allowed:
                raise ValueError('Unsupported recorded RPC')
            remaining = seconds - (time.monotonic()-started)
            if remaining <= 0:
                return {'status':'replay_budget_exhausted', 'rpc_index':index, 'native_changed':changed}
            worker.timeout = min(timeout, remaining)
            target = index == len(calls)-1
            if target and record.get('last_public_hash') and worker.observation is not None:
                if public_hash(worker.observation) != record['last_public_hash']:
                    return {'status':'history_diverged', 'rpc_index':index, 'native_changed':changed}
            try:
                value = getattr(worker, call['method'])(*call['args'])
            except SimulatorError as exc:
                expected = record['exception']
                same = type(exc).__name__ == expected['type'] and str(exc) == expected['message']
                return {'status':('reproduced' if same else 'different_failure') if target else 'earlier_failure',
                        'rpc_index':index, 'method':call['method'], 'error':type(exc).__name__, 'message':str(exc),
                        'fault_record_path':worker.last_fault_path, 'native_changed':changed}
            if target:
                contained = worker._contained_fault(value) if call['method'] in {'panel','afterstates'} else False
                return {'status':'reproduced_contained_fault' if contained else 'not_reproduced',
                        'rpc_index':index, 'native_changed':changed}
            if 'public_hash' in call and public_hash(value[0]) != call['public_hash']:
                return {'status':'history_diverged', 'rpc_index':index, 'native_changed':changed}
            if 'result' in call and value != call['result']:
                return {'status':'teacher_result_diverged', 'rpc_index':index, 'native_changed':changed}
    raise RuntimeError('Empty RPC history')
