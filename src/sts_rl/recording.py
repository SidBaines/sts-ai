"""Launch identities, source bytes and complete resumable training state."""
from __future__ import annotations

import hashlib
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
import zipfile


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return {'path': str(Path(path).resolve()), 'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def deduplicate_snapshot(path, objects):
    """Hard-link only our immutable copies, never a live input or checkpoint."""
    objects.mkdir(parents=True, exist_ok=True)
    target = objects / (file_identity(path)['sha256'] + Path(path).suffix)
    try:
        os.link(path, target)
    except FileExistsError:
        path.unlink()
        os.link(target, path)
    os.chmod(target, 0o444)


def begin_run(out, config, checkpoint=None, inputs=(), build_dir=None):
    """Record each start/resume separately; preserve source and native binary bytes."""
    from sts_env._backend.lightspeed_import import native_metadata, default_build_dir
    root = Path(__file__).resolve().parents[2]
    directory = Path(out).resolve() / 'records' / f'launch-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True)
    source_files = []
    with zipfile.ZipFile(directory/'source.zip', 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for section in ['src', 'scripts', 'tests', 'patches']:
            for path in sorted((root/section).rglob('*')):
                if path.is_file() and path.suffix in {'.py', '.json', '.patch', '.md', '.sh'} and '__pycache__' not in path.parts:
                    archive.write(path, path.relative_to(root).as_posix())
                    source_files.append(file_identity(path))
        archive.write(root/'pyproject.toml', 'pyproject.toml')
        source_files.append(file_identity(root/'pyproject.toml'))
    build_dir = Path(build_dir or os.environ.get('STS_LIGHTSPEED_BUILD', default_build_dir())).resolve()
    native = native_metadata(build_dir)
    shutil.copy2(build_dir/native['module_file'], directory/native['module_file'])
    shutil.copy2(build_dir/'build_manifest.json', directory/'build_manifest.json')
    if file_identity(directory/native['module_file'])['sha256'] != native['module_sha256']:
        raise RuntimeError('Native binary changed while recording launch')
    objects = Path(out).resolve().parent / 'record_objects'
    for path in [directory/'source.zip', directory/native['module_file']]:
        deduplicate_snapshot(path, objects)
    files = []
    for item in inputs:
        path = Path(item)
        files.extend(sorted(path.glob('*.jsonl')) + sorted(path.glob('*.json')) if path.is_dir() else [path])
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    runtime = {name: importlib.metadata.version(name) for name in ['torch', 'numpy']}
    record = {'schema': 'sts_training_launch_v1', 'created_unix': time.time(), 'effective_config': config,
              'argv': sys.argv, 'cwd': os.getcwd(), 'git_commit': commit,
              'source_files': source_files, 'source_archive': file_identity(directory/'source.zip'),
              'runtime': {'python': sys.version, 'platform': platform.platform(), 'packages': runtime},
              'environment': {key: os.environ.get(key) for key in ['STS_LIGHTSPEED_BUILD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED']},
              'parent_checkpoint': file_identity(checkpoint) if checkpoint else None,
              'data_inputs': [file_identity(path) for path in files], 'native': native,
              'native_build_dir': str(build_dir), 'fault_directory': str(directory/'faults'),
              'historical_queue_launch': os.environ.get('STS_RUN_RECORD')}
    path = directory/'launch.json'
    path.write_text(json.dumps(record, indent=2)+'\n')
    os.environ['STS_RUN_RECORD'] = str(path)
    os.environ['STS_FAULT_DIR'] = str(directory/'faults')
    return file_identity(path)


def require_fresh_or_resume(out, resume):
    out = Path(out)
    if resume:
        if not (out/'latest.pt').exists():
            raise ValueError('Cannot resume without latest.pt; choose a fresh output for a new condition')
    elif (out/'latest.pt').exists() or (out/'config.json').exists() or (out/'log.jsonl').exists():
        raise ValueError('Run output already contains training history; use --resume or a fresh output')


def capture_rng(local_rng=None):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'local_python': local_rng.getstate() if local_rng else None,
            'mps': torch.mps.get_rng_state().cpu() if torch.backends.mps.is_available() else None}


def restore_rng(state, local_rng=None):
    import numpy as np
    import torch
    rng = state.get('rng_state')
    if rng is None:
        return False  # legacy RNG fields are inherited and cannot be trusted
    random.setstate(rng['python'])
    torch.set_rng_state(rng['torch'])
    np.random.set_state(rng['numpy'])
    if local_rng is not None and rng['local_python'] is not None:
        local_rng.setstate(rng['local_python'])
    if rng.get('mps') is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng['mps'])
    return True


def resume_payload(model, optimizer, iteration, steps_total, config, run_record, local_rng=None):
    return {'recording_version': 'sts_training_resume_v1', 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'iteration': iteration, 'steps_total': steps_total,
            'config': config, 'run_record': run_record, 'rng_state': capture_rng(local_rng)}


def write_collection(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temporary, 'wt', compresslevel=1) as output:
        json.dump(record, output, separators=(',',':'))
        output.write('\n')
    os.replace(temporary, path)
