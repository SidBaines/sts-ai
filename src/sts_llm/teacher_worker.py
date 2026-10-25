"""JSON-lines subprocess. Only this child imports the non-LLM source tree."""
from __future__ import annotations
import os
import sys

# Run as a script, Python puts this file's directory (src/sts_llm) first on
# sys.path. Modules there, such as sts_llm/statistics.py, would then shadow
# standard-library modules for the teacher code, NumPy and PyTorch. Remove it
# before importing anything else; this works on every Python version.
_PACKAGE_DIRECTORY = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [entry for entry in sys.path
               if os.path.realpath(os.path.abspath(entry or os.curdir)) != _PACKAGE_DIRECTORY]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402

# Used only when the checkpoint does not record its own versions (older files).
FALLBACK_ENCODER_VERSION = 'public_candidate_v2'
FALLBACK_CHECKPOINT_FORMAT_VERSION = 1


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def source_revision(root):
    # A bundle under a different Git checkout must retain its source revision.
    saved = Path(root) / 'teacher_runtime_manifest.json'
    if saved.exists():
        return json.loads(saved.read_text()).get('source_revision')
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def checkpoint_versions(payload) -> dict:
    """Encoder and checkpoint format versions recorded in a policy checkpoint.

    Each field falls back to the historical constant when the checkpoint does
    not record it, and ``version_source`` says which was used.
    """
    metadata = payload.get('metadata') if isinstance(payload, dict) else None
    encoder = metadata.get('encoder_version') if isinstance(metadata, dict) else None
    checkpoint_format = payload.get('format_version') if isinstance(payload, dict) else None
    return {'encoder_version': FALLBACK_ENCODER_VERSION if encoder is None else encoder,
            'checkpoint_format_version': (FALLBACK_CHECKPOINT_FORMAT_VERSION if checkpoint_format is None
                                          else checkpoint_format),
            'version_source': {
                'encoder_version': 'fallback_constant' if encoder is None else 'checkpoint_metadata',
                'checkpoint_format_version': ('fallback_constant' if checkpoint_format is None
                                              else 'checkpoint_metadata')}}


def read_checkpoint_versions(path, expected_identity) -> dict:
    try:
        import torch
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:  # Unreadable here: keep the historical constants, say why.
        return {**checkpoint_versions(None), 'version_source_error': f'{type(exc).__name__}: {exc}'}
    if identity(path) != expected_identity:
        raise RuntimeError('Teacher checkpoint changed while its versions were read')
    return checkpoint_versions(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    root = Path(args.teacher_root).resolve()
    source = root / 'src'
    if not source.is_dir():
        raise ValueError('Teacher root must contain src/')
    # Never combine the clean environment and workstream packages in the parent.
    sys.path.insert(0, str(source))
    from sts_rl.fresh_study import StudyPolicy, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID
    checkpoint = identity(args.checkpoint)
    policy = StudyPolicy(checkpoint['path'])
    versions = read_checkpoint_versions(checkpoint['path'], checkpoint)
    revision = source_revision(root)
    def metadata():
        runtime_files = []
        for module in tuple(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if not filename:
                continue
            path = Path(filename).resolve()
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            if path.suffix == '.py':
                runtime_files.append({**identity(path), 'relative_path': str(relative)})
        return {'kind': 'nonllm_policy_probability_v1', 'checkpoint': checkpoint,
                'source_revision': revision, 'teacher_root': str(root),
                'filter_ids': [FILTER_ID, DEAD_FILTER_ID], 'public_context_id': PUBLIC_CONTEXT_ID,
                **versions,
                'runtime_files': sorted(runtime_files, key=lambda x: x['relative_path']),
                'native_simulator_loaded': any(name == 'slaythespire' or name.startswith('slaythespire.')
                                               for name in sys.modules)}
    print(json.dumps({'ready': True, 'metadata': metadata()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request.get('op') == 'metadata':
                result = metadata()
            elif request.get('op') == 'distributions':
                result = []
                for observation in request['observations']:
                    indices, probabilities = policy.distribution(observation)
                    original = [a['index'] for a in observation['legal_actions']]
                    result.append({'kind': 'nonllm_policy_probability_v1',
                                   'legal_indices': indices, 'probabilities': probabilities,
                                   'excluded_indices': [i for i in original if i not in indices],
                                   'checkpoint_sha256': checkpoint['sha256'],
                                   'source_revision': revision,
                                   'filter_ids': [FILTER_ID, DEAD_FILTER_ID],
                                   'public_context_id': PUBLIC_CONTEXT_ID,
                                   'score_semantics': 'policy_probability_not_action_value'})
            else:
                raise ValueError('Unsupported teacher operation')
            print(json.dumps({'id': request['id'], 'result': result}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'id': request.get('id'), 'error': {'kind': type(exc).__name__,
                                                               'message': str(exc)}}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
