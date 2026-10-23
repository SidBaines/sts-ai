"""JSON-lines subprocess. Only this child imports the non-LLM source tree."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


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
                'encoder_version': 'public_candidate_v2', 'checkpoint_format_version': 1,
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
