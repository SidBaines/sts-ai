"""Prepare state-only teacher targets with a stable split over entire worlds."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from .observations import PROMPT_VERSION, prompt, legal_indices, observation_identity
from .teacher import NonLLMTeacher, validate_teacher

# Version 1 hashed the raw identity string, so an inline integer world (``123``)
# and the same world referenced from its source file (``world:123``) could land
# in different splits. Version 2 hashes one canonical identity. Both versions
# use the same salt, so rows whose identity was already ``world:<seed>`` keep
# their version-1 split.
SPLIT_SCHEME_VERSION = 2
_SEED_IDENTITY = re.compile(r'(?:world:)?([0-9]+)')


def canonical_world_identity(world_identity) -> str:
    """One identity per world: integer seeds become ``world:<seed>``.

    An int, a decimal string or ``world:<decimal>`` all name the same seed.
    Other nonempty strings (synthetic or test worlds) are kept unchanged.
    """
    if isinstance(world_identity, bool) or not isinstance(world_identity, (int, str)):
        raise ValueError('world identity must be an integer seed or a nonempty string')
    if isinstance(world_identity, int):
        if world_identity < 0:
            raise ValueError('world seeds are nonnegative')
        return f'world:{world_identity}'
    if not world_identity:
        raise ValueError('world identity must be an integer seed or a nonempty string')
    match = _SEED_IDENTITY.fullmatch(world_identity)
    return f'world:{int(match.group(1))}' if match else world_identity


def world_split(world_identity) -> str:
    canonical = canonical_world_identity(world_identity)
    bucket = int(hashlib.sha256(('sts-llm-world-split-v1:' + canonical).encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def verified_split(row: dict) -> str:
    """Recompute a prepared row's split from its world and refuse disagreement.

    The stored ``split`` field is never trusted on its own: a hand-edited or
    differently prepared row could otherwise move a held-out world into training.
    """
    version = row.get('split_scheme_version', 1)
    if type(version) is not int or not 1 <= version <= SPLIT_SCHEME_VERSION:
        raise ValueError(f'row {row.get("id")!r} uses unknown split scheme {version!r}')
    if row.get('world_identity') is None:
        raise ValueError(f'row {row.get("id")!r} lacks world_identity, so its split cannot be verified')
    expected = world_split(row['world_identity'])
    if row.get('split') != expected:
        raise ValueError(f'row {row.get("id")!r} declares split {row.get("split")!r}, but world '
                         f'{canonical_world_identity(row["world_identity"])!r} belongs to {expected!r} '
                         f'(split scheme {SPLIT_SCHEME_VERSION}); re-prepare the dataset')
    return expected


def check_row_versions(row: dict, *, required: bool = True) -> None:
    """Refuse rows built for another prompt or public-observation version."""
    from sts_env.env import OBSERVATION_VERSION
    for key, current in (('prompt_version', PROMPT_VERSION), ('observation_version', OBSERVATION_VERSION)):
        value = row.get(key)
        if value is None:
            if required:
                raise ValueError(f'row {row.get("id")!r} lacks {key}; re-prepare it with sts_llm.dataset')
            continue
        if value != current:
            raise ValueError(f'row {row.get("id")!r} has {key} {value!r}, but the current {key} is {current!r}')


def saved_observation(row: dict) -> tuple[dict, str]:
    """Read immutable modern corpus rows, checking the file and record identity."""
    if 'observation' in row:
        world = row.get('world_identity')
        if world is None:
            raise ValueError('Inline observations require a world_identity for leakage-safe splitting')
        return row['observation'], canonical_world_identity(world)
    identity = row['source_record_identity']
    path = Path(identity['path'])
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != identity['file_sha256']:
        raise ValueError('Source world file differs from the recorded identity')
    content = json.loads(gzip.decompress(blob) if path.suffix == '.gz' else blob)
    record = content['labels'][identity['label_ordinal']]
    if record['public_hash'] != identity['public_hash']:
        raise ValueError('Source public record identity differs')
    observation = record['observation']
    if 'observation_sha256' in identity:
        canonical = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != identity['observation_sha256']:
            raise ValueError('Source observation hash differs')
    # The source seed is diagnostic only; it is never incorporated into messages.
    return observation, canonical_world_identity(content['world_seed'])


def prepare_rows(rows: list[dict], teacher: NonLLMTeacher, *, include_observations=False) -> list[dict]:
    observations, worlds = [], []
    for row in rows:
        obs, world = saved_observation(row)
        legal_indices(obs)
        if len(obs['legal_actions']) < 2:
            raise ValueError('Teacher training rows must be informative choices')
        observations.append(obs)
        worlds.append(world)
    targets = teacher.distributions(observations)
    prepared = []
    for original, observation, world, target in zip(rows, observations, worlds, targets):
        identity = observation_identity(observation)
        row = {'id': original.get('id', 'sts_' + identity[:16]),
               'phase': observation['phase'], 'act': observation.get('state', {}).get('act'),
               'legal_indices': legal_indices(observation), 'messages': prompt(observation),
               'teacher': target, 'world_identity': world, 'split': world_split(world),
               'split_scheme_version': SPLIT_SCHEME_VERSION,
               'prompt_version': PROMPT_VERSION,
               'observation_version': observation.get('state', {}).get('observation_version'),
               'public_observation_identity': identity,
               'source_record_identity': original.get('source_record_identity'),
               'diagnostic_only': original.get('diagnostic_only', {})}
        if include_observations:
            row['observation'] = observation
        validate_teacher(row)
        prepared.append(row)
    if len({r['id'] for r in prepared}) != len(prepared):
        raise ValueError('Dataset IDs must be unique')
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', required=True, help='JSONL with observations or immutable source references')
    parser.add_argument('--teacher-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher-python')
    parser.add_argument('--teacher-batch-size', type=int, default=64,
                        help='Observations sent to the teacher per request')
    parser.add_argument('--teacher-timeout', type=float, default=60.,
                        help='Seconds allowed for teacher startup and for each request')
    parser.add_argument('--output', required=True)
    parser.add_argument('--export-runtime', help='Fresh destination for inference-only dependency bundle')
    parser.add_argument('--include-checkpoint', action='store_true')
    parser.add_argument('--include-observations', action='store_true')
    args = parser.parse_args(argv)
    if args.teacher_batch_size <= 0:
        parser.error('--teacher-batch-size must be positive')
    output = Path(args.output)
    if output.exists() or output.with_suffix(output.suffix + '.manifest.json').exists():
        raise FileExistsError('Dataset outputs must be fresh')
    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    with NonLLMTeacher(teacher_root=args.teacher_root, checkpoint=args.checkpoint,
                      python=args.teacher_python, timeout=args.teacher_timeout,
                      batch_size=args.teacher_batch_size) as teacher:
        prepared = prepare_rows(rows, teacher, include_observations=args.include_observations)
        metadata = teacher.refresh_metadata()
        if args.export_runtime:
            teacher.export_runtime(args.export_runtime, include_checkpoint=args.include_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in prepared))
    import collections
    manifest = {'schema': 'sts_llm_teacher_dataset_v1', 'rows': len(prepared),
                'split_scheme_version': SPLIT_SCHEME_VERSION,
                'split_counts': dict(collections.Counter(r['split'] for r in prepared)),
                'phase_counts': dict(collections.Counter(r['phase'] for r in prepared)),
                'worlds': len({r['world_identity'] for r in prepared}), 'teacher': metadata,
                'teacher_batch_size': args.teacher_batch_size,
                'teacher_timeout_seconds_per_request': args.teacher_timeout,
                'source_prompts_sha256': hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'reward_semantics': 'Teacher policy probabilities are preferences, not calibrated action values.',
                'menu_protocol': 'Raw clean-environment menu; excluded teacher-filter actions have probability0.'}
    output.with_suffix(output.suffix + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('rows', 'split_counts', 'phase_counts', 'worlds', 'dataset_sha256')}))


if __name__ == '__main__':
    main()
