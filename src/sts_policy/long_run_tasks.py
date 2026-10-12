"""Immutable rolling TRAIN tasks and new prespecified world-level eval panels."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from .curriculum import TaskSpec, public_hash, validate_provenance, world_split

LONG_TASK_VERSION = "sts_long_curriculum_tasks_v1"
NATIVE_SHA = "40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd"
INTERFACE_SHA = "1fa75a1d3d7de01ceae2ba0461f8ce3cae4051e8e6ba94ff3ff1897a9af02205"
ROUND_STRIDE = 100000
STREAM_STRIDE = 100000000000
STREAMS = {"act2_continuations": 0, "act2_full": 1, "act3": 2, "mixed_full3": 3}
DEFAULT_FRACTIONS = {
    "act2": {"act2": .55, "act3": 0., "scratch": .15, "full3": .20, "rehearsal": .10},
    "act3": {"act2": 0., "act3": .60, "scratch": .30, "rehearsal": .10},
    "full3": {"act2": .15, "act3": .15, "scratch": .60, "rehearsal": .10},
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def task_identity(task: TaskSpec) -> str:
    return hashlib.sha256(json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ContinuationPool:
    by_act: dict[int, tuple[dict[str, Any], ...]]
    provenance: dict[str, Any]
    source_identity: dict[str, Any]


def load_continuations(bank: str | Path, *, metadata_panels: str | Path) -> ContinuationPool:
    """Read TRAIN panel rows only; use their recorded public act, never infer hidden state.

    Earliest public battle per world/act avoids turning late-boss starts into an
    apparent whole-act curriculum. Existing old DEV/FINAL observations are not read.
    """
    bank, panels = Path(bank).resolve(), Path(metadata_panels).resolve()
    manifest_path = bank / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest['provenance']
    if (provenance.get('environment_version') != '0.1.1' or
            provenance.get('python_interface_sha256') != INTERFACE_SHA or
            provenance.get('native_provenance', {}).get('module_sha256') != NATIVE_SHA):
        raise ValueError('Continuation bank is not the corrected native 0.1.1 source')
    if manifest.get('settings', {}).get('max_act') != 3:
        raise ValueError('Continuation bank must contain three-act trajectories')
    sources = {'manifest': {'path': str(manifest_path), 'sha256': file_sha(manifest_path)}, 'panels': {}}
    by_act = {}
    for act in (2, 3):
        path = panels / f'train_act{act}.jsonl'
        sources['panels'][str(act)] = {'path': str(path), 'sha256': file_sha(path)}
        selected = {}
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Filter by world before using any public-state metadata/prefix.
                if row.get('split') != 'train' or world_split(row['seed']) != 'train':
                    raise ValueError('Non-TRAIN world in continuation TRAIN panel')
                metadata = row.get('curriculum_metadata', {})
                if Path(metadata.get('source_bank', '')).resolve() != bank:
                    continue
                floor = metadata.get('floor')
                if (metadata.get('act') != act or not isinstance(floor, int) or
                        not 17 * (act - 1) < floor <= 17 * act or row.get('floor') != floor):
                    raise ValueError('Continuation public act/floor metadata is inconsistent')
                spec = TaskSpec.from_dict(row)
                if (not spec.prefix or not spec.expected_public_hash or
                        len(spec.expected_public_hash) != 64 or spec.kind != 'battle' or
                        spec.max_act != 3 or len(spec.ooc_previews) != len(spec.prefix)):
                    raise ValueError('Continuation lacks a replayable saved battle frontier')
                validate_provenance(provenance, spec.expected_provenance or {})
                if row.get('public_hash', spec.expected_public_hash) != spec.expected_public_hash:
                    raise ValueError('Continuation public hashes disagree')
                previous = selected.get(spec.seed)
                key = (floor, len(spec.prefix), task_identity(spec))
                if previous is None or key < previous[0]:
                    selected[spec.seed] = (key, row)
        if not selected:
            raise ValueError(f'No TRAIN Act {act} continuations in selected bank')
        by_act[act] = tuple(selected[seed][1] for seed in sorted(selected))
    sources['selection'] = 'earliest_public_battle_per_train_world_act'
    return ContinuationPool(by_act, provenance, sources)


def _train_worlds(start: int, count: int, stop: int) -> list[int]:
    if count == 0:
        return []
    worlds = []
    for seed in range(start, stop):
        if world_split(seed) == 'train':
            worlds.append(seed)
            if len(worlds) == count:
                return worlds
    if count == 0:
        return []
    raise ValueError('Training seed block exhausted; refuse overlapping another round')


def _counts(fractions: dict[str, float], episodes: int) -> dict[str, int]:
    values = {key: episodes * fraction for key, fraction in fractions.items()}
    counts = {key: math.floor(value) for key, value in values.items()}
    order = sorted(values, key=lambda key: (-(values[key] - counts[key]), key))
    for key in order[:episodes - sum(counts.values())]:
        counts[key] += 1
    return counts


def sample_training_tasks(pool: ContinuationPool, *, stage: str, round_index: int,
                          seed: int, episodes: int = 128, world_start: int = 1000000,
                          horizon: int = 4096, simulations: int = 200,
                          fractions: dict[str, float] | None = None,
                          stream: str | None = None) -> dict[str, Any]:
    """Scratch worlds never overlap across stream/round blocks; bank replay is explicit.

    A round's condition/episodes/settings must remain unchanged on resume. IDs
    name immutable TaskSpec content, not arbitrary fabricated duplicate row IDs.
    """
    if stage not in DEFAULT_FRACTIONS:
        raise ValueError('Unknown long curriculum stage')
    stream = stream or {'act2': 'act2_continuations', 'act3': 'act3', 'full3': 'mixed_full3'}[stage]
    if stream not in STREAMS or not 0 <= round_index < STREAM_STRIDE // ROUND_STRIDE:
        raise ValueError('Invalid stream/round identity')
    if world_start < 1000000 or not 1 <= episodes <= 10000:
        raise ValueError('TRAIN scratch seeds start at 1000000; bounded episode count required')
    fractions = dict(DEFAULT_FRACTIONS[stage] if fractions is None else fractions)
    for component in ('act2', 'act3', 'scratch', 'full3', 'rehearsal'):
        fractions.setdefault(component, 0.)
    if (set(fractions) != {'act2', 'act3', 'scratch', 'full3', 'rehearsal'} or
            any(not math.isfinite(value) or value < 0 for value in fractions.values()) or
            not math.isclose(sum(fractions.values()), 1., abs_tol=1e-9) or
            (stage == 'act2' and (fractions['act3'] > 0 or fractions['full3'] < .20))):
        raise ValueError('Invalid curriculum mixture fractions')
    counts = _counts(fractions, episodes)
    target_act = 2 if stage == 'act2' else 3
    start = world_start + STREAMS[stream] * STREAM_STRIDE + round_index * ROUND_STRIDE
    worlds = iter(_train_worlds(start, counts['scratch'] + counts['full3'] + counts['rehearsal'], start + ROUND_STRIDE))
    if start + ROUND_STRIDE >= 2**63:
        raise ValueError('Training seed namespace exceeds TaskSpec bounds')
    rng = random.Random(f'{LONG_TASK_VERSION}:{seed}:{stream}:{round_index}')
    pairs = []
    for component in ('scratch', 'full3', 'rehearsal'):
        for _ in range(counts[component]):
            task = TaskSpec(seed=next(worlds), kind='run', horizon=horizon,
                            max_act=1 if component == 'rehearsal' else 3 if component == 'full3' else target_act,
                            battle_simulations=simulations)
            pairs.append((task, {'component': component, 'start_source': 'fresh_scratch', 'start_act': 1, 'start_floor': 0, 'goal_act': task.max_act, 'horizon': task.horizon}))
    for act in (2, 3):
        count = counts[f'act{act}']
        rows = list(pool.by_act[act])
        if count > len(rows):
            raise ValueError('Continuation count exceeds distinct TRAIN worlds in an act')
        rng.shuffle(rows)
        for row in rows[:count]:
            task = replace(TaskSpec.from_dict(row), kind='run', max_act=target_act,
                           horizon=horizon, battle_simulations=simulations, seek_combat=False)
            pairs.append((task, {'component': f'act{act}', 'start_act': act,
                                 'start_floor': row['curriculum_metadata']['floor'], 'start_source': row.get('long_metadata', {}).get('start_source', 'native_teacher'), 'goal_act': task.max_act, 'horizon': task.horizon,
                                 'source_frontier_id': task_identity(TaskSpec.from_dict(row))}))
    rng.shuffle(pairs)
    tasks = [task for task, _ in pairs]
    if len({task_identity(task) for task in tasks}) != episodes:
        raise ValueError('Duplicate immutable task identities in round')
    metadata = {'version': LONG_TASK_VERSION, 'stage': stage, 'stream': stream,
                'round_index': round_index, 'optimizer_seed': seed, 'fractions': fractions,
                'counts': counts, 'scratch_seed_block': [start, start + ROUND_STRIDE],
                'source': pool.source_identity, 'source_provenance': pool.provenance,
                'tasks': [{'task_id': task_identity(task), 'world_seed': task.seed, **row}
                          for task, row in pairs],
                'continuation_reuse': 'finite TRAIN teacher bank; deliberately sampled again across rounds'}
    return {'tasks': tasks, 'metadata': metadata}


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'Immutable task artifact differs: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise ValueError(f'Concurrent immutable task artifact differs: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_task_batch(batch: dict[str, Any], path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    for task, metadata in zip(batch['tasks'], batch['metadata']['tasks'], strict=True):
        if (metadata['task_id'] != task_identity(task) or metadata['world_seed'] != task.seed or
                metadata.get('goal_act', task.max_act) != task.max_act):
            raise ValueError('Task metadata identity or family differs from TaskSpec')
    rows = [{**task.to_dict(), 'long_metadata': metadata}
            for task, metadata in zip(batch['tasks'], batch['metadata']['tasks'], strict=True)]
    content = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows)
    digest = hashlib.sha256(content.encode()).hexdigest()
    manifest = {**batch['metadata'], 'task_file': str(path), 'task_file_sha256': digest}
    _write_immutable(path, content)
    manifest_path = path.with_suffix(path.suffix + '.manifest.json')
    _write_immutable(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return {'path': str(path), 'manifest': str(manifest_path), 'sha256': digest, 'task_ids': [task_identity(task) for task in batch['tasks']]}


def fixed_dev_panels(output: str | Path, *, world_start: int = 200000,
                     episodes: int = 32, horizon: int = 4096,
                     simulations: int = 200, seed: int = 777) -> dict[str, Any]:
    """Define fresh DEV/FINAL worlds before training, without loading their game states."""
    if world_start < 200000 or world_start + 200000 > 1000000 or not 1 <= episodes <= 10000:
        raise ValueError('Evaluation requires separate new bounded seed intervals below TRAIN start')
    output = Path(output).resolve()
    worlds, paths, files = {}, {}, {}
    for split, start in [('dev', world_start), ('final', world_start + 100000)]:
        selected = []
        for world in range(start, start + 100000):
            if world_split(world) == split:
                selected.append(world)
                if len(selected) == episodes:
                    break
        if len(selected) != episodes:
            raise ValueError('Evaluation seed block exhausted')
        random.Random(seed).shuffle(selected)
        worlds[split] = selected
        for act in (1, 2, 3):
            label = f'{split}_act{act}'
            tasks = [TaskSpec(seed=world, kind='run', max_act=act, horizon=horizon,
                              battle_simulations=simulations) for world in selected]
            batch = {'tasks': tasks, 'metadata': {'version': LONG_TASK_VERSION, 'split': split,
                     'target_act': act, 'seed': seed, 'tasks': [{'task_id': task_identity(task),
                     'world_seed': task.seed, 'component': 'prespecified_fresh_evaluation'} for task in tasks]}}
            files[label] = write_task_batch(batch, output / f'{label}.jsonl')
            paths[label] = files[label]['path']
    metadata = {'version': LONG_TASK_VERSION, 'world_start': world_start, 'worlds': worlds,
                'episodes': episodes, 'horizon': horizon, 'simulations': simulations, 'seed': seed,
                'files': files, 'evaluation_embargo': 'FINAL learner outcomes withheld until selection freezes'}
    manifest = output / 'panels.json'
    _write_immutable(manifest, json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    return {'paths': paths, 'metadata': metadata, 'manifest': str(manifest)}


def capture_learner_continuation(task: TaskSpec, previous: dict[str, Any],
                                 observation: dict[str, Any], info: dict[str, Any], *,
                                 terminated: bool, truncated: bool) -> dict[str, Any] | None:
    """Capture only a first live act entry reached by a TRAIN max_act3 actor.

    Terminal-cap observations have empty legal menus and different hashes. They
    are never recast as live snapshots. Collector buffers this row until a whole
    healthy episode finishes, discarding it on any later simulator fault.
    """
    if task.kind != 'run' or task.max_act != 3 or world_split(task.seed) != 'train':
        return None
    act = observation.get('state', {}).get('act')
    old_act = previous.get('state', {}).get('act')
    if (act not in (2, 3) or not isinstance(old_act, int) or old_act >= act or
            terminated or truncated or not observation.get('legal_actions') or
            observation.get('state', {}).get('undefined_behavior_evoked') or
            str(info.get('outcome', '')).rsplit('.', 1)[-1] != 'UNDECIDED'):
        return None
    keys = ('python_interface_sha256', 'runtime_python_version', 'environment_version',
            'observation_version', 'native_provenance')
    if any(key not in info for key in keys):
        raise ValueError('Learner frontier lacks replay provenance')
    provenance = {key: info[key] for key in keys}
    if (provenance['python_interface_sha256'] != INTERFACE_SHA or
            provenance['environment_version'] != '0.1.1' or
            provenance['native_provenance'].get('module_sha256') != NATIVE_SHA):
        raise ValueError('Learner frontier uses a different backend')
    prefix, previews = info.get('action_prefix', []), info.get('ooc_previews', [])
    if not prefix or len(prefix) != len(previews):
        raise ValueError('Learner frontier has incomplete canonical replay history')
    continuation = TaskSpec(seed=task.seed, kind='run', prefix=tuple(prefix),
                            ooc_previews=tuple(previews), expected_public_hash=public_hash(observation),
                            expected_provenance=provenance, horizon=task.horizon, max_act=3,
                            battle_simulations=task.battle_simulations, ascension=task.ascension,
                            seek_combat=False)
    floor = observation['state'].get('floor')
    if not isinstance(floor, int) or not 17 * (act - 1) <= floor <= 17 * act:
        raise ValueError('Learner public act/floor metadata is inconsistent')
    return {**continuation.to_dict(), 'long_metadata': {
        'start_source': 'learner', 'start_act': act, 'start_floor': floor,
        'goal_act': 3, 'horizon': continuation.horizon,
        'source_task_id': task_identity(task), 'task_id': task_identity(continuation),
        'parent_start_source': 'native_or_learner_continuation' if task.prefix else 'fresh_scratch'}}


def add_learner_continuations(pool: ContinuationPool, files: list[str | Path]) -> ContinuationPool:
    """Merge separately collected, fault-quarantined TRAIN starts; preserve source labels."""
    rows = {act: {TaskSpec.from_dict(row).seed: row for row in pool.by_act[act]} for act in (2, 3)}
    sources = json.loads(json.dumps(pool.source_identity))
    sources['learner_files'] = []
    sources['learner_selection'] = 'chosen last valid row in supplied file order per world/act; not chronological recency'
    sources['learner_replacements'] = []
    for reference in files:
        path = Path(reference).resolve()
        sources['learner_files'].append({'path': str(path), 'sha256': file_sha(path)})
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                spec = TaskSpec.from_dict(row)
                metadata = row.get('long_metadata', {})
                act, floor = metadata.get('start_act'), metadata.get('start_floor')
                if (world_split(spec.seed) != 'train' or metadata.get('start_source') != 'learner' or
                        act not in (2, 3) or spec.kind != 'run' or not spec.prefix or
                        not spec.expected_public_hash or spec.max_act != 3 or
                        len(spec.prefix) != len(spec.ooc_previews) or
                        not isinstance(floor, int) or not 17 * (act - 1) <= floor <= 17 * act or
                        metadata.get('task_id') != task_identity(spec)):
                    raise ValueError('Invalid learner continuation source')
                validate_provenance(pool.provenance, spec.expected_provenance or {})
                row['curriculum_metadata'] = {'act': act, 'floor': floor, 'source': 'learner'}
                previous = rows[act].get(spec.seed)
                if previous is not None:
                    sources['learner_replacements'].append({'world_seed': spec.seed, 'act': act,
                        'previous_task_id': task_identity(TaskSpec.from_dict(previous)),
                        'chosen_task_id': task_identity(spec), 'chosen_file': str(path),
                        'previous_start_source': previous.get('long_metadata', {}).get('start_source', 'native_teacher'),
                        'chosen_start_source': 'learner'})
                rows[act][spec.seed] = row
    return ContinuationPool({act: tuple(rows[act][seed] for seed in sorted(rows[act])) for act in (2, 3)},
                            pool.provenance, sources)


def write_learner_continuations(rows: list[dict[str, Any]], path: str | Path, *,
                                source_checkpoint: str | Path | None,
                                source_iteration: int, source_policy_sha256: str | None = None,
                                source_optimizer_step: int | None = None) -> dict[str, Any]:
    """Atomically replace a deduplicated recovery file; capture remains separately sourced."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                existing[task_identity(TaskSpec.from_dict(row))] = row
    source_hash = file_sha(Path(source_checkpoint)) if source_checkpoint is not None else None
    for row in rows:
        spec = TaskSpec.from_dict(row)
        if world_split(spec.seed) != 'train' or row.get('long_metadata', {}).get('start_source') != 'learner':
            raise ValueError('Only TRAIN learner continuations may be written')
        row = json.loads(json.dumps(row))
        row['long_metadata'].update(source_checkpoint_lineage=str(source_checkpoint) if source_checkpoint else None,
                                    source_checkpoint_lineage_sha256=source_hash, source_iteration=source_iteration,
                                    source_policy_sha256=source_policy_sha256, source_optimizer_step=source_optimizer_step)
        existing[task_identity(spec)] = row
    # Keep every selected world/act frontier distinct; state IDs are canonical.
    content = ''.join(json.dumps(existing[key], sort_keys=True, separators=(',', ':')) + '\n'
                      for key in sorted(existing))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content);output.flush();os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'path': str(path), 'sha256': hashlib.sha256(content.encode()).hexdigest(),
            'new_frontiers': len(rows), 'unique_frontiers': len(existing),
            'quarantine': 'saved only after entire source episode completes without a simulator fault'}


def policy_parameters_sha256(model: Any) -> str:
    """Actual frozen collection actor identity, separate from its initial lineage file."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
