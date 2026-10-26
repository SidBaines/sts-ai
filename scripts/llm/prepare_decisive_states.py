"""Prepare decisive combat and out-of-combat states from a saved corpus.

Scans saved champion-play worlds from the non-LLM corpus, samples a few
candidate decisions per world and phase, labels them with the frozen teacher,
and keeps decisions whose top action leads the next by at least `--margin`.
Training states come from training-split worlds and the probe from
validation-split worlds; test-split worlds are skipped entirely. Out-of-combat
states whose potion line hides a menu potion (an env-v0.1.5.1 text fault, fixed
in 0.1.6) are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re

from sts_llm.dataset import SPLIT_SCHEME_VERSION, canonical_world_identity, world_split
from sts_llm.decision_metrics import decisive_margin, teacher_top
from sts_llm.observations import LABELLED_PROMPT_VERSION, legal_indices, observation_identity
from sts_llm.teacher import NonLLMTeacher

POTION_LINE = re.compile(r'^Potions: (.*)$', re.M)
POTION_ACTION = re.compile(r'(?:drink|discard) potion (.+?) \(slot (\d+)\)')


def potion_line_consistent(observation: dict) -> bool:
    """False when an out-of-combat menu potion is not at its slot in the Potions line."""
    if observation['phase'] == 'combat':
        return True
    match = POTION_LINE.search(observation['text'])
    shown = [] if not match or match.group(1).strip() == 'none' else [s.strip() for s in match.group(1).split(',')]
    for action in observation['legal_actions']:
        found = POTION_ACTION.search(action['description'])
        if found:
            name, slot = found.group(1), int(found.group(2))
            if slot >= len(shown) or shown[slot] != name:
                return False
    return True


def decision_type(observation: dict) -> str:
    if observation['phase'] == 'combat':
        room = str(observation['state'].get('room', '')).rsplit('.', 1)[-1]
        return {'BOSS': 'boss', 'ELITE': 'elite'}.get(room, 'hallway')
    descriptions = [a['description'] for a in observation['legal_actions']
                    if not a['description'].startswith(('drink potion ', 'discard potion ', 'recall'))]
    for prefix, kind in (('choose map', 'map'), ('take card ', 'card_reward'), ('event option ', 'event'),
                         ('buy ', 'shop'), ('rest', 'rest'), ('smith', 'rest')):
        if any(d.startswith(prefix) for d in descriptions):
            return kind
    return 'other'


def select(candidates: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Round-robin over decision types, at most one state per world."""
    by_type = defaultdict(list)
    for row in sorted(candidates, key=lambda r: r['id']):
        by_type[row['decision_type']].append(row)
    for rows in by_type.values():
        rng.shuffle(rows)
    chosen, worlds = [], set()
    while len(chosen) < count and any(by_type.values()):
        for kind in sorted(by_type):
            rows = by_type[kind]
            while rows and rows[-1]['world_identity'] in worlds:
                rows.pop()
            if rows and len(chosen) < count:
                row = rows.pop()
                chosen.append(row)
                worlds.add(row['world_identity'])
    if len(chosen) < count:
        raise ValueError(f'only {len(chosen)} decisive states available, {count} requested')
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, help='reuse saved labelled_candidates.jsonl without relabelling')
    parser.add_argument('--corpus', type=Path, required=True, help='directory holding shard_*/worlds/*.json.gz')
    parser.add_argument('--teacher-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--teacher-python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=1000)
    parser.add_argument('--candidates-per-world', type=int, nargs=2, default=(3, 6), metavar=('OOC', 'COMBAT'))
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--train-per-phase', type=int, default=16)
    parser.add_argument('--probe-per-phase', type=int, default=16)
    parser.add_argument('--max-actions', type=int, default=64)
    parser.add_argument('--max-text-chars', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'preparation_config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + '\n')
    rng = random.Random(args.seed)
    if args.candidates:
        files, excluded = [], Counter()
        unique = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
        metadata = json.loads((args.candidates.parent / 'teacher_manifest.json').read_text())
        if metadata['checkpoint']['sha256'] != args.checkpoint_sha256:
            raise ValueError('Cached labels use a different teacher checkpoint')
        for row in unique:
            row['decisive_margin'] = decisive_margin(row)
            row['teacher_top'] = teacher_top(row)
        (args.output / 'teacher_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    else:
        files = sorted(args.corpus.glob('shard_*/worlds/*.json.gz'))[:args.worlds]
        excluded = Counter()
        candidates = []
        for file_number, path in enumerate(files, 1):
            if file_number % 100 == 0:
                print(json.dumps({"event": "scan", "worlds": file_number, "total": len(files)}), flush=True)
            blob = path.read_bytes()
            content = json.loads(gzip.decompress(blob))
            world = canonical_world_identity(content['world_seed'])
            split = world_split(world)
            if split == 'test':
                excluded['test_world_decisions'] += len(content['labels'])
                continue
            per_phase = defaultdict(list)
            for ordinal, record in enumerate(content['labels']):
                observation = record['observation']
                menu = legal_indices(observation)
                if len(menu) < 2:
                    excluded['forced'] += 1
                elif len(menu) > args.max_actions or len(observation['text']) > args.max_text_chars:
                    excluded['too_large'] += 1
                elif not potion_line_consistent(observation):
                    excluded['potion_line_fault'] += 1
                else:
                    per_phase[observation['phase']].append((ordinal, record))
            limits = {'out_of_combat': args.candidates_per_world[0], 'combat': args.candidates_per_world[1]}
            for phase, records in per_phase.items():
                for ordinal, record in rng.sample(records, min(limits[phase], len(records))):
                    observation = record['observation']
                    canonical = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
                    candidates.append({
                        'id': 'sts_' + observation_identity(observation)[:16], 'phase': phase,
                        'act': observation['state'].get('act'), 'decision_type': decision_type(observation),
                        'world_identity': world, 'split': split, 'split_scheme_version': SPLIT_SCHEME_VERSION,
                        'observation': {key: observation[key] for key in ('phase', 'text', 'legal_actions')},
                        'legal_indices': legal_indices(observation),
                        'observation_version': observation['state'].get('observation_version'),
                        'source_record_identity': {'path': str(path), 'file_sha256': hashlib.sha256(blob).hexdigest(),
                                                   'label_ordinal': ordinal, 'public_hash': record['public_hash'],
                                                   'observation_sha256': hashlib.sha256(canonical).hexdigest()}})
        seen, unique = set(), []
        for row in candidates:
            if row['id'] not in seen:
                seen.add(row['id'])
                unique.append(row)
        with NonLLMTeacher(teacher_root=args.teacher_root, checkpoint=args.checkpoint, python=args.teacher_python,
                           expected_checkpoint_sha256=args.checkpoint_sha256, timeout=120) as teacher:
            metadata = teacher.metadata
            (args.output / 'teacher_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
            with (args.output / 'labelled_candidates.jsonl').open('x') as stream:
                for start in range(0, len(unique), 256):
                    batch = unique[start:start + 256]
                    targets = teacher.distributions([row['observation'] for row in batch])
                    for row, target in zip(batch, targets):
                        row['teacher'] = target
                        row['decisive_margin'] = decisive_margin(row)
                        row['teacher_top'] = teacher_top(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(json.dumps({'event': 'label', 'completed': start + len(batch), 'total': len(unique)}), flush=True)
    decisive = [row for row in unique if row['decisive_margin'] >= args.margin]
    sets = {}
    for name, split, count in (('train', 'train', args.train_per_phase), ('probe', 'validation', args.probe_per_phase)):
        rows = []
        for phase in ('out_of_combat', 'combat'):
            rows += select([r for r in decisive if r['split'] == split and r['phase'] == phase], count, rng)
        sets[name] = rows
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = {
        'schema': 'sts_llm_decisive_states_v1', 'prompt_family': LABELLED_PROMPT_VERSION,
        'cached_candidates': {'path': str(args.candidates), 'sha256': hashlib.sha256(args.candidates.read_bytes()).hexdigest()} if args.candidates else None,
        'corpus': str(args.corpus), 'world_files_scanned': len(files), 'seed': args.seed,
        'margin': args.margin, 'candidates_per_world': {'out_of_combat': args.candidates_per_world[0],
                                                        'combat': args.candidates_per_world[1]},
        'candidates_labelled': len(unique), 'exclusions': dict(excluded),
        'decisive_by_split_and_phase': dict(Counter(f"{r['split']}/{r['phase']}" for r in decisive)),
        'candidates_by_split_and_phase': dict(Counter(f"{r['split']}/{r['phase']}" for r in unique)),
        'teacher': {key: value for key, value in metadata.items() if not isinstance(value, (list, dict))},
        'teacher_checkpoint_sha256': args.checkpoint_sha256,
        'sets': {name: {'rows': len(rows), 'phases': dict(Counter(r['phase'] for r in rows)),
                        'decision_types': dict(Counter(r['decision_type'] for r in rows)),
                        'mean_margin': sum(r['decisive_margin'] for r in rows) / len(rows),
                        'teacher_top_is_first_option': sum(r['teacher_top'] == r['legal_indices'][0] for r in rows),
                        'distinct_worlds': len({r['world_identity'] for r in rows})}
                 for name, rows in sets.items()},
    }
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['sets'], indent=1))
    print(json.dumps({k: summary[k] for k in ('candidates_labelled', 'exclusions', 'decisive_by_split_and_phase')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
