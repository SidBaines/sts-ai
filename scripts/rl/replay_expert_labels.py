#!/usr/bin/env python3
"""Replay completed teacher labels and compare HP carried out of paired fights."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import torch
    from sts_policy.curriculum import TaskSpec, public_hash
    from sts_policy.model_agent import collate_public, forward_policy
    from sts_policy.workers import CurriculumWorker
    from sts_rl.common import WORKER_CONFIG, JsonlLog, load_policy
    from sts_rl.recording import begin_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=600.)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() or args.threads < 1 or args.seconds <= 0:
        parser.error('Require fresh output and positive bounds')
    teacher = Path(args.teacher)
    report = json.loads((teacher / 'summary.json').read_text())
    if report['faults'] or report['unstarted'] or report['completed'] != report['requested']:
        raise ValueError('Replay acceptance requires a completed clean teacher panel')
    bank = {(r['task']['seed'], r['floor']): r for r in
            (json.loads(line) for line in Path(args.bank).read_text().splitlines() if line.strip())}
    labels = defaultdict(list)
    for shard in sorted(teacher.glob('shard_*.jsonl')):
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            if row['teacher_protocol'] != 'canonical_rng_resampled_search_played_fight_v2':
                raise ValueError('Require declared canonical teacher labels')
            labels[row['seed'], row['obs']['state']['floor']].append(row)
    assert sum(map(len, labels.values())) == report['labels']
    record = begin_run(out, vars(args), args.checkpoint, inputs=[args.teacher, args.bank])
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    lock = threading.Lock()
    log = JsonlLog(out / 'log.jsonl')
    log.emit({'event': 'start', 'config': vars(args), 'run_record': record})
    started = time.monotonic()

    def policy(obs):
        with lock, torch.no_grad():
            logits, _, _ = forward_policy(model, collate_public([obs]))
        return int(obs['legal_actions'][int(logits[0, :len(obs['legal_actions'])].argmax())]['index'])

    def run(slot):
        worker = CurriculumWorker(WORKER_CONFIG, timeout=60.)
        results = []
        try:
            for fight in report['fights'][slot::args.threads]:
                if time.monotonic() - started >= args.seconds:
                    break
                key = fight['seed'], fight['floor']
                source = bank[key]
                spec = TaskSpec.from_dict(source['task'])
                row = {'seed': key[0], 'floor': key[1], 'group': fight['group']}
                for mode in ['base', 'expert']:
                    obs, _ = worker.reset(spec)
                    initial_hash = public_hash(obs)
                    assert initial_hash == spec.expected_public_hash
                    done = cut = False
                    used = steps = 0
                    sequence = labels[key]
                    while not (done or cut):
                        if mode == 'expert' and used < len(sequence) and public_hash(obs) == public_hash(sequence[used]['obs']):
                            votes = sequence[used]['votes']
                            assert sum(votes.values()) == 24
                            action = int(max(votes.items(), key=lambda item: item[1])[0])
                            assert action in {int(a['index']) for a in obs['legal_actions']}
                            used += 1
                        else:
                            action = policy(obs)
                        obs, _, done, cut, info = worker.step(action)
                        steps += 1
                    assert not cut
                    won = info['task_outcome'] == 'battle_victory'
                    assert won == fight[mode + '_win']
                    if mode == 'base':
                        assert won == source['won']
                    else:
                        assert used == len(sequence) == fight['labels']
                        assert steps == fight['steps']
                    row[mode + '_win'] = won
                    row[mode + '_remaining_hp'] = max(0, int(obs['state']['cur_hp'])) if won else 0
                    row[mode + '_steps'] = steps
                row['hp_difference'] = row['expert_remaining_hp'] - row['base_remaining_hp']
                results.append(row)
                with lock:
                    log.emit({'event': 'fight', **row})
        finally:
            worker.close()
        return results

    with ThreadPoolExecutor(args.threads) as executor:
        results = [row for part in executor.map(run, range(args.threads)) for row in part]
    groups = {}
    for group in sorted({row['group'] for row in results}):
        rows = [r for r in results if r['group'] == group]
        both = [r for r in rows if r['base_win'] and r['expert_win']]
        groups[group] = {'n': len(rows), 'base_wins': sum(r['base_win'] for r in rows),
                         'expert_wins': sum(r['expert_win'] for r in rows),
                         'mean_base_remaining_hp': sum(r['base_remaining_hp'] for r in rows) / len(rows),
                         'mean_expert_remaining_hp': sum(r['expert_remaining_hp'] for r in rows) / len(rows),
                         'mean_hp_gain': sum(r['hp_difference'] for r in rows) / len(rows),
                         'both_win_n': len(both),
                         'both_win_mean_hp_gain': sum(r['hp_difference'] for r in both) / len(both) if both else None}
    summary = {'schema': 'sts_teacher_label_replay_hp_v1', 'run_record': record,
               'requested': report['requested'], 'completed': len(results),
               'unstarted': report['requested'] - len(results), 'groups': groups, 'fights': results,
               'scope': 'Independent fresh-worker replay; every saved label hash and action checked, all final outcomes/teacher step counts checked. HP includes zero on defeats. Groups conditional and overlapping by source world.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    log.emit({'event': 'done', 'completed': len(results), 'unstarted': summary['unstarted']})
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
