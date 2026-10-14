#!/usr/bin/env python3
"""Inspect the existing policy's first forced shop-removal target on fresh TRAIN worlds."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def main():
    import torch
    from sts_env.env import SimulatorError
    from sts_policy.model_agent import collate_public, forward_policy
    from sts_rl.common import make_workers, train_tasks
    from sts_rl.recording import begin_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--world-start', type=int, default=42000000000)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() or args.n < 1:
        parser.error('Fresh output and positive world count required')
    from sts_rl.common import load_policy
    record = begin_run(str(out) + '.record', vars(args), args.checkpoint)
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    worker = make_workers(1, dead_action_filter=True, slim=False)[0]
    results, examples = [], []
    try:
        for task in train_tasks(args.world_start, args.n):
            try:
                obs, _ = worker.submit('reset', task).result()
                term = cut = False
                steps = 0
                while not (term or cut):
                    removals = [a for a in obs['legal_actions'] if a['description'].startswith('buy card remove')]
                    if removals and obs['state']['floor'] <= 33:
                        before = obs
                        obs, _, term, cut, _ = worker.submit('step', removals[0]['index']).result()
                        with torch.no_grad():
                            logits, _, _ = forward_policy(model, collate_public([obs]))
                        pos = int(logits[0, :len(obs['legal_actions'])].argmax())
                        chosen = obs['legal_actions'][pos]
                        results.append({'seed': task.seed, 'floor': before['state']['floor'],
                                        'purchase': removals[0]['description'], 'chosen': chosen['description'],
                                        'selection_screen': obs['state']['screen_state']})
                        examples.append({'seed': task.seed, 'before_purchase': before,
                                         'selection': obs, 'chosen_position': pos,
                                         'logits': logits[0, :len(obs['legal_actions'])].tolist()})
                        break
                    with torch.no_grad():
                        logits, _, _ = forward_policy(model, collate_public([obs]))
                    pos = int(logits[0, :len(obs['legal_actions'])].argmax())
                    obs, _, term, cut, _ = worker.submit('step', obs['legal_actions'][pos]['index']).result()
                    steps += 1
                    if steps > 4096:
                        raise RuntimeError('Bounded diagnostic exhausted its decision limit')
                else:
                    results.append({'seed': task.seed, 'no_eligible_shop': True})
            except SimulatorError as exc:
                results.append({'seed': task.seed, 'fault': str(exc), 'fault_record_path': worker.worker.last_fault_path})
    finally:
        worker.close()
    report = {'schema': 'sts_removal_target_inspection_v1', 'run_record': record,
              'requested': args.n, 'worlds': results,
              'target_counts': dict(Counter(r['chosen'] for r in results if 'chosen' in r)),
              'faults': sum('fault' in r for r in results),
              'scope': 'Policy target predictions after a forced first affordable removal purchase. No removal target executed and no post-intervention game outcome measured.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    with gzip.open(str(out) + '.examples.json.gz', 'wt', compresslevel=1) as sink:
        json.dump(examples, sink)
    print(json.dumps({k: report[k] for k in ('requested', 'target_counts', 'faults')}))


if __name__ == '__main__':
    main()
