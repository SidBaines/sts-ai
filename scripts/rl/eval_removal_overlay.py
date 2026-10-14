#!/usr/bin/env python3
"""Measure one early affordable removal; the existing policy selects its target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def has_unupgraded_strike(obs):
    match = re.search(r'^Deck:.*?\{([^}]*)\}', str(obs.get('text', '')), re.MULTILINE)
    return bool(match and 'Strike' in [name.strip() for name in match.group(1).split(',')])


def unupgraded_strike_choices(obs):
    return [a for a in obs['legal_actions'] if re.match(r'^select card index \d+: Strike \[', a['description'])]


def main():
    import torch
    from sts_rl.common import _MappedFuture, make_workers, close_workers, load_policy
    from sts_rl.evaluate import evaluate, paired, paired_wins
    from sts_rl.recording import begin_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--world-start', type=int, default=5800000000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-floor', type=int, default=33)
    parser.add_argument('--target', choices=['policy', 'strike'], default='policy',
                        help='strike buys only with a visible unupgraded Strike and forces its selection')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() or args.n < 1 or args.workers < 1:
        parser.error('Fresh output and positive counts required')
    record = begin_run(str(out) + '.record', vars(args), args.checkpoint, inputs=[args.baseline])
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()

    class Worker:
        def __init__(self, inner):
            self.worker = inner
            self.used = False
            self.pending = False
            self.restricted = 0
            self.forced_targets = 0

        @property
        def last_fault_path(self):
            return getattr(self.worker.worker, 'last_fault_path', None)

        def map(self, result):
            obs = result[0]
            state = obs.get('state', {})
            if self.pending:
                if 'Choose a card to REMOVE' not in obs.get('text', ''):
                    raise RuntimeError('Expected the purchased removal selection; do not execute an unrelated action')
                kept = unupgraded_strike_choices(obs)
                if not kept:
                    raise RuntimeError('Purchased removal has no expected unupgraded Strike; reject rather than remove another card')
                self.pending = False
                self.forced_targets += 1
                return ({**obs, 'legal_actions': kept}, *result[1:])
            if not self.used and obs.get('phase') != 'combat' and state.get('floor', 999) <= args.max_floor:
                legal = obs['legal_actions']
                kept = [a for a in legal if a['description'].startswith('buy card remove')]
                if args.target == 'strike' and not has_unupgraded_strike(obs):
                    kept = []
                if kept:
                    self.used = True
                    self.pending = args.target == 'strike'
                    self.restricted += 1
                    obs = {**obs, 'legal_actions': kept}
                    result = (obs, *result[1:])
            return result

        def submit(self, method, *arguments):
            if method == 'reset':
                self.used = False
                self.pending = False
            future = self.worker.submit(method, *arguments)
            return _MappedFuture(future, self.map) if method in {'reset', 'step'} else future

        def close(self):
            self.worker.close()

    workers = [Worker(w) for w in make_workers(args.workers, dead_action_filter=True, slim=True)]
    try:
        result = evaluate(model, workers, n=args.n, world_start=args.world_start, chunk=64)
    finally:
        close_workers(workers)
    baseline = json.loads(Path(args.baseline).read_text())
    result.update(run_record=record, overlay=('one_early_affordable_removal_v1' if args.target == 'policy' else 'one_early_unupgraded_strike_removal_v1'),
                  restricted_menus=sum(w.restricted for w in workers),
                  forced_target_menus=sum(w.forced_targets for w in workers),
                  paired=paired(baseline['records'], result['records']),
                  paired_wins=paired_wins(baseline['records'], result['records']),
                  scope=('At most one affordable removal purchase per world, by max-floor. Existing policy chooses the card to remove; no target-quality claim.' if args.target == 'policy' else
                         'At most one affordable removal by max-floor when a visible unupgraded Strike exists. Restrict the purchased removal selector to unupgraded Strikes; native indices preserved.'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + '\n')
    print(json.dumps({k: result[k] for k in ('summary', 'restricted_menus', 'paired', 'paired_wins')}))


if __name__ == '__main__':
    main()
