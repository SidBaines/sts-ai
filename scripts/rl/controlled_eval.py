#!/usr/bin/env python3
"""Fresh receipted full games with original or controlled replacements."""
import json

from fresh_collect import parser
from sts_rl.controlled_policy import policy_factory
from sts_rl.fresh_collection import run_collection, validate_config
from sts_rl.fresh_cohort import validate_cohorts


def main(argv=None):
    config = vars(parser().parse_args(argv))
    validate_config(config)
    if config['mode'] != 'eval':
        raise ValueError('controlled_eval requires --mode eval')
    if not 91_000_000_000 <= config['world_start'] < config['world_start'] + config['n'] <= 91_200_000_000:
        raise ValueError('Controlled games require the development evaluation namespace')
    policy = policy_factory(config['checkpoint'], config['combat'], config['ooc'])
    summary = run_collection(config, policy)  # original receipts, AC, finite RPCs, STOP/deadline
    if summary.get('cohort_complete') is True:
        validate_cohorts([config['out']], world_start=config['world_start'], n=config['n'])
    print(json.dumps({k: summary[k] for k in ('event', 'status', 'cohort_complete', 'completed',
                                             'faults', 'cuts', 'censored', 'unstarted', 'wins')}))


if __name__ == '__main__':
    main()
