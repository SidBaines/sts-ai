#!/usr/bin/env python3
"""Score every option at Act 2 card rewards with the deck oracle.

    option_test.py <games/<arm> folder> <champion.pt> <first seed> <decisions> <salts> <processes> <out.jsonl>

Run with the frozen study runtime and STS_LIGHTSPEED_BUILD set.
"""
import glob
import gzip
import json
import multiprocessing
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'src'
_POLICY = None
PER_WORLD = 2


def _init(champion):
    sys.path.insert(0, str(SRC))
    import torch
    torch.set_num_threads(1)
    from sts_rl.fresh_study import StudyPolicy
    global _POLICY
    _POLICY = StudyPolicy(champion)


def card_decisions(record):
    """Act 2 card rewards before the boss: (step, options, champion's choice) per reward screen."""
    from sts_rl.deck_oracle import episode, run_spec
    ep = episode()
    observation, _ = ep.reset(run_spec(record['world_seed']))
    actions, found, open_reward = record['action_prefix'], [], None
    for i, action in enumerate(actions):
        state, legal = observation['state'], observation['legal_actions']
        if observation['phase'] == 'combat' and state['act'] == 2 and str(state.get('room')).endswith('BOSS'):
            break
        cards = [a for a in legal if a['description'].startswith('take card ')]
        skip = [a for a in legal if a['description'] == 'skip card reward']
        if observation['phase'] == 'out_of_combat' and state['act'] == 2 and cards and skip and open_reward is None:
            open_reward = {'step': i, 'floor': state['floor'],
                           'options': [(a['index'], a['description']) for a in cards + skip]}
        if open_reward is not None:
            chosen = next(a['description'] for a in legal if a['index'] == action)
            if chosen.startswith('take card ') or chosen in ('skip card reward', 'skip rewards / proceed'):
                open_reward['champion'] = 'skip card reward' if chosen.startswith('skip') else chosen
                found.append(open_reward)
                open_reward = None
        observation, *_ = ep.step(action)
    return found


def score_decision(job):
    seed, prefix, decision, salts = job
    from sts_env.env import SimulatorError
    from sts_rl.deck_oracle import episode, score_state
    started = time.perf_counter()
    ep = episode()
    result = {**decision, 'seed': seed, 'scores': {}}
    try:
        for index, description in decision['options']:
            result['scores'][description] = score_state(ep, seed, prefix + [index], range(salts),
                                                        _POLICY.choose_action, variants=('full',))['full']
    except (SimulatorError, ValueError, RuntimeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = time.perf_counter() - started
    return result


def main():
    arm, champion, first_seed, decisions, salts, processes, out = sys.argv[1:8]
    first_seed, decisions, salts, processes, out = int(first_seed), int(decisions), int(salts), int(processes), Path(out)
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    sys.path.insert(0, str(SRC))
    jobs = []
    paths = sorted(glob.glob(f'{arm}/shard_*/worlds/*.json.gz'), key=lambda p: int(Path(p).name.split('.')[0]))
    for path in paths:
        record = json.loads(gzip.open(path).read())
        if record['world_seed'] < first_seed or record['status'] != 'completed' or not record['bosses'].get('2', {}).get('reach'):
            continue
        found = card_decisions(record)
        # Spread over the act: the first and the last card reward of Act 2.
        for decision in ([found[0], found[-1]] if len(found) > 1 else found)[:PER_WORLD]:
            jobs.append((record['world_seed'], record['action_prefix'][:decision['step']], decision, salts))
        if len(jobs) >= decisions:
            break
    jobs = jobs[:decisions]
    print(len(jobs), 'decisions from worlds', jobs[0][0], 'to', jobs[-1][0], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context('spawn')
    started = time.perf_counter()
    with context.Pool(processes, initializer=_init, initargs=(champion,)) as pool, out.open('x') as sink:
        for done, result in enumerate(pool.imap_unordered(score_decision, jobs), 1):
            sink.write(json.dumps(result) + '\n')
            sink.flush()
            if done % 10 == 0:
                print(f'{done}/{len(jobs)} decisions, {time.perf_counter() - started:.0f} s', flush=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
