#!/usr/bin/env python3
"""Replay recorded games and record every card offer: rewards and shops.

    extract_rewards.py <games/<arm> folder> <champion.pt> <processes> <out.jsonl>

One line per world: the card-reward decisions (options with type and rarity,
the policy's probability for each option at the moment of choice, and what was
chosen), the shop card offers and purchases, and the deck at each reward.
Run with the frozen study runtime and STS_LIGHTSPEED_BUILD set.
"""
import glob
import gzip
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'src'
_POLICY = None
CARD = re.compile(r'^take card (.+?) \[(\w+), (\w+)\]$')
BUY = re.compile(r'^buy card (.+?) \[(\w+), (\w+)\]')
DECK = re.compile(r'^Deck: \((\d+)\): \{(.*)\}$', re.MULTILINE)
SKIP = 'skip card reward'


def _init(champion):
    sys.path.insert(0, str(SRC))
    import torch
    torch.set_num_threads(1)
    from sts_rl.fresh_study import StudyPolicy
    global _POLICY
    _POLICY = StudyPolicy(champion)


def deck_of(observation):
    match = DECK.search(observation.get('text', ''))
    if not match:
        return None
    return [name for name in match[2].split(',') if name]


def option(action):
    match = CARD.match(action['description'])
    if match:
        return {'index': action['index'], 'card': match[1], 'type': match[2], 'rarity': match[3]}
    if action['description'] == SKIP:
        return {'index': action['index'], 'card': SKIP, 'type': 'Skip', 'rarity': ''}
    return None


def replay(path):
    from sts_rl.deck_oracle import episode, run_spec
    record = json.loads(gzip.open(path).read())
    started = time.perf_counter()
    out = {'seed': record['world_seed'], 'status': record['status'], 'win': record['win'],
           'floor': record['floor'], 'rewards': [], 'shops': []}
    ep = episode()
    observation, _ = ep.reset(run_spec(record['world_seed']))
    open_reward, shop = None, None
    for step, action in enumerate(record['action_prefix']):
        state, legal = observation['state'], observation['legal_actions']
        if observation['phase'] == 'out_of_combat':
            options = [o for o in (option(a) for a in legal) if o]
            if any(o['card'] == SKIP for o in options) and open_reward is None:
                open_reward = {'step': step, 'floor': state['floor'], 'act': state['act'],
                               'room': str(state.get('room')).rsplit('.', 1)[-1], 'hp': state['cur_hp'],
                               'max_hp': state['max_hp'], 'gold': state['gold'], 'deck': deck_of(observation),
                               'options': options}
            chosen = next(a['description'] for a in legal if a['index'] == action)
            if open_reward is not None and (CARD.match(chosen) or chosen in (SKIP, 'skip rewards / proceed')):
                indices, probs = _POLICY.distribution(observation)
                prob = dict(zip(indices, probs))
                open_reward['options'] = [{**o, 'prob': prob.get(o['index'])} for o in options]
                open_reward['chosen'] = SKIP if chosen.startswith('skip') else CARD.match(chosen)[1]
                open_reward['chosen_type'] = 'Skip' if chosen.startswith('skip') else CARD.match(chosen)[2]
                open_reward['proceeded'] = chosen == 'skip rewards / proceed'
                out['rewards'].append(open_reward)
                open_reward = None
            buys = [a for a in legal if BUY.match(a['description'])]
            if buys and shop is None:
                shop = {'step': step, 'floor': state['floor'], 'act': state['act'], 'gold': state['gold'],
                        'hp': state['cur_hp'], 'max_hp': state['max_hp'], 'deck': deck_of(observation),
                        'offers': [{'card': m[1], 'type': m[2], 'rarity': m[3], 'description': a['description']}
                                   for a in buys for m in [BUY.match(a['description'])]],
                        'bought': [], 'actions': []}
            if shop is not None:
                if chosen.startswith('buy ') or chosen == 'leave shop':
                    shop['actions'].append(chosen)
                if BUY.match(chosen):
                    shop['bought'].append(BUY.match(chosen)[1])
                if chosen == 'leave shop':
                    out['shops'].append(shop)
                    shop = None
        observation, *_ = ep.step(action)
    if open_reward is not None:
        open_reward.update(chosen=None, chosen_type=None, proceeded=None)
        out['rewards'].append(open_reward)
    if shop is not None:
        out['shops'].append(shop)
    out['final_deck'] = deck_of(observation)
    out['seconds'] = time.perf_counter() - started
    return out


def main():
    arm, champion, processes, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    paths = sorted(glob.glob(f'{arm}/shard_*/worlds/*.json.gz'), key=lambda p: int(Path(p).name.split('.')[0]))
    print(len(paths), 'worlds', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context('spawn')
    started = time.perf_counter()
    with context.Pool(processes, initializer=_init, initargs=(champion,)) as pool, out.open('x') as sink:
        for done, result in enumerate(pool.imap_unordered(replay, paths, chunksize=4), 1):
            sink.write(json.dumps(result) + '\n')
            if done % 256 == 0:
                sink.flush()
                print(f'{done}/{len(paths)} worlds, {time.perf_counter() - started:.0f} s', flush=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
