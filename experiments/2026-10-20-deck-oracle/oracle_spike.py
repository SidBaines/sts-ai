#!/usr/bin/env python3
"""Deck-strength oracle spike: replay recorded games and score decks with simulated boss fights.

For each recorded world that reached the Act 2 boss, replay its actions to three
moments (the first out-of-combat decision of Act 2, the first at floor 25 or
later, and the last one before the Act 2 boss fight). At each moment, copy the game, enter this act's boss
fight, and play it with the champion's combat network once per salt, at full
HP and at actual HP. Fight setup and seed salting follow the fight panels
(sts_rl.fresh_panel.FreshPanelEpisode._scratch).

    oracle_spike.py <games/<arm> folder> <champion.pt> <worlds> <salts> <processes> <out.jsonl>

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


def _init(champion):
    sys.path.insert(0, str(SRC))
    import torch
    torch.set_num_threads(1)
    from sts_rl.fresh_study import StudyPolicy
    global _POLICY
    _POLICY = StudyPolicy(champion)


def _episode():
    from sts_policy.workers import CurriculumEpisode
    # The evaluator's worker settings; fast observations are output-identical.
    return CurriculumEpisode({'gamma': 1.0, 'reward_mode': 'progress', 'terminal_hp_weight': 0.0,
                              'progress_floor_weight': 0.05, 'goal_act_clear': True,
                              'fast_observations': True, 'fast_replay': True})


def _spec(seed, prefix=()):
    from sts_policy.curriculum import TaskSpec
    return TaskSpec(seed=seed, kind='run', prefix=tuple(prefix), horizon=4096, max_act=3,
                    battle_simulations=1000, seek_combat=False)


def _fight(ep, raw, base, salt, max_decisions=600):
    """One salted boss fight from a prepared BATTLE-screen GameContext; state restored after."""
    from sts_policy.curriculum import player_hp
    sts, wrapper, episode = raw.sts, ep.episode.env, ep.episode
    saved = [(obj, dict(vars(obj))) for obj in (raw, wrapper, episode)]
    original_exit = sts.BattleContext.exit_battle
    ended = {}

    def record_exit(bc, context):
        ended.update(win=bc.outcome == sts.BattleOutcome.PLAYER_VICTORY, hp_after=int(bc.player_cur_hp))
        context.outcome = sts.GameOutcome.PLAYER_VICTORY if ended['win'] else sts.GameOutcome.PLAYER_LOSS
        context.screen_state = sts.ScreenState.MAP_SCREEN

    try:
        raw.gc, raw.bc = base.copy(), None
        raw.gc.seed = (int(base.seed) + 1_000_003 * (salt + 1)) % (1 << 63)
        raw._combat_recent_actions, raw._combat_history_turn = [], None
        episode.steps, episode.finished = 0, False
        for key, empty in (('_raw_history', []), ('_public_history', []), ('_selection_labels', {})):
            if key in vars(wrapper):
                setattr(wrapper, key, empty)
        sts.BattleContext.exit_battle = record_exit
        raw.advance_to_decision()
        observation = episode._observe()
        entry_hp, decisions, boss_left = player_hp(observation), 0, 1.0
        while not ended and observation['phase'] == 'combat' and decisions < max_decisions:
            boss_left = _boss_fraction(observation, boss_left)
            observation, _, terminated, truncated, _ = episode.step(_POLICY.choose_action(observation))
            decisions += 1
            if terminated or truncated:
                break
        ub = bool(raw._undefined_behavior_evoked or (raw.bc is not None and raw.bc.undefined_behavior_evoked))
        valid = bool(ended) and not ub
        return {'salt': salt, 'win': ended.get('win') if valid else None, 'entry_hp': entry_hp,
                'boss_damage': (1.0 if ended['win'] else 1.0 - boss_left) if valid else None,
                'hp_lost': (entry_hp - (ended['hp_after'] if ended['win'] else 0)) if valid else None,
                'decisions': decisions, 'cut': not ended, 'ub': ub}
    finally:
        sts.BattleContext.exit_battle = original_exit
        for obj, state in saved:
            vars(obj).clear()
            vars(obj).update(state)


def _boss_fraction(observation, previous):
    """HP fraction left on the boss (the enemy with the largest max HP), from the last decision seen."""
    enemies = [e for e in observation['state'].get('combat', {}).get('enemies', []) if e.get('max_hp')]
    if not enemies:
        return previous
    boss = max(enemies, key=lambda e: e['max_hp'])
    return max(0, boss['cur_hp']) / boss['max_hp'] if boss.get('alive', True) else 0.0


def _oracle(ep, seed, prefix, salts):
    """Boss fights from the decision reached by `prefix`, at full and actual HP."""
    observation, _ = ep.reset(_spec(seed, prefix))
    if observation['phase'] != 'out_of_combat':
        raise RuntimeError('oracle moment must be an out-of-combat decision')
    raw = ep.episode.env._env
    base = raw.gc.copy()
    base.cur_room = raw.sts.Room.BOSS
    base.enter_battle(base.boss)
    out = {'hp': int(base.cur_hp), 'max_hp': int(base.max_hp), 'floor': int(base.floor_num),
           'deck_size': len(base.deck), 'boss': str(base.boss).rsplit('.', 1)[-1]}
    for variant in ('full', 'actual'):
        start = base.copy()
        if variant == 'full':
            start.cur_hp = start.max_hp
        out[variant] = [_fight(ep, raw, start, salt) for salt in range(salts)]
    return out


def score_world(job):
    path, salts = job
    from sts_env.env import SimulatorError
    from sts_policy.curriculum import public_hash
    record = json.loads(gzip.open(path).read())
    seed, actions = record['world_seed'], record['action_prefix']
    started = time.perf_counter()
    try:
        ep = _episode()
        observation, _ = ep.reset(_spec(seed))
        act2_start = mid_act2 = before_boss = None
        for i, action in enumerate(actions):
            state = observation['state']
            if observation['phase'] == 'out_of_combat' and state['act'] == 2:
                if act2_start is None:
                    act2_start = i
                if mid_act2 is None and state['floor'] >= 25:
                    mid_act2 = i
                before_boss = i
            if observation['phase'] == 'combat' and state['act'] == 2 and str(state.get('room')).endswith('BOSS'):
                break
            observation, *_ = ep.step(action)
        else:
            raise RuntimeError('replay never reached the Act 2 boss')
        moments = {'act2_start': act2_start, 'mid_act2': mid_act2, 'before_boss': before_boss}
        result = {'seed': seed, 'pass2': record['bosses']['2']['pass'], 'win': record['win'],
                  'entry_hp': record['bosses']['2']['entry_hp'], 'entry_max_hp': record['bosses']['2']['entry_max_hp'],
                  'moments': moments}
        # Check the replay reproduces the recorded game before trusting its states.
        full = _episode()
        full.reset(_spec(seed, actions[:-1]))
        observation, *_ = full.step(actions[-1])
        result['replay_matches'] = public_hash(observation) == record['terminal_hash']
        for name, index in moments.items():
            if index is not None:
                result[name] = _oracle(ep, seed, actions[:index], salts)
    except (SimulatorError, RuntimeError) as exc:
        result = {'seed': seed, 'error': f'{type(exc).__name__}: {exc}'}
    result['seconds'] = time.perf_counter() - started
    return result


def main():
    arm, champion, worlds, salts, processes, out = sys.argv[1:7]
    worlds, salts, processes, out = int(worlds), int(salts), int(processes), Path(out)
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    paths = []
    for path in sorted(glob.glob(f'{arm}/shard_*/worlds/*.json.gz'), key=lambda p: int(Path(p).name.split('.')[0])):
        record = json.loads(gzip.open(path).read())
        if record['status'] == 'completed' and record['bosses'].get('2', {}).get('reach'):
            paths.append(path)
        if len(paths) == worlds:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context('spawn')
    started = time.perf_counter()
    with context.Pool(processes, initializer=_init, initargs=(champion,)) as pool, out.open('x') as sink:
        for done, result in enumerate(pool.imap_unordered(score_world, [(p, salts) for p in paths]), 1):
            sink.write(json.dumps(result) + '\n')
            sink.flush()
            if done % 10 == 0:
                print(f'{done}/{len(paths)} worlds, {time.perf_counter() - started:.0f} s', flush=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
