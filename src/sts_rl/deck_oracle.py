"""Deck-strength oracle: simulated boss fights from any out-of-combat decision.

Copy the game at an out-of-combat decision, enter this act's boss fight with
the current deck, relics and potions, and play it once per salt with a combat
policy. Fight setup and seed salting follow the fight panels
(sts_rl.fresh_panel.FreshPanelEpisode._scratch). The oracle reads simulator
state, so it is a teacher's tool, never a play-time input.
"""
from __future__ import annotations

from typing import Any, Callable

from sts_policy.curriculum import TaskSpec, player_hp
from sts_policy.workers import CurriculumEpisode

Choose = Callable[[dict[str, Any]], int]
VARIANTS = ('full', 'actual')

# The evaluator's worker settings; fast observations are output-identical.
EPISODE_CONFIG = {'gamma': 1.0, 'reward_mode': 'progress', 'terminal_hp_weight': 0.0,
                  'progress_floor_weight': 0.05, 'goal_act_clear': True,
                  'fast_observations': True, 'fast_replay': True}


def episode() -> CurriculumEpisode:
    return CurriculumEpisode(dict(EPISODE_CONFIG))


def run_spec(seed: int, prefix=()) -> TaskSpec:
    return TaskSpec(seed=seed, kind='run', prefix=tuple(prefix), horizon=4096, max_act=3,
                    battle_simulations=1000, seek_combat=False)


def boss_fraction(observation: dict[str, Any], previous: float) -> float:
    """HP fraction left on the boss (the enemy with the largest max HP)."""
    enemies = [e for e in observation['state'].get('combat', {}).get('enemies', []) if e.get('max_hp')]
    if not enemies:
        return previous
    boss = max(enemies, key=lambda e: e['max_hp'])
    return max(0, boss['cur_hp']) / boss['max_hp'] if boss.get('alive', True) else 0.0


def boss_fight(ep: CurriculumEpisode, raw: Any, start: Any, salt: int, choose: Choose,
               max_decisions: int = 600) -> dict[str, Any]:
    """One salted fight from a BATTLE-screen GameContext copy; all episode state is restored."""
    sts, wrapper, inner = raw.sts, ep.episode.env, ep.episode
    saved = [(obj, dict(vars(obj))) for obj in (raw, wrapper, inner)]
    original_exit = sts.BattleContext.exit_battle
    ended: dict[str, Any] = {}

    def record_exit(bc, context):
        ended.update(win=bc.outcome == sts.BattleOutcome.PLAYER_VICTORY, hp_after=int(bc.player_cur_hp))
        context.outcome = sts.GameOutcome.PLAYER_VICTORY if ended['win'] else sts.GameOutcome.PLAYER_LOSS
        context.screen_state = sts.ScreenState.MAP_SCREEN

    try:
        raw.gc, raw.bc = start.copy(), None
        raw.gc.seed = (int(start.seed) + 1_000_003 * (salt + 1)) % (1 << 63)
        raw._combat_recent_actions, raw._combat_history_turn = [], None
        inner.steps, inner.finished = 0, False
        for key, empty in (('_raw_history', []), ('_public_history', []), ('_selection_labels', {})):
            if key in vars(wrapper):
                setattr(wrapper, key, empty)
        sts.BattleContext.exit_battle = record_exit
        raw.advance_to_decision()
        observation = inner._observe()
        entry_hp, decisions, boss_left = player_hp(observation), 0, 1.0
        while not ended and observation['phase'] == 'combat' and decisions < max_decisions:
            boss_left = boss_fraction(observation, boss_left)
            observation, _, terminated, truncated, _ = inner.step(choose(observation))
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


def score_state(ep: CurriculumEpisode, seed: int, prefix, salts, choose: Choose,
                variants=VARIANTS) -> dict[str, Any]:
    """Boss fights from the out-of-combat decision reached by `prefix`.

    `salts` is an iterable of fight salts; reuse the same salts across options
    to compare them on common battle seeds. Variant 'full' sets HP to max HP
    (deck strength alone); 'actual' keeps the current HP.
    """
    observation, _ = ep.reset(run_spec(seed, prefix))
    if observation['phase'] != 'out_of_combat':
        raise ValueError('oracle moment must be an out-of-combat decision')
    raw = ep.episode.env._env
    base = raw.gc.copy()
    base.cur_room = raw.sts.Room.BOSS
    base.enter_battle(base.boss)
    out = {'hp': int(base.cur_hp), 'max_hp': int(base.max_hp), 'floor': int(base.floor_num),
           'deck_size': len(base.deck), 'boss': str(base.boss).rsplit('.', 1)[-1]}
    for variant in variants:
        start = base.copy()
        if variant == 'full':
            start.cur_hp = start.max_hp
        out[variant] = [boss_fight(ep, raw, start, salt, choose) for salt in salts]
    return out
