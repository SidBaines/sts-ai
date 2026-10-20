"""Deck oracle against the native simulator: valid fights, deterministic, state restored."""
import unittest

from sts_policy.curriculum import public_hash
from sts_rl.deck_oracle import episode, run_spec, score_state
from tests.support import requires_simulator

SEED = 91_001_000_000


def first_play_or_end(observation):
    """Deterministic stand-in policy: play the first card, else the first action."""
    legal = observation['legal_actions']
    return next((a['index'] for a in legal if a['description'].startswith('play ')), legal[0]['index'])


@requires_simulator
class DeckOracleTest(unittest.TestCase):
    def test_boss_fights_are_valid_repeatable_and_leave_the_game_unchanged(self):
        ep = episode()
        observation, _ = ep.reset(run_spec(SEED))
        prefix = []
        while observation['phase'] != 'out_of_combat' or not prefix:
            action = observation['legal_actions'][0]['index']
            prefix.append(action)
            observation, *_ = ep.step(action)
        first = score_state(ep, SEED, prefix, range(2), first_play_or_end)
        after = ep.episode._observe()
        second = score_state(ep, SEED, prefix, range(2), first_play_or_end)
        self.assertEqual(first, second)
        self.assertEqual(first['full'][0]['entry_hp'], first['max_hp'])
        self.assertEqual(first['actual'][0]['entry_hp'], first['hp'])
        for fight in first['full'] + first['actual']:
            self.assertIsNotNone(fight['win'])
            self.assertFalse(fight['ub'])
            self.assertTrue(0.0 <= fight['boss_damage'] <= 1.0)
        # The oracle copies the game; the live decision it was asked about is unchanged.
        ep.reset(run_spec(SEED, prefix))
        self.assertEqual(public_hash(ep.episode._observe()), public_hash(after))


if __name__ == '__main__':
    unittest.main()
