from __future__ import annotations

import unittest

from sts_policy.rollout_training import Transition
from sts_rl.combat_trainer import combat_potential, shape_episodes


def obs(player_hp, enemy_hp, enemy_max=100, max_hp=80):
    return {"phase": "combat", "legal_actions": [{"index": 0}, {"index": 1}],
            "state": {"combat": {"player_cur_hp": player_hp, "player_max_hp": max_hp,
                                 "enemies": [{"max_hp": enemy_max, "cur_hp": enemy_hp, "alive": enemy_hp > 0}]}}}


def fight(states, seed, *, terminal_reward):
    out = []
    for i, (hp, ehp) in enumerate(states):
        last = i == len(states) - 1
        out.append(Transition(obs(hp, ehp), 0, 0, -.69, 0.0, [True, True], terminal_reward if last else 0.0,
                              last, False, 0.0, seed))
    return out


class CombatShapingTests(unittest.TestCase):
    def test_monte_carlo_return_telescopes_to_final_minus_initial_potential(self):
        win = fight([(80, 100), (70, 60), (65, 10)], 1, terminal_reward=1.0)
        loss = fight([(30, 100), (12, 70), (3, 40)], 2, terminal_reward=-1.0)
        episodes = [{"world_seed": 1, "steps": 3, "task_outcome": "battle_victory"},
                    {"world_seed": 2, "steps": 3, "task_outcome": "battle_defeat"}]
        shape_episodes(win + loss, episodes, hp_weight=0.3, enemy_weight=0.5, gae_lambda=1.0)
        # Win: +1 + phi(final: hp 65, no enemy HP) - phi(start).
        self.assertAlmostEqual(win[0].returns, 1.0 + 0.3 * 65 / 80 - combat_potential(win[0].observation, .3, .5), places=6)
        # Loss: -1 + phi(final: no player HP, enemy 40/100) - phi(start).
        self.assertAlmostEqual(loss[0].returns, -1.0 - 0.5 * 0.40 - combat_potential(loss[0].observation, .3, .5), places=6)
        # Dealing damage is credited locally even inside a lost fight.
        self.assertGreater(loss[1].advantage - loss[2].advantage, -1.0)

    def test_misaligned_episodes_are_rejected(self):
        a = fight([(80, 100), (70, 0)], 1, terminal_reward=1.0)
        with self.assertRaises(RuntimeError):
            shape_episodes(a, [{"world_seed": 9, "steps": 2, "task_outcome": "battle_victory"}],
                           hp_weight=.3, enemy_weight=.5, gae_lambda=.95)


if __name__ == "__main__":
    unittest.main()
