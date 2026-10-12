"""Bounded arithmetic checks against the campaign reward implementation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from sts_policy.curriculum import reward_components


def episode_return(floor: int, win: bool) -> float:
    previous = {"state": {"floor": 0, "cur_hp": 80, "max_hp": 80}}
    endpoint = {"state": {"floor": floor, "cur_hp": 40 if win else 0, "max_hp": 80}}
    components = reward_components(previous, endpoint, task_outcome="victory" if win else "defeat",
                                   terminated=True, mode="progress", gamma=1.,
                                   progress_floor_weight=.05, terminal_hp_weight=0.)
    return sum(components.values())


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * ((p * (1 - p) + z * z / (4 * trials)) / trials) ** .5 / denominator
    return [max(0., center - half), min(1., center + half)]


def main() -> None:
    result = {
        "purpose": "Synthetic objective counterexample, not an observed pair of policies",
        "return_formula_for_complete_nonfault_fresh_starts": "0.05 * final_floor + 2 * victory - 1",
        "always_die_at_floor_50": {"win_probability": 0., "expected_return": episode_return(50, False)},
        "win_40_percent_else_die_at_floor_10": {
            "win_probability": .4,
            "expected_return": .4 * episode_return(51, True) + .6 * episode_return(10, False)},
        "floor_increase_equivalent_to_win_probability_increase": {"2_floors": .05},
        "wilson_95_intervals": {
            "0_of_21": wilson(0, 21), "20_of_21": wilson(20, 21),
            "2_of_192": wilson(2, 192), "0_of_192": wilson(0, 192)},
        "interval_scope": "Per-world marginal descriptive intervals; not paired comparisons or adjustment for repeated DEV selection",
        "gae_direct_terminal_residual_weights": {str(n): .95 ** n for n in (20, 50, 100, 177, 200)},
    }
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/private/tmp/sts_reward_objective_probe.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
