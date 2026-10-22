"""Native public-search details preserve the original vote protocol."""
import json
from concurrent.futures import ThreadPoolExecutor
import math
import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from sts_policy.curriculum import TaskSpec, public_hash
from sts_rl.fresh_collection import search_plan
from sts_rl.fresh_panel import FreshPanelWorker
from sts_rl.teacher_votes import votes_from_details
from tests.support import requires_simulator


def _native_options():
    try:
        return hasattr(import_lightspeed(None), "set_search_options")
    except Exception:
        return False


requires_search_options = unittest.skipUnless(_native_options(), "native build without set_search_options")

DEFAULT_SEARCH_OPTIONS = {"evaluation": 0, "exploration": 3 * math.sqrt(2), "hp_weight": 1.,
                          "potion_weight": 4., "loss_credit": 1.}


@requires_simulator
class TeacherDetailsTests(unittest.TestCase):
    def test_details_match_votes_and_repeat_in_fresh_workers(self):
        for seed in (3, 7):
            with self.subTest(seed=seed):
                results = []
                for _ in range(2):
                    with FreshPanelWorker({}, timeout=30.) as worker:
                        observation, info = worker.reset(TaskSpec(seed=seed, kind="battle", battle_simulations=16))
                        self.assertEqual(observation["phase"], "combat")
                        plan = search_plan(info["public_hash"], samples=4)
                        votes = worker.search_votes_public(16, plan["samples"], plan["search_seed"])
                        details = worker.search_details_public(16, plan["samples"],
                                                               plan["effective_search_seeds"], plan["search_seed"])
                        self.assertEqual(votes_from_details(details, "best_line"), votes)
                        self.assertEqual(sum(votes.values()), 4)
                        self.assertEqual(details, json.loads(json.dumps(details, allow_nan=False)))
                        results.append(details)
                self.assertEqual(*results)

    @requires_search_options
    def test_options_defaults_visit_changes_and_no_leakage(self):
        changed, positions = 0, 0
        for seed in (3, 7, 11):
            with FreshPanelWorker({}, timeout=30.) as worker:
                observation, info = worker.reset(TaskSpec(seed=seed, kind="battle", battle_simulations=16))
                for _ in range(2):
                    self.assertEqual(observation["phase"], "combat")
                    plan = search_plan(public_hash(observation), samples=4)
                    args = (256, plan["samples"], plan["effective_search_seeds"], plan["search_seed"])
                    before = worker.search_details_public(*args)
                    self.assertEqual(before, worker.search_details_public(*args, options={}))
                    self.assertEqual(before, worker.search_details_public(*args, options=DEFAULT_SEARCH_OPTIONS))
                    votes = worker.search_votes_public(256, plan["samples"], plan["search_seed"])
                    self.assertEqual(votes_from_details(before, "best_line"), votes)
                    alternate = worker.search_details_public(*args, options={"evaluation": 1, "exploration": .05,
                                                                            "hp_weight": 0})
                    visits = lambda details: [[(edge["bits"], edge["visits"]) for edge in row["edges"]] for row in details]
                    changed += visits(before) != visits(alternate)
                    self.assertEqual(before, worker.search_details_public(*args))
                    positions += 1
                    observation, _, terminated, truncated, info = worker.step(before[0]["chosen"])
                    if terminated or truncated or observation["phase"] != "combat":
                        break
        self.assertEqual(positions, 6)
        self.assertGreater(changed, 0)
        print(f"Search options: root visits changed at {changed}/{positions} positions; defaults identical, no leakage.")

    @requires_search_options
    def test_native_options_validate_atomically_and_are_thread_local(self):
        native = import_lightspeed()
        self.addCleanup(native.reset_search_options)
        native.reset_search_options()
        self.assertEqual(native.get_search_options(), DEFAULT_SEARCH_OPTIONS)
        native.set_search_options(evaluation=1, exploration=.05, hp_weight=0)
        customized = native.get_search_options()
        invalid = [{"evaluation": value} for value in (-1, 2, 1., True, "1", None, 10**1000)]
        for key in ("exploration", "hp_weight", "potion_weight", "loss_credit"):
            invalid.extend({key: value} for value in (-1, math.nan, math.inf, -math.inf, True, "1", None, 10**1000))
        invalid.append({"exploration": 0})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                native.set_search_options(**options)
            self.assertEqual(native.get_search_options(), customized)
        def other_thread():
            self.assertEqual(native.get_search_options(), DEFAULT_SEARCH_OPTIONS)
            native.set_search_options(potion_weight=0, loss_credit=0)
            self.assertEqual(native.get_search_options()["potion_weight"], 0)
            native.reset_search_options()
            return native.get_search_options()
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.assertEqual(executor.submit(other_thread).result(), DEFAULT_SEARCH_OPTIONS)
        self.assertEqual(native.get_search_options(), customized)
        native.set_search_options()
        self.assertEqual(native.get_search_options(), DEFAULT_SEARCH_OPTIONS)

    @requires_search_options
    def test_win_first_constant_scores_still_explore(self):
        native = import_lightspeed()
        self.addCleanup(native.reset_search_options)
        gc = native.GameContext(native.CharacterClass.IRONCLAD, 3, 0)
        for index in range(len(gc.deck) - 1, -1, -1):
            gc.remove_card(index)
        for _ in range(10):
            gc.obtain_card(native.Card(native.CardId.DEFEND_RED))
        gc.cur_hp = 1
        gc.enter_battle(native.MonsterEncounter.CULTIST)
        bc = native.BattleContext()
        bc.init(gc)
        native.set_search_options(evaluation=1, hp_weight=0, potion_weight=0, loss_credit=0)
        result = bc.search_best_action(64, search_seed=41, draw_order_seed=43)
        self.assertEqual((result["best_evaluation"], result["min_evaluation"]), (0., 0.))
        self.assertEqual(len(result["root_edges"]), 2)
        self.assertEqual([edge["visits"] for edge in result["root_edges"]], [32, 32])


if __name__ == "__main__":
    unittest.main()
