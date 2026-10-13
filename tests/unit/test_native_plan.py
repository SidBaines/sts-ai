from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from sts_policy.curriculum import TaskSpec
from sts_policy.native_plan import NativePlanEpisode
from tests.unit.test_public_entities import observation


class FakeEnv:
    def __init__(self):
        self.queries=0
        self.descriptions={11:"first",22:"second"}
        self.bc=SimpleNamespace(outcome="Outcome.UNDECIDED",legal_actions=self.legal_actions)

    def legal_actions(self,include_card_type=True):
        def action(bits):
            return SimpleNamespace(bits=bits,describe=lambda battle,include_card_type=True:self.descriptions[bits])
        return [action(11),action(22)]

    def display_index_for_bits(self,bits):
        return {11:0,22:1}[bits]

    def search_best_combat_action(self,simulations,search_seed=None):
        self.queries+=1
        return {"bits":11,"winning_sequence_found":True,"selection_method":"winning_sequence",
            "best_sequence":[{"bits":11,"turn":1,"description":"first"},
                             {"bits":22,"turn":1,"description":"second"}]}


class NativePlanTests(unittest.TestCase):
    def episode(self):
        episode=NativePlanEpisode({});episode.stats={key:0 for key in
            ("search_queries","winning_plans","cache_hits","invalidations","exhausted_in_combat","winning_plan_defeats")}
        episode.spec=TaskSpec(seed=3);episode.finished=False
        episode.observation=observation();episode.observation["state"]["combat"]["turn"]=1
        episode.observation["legal_actions"]=[{"index":0,"description":"first"},{"index":1,"description":"second"}]
        episode.episode=SimpleNamespace(env=FakeEnv())
        return episode

    def test_repeated_query_is_pure_and_next_action_reuses_plan(self):
        episode=self.episode()
        self.assertEqual(episode.teacher_action(1000),0)
        self.assertEqual(episode.teacher_action(1000),0)
        self.assertEqual(episode.episode.env.queries,1)
        # Advance using exactly the recommendation; the new state still has
        # the planned second legal action and expected turn/description.
        result=(episode.observation,0.,False,False,{})
        with patch("sts_policy.workers.CurriculumEpisode.step",return_value=result):
            episode.step(0)
        self.assertEqual(episode.teacher_action(1000),1)
        self.assertEqual(episode.episode.env.queries,1)
        self.assertEqual(episode.stats["cache_hits"],1)

    def test_turn_description_or_settings_change_invalidates_cache(self):
        for changed in ("turn","description","budget"):
            episode=self.episode();episode.teacher_action(1000);episode.pending_index=None
            if changed=="turn":episode.observation["state"]["combat"]["turn"]=2
            if changed=="description":episode.episode.env.descriptions[22]="different"
            episode.teacher_action(500 if changed=="budget" else 1000)
            self.assertEqual(episode.episode.env.queries,2)
            self.assertEqual(episode.stats["invalidations"],1)

    def test_deviation_clears_plan_and_new_stats_contain_no_cache_contents(self):
        episode=self.episode();episode.teacher_action(1000)
        result=(deepcopy(episode.observation),0.,False,False,{})
        with patch("sts_policy.workers.CurriculumEpisode.step",return_value=result):
            returned=episode.step(1)
        self.assertFalse(episode.plan)
        self.assertFalse(episode.known_winning_plan)
        self.assertEqual(returned[4]["native_plan_stats"]["invalidations"],1)
        self.assertTrue(all(isinstance(value,(int,str)) for value in returned[4]["native_plan_stats"].values()))

    def test_public_presentation_labels_do_not_invalidate_native_identity(self):
        episode=self.episode();episode.teacher_action(1000);episode.pending_index=None
        episode.observation["legal_actions"][1]["description"]="second (deal 6 damage)"
        self.assertEqual(episode.teacher_action(1000),1)
        self.assertEqual(episode.episode.env.queries,1)
        self.assertEqual(episode.stats["invalidations"],0)

    def test_predicted_winning_plan_defeat_is_visible_and_boundary_clears_cache(self):
        episode=self.episode();episode.teacher_action(1000)
        episode.episode.env.bc.outcome="Outcome.PLAYER_LOSS"
        end=observation(False)
        with patch("sts_policy.workers.CurriculumEpisode.step",return_value=(end,-1.,True,False,{})):
            result=episode.step(0)
        self.assertEqual(result[4]["native_plan_stats"]["winning_plan_defeats"],1)
        self.assertFalse(episode.plan)


if __name__ == "__main__":
    unittest.main()
