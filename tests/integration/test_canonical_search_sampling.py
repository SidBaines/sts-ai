"""Actual native API regression on public-equivalent hidden-pile variants."""
import importlib.util
import os
from pathlib import Path
import unittest
from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.integration.test_public_combat_teacher import _advance_to_first_combat
from tests.support import requires_simulator

@requires_simulator
class CanonicalSearchSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('STS_SEARCH_TEST_FIXTURE')
        if not path:
            if os.environ.get('STS_REQUIRE_SEARCH_TEST_FIXTURE')=='1':
                raise RuntimeError('Canonical search regression fixture is required')
            raise unittest.SkipTest('Build the test-only hidden-pile fixture for this regression')
        spec=importlib.util.spec_from_file_location('sts_hidden_pile_fixture',path)
        cls.fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.fixture)

    def public(self,bc):
        return {'text':bc.describe_state(public_state=True,include_card_type=True),
                'cards':bc.public_cards(True),'player':bc.public_player(),
                'enemies':bc.enemies(),'relics':bc.public_relics(),'potions':bc.public_potions(),
                'legal_bits':[int(a.bits) for a in bc.legal_actions(True)]}

    def query(self,bc,sample):
        result=bc.search_best_action(100,search_seed=173+sample,draw_order_seed=sample,include_card_type=True)
        self.assertTrue(result['provenance']['hidden_pile_order_canonicalized'])
        return (int(result['bits']),result['selection_method'],
                [(int(e['bits']),int(e['visits']),e['evaluation_sum']) for e in result['root_edges']])

    def test_fixed_sample_queries_are_invariant_and_both_live_copies_unchanged(self):
        changed=0
        for seed in (3,7,11,19):
            with self.subTest(seed=seed):
                env=LightspeedHybridEnv(world_seed=seed,max_act=1,combat_control='llm',combat_observation='combat_public_v3')
                _advance_to_first_combat(env);left=env.bc;right=self.fixture.reversed_copy(left)
                original_order=self.fixture.draw_ids(left);reversed_order=self.fixture.draw_ids(right)
                changed+=original_order!=reversed_order
                self.assertEqual(sorted(original_order),sorted(reversed_order))
                public=self.public(left);self.assertEqual(public,self.public(right))
                for sample in range(1,25):self.assertEqual(self.query(left,sample),self.query(right,sample))
                self.assertEqual(self.public(left),public);self.assertEqual(self.public(right),public)
                self.assertEqual(self.fixture.draw_ids(left),original_order)
                self.assertEqual(self.fixture.draw_ids(right),reversed_order)
        self.assertGreater(changed,0,'Need at least one genuinely different hidden permutation')

    def test_unsampled_teacher_remains_explicitly_privileged(self):
        env=LightspeedHybridEnv(world_seed=3,max_act=1,combat_control='llm')
        _advance_to_first_combat(env);result=env.bc.search_best_action(8,search_seed=123)
        self.assertFalse(result['provenance']['hidden_pile_order_canonicalized'])
        self.assertTrue(result['provenance']['teacher_uses_full_simulator_state'])

    def test_discard_and_exhaust_root_choices_map_to_the_same_public_card(self):
        for exhaust in (False,True):
            with self.subTest(exhaust=exhaust):
                env=LightspeedHybridEnv(world_seed=3,max_act=1,combat_control='llm')
                _advance_to_first_combat(env)
                left=self.fixture.pile_selection_copy(env.bc,exhaust)
                right=self.fixture.reversed_copy(left)
                # Selection menus expose arbitrary routing order. These are
                # equal card multisets, rather than byte-identical observations.
                # Compare semantic choices and preserve each copy's full view.
                before_left=self.public(left);before_right=self.public(right)
                def semantic_view(bc):
                    view=self.public(bc)
                    view['text']=view['text'].split('Selection candidates:')[0]
                    view['cards']['selection']['candidates']=sorted(view['cards']['selection']['candidates'])
                    view.pop('legal_bits')
                    view['option_labels']=sorted(a.describe(bc,include_card_type=True) for a in bc.legal_actions(True))
                    return view
                self.assertEqual(semantic_view(left),semantic_view(right))
                for sample in range(1,9):
                    a=left.search_best_action(32,search_seed=100+sample,draw_order_seed=sample,include_card_type=True)
                    b=right.search_best_action(32,search_seed=100+sample,draw_order_seed=sample,include_card_type=True)
                    self.assertEqual(a['description'],b['description'])
                    self.assertTrue(a['action'].is_valid(left));self.assertTrue(b['action'].is_valid(right))
                    edges=lambda r:sorted((e['description'],int(e['visits']),e['evaluation_sum']) for e in r['root_edges'])
                    self.assertEqual(edges(a),edges(b))
                    self.assertEqual(self.public(left),before_left)
                    self.assertEqual(self.public(right),before_right)
                    # Execute only on throwaway clones. Wrong raw-index routing
                    # would return/exhume different cards despite equal queries.
                    acopy=self.fixture.reversed_copy(self.fixture.reversed_copy(left))
                    bcopy=self.fixture.reversed_copy(self.fixture.reversed_copy(right))
                    a['action'].execute(acopy);b['action'].execute(bcopy)
                    self.assertEqual(self.public(acopy),self.public(bcopy))
