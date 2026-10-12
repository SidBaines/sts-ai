from __future__ import annotations
import copy
import unittest
import torch
from scripts import diagnose_phase_forgetting as diagnostic
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic
from sts_policy.encoding import ENCODER_VERSION
from sts_policy.training import CHECKPOINT_VERSION


class PhaseCrossoverTests(unittest.TestCase):
    def payload(self,seed):
        torch.manual_seed(seed)
        model=CompositeActorCritic(CandidateActorCritic(size='small',context='none'),CandidateActorCritic(size='small',context='none'))
        return {'format_version':CHECKPOINT_VERSION,'model_config':model.metadata(),'model_state':model.state_dict(),'metadata':{'encoder_version':ENCODER_VERSION},'optimizer_state':{'unused':1},'step':0}

    def test_complete_submodel_tensor_swap_and_optimizer_removed(self):
        old,new=self.payload(1),self.payload(2)
        for combat,ooc in (('O','O'),('N','N'),('O','N'),('N','O')):
            result=diagnostic.phase_swap(old,new,combat,ooc)
            diagnostic.verify_phase_payload(result,old,new,combat,ooc)
            self.assertIsNone(result['optimizer_state'])
            for key,tensor in result['model_state'].items():
                expected={'O':old,'N':new}[combat if key.startswith('combat.') else ooc]['model_state'][key]
                self.assertTrue(torch.equal(tensor,expected),key)
                self.assertNotEqual(tensor.data_ptr(),expected.data_ptr())
        self.assertEqual(old['optimizer_state'],{'unused':1})

    def test_cached_wrong_tensor_or_shared_routing_state_rejected(self):
        old,new=self.payload(3),self.payload(4);result=diagnostic.phase_swap(old,new,'O','N')
        result['model_state']['combat.critic.0.weight'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'differs from donor'):diagnostic.verify_phase_payload(result,old,new,'O','N')
        old['model_state']['shared.weight']=torch.zeros(1);new['model_state']['shared.weight']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'Unexpected shared'):diagnostic.phase_swap(old,new,'O','N')

    def test_common_valid_explicit_fault_flags_cuts_and_nonfinite_excluded(self):
        def episode(floor=20,act=2,outcome='defeat'):return {'floor':floor,'act':act,'max_act_reached':act,'outcome':outcome}
        rows={c:{i:episode() for i in 'abcdef'} for c in ('OO','NN','ON','NO')}
        rows['OO']['a']=episode(51,3,'victory');rows['NO']['a']=episode(51,3,'victory')
        rows['ON']['a']=episode(20,2);rows['NN']['a']=episode(12,1)
        rows['NN']['c']['truncated']=True
        rows['NO']['d']['fault']=True
        rows['ON']['e']['termination_reason']='collector_budget'
        rows['OO']['f']['floor']=float('nan')
        results=diagnostic.paired_crossover_statistics(rows,{'ON':{'b'}})
        self.assertEqual(results['common_valid'],1);self.assertEqual(results['common_task_ids'],['a'])
        effect=results['contrasts']['ooc_given_original_combat']
        self.assertEqual(effect['outcomes']['victory']['losses'],1)
        self.assertEqual(effect['outcomes']['act1_clear']['net'],0)
        self.assertEqual(effect['outcomes']['act2_clear']['net'],-1)
        self.assertEqual(effect['mean_floor_delta'],-31)

    def test_no_common_valid_worlds_not_divided_or_reported_as_zero_effect(self):
        rows={c:{'a':{'floor':10,'act':1,'outcome':'defeat'}} for c in ('OO','NN','ON','NO')}
        result=diagnostic.paired_crossover_statistics(rows,{'OO':{'a'}})
        self.assertFalse(result['classifiable']);self.assertEqual(result['common_valid'],0)
        self.assertNotIn('mean_floor',result)


if __name__=='__main__':unittest.main()
