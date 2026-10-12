from __future__ import annotations
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None
if torch is not None:
    from sts_policy.losses import PPOConfig
    from sts_policy.rollout_training import Transition, update_ppo


@unittest.skipIf(torch is None,'optional torch unavailable')
class KLGuardTests(unittest.TestCase):
    def test_over_target_batch_never_updates_under_opt_in_guard(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.logits=torch.nn.Parameter(torch.tensor([3.,0.]));self.value=torch.nn.Parameter(torch.zeros(1))
        def forward(model,count):
            return model.logits[None].expand(count,-1),model.value.expand(count),torch.ones(count,2,dtype=torch.bool)
        observation={'legal_actions':[{'index':0},{'index':1}]}
        rows=[Transition(observation,0,0,-.69314718,0.,[True,True],1.,True,False,0.,1,1.,1.)]*4
        for before in (True,False):
            model=Model();initial=model.logits.detach().clone();optimizer=torch.optim.AdamW(model.parameters(),lr=.05)
            with patch.object(optimizer,'step',wraps=optimizer.step) as step:
                result=update_ppo(model,rows,collate=lambda batch,device:len(batch),forward=forward,
                    optimizer=optimizer,config=PPOConfig(epochs=1,minibatch_size=4,target_kl=.03,kl_guard_before_update=before))
            if before:
                step.assert_not_called();torch.testing.assert_close(model.logits,initial,rtol=0.,atol=0.)
                self.assertFalse(optimizer.state);self.assertEqual(result['steps'],0)
                self.assertEqual(result['rejected_minibatches'],1);self.assertGreater(result['rejected_kl'],.03)
                self.assertEqual(result['algorithm_version'],'masked_ppo_v2_before_update_kl')
            else:
                step.assert_called_once();self.assertEqual(result['steps'],1)
                self.assertFalse(torch.equal(model.logits,initial))
                self.assertEqual(result['algorithm_version'],'masked_ppo_v1_after_update_kl')
