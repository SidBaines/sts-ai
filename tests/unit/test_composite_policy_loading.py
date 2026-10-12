from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from sts_policy.composition import compose_checkpoint
from sts_policy.encoding import ENCODER_VERSION
from sts_policy.losses import PPOConfig, masked_distribution
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import (CompositeActorCritic, ModelAgent, policy_from_config,
                                   policy_from_payload, collate_public, forward_policy)
from sts_policy.rollout_training import Transition, update_ppo
from sts_policy.training import file_identity, load_checkpoint, save_checkpoint
from tests.unit.test_policy_encoding import observation


class CompositePolicyLoadingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def save_candidate(self, path, *, provenance=None):
        model=CandidateActorCritic(context="none")
        metadata={"encoder_version":ENCODER_VERSION,"stage":"bc","data_identity":"fixture"}
        if provenance is not None: metadata["native_provenance"]=provenance
        save_checkpoint(path,model,model_config=model.metadata(),metadata=metadata)
        return model

    def test_legacy_candidate_and_composite_roundtrip_and_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'combat.pt';b=root/'ooc.pt';out=root/'composed.pt'
            combat=self.save_candidate(a);ooc=self.save_candidate(b)
            before=[file_identity(a),file_identity(b)]
            report=compose_checkpoint(a,b,out)
            model=policy_from_payload(load_checkpoint(out))
            self.assertIsInstance(model,CompositeActorCritic)
            self.assertIsInstance(policy_from_payload(load_checkpoint(a)),CandidateActorCritic)
            observations=[observation(),dict(observation(),phase="out_of_combat")]
            batch=collate_public(observations)
            reference=CompositeActorCritic(combat,ooc)
            for actual,expected in zip(model(*batch),reference(*batch)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            agent=ModelAgent(str(out),device="cpu")
            expected=model(*batch)[0].argmax(-1).tolist()
            self.assertEqual(agent.choose_actions(observations),expected)
            self.assertEqual(before,[file_identity(a),file_identity(b)])
            self.assertTrue(report['composition_provenance']['legacy_environment_provenance_missing'])
            with self.assertRaises(ValueError):compose_checkpoint(a,b,a)

    def test_model_kind_encoder_dimensions_and_provenance_are_strict(self):
        with self.assertRaises(ValueError):policy_from_config({'kind':'unknown'})
        with self.assertRaises(ValueError):policy_from_config({'state_dim':7})
        with self.assertRaises(ValueError):policy_from_config({'kind':'composite','combat':{}})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.pt';b=root/'b.pt'
            self.save_candidate(a,provenance={'module_sha256':'a'})
            self.save_candidate(b,provenance={'module_sha256':'b'})
            with self.assertRaisesRegex(ValueError,'provenance'):compose_checkpoint(a,b,root/'out.pt')
            payload=load_checkpoint(a);payload['metadata']['encoder_version']='old'
            with self.assertRaisesRegex(ValueError,'encoder'):policy_from_payload(payload)

    def test_recorded_training_environment_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.pt';b=root/'b.pt'
            model=CandidateActorCritic()
            for path,native in ((a,'native_a'),(b,'native_b')):
                save_checkpoint(path,model,model_config=model.metadata(),metadata={
                    'encoder_version':ENCODER_VERSION,'training_environment':{'native_module_sha256':native}})
            with self.assertRaisesRegex(ValueError,'training-environment'):compose_checkpoint(a,b,root/'out.pt')

    def test_genuine_cpu_ppo_updates_both_heads_and_optimizer_restores(self):
        model=CompositeActorCritic(CandidateActorCritic(),CandidateActorCritic())
        obs=[dict(observation(),phase='combat' if i%2 else 'out_of_combat') for i in range(8)]
        batch=collate_public(obs)
        with torch.no_grad():
            logits,values=model(*batch);logs=masked_distribution(logits,batch[2]).log_prob(torch.ones(8,dtype=torch.long))
        trajectories=[Transition(o,1,1,float(logs[i]),float(values[i]),[True]*4,1.,True,False,0.,i,
                                  advantage=1. if i%3 else -.5,returns=float(values[i])+1.) for i,o in enumerate(obs)]
        before={key:value.detach().clone() for key,value in model.state_dict().items()}
        result=update_ppo(model,trajectories,collate=collate_public,forward=forward_policy,
                          config=PPOConfig(epochs=1,minibatch_size=8,target_kl=10.),device='cpu')
        self.assertEqual(result['steps'],1)
        for head in ('combat.','ooc.'):
            self.assertTrue(any(not torch.equal(value,before[key]) for key,value in model.state_dict().items() if key.startswith(head)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ppo.pt'
            save_checkpoint(path,model,model_config=model.metadata(),optimizer=result['optimizer'],step=1,
                            metadata={'encoder_version':ENCODER_VERSION,'model_kind':'composite','data_identity':'fixture'})
            loaded=policy_from_payload(load_checkpoint(path));optimizer=torch.optim.AdamW(loaded.parameters())
            load_checkpoint(path,loaded,optimizer=optimizer,expected_data_identity='fixture')
            self.assertEqual(len(optimizer.state),len(result['optimizer'].state))
            for actual,expected in zip(loaded(*batch),model(*batch)):torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_training_cli_loads_composite_through_normal_checkpoint(self):
        from scripts.train_nonllm import source_model
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'composite.pt'
            model=CompositeActorCritic(CandidateActorCritic(),CandidateActorCritic())
            save_checkpoint(path,model,model_config=model.metadata(),metadata={'encoder_version':ENCODER_VERSION})
            loaded,payload=source_model(SimpleNamespace(seed=1,checkpoint=path,device='cpu'))
            self.assertIsInstance(loaded,CompositeActorCritic)
            self.assertEqual(payload['model_config'],model.metadata())
