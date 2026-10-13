from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import math
import torch

from scripts.train_combat_residual_v3 import prepare, validate_teacher_manifest, target_loss
from sts_policy.curriculum import world_split
from tests.unit.test_public_entities import observation


class CombatTrainingDataTests(unittest.TestCase):
    def test_consensus_loss_averages_group_mass_and_ignores_padding(self):
        logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
        mask=torch.tensor([[True,True,True]])
        groups=torch.tensor([[[True,True,False],[False,False,True],[False,False,False]]])
        weights=torch.tensor([[.7,.3,0.]])
        loss=target_loss(logits,mask,(groups,weights))
        self.assertAlmostEqual(float(loss.detach()),-.7*math.log(2/3)-.3*math.log(1/3),places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[0,0]),float(logits.grad[0,1]))
        with self.assertRaises(ValueError):
            target_loss(logits,mask,(groups,torch.tensor([[.7,.3,.2]])))
        with self.assertRaises(ValueError):
            target_loss(logits,mask,(groups,torch.tensor([[1.3,-.3,0.]])))

    def test_budget_override_selects_recorded_labels_and_fails_if_missing(self):
        seed=next(s for s in range(10,100) if world_split(s)=="train")
        record={"world_seed":seed,"observation":observation(),"teacher_index":7,
                "teacher_by_budget":{"200":7,"1000":4}}
        with TemporaryDirectory() as folder:
            path=Path(folder)/"teacher.jsonl"; path.write_text(json.dumps(record)+"\n")
            pool,meta=prepare(path,0,10,1000)
            rows=[r for split in pool.values() for group in split.values() for r in group]
            self.assertEqual(rows[0][2],1)
            self.assertEqual(meta["teacher_budget_override"],1000)
            with self.assertRaisesRegex(ValueError,"unavailable"):
                prepare(path,0,10,5000)

    def test_missing_or_mismatched_native_provenance_fails_closed(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"teacher.jsonl"; path.write_text("")
            manifest=Path(folder)/"manifest.json"
            manifest.write_text(json.dumps({"environment":{"native_module_sha256":"wrong"},"purity":{}}))
            with self.assertRaisesRegex(ValueError,"mismatch"):
                validate_teacher_manifest(path,{"native_module_sha256":"right","python_interface_sha256":"right"})
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError,"lacks"):
                validate_teacher_manifest(path,{})


if __name__=="__main__":
    unittest.main()
