from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.entity_model import ResidualOOCPolicy
from sts_policy.strategic_collection import collect
from sts_policy.strategic_data import read_buffer, write_buffer, stratified_rows
from sts_policy.strategic_migration import verify_zero_prior_equivalence
from tests.unit.test_strategic_collection import ScriptedWorker


class StrategicDataTests(unittest.TestCase):
    def test_reference_migration_rejects_learned_residual_and_changed_prior(self):
        model = ResidualOOCPolicy({"size": "small", "context": "none"}, width=32, embedding=4)
        payload = {"model_config": model.metadata(), "model_state": model.state_dict(),
                   "metadata": {"policy_code_identity": {"typed_encoder_sha256": "e", "legacy_bridge_sha256": "b"},
                                "training_environment": {"python_interface_sha256": "p", "native_module_sha256": "n"}}}
        reference = deepcopy(payload)
        verify_zero_prior_equivalence(payload, reference)
        changed = deepcopy(payload)
        changed["model_state"]["actor_score.2.bias"] += .01
        with self.assertRaisesRegex(ValueError, "learned/nonzero"):
            verify_zero_prior_equivalence(changed, reference)
        changed = deepcopy(payload)
        key = next(key for key in changed["model_state"] if key.startswith("prior."))
        changed["model_state"][key] += .01
        with self.assertRaisesRegex(ValueError, "prior weights"):
            verify_zero_prior_equivalence(changed, reference)

    def test_public_buffer_roundtrip_preserves_credit_and_excludes_dev_final(self):
        torch.set_num_threads(1)
        world = next(seed for seed in range(100) if world_split(seed) == "train")
        model = ResidualOOCPolicy({"size": "small", "context": "none"}, width=32, embedding=4)
        result = collect(model, [ScriptedWorker()], [TaskSpec(seed=world, kind="run", max_act=3)])
        rows = stratified_rows(result["transitions"], per_group=8)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "buffer.jsonl"
            write_buffer(path, rows, {"checkpoint": "fixed"})
            loaded, metadata = read_buffer(path)
            self.assertEqual(metadata, {"checkpoint": "fixed"})
            self.assertEqual([r.completed_return for r in loaded], [r.completed_return for r in rows])
            self.assertEqual([r.primitive_duration for r in loaded], [r.primitive_duration for r in rows])
            self.assertEqual(loaded[0].observation, rows[0].observation)
            self.assertIsNone(loaded[0].encoded)
            bad = deepcopy(rows[0])
            bad.world_seed = next(seed for seed in range(100) if world_split(seed) != "train")
            with self.assertRaisesRegex(ValueError, "DEV/final"):
                write_buffer(Path(directory)/"bad.jsonl", [bad], {})
            with self.assertRaisesRegex(ValueError, "TRAIN-only"):
                stratified_rows([bad])


if __name__ == "__main__":
    unittest.main()
