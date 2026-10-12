from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.run_nonllm_campaign import Campaign
from sts_policy.composition import compose_checkpoint
from sts_policy.encoding import ENCODER_VERSION
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import policy_from_payload, CompositeActorCritic
from sts_policy.training import load_checkpoint, save_checkpoint


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointCompositionIdentityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(812)
        self.gpu_guard = patch.object(torch.backends.mps, "is_available", return_value=False)
        self.gpu_guard.start()
        self.addCleanup(self.gpu_guard.stop)

    def sources(self, root: Path):
        paths = root / "combat.pt", root / "ooc.pt"
        models = []
        for path in paths:
            model = CandidateActorCritic(size="small", context="none")
            self.write_candidate(path, model)
            models.append(model)
        return paths, models

    def write_candidate(self, path: Path, model: CandidateActorCritic):
        save_checkpoint(path, model, model_config=model.metadata(),
                        metadata={"encoder_version": ENCODER_VERSION,
                                  "stage": "bc", "data_identity": "cpu-only-identity-fixture"})

    def runner(self, root: Path):
        # Campaign initialization records this string; joint_composition never
        # opens/imports a native build. GPU RNG access is separately disabled.
        with patch.dict(os.environ, {"STS_LIGHTSPEED_BUILD": "unused-no-native-calls"}):
            return Campaign(root / "output")

    def test_unchanged_sources_reuse_without_recomposition_or_mtime_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _ = self.sources(root)
            runner = self.runner(root)
            with patch("sts_policy.composition.compose_checkpoint", wraps=compose_checkpoint) as compose:
                first, report = runner.joint_composition(*paths)
                initial = first.read_bytes(), first.stat().st_mtime_ns
                second, reused = runner.joint_composition(*paths)
                self.assertEqual(compose.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(report, reused)
            self.assertEqual(initial, (second.read_bytes(), second.stat().st_mtime_ns))
            self.assertIsInstance(policy_from_payload(load_checkpoint(first)), CompositeActorCritic)

    def test_changed_source_weights_create_new_identity_and_preserve_old_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, models = self.sources(root)
            runner = self.runner(root)
            old, old_report = runner.joint_composition(*paths)
            old_bytes, old_mtime = old.read_bytes(), old.stat().st_mtime_ns
            old_manifest = old.with_suffix(".composition.json").read_bytes()
            with torch.no_grad():
                next(models[0].parameters()).add_(.125)
            self.write_candidate(paths[0], models[0])
            new, new_report = runner.joint_composition(*paths)
            self.assertNotEqual(old, new)
            self.assertTrue(old.name.startswith("joint_initial_composite_"))
            self.assertTrue(new.name.startswith("joint_initial_composite_"))
            self.assertNotEqual(old_report["composition_provenance"]["sources"]["combat"]["sha256"],
                                new_report["composition_provenance"]["sources"]["combat"]["sha256"])
            self.assertEqual(old_bytes, old.read_bytes())
            self.assertEqual(old_mtime, old.stat().st_mtime_ns)
            self.assertEqual(old_manifest, old.with_suffix(".composition.json").read_bytes())
            self.assertEqual(new_report["sha256"], sha(new))

    def test_tampered_checkpoint_bytes_rejected_before_reuse_or_recompose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _ = self.sources(root)
            runner = self.runner(root)
            checkpoint, _ = runner.joint_composition(*paths)
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            damaged = checkpoint.read_bytes()
            with patch("sts_policy.composition.compose_checkpoint") as compose:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    runner.joint_composition(*paths)
                compose.assert_not_called()
            self.assertEqual(damaged, checkpoint.read_bytes())

    def test_tampered_report_source_sha_rejected_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _ = self.sources(root)
            runner = self.runner(root)
            checkpoint, _ = runner.joint_composition(*paths)
            before = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
            manifest = checkpoint.with_suffix(".composition.json")
            report = json.loads(manifest.read_text())
            report["composition_provenance"]["sources"]["combat"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(report))
            damaged_manifest = manifest.read_bytes()
            with patch("sts_policy.composition.compose_checkpoint") as compose:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    runner.joint_composition(*paths)
                compose.assert_not_called()
            self.assertEqual(before, (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns))
            self.assertEqual(damaged_manifest, manifest.read_bytes())

    def test_partial_composed_artifact_rejected_instead_of_silently_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _ = self.sources(root)
            runner = self.runner(root)
            checkpoint, _ = runner.joint_composition(*paths)
            checkpoint.with_suffix(".composition.json").rename(root / "preserved-report.json")
            with patch("sts_policy.composition.compose_checkpoint") as compose:
                with self.assertRaisesRegex(RuntimeError, "Partial composed artifact"):
                    runner.joint_composition(*paths)
                compose.assert_not_called()
