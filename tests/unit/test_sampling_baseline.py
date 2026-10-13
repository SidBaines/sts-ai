from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts import sample_fight_bank as sample
from sts_ai import fight_lab as lab
from tests.unit.test_fight_lab import record


class BaselineCohortTest(unittest.TestCase):
    def bank(self, root: Path) -> tuple[Path, Path, list[str]]:
        bank, baseline = root / "bank", root / "baseline"
        rows = []
        for seed in (1004, 1005, 1006):
            item = record()
            item["world_seed"] = seed
            rows.extend(lab.index_records([item], f"seed_{seed}_r0.jsonl"))
        lab.write_new(bank / "collection.json", {"states": rows})
        sample.prepare(bank, baseline)
        order = lab.read_json(baseline / "manifest.json")["order"]
        return bank, baseline, order

    def test_rerun_includes_timeout_and_partial_excludes_unattempted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, baseline, order = self.bank(root)
            sample.atomic(baseline / f"{order[0]}.result.json",
                          {"state_id": order[0], "status": "timeout"})
            (baseline / f"{order[1]}_r0.jsonl").write_text("{}\n")
            (baseline / f"{order[2]}_r0.jsonl").touch()
            output = root / "rerun"
            sample.prepare(bank, output, baseline=baseline, combat_observation="combat_public_v4")
            manifest, _, settings = sample.validate(output)
            self.assertEqual(manifest["order"], order[:2])
            self.assertEqual(settings["combat_observation"], "combat_public_v4")
            self.assertIs(settings["thinking"], False)
            self.assertEqual(manifest["sampling_seeds"],
                             {state_id: sample.sampling_seed(state_id) for state_id in order[:2]})
            (baseline / f"{order[1]}_r0.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "baseline artifact changed"):
                sample.validate(output)

    def test_corrupt_completed_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, baseline, order = self.bank(Path(directory))
            sample.atomic(baseline / f"{order[0]}.result.json", {
                "state_id": order[0], "status": "ok", "trace_sha256": "wrong",
                "meta_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "baseline finished trajectory changed"):
                sample.baseline_cohort(bank, baseline)

    def test_empty_baseline_and_different_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, baseline, order = self.bank(Path(directory))
            with self.assertRaisesRegex(ValueError, "no attempted fights"):
                sample.baseline_cohort(bank, baseline)
            (baseline / f"{order[0]}_r0.jsonl").write_text("{}\n")
            manifest = lab.read_json(baseline / "manifest.json")
            manifest["sampling_seeds"][order[0]] += 1
            sample.atomic(baseline / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "different sampling seeds"):
                sample.baseline_cohort(bank, baseline)

    def test_mismatched_sampling_settings_rejected_before_output_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, baseline, order = self.bank(root)
            (baseline / f"{order[0]}_r0.jsonl").write_text("{}\n")
            settings = lab.read_json(baseline / "settings.json")
            settings["thinking"] = True
            sample.atomic(baseline / "settings.json", settings)
            with self.assertRaisesRegex(ValueError, "sampling settings changed"):
                sample.prepare(bank, root / "rerun", baseline=baseline)
            self.assertFalse((root / "rerun").exists())
