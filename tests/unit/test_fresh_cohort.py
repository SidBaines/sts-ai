"""Completed receipts and exact process-shard unions, using tiny pure worlds."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from sts_rl.fresh_cohort import SHARD_FIELDS, merge_cohorts, validate_cohorts
from sts_rl.fresh_collection import play_world
from sts_rl.fresh_panel import _bank_rows, bank_files, load_bank
from sts_rl.fresh_study import atomic_json
from sts_rl.recording import file_identity
from tests.unit.fresh_cohort_fixture import reseal_summary, write_cohort
from tests.unit.test_fresh_collection import (Control, EXPECTED, Policy, Progress, SEED, SOURCE,
                                            Worker, config, observation)


def world(seed=SEED):
    return deepcopy(play_world(Worker([observation(), observation("combat", "BOSS"),
                      observation("combat", "BOSS", hp=20, actions=[])]),
                      Policy(), seed, config(mode="eval"), Control(), Progress(), SOURCE, lambda _: None))


def rewrite_world(directory, change):
    summary = json.loads((directory / "summary.json").read_text())
    brief = summary["records"][0]
    path = directory / brief["shard"]
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    change(value)
    atomic_json(path, value, compressed=True)
    summary["records"][0] = {**{k: v for k, v in value.items() if k not in {"source", "labels", "bank"}},
                             "shard": brief["shard"], "shard_identity": file_identity(path)}
    reseal_summary(directory, summary)


class CohortTests(unittest.TestCase):
    def test_teacher_played_cohorts_require_explicit_analysis_optin(self):
        path = Path(__file__).resolve().parents[2]/"scripts"/"rl"/"fresh_compare.py"
        spec = importlib.util.spec_from_file_location("fresh_compare_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = write_cohort(root/"champion", [world()])
            teacher = write_cohort(root/"teacher", [world(SEED+1)],
                                   config={"mode": "collect", "teacher_plays": True})
            for paths in (teacher, [champion, teacher], [teacher, champion]):
                with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "teacher_plays true"):
                    validate_cohorts(paths)
            for path in (teacher, teacher/"summary.json"):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(ValueError, "teacher_plays true"):
                        module.read_games(path)
                    with self.assertRaisesRegex(ValueError, "teacher_plays true"):
                        module.read_rows(path, "games")
                    self.assertEqual(module.read_games(path, allow_teacher_plays=True)[0]["world_seed"], SEED+1)
            cohort = validate_cohorts(teacher, allow_teacher_plays=True)
            self.assertIs(cohort["config"]["teacher_plays"], True)
            self.assertEqual(len(module.read_games(champion)), 1)

    def test_fight_banks_refuse_teacher_played_cohorts_even_after_optin(self):
        for bank in (False, True):
            with self.subTest(bank=bank), tempfile.TemporaryDirectory() as temporary:
                directory = write_cohort(Path(temporary)/"teacher", [world()],
                                         config={"mode": "collect", "bank": bank, "teacher_plays": True})
                for load in (bank_files, load_bank):
                    with self.assertRaisesRegex(ValueError, "teacher_plays true"):
                        load(directory)
                cohort = validate_cohorts(directory, allow_teacher_plays=True)
                with self.assertRaisesRegex(ValueError, "teacher_plays true cannot be used for a fight bank"):
                    _bank_rows(cohort)

    def test_merge_exact_grid_preserves_bytes_sources_and_comparison_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = write_cohort(root/"a", [world()])
            b = write_cohort(root/"b", [world(SEED+1), world(SEED+2)],
                             config={"workers": 8, "deadline_utc": "2030-01-01T00:00:00Z"})
            cohort = validate_cohorts([b, a], world_start=SEED, n=3)
            self.assertEqual([r["world_seed"] for r in cohort["records"]], list(range(SEED, SEED+3)))
            self.assertEqual(len(load_bank([a, b], world_start=SEED, n=3)), 3)
            result = merge_cohorts([b, a], root/"merged", world_start=SEED, n=3)
            merged = validate_cohorts(root/"merged", world_start=SEED, n=3)
            self.assertTrue(result["cohort_complete"])
            self.assertTrue(result["confidence_claims_allowed"])
            self.assertEqual(result["wins"], 3)
            self.assertEqual(merged["science_identity"], cohort["science_identity"])
            self.assertEqual(len(load_bank(root/"merged")), 3)
            self.assertEqual(len(merged["sources"]), 2)
            for before, after in zip(cohort["shards"], merged["shards"]):
                self.assertEqual(before.read_bytes(), after.read_bytes())
            # fresh_compare.read_rows(summary.json, 'games') unwraps records;
            # fresh_statistics.compare_games requires these terminal fields.
            records = json.loads((root/"merged"/"summary.json").read_text())["records"]
            self.assertEqual([r["world_seed"] for r in records], list(range(SEED, SEED+3)))
            self.assertTrue(all(r["status"] == "completed" and isinstance(r["win"], bool) and
                                r["floor"] == 21 and "bosses" in r for r in records))
            with self.assertRaises(FileExistsError):
                merge_cohorts([a, b], root/"merged", world_start=SEED, n=3)

    def test_only_legitimate_shard_config_fields_are_excluded(self):
        self.assertEqual(SHARD_FIELDS, {"out", "n", "world_start", "workers", "deadline_utc"})
        for key, value in (("timeout", 31.), ("samples", 23), ("simulations", 999),
                           ("teacher_probabilities", {"BOSS2": .5}), ("max_labels", 1),
                           ("unknown_science_setting", "changed")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                a = write_cohort(root/"a", [world()])
                b = write_cohort(root/"b", [world(SEED+1)], config={key: value})
                with self.assertRaisesRegex(ValueError, "Science identity"):
                    validate_cohorts([a, b])

    def test_archive_native_environment_and_policy_identity_must_match(self):
        for kind in ("archive", "native", "environment", "policy", "filter"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root, other = Path(temporary), world(SEED+1)
                kwargs = {}
                if kind == "archive":
                    kwargs["archive"] = b"changed source"
                elif kind == "native":
                    other["expected_provenance"]["native_provenance"] = {"module_sha256": "changed"}
                elif kind == "environment":
                    other["expected_provenance"]["python_interface_sha256"] = "changed"
                elif kind == "filter":
                    kwargs["policy_changes"] = {"filter_ids": ["changed"]}
                else:
                    kwargs["policy_changes"] = {"combat": {"sha256": "replacement", "path": "replacement", "bytes": 1}}
                a, b = write_cohort(root/"a", [world()]), write_cohort(root/"b", [other], **kwargs)
                with self.assertRaisesRegex(ValueError, "identity"):
                    validate_cohorts([a, b])

    def test_exact_requested_seedgrid_rejects_gap_duplicate_missing_and_extra(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = write_cohort(root/"a", [world()])
            b = write_cohort(root/"b", [world(SEED+2)])
            duplicate = write_cohort(root/"duplicate", [world()])
            for paths, start, n in (([a, b], SEED, 3), ([a, duplicate], SEED, 2),
                                    ([a], SEED, 2), ([a], SEED+1, 1)):
                with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "seedgrid"):
                    validate_cohorts(paths, world_start=start, n=n)
            with self.assertRaisesRegex(ValueError, "seedgrid"):
                validate_cohorts([a, b])
            with self.assertRaisesRegex(ValueError, "both"):
                validate_cohorts(a, world_start=SEED)
            for start, n in ((SEED, True), (float(SEED), 1), (SEED, 1.5)):
                with self.assertRaises(ValueError):
                    validate_cohorts(a, world_start=start, n=n)

    def test_merge_cli_validates_expected_cohort_and_publishes_receipts(self):
        from contextlib import redirect_stdout
        import importlib.util
        import io
        path = Path(__file__).resolve().parents[2]/"scripts"/"rl"/"fresh_merge.py"
        spec = importlib.util.spec_from_file_location("fresh_merge_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = write_cohort(root/"a", [world()]), write_cohort(root/"b", [world(SEED+1)])
            args = ["--inputs", str(a), str(b), "--world-start", str(SEED), "--n", "2"]
            for extra in ([], ["--out", str(root/"merged")]):
                output = io.StringIO()
                with redirect_stdout(output):
                    module.main([*args, *extra])
                self.assertEqual(json.loads(output.getvalue())["completed"], 2)
            self.assertTrue((root/"merged"/"completion.json").exists())

    def test_receipt_status_summary_and_shard_hashes_are_required(self):
        for kind in ("completion", "receipt", "summary", "shard", "missing_shard_hash", "incomplete", "extra", "quarantine"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = write_cohort(Path(temporary)/"bank", [world()])
                if kind in {"completion", "receipt"}:
                    (directory/f"{kind}.json").unlink()
                elif kind == "summary":
                    (directory/"summary.json").write_text("{}")
                elif kind == "shard":
                    next((directory/"worlds").glob("*.gz")).write_bytes(b"corrupted")
                elif kind == "missing_shard_hash":
                    summary = json.loads((directory/"summary.json").read_text())
                    summary["records"][0].pop("shard_identity")
                    reseal_summary(directory, summary)
                elif kind == "incomplete":
                    completion = json.loads((directory/"completion.json").read_text())
                    completion["cohort_complete"] = False
                    atomic_json(directory/"completion.json", completion)
                elif kind == "extra":
                    (directory/"worlds"/"extra.json.gz").write_bytes(b"unexpected")
                else:
                    (directory/"quarantine").mkdir()
                    (directory/"quarantine"/"fault.json").write_text("{}")
                with self.assertRaises(ValueError):
                    load_bank(directory)

    def test_completed_receipt_cannot_hide_duplicate_records_or_fault_losses(self):
        for kind in ("duplicate", "fault", "source", "provenance", "summary_status"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = write_cohort(Path(temporary)/"bank", [world()])
                if kind == "duplicate":
                    summary = json.loads((directory/"summary.json").read_text())
                    summary["records"] *= 2
                    reseal_summary(directory, summary)
                elif kind == "summary_status":
                    summary = json.loads((directory/"summary.json").read_text())
                    summary["records"][0]["status"] = "fault"
                    reseal_summary(directory, summary)
                else:
                    def change(row):
                        if kind == "fault":
                            row.update(fault="native fault converted to a loss", win=False)
                        elif kind == "source":
                            row["source"] = {"protocol": "unknown"}
                        else:
                            row["expected_provenance"]["native_provenance"] = {"module_sha256": "wrong"}
                    rewrite_world(directory, change)
                with self.assertRaises(ValueError):
                    validate_cohorts(directory)

    def test_bank_hash_prefix_source_and_entry_validation_survives_receipt_validation(self):
        for field in ("observation", "prefix", "entering_description", "source", "expected_provenance", "first_combat_hash"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = write_cohort(Path(temporary)/"bank", [world()])
                def change(row):
                    bank = row["bank"][0]
                    if field == "observation":
                        bank[field]["text"] = "changed"
                    elif field == "prefix":
                        bank[field] = [77]
                    elif field == "expected_provenance":
                        bank[field]["python_interface_sha256"] = "changed"
                    else:
                        bank[field] = "wrong"
                rewrite_world(directory, change)
                if field == "first_combat_hash":
                    # Bank hash is checked at prepared-start reconstruction,
                    # not invented here from the OOC observation.
                    self.assertEqual(load_bank(directory)[0][field], "wrong")
                else:
                    with self.assertRaises(ValueError):
                        load_bank(directory)


if __name__ == "__main__":
    unittest.main()
