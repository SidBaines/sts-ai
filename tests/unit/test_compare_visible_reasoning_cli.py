from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compare_visible_reasoning.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_compare_visible_reasoning_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


class CompareVisibleReasoningCliTests(unittest.TestCase):
    def test_atomic_write_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("original", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _SCRIPT._write_text_atomically(output, "replacement")

            self.assertEqual(output.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_atomic_write_cleans_temporary_file_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            with mock.patch.object(
                _SCRIPT.os,
                "link",
                side_effect=OSError("link failed"),
            ):
                with self.assertRaisesRegex(OSError, "link failed"):
                    _SCRIPT._write_text_atomically(output, '{"ok":true}\n')

            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_atomic_write_publishes_complete_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.json"
            contents = '{"ok":true}\n'

            _SCRIPT._write_text_atomically(output, contents)

            self.assertEqual(output.read_text(encoding="utf-8"), contents)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_atomic_write_cannot_replace_concurrently_created_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            def create_competing_output(*_args: object) -> None:
                output.write_text("competitor", encoding="utf-8")
                raise FileExistsError(output)

            with mock.patch.object(
                _SCRIPT.os,
                "link",
                side_effect=create_competing_output,
            ):
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    _SCRIPT._write_text_atomically(output, "replacement")

            self.assertEqual(output.read_text(encoding="utf-8"), "competitor")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_expected_model_identity_is_required_by_cli(self) -> None:
        required_actions = {
            action.dest: action.required
            for action in _SCRIPT._parser()._actions
        }

        self.assertTrue(required_actions["expected_model_identity_sha256"])

    def test_embargo_check_rejects_final_window_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embargo_path = root / "embargo.json"
            embargo_path.write_text(
                json.dumps(
                    {
                        "cohort_id": "nob_fresh_final_cohort_v1",
                        "split": {
                            "holdout_window_ids": ["seed_604_r0_w0"],
                        },
                        "embargo": {
                            "teacher_action_scoring_on_holdout": False,
                        },
                        "manifests": {
                            "source": {
                                "path": "source.json",
                                "sha256": "a" * 64,
                            },
                            "validated_primary": {
                                "path": "primary.json",
                                "sha256": "b" * 64,
                            },
                            "validated_repeat": {
                                "path": "repeat.json",
                                "sha256": "c" * 64,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "sha256": "d" * 64,
                "contents": {
                    "dataset_sha256": "e" * 64,
                    "source_labels": {
                        "path": "labels.jsonl",
                        "sha256": "f" * 64,
                        "manifest_path": "labels.manifest.json",
                        "manifest_sha256": "1" * 64,
                        "source_manifest": "development.json",
                        "source_manifest_sha256": "2" * 64,
                    },
                },
            }

            with self.assertRaisesRegex(ValueError, "embargoed final"):
                _SCRIPT._embargo_provenance(
                    embargo_path,
                    rows=[{"window_id": "seed_604_r0_w0"}],
                    dataset_path=root / "data.jsonl",
                    manifest_path=root / "data.manifest.json",
                    manifest=manifest,
                    repo_root=root,
                )

    def test_directory_identity_changes_with_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            weights = model / "model.safetensors"
            weights.write_bytes(b"first")

            first = _SCRIPT._directory_provenance(
                model,
                description="model",
            )
            weights.write_bytes(b"second")
            second = _SCRIPT._directory_provenance(
                model,
                description="model",
            )

            self.assertNotEqual(
                first["identity_sha256"],
                second["identity_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
