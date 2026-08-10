from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import local_task_validate
from sts_ai.local_tasks import base
from sts_ai.local_tasks.start_state import start_state_payload_sha256
from sts_ai.local_tasks.start_state import START_STATE_SIGNATURE_SCHEMA_VERSION
from sts_ai.local_tasks.validation import (
    build_validated_manifest,
    build_validation_report,
    classify_child_result,
    run_window_subprocess,
    source_rollouts_identity,
    validate_one_window,
)


def _manifest() -> dict:
    windows = [
        {
            "window_id": "seed_4_r0_w0",
            "source_stem": "seed_4_r0",
            "world_seed": 4,
            "split": "holdout",
            "label": "won",
            "pre_actions": [],
        },
        {
            "window_id": "seed_5_r0_w0",
            "source_stem": "seed_5_r0",
            "world_seed": 5,
            "split": "train",
            "label": "loss",
            "pre_actions": [],
        },
        {
            "window_id": "seed_6_r0_w0",
            "source_stem": "seed_6_r0",
            "world_seed": 6,
            "split": "train",
            "label": "convincing",
            "pre_actions": [],
        },
    ]
    return {
        "task_id": "gremlin_nob",
        "version": 1,
        "source_rollout_dir": "example",
        "n_windows": len(windows),
        "split_counts": {"holdout": 1, "train": 2},
        "label_counts": {"won": 1, "loss": 1, "convincing": 1},
        "windows": windows,
    }


def _result(window: dict, index: int, status: str) -> dict:
    is_success = status == "success"
    result = {
        "window_index": index,
        "window_id": window["window_id"],
        "world_seed": window["world_seed"],
        "split": window["split"],
        "status": status,
        "returncode": 0 if is_success else (None if status == "timeout" else 1),
        "error_type": None if is_success else "ReplayError",
        "error": None if is_success else f"example {status}",
        "elapsed_seconds": 0.1,
    }
    if is_success:
        signature_payload = {"example": window["window_id"]}
        result.update(
            {
                "start_state_signature": {
                    "schema_version": START_STATE_SIGNATURE_SCHEMA_VERSION,
                    "sha256": start_state_payload_sha256(signature_payload),
                    "payload": signature_payload,
                },
                "simulator_binary": {
                    "path": "/tmp/slaythespire.so",
                    "size_bytes": 123,
                    "sha256": "b" * 64,
                },
            }
        )
    return result


def _write_source_rollouts(root: Path, manifest: dict) -> None:
    rollout_dir = root / "rollouts"
    rollout_dir.mkdir()
    manifest["source_rollout_dir"] = str(rollout_dir)
    for stem in sorted({window["source_stem"] for window in manifest["windows"]}):
        (rollout_dir / f"{stem}.jsonl").write_text(
            json.dumps({"source_stem": stem}) + "\n",
            encoding="utf-8",
        )


class ValidationArtifactTest(unittest.TestCase):
    def test_report_and_filtered_manifest_account_for_every_window(self):
        manifest = _manifest()
        results = [
            _result(manifest["windows"][0], 0, "success"),
            _result(manifest["windows"][1], 1, "failure"),
            _result(manifest["windows"][2], 2, "timeout"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_source_rollouts(Path(tmp), manifest)
            source = Path(tmp) / "source.json"
            base.write_json(source, manifest)
            report = build_validation_report(
                manifest,
                results,
                source_manifest_path=source,
                timeout_seconds=3.0,
                combat_observation="legacy",
                battle_simulations=50,
                max_act=3,
            )
            validated = build_validated_manifest(
                manifest,
                report,
                source_manifest_path=source,
                report_path=Path(tmp) / "report.json",
            )

        self.assertEqual(
            report["status_counts"],
            {"success": 1, "failure": 1, "timeout": 1},
        )
        self.assertTrue(report["all_windows_accounted_for"])
        self.assertEqual(len(report["source_manifest"]["sha256"]), 64)
        self.assertEqual(report["source_rollouts"]["n_files"], 3)
        self.assertEqual(len(report["source_rollouts"]["content_set_sha256"]), 64)
        self.assertEqual(
            report["validator"]["simulator_binary"]["sha256"],
            "b" * 64,
        )
        self.assertEqual(validated["n_windows"], 1)
        self.assertEqual(validated["split_counts"], {"holdout": 1})
        self.assertEqual(validated["label_counts"], {"won": 1})
        self.assertEqual(
            [window["window_id"] for window in validated["windows"]],
            ["seed_4_r0_w0"],
        )
        validation = validated["validation"]
        self.assertEqual(validation["n_source_windows"], 3)
        self.assertEqual(validation["n_validated_windows"], 1)
        self.assertEqual(validation["n_excluded_windows"], 2)
        self.assertEqual(
            validated["windows"][0]["start_state_signature"]["sha256"],
            results[0]["start_state_signature"]["sha256"],
        )
        self.assertEqual(validation["source_rollouts"]["n_files"], 3)
        self.assertEqual(
            [item["status"] for item in validation["excluded_windows"]],
            ["failure", "timeout"],
        )
        self.assertTrue(validation["all_source_windows_accounted_for"])

    def test_report_rejects_missing_reordered_or_duplicate_results(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            _write_source_rollouts(Path(tmp), manifest)
            source = Path(tmp) / "source.json"
            base.write_json(source, manifest)
            kwargs = {
                "source_manifest_path": source,
                "timeout_seconds": 3.0,
                "combat_observation": "legacy",
                "battle_simulations": 50,
                "max_act": 3,
            }
            complete = [
                _result(window, index, "success")
                for index, window in enumerate(manifest["windows"])
            ]
            with self.assertRaisesRegex(ValueError, "every source window"):
                build_validation_report(manifest, complete[:-1], **kwargs)
            with self.assertRaisesRegex(ValueError, "source order"):
                build_validation_report(
                    manifest,
                    [complete[1], complete[0], complete[2]],
                    **kwargs,
                )
            with self.assertRaisesRegex(ValueError, "every source window"):
                build_validation_report(
                    manifest,
                    [complete[0], complete[0], complete[2]],
                    **kwargs,
                )
            missing_error = [dict(result) for result in complete]
            del missing_error[0]["error"]
            with self.assertRaisesRegex(ValueError, "returncode and error"):
                build_validation_report(manifest, missing_error, **kwargs)

            duplicate_manifest = _manifest()
            duplicate_manifest["windows"][1]["window_id"] = duplicate_manifest[
                "windows"
            ][0]["window_id"]
            with self.assertRaisesRegex(ValueError, "duplicate window_id"):
                build_validation_report(
                    duplicate_manifest,
                    complete,
                    **kwargs,
                )

    def test_filtered_manifest_rejects_a_different_source_file(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            _write_source_rollouts(Path(tmp), manifest)
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            base.write_json(first, manifest)
            base.write_json(second, {**manifest, "note": "different bytes"})
            results = [
                _result(window, index, "success")
                for index, window in enumerate(manifest["windows"])
            ]
            report = build_validation_report(
                manifest,
                results,
                source_manifest_path=first,
                timeout_seconds=3.0,
                combat_observation="legacy",
                battle_simulations=50,
                max_act=3,
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                build_validated_manifest(
                    manifest,
                    report,
                    source_manifest_path=second,
                    report_path=Path(tmp) / "report.json",
                )

    def test_source_rollout_identity_hashes_unique_referenced_bytes(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            _write_source_rollouts(Path(tmp), manifest)
            first = source_rollouts_identity(manifest)
            self.assertEqual(first["n_files"], 3)
            path = Path(manifest["source_rollout_dir"]) / "seed_5_r0.jsonl"
            path.write_text('{"changed": true}\n', encoding="utf-8")
            second = source_rollouts_identity(manifest)
        self.assertNotEqual(
            first["content_set_sha256"],
            second["content_set_sha256"],
        )

    def test_filtered_manifest_rejects_changed_source_rollout_bytes(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            _write_source_rollouts(Path(tmp), manifest)
            source = Path(tmp) / "source.json"
            base.write_json(source, manifest)
            results = [
                _result(window, index, "success")
                for index, window in enumerate(manifest["windows"])
            ]
            report = build_validation_report(
                manifest,
                results,
                source_manifest_path=source,
                timeout_seconds=3.0,
                combat_observation="legacy",
                battle_simulations=50,
                max_act=3,
            )
            changed = Path(manifest["source_rollout_dir"]) / "seed_4_r0.jsonl"
            changed.write_text('{"different": "bytes"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-rollout"):
                build_validated_manifest(
                    manifest,
                    report,
                    source_manifest_path=source,
                    report_path=Path(tmp) / "report.json",
                )

    def test_report_rejects_windows_loaded_from_mixed_simulator_binaries(self):
        manifest = _manifest()
        results = [
            _result(window, index, "success")
            for index, window in enumerate(manifest["windows"])
        ]
        results[-1]["simulator_binary"] = {
            **results[-1]["simulator_binary"],
            "sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            _write_source_rollouts(Path(tmp), manifest)
            source = Path(tmp) / "source.json"
            base.write_json(source, manifest)
            with self.assertRaisesRegex(ValueError, "different simulator binaries"):
                build_validation_report(
                    manifest,
                    results,
                    source_manifest_path=source,
                    timeout_seconds=3.0,
                    combat_observation="legacy",
                    battle_simulations=50,
                    max_act=3,
                )


class ChildIsolationTest(unittest.TestCase):
    def test_crashed_child_without_payload_is_an_explicit_failure(self):
        window = _manifest()["windows"][0]
        result = classify_child_result(
            window=window,
            window_index=0,
            returncode=-11,
            result_payload=None,
            stdout=None,
            stderr="segmentation fault",
            elapsed_seconds=0.2,
        )
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["returncode"], -11)
        self.assertEqual(result["error_type"], "ChildProcessError")
        self.assertIn("segmentation fault", result["child_stderr"])

    @patch("sts_ai.local_tasks.validation.subprocess.run")
    def test_timeout_is_an_explicit_result(self, run_mock):
        run_mock.side_effect = subprocess.TimeoutExpired(
            ["python", "validator"],
            2.5,
            output="partial output",
            stderr="native stall",
        )
        window = _manifest()["windows"][0]
        result = run_window_subprocess(
            task_id="gremlin_nob",
            manifest_path=Path("source.json"),
            window=window,
            window_index=0,
            timeout_seconds=2.5,
            combat_observation="legacy",
            battle_simulations=50,
            max_act=3,
            script_path=Path("scripts/local_task_validate.py"),
            repo_root=Path.cwd(),
        )
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["returncode"])
        self.assertIn("2.5 seconds", result["error"])
        self.assertEqual(result["child_stdout"], "partial output")
        self.assertEqual(result["child_stderr"], "native stall")

    @patch(
        "sts_ai.local_tasks.validation.build_start_state_signature",
        return_value={
            "schema_version": START_STATE_SIGNATURE_SCHEMA_VERSION,
            "sha256": start_state_payload_sha256({}),
            "payload": {},
        },
    )
    @patch(
        "sts_ai.local_tasks.validation.simulator_binary_identity",
        return_value={
            "path": "/tmp/slaythespire.so",
            "size_bytes": 123,
            "sha256": "b" * 64,
        },
    )
    @patch("sts_ai.local_tasks.validation.replay_task_start", return_value=4)
    @patch("sts_ai.local_tasks.validation.LightspeedHybridEnv")
    def test_single_window_mode_uses_llm_combat_and_checks_live_task(
        self,
        env_type,
        replay,
        simulator_identity,
        start_signature,
    ):
        env = env_type.return_value
        env.summary.return_value = {"combat": {"enemies": [{"alive": True}]}}
        task = SimpleNamespace(completion_reason=lambda summary: None)
        manifest = _manifest()

        result = validate_one_window(
            manifest,
            window_index=0,
            task=task,
            combat_observation="legacy",
            battle_simulations=17,
            max_act=2,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["n_pre_actions_applied"], 4)
        self.assertEqual(
            result["start_state_signature"]["sha256"],
            start_state_payload_sha256({}),
        )
        self.assertEqual(result["simulator_binary"]["sha256"], "b" * 64)
        env_type.assert_called_once_with(
            world_seed=4,
            combat_control="llm",
            combat_observation="legacy",
            battle_simulations=17,
            max_act=2,
        )
        replay.assert_called_once_with(env, manifest["windows"][0], task)
        self.assertEqual(simulator_identity.call_count, 2)
        simulator_identity.assert_called_with(env)
        start_signature.assert_called_once_with(env)


class ValidationCliTest(unittest.TestCase):
    def test_public_cli_has_optional_filtered_manifest_and_timeout(self):
        args = local_task_validate.parse_args(
            [
                "--task",
                "gremlin_nob",
                "--manifest",
                "source.json",
                "--report",
                "report.json",
                "--timeout-seconds",
                "7.5",
            ]
        )
        self.assertIsNone(args.out_manifest)
        self.assertEqual(args.timeout_seconds, 7.5)
        args = local_task_validate.parse_args(
            [
                "--task",
                "gremlin_nob",
                "--manifest",
                "source.json",
                "--report",
                "report.json",
                "--out-manifest",
                "validated.json",
            ]
        )
        self.assertEqual(args.out_manifest, Path("validated.json"))

    @patch("scripts.local_task_validate.validate_one_window")
    def test_internal_cli_writes_child_result_and_returns_its_code(self, validate):
        manifest = _manifest()
        validate.return_value = _result(manifest["windows"][1], 1, "failure")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            output = Path(tmp) / "result.json"
            base.write_json(source, manifest)
            returncode = local_task_validate.main(
                [
                    "--_validate-window",
                    "--task",
                    "gremlin_nob",
                    "--manifest",
                    str(source),
                    "--window-index",
                    "1",
                    "--result",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(returncode, 1)
        self.assertEqual(payload["window_id"], "seed_5_r0_w0")


if __name__ == "__main__":
    unittest.main()
