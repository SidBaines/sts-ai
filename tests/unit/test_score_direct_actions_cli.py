from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "score_direct_actions.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_score_direct_actions_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


class _Payload:
    pass


class _FakeScorer:
    def __init__(self, model: object, tokenizer: object):
        self._model = model
        self._tokenizer = tokenizer

    def _load(self) -> tuple[object, object]:
        return self._model, self._tokenizer


class ScoreDirectActionsCliTests(unittest.TestCase):
    def test_runtime_tokenizer_does_not_retain_loaded_model(self) -> None:
        model = _Payload()
        model_reference = weakref.ref(model)
        expected_tokenizer = _Payload()
        scorer = _FakeScorer(model, expected_tokenizer)
        del model

        tokenizer = _SCRIPT._runtime_tokenizer(scorer)
        _SCRIPT._drop_scorer_references(scorer)
        gc.collect()

        self.assertIs(tokenizer, expected_tokenizer)
        self.assertIsNone(model_reference())

    def test_checkpoint_provenance_content_addresses_weights_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            weights = checkpoint / "adapters.safetensors"
            weights.write_bytes(b"first")
            (checkpoint / "adapter_config.json").write_text(
                '{"model":"test/model","rank":8}',
                encoding="utf-8",
            )
            (checkpoint / "mlx_lora_config.json").write_text(
                '{"seed":0}',
                encoding="utf-8",
            )

            first = _SCRIPT._checkpoint_provenance(checkpoint)
            weights.write_bytes(b"second")
            second = _SCRIPT._checkpoint_provenance(checkpoint)

            self.assertEqual(len(first["identity_sha256"]), 64)
            self.assertEqual(len(first["mlx_lora_config_sha256"]), 64)
            self.assertNotEqual(
                first["identity_sha256"],
                second["identity_sha256"],
            )
            self.assertNotEqual(
                first["files"]["adapters.safetensors"],
                second["files"]["adapters.safetensors"],
            )

    def test_expected_checkpoint_hashes_are_strict_label_digest_pairs(self) -> None:
        expected = {"arm": "a" * 64}
        self.assertEqual(
            _SCRIPT._parse_label_hashes(
                [f"arm={'a' * 64}"],
                option="--expected-checkpoint-identity",
            ),
            expected,
        )
        bad_values = [
            "arm=abc",
            f"bad label={'a' * 64}",
            f"arm={'A' * 64}",
            f"arm={'a' * 64}",
        ]
        for value in bad_values[:3]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _SCRIPT._parse_label_hashes(
                        [value],
                        option="--expected-checkpoint-identity",
                    )
        with self.assertRaises(ValueError):
            _SCRIPT._parse_label_hashes(
                bad_values[3:4] + bad_values[3:4],
                option="--expected-checkpoint-identity",
            )

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

    def test_float32_report_validation_is_fail_closed(self) -> None:
        valid = {
            "model_logits_dtypes": ["mlx.core.float32"],
            "scoring_dtypes": ["mlx.core.float32"],
            "output_projection_modes": [
                "tied_embedding_full_vocabulary_float32_projection_and_softcap"
            ],
        }
        _SCRIPT._validate_float32_report(valid)

        for key, value in (
            ("model_logits_dtypes", ["mlx.core.bfloat16"]),
            ("scoring_dtypes", ["float16"]),
            ("output_projection_modes", ["posthoc_float32_cast"]),
        ):
            with self.subTest(key=key):
                invalid = dict(valid)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    _SCRIPT._validate_float32_report(invalid)


if __name__ == "__main__":
    unittest.main()
