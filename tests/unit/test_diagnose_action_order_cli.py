from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "diagnose_action_order.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_diagnose_action_order_script",
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


class DiagnoseActionOrderCliTests(unittest.TestCase):
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
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
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


if __name__ == "__main__":
    unittest.main()
