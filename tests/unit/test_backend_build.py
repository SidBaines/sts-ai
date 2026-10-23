from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sts_env._backend import lightspeed_import as loader


class NativeBuildIdentityTest(unittest.TestCase):
    def _build(self, directory):
        path = Path(directory)
        module = path / "slaythespire.test.so"
        module.write_bytes(b"fixture binary")
        metadata = dict(loader.expected_metadata(), module_file=module.name,
                        module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                        python_cache_tag=sys.implementation.cache_tag)
        (path / "build_manifest.json").write_text(json.dumps(metadata))
        return path, module, metadata

    def test_manifest_rejects_changed_patch_abi_and_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path, module, original = self._build(directory)
            self.assertEqual(loader.native_metadata(path), original)
            for key in ("patch_sha256", "python_cache_tag", "upstream_commit"):
                changed = dict(original, **{key: "wrong"})
                (path / "build_manifest.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(RuntimeError, "stale, or incompatible"):
                    loader.native_metadata(path)
            (path / "build_manifest.json").write_text(json.dumps(original))
            module.write_bytes(b"rebuilt binary")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                loader.native_metadata(path)

    def test_rebuilt_same_path_cannot_relabel_already_loaded_module(self):
        with tempfile.TemporaryDirectory() as directory:
            path, module, metadata = self._build(directory)
            loaded = SimpleNamespace(__file__=str(module), _verified_build_metadata=dict(metadata))
            with patch.dict(sys.modules, {"slaythespire": loaded}):
                self.assertIs(loader.import_lightspeed(path), loaded)
                module.write_bytes(b"rebuilt binary")
                metadata["module_sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
                (path / "build_manifest.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(RuntimeError, "restart the process"):
                    loader.import_lightspeed(path)

    def test_preloaded_unverified_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, module, _ = self._build(directory)
            with patch.dict(sys.modules, {"slaythespire": SimpleNamespace(__file__=str(module))}):
                with self.assertRaisesRegex(RuntimeError, "restart the process"):
                    loader.import_lightspeed(path)
