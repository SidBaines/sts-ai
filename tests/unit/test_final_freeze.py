from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from sts_policy.final_freeze import capture_final_freeze, verify_final_freeze, FreezeChangedError


class FinalFreezeTests(unittest.TestCase):
    def setup_inputs(self, root):
        model=root/'model.pt';model.write_bytes(b'checkpoint')
        panel=root/'panel.jsonl';panel.write_text('{"seed":3}\n')
        backend=root/'backend';backend.mkdir()
        module=backend/'slaythespire.test.so';module.write_bytes(b'native binary')
        provenance={'module_file':module.name,'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
                    'upstream_commit':'pinned','patch_sha256':'patched'}
        (backend/'build_manifest.json').write_text(json.dumps(provenance))
        return model,panel,backend,provenance

    def test_identity_survives_json_roundtrip_and_generated_definitions_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            model,panel,backend,provenance=self.setup_inputs(Path(directory))
            definitions={'full':{'kind':'run','split':'final','horizon':2048,'max_act':3}}
            freeze=capture_final_freeze({'shared':model},{'hard':panel},backend,
                                       generated_definitions=definitions,expected_backend_provenance=provenance)
            definitions['full']['horizon']=1
            self.assertEqual(freeze['generated_definitions']['full']['horizon'],2048)
            verify_final_freeze(json.loads(json.dumps(freeze)),backend_build_dir=backend)

    def test_changed_checkpoint_panel_backend_and_record_are_rejected(self):
        for kind in ('checkpoint','panel','module','manifest','record','backend_path'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);model,panel,backend,_=self.setup_inputs(root)
                freeze=capture_final_freeze({'shared':model},{'hard':panel},backend)
                if kind=='checkpoint':model.write_bytes(b'different!')
                elif kind=='panel':panel.write_text('{"seed":4}\n')
                elif kind=='module':(backend/'slaythespire.test.so').write_bytes(b'changed native')
                elif kind=='manifest':(backend/'build_manifest.json').write_text('{}')
                elif kind=='record':freeze['generated_definitions']['edited']=True
                with self.assertRaises(FreezeChangedError):
                    verify_final_freeze(freeze,backend_build_dir=root/'other' if kind=='backend_path' else backend)

    def test_initial_native_manifest_binary_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,panel,backend,_=self.setup_inputs(root)
            (backend/'slaythespire.test.so').write_bytes(b'wrong native')
            with self.assertRaises(ValueError):capture_final_freeze({'shared':model},{'hard':panel},backend)

    def test_retargeted_checkpoint_alias_is_rejected_even_if_old_file_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,panel,backend,_=self.setup_inputs(root)
            alias=root/'chosen.pt';alias.symlink_to(model)
            freeze=capture_final_freeze({'shared':alias},{'hard':panel},backend)
            other=root/'other.pt';other.write_bytes(model.read_bytes())
            alias.unlink();alias.symlink_to(other)
            with self.assertRaises(FreezeChangedError):verify_final_freeze(freeze)
