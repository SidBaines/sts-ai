from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.view_fight_comparison import load_trace, script_json
from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


class FightCompareViewerTest(unittest.TestCase):
    def test_model_text_cannot_close_embedded_data_script(self):
        value = {"thinking": '</script><script>alert("x")</script> & café'}
        encoded = script_json(value)
        self.assertNotIn("<", encoded)
        self.assertEqual(json.loads(encoded), value)

    def test_modified_saved_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "fight_r0.jsonl"
            trace.write_text('{}\n')
            meta = trace.with_suffix('.meta.json')
            lab.write_new(meta, {"extra": {"fight_lab": {"state_id": "fight"}}})
            lab.write_new(root / 'settings.json', {})
            lab.write_new(root / 'fight.result.json', {
                "status": "ok", "state_id": "fight", "trace_sha256": file_sha256(trace),
                "meta_sha256": file_sha256(meta)})
            self.assertEqual(load_trace(root, 'fight')['records'], [{}])
            trace.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                load_trace(root, 'fight')

    @unittest.skipUnless(shutil.which('node'), 'requires node')
    def test_shared_turns_align_by_turn_not_decision_index(self):
        script = Path(__file__).resolve().parents[2] / 'src/sts_ai/interactive/static/fight_compare.js'
        code = """
const assert = require('node:assert/strict');
const {indexForTurn} = require(process.argv[1]);
const old = [0,1,1,1,2,2,3].map(turn=>({turn}));
const thinking = [0,1,2,2].map(turn=>({turn}));
assert.equal(indexForTurn(old,'2'),4);
assert.equal(indexForTurn(thinking,'2'),2);
assert.equal(indexForTurn(thinking,'3'),4); // Finished, never pretend it reached turn 3.
assert.equal(indexForTurn(old,'end'),7);
assert.equal(indexForTurn(old,'0'),0);
"""
        result = subprocess.run([shutil.which('node'), '-e', code, str(script)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
