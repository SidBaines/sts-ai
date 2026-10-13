from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.complete_frontier_replays import read_parent


class ReplayCompletionTests(unittest.TestCase):
    def test_completion_cannot_reintroduce_quarantined_world_or_unpinned_starts(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            starts=root/"starts.jsonl"
            starts.write_text(json.dumps({"seed":1})+"\n"+json.dumps({"seed":2})+"\n")
            (root/"unverified_starts.jsonl").write_text(json.dumps({"seed":1})+"\n"+json.dumps({"seed":3})+"\n")
            (root/"manifest.json").write_text(json.dumps({"starts_sha256":hashlib.sha256(starts.read_bytes()).hexdigest()}))
            (root/"replay_failures.json").write_text(json.dumps({"failures":[{"seed":1}]}))
            previous,pending,blocked=read_parent(root)
            self.assertEqual(previous,[{"seed":2}])
            self.assertEqual(pending,[{"seed":3}])
            self.assertEqual(blocked,{1})
            starts.write_text("changed\n")
            with self.assertRaises(ValueError):
                read_parent(root)


if __name__ == "__main__":
    unittest.main()
