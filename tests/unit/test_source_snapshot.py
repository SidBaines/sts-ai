from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts_policy.source_snapshot import snapshot_experiment_sources


class SourceSnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_executed_bytes_after_source_edits(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            sources={"src/sts_env/env.py":"VERSION = '0.3.0'\n",
                     "src/sts_policy/vocabulary.json":"[1,2,3]\n",
                     "scripts/probe.py":"print('probe')\n",
                     "pyproject.toml":"[project]\n",
                     "patches/sts_lightspeed_python_api.patch":"patch\n"}
            for name,content in sources.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
            destination=root/"data/snapshot"
            hashes=snapshot_experiment_sources(destination,root=root)
            (root/"src/sts_env/env.py").write_text("changed\n")
            for name,content in sources.items():
                self.assertEqual((destination/name).read_text(),content)
                self.assertEqual(hashes[name],hashlib.sha256(content.encode()).hexdigest())
            with self.assertRaises(ValueError):
                snapshot_experiment_sources(destination,root=root)


if __name__ == "__main__":
    unittest.main()
