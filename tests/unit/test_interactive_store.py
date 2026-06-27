"""Unit tests for the interactive session store (synthetic dicts, no simulator)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sts_ai.interactive.store import SessionStore, StoredSession


def _stored(session_id: str, **kw) -> StoredSession:
    base = dict(
        session_id=session_id,
        label="probe",
        world_seed=3,
        combat_control="llm",
        framing="neutral framing",
        methods=["user", "heuristic"],
        created_at="2026-06-27T00:00:00+00:00",
        updated_at="2026-06-27T00:00:01+00:00",
    )
    base.update(kw)
    return StoredSession(**base)


def _decisions() -> list[dict]:
    return [
        {"world_seed": 3, "decision_index": 0, "selected_action": {"index": 0, "bits": 1, "description": "a"}},
        {"world_seed": 3, "decision_index": 1, "selected_action": {"index": 2, "bits": 4, "description": "b"}},
    ]


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self._tmp.name) / "interactive")

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_load_roundtrip(self):
        stored = _stored("sess-a")
        decisions = _decisions()
        meta = {"world_seed": 3, "outcome": "", "final_floor": 1}
        self.store.save(stored, decisions, meta)

        self.assertTrue(self.store.exists("sess-a"))
        loaded = self.store.load_stored("sess-a")
        self.assertEqual(loaded.session_id, "sess-a")
        self.assertEqual(loaded.framing, "neutral framing")
        self.assertEqual(loaded.methods, ["user", "heuristic"])
        self.assertEqual(loaded.n_decisions, 2)
        self.assertEqual(self.store.load_decisions("sess-a"), decisions)

    def test_save_rewrites_decisions_wholesale(self):
        stored = _stored("sess-b")
        self.store.save(stored, _decisions())
        # a branch/edit shrinks history; the file must reflect exactly that
        stored.methods = ["user"]
        self.store.save(stored, _decisions()[:1])
        self.assertEqual(len(self.store.load_decisions("sess-b")), 1)
        self.assertEqual(self.store.load_stored("sess-b").n_decisions, 1)

    def test_list_and_lineage(self):
        self.store.save(_stored("root"), _decisions())
        self.store.save(_stored("child", parent_id="root", branch_point=1), _decisions())
        summaries = {s.session_id: s for s in self.store.list_stored()}
        self.assertEqual(set(summaries), {"root", "child"})
        self.assertEqual(summaries["child"].parent_id, "root")
        self.assertEqual(summaries["child"].branch_point, 1)

    def test_delete(self):
        self.store.save(_stored("gone"), _decisions(), {"x": 1})
        self.assertTrue(self.store.delete("gone"))
        self.assertFalse(self.store.exists("gone"))
        self.assertFalse(self.store.delete("gone"))

    def test_load_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.load_stored("absent")

    def test_corrupt_dir_skipped_in_list(self):
        self.store.save(_stored("ok"), _decisions())
        bad = self.store.session_dir("bad")
        bad.mkdir(parents=True)
        (bad / "session.json").write_text("{not json", encoding="utf-8")
        ids = {s.session_id for s in self.store.list_stored()}
        self.assertEqual(ids, {"ok"})


if __name__ == "__main__":
    unittest.main()
