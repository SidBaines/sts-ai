from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_fight_formats import ordered_pairs
from sts_ai.fight_comparison import infrastructure_annotations, outcome, paired_rows, summarize_pairs
from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


def row(*, win=True, completed=True, invalid=False, hp_loss=4):
    return {"state_id": "a", "world_seed": 3, "hp": 80, "policy_seed": 55,
            "enemies": ["CULTIST"], "floor": 1, "completed": completed,
            "win": win, "hp_loss": hp_loss if completed else None,
            "outcome": "win" if win else "loss" if completed else "incomplete",
            "invalid": invalid, "stopped_reason": "agent_invalid" if invalid else "task_complete",
            "seconds": 4}


class FightComparisonTest(unittest.TestCase):
    def test_infrastructure_timeout_is_excluded_without_removing_other_failures(self):
        failure = row(win=False, completed=False, invalid=True)
        left = {"sleep": failure, "invalid": failure}
        right = {"sleep": row(), "invalid": row()}
        pairs = paired_rows(left, right, {"sleep"})
        self.assertEqual([r["state_id"] for r in pairs], ["invalid"])
        self.assertEqual(summarize_pairs(pairs)["wins_gained"], 1)

    def test_infrastructure_annotation_checks_original_evidence_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab.write_new(root / "a.result.json", {"status": "timeout"})
            (root / "a_r0.jsonl").write_text("{}\n")
            annotation = {"state_id": "a", "kind": "system_sleep_timeout",
                          "result_sha256": file_sha256(root / "a.result.json"),
                          "trace_sha256": file_sha256(root / "a_r0.jsonl")}
            lab.write_new(root / "infrastructure_annotations.json",
                          {"schema_version": 1, "annotations": [annotation]})
            self.assertEqual(infrastructure_annotations(root), [annotation])
            (root / "a_r0.jsonl").write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, "does not match"):
                infrastructure_annotations(root)

    def test_frozen_snapshot_preserves_comparison_display_order(self):
        pairs = {"old_format__new_no_thinking": {"wins": 1},
                 "old_format__new_thinking": {"wins": 2},
                 "new_no_thinking__new_thinking": {"wins": 3}}
        restored = json.loads(json.dumps(pairs, sort_keys=True))
        self.assertNotEqual(list(pairs), list(restored))
        self.assertEqual(list(ordered_pairs(restored).items()), list(pairs.items()))

    def test_invalid_counts_in_success_comparison_but_not_hp(self):
        pairs = paired_rows({"a": row(win=False, completed=False, invalid=True)}, {"a": row()})
        result = summarize_pairs(pairs)
        self.assertEqual(result["paired_finished"], 1)
        self.assertEqual(result["wins_gained"], 1)
        self.assertEqual(result["paired_resolved"], 0)
        self.assertIsNone(result["mean_hp_loss_difference"])

    def test_unfinished_is_not_treated_as_a_loss(self):
        self.assertEqual(paired_rows({"a": row()}, {}), [])
        self.assertEqual(outcome(None), "Not finished")
        self.assertIsNone(summarize_pairs([])["mean_win_difference"])

    def test_hp_changes_have_right_minus_left_sign_and_keep_healing(self):
        result = summarize_pairs(paired_rows({"a": row(hp_loss=4)}, {"a": row(hp_loss=-6)}))
        self.assertEqual(result["mean_hp_loss_difference"], -10)
        self.assertEqual(result["higher_exit_hp"], 1)

    def test_pairing_rejects_state_or_seed_mismatch(self):
        for key in ("world_seed", "hp", "policy_seed"):
            right = copy.deepcopy(row())
            right[key] += 1
            with self.assertRaisesRegex(ValueError, "identity differs"):
                paired_rows({"a": row()}, {"a": right})
