from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rescore_teacher_report import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audit_row(public_state_hash: str, window_id: str) -> dict:
    return {
        "public_state_hash": public_state_hash,
        "window_id": window_id,
        "turn": 2,
        "legal_actions": [
            {"index": 0, "bits": 10, "description": "strike"},
            {"index": 1, "bits": 20, "description": "end turn"},
        ],
        "teacher_queries": [
            {
                "teacher_vote": {
                    "abstained": False,
                    "displayed_action_visits": {"0": 60, "1": 40},
                },
                "search": {
                    "best_sequence": [
                        {"bits": 20, "description": "end turn", "turn": 2}
                    ]
                },
            }
        ],
    }


class RescoreTeacherReportCliTest(unittest.TestCase):
    def test_deterministic_report_with_quarantine_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            per_row = root / "rows.jsonl"
            audit = root / "audit.jsonl"
            quarantine = root / "quarantine.json"
            out_a = root / "report-a.json"
            out_b = root / "report-b.json"
            _write_jsonl(
                per_row,
                [
                    {"public_state_hash": "hash-a", "top1_action_index": 1},
                    {"public_state_hash": "hash-b", "top1_action_index": 0},
                ],
            )
            _write_jsonl(
                audit,
                [
                    _audit_row("hash-a", "window-a"),
                    _audit_row("hash-b", "window-b"),
                ],
            )
            quarantine.write_text(
                json.dumps(
                    {
                        "kind": "state_sanity_quarantine",
                        "version": 1,
                        "quarantined_windows": ["window-a"],
                    }
                ),
                encoding="utf-8",
            )
            common = [
                "--per-row",
                str(per_row),
                "--audit",
                str(audit),
                "--quarantine",
                str(quarantine),
                "--tie-ratio",
                "0.8",
            ]

            with redirect_stdout(io.StringIO()):
                main([*common, "--out", str(out_a)])
                main([*common, "--out", str(out_b)])

            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())
            report = json.loads(out_a.read_text(encoding="utf-8"))
            self.assertEqual(report["kind"], "teacher_regret_report")
            self.assertEqual(report["version"], 1)
            self.assertEqual(report["tie_ratio"], 0.8)
            self.assertEqual(report["overall"]["all"]["n"], 2)
            self.assertEqual(report["overall"]["clean"]["n"], 1)
            self.assertEqual(
                [item["sha256"] for item in report["inputs"]],
                [
                    hashlib.sha256(per_row.read_bytes()).hexdigest(),
                    hashlib.sha256(audit.read_bytes()).hexdigest(),
                    hashlib.sha256(quarantine.read_bytes()).hexdigest(),
                ],
            )

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.json"
            output.write_text("keep", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "output_must_be_fresh"):
                    main(
                        [
                            "--per-row",
                            str(root / "missing-rows.jsonl"),
                            "--audit",
                            str(root / "missing-audit.jsonl"),
                            "--out",
                            str(output),
                        ]
                    )
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
