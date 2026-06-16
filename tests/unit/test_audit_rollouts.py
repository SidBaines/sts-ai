from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_rollouts import audit_paths
from sts_ai.schemas import SCHEMA_VERSION


def _record(**overrides):
    record = {
        "world_seed": 1,
        "decision_index": 0,
        "state": {},
        "state_text": "",
        "legal_actions": [{"index": 0, "bits": 1, "description": "first"}],
        "selected_action": {"index": 0, "bits": 1, "description": "first"},
        "agent": {
            "action_index": 0,
            "raw_response": '{"reasoning": "ok", "action_index": 0}',
            "reasoning": "ok",
            "thinking": "",
            "valid": True,
            "retries": 0,
            "metadata": {"parsed": {"reasoning": "ok", "action_index": 0}},
        },
        "after_state": {},
        "phase": "out_of_combat",
        "affordances": {},
        "policy_seed": 2,
        "rollout_index": 0,
        "action_executed": True,
    }
    record.update(overrides)
    return record


def _write_rollout(root: Path, record: dict, *, meta_overrides: dict | None = None) -> Path:
    path = root / "seed_1_r0.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "n_decisions": 1,
        "n_invalid": 0 if record["agent"].get("valid", True) else 1,
        "stopped_reason": "terminal",
    }
    if meta_overrides:
        meta.update(meta_overrides)
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


class AuditRolloutsTest(unittest.TestCase):
    def test_clean_rollout_has_no_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_rollout(Path(tmp), _record())

            rows, issues = audit_paths([path])

            self.assertEqual(len(rows), 1)
            self.assertEqual(issues, [])

    def test_missing_meta_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_1_r0.jsonl"
            path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

            _, issues = audit_paths([path])

            self.assertEqual([issue.code for issue in issues], ["missing_meta"])

    def test_unclosed_think_and_json_in_thinking_are_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = dict(_record()["agent"])
            agent.update(
                {
                    "raw_response": '<think>thought\n{"reasoning": "ok", "action_index": 0}',
                    "thinking": 'thought\n{"reasoning": "ok", "action_index": 0}',
                }
            )
            path = _write_rollout(Path(tmp), _record(agent=agent))

            _, issues = audit_paths([path])

            self.assertEqual({issue.code for issue in issues}, {"unclosed_think", "json_in_thinking"})
            self.assertTrue(all(issue.severity == "warning" for issue in issues))

    def test_invalid_executed_fallback_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = dict(_record()["agent"])
            agent.update({"valid": False, "metadata": {"error": "no json object"}})
            path = _write_rollout(Path(tmp), _record(agent=agent))

            _, issues = audit_paths([path])

            self.assertIn("invalid_action_executed", {issue.code for issue in issues})

    def test_unexecuted_invalid_requires_agent_invalid_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = dict(_record()["agent"])
            agent.update({"valid": False, "metadata": {"error": "no json object"}})
            path = _write_rollout(
                Path(tmp),
                _record(agent=agent, selected_action={}, action_executed=False),
                meta_overrides={"stopped_reason": "agent_invalid"},
            )

            _, issues = audit_paths([path])

            self.assertEqual(issues, [])

    def test_schema_mismatch_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_rollout(Path(tmp), _record(), meta_overrides={"schema_version": 2})

            _, issues = audit_paths([path])

            self.assertIn("schema_version_mismatch", {issue.code for issue in issues})


if __name__ == "__main__":
    unittest.main()
