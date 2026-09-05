from __future__ import annotations

from dataclasses import asdict
import unittest
import json
import hashlib
from pathlib import Path
import tempfile

from sts_ai.provenance import adapter_provenance

from scripts.experiments.astra_analyze_failure_replay import (
    analyze, failure_category, paired_changes, summarize_arm, validate_pair,
)
from sts_ai.agents import parse_json_action
from sts_ai.schemas import LegalAction


class TestAstraFailureReplay(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {"probe_id": 0, "probe_screen": "MAP_SCREEN", "state_text": "Map: x=3 is available",
                    "legal_actions": [{"index": 0, "bits": 3, "description": "choose map node x=3"}]}

    def prediction(self, raw, row=None):
        row = row or self.row
        decision = parse_json_action(raw, [LegalAction(**a) for a in row["legal_actions"]], output_contract="action_text")
        return {"probe_id": row["probe_id"], "probe_screen": row["probe_screen"], "decision": asdict(decision)}

    def test_invalid_singleton_coordinate_does_not_count_as_valid_intent(self) -> None:
        prediction = self.prediction('{"action":"x=3"}')
        self.assertEqual(failure_category(self.row, prediction["decision"]), "map_coordinate_in_action_field")
        result = summarize_arm([prediction], {0: self.row})
        self.assertEqual(result["by_menu_size"]["singleton"]["invalid"], 1)
        self.assertEqual(result["by_menu_size"]["branching"]["n"], 0)

    def test_valid_prefix_and_index_fallback_are_separate_from_strict_contract(self) -> None:
        prediction = self.prediction('{"action_index":0}')
        strict = [{"probe_id": 0, "valid_json": False, "matched": False}]
        result = summarize_arm([prediction], {0: self.row}, strict)
        self.assertEqual(result["overall"]["valid"], 1)
        self.assertEqual(result["overall"]["valid_resolution_paths"], {"action_index": 1})
        self.assertEqual(result["overall"]["strict_action_contract_valid"], 0)
        self.assertEqual(failure_category(self.row, prediction["decision"]), "valid_live_action")

    def test_categories_distinguish_syntax_schema_and_nonlegal_text(self) -> None:
        cases = [('broken', 'no_parseable_json_object'), ('{"action":0}', 'missing_or_nonstring_action'),
                 ('{"action":"0"}', 'numeric_string_in_action_field'),
                 ('{"action":"Choose map node x=3"}', 'capitalization_only'),
                 ('{"action":"Map: x=3 is available"}', 'observation_text_in_action_field')]
        for raw, expected in cases:
            self.assertEqual(failure_category(self.row, self.prediction(raw)["decision"]), expected)

    def test_paired_change_requires_valid_on_both_sides_to_count_switch(self) -> None:
        a, b = self.prediction('{"action":"x=3"}'), self.prediction('{"action":"choose map node x=3"}')
        result = paired_changes([a], [b], {0: self.row})
        self.assertEqual(result["overall"]["valid_count_delta"], 1)
        self.assertEqual(result["overall"]["action_switches_both_valid"], 0)
        self.assertEqual(result["overall"]["transitions"], {"invalid_to_valid": 1})

    def test_incomplete_paired_cohort_rejected(self) -> None:
        prediction = self.prediction('{"action_index":0}')
        with self.assertRaises(ValueError):
            paired_changes([prediction], [], {0: self.row})

    def test_only_explicit_single_intervention_pairs_allowed(self) -> None:
        config = dict(source="cohort", source_sha256="hash", model="base", temperature=0.,
                      batch_size=16, max_tokens=256, max_retries=1, script_sha256="code",
                      contract="action_text", constrained=False)
        validate_pair("base_temp0.0", "base_temp0.7", config, {**config, "temperature": .7})
        validate_pair("base_temp0.0", "gemma26_temp0.0", config, {**config, "model": "gemma26"})
        with self.assertRaises(ValueError):
            validate_pair("base_temp0.0", "rl5_temp0.7", config, {**config, "temperature": .7})
        with self.assertRaises(ValueError):
            validate_pair("base_temp0.0", "gemma26_temp0.0", config, {**config, "model": "gemma26", "batch_size": 1})
        with self.assertRaises(ValueError):
            validate_pair("base_temp0.0", "base_temp0.7", config, {**config, "temperature": .7, "future_setting": True})

    def test_complete_arm_required_and_raw_decisions_reparsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cohort = [{**self.row, "probe_id": i, "world_seed": 202, "decision_index": i,
                       "source_trace": "original.jsonl", "agent": self.prediction('{"action":"x=3"}')["decision"]}
                      for i in range(56)]
            source = directory / "cohort.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in cohort))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (directory / "cohort.manifest.json").write_text(json.dumps({"n": 56, "sha256": digest}))
            arm = directory / "base_temp0.0"
            arm.mkdir()
            config = dict(mode="probe", contract="action_text", constrained=False, source=str(source),
                          source_sha256=digest, adapter=None, adapter_identity=adapter_provenance(None),
                          temperature=0., model="base", batch_size=16, max_tokens=256,
                          max_retries=1, script_sha256="script")
            (arm / "config.json").write_text(json.dumps(config))
            predictions = [{**{k: row[k] for k in ("probe_id", "probe_screen", "world_seed", "decision_index")},
                            "decision": row["agent"]} for row in cohort]
            path = arm / "predictions.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in predictions[:-1]))
            self.assertEqual(analyze(directory)["completed_arms"], [])
            (arm / "report.json").write_text(json.dumps({"overall": {"n": 56, "valid": 0}}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze(directory)
            path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
            result = analyze(directory)
            self.assertEqual(result["completed_arms"], ["base_temp0.0"])
            self.assertEqual(result["arms"]["base_temp0.0"]["overall"]["invalid"], 56)
            predictions[0]["decision"] = {**predictions[0]["decision"], "retries": 1}
            path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
            with self.assertRaisesRegex(ValueError, "contains retries"):
                analyze(directory)
            predictions[0]["decision"] = {**predictions[0]["decision"], "retries": 0, "valid": True}
            path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
            with self.assertRaisesRegex(ValueError, "disagrees with parsing"):
                analyze(directory)
